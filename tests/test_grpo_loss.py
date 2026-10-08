from types import SimpleNamespace
from collections import defaultdict
from unittest.mock import Mock

import pytest
import torch

from open_r1.trainer.grpo_config import GRPOConfig
from open_r1.trainer.grpo_trainer import VLMGRPOTrainer, reward_breakdown_sums
from open_r1.trainer.grpo_loss import reduce_grpo_loss


def test_grpo_preserves_legacy_response_weighting_and_default():
    losses = torch.tensor([[2.0, 4.0, 99.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert GRPOConfig.__dataclass_fields__["loss_type"].default == "grpo"
    torch.testing.assert_close(reduce_grpo_loss(losses, mask), torch.tensor(3.5))


def test_dr_grpo_gives_each_valid_token_the_same_gradient():
    losses = torch.ones(2, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    loss = reduce_grpo_loss(losses, mask, "dr_grpo", 8)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(6 / 16))
    torch.testing.assert_close(losses.grad, mask.float() / 16)
    # The old objective gives the short response twice as much per-token weight.
    legacy_losses = losses.detach().clone().requires_grad_(True)
    reduce_grpo_loss(legacy_losses, mask).backward()
    assert legacy_losses.grad[0, 0] == 2 * legacy_losses.grad[1, 0]


@pytest.mark.parametrize("normalization_length", [None, 2, 8, 16])
def test_padding_width_does_not_change_fixed_budget_loss(normalization_length):
    losses = torch.tensor([[1.0, 2.0]])
    mask = torch.ones_like(losses)
    expected = reduce_grpo_loss(losses, mask, "dr_grpo", 8, normalization_length)
    padded = torch.tensor([[1.0, 2.0, 1000.0, 1000.0]])
    padded_mask = torch.tensor([[1, 1, 0, 0]])
    torch.testing.assert_close(reduce_grpo_loss(padded, padded_mask, "dr_grpo", 8, normalization_length), expected)


def test_another_fixed_constant_changes_only_overall_gradient_scale():
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    original = torch.ones(2, 4, requires_grad=True)
    scaled = torch.ones(2, 4, requires_grad=True)
    reduce_grpo_loss(original, mask, "dr_grpo", 8).backward()
    reduce_grpo_loss(scaled, mask, "dr_grpo", 8, 2).backward()
    torch.testing.assert_close(scaled.grad, 4 * original.grad)


def test_fully_masked_batch_retains_a_finite_zero_gradient_graph():
    losses = torch.randn(3, 5, requires_grad=True)
    loss = reduce_grpo_loss(losses, torch.zeros_like(losses), "dr_grpo", 8)
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    torch.testing.assert_close(losses.grad, torch.zeros_like(losses))


def test_rank_average_matches_global_gradient_with_unequal_active_rows():
    # Each rank samples the same B; zero-advantage or truncated rows contribute
    # zero without being deleted from the denominator. Rank 0 has one active
    # row and rank 1 has two, with different response lengths on both ranks.
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    values = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    global_param = torch.tensor(0.7, requires_grad=True)
    reduce_grpo_loss(global_param * values, mask, "dr_grpo", 8).backward()
    rank_grads = []
    for rank_slice in (slice(0, 2), slice(2, 4)):
        rank_param = torch.tensor(0.7, requires_grad=True)
        reduce_grpo_loss(rank_param * values[rank_slice], mask[rank_slice], "dr_grpo", 8).backward()
        rank_grads.append(rank_param.grad)
    torch.testing.assert_close(torch.stack(rank_grads).mean(), global_param.grad)


def test_gradient_accumulation_divides_once_and_matches_full_batch():
    mask = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0]])
    values = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    accumulated_param = torch.tensor(0.3, requires_grad=True)
    for microbatch in (slice(0, 2), slice(2, 4)):
        # This factor is applied by Trainer/Accelerate, outside compute_loss.
        (reduce_grpo_loss(accumulated_param * values[microbatch], mask[microbatch], "dr_grpo", 8) / 2).backward()
    full_param = torch.tensor(0.3, requires_grad=True)
    reduce_grpo_loss(full_param * values, mask, "dr_grpo", 8).backward()
    torch.testing.assert_close(accumulated_param.grad, full_param.grad)


@pytest.mark.parametrize("loss_type, normalization_length", [("grpo", None), ("dr_grpo", None), ("dr_grpo", 2)])
def test_standard_compute_loss_uses_selected_reduction(loss_type, normalization_length):
    # Exercise the trainer's standard branch with real autograd, without model
    # weights or rollout generation. A fully masked second response stands in
    # for a filtered rollout and has no gradient.
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.loss_type = loss_type
    trainer.loss_normalization_length = normalization_length
    trainer.max_completion_length = 8
    trainer.num_generations = 2
    trainer.num_iterations = 1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
    trainer._step = 0
    trainer._buffered_inputs = [None]
    trainer.beta = 0.0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.accelerator = SimpleNamespace(
        gather_for_metrics=lambda x: x, is_main_process=False, device="cpu",
    )
    trainer._metrics = {"clip_ratio": []}
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    advantages = torch.tensor([0.4, -0.4])
    logps = torch.full((2, 4), -0.5, requires_grad=True)
    trainer._get_per_token_logps = lambda *args, **kwargs: logps
    trainer._generate_and_score_completions = lambda *args: {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "prompt_mask": torch.ones(2, 1, dtype=torch.long),
        "completion_ids": torch.ones(2, 4, dtype=torch.long),
        "completion_mask": mask,
        "multimodal_inputs": {},
        "advantages": advantages,
        "old_per_token_logps": None,
    }
    loss = trainer.compute_loss(None, [{"prompt": "a"}, {"prompt": "a"}])
    loss.backward()
    denominator = (2 * (normalization_length or 8)) if loss_type == "dr_grpo" else 4
    expected = torch.tensor([[-0.4 / denominator, -0.4 / denominator, 0, 0], [0, 0, 0, 0]])
    torch.testing.assert_close(logps.grad, expected)


@pytest.mark.parametrize("budget", [None, 0, -1, 1.5, True])
def test_dr_grpo_rejects_invalid_fixed_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        reduce_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), "dr_grpo", budget)


def test_rejects_invalid_loss_type_shapes_and_overbudget_completion():
    with pytest.raises(ValueError, match="loss_type"):
        reduce_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), "dapo", 8)
    with pytest.raises(ValueError, match="same"):
        reduce_grpo_loss(torch.ones(1, 2), torch.ones(1, 3), "dr_grpo", 8)
    with pytest.raises(ValueError, match="non-empty"):
        reduce_grpo_loss(torch.empty(0, 2), torch.empty(0, 2), "dr_grpo", 8)
    with pytest.raises(ValueError, match="exceeds"):
        reduce_grpo_loss(torch.ones(1, 9), torch.ones(1, 9), "dr_grpo", 8)
    with pytest.raises(ValueError, match="only supported"):
        reduce_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), "grpo", 8, 4)


@pytest.mark.parametrize("normalization_length", [0, -1, 1.5, True])
def test_rejects_invalid_explicit_fixed_denominator(normalization_length):
    with pytest.raises(ValueError, match="positive integer"):
        reduce_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), "dr_grpo", 8, normalization_length)


def test_invalid_trainer_loss_type_fails_before_model_loading():
    with pytest.raises(ValueError, match="loss_type"):
        VLMGRPOTrainer(model="does-not-exist", reward_funcs=[], args=SimpleNamespace(loss_type="dapo"))


def test_grpo_plus_constant_fails_before_model_loading():
    with pytest.raises(ValueError, match="only supported"):
        VLMGRPOTrainer(
            model="does-not-exist", reward_funcs=[],
            args=SimpleNamespace(loss_type="grpo", loss_normalization_length=8),
        )


def test_dr_grpo_rejects_legacy_variable_phase_row_path():
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.loss_type = "dr_grpo"
    with pytest.raises(ValueError, match="variable number of phase rows"):
        trainer.compute_loss(None, [{"accu_reward_method": "clevr_spotdiff"}])


def test_reward_kind_sums_exclude_inapplicable_qa_dimensions():
    stats = reward_breakdown_sums([
        {"task_kind": "clevr_spy", "answer": 0.7, "grounding": 0.8,
         "answer_exact_match": False, "format_valid": True},
        {"task_kind": "verified_count_qa", "answer": 1.0, "grounding": None,
         "process": None, "consistency": None, "budget": None,
         "answer_exact_match": True, "format_valid": True},
    ])
    torch.testing.assert_close(stats[0, 1], torch.tensor([0.8, 1.0]))
    torch.testing.assert_close(stats[2, 1], torch.zeros(2))
    torch.testing.assert_close(stats[2, 5], torch.tensor([1.0, 1.0]))
    assert reward_breakdown_sums([]).shape == stats.shape


def test_reward_metrics_gather_once_on_rank_with_no_qa_or_no_breakdowns():
    rank0 = [{"task_kind": "clevr_spy", "answer": 0.0, "grounding": 0.6,
              "answer_exact_match": False, "format_valid": True}]
    rank1 = [{"task_kind": "verified_count_qa", "answer": 1.0, "grounding": None,
              "answer_exact_match": True, "format_valid": True}] * 3
    gathered = torch.stack([reward_breakdown_sums(rank0), reward_breakdown_sums(rank1)])
    for local in (rank0, rank1, []):
        trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
        trainer._metrics = defaultdict(list)
        trainer.accelerator = SimpleNamespace(gather=Mock(return_value=gathered))
        trainer._log_reward_breakdown_metrics(local, "cpu")
        trainer.accelerator.gather.assert_called_once()
        assert trainer.accelerator.gather.call_args.args[0].shape == (1, 3, 8, 2)
        assert trainer._metrics["reward/answer"] == [0.75]
        assert trainer._metrics["reward/grounding"] == pytest.approx([0.6])
        assert trainer._metrics["reward/verified_count_qa/sample_fraction"] == [0.75]
        assert trainer._metrics["reward/verified_count_qa/answer_correct"] == [1.0]
        assert "reward/verified_count_qa/grounding" not in trainer._metrics


def test_empty_reward_breakdowns_emit_zero_counts_without_nan():
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer._metrics = defaultdict(list)
    trainer.accelerator = SimpleNamespace(gather=lambda x: x)
    trainer._log_reward_breakdown_metrics([], "cpu")
    assert trainer._metrics["reward/clevr_spy/sample_count"] == [0.0]
    assert trainer._metrics["reward/verified_count_qa/sample_fraction"] == [0.0]
    assert "reward/answer" not in trainer._metrics
