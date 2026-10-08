from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from accelerate.data_loader import BatchSamplerShard
import pytest
import torch
from PIL import Image
from torch.utils.data import BatchSampler

from open_r1.trainer.grpo_config import GRPOConfig
from open_r1.trainer import grpo_trainer as trainer_module
from open_r1.trainer.grpo_trainer import (
    CyclicRepeatSampler, FiniteLogitsProcessor, VLMGRPOTrainer, grouped_reward_stats,
)
from open_r1.trainer.vllm_rollout import completion_tensors, eos_completion_mask
from open_r1.trainer.vllm_rollout_worker import RolloutLogitsProcessor


def init_trainer(monkeypatch, **overrides):
    sharding = overrides.pop("sharding", None)
    world_size = overrides.pop("world_size", 1)
    model_eos = overrides.pop("model_eos", None)
    n_gpu = overrides.pop("n_gpu", None)
    options = dict(output_dir="unused", use_cpu=True, report_to="none", beta=0,
                   use_vllm=False, per_device_train_batch_size=2, num_generations=2)
    args = GRPOConfig(**{**options, **overrides})
    if n_gpu is not None:
        args._n_gpu = n_gpu
    model = torch.nn.Linear(1, 1)
    model.warnings_issued = {}
    model.generation_config = SimpleNamespace(eos_token_id=model_eos)
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=7),
    )
    vlm = SimpleNamespace(
        get_vlm_key=lambda: "qwen",
        get_model_class=lambda *a: SimpleNamespace(from_pretrained=lambda *a, **kw: model),
        get_vision_modules_keywords=lambda: [], post_model_init=lambda *a: None,
    )

    def trainer_init(self, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"]
        self.accelerator = SimpleNamespace(num_processes=world_size, process_index=0)
        self.is_fsdp_enabled = sharding == "fsdp"
        self.is_deepspeed_enabled = sharding == "zero3"

    monkeypatch.setattr(trainer_module.Trainer, "__init__", trainer_init)
    monkeypatch.setattr(trainer_module, "is_deepspeed_zero3_enabled", lambda: sharding == "zero3")
    monkeypatch.setattr(trainer_module, "set_seed", lambda *a, **kw: None)
    return VLMGRPOTrainer(
        model="local-placeholder", args=args, reward_funcs=[],
        processing_class=processor, vlm_module=vlm,
    )


def test_supplied_processor_and_hf_sampling_options(monkeypatch):
    trainer = init_trainer(
        monkeypatch, top_p=0.8, top_k=None, min_p=0.05, repetition_penalty=1.2,
    )
    config = trainer.generation_config
    assert config.pad_token_id == 0
    assert config.use_cache is True
    assert (config.top_p, config.top_k, config.min_p, config.repetition_penalty) == (0.8, 0, 0.05, 1.2)


def test_hf_rollout_preserves_model_stop_token_list(monkeypatch):
    trainer = init_trainer(monkeypatch, model_eos=[7, 8])
    assert trainer.generation_config.eos_token_id == [7, 8]


def test_single_process_multigpu_cannot_scatter_flattened_multimodal_inputs(monkeypatch):
    with pytest.raises(ValueError, match="one GPU per process"):
        init_trainer(monkeypatch, n_gpu=2)


def test_static_sampler_resumes_same_epoch_order_through_accelerate():
    from accelerate.data_loader import DataLoaderShard
    from torch.utils.data import DataLoader

    def make_loader():
        sampler = CyclicRepeatSampler(range(20), mini_repeat_count=2, batch_size=2,
                                      cycle_length=2, seed=42)
        return DataLoaderShard(range(20), sampler=sampler, batch_size=2)

    uninterrupted = make_loader()
    uninterrupted.set_epoch(0)
    epoch_zero = [batch.tolist() for batch in uninterrupted]
    uninterrupted.set_epoch(3)
    expected = [batch.tolist() for batch in uninterrupted]
    resumed = make_loader()
    resumed.set_epoch(3)
    actual = [batch.tolist() for batch in resumed]
    assert actual == expected and actual != epoch_zero


@pytest.mark.parametrize("sharding", ["fsdp", "zero3"])
def test_sharding_does_not_allow_groups_to_span_accumulation_steps(monkeypatch, sharding):
    with pytest.raises(ValueError, match="global train batch size"):
        init_trainer(
            monkeypatch, sharding=sharding, world_size=1,
            gradient_accumulation_steps=4, num_generations=8,
        )


def test_small_image_does_not_resize_later_images():
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.state = SimpleNamespace(global_step=0)
    trainer.processing_class = None
    captured = []

    def prepare(processor, prompts, images, **kwargs):
        captured.extend(images)
        raise RuntimeError("captured")

    trainer.vlm_module = SimpleNamespace(
        prepare_prompt=lambda *a: ["two images"], prepare_model_inputs=prepare,
    )
    original = [Image.new("RGB", (14, 28)), Image.new("RGB", (100, 80))]
    with pytest.raises(RuntimeError, match="captured"):
        trainer._generate_and_score_completions_once([{"prompt": "p", "image": original}], None)
    assert [image.size for image in captured] == [(28, 56), (100, 80)]
    assert [image.size for image in original] == [(14, 28), (100, 80)]


def test_multiple_eos_ids_and_eos_padding():
    tokens = torch.tensor([[1, 8, 7, 7], [2, 7, 7, 7], [3, 4, 5, 6]])
    mask, truncated = eos_completion_mask(tokens, [7, 8])
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    assert truncated.tolist() == [False, False, True]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_logits_fallback_is_dtype_safe_and_skips_invalid_ban_ids(dtype):
    source = torch.tensor([[float("nan")] * 4, [0, 1, float("inf"), float("-inf")]], dtype=dtype)
    hf = FiniteLogitsProcessor([1, -1, 151655])(None, source.clone())
    worker = torch.stack([RolloutLogitsProcessor([1, -1, 151655])([], row.clone()) for row in source])
    torch.testing.assert_close(hf, worker)
    assert torch.isfinite(hf[:, [0, 2, 3]]).all()
    assert torch.isneginf(hf[:, 1]).all()
    probabilities = hf.float().softmax(dim=-1)
    assert torch.isfinite(probabilities).all() and torch.all(probabilities.sum(dim=-1) == 1)


def test_aborted_vllm_output_cannot_become_training_data():
    with pytest.raises(RuntimeError, match="finish normally"):
        completion_tensors(
            [{"prompt_token_ids": [1], "token_ids": [2], "finish_reason": "abort"}],
            [[1]], 0, "cpu",
        )


def test_evaluation_never_reuses_or_advances_training_rollout_buffer():
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.max_completion_length = 8
    trainer.num_generations = trainer.num_iterations = 2
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer._step = 3
    trainer._buffered_inputs = [None, None]
    trainer.beta = 0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x, is_main_process=False, device="cpu")
    trainer._metrics = defaultdict(list)
    trainer._get_per_token_logps = lambda *a, **kw: torch.full((2, 3), -0.5)
    trainer._generate_and_score_completions = lambda *a: {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long), "prompt_mask": torch.ones(2, 1),
        "completion_ids": torch.ones(2, 3, dtype=torch.long), "completion_mask": torch.ones(2, 3),
        "multimodal_inputs": {}, "advantages": torch.tensor([0.5, -0.5]),
        "old_per_token_logps": torch.full((2, 3), -0.5),
    }
    model = torch.nn.Linear(1, 1).eval()
    assert torch.isfinite(trainer.compute_loss(model, [{"prompt": "p"}, {"prompt": "p"}]))
    assert trainer._step == 3 and trainer._buffered_inputs == [None, None]


@pytest.mark.parametrize("world, per_rank, generations, accumulation, iterations", [
    (2, 2, 4, 2, 3), (4, 3, 6, 2, 2), (8, 2, 8, 2, 3),
])
def test_accelerate_sharding_keeps_groups_and_optimizer_replay_aligned(
    world, per_rank, generations, accumulation, iterations,
):
    unique_per_update = world * per_rank * accumulation // generations
    rank_batches = []
    for rank in range(world):
        sampler = CyclicRepeatSampler(
            list(range(unique_per_update * 3)), generations, unique_per_update, iterations, seed=42,
        )
        batches = BatchSampler(sampler, batch_size=per_rank, drop_last=True)
        rank_batches.append(list(BatchSamplerShard(
            batches, num_processes=world, process_index=rank, split_batches=False,
        )))
    global_batches = [sum((rank[index] for rank in rank_batches), [])
                      for index in range(len(rank_batches[0]))]
    for batch in global_batches:
        for start in range(0, len(batch), generations):
            assert len(set(batch[start:start + generations])) == 1
    cycle_width = accumulation * iterations
    for start in range(0, len(global_batches), cycle_width):
        original = global_batches[start:start + accumulation]
        for iteration in range(1, iterations):
            assert global_batches[start + iteration * accumulation:start + (iteration + 1) * accumulation] == original


@pytest.mark.parametrize("values", [
    [float("nan"), 1], [float("inf"), 1], [-float("inf"), 1], [3e38, -3e38],
])
def test_invalid_reward_statistics_fail_before_export_or_resampling(values):
    with pytest.raises(FloatingPointError, match="rewards or group statistics"):
        grouped_reward_stats(torch.tensor(values), 2)


def test_nan_to_num_cannot_repair_nonfinite_backward():
    parameter = torch.tensor(1.0, requires_grad=True)
    unsafe_loss = torch.nan_to_num((parameter * float("nan")).exp(), nan=0.0)
    assert torch.isfinite(unsafe_loss)
    unsafe_loss.backward()
    assert torch.isnan(parameter.grad)


@pytest.mark.parametrize("invalid_field", ["current", "old", "reference", "advantages", "arithmetic_overflow"])
def test_nonfinite_loss_inputs_refuse_backward_even_when_masked(invalid_field):
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.max_completion_length = 4
    trainer.num_generations = trainer.num_iterations = 2
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
    trainer._step = 0
    trainer._buffered_inputs = [None]
    trainer.beta = 0.1
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.accelerator = SimpleNamespace(device="cpu", num_processes=1, gather_for_metrics=lambda x: x)
    trainer._metrics = defaultdict(list)
    parameter = torch.tensor(1.0, requires_grad=True)
    current = parameter * torch.ones(2, 3)
    old, reference = torch.ones(2, 3), torch.ones(2, 3)
    advantages = torch.tensor([0.5, -0.5])
    if invalid_field == "arithmetic_overflow":
        current = parameter * torch.full((2, 3), 5.0)
        advantages = torch.tensor([1e38, -1e38])
    else:
        field = {"current": current, "old": old, "reference": reference, "advantages": advantages}[invalid_field]
        field.data.reshape(-1)[-1] = float("nan")
    trainer._get_per_token_logps = lambda *a, **kw: current
    trainer._generate_and_score_completions = lambda *a: {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long), "prompt_mask": torch.ones(2, 1),
        "completion_ids": torch.ones(2, 3, dtype=torch.long), "completion_mask": torch.zeros(2, 3),
        "multimodal_inputs": {}, "advantages": advantages,
        "old_per_token_logps": old, "ref_per_token_logps": reference,
    }
    with pytest.raises(FloatingPointError, match="unsafe backward"):
        trainer.compute_loss(None, [{"prompt": "p"}, {"prompt": "p"}])
    assert parameter.grad is None


def test_clean_rank_refuses_backward_when_another_rank_reports_invalid_logps():
    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    reduce = Mock(return_value=torch.ones(1))
    trainer.accelerator = SimpleNamespace(device="cpu", num_processes=2, reduce=reduce)
    with pytest.raises(FloatingPointError, match="unsafe backward"):
        trainer._assert_finite_loss_inputs(torch.ones(2, 3), None)
    reduce.assert_called_once()
    torch.testing.assert_close(reduce.call_args.args[0], torch.zeros(1))
    assert reduce.call_args.kwargs == {"reduction": "sum"}
