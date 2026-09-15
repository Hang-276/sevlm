from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model

from open_r1.trainer.vllm_rollout import (
    ColocatedVLLMRollout,
    build_requests,
    completion_tensors,
    iter_policy_weights,
)


def test_requests_keep_multiple_images_with_their_prompt():
    requests = build_requests(["two images", "text only", "one image"], [["a", "b"], [], ["c"]])
    assert requests[0]["multi_modal_data"]["image"] == ["a", "b"]
    assert "multi_modal_data" not in requests[1]
    assert requests[2]["multi_modal_data"]["image"] == ["c"]
    with pytest.raises(ValueError):
        build_requests(["missing images"], [])


def test_completion_masks_use_lengths_even_when_padding_equals_eos():
    outputs = [
        dict(prompt_token_ids=[1, 2], token_ids=[7, 9], finish_reason="stop"),
        dict(prompt_token_ids=[3], token_ids=[5, 6, 8], finish_reason="length"),
    ]
    ids, mask, truncated = completion_tensors(outputs, [[1, 2], [3]], 9, "cpu")
    assert ids.tolist() == [[7, 9, 9], [5, 6, 8]]
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert truncated.tolist() == [False, True]


def test_reject_different_image_token_expansion_and_missing_outputs():
    output = dict(prompt_token_ids=[1, 42, 42, 2], token_ids=[3], finish_reason="stop")
    with pytest.raises(RuntimeError, match="prompt token mismatch"):
        completion_tensors([output], [[1, 42, 2]], 0, "cpu")
    with pytest.raises(RuntimeError, match="number of completions"):
        completion_tensors([], [[1]], 0, "cpu")


def test_hf_preprocessing_does_not_expand_the_prompt_sent_to_vllm():
    from open_r1.trainer import VLMGRPOTrainer

    trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.state = SimpleNamespace(global_step=0)
    trainer.processing_class = None
    trainer.use_vllm = True
    raw = ["look <image> here"]

    def expanding_processor(processor, prompts, images, **kwargs):
        prompts[0] = prompts[0].replace("<image>", "<image>" * 4)
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    trainer.vlm_module = SimpleNamespace(prepare_prompt=lambda *args: raw,
                                        prepare_model_inputs=expanding_processor)
    trainer._generate_with_vllm = Mock(side_effect=RuntimeError("reached vllm"))
    with pytest.raises(RuntimeError, match="reached vllm"):
        trainer._generate_and_score_completions_once([{"prompt": "unused"}], None)
    assert trainer._generate_with_vllm.call_args.args[1] == ["look <image> here"]


def test_lora_sync_matches_effective_policy_without_mutating_training_weights():
    torch.manual_seed(17)
    base = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 3))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["0", "1"]))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    synchronized = dict(iter_policy_weights(model))
    copy = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 3))
    copy.load_state_dict(synchronized, strict=True)
    x = torch.randn(5, 4)
    torch.testing.assert_close(model(x), copy(x))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not m.merged for m in model.modules() if hasattr(m, "merged"))


def test_rollout_syncs_on_optimizer_steps_and_sleeps_after_errors(monkeypatch):
    rollout = ColocatedVLLMRollout.__new__(ColocatedVLLMRollout)
    rollout.last_step = None
    rollout.request = Mock(return_value=[])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = torch.nn.Linear(2, 2)
    rollout.generate(model, 0, [], {})
    first_weights = [c for c in rollout.request.call_args_list if c.args[0] == "weight"]
    assert len(first_weights) == 2
    rollout.request.reset_mock()
    rollout.generate(model, 0, [], {})
    assert not any(c.args[0] == "weight" for c in rollout.request.call_args_list)
    rollout.request.reset_mock()
    rollout.generate(model, 1, [], {})
    assert len([c for c in rollout.request.call_args_list if c.args[0] == "weight"]) == 2

    def fail_generation(command, payload=None):
        if command == "generate":
            raise RuntimeError("generation failed")

    rollout.request = Mock(side_effect=fail_generation)
    with pytest.raises(RuntimeError, match="generation failed"):
        rollout.generate(model, 1, [], {})
    assert rollout.request.call_args_list[-1].args[0] == "sleep"
