import builtins
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from open_r1 import qwen2_5vl_monkey_patch as patch


def test_modern_attention_keeps_native_fp32_rotary():
    attention = patch.qwen_modeling.Qwen2_5_VLVisionAttention
    original = attention.forward
    assert patch.monkey_patch_qwen2_5vl_flash_attn() is False
    assert attention.forward is original
    torch.manual_seed(42)
    q, k = torch.randn(3, 2, 8).bfloat16(), torch.randn(3, 2, 8).bfloat16()
    cos, sin = torch.randn(3, 8).bfloat16(), torch.randn(3, 8).bfloat16()
    actual = patch.qwen_modeling.apply_rotary_pos_emb_vision(q, k, cos, sin)
    expected = patch.qwen_modeling.apply_rotary_pos_emb_vision(q.float(), k.float(), cos.float(), sin.float())
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference.bfloat16())


def test_legacy_attention_still_casts_rotary_to_fp32(monkeypatch):
    class Legacy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.qkv = torch.nn.Linear(8, 24).bfloat16()
            self.proj = torch.nn.Linear(8, 8).bfloat16()

    seen = []

    def rotary(q, k, cos, sin):
        seen.append((cos.dtype, sin.dtype))
        return q, k

    monkeypatch.setattr(patch, "Qwen2_5_VLVisionFlashAttention2", Legacy)
    monkeypatch.setattr(patch, "apply_rotary_pos_emb_flashatt", rotary)
    monkeypatch.setattr(patch, "flash_attn_varlen_func", lambda q, k, v, *args: v)
    assert patch.monkey_patch_qwen2_5vl_flash_attn() is True
    result = Legacy()(torch.randn(3, 8).bfloat16(), torch.tensor([0, 3]),
                      position_embeddings=(torch.ones(3, 8).bfloat16(), torch.zeros(3, 8).bfloat16()))
    assert result.shape == (3, 8)
    assert seen == [(torch.float32, torch.float32)]


def tiny_model():
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=32, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                           out_hidden_size=16, window_size=8, fullatt_block_indexes=[0]),
        image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
        pad_token_id=0, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    return Qwen2_5_VLForConditionalGeneration(config)


def test_modern_forward_keeps_native_image_logits_and_backward(monkeypatch):
    model = tiny_model()
    original = patch.qwen_modeling.Qwen2_5_VLModel.forward
    monkeypatch.setattr(patch.qwen_modeling.Qwen2_5_VLModel, "forward", original)
    conditional_forward = Qwen2_5_VLForConditionalGeneration.forward
    tokens = torch.tensor([[1, 5, 3, 6, 7, 2]])
    inputs = dict(input_ids=tokens, labels=tokens.clone(), pixel_values=torch.randn(4, 12),
                  image_grid_thw=torch.tensor([[1, 2, 2]]), use_cache=False)
    expected = model(**inputs).logits.detach()
    generation_inputs = {key: value for key, value in inputs.items() if key not in {"labels", "use_cache"}}
    expected_generation = model.generate(**generation_inputs, max_new_tokens=3, do_sample=False, use_cache=True)
    assert patch.monkey_patch_qwen2_5vl_forward() is True
    assert patch.monkey_patch_qwen2_5vl_forward() is False
    assert Qwen2_5_VLForConditionalGeneration.forward is conditional_forward
    result = model(**inputs)
    torch.testing.assert_close(result.logits, expected)
    result.loss.backward()
    grads = [p.grad for p in model.model.visual.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    torch.testing.assert_close(
        model.generate(**generation_inputs, max_new_tokens=3, do_sample=False, use_cache=True), expected_generation,
    )


@pytest.mark.parametrize("global_flags, local_image, local_video, accepted", [
    ([2, 0], True, False, True), ([0, 0], False, False, True),
    ([0, 2], False, True, True), ([1, 0], False, False, False),
    ([2, 1], True, False, False),
])
def test_sharded_forward_rejects_mixed_rank_modalities_before_native_call(
    monkeypatch, global_flags, local_image, local_video, accepted,
):
    calls = []

    class Native(torch.nn.Module):
        def forward(self, input_ids=None, pixel_values=None, pixel_values_videos=None, **kwargs):
            calls.append(True)
            return input_ids

    monkeypatch.setattr(patch.qwen_modeling, "Qwen2_5_VLModel", Native)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda flags, **kwargs: flags.copy_(torch.tensor(global_flags)))
    assert patch.monkey_patch_qwen2_5vl_forward()
    model = Native()
    tokens = torch.tensor([[1, 2]])
    kwargs = dict(pixel_values=torch.ones(1) if local_image else None,
                  pixel_values_videos=torch.ones(1) if local_video else None)
    if accepted:
        torch.testing.assert_close(model(tokens, **kwargs), tokens)
        assert calls == [True]
    else:
        with pytest.raises(RuntimeError, match="mixed image/text ranks"):
            model(tokens, **kwargs)
        assert not calls


def test_torch_load_patch_skips_only_absent_deepspeed(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("deepspeed"):
            raise ModuleNotFoundError("No module named 'deepspeed'", name="deepspeed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert patch.monkey_patch_torch_load() is False


def test_torch_load_patch_preserves_transitive_import_failures(monkeypatch):
    original = builtins.__import__

    def missing_dependency(name, *args, **kwargs):
        if name.startswith("deepspeed"):
            raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        patch.monkey_patch_torch_load()


def test_deepspeed_checkpoint_patch_sets_explicit_weights_only_false(monkeypatch, tmp_path):
    original = builtins.__import__
    checkpoint = SimpleNamespace(TorchCheckpointEngine=type("Checkpoint", (), {}))

    def fake_checkpoint(name, *args, **kwargs):
        if name == "deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine":
            return checkpoint
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_checkpoint)
    assert patch.monkey_patch_torch_load()
    path = tmp_path / "checkpoint.pt"
    torch.save({"step": 3, "weights": torch.ones(2)}, path)
    loaded = checkpoint.TorchCheckpointEngine().load(path, map_location="cpu")
    assert loaded["step"] == 3
    torch.testing.assert_close(loaded["weights"], torch.ones(2))
