import json

from datasets import Dataset
from PIL import Image
import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from open_r1.grpo_data import prepare_grpo_sample
from open_r1.trainer.dynamic_dataset import CyclicDynamicDataset, EpochAwareIterableDataset
from open_r1.trainer.grpo_config import GRPOConfig
from open_r1.trainer.grpo_trainer import VLMGRPOTrainer
from open_r1.vlm_modules.qwen_module import Qwen2VLModule
from test_sft_collator import make_processor


@pytest.fixture
def tiny_checkpoint(tmp_path):
    processor = make_processor("left")
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=len(processor.tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=128,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=16, window_size=112, fullatt_block_indexes=[0]),
        image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
        vision_start_token_id=4, vision_end_token_id=5, pad_token_id=2, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model_path = tmp_path / "tiny_qwen2_5_vl"
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(model_path)
    processor.save_pretrained(model_path)
    image = tmp_path / "red.png"
    Image.new("RGB", (28, 28), "red").save(image)
    return model_path, processor, str(image)


@pytest.mark.parametrize("dynamic,beta", [(False, 0.0), (False, 0.04), (True, 0.0)])
def test_real_grpo_train_eval_save_and_resume(tiny_checkpoint, tmp_path, dynamic, beta, monkeypatch):
    model_path, processor, image = tiny_checkpoint

    def sample(index=0):
        return dict(problem="Describe " + ("red" if index % 2 else "blue"), solution="red",
                    image_path=[image])

    if dynamic:
        def make_data(iterations):
            base = EpochAwareIterableDataset(lambda epoch, sample_idx: sample(sample_idx),
                                             epoch_size=8, seed=42, question_prompt="{Question}")
            return CyclicDynamicDataset(base, num_generations=2,
                                        num_iterations=iterations, batch_size=2)

        train_data, eval_data = make_data(2), make_data(1)
    else:
        rows = [prepare_grpo_sample(sample(i), "{Question}") for i in range(4)]
        train_data = eval_data = Dataset.from_list(rows)

    calls = []

    def reward(prompts, completions, **kwargs):
        assert all(prompts[i] == prompts[i + 1] for i in range(0, len(prompts), 2))
        calls.append(len(completions))
        return [float(i % 2) for i in range(len(completions))]

    args = GRPOConfig(
        output_dir=str(tmp_path / "run"), use_cpu=torch.cuda.device_count() != 1, use_vllm=False,
        num_generations=2, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, num_iterations=2, max_steps=3, max_completion_length=4,
        beta=beta, loss_type="dr_grpo", seed=42, report_to="none", disable_tqdm=True,
        save_strategy="steps", save_steps=1, eval_strategy="steps", eval_steps=1, logging_steps=1,
    )

    def make_trainer():
        trainer = VLMGRPOTrainer(model=str(model_path), args=args, reward_funcs=reward,
                                vlm_module=Qwen2VLModule(), train_dataset=train_data,
                                eval_dataset=eval_data, processing_class=processor,
                                attn_implementation="eager", torch_dtype="float32")
        original_generate = trainer.model.generate

        def cached_generate(*inputs, **options):
            assert options["use_cache"] is True
            return original_generate(*inputs, **options)

        monkeypatch.setattr(trainer.model, "generate", cached_generate)
        return trainer

    import subprocess

    def unexpected_upload(*args, **kwargs):
        raise AssertionError("Model saving must not invoke an external upload")

    monkeypatch.setattr(subprocess, "run", unexpected_upload)
    trainer = make_trainer()
    if dynamic:
        assert trainer.accelerator.dispatch_batches is False
    before = trainer.model.lm_head.weight.detach().clone()
    result = trainer.train()
    assert result.global_step == 3 and torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(before, trainer.model.lm_head.weight)
    assert trainer._step == 6
    assert all(call >= 2 and call % 2 == 0 for call in calls)
    assert len([row for row in trainer.state.log_history if "eval_loss" in row]) == 3
    checkpoint = tmp_path / "run/checkpoint-3"
    assert (checkpoint / "model.safetensors").is_file()
    assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 3
    trainer.save_model(str(tmp_path / "final"))
    assert (tmp_path / "final/model.safetensors").is_file()
    args.max_steps = 4
    resumed = make_trainer()
    result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert result.global_step == 4 and torch.isfinite(torch.tensor(result.training_loss))
    assert (tmp_path / "run/checkpoint-4/model.safetensors").is_file()


@pytest.mark.parametrize("options,error", [
    ({"dataloader_num_workers": 1}, "dataloader_num_workers=0"),
    ({"accelerator_config": {"dispatch_batches": True}}, "dispatch_batches=False"),
])
def test_iterable_grpo_rejects_settings_that_corrupt_raw_prompt_batches(tiny_checkpoint, tmp_path, options, error):
    model_path, processor, image = tiny_checkpoint
    base = EpochAwareIterableDataset(lambda: dict(problem="Describe", image_path=[image], solution="red"),
                                     epoch_size=4, seed=42)
    dataset = CyclicDynamicDataset(base, num_generations=2, num_iterations=1)
    args = GRPOConfig(output_dir=str(tmp_path / "run"), use_cpu=True, report_to="none",
                      use_vllm=False, beta=0, num_generations=2, per_device_train_batch_size=2,
                      **options)
    with pytest.raises(ValueError, match=error):
        VLMGRPOTrainer(model=str(model_path), args=args, reward_funcs=[], vlm_module=Qwen2VLModule(),
                        train_dataset=dataset, processing_class=processor,
                        attn_implementation="eager", torch_dtype="float32")


def test_real_eval_does_not_trim_repeated_generations_by_raw_dataset_length(tiny_checkpoint, tmp_path, monkeypatch):
    model_path, processor, image = tiny_checkpoint
    rows = [prepare_grpo_sample(dict(problem="Describe", solution="red", image_path=[image]),
                                "{Question}") for _ in range(3)]
    args = GRPOConfig(output_dir=str(tmp_path / "run"), use_cpu=True, report_to="none", use_vllm=False,
                      beta=0, num_generations=2, per_device_train_batch_size=2, per_device_eval_batch_size=2)
    trainer = VLMGRPOTrainer(model=str(model_path), args=args, reward_funcs=[], vlm_module=Qwen2VLModule(),
                            eval_dataset=Dataset.from_list(rows), processing_class=processor,
                            attn_implementation="eager", torch_dtype="float32")
    batches = []

    def prediction(model, inputs, *args, **kwargs):
        batches.append(len(inputs))
        return torch.tensor(float(len(batches))), None, None

    monkeypatch.setattr(trainer, "prediction_step", prediction)
    gather = trainer.gather_function
    result = trainer.evaluate()
    assert batches == [2, 2, 2]
    assert result["eval_loss"] == pytest.approx(2.0)
    assert trainer.gather_function == gather
