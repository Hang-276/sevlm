"""Opt-in real-model smoke: two-image vLLM sampling + two GRPO updates.

Run from the repo root with a local Qwen2.5-VL model; no model download:
python tests/smoke_grpo_vllm_gpu.py --model /path/to/Qwen2.5-VL-7B-Instruct
The reward alternates 0/1 solely to exercise the optimizer, not task quality.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from peft import LoraConfig
from PIL import Image
import torch

from open_r1.trainer import GRPOConfig, VLMGRPOTrainer
from open_r1.vlm_modules.qwen_module import Qwen2VLModule


def smoke_reward(completions, **kwargs):
    return [float(i % 2) for i in range(len(completions))]


def no_transformers_rollout(*args, **kwargs):
    raise AssertionError("GRPO fell back to Transformers generation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="tmp/grpo-vllm-smoke")
    parser.add_argument("--deepspeed")
    parser.add_argument("--full-parameter", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for i, color in enumerate(["red", "blue"]):
        path = output / f"image{i}.png"
        Image.new("RGB", (320, 240), color).save(path)
        images.append(str(path))
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "image", "text": None},
                {"type": "text", "text": "Compare the colors in these two images."},
            ],
        }
    ]
    data = Dataset.from_list(
        [dict(prompt=prompt, image_path=images, solution="different", problem="colors") for _ in range(4)]
    )
    config = GRPOConfig(
        output_dir=str(output),
        per_device_train_batch_size=2,
        num_generations=2,
        max_steps=2,
        learning_rate=1e-4,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=None,
        max_completion_length=8,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_len=512,
        vllm_max_images=2,
        vllm_max_num_seqs=2,
        dynamic_sampling=False,
        overlong_filtering=False,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        deepspeed=args.deepspeed,
    )
    trainer = VLMGRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=smoke_reward,
        vlm_module=Qwen2VLModule(),
        train_dataset=data,
        peft_config=None if args.full_parameter else LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM"),
        freeze_vision_modules=True,
        attn_implementation="sdpa",
        min_pixels=3136,
        max_pixels=12544,
        script_args=SimpleNamespace(data_generator_type=None),
    )
    trainer.model.generate = no_transformers_rollout
    trainer.train()
    assert trainer.state.global_step == 2
    assert trainer._vllm_rollout is None, "vLLM process was not cleaned up"
    history = [r for r in trainer.state.log_history if "grad_norm" in r]
    assert history and all(torch.isfinite(torch.tensor(r["grad_norm"])) for r in history)
    assert any(r["grad_norm"] > 0 for r in history), "No optimizer gradient was exercised"
    log = (output / f"vllm_rank{trainer.accelerator.process_index}.log").read_text()
    assert "policy_step=0" in log and "policy_step=1" in log
    print("PASS: two-image vLLM GRPO, two optimizer steps, step 0/1 weight sync, finite nonzero gradients, cleanup")


if __name__ == "__main__":
    main()
