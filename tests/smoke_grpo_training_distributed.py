"""Two-rank FSDP2 GRPO smoke: torchrun --standalone --nproc_per_node=2 this_file."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("FSDP_VERSION", "2")
os.environ.setdefault("FSDP_STATE_DICT_TYPE", "FULL_STATE_DICT")

from datasets import Dataset
from PIL import Image
import torch
import torch.distributed as dist
from transformers import AutoProcessor, LogitsProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from open_r1.grpo_data import prepare_grpo_sample
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
from open_r1.trainer.grpo_config import GRPOConfig
from open_r1.trainer.grpo_trainer import VLMGRPOTrainer
from open_r1.vlm_modules.qwen_module import Qwen2VLModule
from test_sft_collator import make_processor


class RankEos(LogitsProcessor):
    def __init__(self, rank, eos, text_token):
        self.tokens_before_eos = 1 + 2 * rank
        self.eos, self.text_token = eos, text_token
        self.prompt_length = None

    def __call__(self, input_ids, scores):
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        token = self.eos if input_ids.shape[1] - self.prompt_length >= self.tokens_before_eos else self.text_token
        scores.fill_(float("-inf"))
        scores[:, token] = 0
        return scores


def make_checkpoint(root):
    processor = make_processor("left")
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=len(processor.tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=128, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=16, window_size=112, fullatt_block_indexes=[0]),
        image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
        vision_start_token_id=4, vision_end_token_id=5, pad_token_id=2, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(root / "tiny_qwen2_5_vl")
    processor.save_pretrained(root / "tiny_qwen2_5_vl")
    Image.new("RGB", (28, 28), "red").save(root / "red.png")


def run(root):
    rank = dist.get_rank()
    if rank == 0:
        make_checkpoint(root)
    dist.barrier()
    monkey_patch_qwen2_5vl_flash_attn()
    monkey_patch_qwen2_5vl_forward()
    processor = AutoProcessor.from_pretrained(root / "tiny_qwen2_5_vl", use_fast=True)
    rows = [prepare_grpo_sample(dict(problem=f"Describe red {i}", solution="red",
                                    image_path=[str(root / "red.png")]), "{Question}") for i in range(4)]
    dataset = Dataset.from_list(rows)

    def rank_reward(prompts, completions, **kwargs):
        gathered = [None] * 2
        dist.all_gather_object(gathered, prompts)
        assert gathered[0] == gathered[1]
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            assert len(text.split()) == 1 + 2 * rank
        return [float(rank)] * len(completions)

    fsdp_config = json.loads((REPO / "local_scripts/fsdp2_qwen2_5vl.json").read_text())
    args = GRPOConfig(
        output_dir=str(root / "run"), use_vllm=False, num_generations=2,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=2, num_iterations=2, max_steps=2, max_completion_length=4,
        beta=0, loss_type="dr_grpo", seed=42, report_to="none", disable_tqdm=True,
        save_strategy="steps", save_steps=1, eval_strategy="steps", eval_steps=1, logging_steps=1,
        fsdp="full_shard auto_wrap", fsdp_config=fsdp_config,
    )
    trainer = VLMGRPOTrainer(
        model=str(root / "tiny_qwen2_5_vl"), args=args, reward_funcs=rank_reward, vlm_module=Qwen2VLModule(),
        train_dataset=dataset, eval_dataset=dataset, processing_class=processor,
        attn_implementation="eager", torch_dtype="float32",
    )
    assert trainer.accelerator.state.fsdp_plugin.fsdp_version == 2
    original_processors = trainer._rollout_logits_processor

    def processors():
        result = original_processors()
        result.append(RankEos(rank, processor.tokenizer.eos_token_id,
                              processor.tokenizer.convert_tokens_to_ids("red" if rank == 0 else "blue")))
        return result

    trainer._rollout_logits_processor = processors
    before = trainer.model.lm_head.weight.detach().cpu().clone()
    result = trainer.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert trainer._step == 4
    after = trainer.model.lm_head.weight.full_tensor().detach().cpu()
    assert not torch.equal(before, after)
    assert len([row for row in trainer.state.log_history if "eval_loss" in row]) == 2
    trainer.save_model(str(root / "final"))
    dist.barrier()
    saved = [(root / "run/checkpoint-2/model.safetensors").is_file()
             and (root / "final/model.safetensors").is_file() if rank == 0 else None]
    dist.broadcast_object_list(saved, src=0)
    assert saved[0]
    print(f"rank={rank} PASS FSDP2 GRPO/train/eval/save with unequal EOS lengths, seed42", flush=True)


def main():
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Run with torchrun --standalone --nproc_per_node=2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        context = TemporaryDirectory(prefix="sevlm-grpo-fsdp2-") if dist.get_rank() == 0 else nullcontext(None)
        with context as folder:
            paths = [folder]
            dist.broadcast_object_list(paths, src=0)
            run(Path(paths[0]))
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
