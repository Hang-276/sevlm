#!/usr/bin/env python
"""
SFT trainer entrypoint for the self-evolving loop (positive buffer → SFT replay).

This is the real SFT entrypoint the closed loop was missing for the *self-evolve
replay schema*. The existing ``src/open_r1/sft.py`` is hard-wired to the CLEVR
**bounding-box grounding** schema (it expects ``example['solution'] == [x1, y1,
x2, y2]`` plus ``normal_caption`` and a single ``image``), so it cannot consume
the self-evolve SFT replay records, which are **multi-image + free-text
``<think>/<answer>`` completions**. Rather than mutate the grounding trainer,
this thin wrapper reuses
``trl.SFTTrainer`` (no fake/placeholder trainer) with a collator that speaks the
self-evolve replay schema.

Data config (YAML, same convention as sft.py / grpo_jsonl.py)::

    datasets:
      - json_path: /path/to/sft_replay.jsonl
        sampling_strategy: all

Each replay record provides ``problem`` (text prompt), ``completion`` (assistant
target, kept verbatim incl. ``<think>...</think><answer>...</answer>``) and
``image_path`` (a list of absolute image paths). The collator builds a single
user turn (images + problem) and an assistant turn (completion); prompt and image
tokens are masked out of the labels so only the completion is supervised.

Usage (text+vision SFT smoke)::

    python -m torch.distributed.run --nproc_per_node=1 src/open_r1/sft_jsonl.py \
        --model_name_or_path <model> --dataset_name <sft.yaml> \
        --output_dir out/sft --max_steps 1 --per_device_train_batch_size 1 \
        --use_peft true --lora_r 16 --lora_alpha 32 \
        --lora_target_modules q_proj k_proj v_proj o_proj

This script does not modify the SFT/GRPO
trainers. If the replay has no positive examples, it exits with a clear error
rather than pretending to train.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rows_from_yaml(data_path: str) -> List[Dict[str, Any]]:
    """Load SFT replay rows from a YAML data-config or a direct JSONL path."""
    p = Path(data_path)
    files: List[Path] = []
    if p.suffix in (".yaml", ".yml"):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for d in cfg.get("datasets", []):
            jp = d.get("json_path")
            if jp:
                files.append(Path(jp))
    else:
        files.append(p)

    rows: List[Dict[str, Any]] = []
    for pf in files:
        if not pf.is_file():
            raise SystemExit(f"SFT replay file not found: {pf}")
        rows.extend(_read_jsonl(pf))
    return rows


def _image_paths(record: Dict[str, Any]) -> List[str]:
    ip = record.get("image_path")
    if isinstance(ip, str):
        return [ip]
    if isinstance(ip, list):
        return [str(x) for x in ip]
    img = record.get("image")
    if isinstance(img, str):
        return [img]
    if isinstance(img, list):
        return [str(x) for x in img]
    return []


@dataclass
class SFTJsonlArguments:
    dataset_name: str = field(
        metadata={"help": "YAML data-config (datasets: - json_path: ...) or a "
                          "direct sft_replay.jsonl path."}
    )


# Module-level processor so the collator can reach it (mirrors sft.py).
processor = None


def _build_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt_text = str(record.get("problem") or record.get("prompt") or "")
    completion = str(record.get("completion") or "")
    user_content: List[Dict[str, Any]] = []
    for ip in _image_paths(record):
        user_content.append({"type": "image", "image": f"file://{ip}"})
    user_content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]


def collate_fn(examples: List[Dict[str, Any]]):
    from qwen_vl_utils import process_vision_info

    messages = [_build_messages(ex) for ex in examples]
    prompt_messages = [m[:1] for m in messages]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    prompt_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]
    image_inputs = []
    for m in prompt_messages:
        imgs, _vids = process_vision_info(m)
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    prompt_batch = processor(
        text=prompt_texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # Supervise only the assistant completion. The previous collator masked
    # padding/image tokens but still trained on the user prompt.
    for i in range(len(examples)):
        prompt_len = int(prompt_batch["attention_mask"][i].sum().item())
        active_len = int(batch["attention_mask"][i].sum().item())
        start = 0 if processor.tokenizer.padding_side == "right" else labels.shape[1] - active_len
        labels[i, start:start + min(prompt_len, active_len)] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoProcessor, set_seed
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_peft_config

    parser = TrlParser((SFTJsonlArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(getattr(training_args, "seed", 42))

    rows = _load_rows_from_yaml(script_args.dataset_name)
    if not rows:
        raise SystemExit(
            "No SFT replay records found (positive buffer empty). SFT is "
            "mandatory but cannot execute without positive trajectories."
        )
    # Keep only records that actually carry a completion and at least the prompt.
    rows = [r for r in rows if (r.get("completion") and (r.get("problem") or r.get("prompt")))]
    if not rows:
        raise SystemExit("No usable SFT replay records (need problem+completion).")
    print(f"[sft_jsonl] loaded {len(rows)} SFT replay record(s) from "
          f"{script_args.dataset_name}")

    import datasets
    train_dataset = datasets.Dataset.from_list(rows)

    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype or torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    model_type = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    ).model_type
    if "Qwen2-VL" in model_args.model_name_or_path or model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path or model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    # save_model only writes processing_class (here processor.tokenizer), so the
    # disk ends up with tokenizer files only — the image processor's
    # preprocessor_config.json / chat_template.json are missing. Qwen2.5-VL is
    # multimodal: save_model does write the full weights (vision tower included)
    # and config, but nobody saves the image-preprocessing half. The next GRPO
    # stage calls AutoProcessor.from_pretrained on this checkpoint and OSErrors
    # without it. Fill that half in from the full processor already in memory
    # (= tokenizer + image_processor) so the SFT->GRPO carry-forward stays
    # loadable. The tokenizer part matches what save_model wrote, so
    # overwriting is side-effect free.
    processor.save_pretrained(training_args.output_dir)
    print(f"[sft_jsonl] saved full processor (image+chat_template) → {training_args.output_dir}")
    print(f"[sft_jsonl] SFT training complete; checkpoint → {training_args.output_dir}")


if __name__ == "__main__":
    main()
