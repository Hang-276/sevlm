#!/usr/bin/env python
"""Merge a LoRA/PEFT adapter into its base model and save a full model dir.

Standalone, side-effect-only script used by the self-evolve runner as a
*training pre-step* so the solver can accumulate weights across iterations
(merge-then-fresh-adapter). It is deliberately a separate process: it loads a
full 7B model to merge, and exiting the process is what guarantees the GPU/CPU
memory is released before the GRPO trainer subprocess starts.

It mirrors ``open_r1.self_evolve.online_solver``'s adapter-aware loading:
  1. read ``base_model_name_or_path`` from ``<adapter>/adapter_config.json``
  2. load that base with ``Qwen2_5_VLForConditionalGeneration``
  3. attach the adapter with ``PeftModel.from_pretrained``
  4. ``merge_and_unload()`` -> full model
  5. ``save_pretrained(output_dir, safe_serialization=True)``
  6. best-effort save tokenizer + processor from the base

Exit code 0 == merge succeeded and a full model dir exists. Any failure to save
the *model* is fatal (non-zero). Tokenizer/processor failures are warned but do
NOT mask a successful model save.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-path", required=True,
                    help="PEFT adapter dir (contains adapter_config.json).")
    ap.add_argument("--output-dir", required=True,
                    help="Destination dir for the merged full model.")
    ap.add_argument("--torch-dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--trust-remote-code", action="store_true", default=False)
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    adapter_cfg_path = adapter_path / "adapter_config.json"
    if not adapter_cfg_path.is_file():
        print(f"[merge_lora] ERROR: no adapter_config.json under {adapter_path}",
              file=sys.stderr)
        return 2

    base_model_name_or_path = json.loads(
        adapter_cfg_path.read_text()
    ).get("base_model_name_or_path")
    if not base_model_name_or_path:
        print(f"[merge_lora] ERROR: adapter_config.json has no "
              f"base_model_name_or_path: {adapter_cfg_path}", file=sys.stderr)
        return 2

    if output_dir.exists():
        if not args.overwrite:
            print(f"[merge_lora] ERROR: output-dir exists and --overwrite not "
                  f"set: {output_dir}", file=sys.stderr)
            return 2
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: F401  (imported for side effects / availability check)
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )
    from peft import PeftModel

    torch_dtype = _resolve_dtype(args.torch_dtype)
    trc = bool(args.trust_remote_code)

    print(f"[merge_lora] adapter_path={adapter_path}")
    print(f"[merge_lora] base_model_name_or_path={base_model_name_or_path}")
    print(f"[merge_lora] output_dir={output_dir}")
    print(f"[merge_lora] loading base model ({args.torch_dtype})…")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trc,
    )
    print(f"[merge_lora] attaching adapter and merging…")
    merged = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = merged.merge_and_unload()

    # Fatal if the model itself fails to save.
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    # Best-effort tokenizer + processor (warn, do not mask a good model save).
    try:
        AutoTokenizer.from_pretrained(
            base_model_name_or_path, trust_remote_code=trc
        ).save_pretrained(str(output_dir))
    except Exception as e:  # noqa: BLE001
        print(f"[merge_lora] WARNING: could not save tokenizer: {e}",
              file=sys.stderr)
    try:
        AutoProcessor.from_pretrained(
            base_model_name_or_path, trust_remote_code=trc
        ).save_pretrained(str(output_dir))
    except Exception as e:  # noqa: BLE001
        print(f"[merge_lora] WARNING: could not save processor: {e}",
              file=sys.stderr)

    has_config = (output_dir / "config.json").is_file()
    weight_files = sorted(
        list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.bin"))
    )
    print(f"[merge_lora] done. config.json={has_config} "
          f"weight_files={len(weight_files)}")
    if not has_config or not weight_files:
        print("[merge_lora] ERROR: merged dir missing config.json or weights.",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
