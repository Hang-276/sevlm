"""Fingerprint the executable self-evolve recipe before resuming a run."""

from __future__ import annotations

import hashlib
from pathlib import Path


def self_evolve_protocol_fingerprint(reward_config_path: str | Path) -> str:
    """Hash reward settings and code that changes tasks, scores or training."""
    repo = Path(__file__).resolve().parents[3]
    sources = sorted((repo / "src/open_r1/self_evolve").glob("*.py"))
    sources += [
        repo / "src/open_r1/qwen_pixels.py",
        repo / "src/open_r1/qwen2_5vl_monkey_patch.py",
        repo / "src/open_r1/sft_jsonl.py",
        repo / "src/open_r1/grpo_jsonl.py",
        repo / "src/open_r1/grpo_data.py",
        repo / "src/open_r1/trainer/grpo_config.py",
        repo / "src/open_r1/trainer/grpo_loss.py",
        repo / "src/open_r1/trainer/dynamic_dataset.py",
        repo / "src/open_r1/trainer/grpo_trainer.py",
        repo / "src/open_r1/trainer/vllm_grpo_trainer.py",
        repo / "src/open_r1/trainer/vllm_rollout.py",
        repo / "src/open_r1/trainer/vllm_rollout_worker.py",
        repo / "src/open_r1/vlm_modules/qwen_module.py",
        repo / "local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py",
        repo / "local_scripts/self_evolve/configs/train_defaults.sh",
        Path(reward_config_path),
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
