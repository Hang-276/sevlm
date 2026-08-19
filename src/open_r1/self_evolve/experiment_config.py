"""Hard defaults for the workflow CLI; an explicit flag always wins."""

from __future__ import annotations

from typing import Any, Dict

# Keyed by argparse dest; applies when the flag is not given.
HARD_DEFAULTS: Dict[str, Any] = {
    # Paths have no sensible default; the entry scripts always pass them from
    # paths.sh, and a missing one should surface rather than point somewhere.
    "dataset_root": None,
    "model_path": None,
    "output_dir": None,
    "num_iterations": 2,
    "allow_more_than_two_iterations": False,
    "num_train_tasks": 8,
    "num_generations": 6,
    "seed": 42,
    "reference_provider": "openrouter",
    "reference_model": "openai/gpt-4o",
    "reference_base_url": None,
    "dry_run_reference_vlm": False,
    "enable_openai_reference_vlm": False,
    "openai_reference_max_tasks": 3,  # DEPRECATED (see SCHEMA note); ablation only
    "max_regenerate_attempts": 2,
    # Judge-until-quota defaults. Batch/budget are coefficients of N
    # (num_train_tasks): first-batch candidates = ceil(1.25 N), judge-call
    # ceiling = ceil(1.5 N). τ is a lenient solvability floor; ambiguity
    # rejection is OFF (CLEVR spot-diff is inherently ambiguous).
    "reference_oversample_factor": 1.25,
    "reference_judge_budget_factor": 1.5,
    "reference_solvability_threshold": 0.4,
    "reference_reject_on_ambiguity": False,
    "disable_reference_quota": False,
    "dry_run_solver": False,
    "dry_run_trainer": False,
    "max_trainer_steps": 1,
    "execute_grpo_smoke": False,
    "execute_sft_smoke": False,
    "use_lora": False,
    # Trainer scale knobs. Defaults give a self-consistent GRPO group of 6
    # (global batch per_device(6) x num_gpus(1) = 6, divisible by gen 6), the
    # mentor-recommended group size — never the degenerate group-of-2. Drop to 4
    # (and per_device 4) on memory-constrained dual-4090; see prod YAML.
    "trainer_num_gpus": 1,
    "trainer_per_device_train_batch_size": 6,
    "trainer_gradient_accumulation_steps": 1,
    "trainer_grpo_num_generations": 6,
    "trainer_deepspeed_config": None,
    # GRPO training backend: "deepspeed" (ZeRO-3, default/unchanged) or "fsdp2"
    # (PyTorch native FSDP2). Only the GRPO path honours this; SFT stays on
    # DeepSpeed. fsdp2 requires trainer_fsdp_config (the HF --fsdp_config JSON).
    "trainer_backend": "deepspeed",
    "trainer_fsdp_config": None,
    "trainer_lora_r": 16,
    "trainer_lora_alpha": 32,
    "trainer_lora_dropout": 0.05,
    "reward_config": None,
}


def resolve_settings(cli_values: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit CLI value wins; ``None`` (not given) falls back to the default."""
    return {dest: cli_values.get(dest) if cli_values.get(dest) is not None else default
            for dest, default in HARD_DEFAULTS.items()}
