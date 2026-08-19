"""Online Transformers-based solver sampler for smoke testing.

This module provides ``OnlineTransformersSolverSampler``, which uses
HuggingFace Transformers (NOT vLLM) to run a single-task / few-task
online rollout against a local Qwen2.5-VL-7B model.

**What this IS:**
- A minimal online solver smoke test.
- Generates 1 trajectory from 1 task prompt using a local VLM.

**What this is NOT:**
- A training pipeline.
- A batch inference engine.
- A replacement for offline replay.
- Connected to SFT / GRPO.
- Connected to reward computation or buffer routing.
- Connected to DINOv2 grounding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import os as _os

# Single source for the base solver model when nothing more specific is given.
# The closed-loop driver exports SELF_EVOLVE_BASE_MODEL (= --model-path) so the
# solver never falls back to a machine-specific hardcoded location.
_DEFAULT_BASE_MODEL = _os.environ.get(
    "SELF_EVOLVE_BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
)


@dataclass
class OnlineSolveConfig:
    """Configuration for a single online solver smoke run."""

    model_path: str = _DEFAULT_BASE_MODEL
    device_map: str = "auto"
    # If set (e.g. "cuda:0"), the whole model is loaded onto that single device
    # and `device_map` is ignored. Used by the multi-GPU data-parallel rollout
    # path so 7B/bf16 fits on one card and the other cards run independent
    # sampler processes over disjoint task shards.
    device: Optional[str] = None
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    # Visual token budget. Must match what GRPO trains with, or the buffers are
    # scored at a different resolution than the policy is optimized at.
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    seed: int = 42
    # If True, only construct the prompt and check paths — do NOT load the model.
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_qwen_vl_messages(
    image_paths: List[str],
    prompt_text: str,
) -> List[Dict[str, Any]]:
    """Build Qwen2.5-VL chat messages for a multi-image task.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to the player images.
    prompt_text : str
        The text prompt (e.g. the ``problem`` field from the raw trajectory).

    Returns
    -------
    list[dict]
        Messages in the Qwen2.5-VL conversation format.
    """
    content: List[Dict[str, Any]] = []
    for ip in image_paths:
        content.append({"type": "image", "image": f"file://{ip}"})
    content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# OnlineTransformersSolverSampler
# ---------------------------------------------------------------------------

class OnlineTransformersSolverSampler:
    """Generate solver trajectories online using a local HF Transformers model.

    This sampler loads the model lazily — only when ``generate_one`` or
    ``sample`` is called for the first time and ``dry_run`` is False.

    Usage::

        config = OnlineSolveConfig(dry_run=True)
        sampler = OnlineTransformersSolverSampler(config)
        # dry_run path — no model loaded
        prompt_info = sampler.inspect_prompt(image_paths, prompt_text)
        print(prompt_info)

        # Real generation
        config.dry_run = False
        sampler = OnlineTransformersSolverSampler(config)
        trajectory = sampler.generate_one(
            task_id="task_0",
            image_paths=["/abs/path/img1.png", ...],
            prompt_text="There are 3 players...",
            metadata={},
        )
    """

    def __init__(self, config: OnlineSolveConfig):
        self._config = config
        self._model = None
        self._processor = None

    def unload(self) -> None:
        """Free the GPU model so a downstream trainer subprocess can use the VRAM.

        The closed loop runs real_solver rollout and then launches an SFT/GRPO
        smoke in the SAME GPU. Without releasing the solver model first, both the
        rollout model and the trainer try to occupy a single 24G card and OOM.
        """
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        try:
            import gc

            import torch as _torch

            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
        except Exception:
            pass

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._config.dry_run:
            return
        if self._model is not None:
            return

        _check_dependencies()

        from pathlib import Path as _Path
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        t0 = time.time()
        torch_dtype = _resolve_dtype(self._config.torch_dtype)

        model_path = self._config.model_path
        # LoRA-adapter awareness: a self-evolve SFT/GRPO checkpoint is a PEFT
        # adapter directory (adapter_config.json + adapter_model.safetensors), NOT
        # a full model. Load the adapter's base model, then attach the adapter, so
        # the closed loop can roll out with the *trained* solver instead of
        # silently falling back to base.
        single_device = self._config.device
        load_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
        if single_device is None:
            load_kwargs["device_map"] = self._config.device_map

        adapter_cfg = _Path(model_path) / "adapter_config.json"
        if adapter_cfg.is_file():
            import json as _json
            from peft import PeftModel

            base_path = _json.loads(adapter_cfg.read_text()).get(
                "base_model_name_or_path"
            ) or _DEFAULT_BASE_MODEL
            print(f"[online_solver] loading LoRA adapter from {model_path} "
                  f"on base {base_path} "
                  f"(device={single_device or self._config.device_map})")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path,
                **load_kwargs,
            )
            self._model = PeftModel.from_pretrained(base_model, model_path)
            self._processor = AutoProcessor.from_pretrained(base_path, **self._processor_kwargs())
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs,
            )
            self._processor = AutoProcessor.from_pretrained(model_path, **self._processor_kwargs())
        if single_device is not None:
            self._model = self._model.to(single_device)
        # Important: the processor needs padding_side = "left" for generation.
        self._processor.tokenizer.padding_side = "left"

        elapsed = time.time() - t0
        print(f"[online_solver] model loaded in {elapsed:.1f}s")

    # -- prompt inspection (no model needed) ---------------------------------

    def inspect_prompt(
        self,
        image_paths: List[str],
        prompt_text: str,
    ) -> Dict[str, Any]:
        """Return the prompt structure without loading the model.

        Useful for ``--dry_run`` validation.
        """
        from pathlib import Path

        missing: List[str] = []
        for ip in image_paths:
            if not Path(ip).is_file():
                missing.append(ip)

        messages = _format_qwen_vl_messages(image_paths, prompt_text)

        return {
            "num_images": len(image_paths),
            "image_paths": image_paths,
            "all_images_exist": len(missing) == 0,
            "missing_images": missing,
            "prompt_text": prompt_text,
            "messages": messages,
        }

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def _processor_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self._config.min_pixels is not None:
            kwargs["min_pixels"] = int(self._config.min_pixels)
        if self._config.max_pixels is not None:
            kwargs["max_pixels"] = int(self._config.max_pixels)
        return kwargs

    def generate_one(
        self,
        task_id: str,
        image_paths: List[str],
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        gen_index: int = 0,
        seed: Optional[int] = None,
        diagnostic_only: bool = True,
        rollout_source: str = "online_transformers_smoke",
    ) -> Dict[str, Any]:
        """Generate a single trajectory from one task.

        Parameters
        ----------
        task_id : str
            Identifier for this task (e.g. ``clevr_epoch_0_sample_0``).
        image_paths : list[str]
            Absolute paths to the image files for this task.
        prompt_text : str
            The text prompt.
        metadata : dict or None
            Additional metadata to attach to the output record.
        gen_index : int
            Index of this trajectory within the per-task sample group (used to
            disambiguate trajectory_id and to derive a per-sample seed).
        seed : int or None
            Explicit decoding seed. When None, falls back to
            ``config.seed + gen_index`` so multiple generations diverge.
        diagnostic_only : bool
            When True the record is flagged as a diagnostic smoke (the default,
            used by the standalone smoke). The closed-loop real rollout sets this
            False because the trajectory is consumed by reward/buffer routing.
        rollout_source : str
            Tag identifying how the trajectory was produced.

        Returns
        -------
        dict
            A trajectory record compatible with the offline self-evolve pipeline.
        """
        if self._config.dry_run:
            raise RuntimeError(
                "generate_one called with dry_run=True. "
                "Set dry_run=False to run real generation."
            )

        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None

        messages = _format_qwen_vl_messages(image_paths, prompt_text)

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision inputs
        from qwen_vl_utils import process_vision_info

        image_inputs, _video_inputs = process_vision_info(messages)

        model_inputs = self._processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # Set seed for reproducibility. Per-sample seed so multiple generations
        # for the same task diverge instead of producing identical text.
        from transformers import set_seed
        effective_seed = seed if seed is not None else (self._config.seed + gen_index)
        set_seed(effective_seed)

        # Generate
        t0 = time.time()
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            do_sample=self._config.temperature > 0,
        )
        elapsed = time.time() - t0

        # Decode (strip the input prompt)
        input_len = model_inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        num_generated_tokens = int(generated_ids_trimmed.shape[1])
        completion = self._processor.tokenizer.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
        ).strip()

        reasoning, final_answer = _split_reasoning_answer(completion)

        trajectory_id = f"{task_id}::{rollout_source}_{gen_index}"
        return {
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "prompt": prompt_text,
            "completion": completion,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "image_path": image_paths,
            "model_path": self._config.model_path,
            "metadata": dict(metadata or {}),
            "rollout_source": rollout_source,
            "diagnostic_only": diagnostic_only,
            "generation_info": {
                "gen_index": gen_index,
                "seed": effective_seed,
                "max_new_tokens": self._config.max_new_tokens,
                "num_generated_tokens": num_generated_tokens,
                "input_tokens": int(input_len),
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "generation_time_s": round(elapsed, 2),
            },
        }

    def sample(
        self,
        tasks: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate ``num_generations`` trajectories per task (smoke-scale only)."""
        results: List[Dict[str, Any]] = []
        for task in tasks:
            for g in range(num_generations):
                traj = self.generate_one(
                    task_id=str(task["task_id"]),
                    image_paths=list(task["image_paths"]),
                    prompt_text=str(task["prompt_text"]),
                    metadata=task.get("metadata"),
                    gen_index=g,
                )
                results.append(traj)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_reasoning_answer(completion: str) -> tuple[Optional[str], Optional[str]]:
    """Extract <think> reasoning and <answer> final answer from a completion.

    Returns (reasoning, final_answer); either may be None if the tag is absent.
    """
    reasoning = None
    final_answer = None
    if "<think>" in completion and "</think>" in completion:
        reasoning = completion.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "<answer>" in completion and "</answer>" in completion:
        final_answer = completion.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return reasoning, final_answer


def _resolve_dtype(name: str) -> torch.dtype:
    mapping: Dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": torch.bfloat16,
    }
    return mapping.get(name, torch.bfloat16)


def _check_dependencies() -> None:
    missing: List[str] = []
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        missing.append("qwen_vl_utils")
    if missing:
        raise ImportError(
            f"OnlineTransformersSolverSampler requires: {', '.join(missing)}. "
            "Install them with: pip install transformers qwen-vl-utils"
        )


# ---------------------------------------------------------------------------
# Multi-GPU data-parallel rollout
# ---------------------------------------------------------------------------
#
# The single-process sampler above loads the 7B model with device_map="auto",
# which shards ONE model across all cards (model parallel): only one card
# computes at a time and the rollout runs one task at a time, so utilisation is
# low and wall-clock is ~N_tasks * N_gen serial generates.
#
# generate_rollouts_multi_gpu() instead spawns one worker process per GPU, each
# loading a FULL model copy onto its own single card (data parallel), and splits
# the accepted tasks into disjoint shards. ~N_gpu speedup. Each worker streams
# its trajectories to a per-worker JSONL shard on disk (robust to the parent not
# draining a Queue); the parent concatenates shards in task order afterwards.

def _rollout_worker(
    rank: int,
    gpu_id: int,
    shard_tasks: List[Dict[str, Any]],
    num_generations: int,
    effective_solver_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    out_path: str,
    progress_position: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Worker: roll out one task shard on a single GPU, stream to out_path."""
    import json as _json
    import os as _os

    # Pin this process to exactly one visible card BEFORE importing torch cuda
    # state is touched, so "cuda:0" inside the worker maps to the intended card.
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from tqdm.auto import tqdm

    sampler = OnlineTransformersSolverSampler(
        OnlineSolveConfig(
            model_path=effective_solver_path,
            device="cuda:0",  # only one card is visible; it's index 0 here
            dry_run=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    )

    total = len(shard_tasks) * num_generations
    pbar = tqdm(
        total=total,
        desc=f"solver gpu{gpu_id}",
        unit="traj",
        position=progress_position,
        dynamic_ncols=True,
        mininterval=2.0,
        leave=False,
    )
    with open(out_path, "w") as fh:
        for task in shard_tasks:
            img_paths = task.get("image_path") or []
            for g in range(num_generations):
                traj = sampler.generate_one(
                    task_id=str(task["task_id"]),
                    image_paths=list(img_paths),
                    prompt_text=str(task.get("prompt") or task.get("problem") or ""),
                    metadata={**task.get("metadata", {}), "self_evolve_task": task},
                    gen_index=g,
                    diagnostic_only=False,
                    rollout_source="real_solver",
                )
                traj["problem"] = task.get("problem")
                traj["prompt"] = task.get("prompt") or task.get("problem")
                traj["solution"] = task.get("solution")
                traj["ground_truth"] = task.get("ground_truth")
                traj["scene_id"] = task.get("scene_id")
                traj["base_task_id"] = task.get("base_task_id")
                traj["reference_reasoning"] = task.get("reference_reasoning")
                traj["rollout_source"] = "real_solver"
                traj["solver_model_path"] = effective_solver_path
                fh.write(_json.dumps(traj, ensure_ascii=False) + "\n")
                fh.flush()
                pbar.update(1)
    pbar.close()
    sampler.unload()


def generate_rollouts_multi_gpu(
    accepted_tasks: List[Dict[str, Any]],
    *,
    num_generations: int,
    effective_solver_path: str,
    num_gpus: int,
    scratch_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    seed: int = 42,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    log: Any = print,
) -> List[Dict[str, Any]]:
    """Data-parallel rollout across ``num_gpus`` cards.

    Splits ``accepted_tasks`` round-robin into ``num_gpus`` shards, spawns one
    worker per shard/card, and returns the concatenated trajectories (shard 0
    first, then 1, ...). Worker output shards are written under ``scratch_dir``.
    Falls back to a single in-process sampler when num_gpus <= 1 or there is
    only one task.
    """
    import json as _json
    import os as _os
    import torch.multiprocessing as mp

    _os.makedirs(scratch_dir, exist_ok=True)

    # Round-robin split keeps shards balanced in count even if tasks vary.
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_gpus)]
    for i, task in enumerate(accepted_tasks):
        shards[i % num_gpus].append(task)

    # Resolve the actual physical GPU ids from CUDA_VISIBLE_DEVICES if the parent
    # set it (the launcher exports e.g. "0,1,2,3,4,5,6,7"). Worker then re-pins to
    # a single one of these. If unset, assume contiguous 0..num_gpus-1.
    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        gpu_ids = [g.strip() for g in visible.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(num_gpus)]
    gpu_ids = gpu_ids[:num_gpus]

    out_paths = [
        _os.path.join(scratch_dir, f"rollout_shard_{r}.jsonl")
        for r in range(num_gpus)
    ]

    log(f"  Solver: data-parallel rollout over {num_gpus} GPU(s) "
        f"(gpu_ids={gpu_ids}); shard sizes={[len(s) for s in shards]}")

    # spawn (not fork): each worker gets a clean CUDA context. Explicit Process
    # objects (rather than mp.spawn) so each worker can take its own gpu_id/shard.
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(num_gpus):
        if not shards[r]:
            continue
        p = ctx.Process(
            target=_rollout_worker,
            args=(
                r,
                int(gpu_ids[r]),
                shards[r],
                num_generations,
                effective_solver_path,
                max_new_tokens,
                temperature,
                top_p,
                seed,
                out_paths[r],
                r,  # progress bar position
                min_pixels,
                max_pixels,
            ),
        )
        p.start()
        procs.append((r, p))

    failed = []
    for r, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((r, p.exitcode))
    if failed:
        raise RuntimeError(
            f"multi-GPU rollout: worker(s) failed with non-zero exit: {failed}"
        )

    # Concatenate shards in rank order.
    trajectories: List[Dict[str, Any]] = []
    for r in range(num_gpus):
        if not shards[r]:
            continue
        with open(out_paths[r]) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trajectories.append(_json.loads(line))
    return trajectories
