"""Online vLLM sampler for self-evolve proposer and solver rollouts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    # Each rollout worker sees one GPU through CUDA_VISIBLE_DEVICES.
    max_model_len: int = 32768
    max_images: int = 8
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
# OnlineVLLMSolverSampler
# ---------------------------------------------------------------------------

class OnlineVLLMSolverSampler:
    """Generate solver trajectories online using a local vLLM engine.

    This sampler loads the model lazily — only when ``generate_one`` or
    ``sample`` is called for the first time and ``dry_run`` is False.

    Usage::

        config = OnlineSolveConfig(dry_run=True)
        sampler = OnlineVLLMSolverSampler(config)
        # dry_run path — no model loaded
        prompt_info = sampler.inspect_prompt(image_paths, prompt_text)
        print(prompt_info)

        # Real generation
        config.dry_run = False
        sampler = OnlineVLLMSolverSampler(config)
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
        self._lora_request = None

    def unload(self) -> None:
        """Free the GPU model so a downstream trainer subprocess can use the VRAM.

        The closed loop runs real_solver rollout and then launches an SFT/GRPO
        smoke in the SAME GPU. Without releasing the solver model first, both the
        rollout model and the trainer try to occupy a single 24G card and OOM.
        """
        if self._model is not None:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            del self._model
            self._model = None
            cleanup_dist_env_and_memory()
        self._processor = None
        self._lora_request = None

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._config.dry_run:
            return
        if self._model is not None:
            return

        from pathlib import Path
        from transformers import AutoProcessor
        from vllm import LLM

        t0 = time.time()
        model_path = self._config.model_path
        load_kwargs: Dict[str, Any] = {}
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.is_file():
            import json
            from vllm.lora.request import LoRARequest

            adapter = json.loads(adapter_cfg.read_text())
            self._lora_request = LoRARequest("solver", 1, model_path)
            load_kwargs.update(enable_lora=True, max_lora_rank=max(8, adapter["r"]))
            model_path = adapter.get("base_model_name_or_path") or _DEFAULT_BASE_MODEL

        self._processor = AutoProcessor.from_pretrained(model_path, **self._processor_kwargs())
        self._model = LLM(
            model=model_path,
            dtype=self._config.torch_dtype,
            tensor_parallel_size=1,
            max_model_len=self._config.max_model_len,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": self._config.max_images},
            mm_processor_kwargs=self._processor_kwargs(),
            seed=self._config.seed,
            **load_kwargs,
        )

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
        rollout_source: str = "online_vllm_smoke",
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

        from vllm import SamplingParams

        effective_seed = seed if seed is not None else (self._config.seed + gen_index)
        request = {"prompt": text}
        if image_inputs:
            request["multi_modal_data"] = {"image": image_inputs}
        t0 = time.time()
        output = self._model.generate(
            [request],
            SamplingParams(
                max_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                seed=effective_seed,
            ),
            lora_request=self._lora_request,
            use_tqdm=False,
        )[0]
        elapsed = time.time() - t0
        input_len = len(output.prompt_token_ids)
        num_generated_tokens = len(output.outputs[0].token_ids)
        completion = output.outputs[0].text.strip()

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


# Multi-GPU rollout: one vLLM engine per GPU, with disjoint task shards.

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
    rollout_source: str = "real_solver",
) -> None:
    """Worker: roll out one task shard on a single GPU, stream to out_path."""
    import json as _json
    import os as _os

    # Pin this process to exactly one visible card BEFORE importing torch cuda
    # state is touched, so "cuda:0" inside the worker maps to the intended card.
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from tqdm.auto import tqdm

    sampler = OnlineVLLMSolverSampler(
        OnlineSolveConfig(
            model_path=effective_solver_path,
            max_images=max((len(t.get("image_path") or []) for t in shard_tasks), default=1),
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
        desc=f"{rollout_source} gpu{gpu_id}",
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
                    rollout_source=rollout_source,
                )
                traj["problem"] = task.get("problem")
                traj["prompt"] = task.get("prompt") or task.get("problem")
                traj["solution"] = task.get("solution")
                traj["ground_truth"] = task.get("ground_truth")
                traj["scene_id"] = task.get("scene_id")
                traj["base_task_id"] = task.get("base_task_id")
                traj["reference_reasoning"] = task.get("reference_reasoning")
                traj["rollout_source"] = rollout_source
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
    rollout_source: str = "real_solver",
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

    log(f"  {rollout_source}: data-parallel rollout over {num_gpus} GPU(s) "
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
                rollout_source,
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
