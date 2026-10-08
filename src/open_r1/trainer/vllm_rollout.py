"""Synchronous, same-GPU vLLM rollouts for the self-evolve GRPO trainer.

Each training rank owns a separate process on its own GPU. The process boundary
isolates vLLM's distributed state from DeepSpeed/FSDP; it is not async training.
vLLM sleeps between rollouts, releasing weights and KV-cache GPU allocations.
"""

import atexit
from contextlib import nullcontext
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import torch


def _full_tensor(tensor):
    # FSDP2 parameters are DTensors. All training ranks must enter full_tensor.
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def iter_policy_weights(model):
    """Yield HF-named, CPU weights, including effective LoRA linear weights.

    ZeRO-3 gathers one parameter (or one LoRA layer) at a time. FSDP2 similarly
    materializes one tensor at a time. Never merge adapters into live parameters:
    that can change the optimizer's weights or introduce rounding drift.
    """
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer

    if isinstance(model, PeftModel):
        configs = model.peft_config.values()
        if any(c.peft_type != "LORA" or c.modules_to_save for c in configs):
            raise ValueError("vLLM rollout supports linear LoRA without modules_to_save.")
        model = model.get_base_model()
    modules = dict(model.named_modules())
    for name, param in model.named_parameters():
        if ".lora_" in name:
            continue
        layer = None
        if ".base_layer." in name:
            layer_name, leaf = name.rsplit(".base_layer.", 1)
            layer = modules[layer_name]
            if not isinstance(layer, LoraLayer) or not isinstance(layer.base_layer, torch.nn.Linear):
                raise ValueError("vLLM rollout currently supports LoRA on Linear layers only.")
            if layer.merged:
                raise ValueError("Unmerge the training LoRA adapter before synchronizing vLLM.")
            name = f"{layer_name}.{leaf}"
        params = [param]
        adapters = []
        if layer is not None and leaf == "weight" and not layer.disable_adapters:
            for adapter in layer.active_adapters:
                if adapter not in layer.lora_A:
                    continue
                if layer.use_dora.get(adapter, False):
                    raise ValueError("DoRA is not supported by the GRPO vLLM weight synchronizer.")
                a, b = layer.lora_A[adapter].weight, layer.lora_B[adapter].weight
                params.extend([a, b])
                adapters.append((a, b, layer.scaling[adapter]))
        context = nullcontext()
        if any(hasattr(p, "ds_id") for p in params):
            from deepspeed import zero

            context = zero.GatheredParameters(params)
        with context, torch.no_grad():
            weight = _full_tensor(param).detach()
            if adapters:
                weight = weight.clone()
                for a, b, scaling in adapters:
                    delta = _full_tensor(b) @ _full_tensor(a)
                    if layer.fan_in_fan_out:
                        delta = delta.T
                    weight.add_(delta.to(weight.dtype), alpha=scaling)
            yield name, weight.to(device="cpu", copy=True).contiguous()


def build_requests(prompts, images_by_prompt):
    if len(prompts) != len(images_by_prompt):
        raise ValueError("Every vLLM prompt must have its own image list.")
    return [
        {"prompt": prompt, **({"multi_modal_data": {"image": images}} if images else {})}
        for prompt, images in zip(prompts, images_by_prompt)
    ]


def finite_rollout_logits(scores, banned_token_ids):
    """Scrub non-finite logits and exclude only in-vocabulary token ids."""
    if not torch.isfinite(scores).all():
        floor = torch.finfo(scores.dtype).min
        scores = torch.nan_to_num(scores, nan=floor, posinf=1e4, neginf=floor)
        bad_rows = scores.max(dim=-1).values <= floor
        scores[bad_rows] = 0.0
    banned = [token for token in banned_token_ids if 0 <= token < scores.shape[-1]]
    scores[..., banned] = float("-inf")
    if not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError("No allowed finite rollout token remains")
    return scores


def eos_completion_mask(completion_ids, eos_token_ids):
    """Include the first EOS and mark rows without one as truncated."""
    if eos_token_ids is None:
        is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    else:
        ids = torch.as_tensor(eos_token_ids, device=completion_ids.device).reshape(-1)
        is_eos = torch.isin(completion_ids, ids)
    before_eos = is_eos.int().cumsum(dim=1) - is_eos.int()
    return (before_eos == 0).int(), ~is_eos.any(dim=1)


def completion_tensors(outputs, expected_prompts, pad_token_id, device):
    """Validate HF/vLLM image-token expansion, then pad only the completions."""
    if len(outputs) != len(expected_prompts):
        raise RuntimeError("vLLM returned a different number of completions than training inputs.")
    for output, expected in zip(outputs, expected_prompts):
        if output["prompt_token_ids"] != expected:
            raise RuntimeError("HF/vLLM prompt token mismatch; check image processing and chat templates.")
        if not output["token_ids"]:
            raise RuntimeError("vLLM returned an empty completion.")
        if output["finish_reason"] not in {"stop", "length"}:
            raise RuntimeError(f"vLLM completion did not finish normally: {output['finish_reason']!r}")
    lengths = torch.tensor([len(o["token_ids"]) for o in outputs], device=device)
    ids = torch.full((len(outputs), int(lengths.max())), pad_token_id, dtype=torch.long, device=device)
    for row, output in zip(ids, outputs):
        row[: len(output["token_ids"])] = torch.tensor(output["token_ids"], device=device)
    mask = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
    truncated = torch.tensor([o["finish_reason"] == "length" for o in outputs], device=device)
    return ids, mask.int(), truncated


class ColocatedVLLMRollout:
    def __init__(self, *, device_index, engine_kwargs, log_path, timeout=600):
        if device_index is None:
            device_index = torch.cuda.current_device()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        device = visible.split(",")[device_index].strip() if visible else str(device_index)
        env = os.environ.copy()
        # The worker has its own single-rank process group, on the same GPU.
        for key in list(env):
            if key in {
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
            } or key.startswith("TORCHELASTIC_"):
                env.pop(key)
        env.update(CUDA_VISIBLE_DEVICES=device, VLLM_USE_V1="0", VLLM_WORKER_MULTIPROC_METHOD="spawn")
        parent, child = multiprocessing.Pipe()
        self.connection = parent
        self.timeout = timeout
        self.last_step = None
        self.log_path = str(log_path)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("vllm_rollout_worker.py")), str(child.fileno())],
                env=env,
                pass_fds=(child.fileno(),),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        child.close()
        atexit.register(self.close)
        try:
            self.request("init", engine_kwargs)
        except BaseException:
            self.close()
            raise

    def request(self, command, payload=None):
        if self.process.poll() is not None:
            raise RuntimeError(f"vLLM worker exited ({self.process.returncode}); see {self.log_path}")
        self.connection.send((command, payload))
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"vLLM worker timed out during {command}; see {self.log_path}")
        try:
            ok, result = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"vLLM worker died during {command}; see {self.log_path}") from exc
        if not ok:
            raise RuntimeError(f"vLLM {command} failed: {result}\nWorker log: {self.log_path}")
        return result

    def generate(self, model, step, requests, sampling):
        from safetensors.torch import save

        torch.cuda.empty_cache()
        self.request("wake")
        try:
            if self.last_step != step:
                count = 0
                for name, weight in iter_policy_weights(model):
                    self.request("weight", save({name: weight}))
                    count += 1
                self.request("synced", step)
                self.last_step = step
                print(f"[GRPO vLLM] synchronized policy step={step}, tensors={count}", flush=True)
            return self.request("generate", (requests, sampling))
        finally:
            self.request("sleep")

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self.connection.send(("close", None))
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.connection.close()
        atexit.unregister(self.close)
