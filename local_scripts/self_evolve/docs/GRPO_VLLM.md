# GRPO vLLM rollout

The self-evolve GRPO path uses vLLM by default. Each training rank has a
synchronous vLLM worker on the **same GPU**. Workers are separate processes to
isolate their distributed communication state from DeepSpeed/FSDP, with one
outstanding request per rank; training and generation do not run asynchronously.

For each fresh rollout batch:

1. Wake vLLM. On a new optimizer step, stream the current policy weights to it.
2. Generate one completion per repeated training sample, preserving rank/sample
   order and the existing `num_generations` grouping.
3. Put vLLM into level-1 sleep: weights move to CPU RAM and KV cache is discarded.
4. Check that vLLM's expanded prompt token IDs match the HF processor's IDs.
5. Compute rewards, reference/current log-probabilities, advantages, GRPO loss,
   gradients and optimizer updates with the existing trainer.

Weights remain unchanged across gradient-accumulation microsteps, so synchronization
is needed once per optimizer step. Dynamic resampling gets a new sampling seed.
LoRA deltas are applied to the transferred weights without merging or changing
the live training parameters. ZeRO-3 and FSDP2 weights are gathered incrementally.
Prefix caches are invalidated after every weight synchronization.

## Configuration

The shared `configs/train_defaults.sh` adds these flags for both the main loop
and experiment wrappers:

| Shell variable | Default | Meaning |
|---|---|---|
| `GRPO_USE_VLLM` | `True` | Use vLLM for GRPO generation |
| `GRPO_VLLM_MEMORY_UTILIZATION` | `0.35` | Fraction of total GPU memory budgeted for active vLLM |
| `GRPO_VLLM_MAX_MODEL_LEN` | `32768` | Prompt + completion token limit |
| `GRPO_VLLM_MAX_IMAGES` | `8` | Maximum images per prompt |
| `GRPO_VLLM_MAX_NUM_SEQS` | `4` | Concurrent sequences per GPU |

Direct `grpo_jsonl.py` arguments use `--use_vllm True`, `--vllm_device auto`,
`--vllm_gpu_memory_utilization`, `--vllm_max_model_len`, `--vllm_max_images`,
and `--vllm_max_num_seqs`. The engine uses vLLM 0.8.2 V0 with eager execution
and sleep enabled. Image processing uses the trainer's `min_pixels`/`max_pixels`.

During sampling, vLLM must fit **alongside training parameters and optimizer
state**. Sleep frees its GPU allocations for the subsequent update, but does
not remove that sampling peak. CPU RAM must also hold each rank's sleeping
vLLM weights. Tune memory utilization, image resolution and concurrency together.

Supported: the current Qwen2/2.5-VL self-evolve task path, full-parameter training
or linear LoRA, DDP/ZeRO-3/FSDP2 weight synchronization. FSDP1, DoRA,
`modules_to_save`, guided decoding and the legacy two-phase `clevr_spotdiff`
game trainer are rejected explicitly rather than silently using HF generation.
For the legacy game path or other VLMs, opt into the previous generation path
with `GRPO_USE_VLLM=False` / `--use_vllm False`.

## Verification

```bash
python -m pytest -q tests/test_vllm_rollout.py

# Requires an available GPU and a local Qwen model. Uses synthetic two-image
# inputs and a diagnostic reward to exercise two real optimizer updates.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python tests/smoke_grpo_vllm_gpu.py \
  --model /path/to/models/Qwen2.5-VL-7B-Instruct
```

The GPU smoke makes HF `generate()` raise an error, so passing requires actual
vLLM generation. It checks two training steps, nonzero finite gradients,
step-0/step-1 synchronization and worker cleanup. Per-rank engine logs are
`<output_dir>/vllm_rank<N>.log`; trainer metrics include `rollout/vllm` and
`rollout/policy_step`.
