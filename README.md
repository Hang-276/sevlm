# Self-evolving VLM visual reasoning agent

A closed training loop on the CLEVR spot-the-difference task. The model sets
its own puzzles and then solves them: it picks which of a scene's changed
objects stay changed, splicing turns that pick into a task whose answer is
computed rather than guessed, and it is trained on the picks that turned out
to sit at the edge of what it can solve. The solver trains with GRPO, with an
SFT replay pass over its own correct trajectories.

## Where everything is

Everything you run lives in one directory:

```
local_scripts/self_evolve/
```

Start there:

| | |
|---|---|
| `RUN_README.md` | the two entry points, and how to start |
| `paths.sh` | **the one file you must edit** — set `WORKSPACE` |
| `experiments/README.md` | every experiment: main, ablation, sensitivity, analysis |
| `docs/REWARD.md` | how the reward and the task generator work, and why |
| `docs/EVALUATION.md` | the evaluation protocol |

The library the loop is built from is `src/open_r1/self_evolve/`.

## Getting started

```bash
cd <repo root>
E=local_scripts/self_evolve

# 1. Install: read setup.sh (it opens with commented-out conda create lines),
#    then set CONDA_ENV in $E/paths.sh to whatever you named the environment.
# 2. Fill in WORKSPACE in $E/paths.sh. Every other path derives from it, and
#    running with the placeholder still in place fails immediately and says so.

# 3. Smoke run: no API, no GPU, no training — checks the loop end to end.
bash $E/run_loop.sh $E/configs/smoke.sh

# 4. Can the base model see these changes at all? Run this before anything
#    else; everything about the reward depends on the answer.
bash $E/experiments/analysis/perception_probe.sh

# 5. The main experiment, then evaluation.
bash $E/experiments/main/ours.sh
bash $E/experiments/main/eval.sh
```

Online proposer/solver rollouts and the self-evolve GRPO trainer's generation
step use vLLM 0.8.2 (installed by `setup.sh`). GRPO synchronizes the current
policy weights before sampling, including LoRA deltas; rewards, log-probabilities,
loss and optimizer updates still use PyTorch/Transformers. Each training rank
runs vLLM synchronously on its own training GPU, then puts it to sleep to release
GPU weights and KV cache before the training forward/backward pass. This needs
no separate rollout node or reserved GPU. See [GRPO rollout configuration](local_scripts/self_evolve/docs/GRPO_VLLM.md).

Outer-loop rollout workers each
load one model on one GPU, with the image limit set from the task shard and
a 32768-token context. The standalone sampler defaults to 8 images.
VLMEvalKit evaluation also defaults to vLLM (`USE_VLLM=0` switches it back).

Training needs no API. Only the evaluation MCQ judge wants a key, and
`JUDGE=exact_matching` scores locally instead.

## What to watch on the first real run

- `parse_rate` — whether the model returns proposals in the expected format
- `subset_diversity` — a proposer that collapses onto one pick has stopped
  being an opponent
- `seeds` in the solvability line — if it stays 0, the curriculum never starts,
  and the perception probe is the first place to look

`experiments/analysis/collapse_diagnostics.sh <run_dir>` reads all of these
off a finished run.

## Origin

Built on [Vision-Zero](https://github.com/wangqinsi1/Vision-Zero) (MIT) and
[open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
(Apache-2.0); the per-file headers under `src/open_r1/` carry the original
notices.
