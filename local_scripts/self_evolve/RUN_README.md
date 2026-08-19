# self_evolve — how to run things

The core logic lives in `src/open_r1/self_evolve/` (the library) and
`workflow/run_real_input_self_evolve_loop.py` (the closed-loop entry);
everything in this directory just wires those together. Run commands from the
repo root — the scripts locate themselves, so it
works from anywhere.

## Two entry points

**Full-parameter experiments (use this for the main results table)**:
`experiments/`, organized into main / ablation / sensitivity / analysis —
see `experiments/README.md`.

**The LoRA path (fast iteration; not comparable with full-parameter, never
put both in one table)**:

```
paths.sh              global paths + environment (the only file where you fill
                      in real paths — change the one WORKSPACE line)
run_loop.sh           closed-loop training entry (runs one hyperparameter config)
configs/
  grpo_h200.sh        default hyperparameters (8x H200)
  sft_grpo_h200.sh    SFT -> GRPO
  grpo_only_h200.sh   GRPO only
  smoke.sh            smoke config (no API / GPU / training — mechanism check only)
  train_defaults.sh   single source of training knobs (shared by experiments/ and run_loop.sh)
  reward/reward_weights.json   reward config (design notes in docs/REWARD.md)
```

## Quick start

```bash
# 0) Fill in WORKSPACE in paths.sh (the placeholder comments say where to
#    download each piece).

# 1) Smoke run: no API, no GPU — confirm the mechanism works end to end.
bash local_scripts/self_evolve/run_loop.sh local_scripts/self_evolve/configs/smoke.sh

# 2) Main experiment (tmux recommended).
bash local_scripts/self_evolve/experiments/main/ours.sh

# 3) Evaluate.
bash local_scripts/self_evolve/experiments/main/eval.sh <run_dir>
```

## Running a different experiment

Full-parameter exps: each wrapper in `experiments/` only changes a few
environment variables — see its README.
LoRA exps: copy `configs/grpo_h200.sh` to `configs/my_exp.sh`, edit the
hyperparameters, then:

```bash
bash local_scripts/self_evolve/run_loop.sh local_scripts/self_evolve/configs/my_exp.sh
```

Output goes to `RUNS_ROOT/<EXP_NAME>_<timestamp>/`, with the full log in
`run.log`.

## Things to know

- **Training needs no API**: task solvability comes from the solver's own
  rollouts. Only the evaluation MCQ judge wants a key, or use
  `JUDGE=exact_matching` to score fully locally.
- **Self-play is on by default** (`SELF_PLAY=1`): the model picks which changes
  each edited task keeps, then solves what it picked, and is trained on the
  picks the solver found learnable. Design notes in docs/REWARD.md;
  `main/ours_no_self_play.sh` is the comparison without it.
- **DPO**: not part of the training path — see the "Training stages"
  section of docs/REWARD.md for why.
- Before running any exp, do one pass of
  `experiments/analysis/perception_probe.sh` first (docs/REWARD.md explains
  why).
