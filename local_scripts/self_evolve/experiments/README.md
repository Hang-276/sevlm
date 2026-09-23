# experiments

Four directories, one per kind of experiment. `lib/` holds the runner that
does the real work; every other script is a dozen-line wrapper that sets a
few environment variables and execs it — so whatever differs between two
exps, you can see it right in the wrapper.

```
lib/         shared runner + tools (not run directly)
main/        one script per row of the main results table
ablation/    knock out one component of the method at a time
sensitivity/ sweep a single knob across a range of values
analysis/    experiments that explain the mechanism; not in the main table
```

Run everything from the repo root; the scripts locate
themselves. Shorthand used below:

```bash
cd <repo root>
E=local_scripts/self_evolve/experiments
```

## main

| exp | script |
|---|---|
| base zero-shot | `main/base_zeroshot.sh` |
| base + GRPO | `main/grpo_baseline.sh` |
| base + vision-zero | `main/vision_zero_baseline.sh` |
| base + ours | `main/ours.sh` |
| base + ours w/o self-play | `main/ours_no_self_play.sh` |
| base + ours w/o process reward | `main/ours_no_process.sh` |

```bash
tmux new -s run
bash $E/main/ours.sh                 && bash $E/main/eval.sh
bash $E/main/grpo_baseline.sh        && MARKER=grpo_baseline bash $E/main/eval.sh
bash $E/main/vision_zero_baseline.sh && MARKER=vision_zero  bash $E/main/eval.sh
bash $E/main/ours_no_self_play.sh    && MARKER=ours_no_self_play bash $E/main/eval.sh
```

`ours_no_self_play` is the row that says what the proposing side is worth:
the model still solves everything, but a regret heuristic picks the edits
instead of the model proposing them. Nothing else changes.

`main/eval.sh` handles three kinds of directory: a self-evolve run (merges
the final round's weights), a trainer output dir (picks the
highest-numbered checkpoint), or a directory that already is a full model.

`main/eval.sh` is the single evaluation entry every exp shares — datasets,
judge and decoding settings all live there. Don't change them inside one
exp, or the main table stops being comparable. With no run_dir it uses the
most recent recorded run; `MARKER=xxx` selects which one.

## ablation

Each script reverts exactly one component; everything else stays the full
method.

```bash
bash $E/ablation/training_stages.sh grpo             # drop SFT replay
bash $E/ablation/training_stages.sh sft              # drop GRPO
bash $E/ablation/reward_component.sh gating          # drop the outcome gate
bash $E/ablation/reward_component.sh answer_fields   # answer back to whole-string match
bash $E/ablation/reward_component.sh box_f1          # grounding back to recall
bash $E/ablation/reward_component.sh format_floor    # restore the unconditional format bonus
bash $E/ablation/reward_component.sh field_consistency
bash $E/ablation/reward_component.sh old_reward      # the whole old scoring at once
bash $E/ablation/self_play.sh no_feedback             # model proposes blind to the solver's rate
bash $E/ablation/self_play.sh greedy                  # proposals sampled greedily
bash $E/ablation/generator.sh no_edit                # no regret-driven editing
bash $E/ablation/generator.sh no_pairs               # edit, but unpaired
bash $E/ablation/generator.sh no_label_balance       # no label balancing
bash $E/ablation/generator.sh plain                  # all three off
bash $E/ablation/grpo_algorithm.sh scale_rewards     # advantages divided by std again
bash $E/ablation/grpo_algorithm.sh no_overlong
bash $E/ablation/grpo_algorithm.sh batch_retry
bash $E/ablation/grpo_algorithm.sh old_grpo          # all three GRPO changes reverted
bash $E/ablation/rollout_budget.sh 256               # rollouts capped at 256 tokens
```

Evaluate with `main/eval.sh` as usual; `MARKER` is whatever name the
wrapper exported.

## sensitivity

One axis per script, one independent training + evaluation run per value.
`DRY=1` just prints the plan.

```bash
DRY=1 bash $E/sensitivity/edit_fraction.sh      # see how many runs it'd take
bash $E/sensitivity/grounding_timing.sh         # 0 80 160 320
bash $E/sensitivity/reward_weight.sh            # answer 0.45 0.60 0.75
bash $E/sensitivity/edit_fraction.sh            # 0 0.25 0.5 0.75
bash $E/sensitivity/group_size.sh               # 4 8 16
bash $E/sensitivity/rollout_length.sh           # 256 512 1024 2048
bash $E/sensitivity/rollout_temperature.sh      # 0.6 0.9 1.2
bash $E/sensitivity/image_resolution.sh         # 1x 2x 4x
bash $E/sensitivity/proposer_group_size.sh      # 2 4 8
bash $E/sensitivity/learnability_threshold.sh   # 0 0.25 0.5 0.75
bash $E/sensitivity/group_size.sh 8 16 32       # or pass your own values
```

## analysis

Nothing here produces a main-table number; these explain what's going on.

```bash
# The perception ceiling: base-model count accuracy at each resolution vs
# the always-answer-the-mode prior. Run this before any other exp.
bash $E/analysis/perception_probe.sh

# How much reward a policy that never looks at the image can collect.
# Run this after every reward change.
bash $E/analysis/reward_hackability.sh

# Controls: how much you gain when the reward carries no information about
# correctness. Once your numbers start going up, you need these.
bash $E/analysis/spurious_reward_control.sh random
bash $E/analysis/spurious_reward_control.sh format_only

# Collapse diagnostics read from an existing run — no retraining.
bash $E/analysis/collapse_diagnostics.sh <run_dir>

# Custom-decoding evaluation of one checkpoint (separate from the
# main-table eval path).
bash $E/analysis/eval_checkpoint.sh

# Assemble finished evaluations into one model x benchmark table.
bash $E/analysis/collect_results.sh
bash $E/analysis/collect_results.sh base ours vision_zero   # just these
VERBOSE=1 bash $E/analysis/collect_results.sh               # show which column each score came from
```

`collect_results.sh` computes AVG only over benchmarks that every model has
a score for — if one model is missing an entry, rows averaged over
different subsets can't be compared. The table states which benchmarks
went into AVG and which were excluded.

## Where to change hyperparameters

`../configs/train_defaults.sh` is the single home for training knobs (lr,
lengths, sampling, advantage handling, resolution, task generation). Both
`lib/common.sh` and `../run_loop.sh` source it, so every exp starts from
the same defaults. Change things there, not inside individual scripts.

The main-method reward config is `../configs/reward/reward_weights.json`.
`lib/reward_outcome_only.json` is reserved for the ours-without-process
ablation. The plain GRPO baseline has its own binary config at
`../configs/reward/baselines/grpo_binary_outcome.json`: it parses `spy` and
`changed_attributes`, then returns 1 only when both match gold. Ablation
scripts patch the default config on the fly via `lib/patch_reward_config.py`.

Generation defaults to half edited, half freshly sampled. With `SELF_PLAY=1`
(the default) the model itself picks which changes each edited task keeps;
`SELF_PLAY=0` hands that back to the regret heuristic. `EDIT_FRACTION=0`
disables editing altogether, which also switches the proposing side off since
no proposal could become a task.

Which stages run is controlled by `STAGES` (`sft` / `grpo`,
comma-separated, executed sft→grpo). The training path is SFT replay plus
GRPO — why there's no DPO is in `../docs/REWARD.md`.

## The LoRA path

`../run_loop.sh` + `../configs/*_h200.sh` is a separate entry point that
trains LoRA, not full-parameter. Not comparable with the scripts above —
**never put the two in the same results table**.

## Prerequisites

Moving to another machine means editing one line: `WORKSPACE` in
`../paths.sh` — every other path derives from it (and can still be
overridden by its own env var). Running with the placeholder unfilled
fails immediately with a clear message instead of crashing halfway
through.

```
$WORKSPACE/
  data/Vision-Zero-clevr-dataset/    # https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset
  models/Qwen2.5-VL-7B-Instruct/     # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
  eval/VLMEvalKit/                   # https://github.com/open-compass/VLMEvalKit
  eval/LMUData/                      # benchmark tsvs, see ../eval/fetch_vlmeval_tsv.sh
  eval/results/                      # evaluation output
  runs/                              # training output
```

Install dependencies via the repo's root `setup.sh` (read it rather than
running it blindly — it opens with commented-out `conda create` lines).
Then point `CONDA_ENV` in `paths.sh` at whatever you named the env; if it
isn't found, the scripts fall back to the `python3` on PATH and say so.

API keys go in `$REPO/.env`. **Training needs no API** — with the
Reference VLM off, task screening falls back to local rules, and difficulty
signals come from the solver's own rollouts. Only the evaluation MCQ judge
wants a key, or use `JUDGE=exact_matching` to score fully locally.

Handy overrides: `DATASETS="MMVP"` for a single-benchmark smoke run,
`WITH_BASE=0` to skip re-evaluating the base model,
`NUM_ITERATIONS` / `GRPO_MAX_STEPS` to shrink a smoke run.
