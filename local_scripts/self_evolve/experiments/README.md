# experiments

Four directories, one per kind of experiment. `lib/` holds the shared sevlm
runners. Most wrappers set a few variables and call those runners.
Experiment 3 is independent: `main/vision_zero_baseline.sh` launches a pinned
official Vision-Zero checkout in its own environment. See [VISION_ZERO.md](VISION_ZERO.md).

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
cd /path/to/sevlm
E=local_scripts/self_evolve/experiments
```

## main

| exp | script |
|---|---|
| base zero-shot | `main/base_zeroshot.sh` |
| base + GRPO | `main/grpo_baseline.sh` |
| base + vision-zero (official implementation and recipe) | `main/vision_zero_baseline.sh` — [separate setup](VISION_ZERO.md) |
| base + ours | `main/ours.sh` |
| base + ours + verified visual facts (experimental) | `main/ours_visual_facts.sh` |
| base + visual facts + visual curriculum (experimental) | `main/ours_visual_curriculum.sh` — [research and ablations](VISUAL_CURRICULUM.md) |
| base + ours w/o self-play | `main/ours_no_self_play.sh` |
| base + ours w/o process reward | `main/ours_no_process.sh` |

```bash
tmux new -s run
bash $E/main/ours.sh                 && bash $E/main/eval.sh
bash $E/main/grpo_baseline.sh        && MARKER=grpo_baseline bash $E/main/eval.sh
bash $E/main/ours_no_self_play.sh    && MARKER=ours_no_self_play bash $E/main/eval.sh
```

For experiment 3, follow [VISION_ZERO.md](VISION_ZERO.md) to set the official
checkout, Python, model, data and output paths, then run
`bash $E/main/vision_zero_baseline.sh`. It does not use the shared training
defaults or create a `vision_zero` marker. Evaluate its explicit output model
path under the same evaluation protocol used for all five rows. The launcher
runs the official implementation at the training budget the VISION-ZERO paper
reports (100 iterations, batch 128 games, lr 1e-5, beta 0.04, official ZeRO
config), so it is deliberately NOT bound by `configs/train_defaults.sh`: a paper
baseline is most defensible on its published configuration, and mixing in ours'
learning rate and beta would invite exactly the wrong reviewer question. Note
the repo's own `run_grpo_vision_zero.sh` asks for 40 epochs, which the data
pipeline turns into 280 optimizer steps -- 2.8x the paper -- so the two "official"
budgets disagree and VISION_ZERO.md explains why the paper's wins. It does not
equate generated tokens/FLOPs with the other rows either; report the actual
step counts.

Experiment 3 is also the one row that patches upstream code:
`patches/vision_zero_paper_alignment.patch` implements the paper's stage
switching (RAE alpha 0.95, the rho=0.95 accuracy/"n/a" EMAs, the threshold
hysteresis and its dwell gate), because the released code cannot produce it from
flags alone. The launcher accepts only a clean checkout or exactly that patch,
records both hashes in the run directory, and `patches/revert_vision_zero_patch.sh`
restores upstream. Do not hand-edit the official checkout; see VISION_ZERO.md.
See [EVAL_DATASETS.md](EVAL_DATASETS.md)
for the shared nine-benchmark evaluation entry point and dataset names.

`ours_no_self_play` is the row that says what the proposing side is worth:
the model still solves everything, but a regret heuristic picks the edits
instead of the model proposing them. Nothing else changes.

`main/ours_visual_facts.sh` is a separate experimental extension, not a row
of the original main table. It asks the solver for one
`<change>attribute:before->after</change>` tag per changed attribute and checks
those tags against the exact CLEVR scene metadata using multiset F1. Extra,
repeated and malformed tags cost precision. In this experiment, a malformed
completion receives answer/budget credit but no grounding, process, or
consistency credit, matching the positive-buffer format requirement.
A capped, deterministic SFT set of
verified certificates teaches the new syntax before GRPO. A second capped SFT
source asks count, left/right and depth questions about the unedited civilian
image, with answers computed from its original CLEVR scene. Each source defaults
to at most 64 records per round; set `SELF_EVOLVE_ORACLE_SFT_MAX=0` and
`SELF_EVOLVE_SCENE_QA_MAX=0` to disable them independently. In rounds with
real solver positives, `SELF_EVOLVE_AUX_SFT_MAX_RATIO=1.0` caps the combined
auxiliary count at the real-positive count, splitting slots between the two
sources when possible. A zero-positive round keeps both pools for cold start;
set the ratio to `0` to remove auxiliaries once positives appear, or unset it
to recover the uncapped export. Whenever no solved editing seeds are available,
the launcher reserves 25% of generated candidate-task slots for matched
one-object/two-object task pairs via
`SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION=0.25`; set it to `0` for an ablation.
The script sets
`SFT_MIN_PIXELS=401408` and `SFT_MAX_PIXELS=602112` (512–768 merged Qwen
visual tokens per image) with `SFT_PER_DEVICE_BATCH=2`. GRPO uses at least
1024 merged visual tokens per image; the SFT setting is deliberately smaller because a batch
can contain several player images and full-parameter SFT has a different
memory footprint. On hardware with headroom, test matching GRPO's image scale
by overriding these two SFT variables, while lowering the batch if needed.
The shared SFT defaults remain unchanged for the original main experiments.
It uses `configs/reward/reward_visual_facts.json` and leaves `main/ours.sh`
unchanged. For attribution, compare against `main/ours.sh` at the same
backbone, task budget, GRPO steps and evaluation settings. The loop records a
code, reward, and run-settings fingerprint. Resume into an
older or changed run directory is refused; use a fresh `RUN_TAG` for this
version so old scored trajectories and checkpoints cannot be silently reused.
The following controls use the same launcher; assign each one a distinct
`RUN_TAG` and `MARKER`, and use a matching unique `LABEL` when evaluating:

| control | launcher overrides | what it isolates |
|---|---|---|
| no bootstrap pairs | `SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION=0` | effect of task pairing when no solved editing seeds are available |
| no auxiliary SFT | `SELF_EVOLVE_ORACLE_SFT_MAX=0 SELF_EVOLVE_SCENE_QA_MAX=0` | certificate prompt plus verified reward without synthetic SFT |
| no auxiliary SFT after cold start | `SELF_EVOLVE_AUX_SFT_MAX_RATIO=0` | synthetic SFT in later rounds, preserving first-round cold start |
| prompt only | `REWARD_JSON="$PWD/local_scripts/self_evolve/configs/reward/reward_weights.json" SELF_EVOLVE_ORACLE_SFT_MAX=0 SELF_EVOLVE_SCENE_QA_MAX=0 SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION=0` | certificate instructions under the original reward |
| reward only | `SELF_EVOLVE_VISUAL_FACTS=0 SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION=0` | verified reward without certificate instructions or auxiliary SFT; this is a hard exploration control because tags are not requested |

The new SFT image scale and batch apply to these controls too. For strict
comparison to `main/ours.sh`, pass the same `SFT_MIN_PIXELS`, `SFT_MAX_PIXELS`
and `SFT_PER_DEVICE_BATCH` there as well. These scores are new experiments,
not established improvements.

`main/ours_visual_curriculum.sh` adds paired single-image count QA, joint
visual-evidence mastery, and fixed-denominator Dr. GRPO loss. It preserves the
GRPO prompt count and existing game pairs. See [VISUAL_CURRICULUM.md](VISUAL_CURRICULUM.md)
for the paper survey, exact settings, ablations and performance limits.

The SFT loader preserves the full multi-image prompt and answer. It skips
TRL's generic dataset tokenization and sets `max_length=None`; the 1024-token
text default would otherwise cut 5–8-image examples before the answer.

For the nine requested benchmarks, evaluate every full checkpoint with the
same explicit dataset list and decoding settings (`MMStar` is the evaluator's
name for MMStar):

```bash
DATASETS="MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST MMMU_Pro_10c CV-Bench-2D CV-Bench-3D" \
  LABEL=ours_visual_facts bash $E/analysis/eval_checkpoint.sh /path/to/full-checkpoint
```

`main/eval.sh` handles three kinds of directory: a self-evolve run (merges
the final round's weights), a trainer output dir (picks the
highest-numbered checkpoint), or a directory that already is a full model.

`main/eval.sh` delegates to the same `analysis/eval_checkpoint.sh` protocol.
With no run directory, it uses the most recent recorded run (`MARKER=xxx`
selects one). Use a unique `LABEL` per checkpoint and keep datasets, decoding,
judge, and `WORK_DIR` fixed across comparisons.

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
bash $E/analysis/collapse_diagnostics.sh /path/to/run_dir

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
