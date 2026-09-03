#!/usr/bin/env bash
# =============================================================================
# Training: base + GRPO — plain single-round GRPO post-training, no
# self-evolve mechanism.
#
# Difference from ours (same loop entry, with the self-evolve parts switched
# off):
#   - single round (NUM_ITERATIONS=1), no cross-round carry-forward
#   - GRPO stage only (STAGES=grpo)
#   - Reference-VLM screening off (local dry-run heuristic, no GPT-4o call)
#   - outcome-only reward (answer=1.0, see reward_outcome_only.json)
# Data is the same CLEVR source, so it stays comparable with ours.
#
# Hardware defaults assume 8x H200; edit the variable block at the top to
# change hyperparameters.
# After training, evaluate with  main/eval.sh <run_dir>.
#
# Usage (tmux recommended):
#   tmux new -s grpo
#   bash local_scripts/self_evolve/experiments/main/grpo_baseline.sh
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ---------------- hyperparameters (edit here for a different scale) ----------------
EXP_NAME="base_grpo"
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-$MAIN_NUM_TRAIN_TASKS}"
NUM_GENERATIONS="${NUM_GENERATIONS:-$MAIN_NUM_GENERATIONS}"  # GRPO group size; must be >=4 and divide PER_DEVICE_BATCH*NUM_GPUS
SEED=42
# --- Fair comparison ---
# TOTAL_GRPO_STEPS, group size, task count and adaptation method all come from
# the main-table protocol in configs/train_defaults.sh, so this exp and ours
# train on the same budget in the same regime. Only the reward and the loop
# differ. Changing any of these here breaks the main table.
MAX_STEPS="${MAX_STEPS:-$TOTAL_GRPO_STEPS}"
NUM_GPUS=8                 # H200 count
PER_DEVICE_BATCH=2
GRAD_ACCUM=2
LORA_R=256; LORA_ALPHA=512; LORA_DROPOUT=0.05
# Full-parameter by default, matching ours; MAIN_USE_LORA=1 switches to LoRA
# (then it is no longer comparable with a full-parameter ours).
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO/local_scripts/zero3.json}"
REWARD_JSON="${REWARD_JSON:-$LIB_DIR/reward_outcome_only.json}"   # outcome-based reward
# ---------------------------------------------------------

require_paths
load_env
[ -f "$REWARD_JSON" ] || { echo "[ERROR] reward config does not exist: $REWARD_JSON" >&2; exit 2; }

# Trainer knobs come from configs/train_defaults.sh, identical to the ours
# exps: baseline and main method should differ only in reward config, never
# in rollout length / sampling / advantage handling.
STAGES=grpo
# This exp is defined as plain GRPO with none of the self-evolve machinery, so
# the two things that machinery adds are switched off explicitly. Without this
# the baseline would inherit them from train_defaults and stop being a baseline.
SELF_PLAY=0
EDIT_FRACTION=0
GRPO_MAX_STEPS="$MAX_STEPS"
export_train_extra_args

RUN_TAG="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$EXP_RUNS_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

# --- wandb training curves (on by default, matching ours full-pipeline; WANDB=0
# disables). The GRPO trainer reports standard TRL curves (loss/lr/grad_norm/
# reward). Loop-level signals stay in run.log. Needs WANDB_API_KEY (in .env or a
# prior `wandb login`). Auto-login before the run (idempotent; a failure is
# non-fatal — training still runs and wandb caches offline). ---
if [ "${WANDB:-1}" = "1" ]; then
  export SELF_EVOLVE_REPORT_TO=wandb
  export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
  export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}_${NUM_TRAIN_TASKS}t_${MAX_STEPS}step}"
  : "${WANDB_API_KEY:?WANDB=1 needs WANDB_API_KEY (put it in $ENV_FILE or run wandb login first)}"
  wandb login --relogin "$WANDB_API_KEY" 2>/dev/null \
    && echo "[wandb] logged in" || echo "[wandb] login skipped/failed (training continues)" >&2
  echo "[wandb] project=$WANDB_PROJECT name=$WANDB_NAME"
else
  export SELF_EVOLVE_REPORT_TO=none
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
fi

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$BASE_MODEL"
  --output-dir "$OUT_DIR"
  --reward-config "$REWARD_JSON"
  --num-iterations 1
  --num-train-tasks "$NUM_TRAIN_TASKS"
  --num-generations "$NUM_GENERATIONS"
  --seed "$SEED"
  --max-trainer-steps "$MAX_STEPS"
  --trainer-num-gpus "$NUM_GPUS"
  --trainer-per-device-train-batch-size "$PER_DEVICE_BATCH"
  --trainer-gradient-accumulation-steps "$GRAD_ACCUM"
  --trainer-grpo-num-generations "$NUM_GENERATIONS"
  $(solver_cli_args)
  $(generator_cli_args)
  # --- Strip the self-evolve parts: GRPO-only + no reference screening ---
  --dry-run-reference-vlm     # no GPT-4o task screening (local heuristic)
  $(stage_cli_args)
)
if [ "${MAIN_USE_LORA:-0}" = "1" ]; then
  ARGS+=(--use-lora --trainer-lora-r "$LORA_R" --trainer-lora-alpha "$LORA_ALPHA" --trainer-lora-dropout "$LORA_DROPOUT")
fi
[ -n "$DEEPSPEED_CONFIG" ] && ARGS+=(--trainer-deepspeed-config "$DEEPSPEED_CONFIG")

banner "train_grpo_baseline  plain single-round GRPO, outcome-only reward
  RUN_TAG=$RUN_TAG
  OUT_DIR=$OUT_DIR
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  tasks=$NUM_TRAIN_TASKS gen=$NUM_GENERATIONS steps=$MAX_STEPS gpus=$NUM_GPUS mode=$([ "${MAIN_USE_LORA:-0}" = 1 ] && echo "lora_r=$LORA_R" || echo FULL-PARAM)
  reward=$REWARD_JSON"

cd "$REPO"
"$PY" "$LOOP_ENTRY" "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"

record_run "${MARKER:-grpo_baseline}" "$OUT_DIR"
banner "Training done. Evaluate: bash local_scripts/self_evolve/experiments/main/eval.sh $OUT_DIR"
