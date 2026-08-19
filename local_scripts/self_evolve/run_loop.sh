#!/usr/bin/env bash
# =============================================================================
# Self-evolve loop training entry.
#
# Usage:
#   bash self_evolve/run_loop.sh                          # default config configs/grpo_h200.sh
#   bash self_evolve/run_loop.sh configs/my_exp.sh        # a specific config
#   Run long jobs in tmux:  tmux new -s train
#
# All real paths live in paths.sh; all hyperparameters in configs/*.sh. This
# script only assembles the command.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/paths.sh"
source "$HERE/configs/train_defaults.sh"

# Pick the hyperparameter config (default grpo_h200.sh)
CFG="${1:-$HERE/configs/grpo_h200.sh}"
[ -f "$CFG" ] || { echo "[ERROR] config not found: $CFG" >&2; exit 2; }
source "$CFG"
echo "[cfg] using $CFG (EXP_NAME=$EXP_NAME)"
export_train_extra_args

# Basic existence checks (unfilled placeholder paths fail here, not halfway
# through training)
[ -e "$DATASET_ROOT" ] || { echo "[ERROR] DATASET_ROOT does not exist: $DATASET_ROOT (see the comments in paths.sh; download the dataset first)" >&2; exit 2; }
[ -d "$DATASET_ROOT/output/replacement_images" ] || { echo "[ERROR] $DATASET_ROOT/output/replacement_images not found. DATASET_ROOT must be the directory CONTAINING output/, not output/ itself." >&2; exit 2; }
[ -e "$BASE_MODEL" ]   || { echo "[ERROR] BASE_MODEL does not exist: $BASE_MODEL (see the comments in paths.sh; download the model first)" >&2; exit 2; }

# Load .env (API keys etc.; anything already exported in the shell wins)
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

# Output directory
RUN_TAG="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$RUNS_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

# carry-forward: keep every round's merged model for per-round evaluation
if [ "${KEEP_ALL_MERGED:-1}" = "1" ]; then export SELF_EVOLVE_DISABLE_MERGE_GC=1; fi

# Visible GPUs (0..NUM_GPUS-1); respect an externally pinned set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
fi

# ---- Assemble CLI flags ----
ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$BASE_MODEL"
  --output-dir "$OUT_DIR"
  --reward-config "$REWARD_CONFIG"
  --num-iterations "$NUM_ITERATIONS"
  --num-train-tasks "$NUM_TRAIN_TASKS"
  --num-generations "$NUM_GENERATIONS"
  --seed "$SEED"
  --reference-provider "$REFERENCE_PROVIDER"
  --reference-model "$REFERENCE_MODEL"
  --reference-base-url "$REFERENCE_BASE_URL"
  --max-trainer-steps "$MAX_STEPS"
  --trainer-num-gpus "$NUM_GPUS"
  --trainer-per-device-train-batch-size "$PER_DEVICE_BATCH"
  --trainer-gradient-accumulation-steps "$GRAD_ACCUM"
  --trainer-grpo-num-generations "$NUM_GENERATIONS"
  $(solver_cli_args)
  $(generator_cli_args)
  --trainer-lora-r "$LORA_R"
  --trainer-lora-alpha "$LORA_ALPHA"
  --trainer-lora-dropout "$LORA_DROPOUT"
)

# More than 2 rounds needs an explicit unlock
[ "$NUM_ITERATIONS" -gt 2 ] && ARGS+=(--allow-more-than-two-iterations)

# Reference VLM: live vs dry-run
if [ "${REFERENCE_LIVE:-0}" = "1" ]; then ARGS+=(--enable-openai-reference-vlm); else ARGS+=(--dry-run-reference-vlm); fi
# Solver: live vs dry-run
[ "${SOLVER_LIVE:-0}" = "1" ] || ARGS+=(--dry-run-solver)
# Trainer: live vs dry-run + which stages run
if [ "${TRAINER_LIVE:-0}" = "1" ]; then
  ARGS+=($(stage_cli_args))
else
  ARGS+=(--dry-run-trainer)
fi
# LoRA
[ "${USE_LORA:-1}" = "1" ] && ARGS+=(--use-lora)
# DeepSpeed (optional)
[ -n "${DEEPSPEED_CONFIG:-}" ] && ARGS+=(--trainer-deepspeed-config "$DEEPSPEED_CONFIG")
# carry-forward ablation
[ "${DISABLE_CARRY_FORWARD:-0}" = "1" ] && ARGS+=(--disable-grpo-carry-forward)

echo "======================================================"
echo "  RUN_TAG=$RUN_TAG"
echo "  OUT_DIR=$OUT_DIR"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  iterations=$NUM_ITERATIONS tasks=$NUM_TRAIN_TASKS gen=$NUM_GENERATIONS"
echo "  trainer_live=$TRAINER_LIVE reference_live=$REFERENCE_LIVE solver_live=$SOLVER_LIVE"
echo "======================================================"

cd "$REPO"
"$PY" "$LOOP_ENTRY" "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/run.log"
