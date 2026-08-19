#!/usr/bin/env bash
# Main experiment: base + vision-zero.
#
# Runs upstream Vision-Zero's own self-play recipe: alternating clue/decision
# phases with its own two rewards (clue format with votes + decision
# accuracy) — none of our loop, no five-dim reward, no bbox.
#
# Aligned with ours (keeps the main table comparable): base model, image
# pool, num_players, num_generations, GPU count, per-device batch,
# max_completion_length, total steps. The only thing not aligned is the
# method itself — which is exactly what this exp measures.
#
# The trainer is invoked directly rather than through upstream's launcher: that
# one assigns `MODEL = ...` with spaces (not valid bash), passes the literal
# string as `--model_name_or_path`, and sets `--dispatch_batches`, which is not
# a TrainingArguments parameter in current transformers.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

NUM_PLAYERS="${NUM_PLAYERS:-5}"          # matches ours' task generation
NUM_ROUNDS="${NUM_ROUNDS:-2}"            # Vision-Zero's own clue round count
NUM_GENERATIONS="${NUM_GENERATIONS:-$MAIN_NUM_GENERATIONS}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-$TOTAL_GRPO_STEPS}"   # same total GRPO budget as every main exp
EPOCH_SIZE="${EPOCH_SIZE:-450}"
TRAINING_PHASE="${TRAINING_PHASE:-interactive}"
INTERACTIVE_CYCLE_LENGTH="${INTERACTIVE_CYCLE_LENGTH:-1}"
LR="${LR:-1e-6}"                         # aligned with ours' GRPO_LR
BETA="${BETA:-0.04}"

RUN_TAG="${RUN_TAG:-vision_zero}"
OUT_DIR="$EXP_RUNS_ROOT/$RUN_TAG"
IMAGES_DIR="$DATASET_ROOT/output/replacement_images"
SCENES_DIR="$DATASET_ROOT/output/replacement_scenes"

require_paths
load_env
[ -d "$IMAGES_DIR" ] || { echo "[ERROR] CLEVR images directory does not exist: $IMAGES_DIR" >&2; exit 2; }
[ -d "$SCENES_DIR" ] || { echo "[ERROR] CLEVR scenes directory does not exist: $SCENES_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
fi
export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "${WANDB:-0}" = "1" ]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
  export WANDB_NAME="${WANDB_NAME:-$RUN_TAG}"
  REPORT_TO=wandb
else
  REPORT_TO=none
fi

banner "vision_zero baseline
  OUT_DIR=$OUT_DIR
  players=$NUM_PLAYERS rounds=$NUM_ROUNDS gen=$NUM_GENERATIONS steps=$MAX_STEPS gpus=$NUM_GPUS
  phase=$TRAINING_PHASE  reward=clevr_clue_format_with_votes + clevr_decision_accuracy"

cd "$REPO"
"$PY" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" \
  --master_port="${MASTER_PORT:-12350}" \
  src/open_r1/grpo_jsonl.py \
  --deepspeed "$REPO/local_scripts/zero3.json" \
  --output_dir "$OUT_DIR" \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_name dynamic_clevr_spotdiff \
  --use_dynamic_dataset \
  --data_generator_type clevr_spotdiff \
  --clevr_images_dir "$IMAGES_DIR" \
  --clevr_scenes_dir "$SCENES_DIR" \
  --clevr_num_players "$NUM_PLAYERS" \
  --clevr_num_rounds "$NUM_ROUNDS" \
  --training_phase "$TRAINING_PHASE" \
  --interactive_cycle_length "$INTERACTIVE_CYCLE_LENGTH" \
  --epoch_size "$EPOCH_SIZE" \
  --data_generator_seed "${SEED:-42}" \
  --reward_funcs clevr_clue_format_with_votes clevr_decision_accuracy \
  --max_prompt_length "${GRPO_MAX_PROMPT_LEN:-8000}" \
  --max_completion_length "$GRPO_MAX_COMPLETION_LEN" \
  --min_pixels "$GRPO_MIN_PIXELS" --max_pixels "$GRPO_MAX_PIXELS" \
  --num_generations "$NUM_GENERATIONS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" --beta "$BETA" \
  --warmup_ratio "$GRPO_WARMUP_RATIO" --lr_scheduler_type "$GRPO_LR_SCHEDULER" \
  --bf16 --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_grad_norm 0.3 \
  --logging_steps 1 --save_steps 10 --save_only_model true \
  --report_to "$REPORT_TO" --run_name "$RUN_TAG" \
  --num_iterations 1 --val_split_ratio 0.0 \
  2>&1 | tee -a "$OUT_DIR/run.log"

record_run "${MARKER:-vision_zero}" "$OUT_DIR"
banner "Training done. Evaluate: MARKER=vision_zero bash $EXP_DIR/main/eval.sh"
