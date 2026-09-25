#!/usr/bin/env bash
# Experiment 3: the official Vision-Zero recipe in a separate checkout/env.
# Does not source sevlm paths.sh, common.sh, train_defaults.sh or .env.
# Setup and evaluation: ../VISION_ZERO.md. DRY_RUN=1 only prints the command.
set -euo pipefail

readonly OFFICIAL_COMMIT="386fa20711130b9c7d8a340edd285c6242d8d255"
: "${VISION_ZERO_REPO:?Set VISION_ZERO_REPO to the official vision-zero checkout}"
: "${VISION_ZERO_PY:?Set VISION_ZERO_PY to the absolute Python path in the separate environment}"
: "${VISION_ZERO_MODEL:?Set VISION_ZERO_MODEL to the original Qwen2.5-VL-7B-Instruct directory}"
: "${VISION_ZERO_DATASET:?Set VISION_ZERO_DATASET to the directory containing output/}"
: "${VISION_ZERO_OUTPUT:?Set VISION_ZERO_OUTPUT to a new absolute run directory}"

fail() { echo "[ERROR] $*" >&2; exit 2; }
for value in "$VISION_ZERO_REPO" "$VISION_ZERO_PY" "$VISION_ZERO_MODEL" \
             "$VISION_ZERO_DATASET" "$VISION_ZERO_OUTPUT"; do
  case "$value" in /*) ;; *) fail "Use absolute paths: $value" ;; esac
done
[ -x "$VISION_ZERO_PY" ] || fail "Python is not executable: $VISION_ZERO_PY"
[ -f "$VISION_ZERO_MODEL/config.json" ] || fail "Model config.json not found"
IMAGES_DIR="$VISION_ZERO_DATASET/output/replacement_images"
SCENES_DIR="$VISION_ZERO_DATASET/output/replacement_scenes"
[ -d "$IMAGES_DIR" ] || fail "Missing image directory: $IMAGES_DIR"
[ -d "$SCENES_DIR" ] || fail "Missing scene directory: $SCENES_DIR"

TRAIN_ROOT="$VISION_ZERO_REPO/src/open-r1-multimodal"
ENTRY="$TRAIN_ROOT/src/open_r1/grpo_jsonl.py"
ZERO_CONFIG="$TRAIN_ROOT/local_scripts/zero3_model_parallel.json"
[ -f "$ENTRY" ] && [ -f "$ZERO_CONFIG" ] || fail "Not an official Vision-Zero checkout"
ACTUAL_COMMIT="$(git -C "$VISION_ZERO_REPO" rev-parse HEAD)"
[ "$ACTUAL_COMMIT" = "$OFFICIAL_COMMIT" ] || fail "Expected official commit $OFFICIAL_COMMIT, got $ACTUAL_COMMIT"
git -C "$VISION_ZERO_REPO" diff --quiet HEAD -- src/open-r1-multimodal \
  || fail "Official training code/config has local changes; use a clean checkout"
[ ! -e "$VISION_ZERO_OUTPUT" ] || fail "Output already exists; choose a new VISION_ZERO_OUTPUT to avoid implicit resume"

RUN_NAME="${VISION_ZERO_RUN_NAME:-Qwen2.5-VL-7B-Vision-Zero-official}"
REPORT_TO="${VISION_ZERO_REPORT_TO:-none}"
# Values below reproduce the published launcher, not sevlm's main protocol.
# Explicit seed, pixel limits, vLLM=False and max_steps=-1 match upstream defaults.
CMD=(
  "$VISION_ZERO_PY" -m torch.distributed.run
  --nproc_per_node=8 --nnodes=1 --node_rank=0
  --master_addr=127.0.0.1 --master_port="${VISION_ZERO_MASTER_PORT:-12350}"
  "$ENTRY"
  --deepspeed "$ZERO_CONFIG"
  --output_dir "$VISION_ZERO_OUTPUT" --model_name_or_path "$VISION_ZERO_MODEL"
  --dataset_name dynamic_clevr_spotdiff --use_dynamic_dataset
  --epoch_size 450 --data_generator_type clevr_spotdiff
  --clevr_images_dir "$IMAGES_DIR" --clevr_scenes_dir "$SCENES_DIR"
  --clevr_num_players 4 --clevr_num_rounds 2
  --training_phase interactive --interactive_cycle_length 1
  --data_generator_seed 42 --seed 42 --max_anyres_num 6
  --max_prompt_length 8000 --max_completion_length 512
  --min_pixels 3136 --max_pixels 12845056
  --num_generations 8 --per_device_train_batch_size 1
  --gradient_accumulation_steps 8 --logging_steps 1
  --bf16 --torch_dtype bfloat16 --beta 0.04
  --report_to "$REPORT_TO" --gradient_checkpointing true
  --attn_implementation flash_attention_2 --use_vllm False
  --num_train_epochs 40 --max_steps -1
  --learning_rate 1e-5 --warmup_ratio 0.1 --lr_scheduler_type cosine
  --run_name "$RUN_NAME" --save_steps 5 --save_only_model true
  --reward_funcs clevr_clue_format_with_votes clevr_decision_accuracy
  --val_split_ratio 0.0 --num_iterations 1
)

echo "[vision-zero] official source=$OFFICIAL_COMMIT"
echo "[vision-zero] 8 GPUs; 40 epochs; 4 players; G=8; no sevlm trainer imports"
printf '%q ' "${CMD[@]}"; printf '\n'
if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi

# Select only the official source, even if this shell previously ran sevlm.
export PYTHONPATH="$TRAIN_ROOT/src"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPU_IDS[@]}" -eq 8 ] || fail "This official recipe requires exactly 8 visible GPUs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_ALLOC_CONF
export DEBUG_MODE=true
mkdir -p "$VISION_ZERO_OUTPUT"
export LOG_PATH="$VISION_ZERO_OUTPUT/debug_log.txt"
printf '%s\n' "$OFFICIAL_COMMIT" > "$VISION_ZERO_OUTPUT/official_commit.txt"
printf '%q ' "${CMD[@]}" > "$VISION_ZERO_OUTPUT/launch_command.txt"
printf '\n' >> "$VISION_ZERO_OUTPUT/launch_command.txt"
cd "$TRAIN_ROOT"
"${CMD[@]}" 2>&1 | tee "$VISION_ZERO_OUTPUT/run.log"
echo "[done] Full model: $VISION_ZERO_OUTPUT"
echo "[eval] Evaluate this checkpoint with the same protocol as all other main-table models."
