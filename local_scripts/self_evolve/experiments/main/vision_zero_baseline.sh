#!/usr/bin/env bash
# Experiment 3: official Vision-Zero implementation with sevlm-aligned settings.
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
ZERO_SOURCE="$TRAIN_ROOT/local_scripts/zero3_model_parallel.json"
[ -f "$ENTRY" ] && [ -f "$ZERO_SOURCE" ] || fail "Not an official Vision-Zero checkout"
ZERO_CONFIG="$VISION_ZERO_OUTPUT/zero3_vision_zero.json"
ACTUAL_COMMIT="$(git -C "$VISION_ZERO_REPO" rev-parse HEAD)"
[ "$ACTUAL_COMMIT" = "$OFFICIAL_COMMIT" ] || fail "Expected official commit $OFFICIAL_COMMIT, got $ACTUAL_COMMIT"
git -C "$VISION_ZERO_REPO" diff --quiet HEAD -- src/open-r1-multimodal \
  || fail "Official training code/config has local changes; use a clean checkout"
[ ! -e "$VISION_ZERO_OUTPUT" ] || fail "Output already exists; choose a new VISION_ZERO_OUTPUT to avoid implicit resume"

RUN_NAME="${VISION_ZERO_RUN_NAME:-Qwen2.5-VL-7B-Vision-Zero-aligned}"
REPORT_TO="${VISION_ZERO_REPORT_TO:-none}"
# Snapshots of the two sevlm entry points at commit 346490d. Do not source
# their launchers: doing so would activate the main environment/start training.
# Default: ours_full_pipeline_train.sh, 2 rounds x 120 GRPO steps.
# main_ours: main/ours.sh -> lib/run_self_evolve.sh + train_defaults.sh.
PROTOCOL="${VISION_ZERO_PROTOCOL:-ours_full_pipeline}"
case "$PROTOCOL" in
  ours_full_pipeline) DEFAULT_STEPS=240; DEFAULT_G=8; DEFAULT_BATCH=2; DEFAULT_ACCUM=8 ;;
  main_ours) DEFAULT_STEPS=320; DEFAULT_G=16; DEFAULT_BATCH=8; DEFAULT_ACCUM=1 ;;
  *) fail "VISION_ZERO_PROTOCOL must be ours_full_pipeline or main_ours" ;;
esac
# Prefix overrides explicitly; unrelated main-run environment variables cannot leak in.
TRAIN_STEPS="${VISION_ZERO_MAX_STEPS:-$DEFAULT_STEPS}"
GENERATIONS="${VISION_ZERO_NUM_GENERATIONS:-$DEFAULT_G}"
DEVICE_BATCH="${VISION_ZERO_PER_DEVICE_BATCH:-$DEFAULT_BATCH}"
ACCUM="${VISION_ZERO_GRAD_ACCUM:-$DEFAULT_ACCUM}"
for value in "$TRAIN_STEPS" "$GENERATIONS" "$DEVICE_BATCH" "$ACCUM"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "Steps, G, batch and accumulation must be positive decimal integers"
done
EFFECTIVE_BATCH=$((8 * DEVICE_BATCH * ACCUM))
[ "$GENERATIONS" -ge 4 ] || fail "Use G >= 4, matching the main training protocol"
[ $((8 * DEVICE_BATCH % GENERATIONS)) -eq 0 ] || fail "8 * per-device batch must be divisible by G"
# Match common optimizer/length/pixel settings; retain official rewards, gameplay,
# advantage computation and CPU-offloaded ZeRO-3. This does NOT equalize total FLOPs.
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
  --max_prompt_length 10240 --max_completion_length 2048
  --min_pixels 802816 --max_pixels 1003520
  --num_generations "$GENERATIONS" --per_device_train_batch_size "$DEVICE_BATCH"
  --gradient_accumulation_steps "$ACCUM" --logging_steps 1
  --bf16 --torch_dtype bfloat16 --beta 0.06
  --report_to "$REPORT_TO" --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --attn_implementation flash_attention_2 --use_vllm False
  --max_steps "$TRAIN_STEPS"
  --learning_rate 1e-6 --warmup_ratio 0.1 --lr_scheduler_type cosine
  --weight_decay 0.0 --temperature 1.0 --max_grad_norm 0.3
  --run_name "$RUN_NAME" --save_steps 10 --save_only_model true
  --reward_funcs clevr_clue_format_with_votes clevr_decision_accuracy
  --val_split_ratio 0.0 --num_iterations 1
)

echo "[vision-zero] official source=$OFFICIAL_COMMIT"
echo "[vision-zero] protocol=$PROTOCOL; steps=$TRAIN_STEPS; G=$GENERATIONS; per_device_batch=$DEVICE_BATCH; grad_accum=$ACCUM; nominal_effective_batch=$EFFECTIVE_BATCH"
echo "[vision-zero] 8 GPUs; official 4-player gameplay; no sevlm trainer imports"
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
# Upstream pins gradient_clipping=1.0; HF would reject max_grad_norm=0.3.
# Write a run-local copy with automatic clipping alignment; never edit upstream.
"$VISION_ZERO_PY" - "$ZERO_SOURCE" "$ZERO_CONFIG" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    config = json.load(source)
config["gradient_clipping"] = "auto"
with open(sys.argv[2], "w") as output:
    json.dump(config, output, indent=2)
PY
export LOG_PATH="$VISION_ZERO_OUTPUT/debug_log.txt"
printf '%s\n' "$OFFICIAL_COMMIT" > "$VISION_ZERO_OUTPUT/official_commit.txt"
printf 'protocol=%s\nmax_steps=%s\nnum_generations=%s\nper_device_batch=%s\ngrad_accum=%s\nnominal_effective_batch=%s\n' \
  "$PROTOCOL" "$TRAIN_STEPS" "$GENERATIONS" "$DEVICE_BATCH" "$ACCUM" "$EFFECTIVE_BATCH" \
  > "$VISION_ZERO_OUTPUT/alignment_config.txt"
printf '%q ' "${CMD[@]}" > "$VISION_ZERO_OUTPUT/launch_command.txt"
printf '\n' >> "$VISION_ZERO_OUTPUT/launch_command.txt"
cd "$TRAIN_ROOT"
"${CMD[@]}" 2>&1 | tee "$VISION_ZERO_OUTPUT/run.log"
echo "[done] Full model: $VISION_ZERO_OUTPUT"
echo "[eval] Evaluate this checkpoint with the same protocol as all other main-table models."
