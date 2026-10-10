#!/usr/bin/env bash
# External baseline: the official Vision-Zero implementation, run at the
# training budget the Vision-Zero paper reports (100 iterations, batch 128
# games). NOT aligned to the sevlm RL budget. See ../VISION_ZERO.md.
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
warn() { echo "[WARN] $*" >&2; }
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
[ ! -e "$VISION_ZERO_OUTPUT" ] || fail "Output already exists; choose a new VISION_ZERO_OUTPUT to avoid implicit resume"

# --- Paper-alignment patch -------------------------------------------------
# The released code cannot reproduce the paper's stage switching from flags
# alone: the RAE coefficient is 0.9 rather than Table 5's 0.95, and the
# threshold/EMA switching of App. A.2.3 is absent (upstream ships a fixed cycle).
# PATCH_FILE is the ONLY change this launcher is allowed to make to the checkout;
# any other local modification is refused, so nothing undocumented can slip in.
# Revert with: git -C "$VISION_ZERO_REPO" checkout -- src/open-r1-multimodal
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/vision_zero_paper_alignment.patch"
[ -f "$PATCH_FILE" ] || fail "Missing patch file: $PATCH_FILE"
PATCH_SHA="$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
if git -C "$VISION_ZERO_REPO" diff --quiet HEAD -- src/open-r1-multimodal; then
  PATCH_STATE="clean"
  git -C "$VISION_ZERO_REPO" apply --check "$PATCH_FILE" \
    || fail "Patch does not apply to this checkout; resolve it by hand"
elif git -C "$VISION_ZERO_REPO" apply --check --reverse "$PATCH_FILE" 2>/dev/null; then
  PATCH_STATE="applied"
  APPLIED_SHA="$(git -C "$VISION_ZERO_REPO" diff -- src/open-r1-multimodal | sha256sum | cut -d' ' -f1)"
  [ "$APPLIED_SHA" = "$PATCH_SHA" ] \
    || fail "Checkout differs from the recorded patch: diff=$APPLIED_SHA patch=$PATCH_SHA"
else
  fail "Official checkout has local changes that are not the paper-alignment patch; refusing to run"
fi

# --- Training budget -------------------------------------------------------
# The Vision-Zero paper (arXiv 2509.25541) reports "100 iterations with a batch
# size of 128" (Table 5), on 8xA100. The repo's own run_grpo_vision_zero.sh says
# 40 epochs x epoch_size 450, which works out to 280 optimizer steps -- 2.8x the
# paper. We follow the paper's own number: it is the budget the baseline's
# published results come from. See ../VISION_ZERO.md.
STEPS="${VISION_ZERO_MAX_STEPS:-100}"
EPOCH_SIZE="${VISION_ZERO_EPOCH_SIZE:-450}"
GENERATIONS="${VISION_ZERO_NUM_GENERATIONS:-8}"
# Paper lists --per_device_train_batch_size 8, but grpo_trainer.py truncates
# every micro-batch to its first game ("inputs = [inputs[0]]", "to avoid OOM"),
# so anything above 1 is generated and then discarded. 8 ranks x 1 game x
# accum 16 = 128 games per optimizer step, which is the paper's batch size.
DEVICE_BATCH="${VISION_ZERO_PER_DEVICE_BATCH:-1}"
ACCUM="${VISION_ZERO_GRAD_ACCUM:-16}"
LR="${VISION_ZERO_LR:-1e-5}"
BETA="${VISION_ZERO_BETA:-0.04}"
SEED="${VISION_ZERO_SEED:-42}"
# The paper's command, now affordable: 100 steps x ~16 GB per full-model
# checkpoint means every 5 steps is ~320 GB against ~2.2 TB free.
SAVE_STEPS="${VISION_ZERO_SAVE_STEPS:-5}"
# The paper reports to wandb. Checked before launch: if the environment has no
# credentials this falls back to none rather than hang a 24 h run on a login
# prompt (the substitution is recorded in official_recipe.txt).
REPORT_TO="${VISION_ZERO_REPORT_TO:-wandb}"
RUN_NAME="${VISION_ZERO_RUN_NAME:-Qwen2.5-VL-7B-GRPO-Vision-Zero}"

for value in "$STEPS" "$EPOCH_SIZE" "$GENERATIONS" "$DEVICE_BATCH" "$ACCUM" "$SAVE_STEPS" "$SEED"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "Steps, sizes, G, batch, accumulation, save steps and seed must be positive decimal integers"
done
[ "$GENERATIONS" -ge 2 ] || fail "num_generations must be at least 2"
[ $((8 * DEVICE_BATCH % GENERATIONS)) -eq 0 ] || fail "8 * per-device batch must be divisible by num_generations"
[ "$DEVICE_BATCH" -eq 1 ] || warn "per-device batch $DEVICE_BATCH: only the first game of each micro-batch reaches the gradient (grpo_trainer.py truncates). Set it to 1 to avoid wasted generation."
NOMINAL_GAMES=$((8 * DEVICE_BATCH * ACCUM))
ACTUAL_GAMES=$((8 * 1 * ACCUM))

# WandB needs a working client and credentials, and a tmux run cannot answer a
# login prompt. The env's own wandb (0.18.3) creates a run but uploads no
# history, so stage the 0.30.0 package the sevlm runs use (WANDB_PKG_DIR, see
# ours_full_pipeline_train.sh) and log in with WANDB_API_KEY, as that script
# does. Both are resolved before the command line is built so the substitution
# is recorded.
REPORT_TO_NOTE=""
if [ "$REPORT_TO" = "wandb" ]; then
  WANDB_PKG_DIR="${WANDB_PKG_DIR:-/tmp/sevlm_wandb_py311}"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[vision-zero] DRY_RUN: not checking wandb credentials (the real run does)"
  elif [ ! -d "$WANDB_PKG_DIR/wandb" ]; then
    REPORT_TO_NOTE="W&B package not found at $WANDB_PKG_DIR; fell back to none"
    echo "[WARN] $REPORT_TO_NOTE" >&2
    REPORT_TO="none"
  elif [ -z "${WANDB_API_KEY:-}" ]; then
    REPORT_TO_NOTE="WANDB_API_KEY is not set; fell back to none"
    echo "[WARN] $REPORT_TO_NOTE" >&2
    REPORT_TO="none"
  else
    export PYTHONPATH="$WANDB_PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
    if "$VISION_ZERO_PY" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
      echo "[vision-zero] wandb logged in from $WANDB_PKG_DIR; project=$WANDB_PROJECT"
    else
      REPORT_TO_NOTE="wandb login failed; fell back to none"
      echo "[WARN] $REPORT_TO_NOTE" >&2
      REPORT_TO="none"
    fi
  fi
fi

# Do not add --min_pixels/--max_pixels: the paper's command omits them and the
# entry defaults (3136 / 12845056) are the Qwen2.5-VL values. Smaller values
# would downscale every image and silently weaken the baseline.
# --max_prompt_length is accepted but the trainer forces it to None, and the
# CLEVR interactive path hardcodes max_new_tokens=1024 / temperature=0.8 for
# its own generations, so --max_completion_length and --temperature never
# reach them. They are passed only to mirror the paper's command.
CMD=(
  "$VISION_ZERO_PY" -m torch.distributed.run
  --nproc_per_node=8 --nnodes=1 --node_rank=0
  --master_addr=127.0.0.1 --master_port="${VISION_ZERO_MASTER_PORT:-12350}"
  "$ENTRY"
  --deepspeed "$ZERO_CONFIG"
  --output_dir "$VISION_ZERO_OUTPUT" --model_name_or_path "$VISION_ZERO_MODEL"
  --dataset_name dynamic_clevr_spotdiff --use_dynamic_dataset
  --epoch_size "$EPOCH_SIZE" --data_generator_type clevr_spotdiff
  --clevr_images_dir "$IMAGES_DIR" --clevr_scenes_dir "$SCENES_DIR"
  --clevr_num_players 4 --clevr_num_rounds 2
  --training_phase interactive
  --data_generator_seed "$SEED" --max_anyres_num 6
  --max_prompt_length 8000 --max_completion_length 512
  --num_generations "$GENERATIONS" --per_device_train_batch_size "$DEVICE_BATCH"
  --gradient_accumulation_steps "$ACCUM" --logging_steps 1
  --bf16 --torch_dtype bfloat16 --beta "$BETA"
  --report_to "$REPORT_TO" --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --attn_implementation flash_attention_2
  # --use_vllm is omitted: GRPOConfig defaults it to False, as the paper does.
  # Required: accelerate defaults dispatch_batches=True for IterableDataset,
  # which concatenates the per-process batches on rank 0. These batches are
  # dicts holding strings, so that raises TypeError. Upstream passes it too.
  --dispatch_batches False
  --max_steps "$STEPS"
  --learning_rate "$LR" --warmup_ratio 0.1 --lr_scheduler_type cosine
  --run_name "$RUN_NAME" --save_steps "$SAVE_STEPS" --save_only_model true
  --reward_funcs clevr_clue_format_with_votes clevr_decision_accuracy
  --val_split_ratio 0.0 --num_iterations 1
)

echo "[vision-zero] official source=$OFFICIAL_COMMIT; alignment patch=$PATCH_SHA ($PATCH_STATE)"
echo "[vision-zero] recipe=paper (100 iterations, batch 128 games); steps=$STEPS; epoch_size=$EPOCH_SIZE; G=$GENERATIONS; per_device_batch=$DEVICE_BATCH; grad_accum=$ACCUM"
echo "[vision-zero] games per optimizer step: nominal=$NOMINAL_GAMES actual=$ACTUAL_GAMES; lr=$LR beta=$BETA seed=$SEED; save every $SAVE_STEPS steps; report_to=$REPORT_TO"
echo "[vision-zero] 8 GPUs; official 4-player gameplay; no sevlm trainer imports"
printf '%q ' "${CMD[@]}"; printf '\n'
if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi

# Apply the paper-alignment patch (dry runs must not mutate the checkout), then
# confirm the resulting diff is still exactly the recorded patch.
if [ "$PATCH_STATE" = "clean" ]; then
  git -C "$VISION_ZERO_REPO" apply "$PATCH_FILE" || fail "Failed to apply $PATCH_FILE"
  echo "[vision-zero] applied paper-alignment patch $PATCH_SHA"
fi
APPLIED_SHA="$(git -C "$VISION_ZERO_REPO" diff -- src/open-r1-multimodal | sha256sum | cut -d' ' -f1)"
[ "$APPLIED_SHA" = "$PATCH_SHA" ] || fail "Post-apply diff $APPLIED_SHA != recorded patch $PATCH_SHA"

# Select only the official source, even if this shell previously ran sevlm;
# keep any prefix added above (the staged wandb package) behind it.
export PYTHONPATH="$TRAIN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPU_IDS[@]}" -eq 8 ] || fail "This official recipe requires exactly 8 visible GPUs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_ALLOC_CONF
export DEBUG_MODE=true
# DeepSpeed has no prebuilt cpu_adam for this env and builds it with the system
# nvcc (12.9), while torch is cu124, so the op builder's version check aborts
# the run. cpu_adam is a CPU-only operator compiled with g++; skipping the
# check is safe and changes nothing about the recipe.
export DS_SKIP_CUDA_CHECK=1
mkdir -p "$VISION_ZERO_OUTPUT"
export LOG_PATH="$VISION_ZERO_OUTPUT/debug_log.txt"
printf '%s\n' "$OFFICIAL_COMMIT" > "$VISION_ZERO_OUTPUT/official_commit.txt"
printf 'recipe=paper_100_iterations_batch128\nmax_steps=%s\nepoch_size=%s\nnum_generations=%s\nper_device_batch=%s\ngrad_accum=%s\nnominal_games_per_step=%s\nactual_games_per_step=%s\nlearning_rate=%s\nbeta=%s\nseed=%s\nsave_steps=%s\nreport_to=%s\n' \
  "$STEPS" "$EPOCH_SIZE" "$GENERATIONS" "$DEVICE_BATCH" "$ACCUM" "$NOMINAL_GAMES" "$ACTUAL_GAMES" \
  "$LR" "$BETA" "$SEED" "$SAVE_STEPS" "$REPORT_TO" \
  > "$VISION_ZERO_OUTPUT/official_recipe.txt"
printf 'note=%s\n' "repo run_grpo_vision_zero.sh says 40 epochs = 280 steps; paper reports 100 iterations at batch 128. This run follows the paper." \
  >> "$VISION_ZERO_OUTPUT/official_recipe.txt"
[ "$REPORT_TO" = "wandb" ] && printf 'wandb_pkg_dir=%s\nwandb_project=%s\n' \
  "$WANDB_PKG_DIR" "$WANDB_PROJECT" >> "$VISION_ZERO_OUTPUT/official_recipe.txt"
[ -n "$REPORT_TO_NOTE" ] && printf 'report_to_note=%s\n' "$REPORT_TO_NOTE" >> "$VISION_ZERO_OUTPUT/official_recipe.txt"
{
  printf 'patch_file=%s\n' "$PATCH_FILE"
  printf 'patch_sha256=%s\n' "$PATCH_SHA"
  printf 'checkout_diff_sha256=%s\n' "$APPLIED_SHA"
  printf 'base_commit=%s\n' "$OFFICIAL_COMMIT"
  printf 'revert=%s\n' "git -C $VISION_ZERO_REPO checkout -- src/open-r1-multimodal"
} > "$VISION_ZERO_OUTPUT/paper_alignment_patch.txt"
printf '%q ' "${CMD[@]}" > "$VISION_ZERO_OUTPUT/launch_command.txt"
printf '\n' >> "$VISION_ZERO_OUTPUT/launch_command.txt"
cd "$TRAIN_ROOT"
"${CMD[@]}" 2>&1 | tee "$VISION_ZERO_OUTPUT/run.log"
echo "[done] Full model: $VISION_ZERO_OUTPUT"
echo "[eval] Evaluate this checkpoint with the same protocol as all other main-table models."
