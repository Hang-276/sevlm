#!/usr/bin/env bash
# =============================================================================
# Shared training runner: the full self-evolve loop (multi-round carry-forward
# + Reference-VLM screening + SFT/GRPO). Not run directly — the wrappers in
# main/ / ablation/ / sensitivity/ set a few environment variables
# (REWARD_JSON, STAGES, RUN_TAG, MARKER, ...) and exec this.
#
# Which reward an exp trains with is exactly its REWARD_JSON:
#   main/ours.sh:            reward_weights.json (design in docs/REWARD.md)
#   main/ours_no_process.sh: reward_outcome_only.json (answer=1.0, all else 0)
# Loop mechanics / trainer stages / hyperparameters stay identical across
# exps, so any two exps differ only in what their wrappers export.
#
# After training, evaluate with  main/eval.sh <run_dir>.
# =============================================================================
set -euo pipefail

# If the GPU box has no public egress, Reference-VLM / answer-judge calls go
# through a proxy. Set SELF_EVOLVE_PROXY (e.g. socks5h://127.0.0.1:18080);
# unset means direct connection.
if [ -n "${SELF_EVOLVE_PROXY:-}" ]; then
  export HTTPS_PROXY="$SELF_EVOLVE_PROXY" HTTP_PROXY="$SELF_EVOLVE_PROXY" \
         ALL_PROXY="$SELF_EVOLVE_PROXY" NO_PROXY="localhost,127.0.0.1"
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# DATASET_ROOT / BASE_MODEL / RUNS_ROOT come from paths.sh; change them there
# or override via env.
EXP_RUNS_ROOT="$RUNS_ROOT"

# ---------------- hyperparameters ----------------
EXP_NAME="${EXP_NAME:-ours_final}"
NUM_ITERATIONS="${NUM_ITERATIONS:-$LOOP_ITERATIONS}" # rounds of the loop: generate-screen-solve-score-train
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-$MAIN_NUM_TRAIN_TASKS}" # candidate tasks generated per round, before screening
NUM_GENERATIONS="${NUM_GENERATIONS:-$MAIN_NUM_GENERATIONS}"  # solver rollouts per task
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-60}"          # fallback/default step count (used by any stage without its own; also names banner/wandb)
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-$GRPO_STEPS_PER_ITER}"     # GRPO step count (overrides MAX_STEPS)
SFT_MAX_STEPS="${SFT_MAX_STEPS:-2}"      # SFT step count (overrides MAX_STEPS)
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
# Full-parameter training. 7B full-parameter GRPO (policy + reference model +
# rollout + Adam state) must be sharded across 8 cards or it OOMs.
#
# GRPO trainer backend (TRAINER_BACKEND):
#   fsdp2    — native PyTorch FSDP2. Chosen because DeepSpeed ZeRO-3 buries
#              the optimizer step inside backward, so when a degenerate batch
#              yields nan gradients under bf16 full-parameter GRPO there is no
#              clean way to skip the step (an old run died exactly like this
#              at step 36: finite loss but grad_norm=nan, weights turned nan,
#              next rollout's logits all nan, sampler emitted <|image_pad|>,
#              crash with "Image features and image tokens do not match").
#              Under FSDP2 the step happens outside backward, so
#              grpo_trainer.py's training_step can check gradient finiteness
#              before stepping and skip in sync across all cards.
#   deepspeed — kept as a switchable fallback (ZeRO-3): TRAINER_BACKEND=deepspeed.
# Either way, grpo_jsonl.py applies the qwen2_5vl forward monkey-patch (under
# sharding the vision tower must all-gather in sync across cards). If memory
# is tight on deepspeed, switch to zero3_offload.json.
# Note: SFT always runs on DeepSpeed (unaffected by TRAINER_BACKEND), so
# DEEPSPEED_CONFIG must stay set for it.
TRAINER_BACKEND="${TRAINER_BACKEND:-deepspeed}"
DEEPSPEED_CONFIG="$REPO/local_scripts/zero3.json"
FSDP_CONFIG="$REPO/local_scripts/fsdp2_qwen2_5vl.json"
# Reference VLM (task screening; live use needs the key in .env)
REFERENCE_PROVIDER="${REFERENCE_PROVIDER:-openai}"
REFERENCE_MODEL="${REFERENCE_MODEL:-gpt-4o}"
# REFERENCE_BASE_URL comes from paths.sh
# The full method trains on the default reward config; wrappers override
# REWARD_JSON for other exps. Design notes: local_scripts/self_evolve/docs/REWARD.md.
REWARD_JSON="${REWARD_JSON:-$SE_DIR/configs/reward/reward_weights.json}"
# How many rollouts per task the answer judge scores. Must equal
# NUM_GENERATIONS so every rollout gets judged (otherwise the answer reward
# dimension is biased).
ANSWER_JUDGE_SAMPLE_N=$NUM_GENERATIONS

# ---- per-stage hyperparameters ----
# Everything defaults to configs/train_defaults.sh (sourced via common.sh);
# only overrides for this run go here. That way baselines / ablations / main
# exps share one trainer config, and exps differ only in reward config and
# which stages run.
# Stage order: SFT -> GRPO; GRPO starts from the SFT checkpoint.
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-0.9}"

GRPO_EXTRA="$(build_grpo_extra)"
SFT_EXTRA="$(build_sft_extra)"
# --------------------------------------

require_paths
load_env
[ -f "$REWARD_JSON" ] || { echo "[ERROR] reward config does not exist: $REWARD_JSON" >&2; exit 2; }

# ---- RUN_TAG (fixed name, no timestamp) ----
# If OUT_DIR exists, the run resumes automatically (two levels of resume):
#   - iter_NNN/iteration_state.json present => that round is complete, skip it;
#     re-enter at the first unfinished iter
#   - inside the re-entered round: candidate/accepted/reference jsonl all
#     present => skip generator + Reference VLM; raw_solver_trajectories.jsonl
#     present => skip solver rollouts
# New experiment: change the name below (a fresh directory starts from scratch).
# Force a fresh run in the same directory: SELF_EVOLVE_DISABLE_RESUME=1.
# Full-parameter training: use a fresh directory so you don't resume onto an
# old LoRA run's adapter checkpoints.
RUN_TAG="${RUN_TAG:-ours_final}"
OUT_DIR="$EXP_RUNS_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

# Loop-mechanism switches (aligned with the launcher described in docs)
export SELF_EVOLVE_TOO_HARD_GAP="${SELF_EVOLVE_TOO_HARD_GAP:-2}"
export SELF_EVOLVE_ANSWER_JUDGE="${SELF_EVOLVE_ANSWER_JUDGE:-1}"
export SELF_EVOLVE_ANSWER_JUDGE_LIVE="${SELF_EVOLVE_ANSWER_JUDGE_LIVE:-1}"
export SELF_EVOLVE_ANSWER_JUDGE_FORCE="${SELF_EVOLVE_ANSWER_JUDGE_FORCE:-0}"
export SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N="${SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N:-$ANSWER_JUDGE_SAMPLE_N}"

# --- Per-step GRPO trajectory + score dumps (on by default) ---
# When on, every GRPO step writes each rollout of that step to disk (full
# trajectory text + five-dim reward breakdown answer/grounding/process/
# consistency/budget + scalar + advantage + bbox audit fields). Layout: one
# folder per step, each of the 8 cards writes its own rank file (no locking):
#   $OUT_DIR/grpo_dumps/step_000042/rank0.jsonl ... rank7.jsonl
# The step's full batch is the union of all rankN.jsonl in that folder.
# Dumps are best-effort: a failure logs and training continues. To turn off:
# export SELF_EVOLVE_GRPO_DUMP_DIR= (empty).
export SELF_EVOLVE_GRPO_DUMP_DIR="${SELF_EVOLVE_GRPO_DUMP_DIR:-$OUT_DIR/grpo_dumps}"

# Trainer hyperparameter pass-through (the loop entry appends these to each
# stage's command line, overriding built-in defaults)
export SELF_EVOLVE_GRPO_EXTRA_ARGS="$GRPO_EXTRA"
export SELF_EVOLVE_SFT_EXTRA_ARGS="$SFT_EXTRA"

# --- wandb training curves (off by default; WANDB=1 enables) ---
# When on, the GRPO/SFT trainers report to wandb (standard TRL curves:
# loss/lr/grad_norm/reward). Loop-level signals (reward_std /
# bbox_valid_rate etc.) stay in run.log. Needs WANDB_API_KEY (in .env, or
# `wandb login` first).
if [ "${WANDB:-0}" = "1" ]; then
  export SELF_EVOLVE_REPORT_TO=wandb
  export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
  export WANDB_NAME="${WANDB_NAME:-ours_${NUM_ITERATIONS}iters_${MAX_STEPS}step}"
  : "${WANDB_API_KEY:?WANDB=1 needs WANDB_API_KEY (put it in $ENV_FILE or run wandb login first)}"
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
  --num-iterations "$NUM_ITERATIONS"
  --allow-more-than-two-iterations
  --num-train-tasks "$NUM_TRAIN_TASKS"
  --num-generations "$NUM_GENERATIONS"
  $(solver_cli_args)
  $(generator_cli_args)
  --seed "$SEED"
  --reference-provider "$REFERENCE_PROVIDER"
  --reference-model "$REFERENCE_MODEL"
  --reference-base-url "$REFERENCE_BASE_URL"
  --enable-openai-reference-vlm
  --max-trainer-steps "$MAX_STEPS"
  --trainer-num-gpus "$NUM_GPUS"
  --trainer-per-device-train-batch-size "$PER_DEVICE_BATCH"
  --trainer-gradient-accumulation-steps "$GRAD_ACCUM"
  --trainer-grpo-num-generations "$NUM_GENERATIONS"
  $(stage_cli_args)
)
# GRPO backend selection. SFT always uses DeepSpeed, so
# --trainer-deepspeed-config is always passed; GRPO picks fsdp2 or deepspeed
# per TRAINER_BACKEND.
ARGS+=(--trainer-backend "$TRAINER_BACKEND")
[ -n "$DEEPSPEED_CONFIG" ] && ARGS+=(--trainer-deepspeed-config "$DEEPSPEED_CONFIG")
if [ "$TRAINER_BACKEND" = "fsdp2" ]; then
  [ -f "$FSDP_CONFIG" ] || { echo "[ERROR] FSDP2 config does not exist: $FSDP_CONFIG" >&2; exit 2; }
  ARGS+=(--trainer-fsdp-config "$FSDP_CONFIG")
fi

banner "train  self-evolve loop
  RUN_TAG=$RUN_TAG
  OUT_DIR=$OUT_DIR
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  iters=$NUM_ITERATIONS tasks=$NUM_TRAIN_TASKS gen=$NUM_GENERATIONS steps=$MAX_STEPS gpus=$NUM_GPUS mode=FULL-PARAM(GRPO=$([ "$TRAINER_BACKEND" = fsdp2 ] && echo "fsdp2=$(basename "$FSDP_CONFIG")" || echo "deepspeed=$(basename "$DEEPSPEED_CONFIG")"); SFT=deepspeed=$(basename "$DEEPSPEED_CONFIG"))
  stages=$STAGES  (sft=$SFT_MAX_STEPS grpo=$GRPO_MAX_STEPS step)
  reward=$REWARD_JSON  reference=$REFERENCE_MODEL(live)"

cd "$REPO"
# On resume, keep the historical log: append rather than overwrite (-a).
"$PY" "$LOOP_ENTRY" "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"

record_run "${MARKER:-ours}" "$OUT_DIR"
banner "Training done. Evaluate: bash local_scripts/self_evolve/experiments/main/eval.sh $OUT_DIR"
