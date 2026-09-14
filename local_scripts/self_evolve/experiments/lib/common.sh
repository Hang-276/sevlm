#!/usr/bin/env bash
# =============================================================================
# Shared helpers for everything under experiments/. Sourced by the runners in
# lib/ and the thin wrappers in the subdirectories. Never executed directly,
# so no set -e here (the caller sets it).
# =============================================================================

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$LIB_DIR/.." && pwd)"      # local_scripts/self_evolve/experiments
SE_DIR="$(cd "$EXP_DIR/.." && pwd)"       # local_scripts/self_evolve
RUNNER="$LIB_DIR/run_self_evolve.sh"
source "$SE_DIR/paths.sh"                  # REPO / PY / DATASET_ROOT / BASE_MODEL / RUNS_ROOT / REWARD_CONFIG / ENV_FILE ...
source "$SE_DIR/configs/train_defaults.sh" # training knobs + build_*_extra / solver_cli_args

MERGE_LORA="$SE_DIR/workflow/merge_lora.py"

# One root for all experiment artifacts (training runs + evaluation results)
EXP_RUNS_ROOT="${EXP_RUNS_ROOT:-$RUNS_ROOT}"
# Marker files recording each exp's run dir, used by eval when called without args.
MARKER_DIR="${MARKER_DIR:-$EXP_RUNS_ROOT/.markers}"

record_run() { mkdir -p "$MARKER_DIR"; echo "$2" > "$MARKER_DIR/$1"; }
last_run() { cat "$MARKER_DIR/$1" 2>/dev/null || true; }

# Patch the default reward config into a temp file.
# Usage: patched_reward_config <dotted.key=value> [...]
patched_reward_config() {
  local base="${REWARD_JSON_BASE:-$SE_DIR/configs/reward/reward_weights.json}"
  local out; out="$(mktemp /tmp/reward_XXXXXX.json)"
  if ! "$PY" "$LIB_DIR/patch_reward_config.py" "$base" "$out" "$@" >/dev/null; then
    echo "[ERROR] failed to build reward config: $* (base=$base)" >&2
    rm -f "$out"; return 3
  fi
  [ -s "$out" ] || { echo "[ERROR] reward config came out empty: $out" >&2; rm -f "$out"; return 3; }
  echo "$out"
}

banner() {
  echo "======================================================"
  echo "  $*"
  echo "======================================================"
}

# Load .env (API keys etc.); anything already exported in the shell wins.
load_env() { [ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }; }

# Existence checks up front — unfilled placeholder paths fail here, not
# halfway through training.
require_paths() {
  case "$WORKSPACE" in */PATH/TO/*) echo "[ERROR] WORKSPACE in paths.sh is still the placeholder; fill in the real path first" >&2; exit 2;; esac
  [ -e "$DATASET_ROOT" ] || { echo "[ERROR] DATASET_ROOT does not exist: $DATASET_ROOT (download the CLEVR dataset first, see the comments in paths.sh)" >&2; exit 2; }
  [ -d "$DATASET_ROOT/output/replacement_images" ] || { echo "[ERROR] $DATASET_ROOT/output/replacement_images not found. DATASET_ROOT must be the directory CONTAINING output/, not output/ itself." >&2; exit 2; }
  [ -e "$BASE_MODEL" ]   || { echo "[ERROR] BASE_MODEL does not exist: $BASE_MODEL (download Qwen2.5-VL-7B-Instruct first, see the comments in paths.sh)" >&2; exit 2; }
}

# ---------------------------------------------------------------------------
# Merge one iteration's GRPO LoRA adapter into a full model directory that
# evaluation can load directly. The training entry only auto-merges the
# previous round's adapter when the next round starts, so the final round's
# adapter is never merged automatically — evaluation must merge it once
# by hand (same final_after logic as in the docs).
#
# Usage: merge_final_adapter <run_dir> <out_full_model_dir>
#   Picks the highest-numbered iter_NNN/checkpoints/grpo under <run_dir>.
# On success, echoes the full-model directory path for the caller to capture.
# ---------------------------------------------------------------------------
merge_final_adapter() {
  local run_dir="$1" out_dir="$2"
  [ -d "$run_dir" ] || { echo "[ERROR] run_dir does not exist: $run_dir" >&2; return 2; }

  # Highest-numbered round's GRPO adapter
  local adapter
  adapter="$(ls -d "$run_dir"/iter_*/checkpoints/grpo 2>/dev/null | sort | tail -1)"
  [ -n "$adapter" ] || { echo "[ERROR] no iter_*/checkpoints/grpo found under $run_dir" >&2; return 2; }
  # Neither a LoRA adapter nor a full model -> invalid directory, stop here.
  if [ ! -f "$adapter/adapter_config.json" ] && [ ! -f "$adapter/config.json" ]; then
    echo "[ERROR] $adapter has neither adapter_config.json nor config.json (not a LoRA adapter, not a full model)" >&2
    return 2
  fi

  # Full-parameter training: the final checkpoints/grpo already is a full
  # model (config.json, no adapter_config.json) — no merge needed, return it
  # as the evaluation model directly.
  if [ ! -f "$adapter/adapter_config.json" ] && [ -f "$adapter/config.json" ]; then
    echo "[merge] final checkpoint already a full model (full-parameter training), skipping merge: $adapter" >&2
    echo "$adapter"
    return 0
  fi

  echo "[merge] final adapter: $adapter" >&2
  echo "[merge] output full model: $out_dir" >&2
  if [ -d "$out_dir" ] && [ -f "$out_dir/config.json" ]; then
    echo "[merge] already exists, skipping merge (delete $out_dir to re-merge)" >&2
  else
    "$PY" "$MERGE_LORA" --adapter-path "$adapter" --output-dir "$out_dir" --torch-dtype bf16 --overwrite 1>&2 \
      || { echo "[ERROR] merge_lora failed" >&2; return 3; }
  fi
  echo "$out_dir"   # stdout carries only the path, for $(...) capture
}

# ---------------------------------------------------------------------------
# Resolve a directory into a model path evaluation can load. Three cases:
#   1. self-evolve run dir (has iter_*/checkpoints/grpo)  -> merge the final round
#   2. trainer output dir (has checkpoint-N)              -> take the highest-numbered one
#   3. already a full model dir (has config.json)         -> return as-is
# On success, stdout carries only the path.
# ---------------------------------------------------------------------------
resolve_eval_model() {
  local dir="$1" out_dir="${2:-$1/final_merged_model}"
  [ -d "$dir" ] || { echo "[ERROR] directory does not exist: $dir" >&2; return 2; }

  if ls -d "$dir"/iter_*/checkpoints/grpo >/dev/null 2>&1; then
    merge_final_adapter "$dir" "$out_dir"
    return $?
  fi

  local ckpt
  ckpt="$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
  if [ -n "$ckpt" ]; then
    echo "[eval] using latest checkpoint: $ckpt" >&2
    if [ -f "$ckpt/adapter_config.json" ]; then
      "$PY" "$MERGE_LORA" --adapter-path "$ckpt" --output-dir "$out_dir" \
        --torch-dtype bf16 --overwrite 1>&2 || return 3
      echo "$out_dir"
    else
      echo "$ckpt"
    fi
    return 0
  fi

  if [ -f "$dir/config.json" ]; then
    echo "$dir"
    return 0
  fi

  echo "[ERROR] $dir has no iter_*/checkpoints/grpo, no checkpoint-*, and is not a model directory" >&2
  return 2
}

# ---------------------------------------------------------------------------
# Run the Tier-1 VLMEvalKit evaluation on arbitrary model paths
# (self-contained temp config). Pass one "label=model_path" per model.
#
# Usage: run_tier1_eval <label1=path1> [<label2=path2> ...]
#   Env overrides: DATASETS / JUDGE / VLMK / LMUData / WORK_DIR / EVAL_MAX_PIXELS
# ---------------------------------------------------------------------------
run_tier1_eval() {
  local VLMK="${VLMK:?VLMEvalKit path not set, see paths.sh}"
  export LMUData="${LMUData:-$LMUDATA}"
  local WORK_DIR="${WORK_DIR:-$EVAL_WORK_DIR}"
  local DATASETS="${DATASETS:-MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST}"
  local JUDGE="${JUDGE:-gpt-4o-mini}"
  local MAXPX="${EVAL_MAX_PIXELS:-1003520}"   # 1280*28*28, safe on a single 24G card; raise on H200
  local MINPX="${EVAL_MIN_PIXELS:-200704}"

  [ -d "$VLMK" ] || { echo "[ERROR] VLMEvalKit not installed: $VLMK (see experiments/README.md)" >&2; return 2; }
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  load_env
  mkdir -p "$WORK_DIR"

  # Build a self-contained temp config (label -> model_path). A missing
  # model path fails right here.
  local TMP_CFG; TMP_CFG="$(mktemp /tmp/vlmeval_cfg_XXXX.json)"
  # shellcheck disable=SC2064
  trap "rm -f '$TMP_CFG'" RETURN
  "$PY" - "$TMP_CFG" "$DATASETS" "$MINPX" "$MAXPX" "$@" <<'PY'
import json, os, sys
dst, datasets, minpx, maxpx = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4])
pairs = sys.argv[5:]
out = {"model": {}, "data": {}}
for p in pairs:
    if "=" not in p:
        sys.exit(f"[ERROR] expected 'label=model_path', got: {p}")
    label, mp = p.split("=", 1)
    if not os.path.isdir(mp):
        sys.exit(f"[ERROR] model '{label}' path does not exist: {mp}")
    out["model"][label] = {
        "class": "Qwen2VLChat", "model_path": mp,
        "min_pixels": minpx, "max_pixels": maxpx, "use_custom_prompt": False,
    }
_DATA = {
    "MMVP": {"class": "ImageMCQDataset", "dataset": "MMVP"},
    "MMStar": {"class": "ImageMCQDataset", "dataset": "MMStar"},
    "BLINK": {"class": "ImageMCQDataset", "dataset": "BLINK"},
    "RealWorldQA": {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
    "AI2D_TEST": {"class": "ImageMCQDataset", "dataset": "AI2D_TEST"},
    "ChartQA_TEST": {"class": "ImageVQADataset", "dataset": "ChartQA_TEST"},
}
for d in datasets:
    if d not in _DATA:
        sys.exit(f"[ERROR] unknown dataset '{d}', Tier-1 options: {list(_DATA)}")
    out["data"][d] = _DATA[d]
json.dump(out, open(dst, "w"), indent=2)
print(f"[cfg] models={list(out['model'])} datasets={list(out['data'])}", file=sys.stderr)
PY
  [ $? -eq 0 ] || return 3

  echo "[eval] LMUData=$LMUData work-dir=$WORK_DIR" >&2
  echo "[eval] datasets: $DATASETS" >&2
  echo "[eval] models:   $*" >&2

  local JUDGE_ARGS=(--judge "$JUDGE")
  if [ "$JUDGE" != "exact_matching" ]; then
    : "${OPENAI_BASE_URL:?live judge needs OPENAI_BASE_URL (from .env); or use JUDGE=exact_matching to stay local}"
    : "${OPENAI_API_KEY:?live judge needs OPENAI_API_KEY (from .env); or use JUDGE=exact_matching to stay local}"
    JUDGE_ARGS+=(--judge-base-url "$OPENAI_BASE_URL")
    echo "[judge] $JUDGE via $OPENAI_BASE_URL" >&2
  else
    echo "[judge] exact_matching (no API)" >&2
  fi

  local RUN_ARGS=(--config "$TMP_CFG" --work-dir "$WORK_DIR" --verbose "${JUDGE_ARGS[@]}")
  [ "${USE_VLLM:-1}" = "1" ] && RUN_ARGS+=(--use-vllm)
  ( cd "$VLMK" && "$PY" run.py "${RUN_ARGS[@]}" )
  echo "[done] results in $WORK_DIR (per-model subdirectories; *_acc.csv holds the scores)" >&2
}
