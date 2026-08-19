#!/usr/bin/env bash
# =============================================================================
# General single-model evaluation (standalone, self-contained)
#   Given one CKPT (a full model directory), runs a set of Tier-1 VLMEvalKit
#   evaluations serially.
#   — No LoRA merge (a full-parameter checkpoint already is a full model);
#   — No base-model run (evaluates only the CKPT you point it at);
#   — To change model / sampling: edit the "things you change" block below,
#     or override via command line / environment variables.
#
# Difference from main/eval.sh: that one auto-finds the final GRPO adapter,
# merges it, and evaluates base alongside for comparison. This script strips
# all of that — pure "feed a model path -> get scores" — handy for evaluating
# any checkpoint on the fly (base, a mid-run checkpoint, someone else's model).
#
# Usage:
#   # 1) Run as-is (default CKPT = base model; verifies the pipeline first)
#   bash local_scripts/self_evolve/experiments/analysis/eval_checkpoint.sh
#
#   # 2) Different checkpoint: first CLI arg (beats the CKPT default below)
#   bash .../eval_checkpoint.sh /path/to/runs/xxx/iter_000/checkpoints/grpo
#
#   # 3) Subset of datasets / different sampling / different judge (all env)
#   DATASETS="MMStar MMVP" bash .../eval_checkpoint.sh <ckpt>
#   TEMPERATURE=0.0 DO_SAMPLE=false bash .../eval_checkpoint.sh <ckpt>   # pure greedy
#   JUDGE=exact_matching bash .../eval_checkpoint.sh <ckpt>              # no API, local scoring
#
# Run it in tmux (the datasets run serially; a 7B model can take hours).
# =============================================================================
set -euo pipefail

# Evaluation uses whatever conda environment you have currently activated
# (activate the one with VLMEvalKit deps, e.g. vllmeval). Grab the current
# python first, before paths.sh's conda activate replaces it.
PY_CURRENT="$(command -v python || true)"

# paths.sh: for REPO / BASE_MODEL / ENV_FILE etc. Note it conda-activates the
# training environment (vision-zero) and points PY there — evaluation doesn't
# need that, so PY is set back to the current environment below.
# Only paths.sh is sourced, not experiments/lib/common.sh (that one carries merge
# logic this script doesn't use).
SE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # local_scripts/self_evolve
source "$SE_DIR/paths.sh"

# Override the PY that paths.sh set with the current environment's python
# (evaluation runs in your own environment; training is unaffected).
[ -n "$PY_CURRENT" ] && PY="$PY_CURRENT"
echo "[env] using python: $PY"

# =============================================================================
#  vvv Things you change: checkpoint / sampling / datasets — mostly just here vvv
# =============================================================================
# --- Which model to evaluate ---
# Put the full model directory here (edit this one line). Empty "" means
# evaluate the base model.
# Example: CKPT="$RUNS_ROOT/ours_final/iter_000/checkpoints/grpo/checkpoint-160"
CKPT=""

# Fallbacks: first CLI arg / env CKPT still override the value above; if
# nothing is set, evaluate base.
CKPT="${1:-${CKPT:-$BASE_MODEL}}"
[ -n "$CKPT" ] || CKPT="$BASE_MODEL"

# --- Result label (the results directory name, edit this one line) ---
# Hardcode a name if you want, e.g. LABEL="grpo_ckpt160". Empty "" derives it
# from the checkpoint directory name.
LABEL=""
# Fallback: env LABEL still overrides the value above.
LABEL="${LABEL:-}"

# --- Datasets (run serially, one by one). Options: the _DATA map below. ---
# Default is all 10: the 6 Tier-1 (MCQ + RealWorldQA + ChartQA)
#   + the locally prepared VStarBench / MMMU_Pro_10c / CV-Bench-2D / CV-Bench-3D.
# Already-run datasets reuse their old predictions (no re-inference); only the
# missing ones run. Use DATASETS=... to run a subset.
DATASETS="${DATASETS:-MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST VStarBench MMMU_Pro_10c CV-Bench-2D CV-Bench-3D}"

# --- Sampling (passed through to Qwen2VLChat; all effective on the default
#     transformers backend) ---
# Qwen2VLChat ships "near-greedy" defaults (top_p=0.001, top_k=1, temp=0.01,
# do_sample=true); they're written out here so they're easy to tweak. For
# strict greedy / reproducibility: DO_SAMPLE=false.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.01}"
TOP_P="${TOP_P:-0.001}"
TOP_K="${TOP_K:-1}"
DO_SAMPLE="${DO_SAMPLE:-true}"           # true / false (false = greedy; temp/top_p/top_k ignored)
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# --- Image resolution bounds (pixels). Default 1003520 (=1280*28*28) is safe
#     on a single 24G card; raise on H200. ---
MAX_PIXELS="${EVAL_MAX_PIXELS:-1003520}"
MIN_PIXELS="${EVAL_MIN_PIXELS:-200704}"

# --- Judge. Default gpt-4o-mini (needs OPENAI_* in .env); use exact_matching
#     to avoid the API. ---
JUDGE="${JUDGE:-gpt-4o-mini}"

# --- Run mode (passed to VLMEvalKit run.py --mode) ---
#   all   : inference + scoring (default)
#   infer : inference only, skip scoring (no judge / OPENAI_* needed; produces prediction files)
#   eval  : score existing inference results (no re-inference)
MODE="${MODE:-all}"
# vLLM backend: USE_VLLM=1 passes --use-vllm to run.py (default: transformers).
USE_VLLM="${USE_VLLM:-0}"
# =============================================================================
#  ^^^ Usually only the block above needs editing ^^^
# =============================================================================

# --- Tool/output paths (rarely changed; override via env if installed elsewhere) ---
export LMUData="${LMUData:-$LMUDATA}"
WORK_DIR="${WORK_DIR:-$EVAL_WORK_DIR}"

# ---- Up-front checks ----
[ -d "$CKPT" ] || { echo "[ERROR] CKPT directory does not exist: $CKPT" >&2; exit 2; }
[ -f "$CKPT/config.json" ] || { echo "[ERROR] $CKPT has no config.json (not a full model directory; this script does no LoRA merge — point it at a full model)" >&2; exit 2; }
[ -d "$VLMK" ] || { echo "[ERROR] VLMEvalKit not found: $VLMK" >&2; exit 2; }

# Default label = checkpoint directory name (base model special-cased to "base")
if [ -z "$LABEL" ]; then
  if [ "$CKPT" = "$BASE_MODEL" ]; then LABEL="base"; else LABEL="$(basename "$CKPT")"; fi
fi

# Results directory (VLMEvalKit writes <LABEL>_<DATASET>_*.csv there;
# *_acc.csv holds the scores)
RESULT_DIR="$WORK_DIR/$LABEL"

# MODE validity (run.py only accepts all/infer/eval)
case "$MODE" in
  all|infer|eval) ;;
  *) echo "[ERROR] invalid MODE='$MODE', options: all / infer / eval" >&2; exit 2;;
esac

# Load .env (judge API key); anything already exported wins
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$WORK_DIR" "$LMUData"

echo "======================================================"
echo "  checkpoint eval  |  label=$LABEL"
echo "  CKPT=$CKPT"
echo "  datasets=$DATASETS"
echo "  sampling: max_new_tokens=$MAX_NEW_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K do_sample=$DO_SAMPLE rep=$REPETITION_PENALTY"
echo "  pixels: min=$MIN_PIXELS max=$MAX_PIXELS   judge=$JUDGE"
echo "  mode=$MODE   use_vllm=$USE_VLLM"
echo "  work-dir=$WORK_DIR   LMUData=$LMUData"
echo "======================================================"

# ---- Build a self-contained temp config (label -> model_path + sampling;
#      leaves the repo's configs untouched) ----
TMP_CFG="$(mktemp /tmp/vlmeval_cfg_XXXX.json)"
# shellcheck disable=SC2064
trap "rm -f '$TMP_CFG'" EXIT
"$PY" - "$TMP_CFG" <<PY
import json, os, sys
dst = sys.argv[1]
# do_sample as a python bool
_ds = "${DO_SAMPLE}".strip().lower() in ("1", "true", "yes", "on")
out = {"model": {}, "data": {}}
out["model"]["${LABEL}"] = {
    "class": "Qwen2VLChat",
    "model_path": "${CKPT}",
    "min_pixels": ${MIN_PIXELS},
    "max_pixels": ${MAX_PIXELS},
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P},
    "top_k": ${TOP_K},
    "do_sample": _ds,
    "repetition_penalty": ${REPETITION_PENALTY},
    # Aligned with main/eval.sh: disable VLMEvalKit's custom prompt template
    # so the numbers stay comparable.
    "use_custom_prompt": False,
}
# Tier-1 dataset -> VLMEvalKit dataset class (MCQ via ImageMCQDataset, ChartQA
# via VQA). The class must be the real registered VLMEvalKit name: CV-Bench
# uses CVBench, MMMU-Pro uses MMMUProDataset.
_DATA = {
    "MMVP":         {"class": "ImageMCQDataset", "dataset": "MMVP"},
    "MMStar":       {"class": "ImageMCQDataset", "dataset": "MMStar"},
    "BLINK":        {"class": "ImageMCQDataset", "dataset": "BLINK"},
    "RealWorldQA":  {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
    "AI2D_TEST":    {"class": "ImageMCQDataset", "dataset": "AI2D_TEST"},
    "ChartQA_TEST": {"class": "ImageVQADataset", "dataset": "ChartQA_TEST"},
    # These 4 have tsvs already prepared in local LMUData.
    "VStarBench":   {"class": "ImageMCQDataset", "dataset": "VStarBench"},
    "MMMU_Pro_10c": {"class": "MMMUProDataset",  "dataset": "MMMU_Pro_10c"},
    "CV-Bench-2D":  {"class": "CVBench",         "dataset": "CV-Bench-2D"},
    "CV-Bench-3D":  {"class": "CVBench",         "dataset": "CV-Bench-3D"},
}
for d in "${DATASETS}".split():
    if d not in _DATA:
        sys.exit(f"[ERROR] unknown dataset '{d}', options: {list(_DATA)}")
    out["data"][d] = _DATA[d]
json.dump(out, open(dst, "w"), indent=2)
print(f"[cfg] model={list(out['model'])} datasets={list(out['data'])}", file=sys.stderr)
PY

# ---- Judge args: live judge needs OPENAI_*; exact_matching is fully local ----
# mode=infer does no scoring, so judge args and OPENAI_* checks are skipped.
JUDGE_ARGS=()
if [ "$MODE" = "infer" ]; then
  echo "[judge] mode=infer, scoring skipped (no judge / OPENAI_* needed)"
else
  JUDGE_ARGS+=(--judge "$JUDGE")
  if [ "$JUDGE" != "exact_matching" ]; then
    : "${OPENAI_BASE_URL:?live judge needs OPENAI_BASE_URL (put it in $ENV_FILE); or use JUDGE=exact_matching to stay local}"
    : "${OPENAI_API_KEY:?live judge needs OPENAI_API_KEY (put it in $ENV_FILE); or use JUDGE=exact_matching to stay local}"
    JUDGE_ARGS+=(--judge-base-url "$OPENAI_BASE_URL")
    echo "[judge] $JUDGE via $OPENAI_BASE_URL"
  else
    echo "[judge] exact_matching (no API)"
  fi
fi

# ---- Assemble run.py args (--mode passed through; USE_VLLM=1 adds --use-vllm) ----
# No --verbose: keep just the tqdm bar instead of printing every sample's
# full output (that floods the terminal).
RUN_ARGS=(--config "$TMP_CFG" --work-dir "$WORK_DIR" --mode "$MODE")
[ "$USE_VLLM" = "1" ] && RUN_ARGS+=(--use-vllm)
RUN_ARGS+=("${JUDGE_ARGS[@]}")

# ---- Run (VLMEvalKit run.py evaluates the datasets serially, in config order) ----
( cd "$VLMK" && "$PY" run.py "${RUN_ARGS[@]}" )

echo "======================================================"
echo "  [done] results in $RESULT_DIR/ (*_acc.csv holds the scores)"
echo "======================================================"
