#!/usr/bin/env bash
# =============================================================================
# General single-model evaluation (standalone, self-contained)
#   Given one CKPT (a full model directory), runs the nine target VLMEvalKit
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

# Standalone evaluation: do not source paths.sh or activate the training env.
# EVAL_PY (or PY) overrides the currently activated evaluation interpreter.
SE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${REPO:-$(cd "$SE_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$REPO}"
PY="${EVAL_PY:-${PY:-$(command -v python || command -v python3 || true)}}"
[ -n "$PY" ] || { echo "[ERROR] Activate the evaluation environment or set EVAL_PY" >&2; exit 2; }
PY="$(command -v "$PY")"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/models/Qwen2.5-VL-7B-Instruct}"
VLMK="${VLMK:-$WORKSPACE/eval/VLMEvalKit}"
LMUDATA="${LMUDATA:-$WORKSPACE/eval/LMUData}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-$WORKSPACE/eval/results}"
ENV_FILE="${ENV_FILE:-$REPO/.env}"
echo "[env] using python: $PY"

# Priority: explicit CLI model > CKPT environment > base model.
CKPT="${1:-${CKPT:-$BASE_MODEL}}"
LABEL="${LABEL:-}"

# --- Datasets (run serially, one by one). Options: the _DATA map below. ---
# The default is exactly the nine target benchmarks. VStarBench remains an
# opt-in extra via DATASETS. Use a separate LABEL for a different protocol.
DATASETS="${DATASETS:-MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST MMMU_Pro_10c CV-Bench-2D CV-Bench-3D}"

# --- Sampling (passed through to Qwen2VLChat) ---
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
# vLLM backend by default; USE_VLLM=0 selects Transformers.
USE_VLLM="${USE_VLLM:-1}"
# =============================================================================
#  ^^^ Usually only the block above needs editing ^^^
# =============================================================================

# --- Tool/output paths (rarely changed; override via env if installed elsewhere) ---
export LMUData="${LMUData:-$LMUDATA}"
WORK_DIR="${WORK_DIR:-$EVAL_WORK_DIR}"

# ---- Up-front checks ----
[ -d "$CKPT" ] || { echo "[ERROR] CKPT directory does not exist: $CKPT" >&2; exit 2; }
[ -f "$CKPT/config.json" ] || { echo "[ERROR] $CKPT has no config.json (not a full model directory; this script does no LoRA merge — point it at a full model)" >&2; exit 2; }
[ -f "$VLMK/run.py" ] || { echo "[ERROR] VLMEvalKit not found: $VLMK" >&2; exit 2; }
CKPT="$(cd "$CKPT" && pwd -P)"
VLMK="$(cd "$VLMK" && pwd -P)"
if [ -d "$BASE_MODEL" ]; then BASE_MODEL="$(cd "$BASE_MODEL" && pwd -P)"; fi

# Default label = checkpoint directory name (base model special-cased to "base")
if [ -z "$LABEL" ]; then
  if [ "$CKPT" = "$BASE_MODEL" ]; then LABEL="base"; else LABEL="$(basename "$CKPT")"; fi
fi

# Results directory (VLMEvalKit writes <LABEL>_<DATASET>_*.csv there;
# *_acc.csv holds the scores)
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "[ERROR] LABEL must contain letters, numbers, dots, underscores or hyphens" >&2; exit 2; }
mkdir -p "$WORK_DIR" "$LMUData"
WORK_DIR="$(cd "$WORK_DIR" && pwd -P)"
export LMUData="$(cd "$LMUData" && pwd -P)"
RESULT_DIR="$WORK_DIR/$LABEL"

# MODE validity (run.py only accepts all/infer/eval)
case "$MODE" in
  all|infer|eval) ;;
  *) echo "[ERROR] invalid MODE='$MODE', options: all / infer / eval" >&2; exit 2;;
esac

# Optional .env; preserve judge values explicitly supplied in the shell.
KEY_SET="${OPENAI_API_KEY+x}"; KEY_VALUE="${OPENAI_API_KEY:-}"
URL_SET="${OPENAI_BASE_URL+x}"; URL_VALUE="${OPENAI_BASE_URL:-}"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
if [ "$KEY_SET" = x ]; then export OPENAI_API_KEY="$KEY_VALUE"; fi
if [ "$URL_SET" = x ]; then export OPENAI_BASE_URL="$URL_VALUE"; fi
unset KEY_VALUE URL_VALUE
# Keep artifact formats fixed: the collector reads *_acc.csv, and changing the
# prediction format within a LABEL would defeat VLMEvalKit's reuse lookup.
export PRED_FORMAT=xlsx EVAL_FORMAT=csv
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RESULT_DIR"

echo "======================================================"
echo "  checkpoint eval  |  label=$LABEL"
echo "  CKPT=$CKPT"
echo "  datasets=$DATASETS"
echo "  sampling: max_new_tokens=$MAX_NEW_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K do_sample=$DO_SAMPLE rep=$REPETITION_PENALTY"
echo "  pixels: min=$MIN_PIXELS max=$MAX_PIXELS   judge=$JUDGE"
echo "  mode=$MODE   use_vllm=$USE_VLLM"
echo "  work-dir=$WORK_DIR   LMUData=$LMUData"
echo "======================================================"

# Save the exact protocol; stdlib-only, safe to preview on a CPU machine.
# Pass values as arguments, never interpolate paths/labels into Python source.
TMP_CFG="$RESULT_DIR/eval_config.json"
"$PY" - "$TMP_CFG" "$LABEL" "$CKPT" "$MIN_PIXELS" "$MAX_PIXELS" \
  "$MAX_NEW_TOKENS" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$DO_SAMPLE" \
  "$REPETITION_PENALTY" "$DATASETS" "$JUDGE" "$USE_VLLM" "$VLMK" "$LMUData" \
  "${OPENAI_BASE_URL:-}" "$SE_DIR/experiments/analysis" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
(dst, label, ckpt, minpx, maxpx, length, temp, top_p, top_k,
 sample, repetition, datasets, judge, use_vllm, vlmk, lmu_data, judge_url, analysis) = sys.argv[1:]
sys.path.insert(0, analysis)
from check_eval_results import dataset_fingerprints
if sample.lower() not in ("true", "false", "1", "0"):
    sys.exit("[ERROR] DO_SAMPLE must be true/false or 1/0")
out = {"model": {label: {
    "class": "Qwen2VLChat", "model_path": ckpt,
    "min_pixels": int(minpx), "max_pixels": int(maxpx),
    "max_new_tokens": int(length), "temperature": float(temp),
    "top_p": float(top_p), "top_k": int(top_k),
    "do_sample": sample.lower() in ("true", "1"),
    "repetition_penalty": float(repetition), "use_custom_prompt": False,
}}, "data": {}}
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
requested = datasets.split()
if len(requested) != len(set(requested)):
    sys.exit("[ERROR] DATASETS contains duplicate names")
for d in requested:
    if d not in _DATA:
        sys.exit(f"[ERROR] unknown dataset '{d}', options: {list(_DATA)}")
    out["data"][d] = _DATA[d]
if not out["data"]:
    sys.exit("[ERROR] DATASETS is empty")
# Refuse changed settings in a cache directory. Dataset subsets may be added
# incrementally; every dataset alias retains its original construction config.
try:
    revision = subprocess.check_output(
        ["git", "-C", vlmk, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    revision = None
model_dir = Path(ckpt)
weight_files = sorted({*model_dir.glob("*.safetensors"), *model_dir.glob("*.bin"),
                       *model_dir.glob("*.index.json"), model_dir / "config.json"})
fingerprint = [{"name": p.name, "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns} for p in weight_files if p.is_file()]
kit_hash = hashlib.sha256()
for path in sorted([Path(vlmk) / "run.py", *(Path(vlmk) / "vlmeval").rglob("*.py")]):
    kit_hash.update(str(path.relative_to(vlmk)).encode())
    kit_hash.update(b"\0")
    kit_hash.update(path.read_bytes())
    kit_hash.update(b"\0")
protocol = {"config": out, "judge": judge, "use_vllm": use_vllm,
            "judge_base_url": judge_url.rstrip("/") if judge != "exact_matching" else None,
            "vlmeval_path": vlmk, "vlmeval_commit": revision,
            "vlmeval_code_sha256": kit_hash.hexdigest(),
            "lmu_data_path": lmu_data, "checkpoint_fingerprint": fingerprint}
meta = Path(dst).with_name("eval_protocol.json")
if meta.exists():
    previous = json.loads(meta.read_text())
    old_data = previous.get("config", {}).get("data", {})
    old_base = {key: value for key, value in previous.items() if key != "dataset_fingerprints"}
    old_base["config"] = {**previous.get("config", {}), "data": {}}
    new_base = {**protocol, "config": {**protocol["config"], "data": {}}}
    if old_base != new_base:
        sys.exit("[ERROR] Evaluation protocol changed. Use a new LABEL or WORK_DIR to avoid cached predictions.")
    for name, config in out["data"].items():
        if name in old_data and old_data[name] != config:
            sys.exit(f"[ERROR] Dataset protocol changed for {name}; use a new LABEL or WORK_DIR.")
    protocol["config"]["data"] = {**old_data, **out["data"]}
    current_data = dataset_fingerprints(Path(lmu_data), protocol["config"]["data"])
    for name, fingerprint in previous.get("dataset_fingerprints", {}).items():
        if fingerprint is not None and fingerprint != current_data.get(name):
            sys.exit(f"[ERROR] Dataset source changed for {name}; use a new LABEL or WORK_DIR.")
else:
    if any(path for suffix in (".xlsx", ".tsv", ".csv", ".json")
           for path in meta.parent.rglob(f"{label}_*{suffix}")):
        sys.exit("[ERROR] Existing evaluation artifacts have no protocol; use a new LABEL or WORK_DIR.")
    current_data = dataset_fingerprints(Path(lmu_data), protocol["config"]["data"])
protocol["dataset_fingerprints"] = current_data
Path(dst).write_text(json.dumps(out, indent=2))
meta.write_text(json.dumps(protocol, indent=2))
print(f"[cfg] saved {dst}; datasets={list(out['data'])}", file=sys.stderr)
PY

# ---- Judge args: live judge needs OPENAI_*; exact_matching is fully local ----
# mode=infer does no scoring, so judge args and OPENAI_* checks are skipped.
JUDGE_ARGS=()
if [ "$MODE" = "infer" ] || [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[judge] inference/preview: no judge request or API-key requirement"
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
# Current VLMEvalKit creates one eval_id directory per invocation. Explicit
# --reuse is required to pick up a completed prediction from an earlier run.
if grep -q -- "'--reuse'" "$VLMK/run.py"; then
  RUN_ARGS+=(--reuse)
else
  echo "[WARN] This VLMEvalKit has no --reuse option; check its cache behavior before rerunning a LABEL." >&2
fi
if [ "$MODE" != infer ] && [ "${DRY_RUN:-0}" != 1 ]; then
  RUN_ARGS+=("${JUDGE_ARGS[@]}")
fi

# Preview does not import VLMEvalKit/torch, load a model, or start inference.
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[preview] saved config=$TMP_CFG; no model loaded; live judge credentials not checked"
  exit 0
fi

# ---- Run (VLMEvalKit run.py evaluates the datasets serially, in config order) ----
CHECK_RESULTS="$SE_DIR/experiments/analysis/check_eval_results.py"
STATUS_BEFORE="$("$PY" "$CHECK_RESULTS" --result-dir "$RESULT_DIR" --snapshot)"
INVOCATION_START="$("$PY" -c 'import time; print(time.time())')"
( cd "$VLMK" && "$PY" run.py "${RUN_ARGS[@]}" ) 2>&1 | tee -a "$RESULT_DIR/eval.log"
CHECK_ARGS=(--result-dir "$RESULT_DIR" --label "$LABEL" --mode "$MODE"
  --lmu-data "$LMUData" --previous-status "$STATUS_BEFORE" --since "$INVOCATION_START")
[ ! -f "$VLMK/vlmeval/smp/status_report.py" ] || CHECK_ARGS+=(--require-status)
read -r -a REQUESTED_DATASETS <<< "$DATASETS"
"$PY" "$CHECK_RESULTS" "${CHECK_ARGS[@]}" --datasets "${REQUESTED_DATASETS[@]}"

echo "======================================================"
echo "  [done] results in $RESULT_DIR/ (keep predictions and all dataset-specific score files)"
echo "======================================================"
