#!/usr/bin/env bash
# Evaluate any run. A self-evolve run gets its final round's weights merged
# first; a trainer output directory uses its latest checkpoint; a plain model
# directory is used as-is.
#
#   bash .../main/eval.sh <run_dir>
#   bash .../main/eval.sh                    # the most recent ours run
#   MARKER=grpo_baseline bash .../main/eval.sh
#   WITH_BASE=0 DATASETS="MMVP" bash .../main/eval.sh <run_dir>
#
# Every exp shares the isolated checkpoint evaluator, so all nine target
# datasets and decoding/judge settings stay identical.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

RUN_DIR="${1:-$(last_run "${MARKER:-ours}")}"
[ -n "$RUN_DIR" ] || { echo "[ERROR] no run_dir given and no recorded run for ${MARKER:-ours}. Usage: eval.sh <run_dir>" >&2; exit 2; }
[ -d "$RUN_DIR" ] || { echo "[ERROR] run_dir does not exist: $RUN_DIR" >&2; exit 2; }

LABEL="${LABEL:-$(basename "$RUN_DIR")}"
banner "eval  |  run=$RUN_DIR"
MODEL="$(resolve_eval_model "$RUN_DIR")"
export WORKSPACE BASE_MODEL VLMK LMUDATA
export EVAL_PY="${EVAL_PY:-$PY}"
export DATASETS="${DATASETS:-MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST MMMU_Pro_10c CV-Bench-2D CV-Bench-3D}"
case "${WITH_BASE:-1}" in
  0|1) ;;
  *) echo "[ERROR] WITH_BASE must be 0 or 1" >&2; exit 2;;
esac
LABEL="$LABEL" bash "$EXP_DIR/analysis/eval_checkpoint.sh" "$MODEL"
if [ "${WITH_BASE:-1}" = "1" ] && [ "$LABEL" != base ]; then
  LABEL=base bash "$EXP_DIR/analysis/eval_checkpoint.sh" "$BASE_MODEL"
fi
