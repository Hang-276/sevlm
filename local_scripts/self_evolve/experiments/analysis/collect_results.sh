#!/usr/bin/env bash
# Assemble finished evaluations into one model x benchmark table.
#
#   bash .../analysis/collect_results.sh                       # everything
#   bash .../analysis/collect_results.sh base ours vision_zero # just these
#   VERBOSE=1 bash .../analysis/collect_results.sh             # show how each score was read
#   OUT_JSON=table.json bash .../analysis/collect_results.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

WORK_DIR="${WORK_DIR:-$EVAL_WORK_DIR}"
ARGS=(--work-dir "$WORK_DIR" --format "${FORMAT:-md}")
[ $# -gt 0 ] && ARGS+=(--labels "$@")
[ "${VERBOSE:-0}" = "1" ] && ARGS+=(--verbose)
[ -n "${OUT_JSON:-}" ] && ARGS+=(--out "$OUT_JSON")

banner "collect results  |  work_dir=$WORK_DIR"
cd "$REPO"
"$PY" "$HERE/collect_results.py" "${ARGS[@]}"
