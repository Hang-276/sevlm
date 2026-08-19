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
# Every exp shares this single entry, so dataset / judge / decoding settings
# stay identical.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

RUN_DIR="${1:-$(last_run "${MARKER:-ours}")}"
[ -n "$RUN_DIR" ] || { echo "[ERROR] no run_dir given and no recorded run for ${MARKER:-ours}. Usage: eval.sh <run_dir>" >&2; exit 2; }
[ -d "$RUN_DIR" ] || { echo "[ERROR] run_dir does not exist: $RUN_DIR" >&2; exit 2; }

LABEL="${LABEL:-$(basename "$RUN_DIR")}"
banner "eval  |  run=$RUN_DIR"
require_paths
MODEL="$(resolve_eval_model "$RUN_DIR")"

PAIRS=("$LABEL=$MODEL")
[ "${WITH_BASE:-1}" = "1" ] && PAIRS+=("base=$BASE_MODEL")
run_tier1_eval "${PAIRS[@]}"
