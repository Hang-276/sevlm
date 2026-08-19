#!/usr/bin/env bash
# Analysis: compute collapse diagnostics from a run's artifacts — no
# retraining.
#
#   bash .../analysis/collapse_diagnostics.sh <run_dir>
#   bash .../analysis/collapse_diagnostics.sh          # the most recent ours run
#
# Per round it reports: label-prior share, per-field accuracy, regret,
# advantage collapse rate, counterfactual sensitivity, predicted-box count
# distribution.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

RUN_DIR="${1:-$(last_run "${MARKER:-ours}")}"
[ -n "$RUN_DIR" ] || { echo "[ERROR] no run_dir given and none recorded. Usage: collapse_diagnostics.sh <run_dir>" >&2; exit 2; }
[ -d "$RUN_DIR" ] || { echo "[ERROR] run_dir does not exist: $RUN_DIR" >&2; exit 2; }

banner "collapse diagnostics  |  run=$RUN_DIR"
cd "$REPO"
"$PY" "$HERE/collapse_diagnostics.py" --run-dir "$RUN_DIR" --out "$RUN_DIR/collapse_diagnostics.json"
