#!/usr/bin/env bash
# Main experiment: base + ours (w/o process-level reward).
# Only the reward changes to outcome-only (answer=1.0, all else 0); the loop
# and training config stay the same.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
export REWARD_JSON="$LIB_DIR/reward_outcome_only.json"
export MARKER=ours_no_process
export RUN_TAG="${RUN_TAG:-ours_no_process}"
exec bash "$RUNNER"
