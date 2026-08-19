#!/usr/bin/env bash
# Sensitivity: how learnable a puzzle must have been for the proposer to be
# trained on it. 0 trains on every pick, 1 only on a perfect coin flip.
#   bash .../sensitivity/learnability_threshold.sh            # default values
#   bash .../sensitivity/learnability_threshold.sh 0 0.25 0.5 0.75
#   DRY=1 bash .../sensitivity/learnability_threshold.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export SELF_PLAY=1 PROPOSER_LEARNABILITY_THRESHOLD=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0 0.25 0.5 0.75)
run_sweep "learnability_threshold" "${VALUES[@]}"
