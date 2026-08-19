#!/usr/bin/env bash
# Sensitivity: what fraction of each round's tasks comes from editing
#   bash .../sensitivity/edit_fraction.sh            # default values
#   bash .../sensitivity/edit_fraction.sh 0 0.25 0.5 0.75
#   DRY=1 bash .../sensitivity/edit_fraction.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export EDIT_FRACTION=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0 0.25 0.5 0.75)
run_sweep "edit_fraction" "${VALUES[@]}"
