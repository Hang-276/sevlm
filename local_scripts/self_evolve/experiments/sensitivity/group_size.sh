#!/usr/bin/env bash
# Sensitivity: GRPO group size
#   bash .../sensitivity/group_size.sh            # default values
#   bash .../sensitivity/group_size.sh 4 8 16
#   DRY=1 bash .../sensitivity/group_size.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export NUM_GENERATIONS=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(4 8 16)
run_sweep "num_generations" "${VALUES[@]}"
