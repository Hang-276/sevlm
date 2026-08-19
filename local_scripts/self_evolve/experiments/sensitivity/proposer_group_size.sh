#!/usr/bin/env bash
# Sensitivity: proposals sampled per scene (the proposing group size)
#   bash .../sensitivity/proposer_group_size.sh            # default values
#   bash .../sensitivity/proposer_group_size.sh 2 4 8
#   DRY=1 bash .../sensitivity/proposer_group_size.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export SELF_PLAY=1 PROPOSER_GROUP_SIZE=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(2 4 8)
run_sweep "proposer_group_size" "${VALUES[@]}"
