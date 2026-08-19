#!/usr/bin/env bash
# Sensitivity: rollout sampling temperature
#   bash .../sensitivity/rollout_temperature.sh            # default values
#   bash .../sensitivity/rollout_temperature.sh 0.6 0.9 1.2
#   DRY=1 bash .../sensitivity/rollout_temperature.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export GRPO_TEMPERATURE=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0.6 0.9 1.2)
run_sweep "grpo_temperature" "${VALUES[@]}"
