#!/usr/bin/env bash
# Sensitivity: rollout length cap
#   bash .../sensitivity/rollout_length.sh            # default values
#   bash .../sensitivity/rollout_length.sh 256 512 1024 2048
#   DRY=1 bash .../sensitivity/rollout_length.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export GRPO_MAX_COMPLETION_LEN=$1 SOLVER_MAX_NEW_TOKENS=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(256 512 1024 2048)
run_sweep "max_completion_len" "${VALUES[@]}"
