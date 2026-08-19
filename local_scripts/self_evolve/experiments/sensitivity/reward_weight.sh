#!/usr/bin/env bash
# Sensitivity: answer vs grounding weight split (other dims scale along)
#   bash .../sensitivity/reward_weight.sh            # default values
#   bash .../sensitivity/reward_weight.sh 0.45 0.60 0.75
#   DRY=1 bash .../sensitivity/reward_weight.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { REWARD_JSON="$(patched_reward_config weights.answer=$1 weights.grounding=$(python -c "print(round(0.85-$1,4))"))"; export REWARD_JSON; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0.45 0.60 0.75)
run_sweep "answer_weight" "${VALUES[@]}"
