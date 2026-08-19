#!/usr/bin/env bash
# Sensitivity: at which step grounding pressure starts
#   bash .../sensitivity/grounding_timing.sh            # default values
#   bash .../sensitivity/grounding_timing.sh 0 80 160 320
#   DRY=1 bash .../sensitivity/grounding_timing.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { REWARD_JSON="$(patched_reward_config components.grounding.warmup_steps=$1)"; export REWARD_JSON; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0 80 160 320)
run_sweep "grounding_warmup" "${VALUES[@]}"
