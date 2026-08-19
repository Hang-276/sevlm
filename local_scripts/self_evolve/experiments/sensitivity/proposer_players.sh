#!/usr/bin/env bash
# Sensitivity: how wide a player range the proposer may choose from.
# 5 5 pins it (subset choice only, 3 options); wider ranges give it a real
# action space.
#   bash .../sensitivity/proposer_players.sh            # default values
#   bash .../sensitivity/proposer_players.sh 5 4 8
#   DRY=1 bash .../sensitivity/proposer_players.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

# One value = the half-width around 5 players; 0 pins the count.
set_axis() {
  export SELF_PLAY=1
  export PROPOSER_MIN_PLAYERS=$((5 - $1))
  export PROPOSER_MAX_PLAYERS=$((5 + $1))
}

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(0 1 3)
run_sweep "proposer_player_span" "${VALUES[@]}"
