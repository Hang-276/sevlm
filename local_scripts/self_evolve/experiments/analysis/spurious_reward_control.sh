#!/usr/bin/env bash
# Analysis: control exps. Train on a reward that carries no information
# about correctness and see how much the numbers go up.
#
#   bash .../analysis/spurious_reward_control.sh random       # random reward
#   bash .../analysis/spurious_reward_control.sh format_only  # rewards valid format only
#
# Why: if ours gains about as much as these two, the gain can't be credited
# to the reward design. Once your numbers start going up, these exps are
# mandatory.
set -euo pipefail
MODE="${1:?usage: spurious_reward_control.sh <random|format_only>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

case "$MODE" in
  random|format_only) ;;
  *) echo "[ERROR] unknown mode: $MODE" >&2; exit 2 ;;
esac
REWARD_JSON="$(patched_reward_config components.control.mode=\"$MODE\")"
export REWARD_JSON MARKER="control_$MODE" RUN_TAG="${RUN_TAG:-control_$MODE}"
echo "[control] $MODE  config=$REWARD_JSON"
exec bash "$RUNNER"
