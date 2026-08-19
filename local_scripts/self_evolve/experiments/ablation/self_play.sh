#!/usr/bin/env bash
# Ablation: how the proposing side proposes.
#
#   no_feedback  model still proposes, but is never told how the solver is doing
#   greedy       proposals sampled greedily, so the group collapses to one pick
#
# Turning proposing off entirely is a main-table row, not an ablation:
# main/ours_no_self_play.sh.
set -euo pipefail
WHICH="${1:?usage: self_play.sh <no_feedback|greedy>}"
case "$WHICH" in
  no_feedback) export SELF_PLAY=1 PROPOSER_NO_COMPETENCE=1 ;;
  greedy)      export SELF_PLAY=1 PROPOSER_TEMPERATURE=0.01 ;;
  *) echo "[ERROR] unknown option: $WHICH" >&2; exit 2 ;;
esac
export MARKER="self_play_$WHICH"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
