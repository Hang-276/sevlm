#!/usr/bin/env bash
# Main experiment: base + ours (w/o self-play).
# The model still solves, but a regret heuristic picks the edits instead of the
# model proposing them. Everything else is the full method — this is the exp
# that isolates what the proposing side is worth.
set -euo pipefail
export SELF_PLAY=0
export MARKER=ours_no_self_play
export RUN_TAG="${RUN_TAG:-ours_no_self_play}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
