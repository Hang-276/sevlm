#!/usr/bin/env bash
# Ablation: task generation.
#
#   no_edit          turn off regret-driven editing; pure sampled generation
#   no_pairs         still edit, but unpaired (drops the counterfactual pairs)
#   no_label_balance turn off label balancing; use the pool's own distribution
#   plain            all three off (closest to the pre-change generator)
set -euo pipefail
WHICH="${1:?usage: generator.sh <no_edit|no_pairs|no_label_balance|plain>}"
case "$WHICH" in
  no_edit)          export EDIT_FRACTION=0 ;;
  no_pairs)         export COUNTERFACTUAL_PAIRS=0 ;;
  no_label_balance) export LABEL_BALANCE=none ;;
  plain)            export EDIT_FRACTION=0 COUNTERFACTUAL_PAIRS=0 LABEL_BALANCE=none ;;
  *) echo "[ERROR] unknown option: $WHICH" >&2; exit 2 ;;
esac
export MARKER="generator_$WHICH"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
