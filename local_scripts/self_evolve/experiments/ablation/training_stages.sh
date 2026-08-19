#!/usr/bin/env bash
# Ablation: training stages. Only STAGES changes; reward / loop / trainer
# knobs all match main/ours.sh.
#
#   bash .../ablation/training_stages.sh grpo       # drop SFT replay
#   bash .../ablation/training_stages.sh sft        # drop GRPO
#   bash .../ablation/training_stages.sh sft,grpo   # full (= main/ours.sh)
set -euo pipefail
STAGES="${1:?usage: training_stages.sh <sft|grpo|sft,grpo>}"
export STAGES
export MARKER="stages_$(echo "$STAGES" | tr ',' '_')"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
