#!/usr/bin/env bash
# Ablation: rollout length back to the pre-change 256 tokens.
# Quantifies how much "the format block at the end gets truncated" contributes
# on its own.
set -euo pipefail
export GRPO_MAX_COMPLETION_LEN="${1:-256}"
export SOLVER_MAX_NEW_TOKENS="$GRPO_MAX_COMPLETION_LEN"
export MARKER="rollout_len_$GRPO_MAX_COMPLETION_LEN"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
