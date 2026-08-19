#!/usr/bin/env bash
# Ablation: the three GRPO-side changes, reverted one at a time.
#
#   scale_rewards     advantages divided by std again (original GRPO)
#   no_overlong       truncated rollouts train as usual
#   batch_retry       resample the whole batch on a degenerate group instead
#                     of just zeroing its advantages
#   old_grpo          all three reverted at once
set -euo pipefail
WHICH="${1:?usage: grpo_algorithm.sh <scale_rewards|no_overlong|batch_retry|old_grpo>}"
case "$WHICH" in
  scale_rewards) export GRPO_SCALE_REWARDS=True ;;
  no_overlong)   export GRPO_OVERLONG_FILTERING=False ;;
  batch_retry)   export GRPO_DYNAMIC_MODE=batch_retry GRPO_DYNAMIC_STD_THRESHOLD=1e-6 ;;
  old_grpo)      export GRPO_SCALE_REWARDS=True GRPO_OVERLONG_FILTERING=False \
                        GRPO_DYNAMIC_MODE=batch_retry GRPO_DYNAMIC_STD_THRESHOLD=1e-6 ;;
  *) echo "[ERROR] unknown option: $WHICH" >&2; exit 2 ;;
esac
export MARKER="grpo_$WHICH"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
