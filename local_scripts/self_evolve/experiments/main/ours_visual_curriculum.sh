#!/usr/bin/env bash
# Visual-facts extension: paired QA, evidence-aware mastery, fixed loss divisor.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MARKER="${MARKER:-ours_visual_curriculum}"
export RUN_TAG="${RUN_TAG:-ours_visual_curriculum}"
export SELF_EVOLVE_MASTERY_MODE="${SELF_EVOLVE_MASTERY_MODE:-verified_visual}"
# Replace at most 1/8 of training rows; preserve game pairs and total row count.
export SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION="${SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION:-0.125}"
export GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
# Fixed gradient scale, independent of the 2048-token generation cap.
if [ "$GRPO_LOSS_TYPE" = "dr_grpo" ]; then
  export GRPO_LOSS_NORMALIZATION_LENGTH="${GRPO_LOSS_NORMALIZATION_LENGTH-512}"
fi
exec bash "$SCRIPT_DIR/ours_visual_facts.sh"
