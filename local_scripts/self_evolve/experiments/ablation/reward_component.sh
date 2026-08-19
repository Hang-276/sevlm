#!/usr/bin/env bash
# Ablation: one reward component at a time. Each option reverts exactly one
# piece to the old scoring; everything else stays at the defaults.
#
#   gating            drop the outcome gate (auxiliary dims pay unconditionally)
#   answer_fields     answer back to whole-string exact match
#   box_f1            grounding back to recall (no precision; box spam unpunished)
#   format_floor      grounding gets its 0.1+0.1 unconditional floor back
#   field_consistency consistency back to the group-modal share
#   old_reward        the whole old scoring at once (whole-string match + no
#                     gate + recall + format floor + group modal)
set -euo pipefail
WHICH="${1:?usage: reward_component.sh <gating|answer_fields|box_f1|format_floor|field_consistency|old_reward>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

case "$WHICH" in
  gating)            OVERRIDES=(components.gating.mode=\"none\") ;;
  answer_fields)     OVERRIDES=(components.answer.mode=\"exact_match\") ;;
  box_f1)            OVERRIDES=(components.grounding.match_mode=\"recall\"
                                components.grounding.iou_threshold=0.0) ;;
  format_floor)      OVERRIDES=(components.grounding.format_credit=0.1
                                components.grounding.player_id_credit=0.1
                                components.grounding.iou_credit=0.8) ;;
  field_consistency) OVERRIDES=(components.consistency.mode=\"group_modal\") ;;
  old_reward)        OVERRIDES=(components.answer.mode=\"exact_match\"
                                components.gating.mode=\"none\"
                                components.grounding.format_credit=0.1
                                components.grounding.player_id_credit=0.1
                                components.grounding.iou_credit=0.8
                                components.grounding.match_mode=\"recall\"
                                components.grounding.iou_threshold=0.0
                                components.consistency.mode=\"group_modal\"
                                weights.answer=0.5 weights.grounding=0.25
                                weights.process=0.15 weights.consistency=0.07
                                weights.budget=0.03) ;;
  *) echo "[ERROR] unknown component: $WHICH" >&2; exit 2 ;;
esac

REWARD_JSON="$(patched_reward_config "${OVERRIDES[@]}")"
export REWARD_JSON
echo "[ablation] reward=$WHICH  config=$REWARD_JSON"
export MARKER="reward_wo_$WHICH"
export RUN_TAG="${RUN_TAG:-$MARKER}"
exec bash "$RUNNER"
