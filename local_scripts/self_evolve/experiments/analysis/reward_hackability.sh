#!/usr/bin/env bash
# Analysis: how much reward a policy that never looks at the image collects
# under the current config. Run after every reward change; this number has
# to stay low.
#
#   bash .../analysis/reward_hackability.sh              # default config
#   bash .../analysis/reward_hackability.sh <config.json>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

CFG="${1:-$SE_DIR/configs/reward/reward_weights.json}"
banner "reward hackability  |  config=$(basename "$CFG")"
cd "$REPO"
"$PY" "$SE_DIR/reward_alignment/audit_reward_hackability.py" \
  --reward-config "$CFG" --max-blind-reward "${MAX_BLIND_REWARD:-0.05}"
