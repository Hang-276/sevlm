#!/usr/bin/env bash
# Sensitivity: input resolution (200704 = 1x)
#   bash .../sensitivity/image_resolution.sh            # default values
#   bash .../sensitivity/image_resolution.sh 200704 401408 802816
#   DRY=1 bash .../sensitivity/image_resolution.sh      # just print the plan
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
source "$HERE/_sweep.sh"
SWEEP_EVAL="$EXP_DIR/main/eval.sh"

set_axis() { export GRPO_MIN_PIXELS=$1; }

VALUES=("${@:-}")
[ -n "${VALUES[0]:-}" ] || VALUES=(200704 401408 802816)
run_sweep "min_pixels" "${VALUES[@]}"
