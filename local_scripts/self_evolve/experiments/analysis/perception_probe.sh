#!/usr/bin/env bash
# =============================================================================
# Perception-ceiling probe — run this before any other experiment.
#
# Can the base model see these attribute changes at all? Everything about the
# reward depends on the answer. The script prints how to read its output.
# No API needed.
#
# Usage:
#   bash local_scripts/self_evolve/experiments/analysis/perception_probe.sh
#   NUM_TASKS=50 bash .../analysis/perception_probe.sh          # quick smoke run
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

NUM_TASKS="${NUM_TASKS:-200}"
NUM_PLAYERS="${NUM_PLAYERS:-5}"
MIN_PIXELS="${MIN_PIXELS:-200704 802816}"
MAX_PIXELS="${MAX_PIXELS:-4014080}"
OUT_JSON="${OUT_JSON:-$EXP_RUNS_ROOT/perception_probe.json}"

require_paths
mkdir -p "$(dirname "$OUT_JSON")"

banner "probe_perception  perception-ceiling probe
  model=$BASE_MODEL
  tasks=$NUM_TASKS  players=$NUM_PLAYERS
  min_pixels=$MIN_PIXELS  (200704 = 1x)"

cd "$REPO"
"$PY" "$EXP_DIR/analysis/perception_probe.py" \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$BASE_MODEL" \
  --num-tasks "$NUM_TASKS" \
  --num-players "$NUM_PLAYERS" \
  --min-pixels $MIN_PIXELS \
  --max-pixels "$MAX_PIXELS" \
  --max-new-tokens "$SOLVER_MAX_NEW_TOKENS" \
  --out "$OUT_JSON"
