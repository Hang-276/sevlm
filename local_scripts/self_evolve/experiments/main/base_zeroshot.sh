#!/usr/bin/env bash
# =============================================================================
# Base model zero-shot: no training at all; inference straight on the
# Tier-1 benchmarks.
#
# The six Tier-1 general-ability benchmarks (see docs/EVALUATION.md):
#   MMVP  MMStar  BLINK  RealWorldQA  AI2D_TEST  ChartQA_TEST
#
# Usage:
#   bash local_scripts/self_evolve/experiments/main/base_zeroshot.sh
#   DATASETS="MMVP" bash .../base_zeroshot.sh            # single-benchmark smoke run
#   JUDGE=exact_matching bash .../base_zeroshot.sh       # local scoring (no API)
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"

case "$WORKSPACE" in */PATH/TO/*) echo "[ERROR] WORKSPACE in paths.sh is still the placeholder; fill in the real path first" >&2; exit 2;; esac
[ -e "$BASE_MODEL" ] || { echo "[ERROR] BASE_MODEL does not exist: $BASE_MODEL (download Qwen2.5-VL-7B-Instruct first, see paths.sh)" >&2; exit 2; }

banner "eval_base  zero-shot  |  model=$BASE_MODEL"
run_tier1_eval "base=$BASE_MODEL"
