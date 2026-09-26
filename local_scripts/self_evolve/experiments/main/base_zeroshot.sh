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
# Use the isolated evaluator; never load training defaults or activate sevlm.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASETS="${DATASETS:-MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST}"
export LABEL="${LABEL:-base}"
# Experiment 1 always evaluates BASE_MODEL, not a CKPT left in the shell.
unset CKPT
exec bash "$HERE/../analysis/eval_checkpoint.sh"
