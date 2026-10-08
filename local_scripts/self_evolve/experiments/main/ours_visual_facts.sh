#!/usr/bin/env bash
# Experimental extension of the full self-evolve loop: verifiable CLEVR
# attribute-change certificates plus a small oracle-format SFT warm start.
# Inherits the main experiment's rounds, GRPO budget and backbone settings.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MARKER="${MARKER:-ours_visual_facts}"
export RUN_TAG="${RUN_TAG:-ours_visual_facts}"
export SELF_EVOLVE_VISUAL_FACTS="${SELF_EVOLVE_VISUAL_FACTS:-1}"
export SELF_EVOLVE_ORACLE_SFT_MAX="${SELF_EVOLVE_ORACLE_SFT_MAX:-64}"
export SELF_EVOLVE_SCENE_QA_MAX="${SELF_EVOLVE_SCENE_QA_MAX:-64}"
# In later rounds, let real grounded solver positives displace synthetic SFT.
# A zero-positive round still gets the full cold-start pools.
export SELF_EVOLVE_AUX_SFT_MAX_RATIO="${SELF_EVOLVE_AUX_SFT_MAX_RATIO-1.0}"
# Without solved editing seeds, reserve a quarter of candidate slots for
# matched one-object versus two-object variants.
export SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION="${SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION:-0.25}"
# The base SFT processor would see the 320x240 source image at ~98 merged
# visual tokens, while this experiment's GRPO sees >=1024. Use a conservative
# 512-768-token SFT range and a smaller per-device batch for multi-image SFT.
# Both can be overridden when hardware headroom has been measured.
export SFT_MIN_PIXELS="${SFT_MIN_PIXELS:-401408}"
export SFT_MAX_PIXELS="${SFT_MAX_PIXELS:-602112}"
export SFT_PER_DEVICE_BATCH="${SFT_PER_DEVICE_BATCH:-2}"
# Keep offline selection aligned with the deterministic answer reward used
# inside GRPO; the optional external answer judge can be re-enabled explicitly.
export SELF_EVOLVE_ANSWER_JUDGE="${SELF_EVOLVE_ANSWER_JUDGE:-0}"
export REWARD_JSON="${REWARD_JSON:-$SCRIPT_DIR/../../configs/reward/reward_visual_facts.json}"
exec bash "$SCRIPT_DIR/../lib/run_self_evolve.sh"
