#!/usr/bin/env bash
# =============================================================================
# Smoke config: no API, no GPU, no training. Only verifies the loop mechanism
# runs end to end.
# Usage: bash self_evolve/run_loop.sh self_evolve/configs/smoke.sh
# =============================================================================
EXP_NAME="smoke_dryrun"

NUM_ITERATIONS=2
NUM_TRAIN_TASKS=8
NUM_GENERATIONS=4
SEED=42

REFERENCE_LIVE=0          # local heuristic screening, no API calls
REFERENCE_PROVIDER="openai"
REFERENCE_MODEL="gpt-4o"
REFERENCE_BASE_URL="https://api.openai.com/v1"

SOLVER_LIVE=0             # no model load
TRAINER_LIVE=0            # builds the command, doesn't train

MAX_STEPS=1
NUM_GPUS=1
PER_DEVICE_BATCH=4
GRAD_ACCUM=1
USE_LORA=1
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
DEEPSPEED_CONFIG=""

DISABLE_CARRY_FORWARD=0
KEEP_ALL_MERGED=0
