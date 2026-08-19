#!/usr/bin/env bash
# =============================================================================
# Experiment hyperparameter config — for a different experiment, copy this
# file and change a few values. run_loop.sh sources one config file like this
# (this one by default).
#
# Hardware: 8x H200 (143GB each) — batch / group / rank sized accordingly.
# Everything is passed to run_real_input_self_evolve_loop.py as CLI flags;
# precedence: CLI > built-in defaults.
# =============================================================================

# --- Experiment tag (output directory name prefix; pick a recognizable name
#     per experiment) ---
EXP_NAME="grpo_h200"

# --- Loop rounds ---
NUM_ITERATIONS=8            # one round = generate -> screen -> solve -> score -> SFT/GRPO -> policy update
NUM_TRAIN_TASKS=128         # candidate tasks per round (the main diversity lever)
NUM_GENERATIONS=4           # trajectories per accepted task = GRPO group size (must be >=4)
SEED=42

# --- Reference VLM (GPT-4o screening, at task-generation time) ---
REFERENCE_LIVE=1            # 1 = real GPT-4o API calls (needs key); 0 = dry-run local heuristic
REFERENCE_PROVIDER="openai" # openai | openrouter
REFERENCE_MODEL="gpt-4o"
REFERENCE_BASE_URL="${REFERENCE_BASE_URL:-https://api.openai.com/v1}"  # for a compatible relay, put its address

# --- Solver rollouts ---
SOLVER_LIVE=1              # 1 = real GPU rollouts; 0 = dry-run (no model load, mechanism check only)

# --- Trainer (SFT -> GRPO; GRPO-only experiments set STAGES=grpo) ---
TRAINER_LIVE=1            # 1 = real training; 0 = dry-run (builds and prints the command, doesn't launch)
MAX_STEPS=100            # GRPO steps per round
NUM_GPUS=8               # H200 count
PER_DEVICE_BATCH=2       # per-card batch. Constraint: PER_DEVICE_BATCH*NUM_GPUS divisible by NUM_GENERATIONS
GRAD_ACCUM=2
USE_LORA=1               # 1 = LoRA (recommended here); 0 = full-parameter (needs more memory / ZeRO-3)
LORA_R=256               # LoRA rank. Keep LORA_ALPHA = 2*LORA_R when changing
LORA_ALPHA=512
LORA_DROPOUT=0.05
# DeepSpeed config (empty = no deepspeed; usually fine on large-memory H200)
# For multi-card ZeRO, set a json path, e.g. local_scripts/zero2_dual4090_lora.json
DEEPSPEED_CONFIG=""

# --- carry-forward (weight accumulation across rounds) ---
DISABLE_CARRY_FORWARD=0   # 0 = on (round N continues from round N-1's merged weights); 1 = retrain from base each round (ablation)
KEEP_ALL_MERGED=1         # 1 = keep every round's merged model for per-round eval (~16GB/round of disk); 0 = keep only the latest

# --- Stages ---
STAGES="sft,grpo"
