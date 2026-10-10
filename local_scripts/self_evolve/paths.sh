#!/usr/bin/env bash
# =============================================================================
# Global paths + environment (the only place you fill in real paths)
# Every run script sources this file. Each placeholder below says where to
# download the thing it points at.
# =============================================================================

# --- Repo and workspace roots; environment overrides take precedence. ---
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$REPO}"

# --- Training env on this machine. Set CONDA_BASE= (empty) to keep whatever
# environment is already active instead. ---
CONDA_BASE="${CONDA_BASE:-/jizhicfs/rtliu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-sevlm}"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -n "$CONDA_BASE" ]; then
  echo "[paths.sh][WARN] conda.sh not found at $CONDA_BASE (set CONDA_BASE if conda lives elsewhere)" >&2
fi
# --- Python interpreter ---
PY="${PY:-$(command -v python3 || command -v python || true)}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] || { echo "[paths.sh][ERROR] no python found. Set PY=/path/to/python, or CONDA_BASE/CONDA_ENV." >&2; return 2 2>/dev/null || exit 2; }
  echo "[paths.sh][WARN] requested Python is unavailable; using $PY. Set PY=... to pick another." >&2
fi

# --- CLEVR dataset root ---
# Download the CLEVR dataset; after unpacking,
# the directory should contain output/CLEVR_scenes.json.
# Source (HF): https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset
# Point this at the top-level directory (the parent of output/, not output/).
DATASET_ROOT="${DATASET_ROOT:-/jizhicfs/rtliu/data/Vision-Zero-clevr-dataset}"

# --- Base solver model ---
# TODO placeholder: Qwen2.5-VL-7B-Instruct
# Source (HF): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
BASE_MODEL="${BASE_MODEL:-/jizhicfs/rtliu/models/Qwen2.5-VL-7B-Instruct}"

# --- Output root (each run creates a <RUN_TAG>/ subdirectory under it) ---
RUNS_ROOT="${RUNS_ROOT:-$WORKSPACE/runs}"

# --- Reward weight config (single source; rarely touched) ---
# Design notes: docs/REWARD.md
REWARD_CONFIG="${REWARD_CONFIG:-$REPO/local_scripts/self_evolve/configs/reward/reward_weights.json}"

# --- Evaluation (VLMEvalKit) ---
# Framework: https://github.com/open-compass/VLMEvalKit
# Benchmark tsvs go in LMUData, see eval/fetch_vlmeval_tsv.sh
VLMK="${VLMK:-$WORKSPACE/eval/VLMEvalKit}"
LMUDATA="${LMUDATA:-$WORKSPACE/eval/LMUData}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-$WORKSPACE/eval/results}"

# --- Reference VLM API (needed for live mode only; training itself doesn't) ---
# Official endpoint is https://api.openai.com/v1; for a compatible relay,
# put the relay's address here.
REFERENCE_BASE_URL="${REFERENCE_BASE_URL:-https://api.openai.com/v1}"

# --- Reference VLM API key (GPT-4o screening, live mode only) ---
# Read OPENAI_API_KEY from .env or the environment.
ENV_FILE="$REPO/.env"

# =============================================================================
# Everything below is set automatically — no need to change.
# =============================================================================
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Enable allocator defragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOOP_ENTRY="$REPO/local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py"
