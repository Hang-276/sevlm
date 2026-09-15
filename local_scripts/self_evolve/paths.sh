#!/usr/bin/env bash
# =============================================================================
# Global paths + environment (the only place you fill in real paths)
# Every run script sources this file. Each placeholder below says where to
# download the thing it points at.
# =============================================================================

# --- Workspace root: data, models, outputs and evaluation all hang off it.
#     Moving machines = changing this one line. ---
# Can also be overridden from outside via export WORKSPACE=...
WORKSPACE="${WORKSPACE:-/jizhicfs/rtliu}"

# --- Repo root. Derived from this file's location;
#     rarely needs changing. ---
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# --- Conda environment (this machine uses the sevlm environment) ---
# Activate it directly so torchrun/python subprocesses inside training also
# use this env (setting PY alone isn't enough).
# CONDA_ENV must match the env you actually created — the repo's root setup.sh
# suggests the name vlm-r1, so set CONDA_ENV=vlm-r1 (or export PY) if that's
# what you used.
CONDA_BASE="${CONDA_BASE:-/jizhicfs/rtliu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-sevlm}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "[paths.sh][WARN] conda.sh not found at $CONDA_BASE (set CONDA_BASE if conda lives elsewhere)" >&2
fi
# --- Python interpreter ---
# The conda env's python works without activation. If that env isn't there,
# fall back to whatever python is on PATH, and say so — otherwise every script
# dies much later with a bare "No such file or directory".
PY="${PY:-$CONDA_BASE/envs/$CONDA_ENV/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] || { echo "[paths.sh][ERROR] no python found. Set PY=/path/to/python, or CONDA_BASE/CONDA_ENV." >&2; return 2 2>/dev/null || exit 2; }
  echo "[paths.sh][WARN] conda env '$CONDA_ENV' not found; using $PY. Set PY=... to pick another." >&2
fi

# --- CLEVR dataset root ---
# TODO placeholder: download the Vision-Zero CLEVR dataset; after unpacking,
# the directory should contain output/CLEVR_scenes.json.
# Source (HF): https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset
# Point this at the top-level directory (the parent of output/, not output/).
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE/data/Vision-Zero-clevr-dataset}"

# --- Base solver model ---
# TODO placeholder: Qwen2.5-VL-7B-Instruct
# Source (HF): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/models/Qwen2.5-VL-7B-Instruct}"

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
# Put it in .env (gitignored), or export it here. A shell export always wins.
# export OPENAI_API_KEY="sk-..."
ENV_FILE="$REPO/.env"

# =============================================================================
# Everything below is set automatically — no need to change.
# =============================================================================
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# H200 uses sdpa (no flash-attn); enable allocator defragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOOP_ENTRY="$REPO/local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py"
