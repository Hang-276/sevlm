#!/usr/bin/env bash
# =============================================================================
# GRPO baseline（base model + 纯 GRPO）单文件全链路训练脚本 —— 自包含版
#
# 这是 ours_full_pipeline_train.sh 的「baseline 版」：同一个 loop 入口，同样的
# GRPO 训练协议（lr / beta / warmup / 长度 / 采样 / advantage 全部逐项对齐 ours），
# 单轮一次性出题 + GRPO 跑到头。只有 reward 和「有没有 self-evolve 闭环」与 ours 不同。
#
# 用法（建议 tmux）：
#   tmux new -s grpo_baseline
#   bash local_scripts/self_evolve/experiments/grpo_baseline_train.sh
#
# 训练完评测：bash local_scripts/self_evolve/experiments/main/eval.sh <run_dir>
# =============================================================================
set -euo pipefail

# =============================================================================
# 1) 路径 + 环境（对齐 ours_full_pipeline_train.sh）。全部可用 env 覆盖。
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$REPO}"

# --- conda 环境 ---
CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV="${CONDA_ENV:-easy-r1}"
WANDB_PKG_DIR="${WANDB_PKG_DIR:-/tmp/sevlm_wandb_py311}"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -n "$CONDA_BASE" ]; then
  echo "[WARN] conda.sh not found at $CONDA_BASE (set CONDA_BASE if conda lives elsewhere)" >&2
fi
PY="${PY:-$(command -v python3 || command -v python || true)}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] || { echo "[ERROR] no python found. Set PY=/path/to/python, or CONDA_BASE/CONDA_ENV." >&2; exit 2; }
  echo "[WARN] requested Python is unavailable; using $PY." >&2
fi

# --- 数据 / 模型 / 输出 / reward ---
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE/data/Vision-Zero-clevr-dataset}"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/models/Qwen2.5-VL-7B-Instruct}"
RUNS_ROOT="${RUNS_ROOT:-$WORKSPACE/self_evolve_runs}"
# Dedicated baseline reward: parse the two answer fields, then return 1 only
# when BOTH match gold.  Main-method / ablation reward JSON files are untouched.
REWARD_JSON="${REWARD_JSON:-$REPO/local_scripts/self_evolve/configs/reward/baselines/grpo_binary_outcome.json}"
# API key 从 .env 或环境变量读取。
ENV_FILE="$REPO/.env"

# --- 自动派生 ---
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# vLLM's colocated sleep-mode rollout uses CuMemAllocator. PyTorch expandable
# segments are incompatible with that memory pool and make every worker abort
# during vLLM initialization, before the first rollout.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
LOOP_ENTRY="$REPO/local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py"
DEEPSPEED_CONFIG="$REPO/local_scripts/zero3.json"
FSDP_CONFIG="$REPO/local_scripts/fsdp2_qwen2_5vl.json"

# =============================================================================
# 2) 超参（
# =============================================================================
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"                          # 单轮 GRPO 总步数（= ours 累计步数）
# 题量：对齐 ours 整个 run 的累计题量（默认 10 轮 × 256 = 2560），单轮一次性出题。
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-2560}"
NUM_PLAYERS="${NUM_PLAYERS:-3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"                 # GRPO group size；须 >=4 且整除 PER_DEVICE_BATCH*NUM_GPUS
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
TRAINER_BACKEND="${TRAINER_BACKEND:-deepspeed}"
# 1 = exercise the full data/export/command wiring without launching torchrun.
DRY_RUN="${DRY_RUN:-0}"

# ---- 训练阶段：baseline 只跑 GRPO（无 SFT）----
STAGES="${STAGES:-grpo}"

# ---- GRPO（逐项对齐 ours；温度已随 ours 统一为 1.0，长度 2048）----
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-$MAX_STEPS}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_WARMUP_RATIO="${GRPO_WARMUP_RATIO:-0.1}"
GRPO_LR_SCHEDULER="${GRPO_LR_SCHEDULER:-cosine}"
GRPO_BETA="${GRPO_BETA:-0.06}"                          # KL 系数
GRPO_WEIGHT_DECAY="${GRPO_WEIGHT_DECAY:-0.0}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"             # rollout 采样温度（对齐改后的 ours）
GRPO_MAX_PROMPT_LEN="${GRPO_MAX_PROMPT_LEN:-10240}"
GRPO_MAX_COMPLETION_LEN="${GRPO_MAX_COMPLETION_LEN:-2048}"  # 对齐 ours（放得下五图 CoT + 尾部 <bbox>）
GRPO_MIN_PIXELS="${GRPO_MIN_PIXELS:-602112}"            # 1024*28*28（4x）
GRPO_MAX_PIXELS="${GRPO_MAX_PIXELS:-1003520}"           # 640*28*28
GRPO_SCALE_REWARDS="${GRPO_SCALE_REWARDS:-False}"       # 减组均值，不除标准差
GRPO_OVERLONG_FILTERING="${GRPO_OVERLONG_FILTERING:-True}"
GRPO_DYNAMIC_SAMPLING="${GRPO_DYNAMIC_SAMPLING:-True}"
GRPO_DYNAMIC_MODE="${GRPO_DYNAMIC_MODE:-mask_degenerate}"
GRPO_DYNAMIC_STD_THRESHOLD="${GRPO_DYNAMIC_STD_THRESHOLD:-0.02}"
GRPO_DYNAMIC_MAX_RETRIES="${GRPO_DYNAMIC_MAX_RETRIES:-3}"

# ---- 离线 solver rollout（打分 / 建 buffer 的那一遍；长度须与 GRPO 一致）----
SOLVER_MAX_NEW_TOKENS="${SOLVER_MAX_NEW_TOKENS:-2048}"
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-1.0}"
SOLVER_TOP_P="${SOLVER_TOP_P:-0.95}"

# ---- GRPO 共用 ----
GC_KWARGS="${GC_KWARGS:---gradient_checkpointing_kwargs '{\"use_reentrant\": false}'}"
CLIP="${CLIP:---max_grad_norm 0.3}"                     # 梯度裁剪，防 bf16 全参 grad_norm 突然 nan
SAVE_STEPS="${SAVE_STEPS:-50}"

# ---- 组装 GRPO 透传串（主入口 append 到命令末尾，覆盖内置默认）----
GRPO_STEPS_ARG=""; [ -n "${GRPO_MAX_STEPS:-}" ] && GRPO_STEPS_ARG="--max_steps $GRPO_MAX_STEPS"
GRPO_EXTRA="$GRPO_STEPS_ARG --learning_rate $GRPO_LR --warmup_ratio $GRPO_WARMUP_RATIO \
--lr_scheduler_type $GRPO_LR_SCHEDULER --beta $GRPO_BETA \
--max_prompt_length $GRPO_MAX_PROMPT_LEN --max_completion_length $GRPO_MAX_COMPLETION_LEN \
--min_pixels $GRPO_MIN_PIXELS --max_pixels $GRPO_MAX_PIXELS \
--weight_decay $GRPO_WEIGHT_DECAY --temperature $GRPO_TEMPERATURE \
--scale_rewards $GRPO_SCALE_REWARDS --overlong_filtering $GRPO_OVERLONG_FILTERING \
--dynamic_sampling $GRPO_DYNAMIC_SAMPLING --dynamic_sampling_mode $GRPO_DYNAMIC_MODE \
--dynamic_sampling_std_threshold $GRPO_DYNAMIC_STD_THRESHOLD \
--dynamic_sampling_max_retries $GRPO_DYNAMIC_MAX_RETRIES \
--save_steps $SAVE_STEPS $CLIP $GC_KWARGS"

# =============================================================================
# 3) 小工具
# =============================================================================
banner() {
  echo "======================================================"
  echo "  $*"
  echo "======================================================"
}
load_env() {
  if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  fi
}
require_paths() {
  case "$WORKSPACE" in */PATH/TO/*) echo "[ERROR] WORKSPACE 还是占位符，先填真实路径" >&2; exit 2;; esac
  [ -e "$DATASET_ROOT" ] || { echo "[ERROR] DATASET_ROOT 不存在: $DATASET_ROOT（先下 CLEVR 数据集）" >&2; exit 2; }
  [ -d "$DATASET_ROOT/output/replacement_images" ] || { echo "[ERROR] $DATASET_ROOT/output/replacement_images 缺失。DATASET_ROOT 要指向 output/ 的父目录，不是 output/ 本身。" >&2; exit 2; }
  [ -e "$BASE_MODEL" ] || { echo "[ERROR] BASE_MODEL 不存在: $BASE_MODEL（先下 Qwen2.5-VL-7B-Instruct）" >&2; exit 2; }
}

require_paths
load_env
[ -f "$REWARD_JSON" ] || { echo "[ERROR] reward 配置不存在: $REWARD_JSON" >&2; exit 2; }

# ---- RUN_TAG（带时间戳，每次起新目录；换名从头跑）----
RUN_TAG="${RUN_TAG:-base_grpo_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$RUNS_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

# =============================================================================
# 4) 环境开关
# =============================================================================
# GRPO 每-step 轨迹+打分落盘（默认开）。关掉：export SELF_EVOLVE_GRPO_DUMP_DIR=（置空）。
export SELF_EVOLVE_GRPO_DUMP_DIR="${SELF_EVOLVE_GRPO_DUMP_DIR:-$OUT_DIR/grpo_dumps}"

# trainer 超参透传
export SELF_EVOLVE_GRPO_EXTRA_ARGS="$GRPO_EXTRA"

# 注意：不设置 SELF_EVOLVE_ANSWER_JUDGE* —— 取代码默认（关闭），reward 走本地
# local binary structured-answer rubric；默认仅 W&B 上报需要联网，由 GPU 机器直接访问。

# --- wandb 训练曲线（默认开；WANDB=0 关闭。key 放 .env 的 WANDB_API_KEY）---
if [ "${WANDB:-1}" = "1" ]; then
  [ -d "$WANDB_PKG_DIR/wandb" ] || {
    echo "[ERROR] W&B package not found: $WANDB_PKG_DIR/wandb" >&2
    exit 2
  }
  # The conda env's bundled wandb is unusable on this machine. Always load the
  # Python 3.11-compatible package staged under /tmp for both login and trainer.
  export PYTHONPATH="$WANDB_PKG_DIR:$PYTHONPATH"
  export SELF_EVOLVE_REPORT_TO=wandb
  export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
  export WANDB_NAME="${WANDB_NAME:-base_grpo_${NUM_TRAIN_TASKS}t_${MAX_STEPS}step}"
  : "${WANDB_API_KEY:?WANDB=1 requires WANDB_API_KEY in the environment}"
  export WANDB_API_KEY
  echo "[wandb] package=$WANDB_PKG_DIR"
  "$PY" -m wandb login --relogin "$WANDB_API_KEY" || {
    echo "[ERROR] W&B login failed; refusing to start an untracked training run." >&2
    exit 2
  }
  echo "[wandb] logged in"
  echo "[wandb] project=$WANDB_PROJECT name=$WANDB_NAME"
else
  export SELF_EVOLVE_REPORT_TO=none
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
fi

# =============================================================================
# 5) 组装 loop 入口参数
# =============================================================================
ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$BASE_MODEL"
  --output-dir "$OUT_DIR"
  --reward-config "$REWARD_JSON"
  --num-iterations "$NUM_ITERATIONS"
  --num-train-tasks "$NUM_TRAIN_TASKS"
  --num-players "$NUM_PLAYERS"
  --num-generations "$NUM_GENERATIONS"
  --seed "$SEED"
  --max-trainer-steps "$MAX_STEPS"
  --trainer-num-gpus "$NUM_GPUS"
  --trainer-per-device-train-batch-size "$PER_DEVICE_BATCH"
  --trainer-gradient-accumulation-steps "$GRAD_ACCUM"
  --trainer-grpo-num-generations "$NUM_GENERATIONS"
)

# --- solver 离线 rollout 采样参数 ---
ARGS+=(
  --solver-max-new-tokens "$SOLVER_MAX_NEW_TOKENS"
  --solver-temperature "$SOLVER_TEMPERATURE"
  --solver-top-p "$SOLVER_TOP_P"
  --solver-min-pixels "$GRPO_MIN_PIXELS"
  --solver-max-pixels "$GRPO_MAX_PIXELS"
)

# --- 关掉 self-evolve 的东西（否则从默认继承，就不是 baseline 了）---
# 1) 无出题进化：edit-fraction=0（全部新采样，不由上一轮 regret 编辑）；不开 self-play。
ARGS+=(--edit-fraction 0)
# 2) 无 Reference-VLM 出题把关：走本地启发式（不联网、无 API）。
ARGS+=(--dry-run-reference-vlm)
# 3) 跳过离线 solver rollout（题生完直接 GRPO）。
#    这一步在 loop 里是为 self-evolve 服务的：用 base model 把每题各 rollout 8 次、
#    打分 -> 算 solvability/regret 喂下一轮出题、建 positive buffer 给 SFT。但 baseline
#    是单轮、无 SFT、无下一轮，这些产物没人消费。而 GRPO 训练读的导出 JSONL 只来自
#    accepted_tasks（题目 problem/solution/图），完全不含离线轨迹——GRPO 每个 step 会用
#    当时最新权重在训练循环内部自己 on-policy rollout num_generations 次。所以这段离线
#    rollout 对 baseline 纯浪费 GPU 时间。--dry-run-solver 让它不加载模型、只生成占位
#    轨迹（瞬间完成），GRPO 的训练数据与结果完全不变。
ARGS+=(--dry-run-solver)

# --- 训练阶段（STAGES -> --execute-*-smoke）。baseline 只有 grpo。---
case ",$STAGES," in *grpo*) ;; *) echo "[ERROR] STAGES=$STAGES 不含 grpo（baseline 必须跑 grpo）" >&2; exit 2;; esac
if [ "$DRY_RUN" = "1" ]; then
  ARGS+=(--dry-run-trainer)
else
  case ",$STAGES," in *,sft,*)  ARGS+=(--execute-sft-smoke);; esac
  case ",$STAGES," in *,grpo,*) ARGS+=(--execute-grpo-smoke);; esac
fi

# --- 后端 ---
ARGS+=(--trainer-backend "$TRAINER_BACKEND")
[ -n "$DEEPSPEED_CONFIG" ] && ARGS+=(--trainer-deepspeed-config "$DEEPSPEED_CONFIG")
if [ "$TRAINER_BACKEND" = "fsdp2" ]; then
  [ -f "$FSDP_CONFIG" ] || { echo "[ERROR] FSDP2 配置不存在: $FSDP_CONFIG" >&2; exit 2; }
  ARGS+=(--trainer-fsdp-config "$FSDP_CONFIG")
fi

banner "train  GRPO baseline（纯单轮 GRPO，无 self-evolve；出题 -> 本地筛选 -> [跳过离线 rollout] -> GRPO 训练内部 on-policy rollout+outcome reward）
  RUN_TAG=$RUN_TAG
  OUT_DIR=$OUT_DIR
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  dry_run=$DRY_RUN
  iters=$NUM_ITERATIONS tasks=$NUM_TRAIN_TASKS gen=$NUM_GENERATIONS steps=$MAX_STEPS gpus=$NUM_GPUS mode=FULL-PARAM(GRPO=$([ "$TRAINER_BACKEND" = fsdp2 ] && echo "fsdp2=$(basename "$FSDP_CONFIG")" || echo "deepspeed=$(basename "$DEEPSPEED_CONFIG")"))
  stages=$STAGES  reward=$REWARD_JSON(binary structured outcome, 本地 rubric, 不联网)"

cd "$REPO"
"$PY" "$LOOP_ENTRY" "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"

banner "训练完成。评测：bash local_scripts/self_evolve/experiments/main/eval.sh $OUT_DIR"
