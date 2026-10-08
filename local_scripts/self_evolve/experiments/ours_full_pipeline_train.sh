#!/usr/bin/env bash
# =============================================================================
# ours（完整方法）单文件全链路训练脚本 —— 自包含版
#
# 这是新库 experiments/main/ours.sh 的「摊平」版本：把原本拆在
#   main/ours.sh  ->  lib/run_self_evolve.sh  ->  lib/common.sh
#                                              ->  paths.sh
#                                              ->  configs/train_defaults.sh
# 用法（建议 tmux）：
#   tmux new -s ours
#   bash local_scripts/self_evolve/experiments/ours_full_pipeline_train.sh
#
# 训练完评测：bash local_scripts/self_evolve/experiments/main/eval.sh <run_dir>
# =============================================================================
set -euo pipefail

# =============================================================================
# 1) 路径 + 环境（原 paths.sh）。全部可用 env 覆盖。
# =============================================================================
# REPO = sevlm 仓库根（从本脚本位置推导：experiments/ 向上三级）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$REPO}"

# --- 已有 conda 环境 ---
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
# --- 使用 PY 或当前环境的 Python ---
PY="${PY:-$(command -v python3 || command -v python || true)}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] || { echo "[ERROR] no python found. Set PY=/path/to/python, or CONDA_BASE/CONDA_ENV." >&2; exit 2; }
  echo "[WARN] requested Python is unavailable; using $PY." >&2
fi

DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE/data/Vision-Zero-clevr-dataset}"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/models/Qwen2.5-VL-7B-Instruct}"
RUNS_ROOT="${RUNS_ROOT:-$WORKSPACE/self_evolve_runs}"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOOP_ENTRY="$REPO/local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py"
DEEPSPEED_CONFIG="$REPO/local_scripts/zero3.json"
FSDP_CONFIG="$REPO/local_scripts/fsdp2_qwen2_5vl.json"

# =============================================================================
# 2) API / 外部服务
# =============================================================================

# ---- Reference VLM：出题把关，要读图（判题目可解 / 歧义 / 难度）----
REFERENCE_VLM_BASE_URL="${REFERENCE_VLM_BASE_URL:-https://api.deepseek.com}"
REFERENCE_VLM_API_KEY="${REFERENCE_VLM_API_KEY:-}"
REFERENCE_VLM_MODEL="${REFERENCE_VLM_MODEL:-deepseek-flash}"
REFERENCE_PROVIDER="${REFERENCE_PROVIDER:-openai}"

# ---- Answer judge：GRPO 打分时兜 exact-match 的语义等价判断，纯文本、无图 ----
# OpenAI 兼容接口需包含 /v1。
ANSWER_JUDGE_BASE_URL="${ANSWER_JUDGE_BASE_URL:-https://api.openai.com/v1}"
ANSWER_JUDGE_API_KEY="${ANSWER_JUDGE_API_KEY:-}"
ANSWER_JUDGE_MODEL="${ANSWER_JUDGE_MODEL:-gpt-4o-mini}"

# ---- wandb ----
WANDB_API_KEY="${WANDB_API_KEY:-}"

# 这些要给 loop / trainer 子进程读，必须 export。
export REFERENCE_VLM_BASE_URL REFERENCE_VLM_API_KEY REFERENCE_VLM_MODEL \
       REFERENCE_PROVIDER ANSWER_JUDGE_BASE_URL ANSWER_JUDGE_API_KEY \
       ANSWER_JUDGE_MODEL WANDB_API_KEY

# =============================================================================
# 3) 超参（原 train_defaults.sh）。每项都可用 env 覆盖。
# =============================================================================
LOOP_ITERATIONS="${LOOP_ITERATIONS:-1}"                # self-evolve 轮数（出题-筛选-解题-打分-训练）
GRPO_STEPS_PER_ITER="${GRPO_STEPS_PER_ITER:-240}"       # 每轮 GRPO 步数
MAIN_NUM_GENERATIONS="${MAIN_NUM_GENERATIONS:-8}"      # GRPO group size / solver rollout n
MAIN_NUM_TRAIN_TASKS="${MAIN_NUM_TRAIN_TASKS:-256}"     # 每轮筛选前生成的候选题目数
# 每题给几张图。对齐 baseline 的 NUM_PLAYERS=3
NUM_PLAYERS="${NUM_PLAYERS:-3}"

NUM_ITERATIONS="${NUM_ITERATIONS:-$LOOP_ITERATIONS}"
NUM_TRAIN_TASKS="${NUM_TRAIN_TASKS:-$MAIN_NUM_TRAIN_TASKS}"
NUM_GENERATIONS="${NUM_GENERATIONS:-$MAIN_NUM_GENERATIONS}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-60}"                            # 兜底/默认步数（未单独指定的阶段用它；也用于 banner/wandb 命名）
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

TRAINER_BACKEND="${TRAINER_BACKEND:-deepspeed}"

# ---- attention 实现（两个 trainer 都透传 --attn_implementation）----
# 需安装 flash-attn；可通过 ATTN_IMPL=sdpa 切换。
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# ← 完整方法用默认 reward（五维加权）。
REWARD_JSON="${REWARD_JSON:-$REPO/local_scripts/self_evolve/configs/reward/reward_weights.json}"
# answer-judge 每题采样多少条 rollout 送判；必须 = NUM_GENERATIONS 才能判全每条。
ANSWER_JUDGE_SAMPLE_N="${ANSWER_JUDGE_SAMPLE_N:-$NUM_GENERATIONS}"

# ---- 训练阶段（sft -> grpo 顺序执行；GRPO 从 SFT 的 checkpoint 起训）----
# 这是 ablation 唯一该改的东西之一（另一个是 REWARD_JSON）。**已无 DPO。**
STAGES="${STAGES:-grpo}"  # sft，dpo

# ---- GRPO ----
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-$GRPO_STEPS_PER_ITER}"  # GRPO 单独步数
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_WARMUP_RATIO="${GRPO_WARMUP_RATIO:-0.1}"
GRPO_LR_SCHEDULER="${GRPO_LR_SCHEDULER:-cosine}"
GRPO_BETA="${GRPO_BETA:-0.06}"                           # KL 系数
GRPO_WEIGHT_DECAY="${GRPO_WEIGHT_DECAY:-0.0}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"              # rollout 采样温度
GRPO_MAX_PROMPT_LEN="${GRPO_MAX_PROMPT_LEN:-10240}"      # 要放得下 num_players 张图 + prompt 文本
GRPO_MAX_COMPLETION_LEN="${GRPO_MAX_COMPLETION_LEN:-2048}"  # 库默认 256 放不下五图对比 CoT + 尾部 <bbox>
# 分辨率对齐 baseline 的 GRPO_MIN/MAX_PIXELS
GRPO_MIN_PIXELS="${GRPO_MIN_PIXELS:-602112}"            # 768*28*28
GRPO_MAX_PIXELS="${GRPO_MAX_PIXELS:-1003520}"           # 1280*28*28（CLEVR 物体在此分辨率仍可辨）
GRPO_SCALE_REWARDS="${GRPO_SCALE_REWARDS:-False}"        # 减组均值，不除标准差
GRPO_OVERLONG_FILTERING="${GRPO_OVERLONG_FILTERING:-True}"  # 被 max_completion_length 截断（无 EOS）的 rollout 给 0 advantage
GRPO_DYNAMIC_SAMPLING="${GRPO_DYNAMIC_SAMPLING:-True}"
GRPO_DYNAMIC_MODE="${GRPO_DYNAMIC_MODE:-mask_degenerate}"   # 只把退化组 advantage 置 0，保留健康组
GRPO_DYNAMIC_STD_THRESHOLD="${GRPO_DYNAMIC_STD_THRESHOLD:-0.02}"
GRPO_DYNAMIC_MAX_RETRIES="${GRPO_DYNAMIC_MAX_RETRIES:-3}"

# ---- SFT（replay 数据监督微调）----
SFT_MAX_STEPS="${SFT_MAX_STEPS:-2}"                      # SFT 单独步数
SFT_LR="${SFT_LR:-1e-6}"
SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.0}"
SFT_LR_SCHEDULER="${SFT_LR_SCHEDULER:-cosine}"
SFT_PER_DEVICE_BATCH="${SFT_PER_DEVICE_BATCH:-8}"
SFT_WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-0.0}"

# ---- 出题（task generation）----
EDIT_FRACTION="${EDIT_FRACTION:-0.5}"                   # 每轮由「上一轮最高 regret 场景」编辑生成的比例；其余新采样
COUNTERFACTUAL_PAIRS="${COUNTERFACTUAL_PAIRS:-1}"       # 编辑变体成对出现（差一个物体），测反事实敏感度
LABEL_BALANCE="${LABEL_BALANCE:-uniform}"               # uniform=生成时拉平 gold changed_attributes 分布

# ---- Self-play（模型自己挑改动->解自己挑的->只训 solver 觉得可学的）----
SELF_PLAY="${SELF_PLAY:-1}"
PROPOSER_GROUP_SIZE="${PROPOSER_GROUP_SIZE:-4}"         # 每场景提案数
PROPOSER_MAX_SCENES="${PROPOSER_MAX_SCENES:-32}"         # 每轮提案的场景数，256题目，edit budget是128，因此这里要填入32合理
PROPOSER_TEMPERATURE="${PROPOSER_TEMPERATURE:-1.0}"
PROPOSER_MAX_NEW_TOKENS="${PROPOSER_MAX_NEW_TOKENS:-512}"
PROPOSER_LEARNABILITY_THRESHOLD="${PROPOSER_LEARNABILITY_THRESHOLD:-0.5}"  # 提案要被训练需达到的最低可学性
PROPOSER_NO_COMPETENCE="${PROPOSER_NO_COMPETENCE:-0}"   # 1=对提案者隐藏 solver 当前解题率（ablation）
PROPOSER_MIN_PLAYERS="${PROPOSER_MIN_PLAYERS:-3}"
PROPOSER_MAX_PLAYERS="${PROPOSER_MAX_PLAYERS:-8}"

# ---- 离线 solver rollout（打分 / 建 buffer 的那一遍）----
SOLVER_MAX_NEW_TOKENS="${SOLVER_MAX_NEW_TOKENS:-2048}"  # 长度须与 GRPO 一致
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-1.0}"         # 温度太低 group 会几乎相同、advantage 全 0
SOLVER_TOP_P="${SOLVER_TOP_P:-0.95}"

# ---- 三阶段共用 ----
# non-reentrant checkpointing：8 卡 DDP 下 reentrant 会把同一参数 mark ready 两次。
GC_KWARGS="${GC_KWARGS:---gradient_checkpointing_kwargs '{\"use_reentrant\": false}'}"
CLIP="${CLIP:---max_grad_norm 0.3}"                     # 梯度裁剪，防 bf16 全参 grad_norm 突然 nan 打崩权重
SAVE_STEPS="${SAVE_STEPS:-40}"

# ---- 组装各阶段透传串（主入口把这些 append 到各自命令末尾，覆盖内置默认）----
GRPO_STEPS_ARG=""; [ -n "${GRPO_MAX_STEPS:-}" ] && GRPO_STEPS_ARG="--max_steps $GRPO_MAX_STEPS"
GRPO_EXTRA="$GRPO_STEPS_ARG --learning_rate $GRPO_LR --warmup_ratio $GRPO_WARMUP_RATIO \
--lr_scheduler_type $GRPO_LR_SCHEDULER --beta $GRPO_BETA \
--max_prompt_length $GRPO_MAX_PROMPT_LEN --max_completion_length $GRPO_MAX_COMPLETION_LEN \
--min_pixels $GRPO_MIN_PIXELS --max_pixels $GRPO_MAX_PIXELS \
--attn_implementation $ATTN_IMPL \
--weight_decay $GRPO_WEIGHT_DECAY --temperature $GRPO_TEMPERATURE \
--scale_rewards $GRPO_SCALE_REWARDS --overlong_filtering $GRPO_OVERLONG_FILTERING \
--dynamic_sampling $GRPO_DYNAMIC_SAMPLING --dynamic_sampling_mode $GRPO_DYNAMIC_MODE \
--dynamic_sampling_std_threshold $GRPO_DYNAMIC_STD_THRESHOLD \
--dynamic_sampling_max_retries $GRPO_DYNAMIC_MAX_RETRIES \
--save_steps $SAVE_STEPS $CLIP $GC_KWARGS"

SFT_STEPS_ARG=""; [ -n "${SFT_MAX_STEPS:-}" ] && SFT_STEPS_ARG="--max_steps $SFT_MAX_STEPS"
SFT_EXTRA="$SFT_STEPS_ARG --learning_rate $SFT_LR --warmup_ratio $SFT_WARMUP_RATIO \
--lr_scheduler_type $SFT_LR_SCHEDULER --per_device_train_batch_size $SFT_PER_DEVICE_BATCH \
--weight_decay $SFT_WEIGHT_DECAY --save_steps 2 $CLIP $GC_KWARGS \
--attn_implementation $ATTN_IMPL"

# =============================================================================
# 4) 小工具（原 common.sh 里用到的三个函数）
# =============================================================================
banner() {
  echo "======================================================"
  echo "  $*"
  echo "======================================================"
}
# 前置存在性检查：占位路径在这里就报错，而不是训练跑一半才崩。
require_paths() {
  case "$WORKSPACE" in */PATH/TO/*) echo "[ERROR] WORKSPACE 还是占位符，先填真实路径" >&2; exit 2;; esac
  [ -e "$DATASET_ROOT" ] || { echo "[ERROR] DATASET_ROOT 不存在: $DATASET_ROOT（先下 CLEVR 数据集）" >&2; exit 2; }
  [ -d "$DATASET_ROOT/output/replacement_images" ] || { echo "[ERROR] $DATASET_ROOT/output/replacement_images 缺失。DATASET_ROOT 要指向 output/ 的父目录，不是 output/ 本身。" >&2; exit 2; }
  [ -e "$BASE_MODEL" ] || { echo "[ERROR] BASE_MODEL 不存在: $BASE_MODEL（先下 Qwen2.5-VL-7B-Instruct）" >&2; exit 2; }
}

# 凭据检查（第 2 节声明）。空值在这里就报错，而不是等 Reference VLM 筛题时才崩。
require_api_keys() {
  [ -n "$REFERENCE_VLM_API_KEY" ] || {
    echo "[ERROR] REFERENCE_VLM_API_KEY 为空（Reference VLM 出题把关要用）。填在第 2 节，或跑之前 export。" >&2
    exit 2
  }
  [ -n "$ANSWER_JUDGE_API_KEY" ] || {
    echo "[ERROR] ANSWER_JUDGE_API_KEY 为空（GRPO 打分的 answer judge 要用）。填在第 2 节，或跑之前 export。" >&2
    exit 2
  }
  [ -n "$REFERENCE_VLM_BASE_URL" ] || { echo "[ERROR] REFERENCE_VLM_BASE_URL 为空" >&2; exit 2; }
  [ -n "$ANSWER_JUDGE_BASE_URL" ]  || { echo "[ERROR] ANSWER_JUDGE_BASE_URL 为空" >&2; exit 2; }
}

require_paths
require_api_keys
[ -f "$REWARD_JSON" ] || { echo "[ERROR] reward 配置不存在: $REWARD_JSON" >&2; exit 2; }

# ---- RUN_TAG（固定名，不带时间戳；OUT_DIR 存在则自动两级续跑）----
# 起新实验：换名字（新目录自然从头跑）。强制重头跑同目录：SELF_EVOLVE_DISABLE_RESUME=1。
RUN_TAG="${RUN_TAG:-ours_0926}"
OUT_DIR="$RUNS_ROOT/$RUN_TAG"
mkdir -p "$OUT_DIR"

# =============================================================================
# 5) 闭环机制的环境开关
# =============================================================================
export SELF_EVOLVE_TOO_HARD_GAP="${SELF_EVOLVE_TOO_HARD_GAP:-2}"
export SELF_EVOLVE_ANSWER_JUDGE="${SELF_EVOLVE_ANSWER_JUDGE:-1}"
export SELF_EVOLVE_ANSWER_JUDGE_LIVE="${SELF_EVOLVE_ANSWER_JUDGE_LIVE:-1}"
export SELF_EVOLVE_ANSWER_JUDGE_FORCE="${SELF_EVOLVE_ANSWER_JUDGE_FORCE:-0}"
export SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N="${SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N:-$ANSWER_JUDGE_SAMPLE_N}"
export SELF_EVOLVE_ANSWER_JUDGE_CONCURRENCY="${SELF_EVOLVE_ANSWER_JUDGE_CONCURRENCY:-16}"
# （answer judge 的 endpoint / 模型 / key 已在第 2 节统一声明。）

# GRPO 每-step 轨迹+打分落盘（默认开）。每 step 一个文件夹，8 卡各写各的 rankN.jsonl。
# 关掉：export SELF_EVOLVE_GRPO_DUMP_DIR=（置空）。
export SELF_EVOLVE_GRPO_DUMP_DIR="${SELF_EVOLVE_GRPO_DUMP_DIR:-$OUT_DIR/grpo_dumps}"

# trainer 超参透传（主入口 append 到各自命令末尾）
export SELF_EVOLVE_GRPO_EXTRA_ARGS="$GRPO_EXTRA"
export SELF_EVOLVE_SFT_EXTRA_ARGS="$SFT_EXTRA"

# --- wandb 训练曲线（默认开；WANDB=0 关闭。key 在第 2 节声明）---
if [ "${WANDB:-1}" = "1" ]; then
  # 对齐 baseline：conda env 自带的 wandb 在这台机器上不能用，login 和 trainer
  # 都从 $WANDB_PKG_DIR 的 staged 包加载。加载/登录失败直接退出，不跑一个没有
  # 曲线的训练（baseline 也是 fail-hard）。
  [ -d "$WANDB_PKG_DIR/wandb" ] || {
    echo "[ERROR] W&B package not found: $WANDB_PKG_DIR/wandb" >&2
    exit 2
  }
  export PYTHONPATH="$WANDB_PKG_DIR:$PYTHONPATH"
  export SELF_EVOLVE_REPORT_TO=wandb
  export WANDB_PROJECT="${WANDB_PROJECT:-self-evolve-vlm}"
  # 用每轮步数命名（MAX_STEPS=60 只是兜底值，拿它命名会误导）。
  export WANDB_NAME="${WANDB_NAME:-ours_${NUM_ITERATIONS}iters_${GRPO_MAX_STEPS}step}"
  : "${WANDB_API_KEY:?WANDB=1 需要 WANDB_API_KEY（填在第 2 节，或 export；不想用 wandb 就 WANDB=0）}"
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
# 6) 组装 loop 入口参数（内联 stage_cli_args / solver_cli_args / generator_cli_args）
# =============================================================================
ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$BASE_MODEL"
  --output-dir "$OUT_DIR"
  --reward-config "$REWARD_JSON"
  --num-iterations "$NUM_ITERATIONS"
  --allow-more-than-two-iterations
  --num-train-tasks "$NUM_TRAIN_TASKS"
  --num-generations "$NUM_GENERATIONS"
  --num-players "$NUM_PLAYERS"
  --seed "$SEED"
  --reference-provider "$REFERENCE_PROVIDER"
  --reference-model "$REFERENCE_VLM_MODEL"
  --reference-base-url "$REFERENCE_VLM_BASE_URL"
  --enable-openai-reference-vlm
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

# --- 出题器 + self-play 参数 ---
ARGS+=(--edit-fraction "$EDIT_FRACTION" --label-balance "$LABEL_BALANCE")
[ "$COUNTERFACTUAL_PAIRS" = "1" ] || ARGS+=(--no-counterfactual-pairs)
if [ "$SELF_PLAY" = "1" ]; then
  ARGS+=(
    --self-play
    --proposer-group-size "$PROPOSER_GROUP_SIZE"
    --proposer-max-scenes "$PROPOSER_MAX_SCENES"
    --proposer-temperature "$PROPOSER_TEMPERATURE"
    --proposer-max-new-tokens "$PROPOSER_MAX_NEW_TOKENS"
    --proposer-learnability-threshold "$PROPOSER_LEARNABILITY_THRESHOLD"
    --proposer-min-players "$PROPOSER_MIN_PLAYERS"
    --proposer-max-players "$PROPOSER_MAX_PLAYERS"
  )
  [ "$PROPOSER_NO_COMPETENCE" = "1" ] && ARGS+=(--proposer-no-competence)
fi

# --- 训练阶段（STAGES -> --execute-*-smoke）---
case ",$STAGES," in *,sft,*)  ARGS+=(--execute-sft-smoke);; esac
case ",$STAGES," in *,grpo,*) ARGS+=(--execute-grpo-smoke);; esac
case ",$STAGES," in *sft*|*grpo*) ;; *) echo "[ERROR] STAGES=$STAGES 不含有效训练阶段" >&2; exit 2;; esac

# --- GRPO 后端选择。SFT 始终用 DeepSpeed，故 --trainer-deepspeed-config 总是传。---
ARGS+=(--trainer-backend "$TRAINER_BACKEND")
[ -n "$DEEPSPEED_CONFIG" ] && ARGS+=(--trainer-deepspeed-config "$DEEPSPEED_CONFIG")
if [ "$TRAINER_BACKEND" = "fsdp2" ]; then
  [ -f "$FSDP_CONFIG" ] || { echo "[ERROR] FSDP2 配置不存在: $FSDP_CONFIG" >&2; exit 2; }
  ARGS+=(--trainer-fsdp-config "$FSDP_CONFIG")
fi

banner "train  self-evolve loop（完整方法 ours，全链路：出题/self-play -> 筛选 -> solver rollout -> 打分 -> SFT -> GRPO）
  RUN_TAG=$RUN_TAG
  OUT_DIR=$OUT_DIR
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  iters=$NUM_ITERATIONS tasks=$NUM_TRAIN_TASKS players=$NUM_PLAYERS gen=$NUM_GENERATIONS gpus=$NUM_GPUS mode=FULL-PARAM(GRPO=$([ "$TRAINER_BACKEND" = fsdp2 ] && echo "fsdp2=$(basename "$FSDP_CONFIG")" || echo "deepspeed=$(basename "$DEEPSPEED_CONFIG")"); SFT=deepspeed=$(basename "$DEEPSPEED_CONFIG"))
  stages=$STAGES  (sft=$SFT_MAX_STEPS grpo=$GRPO_MAX_STEPS step/iter → 累计 $((NUM_ITERATIONS * GRPO_MAX_STEPS)) grpo step, $((NUM_ITERATIONS * NUM_TRAIN_TASKS)) 题)
  reward=$REWARD_JSON  reference=$REFERENCE_VLM_MODEL(live)  answer_judge=$ANSWER_JUDGE_MODEL(live)"

cd "$REPO"
# resume 场景保留历史 log：append 而非覆盖（-a）。
"$PY" "$LOOP_ENTRY" "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"

banner "训练完成。评测：bash local_scripts/self_evolve/experiments/main/eval.sh $OUT_DIR"
