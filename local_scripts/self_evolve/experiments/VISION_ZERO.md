# 实验③：官方 Vision-Zero 实现，对齐主实验设置

入口：`main/vision_zero_baseline.sh`。该入口仅负责启动官方训练代码，不使用
sevlm 的 trainer、`paths.sh`、`common.sh`、`train_defaults.sh` 或 `.env`，
不会改变 ours 主实验的实现和配置。使用固定的官方算法实现，更新步数、batch、
学习率及输入/输出长度等公共参数按下述主实验配置对齐，不再默认使用官方 40 epochs 配方。

## 固定官方版本

官方仓库已更名为 [RLSVR](https://github.com/wangqinsi1/RLSVR)，Vision-Zero
代码位于 `vision-zero` 分支；默认 `main` 是另一套方法，不能用于本实验。

在 GPU 服务器上准备独立目录（以下路径均为示例，需替换）：

```bash
mkdir -p /data/code
git clone --branch vision-zero --single-branch https://github.com/wangqinsi1/RLSVR.git /data/code/Vision-Zero-official
git -C /data/code/Vision-Zero-official checkout --detach 386fa20711130b9c7d8a340edd285c6242d8d255
```

启动器要求这个提交及未修改的官方训练源码。不要在官方目录复制 sevlm 的
trainer、reward 或配置文件。

## 独立环境与数据

创建独立 Conda 环境，不要在 ours 的环境里安装官方包：两边包名都叫 `open-r1`。
以下安装只应在有 CUDA 的训练服务器上执行：

```bash
conda create -n vision-zero-official python=3.11 -y
conda activate vision-zero-official
cd /data/code/Vision-Zero-official
bash setup.sh
python -m pip install json-repair
python -m pip check
```

`setup.sh` 是上游安装入口，并非完全锁定的环境文件；其中要求
`transformers==4.49.0`、`trl==0.16.0`、`deepspeed==0.15.4`，并安装
FlashAttention。上面的 `json-repair` 补足官方训练入口直接导入、但安装脚本未列出的依赖。
安装是否兼容目标服务器的 CUDA/PyTorch 仍须在服务器检查。
训练成功后保存环境版本用于复现；本仓库未声称已在 A100 上完成实测。

分别准备：

- 初始模型：[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)，
  应与主实验使用同一版本，不能换成已训练的 Vision-Zero checkpoint。
- 数据：[Vision-Zero CLEVR](https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset)，
  解压后应含 `output/replacement_images/` 和 `output/replacement_scenes/`。
- 8 张 GPU；本次计划使用 8×A100 80GB。官方 ZeRO 配置同时卸载参数和优化器到 CPU，
  因而还需要充足的主机内存。显存、主机内存和耗时仍需服务器试跑确认。
- 独立输出目录；本入口每 10 步保存全模型，需要预留足够磁盘空间。

## 预览与正式启动

```bash
conda activate vision-zero-official
export VISION_ZERO_REPO=/data/code/Vision-Zero-official
export VISION_ZERO_PY="$(command -v python)"
export VISION_ZERO_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export VISION_ZERO_DATASET=/data/datasets/Vision-Zero-clevr-dataset
export VISION_ZERO_OUTPUT=/data/runs/vision_zero_aligned_seed42

cd /path/to/sevlm
# 只检查路径、官方版本并打印参数，不加载模型、不启动 GPU、不创建输出目录。
DRY_RUN=1 bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh

# 确认服务器环境后正式执行。
bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh
```

所有路径必须是绝对路径。输出目录必须不存在，避免无意间续训旧 checkpoint；
当前入口不提供自动续训。可选机器参数：`CUDA_VISIBLE_DEVICES`（必须列出 8 张卡）、
`VISION_ZERO_MASTER_PORT`、`VISION_ZERO_RUN_NAME`。
默认 `VISION_ZERO_REPORT_TO=none`；需要 WandB 时，在独立环境登录并设置为 `wandb`。
训练不依赖 sevlm 的 `.env` 或 Reference VLM API。

本入口有意不读取主实验的 `MAX_STEPS`、`GRPO_LR`、`NUM_GENERATIONS` 等环境变量，
防止继承另一实验的设置。默认对齐两个 ours 入口现已统一的 RL 预算。

原有 `main_ours` 名称保留为同一预算的别名，以下设置现在不会改变训练预算：

```bash
export VISION_ZERO_PROTOCOL=main_ours
```

如果朋友运行时覆盖了默认值，以其真实启动日志为准，用专用变量覆盖：
`VISION_ZERO_MAX_STEPS`（所有轮累计 GRPO 更新数）、`VISION_ZERO_NUM_GENERATIONS`、
`VISION_ZERO_PER_DEVICE_BATCH`、`VISION_ZERO_GRAD_ACCUM`。
如果还改了学习率、图像像素或长度，需要同步调整③的独立启动参数。
所有最终参数写入 `launch_command.txt`，所选协议和 batch 写入 `alignment_config.txt`。

## 对齐范围与保留的原方法

来源：[官方启动脚本](https://github.com/wangqinsi1/RLSVR/blob/386fa20711130b9c7d8a340edd285c6242d8d255/run_scripts/run_grpo_vision_zero.sh)。
训练代码使用该官方版本；启动参数是本仓库的受控对比配置，不应标注为“官方原生超参数”。
以 `ours_full_pipeline_train.sh` 为基准统一 RL 默认值，没有执行或 source 主实验脚本。
这不改变旧运行：旧提交的 `main/ours.sh` 曾默认320步、G=16、每卡batch=8、累积1；
如需对照已有旧模型，必须根据其日志显式设置 `VISION_ZERO_*` 覆盖值。

| 设置 | 默认 `ours_full_pipeline` | 可选 `main_ours` |
|---|---|---|
| 对齐入口 | `ours_full_pipeline_train.sh` | `main/ours.sh` → 公共 runner |
| 累计 GRPO 更新数 | 2 × 120 = **240** | 2 × 120 = **240** |
| G | 8 | 8 |
| 每卡 batch / 梯度累积 | 2 / 8 | 2 / 8 |
| 名义有效 batch（8 卡） | 128 | 128 |

两种 profile 共用以下设置：

| 参数 | 值 |
|---|---|
| 模型与适配 | Qwen2.5-VL-7B-Instruct，全参数 BF16 |
| GPU / ZeRO | 8 卡；基于官方 ZeRO-3，参数及优化器 CPU offload |
| 玩家 / clue rounds | 4 / 2 |
| 阶段 | interactive；decision、clue 每 1 个更新步切换 |
| 奖励 | clevr_clue_format_with_votes + clevr_decision_accuracy |
| Epoch size | 保留官方 450；只是动态数据长度，训练由 max_steps 停止 |
| LR / beta / weight decay | 1e-6 / 0.06 / 0 |
| Warmup / scheduler | 0.1 / cosine |
| Prompt / completion 参数 | 10240 / 2048；实际截断行为由官方训练器决定 |
| 图像像素上下限 | 802816 / 1003520 |
| Temperature / gradient clipping | 1.0 / 0.3 |
| Attention / vLLM | flash_attention_2 / False |
| Seed / num_iterations | 42 / 1 |
| 保存 | 每 10 步，只保存模型 |

保留官方的 4 玩家、2 clue rounds、交替阶段、奖励和 advantage 算法，
不引入 ours 的 SFT、Reference VLM、proposer 或筛题机制。
官方 ZeRO 配置固定 clipping=1.0，与对齐后的 `max_grad_norm=0.3` 冲突；
入口会在输出目录写一份只将 `gradient_clipping` 改为 `auto` 的配置副本，
不修改官方文件或主实验文件。梯度检查点显式使用 non-reentrant 模式。
模型路径和上游 shell 的变量错误也已修正，省略了不兼容的 `--dispatch_batches False`。

这只是对齐共同参数，不代表等 FLOPs、等轨迹数或等端到端成本：

- 默认 ours 有两轮，学习率调度会按每轮重启；③是一次 240 步连续训练。
- Vision-Zero 的一个输入游戏会展开成交互和多条生成，名义 batch 不等于实际轨迹数。
- ours 的每轮 256 候选题与 Vision-Zero 的动态 `epoch_size=450` 含义不同；
  不能把它们设成同一个数字就声称等训练数据。两边应使用同一 CLEVR 数据来源/划分。
- ours 还有 SFT、筛题与离线 solver；应额外记录生成 token、轨迹数和总资源消耗。
- 官方训练器可能忽略 prompt 截断上限；这里对齐传入值，不宣称实际最大长度被强制一致。

## 评测与论文记录

训练成功后，`VISION_ZERO_OUTPUT` 是可供评测的完整模型目录。
本入口不写旧公共 runner 的 marker；评测时应显式传入该模型路径。
切回已固定版本的评测环境，与①②④⑤共用同一份 benchmark、prompt、
分辨率、解码及评分配置。不要在官方训练环境中安装 sevlm 来运行评测。

使用独立 `analysis/eval_checkpoint.sh "$VISION_ZERO_OUTPUT"`，设置 `LABEL=vision_zero`。
该入口的路径、变量覆盖和 `.env` 处理已修复，经过无GPU启动回归检查；①也使用同一评测器。
默认 benchmark 数量仍不同，必须显式统一 `DATASETS`、解码和 judge。
主实验训练和旧公共评测入口未改动；独立入口不使用公共 `load_env`。

论文可写为“使用官方 Vision-Zero 实现，并对齐共同训练超参数及累计 GRPO 更新数”。
它与 ours 属于方法对比，不是单因素消融；不能仅凭 step 相同声称等计算预算。
当前仍使用官方非 vLLM、CPU offload 路径，时间需在目标服务器重新测量。
具体 benchmark 清单与训练/评测数据区别见 [EVAL_DATASETS.md](EVAL_DATASETS.md)。
