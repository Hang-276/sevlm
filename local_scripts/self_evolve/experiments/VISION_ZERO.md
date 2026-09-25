# 实验③：官方 Vision-Zero 独立运行

入口：`main/vision_zero_baseline.sh`。该入口仅负责启动官方训练代码，不使用
sevlm 的 trainer、`paths.sh`、`common.sh`、`train_defaults.sh` 或 `.env`，
不会改变 ours 主实验的实现和配置。它替代了原先调用 sevlm 本地 trainer 的入口。

## 固定官方版本

官方仓库已更名为 [RLSVR](https://github.com/wangqinsi1/RLSVR)，Vision-Zero
代码位于 `vision-zero` 分支；默认 `main` 是另一套方法，不能用于本实验。

在 GPU 服务器上准备独立目录（以下路径均为示例，需替换）：

```bash
git clone --branch vision-zero --single-branch https://github.com/wangqinsi1/RLSVR.git /data/Vision-Zero-official
git -C /data/Vision-Zero-official checkout --detach 386fa20711130b9c7d8a340edd285c6242d8d255
```

启动器要求这个提交及未修改的官方训练源码。不要在官方目录复制 sevlm 的
trainer、reward 或配置文件。

## 独立环境与数据

创建独立 Conda 环境，不要在 ours 的环境里安装官方包：两边包名都叫 `open-r1`。
以下安装只应在有 CUDA 的训练服务器上执行：

```bash
conda create -n vision-zero-official python=3.11 -y
conda activate vision-zero-official
cd /data/Vision-Zero-official
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
- 独立输出目录；官方每 5 步保存全模型，需要预留足够磁盘空间。

## 预览与正式启动

```bash
export VISION_ZERO_REPO=/data/Vision-Zero-official
export VISION_ZERO_PY=/path/to/conda/envs/vision-zero-official/bin/python
export VISION_ZERO_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export VISION_ZERO_DATASET=/data/Vision-Zero-clevr-dataset
export VISION_ZERO_OUTPUT=/data/runs/vision_zero_official_seed42

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
防止继承另一实验的设置。训练超参数固定为下表所示官方配方。

## 官方配方与启动修正

来源：[官方启动脚本](https://github.com/wangqinsi1/RLSVR/blob/386fa20711130b9c7d8a340edd285c6242d8d255/run_scripts/run_grpo_vision_zero.sh)。
这是公开代码的配置，不保证等于论文中每组结果的完整配方。

| 参数 | 值 |
|---|---|
| 模型与适配 | Qwen2.5-VL-7B-Instruct，全参数 BF16 |
| GPU / ZeRO | 8 卡；官方 `zero3_model_parallel.json`，参数及优化器 CPU offload |
| 玩家 / clue rounds | 4 / 2 |
| 阶段 | interactive；decision、clue 每 1 个更新步切换 |
| 奖励 | clevr_clue_format_with_votes + clevr_decision_accuracy |
| Epoch size / epochs | 450 / 40；不以 320 步替代 |
| G / 每卡 batch / 梯度累积 | 8 / 1 / 8 |
| LR / beta | 1e-5 / 0.04 |
| Warmup / scheduler | 0.1 / cosine |
| Prompt / completion 参数 | 8000 / 512；实际截断行为由官方训练器决定 |
| 图像像素上下限 | 3136 / 12845056，来自官方程序默认值 |
| Attention / vLLM | flash_attention_2 / False |
| Seed / num_iterations | 42 / 1 |
| 保存 | 每 5 步，只保存模型 |

与上游 shell 的差异仅在启动和记录层面：正确传递模型变量、使用绝对路径、
省略不兼容的 `--dispatch_batches False`、将上游默认值显式写出、默认关闭 WandB，
以及保存 `official_commit.txt`、`launch_command.txt`、`run.log`。
官方源码及其 ZeRO 配置不作修改；没有将 ours 的奖励、SFT 或筛题流程带入③。

## 评测与论文记录

训练成功后，`VISION_ZERO_OUTPUT` 是可供评测的完整模型目录。
本入口不写旧公共 runner 的 marker；评测时应显式传入该模型路径。
切回已固定版本的评测环境，与①②④⑤共用同一份 benchmark、prompt、
分辨率、解码及评分配置。不要在官方训练环境中安装 sevlm 来运行评测。

仓库 `analysis/eval_checkpoint.sh <模型目录>` 可作为显式路径入口，但它与
`main/eval.sh` 的默认 benchmark/解码设置不完全相同，不能在五组之间混用默认值。
本次仅替换③的训练入口，不修改公共评测代码；公共 `load_env` 在缺少 `.env` 时
提前退出的问题仍需另行处理。

论文记录应写为“使用官方公开训练配方复现的 Vision-Zero”。它与 ours 属于方法对比，
不是单因素消融；如果训练预算不同，应报告实际更新步数、轨迹/token 数和计算资源，
不能声称是等预算实验。先前依据 sevlm 的 320 步/vLLM 配置给出的时间预算不适用于
这套 40 epochs、非 vLLM、CPU offload 配方。
