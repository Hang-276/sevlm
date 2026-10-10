# 实验③：官方 Vision-Zero 实现，按论文报告的配置运行

入口：`main/vision_zero_baseline.sh`。该入口仅负责启动官方训练代码，不使用
sevlm 的 trainer、`paths.sh`、`common.sh`、`train_defaults.sh` 或 `.env`，
不会改变 ours 主实验的实现和配置。

作为论文中的 baseline 对照，③采用**官方实现 + 官方超参**（lr 1e-5、beta 0.04、
官方 ZeRO 配置原样），不把 ours 的 lr/beta 搬进来：对照实验里最容易被质疑的就是
"baseline 被换了次优超参"，而把 lr 从 1e-5 改成 1e-6、beta 从 0.04 改成 0.06，
换不来等 FLOPs 的结论，只换来风险。

**配方已定：按论文报告的 100 iterations / batch 128 games 跑。** 仓库脚本给的是另一套
（40 epochs，实为 280 步），两者差 2.8 倍，理由见下节。

## 官方配方的步数：仓库脚本与论文差 2.8 倍

| 来源 | 配置 | 优化步数 | 8×H20 实测估算 |
|---|---|---|---|
| 仓库 `run_scripts/run_grpo_vision_zero.sh` | 40 epochs，accum 8 | **280** | 约 52 小时 |
| 论文附录 A.3.2 命令 | `--num_train_epochs 15`，accum 16 | **45** | 约 17 小时 |
| 论文正文 §3 与表 5 | "100 iterations, batch size 128" | **100** | 约 37 小时 |

推导（2026-10-09 用 smoke 实测校正过，先前一版把 micro-batch 当成优化步、错了 8 倍）：
`CyclicDynamicDataset.__len__` = (450 // G 8) × 8 × num_iterations 1 = **448**，
但 `dispatch_batches=False` 下 accelerate 会把这个 IterableDataset 按 8 个进程**切片**，
所以每进程 `len(train_dataloader)` = 448 / 8 = **56**（实测印证：epoch 计数在 step 1/2
后分别是 0.29/0.57 ⇒ `steps_per_epoch` = 56/16 = 3.5）。
`num_update_steps_per_epoch = 56 // accum`，再乘 `num_train_epochs`
（`.../transformers/trainer.py:5219-5223`）→ 40 × (56//8) = **280 步**；
论文附录的 15 epochs × (56//16) = **45 步**。

每个优化步的算力 = 8 卡 × 1 game × accum 个 micro-batch。实测 1330 s/步（accum 16，
即 83 s/micro-batch），据此估算上表；仓库配置 accum 8 每步约 665 s。**先前"2240 步 /
22 天"的说法是错的**：2240 其实是仓库配置的 micro-batch 总数，不是优化步数。

论文自身有两处不自洽：正文说 batch 128 是 `nproc 8 × accum 16 × G 8` 的乘积（= 1024，不是 128）；
附录给的命令是 `--num_train_epochs 15 --per_device_train_batch_size 8`，而代码会把
per-device >1 丢弃，照附录跑也得不到论文的配置。论文报的 127 A100-hours 反推约 16 小时
wall-clock（8 卡，约 573 s/步），与 100 步量级相符；仓库脚本的 280 步按 A100 也会更快，
但它与论文正文的数字不同。

**已定（2026-10-09）**：③按**论文的配置**跑 —— 100 步 / batch 128 game（= 8 卡 ×
per-device 1 × accum 16）/ G 8 / beta 0.04 / lr 1e-5。这是论文实际汇报的预算，与 ours 的
240 步同量级，论文可写"按论文报告的训练配置复现官方实现"。仓库脚本的 40 epochs（280 步）
不用：它比论文正文多 2.8 倍，不是论文汇报的数字（虽然代价可承受，见上表）。

## 固定官方版本

官方仓库已更名为 [RLSVR](https://github.com/wangqinsi1/RLSVR)，Vision-Zero
代码位于 `vision-zero` 分支；默认 `main` 是另一套方法，不能用于本实验。

在 GPU 服务器上准备独立目录。本机（`/jizhicfs` 为共享存储）已就位于
`/jizhicfs/rtliu/code/Vision-Zero-official`；换机器时把根路径替换成该机的存储根，
不要把资产放到 `/data` 之类被清空的根分区：

```bash
mkdir -p /jizhicfs/rtliu/code
git clone --branch vision-zero --single-branch https://github.com/wangqinsi1/RLSVR.git /jizhicfs/rtliu/code/Vision-Zero-official
git -C /jizhicfs/rtliu/code/Vision-Zero-official checkout --detach 386fa20711130b9c7d8a340edd285c6242d8d255
```

该路径同时是环境里 `open-r1` 的 editable 安装目标（`pip list` 显示
`open-r1 0.1.0.dev0 /…/Vision-Zero-official/src/open-r1-multimodal`），
删掉或换路径会让训练入口导入不到上游代码。

启动器要求这个提交，并且**只允许一处改动**：`patches/vision_zero_paper_alignment.patch`
（见下节）。除此之外编辑官方目录会被入口拒绝，避免出现未记录的修改。

## 论文对齐补丁

**为什么必须打补丁**：官方发布代码无法从命令行复现论文的算法。三处硬伤——
RAE 系数 α 是 `0.9` 而表 5 是 `0.95`；ρ=0.95 的准确率/n-a 滑动统计完全不存在；
阶段切换是固定周期 `step % (cycle*2)`，而论文 App. A.2.3 是阈值滞回 + `K_min`
最小停留。这三处正是论文自称的 Iterative-SPO 贡献。

补丁（`patches/vision_zero_paper_alignment.patch`）改两个文件：

| 文件 | 改动 |
|---|---|
| `clevr_spotdiff_generator.py` | `self.alpha` 0.9 → 0.95；在 clue 指标里增加 `decision_correct` / `decision_na_rate` / `decision_votes_cast`（由已有的 G=8 投票统计算出，不新增一遍解析） |
| `trainer/grpo_trainer.py` | 新增 `VisionZeroStageSwitcher`（ρ/阈值/K_min/P 具名常量 + 纯函数式的切换规则）；`_handle_clevr_spotdiff_training` 用滞回阶段取代固定周期；每个优化步归约一次 acc/na 并更新 EMA；**修掉损失门控**（见下） |

**为什么还要修门控**：`compute_loss` 从 `script_args.training_phase` 重读阶段，
在 interactive 模式下永远拿到字符串 `'interactive'`，于是按阶段过滤样本的 mask
落进 `else` 分支训练**全部**样本，clue 阶段里标为 `god_decision_for_clue_reward`
的样本照样吃到梯度。这与代码自己打印的 "included for reward calculation only"
以及论文的 `L_t = m_t·L_clue + (1−m_t)·L_dec` 都不符。

**论文没写清、由我们定的三处**（都在补丁里做成具名常量，改一行即可调整）：

- `P = 20`：表 5 叫它 "rounds before forcing change"，但算法框从未引用它。按字面
  实现为"连续 P 轮未触发阈值就强制切换"，`PATIENCE = None` 可关闭。**注意副作用**：
  `acc̄` 从 0 按 `1−0.95^k` 爬，要 45 步才够 `τ↑acc=0.9`，所以 P=20 会先触发，实际
  效果接近每 20 步交替一次。
- `acc_t`/`na_t` 的口径：正文说 held-out mini-batch，算法框说 batch。取当前步自己的
  决策投票。
- `q_θ(∅|H)`：代码没有对 {玩家, ∅} 的概率头，用 G=8 次采样的经验频率近似。

**回退**：

```bash
VISION_ZERO_REPO=/jizhicfs/rtliu/code/Vision-Zero-official \
  bash local_scripts/self_evolve/experiments/patches/revert_vision_zero_patch.sh
```

回退脚本只接受"本地改动恰好等于该补丁"的 checkout，否则拒绝动手。入口下次运行会自动
重新应用补丁，校验方式为：checkout 要么干净、要么 `git diff` 的 sha256 与补丁文件
**逐字节相等**；补丁与 diff 的 sha256 会写进输出目录的 `paper_alignment_patch.txt`。
运行时 `VZ_STAGE_SWITCH=cycle` 可临时退回上游的固定周期做 A/B，无需回退补丁。

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
export VISION_ZERO_REPO=/jizhicfs/rtliu/code/Vision-Zero-official
export VISION_ZERO_PY=/jizhicfs/rtliu/miniconda3/envs/vision-zero-official/bin/python
export VISION_ZERO_MODEL=/jizhicfs/rtliu/models/Qwen2.5-VL-7B-Instruct
export VISION_ZERO_DATASET=/jizhicfs/rtliu/data/Vision-Zero-clevr-dataset
export VISION_ZERO_OUTPUT=/jizhicfs/rtliu/self_evolve_runs/vision_zero_official_seed42

cd /jizhicfs/rtliu/sevlm/sevlm
# 只检查路径、官方版本并打印参数，不加载模型、不启动 GPU、不创建输出目录。
DRY_RUN=1 bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh

# 确认服务器环境后正式执行（必须在 28.59.7.41 上，8 卡）。
bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh
```

所有路径必须是绝对路径。输出目录必须不存在，避免无意间续训旧 checkpoint；
当前入口不提供自动续训。可选机器参数：`CUDA_VISIBLE_DEVICES`（必须列出 8 张卡）、
`VISION_ZERO_MASTER_PORT`、`VISION_ZERO_RUN_NAME`。
默认 `VISION_ZERO_REPORT_TO=none`；需要 WandB 时，在独立环境登录并设置为 `wandb`
（注意该环境的 wandb 是 0.18.3，能否上报未验证）。
训练不依赖 sevlm 的 `.env` 或 Reference VLM API。

本入口有意不读取主实验的 `MAX_STEPS`、`GRPO_LR`、`NUM_GENERATIONS` 等环境变量，
防止继承另一实验的设置。默认值就是论文报告的配置；要跑多种子或有意偏离时，按
`VISION_ZERO_` 前缀覆盖，并把实际启动参数记录下来：

| 变量 | 默认（= 论文） | 说明 |
|---|---|---|
| `VISION_ZERO_MAX_STEPS` | 100 | 论文的 "100 iterations"；官方脚本用的是 `--num_train_epochs 40` |
| `VISION_ZERO_EPOCH_SIZE` | 450 | 动态数据集的长度参数，不是训练长度 |
| `VISION_ZERO_NUM_GENERATIONS` | 8 | |
| `VISION_ZERO_PER_DEVICE_BATCH` | 1 | **>1 会被官方代码丢弃，见下** |
| `VISION_ZERO_GRAD_ACCUM` | 16 | 8 卡 × 1 × 16 = 128 game/步 = 论文的 batch 128 |
| `VISION_ZERO_LR` / `VISION_ZERO_BETA` | 1e-5 / 0.04 | |
| `VISION_ZERO_SEED` | 42 | 同时写入 `--seed` 和 `--data_generator_seed` |
| `VISION_ZERO_SAVE_STEPS` | 10 | 论文命令是 5；不改变训练出的模型 |

所有最终参数写入 `launch_command.txt`，配方记录写入 `official_recipe.txt`。

## 配方与相对官方的改动

训练代码：[官方 pinned 版本](https://github.com/wangqinsi1/RLSVR/blob/386fa20711130b9c7d8a340edd285c6242d8d255/run_scripts/run_grpo_vision_zero.sh)。
超参：论文（arXiv 2509.25541）正文与表 5。论文可写为“按论文报告的训练配置复现官方实现”。

| 参数 | 值 | 来源 |
|---|---|---|
| 优化步数 | **100** | 论文 "100 iterations" |
| 名义 batch | **128 games** | 论文 "batch size 128"；= 8 卡 × per-device 1 × accum 16 |
| G | 8 | 论文 |
| LR / beta / warmup / scheduler | 1e-5 / 0.04 / 0.1 / cosine | 论文附录 A.3.2 |
| ZeRO | 官方 `zero3_model_parallel.json` 原样，参数与优化器 CPU offload，clipping 1.0 | 官方 |
| max_grad_norm | 不传，沿用官方 ZeRO 的 1.0 | 官方 |
| 图像像素 | 不传，用入口默认 3136 / 12845056（Qwen2.5-VL 默认） | 论文命令未传 |
| 模型与适配 | Qwen2.5-VL-7B-Instruct，全参数 BF16 | 论文 |
| 玩家 / clue rounds / 阶段 | 4 / 2 / interactive，由论文的滞回判据切换 | 论文 + 补丁 |
| 奖励 | clevr_clue_format_with_votes + clevr_decision_accuracy | 官方 |
| Attention / vLLM / num_iterations | flash_attention_2 / False / 1 | 论文 |
| 保存 | 每 5 步只存模型（约 320 GB） | 论文 |

保留官方的 advantage 算法、4 玩家、2 clue rounds 与交替阶段，
不引入 ours 的 SFT、Reference VLM、proposer 或筛题机制，也不改写官方 ZeRO 配置。

相对论文附录命令，本入口保留的差异只剩四处，全部记入 `official_recipe.txt`：

1. **用 `--max_steps 100`，而不是 `--num_train_epochs 15`**。论文表 5 与正文都写
   "100 iterations"，附录命令的 15 epochs 换算成 420 步；取正文/表 5 的数字（见上文
   的步数一节）。
2. **`per_device_train_batch_size` 用 1，论文命令写的是 8**。官方代码每个 micro-batch 只取
   第一个 game（`inputs = [inputs[0]]`），写 8 会被静默丢弃、白烧 7/8 的生成；accum 16
   让有效 batch 仍是论文的 128 game。
3. **`--dispatch_batches False`**：论文附录命令自己也带，不算偏离；列出只因上游
   `run_grpo_vision_zero.sh` 之外的人容易漏掉（accelerate 对 `IterableDataset` 默认
   `dispatch_batches=True`，会在 rank 0 拼接含字符串的 dict 并 `TypeError`）。
4. **`--gradient_checkpointing_kwargs '{"use_reentrant": false}'`**：论文命令没有，
   与 transformers 4.49 配合所需，不改变数值行为。

已按论文去掉的：`--seed`（论文未传；`TrainingArguments.seed` 默认就是 42，等价）、
`--use_vllm False`（论文未传；`GRPOConfig` 默认即 False，等价）、
`--interactive_cycle_length`（论文未传；补丁后由滞回判据接管，该参数不再起作用）。

启动环境由入口自己 `export DS_SKIP_CUDA_CHECK=1`：该环境没有预编译的 deepspeed 算子，
cpu_adam 要用系统 nvcc 12.9 现场编译而 torch 是 cu124，版本检查会直接终止运行；
cpu_adam 是纯 CPU 算子（只用 g++ 编译），跳过检查不影响配方。

### 仍然无法对齐、论文里不能声称一致的东西

- `--max_prompt_length` 会被 trainer 强制置 None 并告警（`grpo_trainer.py:452`），传了也不生效。
- CLEVR 交互路径自带 `GenerationConfig`，写死 `max_new_tokens=1024`、`temperature=0.8`、
  `min_new_tokens=20`（`grpo_trainer.py:1281/1537/2711` 等），命令行传的 `--temperature`
  和 `--max_completion_length` 到不了那里。**论文完全没给这三个值**，所以连"对齐"的
  靶子都没有，只能如实说明用的是代码里的值。
- **每卡 batch > 1 无效**：`grpo_trainer.py:976` 是 `inputs = [inputs[0]]`（注释写 "avoid OOM"），
  一个 micro-batch 只有第一个 game 进梯度，其余照样生成、照样占显存。入口默认 1，并在 >1 时告警。
- 阶段切换的三处重建选择（`P` 的作用、`acc_t`/`na_t` 的口径、`q_θ(∅|H)` 的近似）
  见[论文对齐补丁](#论文对齐补丁)一节。

### 与 ours 的可比性

③按论文跑 100 步，ours 的 RL 预算是 240 步。两者量级接近但不相等（仓库脚本那套 280 步
则与 ours 更接近，但仍不采用，因为论文正文报的是 100）。按各自实际配置如实报告，不声称等 FLOPs、等轨迹数或等
端到端成本：

- ours 有两轮，学习率调度按每轮重启；③是一次连续训练。
- Vision-Zero 的一个 game 会展开成交互和多条生成，名义 batch 不等于实际轨迹数。
- ours 的每轮 256 候选题与 Vision-Zero 的动态 `epoch_size=450` 含义不同；
  两边应使用同一 CLEVR 数据来源/划分，但不能声称等训练数据。
- ours 还有 SFT、筛题与离线 solver；应额外记录生成 token、轨迹数和总资源消耗。
- ③每个优化步消耗 128 个 game 的生成（每 game 展开成 4 玩家 × 2 轮交互），是官方脚本
  每步 64 个 game 的两倍。空机 smoke 实测约 850 s/步（当时每步 64 game）→ 100 步约
  **24–48 小时/seed**。850 s 来自 2026-09-27 的单步 smoke，正式跑前应重新测量。

## 评测与论文记录

训练成功后，`VISION_ZERO_OUTPUT` 是可供评测的完整模型目录。
本入口不写旧公共 runner 的 marker；评测时应显式传入该模型路径。
切回已固定版本的评测环境，与①②④⑤共用同一份 benchmark、prompt、
分辨率、解码及评分配置。不要在官方训练环境中安装 sevlm 来运行评测。

使用独立 `analysis/eval_checkpoint.sh "$VISION_ZERO_OUTPUT"`，设置 `LABEL=vision_zero`。
该入口的路径、变量覆盖和 `.env` 处理已修复，经过无GPU启动回归检查；①也使用同一评测器。
默认 benchmark 数量仍不同，必须显式统一 `DATASETS`、解码和 judge。
主实验训练和旧公共评测入口未改动；独立入口不使用公共 `load_env`。

论文可写为“使用官方 Vision-Zero 实现，按论文报告的训练配置（100 iterations、batch 128）
复现”。仓库脚本里的 40 epochs 与论文报的预算不一致，若要引用需在附录说明取舍。
它与 ours 属于方法对比，不是单因素消融；两者训练预算不同，不能声称等计算预算。
当前仍使用官方非 vLLM、CPU offload 路径，时间需在目标服务器重新测量。
具体 benchmark 清单与训练/评测数据区别见 [EVAL_DATASETS.md](EVAL_DATASETS.md)。
