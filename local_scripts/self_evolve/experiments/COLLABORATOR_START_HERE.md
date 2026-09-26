# 新合作者上手：理解研究，并完成实验①和③

你负责两个结果：**未训练的 base model 得分（①）**，以及 **Vision-Zero 训练后的得分（③）**。
①不训练；③需要先训练，再用与①相同的评测配置测分。两组是主表基线，不是 ours 的奖励消融。
阅读顺序：本文 → [实验③的安装和参数细节](VISION_ZERO.md) → [评测清单](EVAL_DATASETS.md)。

本文对应新增上手文档及独立评测修复后的代码。已检查命令生成和无GPU启动流程，未在八卡 A100 上完成训练验证；
“有启动脚本”不代表已经验证整个环境或能稳定训练到结束。

## 1. 我们想研究什么

研究问题是：**VLM 能否利用自己的解题表现，持续构造有学习价值的视觉任务，
并通过更细的反馈提升视觉推理能力，而不是只学会输出格式或猜答案？**

训练素材是 CLEVR 的合成场景图片。数据里有原图与修改图，可构造“谁是 spy”游戏：
多数玩家看到原图，一个玩家看到修改图，模型观察、描述并判断差异。
这里主要是从已有图像/场景构造问题和游戏，不是每一步重新渲染图片。

Vision-Zero 的基线用官方多玩家 clue（描述）/decision（判断）交互和相应奖励训练。
ours 在同一数据来源上加入任务编辑和 proposer：模型选择场景改动及难度，再根据 solver
的表现筛选任务；用合格轨迹做 SFT replay，再进行 GRPO，之后继续下一轮。
完整版奖励综合 answer、grounding、process、consistency、budget 等信号。
这不是一个外置的逐 token 人工标注 PRM；奖励具体实现见 [REWARD.md](../docs/REWARD.md)。

**这些是方法设计和待验证假设，不是已经证明的提升。** 最终要看公开 benchmark 的结果。
“Base”在本项目指统一的 **Qwen2.5-VL-7B-Instruct 初始 checkpoint**，不是另一个非 Instruct 模型。

| 编号 | 实验 | 用来回答什么 | 你的任务 |
|---|---|---|---|
| ① | Base zero-shot | 不经过本项目训练，原模型能做到什么程度？ | 用原始 checkpoint 推理、评分 |
| ② | Base + GRPO | 普通 outcome-based GRPO 能带来多少提升？ | 其他成员负责 |
| ③ | Base + Vision-Zero | 已有 self-play 方法与 ours 相比如何？ | 独立训练官方实现，再评测 |
| ④ | Ours，outcome-only | 在保留 ours 流程时，奖励设计贡献多少？ | 其他成员负责 |
| ⑤ | Ours，完整版 | 完整方法的表现如何？ | 其他成员负责 |

①没有训练过程，也不要使用 few-shot 示例或先在 benchmark 上微调。
③必须从与⑤相同版本的原始模型开始，不能拿别人已训练的 Vision-Zero 权重当作自己复现的结果。

## 2. 先拿到四项团队约定

正式长跑前，与负责主实验的同学核对以下信息并记录到本次实验目录。
这是复现所需的信息；不需要为这些基线修改主实验代码。

1. **模型和数据版本**：同一初始模型 revision、同一 CLEVR 图像/场景来源及划分。
2. **主实验真实配置**：两个 ours 入口的默认 RL 预算现已统一为两轮、每轮120步，
   累计240步，G=8，每卡batch=2，梯度累积=8，八卡名义有效batch=128。
   ③默认连续训练240步，两个 `VISION_ZERO_PROTOCOL` 名称现在是同一预算的别名。
   旧提交的 `main/ours.sh` 曾默认320步、G=16、名义batch=64；已启动任务、旧模型
   和环境变量覆盖不因本次修改而改变，仍要以主实验真实启动日志为准。
3. **统一评测协议**：固定 VLMEvalKit 提交、6项还是10项 benchmark、解码、像素范围、judge，
   以及 checkpoint 选择规则。①③的协议也必须与②④⑤一致。
4. **机器及存储**：本次计划 8×A100 80GB；确认 CPU 内存和磁盘，③默认有 CPU offload，
   每10步保存全模型。时间和峰值内存需服务器试跑估计，不套用旧 H200/vLLM 日志。

脚本只对齐共同参数和累计 GRPO 更新数，**不是严格等 FLOPs/等轨迹量**。
③保留官方4玩家和交互；⑤还包含 SFT、proposer、筛题、离线 solver，学习率按轮重启。
这些区别应在论文中披露，不能为了“所有参数一样”把③改成 ours 的算法。
本次仅统一 RL 步数、G 和名义有效 batch，未统一两个 ours 入口的 API、生成后端、
proposer 或离线 solver 设置；完整主实验及其奖励消融应继续固定同一启动流程。

论文写法可参考 [Vision-Zero 的 Figure 7](https://arxiv.org/html/2509.25541v2#S3)：
作者用相同8×A100、batch 128、100 iterations对比其方法与普通GRPO。
我们采用相同RL更新数和启动器batch参数的对照口径，另行说明交互及SFT等方法开销；
不据此声称相同训练时间或总计算量。若比较采样效率，还须核对每步实际rollout数；
[MSSR](https://arxiv.org/html/2512.18215v1#S5)明确按每步rollout总量对齐不同方法。

## 3. 目录、环境和数据准备

先接受私有仓库的协作邀请，并用自己的 GitHub 账号配置访问权限；不要共享负责人凭据。
首次获取本项目（已有目录时检查提交版本，不要重复克隆）：

```bash
mkdir -p /data/code
git clone https://github.com/Hang-276/sevlm.git /data/code/sevlm
git -C /data/code/sevlm rev-parse HEAD
```

建议目录（示例路径，按服务器修改）：

```text
/data/
  code/sevlm/                       # 本项目：只用③的独立启动器
  code/Vision-Zero-official/         # 官方 vision-zero 分支，固定提交
  code/VLMEvalKit/                   # 团队统一评测版本
  models/Qwen2.5-VL-7B-Instruct/     # 统一初始模型，含 config.json 和完整权重
  datasets/Vision-Zero-clevr-dataset/output/
    replacement_images/
    replacement_scenes/
  eval/LMUData/                     # benchmark 数据，与 CLEVR 训练数据分开
  eval/protocol_v1/                 # 本次统一评测配置和结果
  runs/vision_zero_aligned_seed42/  # ③的训练输出，启动前不应存在
```

使用两个独立环境：

- `vision-zero-official`：只安装固定官方 Vision-Zero 的训练依赖，安装步骤见
  [VISION_ZERO.md](VISION_ZERO.md#独立环境与数据)。不要在其中安装 sevlm，包名会冲突。
- `vlmeval`：安装团队固定版本的 VLMEvalKit 及其模型推理依赖。优先复用团队验证过的
  环境文件；仓库目前没有完整锁定的评测环境，不要默认安装最新版本后与旧结果混用。

官方仓库现名 RLSVR，必须使用其 `vision-zero` 分支，并固定提交
`386fa20711130b9c7d8a340edd285c6242d8d255`，不能用默认 `main`。
③不需要 sevlm 的 `.env`、Reference VLM API 或 WandB；默认不上传 WandB。
评测默认的 API judge 则需要团队提供可用的端点配置，凭据不要写进提交或交付材料。

## 4. 实验①：原始模型直接评测

### 安装独立评测环境

在 GPU 服务器上准备固定版本（若团队已有统一版本/环境，所有组使用团队版本）：

```bash
conda create -n vlmeval python=3.11 -y
conda activate vlmeval
git clone https://github.com/open-compass/VLMEvalKit.git /data/code/VLMEvalKit
git -C /data/code/VLMEvalKit checkout --detach 6f0370704932cec0d9aa1e73f01954cd786c57ea
cd /data/code/VLMEvalKit
python -m pip install -e .
python -m pip install vllm
python -m pip check
python run.py --help
```

这个 VLMEvalKit 提交的接口已核对，但依赖并未形成服务器实测 lockfile；安装中若遇到
CUDA/PyTorch 冲突，应保留完整报错，不要忽略 `pip check`。已验证的团队容器/环境优先。
当前 Qwen 的 Transformers 后端使用 FlashAttention；直接切 `USE_VLLM=0` 不能保证免装它。
模型/数据较大且安装涉及 CUDA，以下命令都在服务器运行，不在无GPU的个人电脑训练。

### 设置一次共同评测参数

新开终端时重新设置这些变量。复制到服务器自己的环境文件也可以，但不要修改主实验脚本。

```bash
conda activate vlmeval
unset PYTHONPATH
export SEVLM_REPO=/data/code/sevlm
export EVAL_PY="$(command -v python)"
export VLMK=/data/code/VLMEvalKit
export LMUData=/data/eval/LMUData
export BASE_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export WORK_DIR=/data/eval/protocol_v1
export DATASETS="MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST"
export MAX_NEW_TOKENS=2048 TEMPERATURE=0.01 TOP_P=0.001 TOP_K=1 DO_SAMPLE=true
export EVAL_MIN_PIXELS=200704 EVAL_MAX_PIXELS=1003520
export REPETITION_PENALTY=1.0 USE_VLLM=1
export JUDGE=gpt-4o-mini
```

上面采用6项、显式 near-greedy 解码。若五组已约定10项，将
`VStarBench MMMU_Pro_10c CV-Bench-2D CV-Bench-3D` 追加到 `DATASETS`，所有组一致。
默认 API judge 还需要当前环境中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；
也可放到个人 `.env` 并用 `ENV_FILE=/绝对路径/个人.env` 指定，勿上传凭据。
如果团队统一使用本地评分，设置 `JUDGE=exact_matching`，无需 `.env` 或 API。
正式比较时不能①用一种 judge、③用另一种。

benchmark 数据由 VLMEvalKit 从 `LMUData` 读取，不存在时按其实现下载。
服务器网络受限时，先将相同版本 benchmark TSV/关联图片准备到该目录；
不要把 CLEVR 数据放在这里冒充评测数据。

### 先预览，再做一次单 benchmark 验证，最后跑完整①

```bash
cd "$SEVLM_REPO"
DRY_RUN=1 bash local_scripts/self_evolve/experiments/main/base_zeroshot.sh

# 单项验证输出与正式结果隔离；该验证会在服务器实际推理。
CUDA_VISIBLE_DEVICES=0 DATASETS=MMVP WORK_DIR=/data/eval/smoke_base \
  bash local_scripts/self_evolve/experiments/main/base_zeroshot.sh

# 正式①，不训练，只推理和评分。
CUDA_VISIBLE_DEVICES=0 bash local_scripts/self_evolve/experiments/main/base_zeroshot.sh
```

预览只用Python标准库生成 `WORK_DIR/base/eval_config.json` 和 `eval_protocol.json`，
不会导入torch或模型，也不会验证显存或live judge连通性。正式运行将终端日志自动保存在
`WORK_DIR/base/eval.log`，预测与评分也在对应的结果目录下。
只有拿到所有约定 benchmark 的有效分数才算①完成；不要只看进度条或退出码。

这是单卡评测示例。八卡不会自动并行；要并行可按 benchmark 分配不同进程和独立
`WORK_DIR`，不能多个进程同时写同一结果目录。不要先给①改分辨率来追求速度。

同一模型标签/结果目录下变更解码、数据清单、judge或评测版本时，脚本会拒绝复用旧协议；
换一个新的 `WORK_DIR`。如果在相同路径替换了模型权重，也必须自己换结果目录。

## 5. 实验③：独立训练 Vision-Zero

按 [VISION_ZERO.md](VISION_ZERO.md) 完成官方固定版本和独立环境安装后，在新终端设置：

```bash
conda activate vision-zero-official
export VISION_ZERO_REPO=/data/code/Vision-Zero-official
export VISION_ZERO_PY="$(command -v python)"
export VISION_ZERO_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export VISION_ZERO_DATASET=/data/datasets/Vision-Zero-clevr-dataset
export VISION_ZERO_OUTPUT=/data/runs/vision_zero_aligned_seed42
export VISION_ZERO_PROTOCOL=ours_full_pipeline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/code/sevlm

# 只打印命令并检查路径/提交，不加载模型。
DRY_RUN=1 bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh
```

先确认打印出的步数、G、batch 与团队约定一致。若要在服务器做2步环境验证，
使用另一个输出目录，不能把测试 checkpoint 接着当作正式初始模型：

```bash
VISION_ZERO_MAX_STEPS=2 \
VISION_ZERO_OUTPUT=/data/runs/vision_zero_smoke_seed42 \
bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh

# 正式运行会使用先前 export 的正式输出目录和完整步数，需在持久终端中执行。
bash local_scripts/self_evolve/experiments/main/vision_zero_baseline.sh
```

两步运行只验证基本执行路径，不证明长时间稳定。预览不会检验显存；模型加载、
loss/backward、保存成功仍需服务器验证。当前入口拒绝已有输出目录，不提供自动续训。
失败时保留日志，与负责人定位原因；不要悄悄改成 LoRA、换模型、缩小分辨率或跳过错误。

本次正式配置的关键值：240步、G=8、每卡batch=2、累积8、LR=1e-6、beta=0.06、
completion上限2048、图像像素范围802816–1003520；4玩家、2 clue rounds及官方奖励保留。
具体含义和预算覆盖方式见 [VISION_ZERO.md](VISION_ZERO.md)。

## 6. 用与①相同的协议评测③

确认训练无异常退出、达到约定正式步数，输出目录包含 `config.json`、完整模型权重
及 tokenizer/processor 文件。中途 checkpoint 不自动等于最终模型；按事先约定选择
最终导出的模型，不根据测试集成绩挑最好的一步。

切回评测环境，重新设置第4节的环境变量，保留完全相同的 `DATASETS`、解码和 judge：

```bash
conda activate vlmeval
unset PYTHONPATH
export EVAL_PY="$(command -v python)"
export VISION_ZERO_OUTPUT=/data/runs/vision_zero_aligned_seed42
cd "$SEVLM_REPO"

DRY_RUN=1 LABEL=vision_zero \
  bash local_scripts/self_evolve/experiments/analysis/eval_checkpoint.sh "$VISION_ZERO_OUTPUT"
CUDA_VISIBLE_DEVICES=0 LABEL=vision_zero \
  bash local_scripts/self_evolve/experiments/analysis/eval_checkpoint.sh "$VISION_ZERO_OUTPUT"
```

①入口现在委托同一个独立评测器，默认6项；单 checkpoint 入口不指定 `DATASETS` 时仍默认10项，
所以第4节必须显式设置并在两个实验间保持一致。新评测器不加载 `paths.sh` 或切换训练环境，
`CKPT`、`LABEL` 和包含空格/引号的模型路径也可正确传入。

结果在 `WORK_DIR/vision_zero/`，包含 `eval_config.json`、`eval_protocol.json`、`eval.log`
以及预测和各 dataset 的评分文件。②④⑤也应按同一显式协议评测，不能假设旧的
`main/eval.sh` 框架默认解码与本协议自动相同。

**评测数据不是 CLEVR 生题结果。** VLMEvalKit 读取公开 benchmark 原有的图片和题目，
按各 benchmark 的规则评分；这测的是 CLEVR 后训练后的跨数据集迁移能力。
API judge 通常辅助答案/选项提取，不是直接用训练的五维 reward 评分。

## 7. 什么算完成，交付哪些东西

交付给项目负责人（文件路径或项目共享存储即可；不要只发截图）：

- ①③在每个约定 benchmark 上的原始预测和评分文件；保留框架完整输出，
  不要假设所有指标文件都一定叫 `*_acc.csv`。注明运行失败、缺失或答案解析失败。
- 一张①/③ × benchmark 的分数表，注明指标与单位；缺项不能填0，平均分只在同一组任务上算。
- ③最终 checkpoint 路径、训练 `run.log`、`launch_command.txt`、`alignment_config.txt`、
  `official_commit.txt`、运行时生成的 `zero3_vision_zero.json`。
- ①③的评测 JSON、日志、VLMEvalKit 提交、sevlm 提交、模型/data revision、
  Python/PyTorch/CUDA/依赖版本、GPU型号、耗时、seed。运行中所有配置偏离都应说明。

“生成出了几个样例”“进度条动了”“退出码为0”都不能单独代表实验完成：还应核对
训练实际步数、模型导出、所有 benchmark 的预测覆盖和可用分数。

## 8. 常见问题速查

| 现象 | 先检查 |
|---|---|
| 官方仓库变成 SpyRL/verl | 是否取了默认 main；应使用固定 vision-zero 提交 |
| 提示输出目录存在 | 当前③入口不自动续训；不要覆盖或删除旧结果来掩盖失败 |
| `experiments/paths.sh` 不存在 | 使用了旧脚本；更新到本次独立评测修复版本 |
| 没有 `.env` 就退出 | ①③的新评测入口不依赖它；检查是否误用了旧公共入口 |
| CPU内存/显存不足 | 记录完整配置和报错；CPU offload 会用主机内存，预览不保证能装下 |
| base 和③分数不可比 | 检查6项/10项、模型版本、解码、judge、像素和缓存目录是否一致 |
| 未达到预期涨分 | 正常记录结果；这是研究假设的检验，不要通过改评测协议追求正向结果 |

## 9. 负责人需要另行确认的消融配置

本次没有改动实验④或主实验的奖励实现。当前 `lib/reward_outcome_only.json` 的
`positive_min_grounding / positive_min_process` 为0.3 / 0.5，而完整版
`configs/reward/reward_weights.json` 为0.2 / 0.4；④还未显式复制完整版的 `components` 设置。
因此，即使两组RL步数和batch一致，也不能直接声明④只改变奖励权重。
是否统一筛选阈值及answer定义，需要负责人确定消融意图后另行修改；不影响你按本文准备①③。
