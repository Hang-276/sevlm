# 实际评测入口与训练/评测数据

本说明核对的是脚本实际构造的 VLMEvalKit 配置，而不是历史文档的计划清单。

## 两个入口的默认 benchmark 不同

`main/eval.sh` 调用 `lib/common.sh::run_tier1_eval`：默认且当前支持6项。
`main/base_zeroshot.sh` 现已委托独立的 `analysis/eval_checkpoint.sh`，仍默认6项，
显式设置 `DATASETS` 时可选择独立入口支持的10项。

`analysis/eval_checkpoint.sh <完整模型目录>` 的默认列表及映射支持 10 个 benchmark。
路径推导和环境变量覆盖问题已修复；独立入口不再 source 训练环境或依赖可选 `.env`。
支持 `DRY_RUN=1` 保存评测配置但不启动模型。具体操作见
[COLLABORATOR_START_HERE.md](COLLABORATOR_START_HERE.md)。

| Benchmark 配置名 | 主表入口（6项） | 单 checkpoint 入口（10项） | 主要任务 |
|---|---|---|---|
| MMVP | 是 | 是 | 视觉细节与模式辨别 |
| MMStar | 是 | 是 | 综合多模态理解与推理 |
| BLINK | 是 | 是 | 视觉感知、多图关系等 |
| RealWorldQA | 是 | 是 | 真实场景、空间理解 |
| AI2D_TEST | 是 | 是 | 科学示意图问答 |
| ChartQA_TEST | 是 | 是 | 图表读数与推理问答 |
| VStarBench | 否 | 是 | 高分辨率图像中的细节查找与理解 |
| MMMU_Pro_10c | 否 | 是 | 跨学科多模态题，10选项版本 |
| CV-Bench-2D | 否 | 是 | 二维空间关系与计数 |
| CV-Bench-3D | 否 | 是 | 深度顺序与相对距离 |

`docs/EVALUATION.md` 提到的 13 项 Tier-2 清单不是当前这两个脚本的完整可执行配置。
`eval/fetch_vlmeval_tsv.sh` 也只预下载前 6 项。评测数据位置由 `LMUData` 指定，
默认按 `$WORKSPACE/eval/LMUData` 推导；独立入口没有 WORKSPACE 时使用仓库目录，
因此新机器建议显式设置 `LMUData`。是否已下载齐全需在服务器检查。
脚本中“本机已经准备”的注释不代表其他服务器已有数据。

## 训练时如何用数据

实验③读取 Vision-Zero CLEVR 数据集中的
`output/replacement_images/` 和 `output/replacement_scenes/`，
用官方生成器把原图/修改图分配给不同玩家，动态构造“谁是 spy”的游戏，
再通过 clue 和 decision 交互生成训练轨迹。

ours 使用同一 CLEVR 图像/场景来源，但有自己的编辑、proposer、筛题、
reward 和训练闭环。因此“同一原始数据来源”不等于“两边产生完全相同的训练题目”。
这里的生题主要是使用已有图像构造游戏/问答任务，不是每步都重新渲染 CLEVR 场景。

## 评测时如何用数据

评测脚本加载原始模型或训练完成的 checkpoint，由 VLMEvalKit 读取上述公开
benchmark 的既定图片、问题、选项/答案，再进行推理和评分。
这条路径不调用 CLEVR 生题器，也不使用训练时的五维 reward 给最终 benchmark 打分。

默认 judge 为 `gpt-4o-mini`；支持 `JUDGE=exact_matching` 切换本地评分。
judge 通常用于选项/答案提取，具体评分由各 dataset 类处理；不能认为每道题都交给
GPT 主观评分。不同 benchmark 的分数应使用其实际评测实现与指标。

因此实验主张是“在 CLEVR 任务上后训练后，对公开 benchmark 的迁移/泛化表现”，
不是“在相同 CLEVR 训练题上的得分”。这些脚本没有把上述 benchmark 作为本次后训练数据；
这不等于证明基础模型预训练时从未见过相关数据。

## 五组模型必须统一评测

旧公共主表入口与独立入口的默认解码不完全一致：`main/eval.sh` 没有显式固定 temperature、
max_new_tokens 等参数，依赖已安装的 VLMEvalKit；单 checkpoint 入口显式设为
`max_new_tokens=2048`、`temperature=0.01`、`do_sample=true`。
两者均默认 `min_pixels=200704`、`max_pixels=1003520`、`use_custom_prompt=false`。

决定用 6 项还是 10 项后，五组都应使用同一个入口、同一 VLMEvalKit 提交和
同一解码/评分配置。若希望覆盖当前全部 10 项，可对五组完整模型分别调用
`analysis/eval_checkpoint.sh`，并显式统一 `DATASETS`、`DO_SAMPLE`、
`TEMPERATURE`、`MAX_NEW_TOKENS`、像素范围及 judge。
不要把①默认的 6 项平均分与③④⑤默认的 10 项平均分比较。

独立评测入口的启动和参数传递已经过无GPU回归检查，没有运行真实模型。
`main/eval.sh` 及主实验训练代码未改动；公共 `load_env` 等历史问题不影响本文推荐的独立入口。

原始 benchmark 说明可参考：
[MMVP](https://tsb0601.github.io/mmvp_blog/)、
[MMStar](https://github.com/MMStar-Benchmark/MMStar)、
[BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)、
[ChartQA](https://github.com/vis-nlp/ChartQA)、
[MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU)、
[CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)。
