# 实际评测入口与训练/评测数据

本说明核对的是脚本实际构造的 VLMEvalKit 配置，而不是历史文档的计划清单。

## 九项 benchmark 的统一入口

`main/eval.sh`、`main/base_zeroshot.sh` 均委托
`analysis/eval_checkpoint.sh`：默认恰好跑用户指定的九项，使用同一解码、
像素范围和评分配置。`main/eval.sh` 只负责找到最终完整模型并可选地再跑 base。

`analysis/eval_checkpoint.sh <完整模型目录>` 还支持显式选 `VStarBench` 作为第十项，
但它不进入默认九项平均分。独立入口不加载训练环境，支持 `DRY_RUN=1`
保存配置但不启动模型。路径、VLMEvalKit 提交、judge 与解码保存在每个模型的
`eval_protocol.json`；同一 LABEL 下可增量增加数据集，但不允许改变已有协议。

| Benchmark 配置名 | 默认九项 | 主要任务 |
|---|---|---|
| MMVP | 是 | 视觉细节与模式辨别；使用成对准确率 Overall |
| MMStar | 是 | 综合多模态理解与推理 |
| BLINK | 是 | 视觉感知、多图关系等 |
| RealWorldQA | 是 | 真实场景、空间理解 |
| AI2D_TEST | 是 | 科学示意图问答 |
| ChartQA_TEST | 是 | 图表读数与推理问答；VLMEvalKit 已输出百分数 |
| MMMU_Pro_10c | 是 | 跨学科多模态题，10选项版本 |
| CV-Bench-2D | 是 | 二维空间关系与计数；按来源宏平均 |
| CV-Bench-3D | 是 | 深度顺序与相对距离 |
| VStarBench | 否，显式选用 | 高分辨率图像中的细节查找与理解 |

`docs/EVALUATION.md` 的 13 项 Vision-Zero 文献清单是历史参考，不是当前九项协议。
`eval/fetch_vlmeval_tsv.sh` 只预下载前 6 项；其余可由 VLMEvalKit 自行下载，
或者预放进 `LMUData`。评测数据位置由 `LMUData` 指定，默认是
`$WORKSPACE/eval/LMUData`。新机器应显式检查九个 TSV 的完整性及 VLMEvalKit
版本；脚本配置中的数据集名不代表数据已下载。

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

## 模型比较与结果汇总

入口默认固定 `max_new_tokens=2048`、`temperature=0.01`、`top_p=0.001`、
`top_k=1`、`do_sample=true`，像素范围为 `200704–1003520`；若需要严格贪心，
统一设置 `DO_SAMPLE=false`。默认 judge 是 `gpt-4o-mini`，无 API 时可统一改
`JUDGE=exact_matching`。所有模型必须用相同设置跑九项。

`analysis/collect_results.sh` 默认只汇总九项，扫描新旧 VLMEvalKit 输出层级，
分别读取各数据集明确的 Overall 指标。只要任何模型缺项、重复跑出的得分冲突，
或 `eval_protocol.json` 缺失/不一致，就暂不计算 AVG(9)。单项分数仍展示以便排查；
使用 `VERBOSE=1` 可查看源文件和原因。

入口和汇总的参数传递已经过无 GPU 回归检查，尚未运行真实模型。

原始 benchmark 说明可参考：
[MMVP](https://tsb0601.github.io/mmvp_blog/)、
[MMStar](https://github.com/MMStar-Benchmark/MMStar)、
[BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)、
[ChartQA](https://github.com/vis-nlp/ChartQA)、
[MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU)、
[CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)。
