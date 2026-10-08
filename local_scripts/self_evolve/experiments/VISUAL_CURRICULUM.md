# 视觉课程增强：论文依据与实验方案

检索截至 **2026-10-03**，以论文原文和官方实现为依据。下面覆盖最相关的机制，不声称穷尽所有论文。改动基于上游代码及已有的 visual-facts 实现。

## 判断

当前最值得增强的是“视觉变化能否改变答案”和“感知技能是否真正掌握”，再修正优化目标中的长度偏置。把更多算法同时接入会增加归因和稳定性难度。下面的课程与数据混合是结合本仓库作出的设计推断；论文结果不能替代本实验的九项评估。

| 论文（首次提交日期） | 可借鉴机制 | 本框架中的处理 |
|---|---|---|
| [Same Reward, Different Skills](https://arxiv.org/abs/2610.01908)，2026-10-01 | 问题固定、图像改变、答案改变，同时保证可学性 | 落地同问题反事实 count QA；刚发布的预印本 |
| [Evidence-RL](https://arxiv.org/abs/2608.08021)，2026-08-08，09-28 修订 | 对照证据区和等面积非证据区干预后的答案支持下降 | 后续独立消融；需额外 forward，不能把事实标签正确当成因果 grounding |
| [PointRL](https://arxiv.org/abs/2608.25299)，2026-08-26 | 将定位监督转换为可验证的点预测 | 后续考虑 bbox 难学时的轻量定位目标 |
| [Staying VIGILant](https://arxiv.org/abs/2606.26387)，2026-06-24 | seeing/blind 路径的反事实视觉对齐 | 离线 DPO，暂不混入现有 GRPO |
| [SSL-R1](https://arxiv.org/abs/2604.20705)，2026-04-22 | 图像自监督任务产生可验证 RL 奖励 | 支持视觉代理任务方向；本轮 count QA 是机制借鉴 |
| [PuzzleCraft / PC-GRPO](https://arxiv.org/abs/2512.14944)，2025-12-16 | 可学难度课程与探索能力 | 保留 spy 可学性，增加证据掌握状态 |
| [VisPlay](https://arxiv.org/abs/2511.15661)，2025-11-19 | Questioner/Reasoner 交替演化，难度与多样性反馈 | 保留当前 proposer；可信 scene gold 优先于多数投票标签 |
| [Vision-Zero](https://arxiv.org/abs/2509.25541)，2025-09-29，2026-03-04 修订 | CLEVR、图表、真实图像的多域视觉游戏 | 当前 CLEVR 路径可直接增强；多域覆盖是下一阶段 |
| [PAPO](https://arxiv.org/abs/2507.06448)，2025-07-08 | 视觉策略差异与双熵正则 | 暂缓：额外 forward，且论文报告 perception KL hacking 风险 |
| [ViCrit](https://arxiv.org/abs/2506.10128)，2025-06-11 | 属性、计数、空间幻觉的可验证诊断代理任务 | 支持细粒度感知监督；未来可扩展唯一指代的属性 QA |
| [Perception-R1: Visual Perception Reward](https://arxiv.org/abs/2506.07218)，2025-06-08 | 教师视觉事实对齐与奖励 | 现有 verified-facts 用确定性 scene gold 替代教师/judge |
| [SATORI-R1](https://arxiv.org/abs/2505.19094)，2025-05-25 | Caption、BBox、Answer 的视觉锚定 | 与已有证据框及事实监督相容，非完整复现 |
| [Perception-R1: Perception Policy](https://arxiv.org/abs/2504.07954)，2025-04-10 | 感知任务的规则奖励与难度 | 支持 count/grounding 直接监督；与上面的同名论文不同 |
| [VL-Rethinker](https://arxiv.org/abs/2504.08837)，2025-04-10 | Selective Sample Replay、Forced Rethinking | 已有退化组处理；暂不强制长反思或回放旧策略 rollout |
| [Dr. GRPO](https://arxiv.org/abs/2503.20783)，2025-03-26 | 去除 reward std 和回答实际长度归一化 | 落地固定分母；原代码只去除了 std |
| [DAPO](https://arxiv.org/abs/2503.14476)，2025-03-18 | 动态采样、非对称 clipping、token 归一化、截断处理 | 保留现有组 mask；不声称已完整复现 DAPO |
| [GSPO](https://arxiv.org/abs/2507.18071)，2025-07-24 | sequence ratio 与 sequence clipping | 暂缓完整目标切换 |
| [SAPO](https://arxiv.org/abs/2511.20347)，2025-11-25 | token soft gate 与非对称温度 | 当前一次更新路径 ratio=1，直接替换 gate 收益依据不足 |
| [Entropy-Preserving RL](https://arxiv.org/abs/2603.11682)，2026-03-12 | 熵控制与精度/backend 影响 | 先记录分任务训练信号；暂不加未经验证的控制器 |
| [Scaling Laws for Collapse in Async GRPO](https://arxiv.org/abs/2607.01083)，2026-07-01，09-27 修订 | 同步间隔和学习率共同影响异步稳定性 | 保留当前同步 rollout，不新增异步 staleness |

## 本轮实现

**反事实单图 QA。** 同一场景的原图与实际接受的编辑图使用完全相同的计数问题，正确答案必须不同。核对对象索引、原始属性、replacement、keep subset、变化证书和图片路径；两个样本整体接受或丢弃。每 scene 每 attribute 最多一个 pair。正确性依赖可信 render/edit 元数据，路径检查不能证明每个像素的语义。

QA 采用 `r = 1[格式有效且计数正确]`，两侧分别在原有 GRPO 组内打分，不依赖两侧出现在同一 batch。保留游戏的五维奖励；QA 没有 bbox/过程目标，相关指标记为不适用。默认最多替换 **12.5%** 的训练 rows，只替换未配对的游戏任务，保持游戏 pairs、总 GRPO rows、G 和训练步数。可用 slot 不足或证书失败时回退游戏 rows。没有额外离线 solver、外部 judge 或干预 forward。

**联合掌握课程。** spy 成功率继续判断任务是否过难；全 spy 正确但计数、格式、框或事实证书不完整时保留为 evidence frontier。只在所有 rollout 的完整证据均通过时判定掌握；同 scene 的其他已评估 variant 未掌握时不能退休整个 scene。regret 编辑的 `hold` 保留当前 subset 和 player count，并提供相邻 contrast。自博弈 proposer 仍能自由选题，其 learnability 和 competence feedback 同步识别证据 frontier。spy 已熟练但证据不熟练时，全局难度分布也保持。

**固定分母 loss。** `dr_grpo` 使用 `sum(mask * token_loss) / (B * C)`，不按每条实际长度重加权；与 `scale_rewards=False` 一起启用。默认 `C=max_completion_length`，新实验采用固定 `C=512`，生成上限仍为 2048。512 是待消融的梯度尺度选择，不是论文保证的最佳值；不同 C 只改变整体尺度。padding、退化组和截断 advantage mask 不改变 B；GA 仍由 Trainer 处理。现有截断/退化组继续付 KL，截断奖励仍进入组 baseline。

新增代码的 docstring 和注释保持简短，研究依据集中在本文件。

## 运行与消融

从仓库根目录，配置真实模型、训练数据和硬件路径后运行：

```bash
E=local_scripts/self_evolve/experiments
SEED=42 RUN_TAG=visual_curriculum_s42 MARKER=visual_curriculum_s42 \
  bash "$E/main/ours_visual_curriculum.sh"
MARKER=visual_curriculum_s42 LABEL=visual_curriculum_s42 bash "$E/main/eval.sh"
```

新入口继承 `ours_visual_facts.sh` 的 SFT、reward、图像尺度和预算；原实验入口默认不启用新课程、QA 或固定分母。每个实验使用独立 `RUN_TAG` / `MARKER` / `LABEL`，代码或参数变化不能续接旧 buffers/checkpoints。

| 消融 | 新入口上的环境变量 |
|---|---|
| 完整增强 | 默认 |
| 去反事实 QA | `SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION=0` |
| 去联合掌握课程 | `SELF_EVOLVE_MASTERY_MODE=spy` |
| 回原 loss | `GRPO_LOSS_TYPE=grpo GRPO_LOSS_NORMALIZATION_LENGTH=` |
| 论文默认固定分母 | `GRPO_LOSS_NORMALIZATION_LENGTH=`，使用生成预算 2048 |
| 仅 QA | `SELF_EVOLVE_MASTERY_MODE=spy GRPO_LOSS_TYPE=grpo GRPO_LOSS_NORMALIZATION_LENGTH=` |

固定使用一个种子 `SEED=42`，先与上一版 visual-facts 对照，再与原 ours / outcome-only GRPO 对照。报告每个数据集、九项平均及相对各对照的差值。保持 backbone、SFT 预算、GRPO 步数、G、评估 pixels、解码、judge 与 VLMEvalKit commit 一致。任务数和步数相同不等于 token/FLOPs 完全相同，QA 单图较便宜，应同时记录计算成本。

九项名称固定为 `MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST MMMU_Pro_10c CV-Bench-2D CV-Bench-3D`；参见 [EVAL_DATASETS.md](EVAL_DATASETS.md)。不使用这些评估集的样本或标签构造训练题。训练日志增加分任务正确率、格式率和 sample fraction，离线诊断保留 verified pass rate 与 frontier 数量。

## 如何判断稳定提分

目前验证的是数据/奖励/优化接口正确性，尚无新训练分数。合成计数与空间技能对 MMVP、BLINK、CV-Bench 的迁移是较直接的假设；MMStar、RealWorldQA 还涉及真实图像差异，ChartQA、AI2D、MMMU-Pro 更需要图表、图解与领域知识覆盖。不能由 CLEVR 内部奖励上升推出九项均提升。

下一阶段按结果选择：先在保留的训练来源 scenes 上做完整 pair accuracy 与灰图对照；若视觉依赖不足，再单独测试 Evidence-RL 风格的证据区/等面积干扰区 forward；若 ChartQA/AI2D 仍弱，扩展独立训练来源的程序化图表/图解任务。每次只加一类机制，并保留现有方法作为对照。

当前工作区缺少目标 7B 模型权重，因此未执行真实训练或九项评估。两轮真实 CLEVR CPU dry-run 每轮保持 32 GRPO rows，其中 28 game + 4 QA；这些诊断输出不代表目标模型能力或提分。

验证：全套 139 tests、81 subtests 通过；旧奖励对齐 120 checks 通过。静态与动态 loader、混合 Arrow schema、同 scene 退休、固定分母梯度和 rank/GA 数学一致性均有回归覆盖；分布式 GPU 运行尚未验证。临时 dry-run 与生成图片缓存已清理，后续 variant cache 限定在各轮 run 目录。
