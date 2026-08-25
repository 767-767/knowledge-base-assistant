# Sci-RAG 评估报告（RAGAS）

- 论文：DrugR: Optimizing Molecular Drugs through LLM-based Explicit Reasoning and Reinforcement Learning（`2602.08213v1.pdf`）
- 生成时间：2026-08-24 21:10:49
- RAGAS 版本：0.4.3；评判模型：deepseek-chat；Embedding：BAAI/bge-small-zh-v1.5
- 测试题数：11；每题参与打分的检索上下文数：10（重排序后 top-N）
- 总耗时：188.0 秒

## 指标汇总

| 指标 | 平均分 | 有效样本 |
| --- | --- | --- |
| Context Relevance（上下文相关性） | **0.8864** | 11/11 |
| Answer Faithfulness（答案忠实度） | **0.5833** | 8/11 |
| Answer Relevance（答案相关性） | **0.8543** | 11/11 |

## 逐题明细

### 1. [table] Table 1 中 DrugR 的整体优化得分（Overall Optimization Score）是多少？

- 检索上下文：56 个（打分取前 10 个）
- 标准答案：0.2712（Table 1 所有基线中的最高值）
- 模型回答：根据参考片段 [1] 所示，Table 1 中 DrugR 的整体优化得分（Overall Optimization Score）为 **0.2712**。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：1.0000；Answer Relevance（答案相关性）：0.7634

### 2. [table] Table 2 中 DrugR*（在少量新数据上微调后）的整体优化得分是多少？

- 检索上下文：55 个（打分取前 10 个）
- 标准答案：0.2060
- 模型回答：根据参考片段 [1] 所示，Table 2 中 DrugR*（在少量新数据上微调后）的整体优化得分（Overall Optimization Score）为 **0.2060**。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：0.0000；Answer Relevance（答案相关性）：0.6254

### 3. [table] Table 2 中 DrugR* 的 Target property F1 score 是多少？

- 检索上下文：54 个（打分取前 10 个）
- 标准答案：0.3404
- 模型回答：根据参考片段 [5] 所示，Table 2 中 DrugR 的 Target property F1 score 为 **0.4364**。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：0.5000；Answer Relevance（答案相关性）：0.8730

### 4. [table] Table 1 中 LLaMA3-8B 基线模型的 Fingerprint Similarity 是多少？

- 检索上下文：57 个（打分取前 10 个）
- 标准答案：0.2843
- 模型回答：根据参考片段 [1] 所示，Table 1 中 LLaMA3-8B 基线模型的 Fingerprint Similarity 为 0.2843。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：1.0000；Answer Relevance（答案相关性）：0.7884

### 5. [table] Table 6 中 Unique SMILES 的数量和占比是多少？

- 检索上下文：55 个（打分取前 10 个）
- 标准答案：3,863 / 4,826，占比 80.05%
- 模型回答：根据参考片段 [2] 所示，Table 6 中 Unique SMILES 的数值为 3,863 / 4,826，占比为 80.05%。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：1.0000；Answer Relevance（答案相关性）：0.6810

### 6. [table] Table 5 中抗高血压药物（Antihypertensive）类别用于结合亲和力评估的靶点集合有哪些？

- 检索上下文：57 个（打分取前 10 个）
- 标准答案：ACE、AGTR1、ADRB1、ADRB2
- 模型回答：根据参考片段 [1] 所示，Table 5 中抗高血压药物（Antihypertensive）类别用于结合亲和力评估的靶点集合为：ACE, AGTR1, ADRB1, ADRB2。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：0.0000；Answer Relevance（答案相关性）：0.9541

### 7. [method] DrugR 在强化学习训练阶段采用了哪种强化学习算法？

- 检索上下文：60 个（打分取前 10 个）
- 标准答案：GRPO（Group Relative Policy Optimization）
- 模型回答：根据参考片段 [1] 和 [3] 所示，DrugR 在强化学习训练阶段采用了 **Group Relative Policy Optimization (GRPO)** 算法。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：0.5000；Answer Relevance（答案相关性）：0.9762

### 8. [method] DrugR 的训练过程分为哪几个阶段？各阶段的目标是什么？

- 检索上下文：60 个（打分取前 10 个）
- 标准答案：三个阶段：(1) 持续预训练（CPT）——用知识库、文献记录等领域语料增强模型的化学知识，同时混入通用与领域指令数据防止灾难性遗忘；(2) 监督微调（SFT）——用反向数据工程构造的任务数据训练模型识别不良性质并设计更好分子的能力；(3) 强化学习（RL）——多目标强化学习进一步优化。
- 模型回答：根据参考片段，DrugR 的训练过程分为三个阶段，各阶段目标如下：

第一，**持续预训练（CPT）**：该阶段旨在通过领域语料（如知识库和文献记录）增强模型的化学知识，同时保留通用语料以维持广泛的语言能力和指令遵循行为，防止灾难性遗忘。根据参考片段 [9] 和 [5] 所示。

第二，**监督微调（SFT）**：该阶段使用通过逆向数据工程构建的任务数据，训练模型识别不良性质并设计更好的分子结构。具体采用两阶段课程：先仅监督 SMILES 生成以聚焦分子语法和稳定结构表示，再从该检查点继续联合监督推理过程和优化后的 SMILES 字符串，以对齐解释与分子优化的因果结构。根据参考片段 [9]、[24] 和 [7] 所示。

第三，**强化学习（RL）**：该阶段采用 Group Relative Policy Optimization（GRPO）算法，结合多粒度奖励函数，对模型输出的不同字段（如特征定位准确性、推理多样性、设计有效性、药效保留等）分配不同奖励内容，并通过 Pareto 改进策略实现多奖励问题间的自平衡。根据参考片段 [3] 和 [6] 所示。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：N/A；Answer Relevance（答案相关性）：0.9972

### 9. [method] DrugR 的显式推理数据集（explicit-reasoning dataset）包含多少个样本？推理标注是通过什么管道构建的？

- 检索上下文：60 个（打分取前 10 个）
- 标准答案：4,855 个样本。通过闭环反向数据工程（reverse-engineering）管道构建：从成熟药物分子出发，由 DeepSeek-R1 提出结构相似候选分子，经指纹相似度过滤（保留相似度 > 0.6 的分子对），用 ADMETLab 预测 ADMET 性质，按统一阈值规则筛选出改进的分子对，再由化学适配语言模型根据性质增量与 SMILES 反向生成机理性理由。
- 模型回答：根据参考片段 [13] 所示，DrugR 的显式推理数据集包含 **4,855** 个样本。

根据参考片段 [13] 所示，推理标注是通过一个**闭环逆向工程管道**（closed-loop reverse-engineering pipeline）构建的。该管道从已确立的药物分子出发，生成结构上可比较的候选分子，通过 ADMET 评估验证是否发生优化，然后生成推理标注。
- 得分：Context Relevance（上下文相关性）：0.0000；Answer Faithfulness（答案忠实度）：0.6667；Answer Relevance（答案相关性）：0.9364

### 10. [comprehensive] 结合 Table 1 的数据，DrugR 相比各基线模型在哪些指标上取得了最优结果？

- 检索上下文：55 个（打分取前 10 个）
- 标准答案：DrugR 在 Overall Optimization Score（0.2712）、Target property F1 score（0.4364）、reasoning LMS score（0.7712）、reasoning richness（0.9877）四项指标上均为最优；Fingerprint Similarity（0.6409）并非最高（如 mobo 为 0.9992、SFT 为 0.7992），但论文认为其保持了合理的指纹相似度。
- 模型回答：根据参考片段 [1]（即 Table 1）所示，DrugR 在以下指标上取得了最优结果：

- **Overall Optimization Score**：DrugR 为 **0.2712**，高于所有基线模型（如 GPT5 的 0.1969、DeepSeek-R1 的 0.1787、ChemDFM 的 0.1997、ether0 的 0.2149 等）。
- **Target property F1 score**：DrugR 为 **0.4364**，同样为最高值（如 GPT5 为 0.2209、DeepSeek-R1 为 0.1911）。
- **reasoning LMS score**：DrugR 为 **0.7712**，高于其他基线（如 GPT5 的 0.5721、DeepSeek-R1 的 0.7011）。
- **reasoning richness**：DrugR 为 **0.9877**，为最高值（如 GPT5 的 0.8570、DeepSeek-R1 的 0.8462）。

在 **Fingerprint Similarity** 指标上，DrugR 为 0.6409，并非最高（如 diffusion 为 0.8083、mobo 为 0.9992），但根据片段 [4] 所述，DrugR 在保持合理相似度的同时实现了最优的优化性能。
- 得分：Context Relevance（上下文相关性）：1.0000；Answer Faithfulness（答案忠实度）：N/A；Answer Relevance（答案相关性）：0.8337

### 11. [comprehensive] DrugR 的强化学习阶段如何解决多目标训练中的目标主导（objective domination）与目标饥饿（starvation）问题？

- 检索上下文：60 个（打分取前 10 个）
- 标准答案：采用两级自适应自平衡机制：(1) 样本级 Pareto 感知重加权——在 on-policy rollout 批次内按支配关系估计经验 Pareto 集，Pareto 上的轨迹获得乘性增益，被支配轨迹按到 Pareto 锚点的最小距离做指数衰减调整；(2) 批级自适应——按推理组/SMILES 组聚合的批次统计计算组级缩放因子以加强对落后组的强调，并对批次均值低于预设目标的目标施加平滑指数控制、带显式上限的通道级短fall 加权 boost，防止过度校正。
- 模型回答：根据参考片段，DrugR 的强化学习阶段通过一个**两级自适应自平衡机制**来解决多目标训练中的目标主导与目标饥饿问题，具体步骤如下：

**第一，样本级 Pareto 感知重加权（Sample-level Pareto-aware reweighting）。**  
根据片段 [50] 所示，在每个策略内 rollout 批次中，模型构建每个轨迹的目标向量，并通过支配关系估计经验 Pareto 集。Pareto 前沿上的轨迹获得乘法提升，而被支配的轨迹则根据其到 Pareto 锚点的最小距离通过指数衰减进行软调整。这产生一个样本级标量权重，统一缩放该轨迹的所有奖励通道（包括 token 级奖励）。这种统一缩放至关重要：它增加了多目标一致 rollouts 的梯度贡献，而不会扭曲轨迹内各奖励通道的比例，从而避免快速改进的通道压倒较慢但关键的通道。

**第二，批次级自适应调整（Batch-level adaptations）。**  
根据片段 [34] 所示，由于样本级 Pareto 重加权不能完全消除目标间异质学习动态造成的不平衡，模型进一步引入批次级调整。具体而言，从聚合的批次性能中计算 Reasoning 组和 SMILES 组的组级缩放因子，增加对落后组的相对强调。同时，对批次均值低于预设目标的目标应用有上限的、通道级的短缺提升（shortfall boost），使用平滑指数控制激进程度，并设置显式上限防止过度修正。这一层级缓解了因收敛速度不匹配导致的支配效应，稳定了异质奖励下的训练，并减少了对逐通道手动校准的敏感性。

**第三，Pareto 改进策略（Pareto improvement strategy）。**  
根据片段 [2] 所示，在强化学习阶段，DrugR 采用 Group Relative Policy Optimization (GRPO) 算法，并发现 Pareto 改进策略有助于在该阶段实现多个奖励问题之间的自平衡。这进一步支持了上述两级机制在整体训练动态中的有效性。
- 得分：Context Relevance（上下文相关性）：0.7500；Answer Faithfulness（答案忠实度）：N/A；Answer Relevance（答案相关性）：0.9682
