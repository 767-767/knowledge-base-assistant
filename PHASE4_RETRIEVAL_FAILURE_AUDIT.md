# Phase 4.0 Hybrid+CE+RRF 检索失败审计

日期：2026-08-29  
分支：`develop`  
范围：五篇论文、53 道 benchmark 问题；只审计 `Hybrid + BAAI/bge-reranker-base + RRF` 的 `@10` 未完整案例。

## 结论先行

P0 alias/数学表面形式修正前，Hybrid+CE+RRF 的 `@10` 完整 required-fact coverage 为 `0.698`，共有
`16/53` 道题未完整覆盖。修正 alias 后在**不改变最终排名**的情况下，coverage 更新为 `0.717`，未完整
案例为 `15/53`；剩余失败仍不能统归为“reranker 不好”：

- 有些事实在现行解析文本中存在，但 benchmark 的中文/数学/连字符表面形式没有匹配到；
- 有些事实进入 Hybrid 的 top-50 候选，却在 cross-encoder 排序或最终 RRF 融合中被丢掉；
- 有些事实在目标论文的解析块中存在，但根本没有进入 Hybrid top-50；
- 个别通用词可能在参考文献或其他非证据块中产生词法假阳性。

本阶段已落地经过 PDF 核对的 benchmark alias 修正，以及防止多论文 section expansion 污染的安全边界；
weighted RRF 和 section expansion 的检索增强仍是可选对照，没有把任何单一融合策略设为默认。

## 审计输入和可复核性

- 基准结果：`/private/tmp/sci_rag_benchmark_hybrid_reranker_20260829.json`
- 阶段重建结果（candidate-50、reranked-50、final fused-10）：`/private/tmp/phase4_failure_retrieval_stages.json`
- 事实在现行解析块中的位置：`/private/tmp/phase4_failure_fact_locations.json`
- alias 修正后按相同最终排名重算的结果：`/private/tmp/sci_rag_benchmark_hybrid_reranker_aliasfix_v2_20260829.json`
- 同 candidate-50、同 cross-encoder、仅关闭最终 RRF 的 CE-only 对照：`/private/tmp/sci_rag_benchmark_hybrid_reranker_ceonly_20260829.json`
- 同 candidate-50、同 cross-encoder 的加权 RRF 对照：`/private/tmp/sci_rag_benchmark_hybrid_reranker_weight2_20260829.json`、
  `..._weight4_20260829.json`、`..._weight8_20260829.json`
- candidate-k=80 的同条件对照：`/private/tmp/sci_rag_benchmark_hybrid_reranker_candidate80_20260830.json`
- 清单：`evaluation/benchmark/manifest.json`、`evaluation/benchmark/cases.jsonl`
- 外部 PDF：桌面根目录的 `2602.08213v1.pdf`，以及 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`

本次使用同一现行解析器重新读取五篇 PDF，共得到 479 个块：DrugR 101、SciDQA 112、Scientific
Table LLM 65、MgNO 94、AlphaFold 3 107。没有读取或写入 ChromaDB，也没有调用 DeepSeek、RAGAS
或 Gradio。阶段重建固定使用 `BAAI/bge-small-zh-v1.5` 和
`BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`。

“candidate-50”表示 Hybrid RRF 进入 cross-encoder 的候选位置；“reranked-50”表示只按 cross-encoder
分数排序；“fused-10”表示当前实现将 reranked list 与原始 candidate list 再做一次等权 RRF 后的最终
top-10。阶段标签按确定性词法事实位置计算，不能替代语义或引用正确性。

## 逐题分类

| case | `@10` 缺失事实 | 阶段证据 | 主要分类 | 后续方向 |
|---|---|---|---|---|
| `drugr-11` | `Pareto`, `shortfall` | 两类事实均在 candidate-50；`Pareto`/`shortfall` 的 CE 排名约 17–23，未进最终 top-10 | CE 降权 | 比较 CE-only、加权 RRF；保留 Reasoning/SMILES 全组保护 |
| `scidqa-06` | `80%`, `25%` | 同一证据块 candidate 排名 42、CE 排名 1，但 final fused-10 仍未出现 | RRF 融合稀释 | 评估加权 RRF 或 CE-only，不能只看 CE 排名 |
| `scidqa-07` | `closed-book`, `title-abs`, `RAG`, `full-text` | 前三类证据不在 candidate-50；`full-text` 证据虽进入 candidate，CE 后仍未进 final top-10 | 候选召回 + CE 降权 | 同小节/兄弟块扩展，另测 candidate-k |
| `scidqa-10` | `第三人称` | 原文为 `third-person`；当前 alias 只有 `third person`，OpenReview/作者片段本身已进 final top-10 | 表面形式/标注 | 增加连字符 alias，并把别名修正与检索改动分开评估 |
| `table-llm-03` | `T5`, `LLaMA-2` | `FlanT5` 相关块可进入候选，另外两类模型事实未进入 candidate-50 | 多事实候选召回 | 对同一方法段做邻接块扩展，避免只返回一个架构 |
| `table-llm-09` | `SciGen`, `Explain the given table`, `参考` | `SciGen` 泛化块进入 candidate，部分被 CE 排到前列但 final RRF 丢失；精确短语所在块未进 candidate | 候选召回 + RRF 稀释 | 精确短语保护、邻接块扩展、比较融合策略 |
| `mgno-01` | `(0,1)` | 区域块 candidate 排名 36、CE 排名 9，final fused-10 丢失；边界条件块已在 final top-10 | RRF 融合稀释 | 数学区域/边界条件应成组保留 |
| `mgno-02` | `3×3` | 公式块 candidate 排名 46、CE 排名 6，final fused-10 丢失；`A/u/f` 已在 final top-10 | RRF 融合稀释 | 对公式数值和符号增加结构化证据保护 |
| `mgno-03` | `初始化`, `残差` | 原文块存在；`初始化` alias 不匹配 `Initialization`，残差块未进 candidate-50 | 表面形式 + 候选召回 | 修正 alias，同时扩展初始化/迭代邻接块 |
| `mgno-04` | `stride=2`, `反卷积`, `Backslash-cycle` | 原文使用 `stride of 2`；相关段未进 candidate-50，只有 `V-cycle` 片段进入 final top-10 | 表面形式 + 候选召回 | 修正 alias，保护整段 restriction/prolongation 证据 |
| `mgno-05` | `GELU` | GELU 块在 candidate-50（约 15–24），但 CE 后仍未进 final top-10；非线性激活块已进入 | CE 降权 | 测试同段实体共现和 CE 融合权重 |
| `mgno-06` | `d²n²`（P0 修正后已覆盖） | 同一 p6 公式块已是 CE/final top-1；现行 HTML/sup 规范化把 `d² n²` 变成带空格形式，原先未匹配无空格 gold | 数学表面形式（已修正） | 保留 `d 2 n 2` alias；不要误判为召回失败 |
| `mgno-11` | `三次`, `随机种子` | p10 可复现性块存在，但不在 Hybrid candidate-50 | 候选召回 | 对 `reproducibility`/seed 术语增加词法保护或邻接窗口 |
| `af3-04` | `随机噪声`, `去噪`, `答案分布` | 实际推理证据在 p3 块且未进 candidate；`去噪` 的一个命中来自参考文献块，CE 也未稳定保留 | 候选召回 + 词法假阳性 | 加 section/source 约束；不能仅靠字符串命中 |
| `af3-10` | `手性`, `原子重叠`, `4.4%` | p5 原文块存在但未进 candidate；`原子重叠` 原文为 `overlapping (clashing) atoms`，现有 alias 不覆盖括号形式 | 表面形式 + 候选召回 | 修正 alias，优先保留同一 stereochemistry 段落 |
| `af3-11` | `静态结构`, `动力学行为` | p6 限制段未进 candidate；`closed` 相关块已进入 final top-10 | 候选召回 | 对 limitation 小节做邻接段落扩展 |

## 原文排版核对

以下结论已通过对应 PDF 页面视觉核对，说明它们不是 benchmark 自行臆造的事实：

- MgNO 印刷页 5 明确写出 `Initialization`、基于 residual 的迭代、`stride of 2`、`de-convolution`、
  `Backslash-cycle` 和 `V-cycle`；
- MgNO 印刷页 6 明确写出参数量 `O(log(d)n²)`、复杂度 `O(d²n²)`，并说明多通道 `W_Mg` 不包含非线性激活；
- SciDQA 印刷页 3 使用 `third-person point of view`；
- AlphaFold 3 印刷页 5 使用 `overlapping (clashing) atoms`，并报告 chirality violation rate `4.4%`；
- AlphaFold 3 印刷页 6 说明 static structures、dynamical behaviour，以及 cereblon 的 apo/holo 都预测为
  closed conformation。

## 汇总发现

### P0 alias 修正的独立影响

本阶段补充了 5 类经 PDF 核对的表面形式：`third-person`、`Initialization/initialize`、`stride of 2`、
`overlapping (clashing) atoms` 和解析器产生的 `d 2 n 2`。在保持 candidate、CE 和 final RRF 排名完全
不变的前提下，指标变为：

| 版本 | fact macro | fact micro | 完整覆盖 | 未完整案例 |
|---|---:|---:|---:|---:|
| alias 修正前 | 0.785 | 0.776 | 0.698 | 16/53 |
| alias 修正后 | 0.794 | 0.782 | 0.717 | 15/53 |

这 0.019 的完整覆盖变化属于评估匹配修正，不能解释为检索器或 reranker 的收益。后续检索实验应以
alias 修正后的 `0.794/0.782/0.717` 作为新控制组。

### CE-only 受控对照

在相同解析块、相同 Hybrid candidate-50、相同 cross-encoder revision 下，只关闭“与原始 candidate
排名再次 RRF”的步骤，得到：

| 最终排序 | reference recall | target doc | page hit | Table N hit | fact macro | fact micro | 完整覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + 原始排名 RRF | 0.657 | 0.981 | 0.905 | 0.944 | 0.794 | 0.782 | 0.717 |
| CE-only | 0.695 | 1.000 | 0.810 | 0.944 | 0.792 | 0.782 | 0.717 |

两种策略都留下 15/53 个未完整案例，但并不是同一批：

- CE-only 修复了 `scidqa-06` 的 80%/25% 阈值，以及 `mgno-01` 的 `(0,1)` 和 `mgno-02` 的 `3×3`；
- CE-only 同时丢失了当前 RRF 能覆盖的 `scidqa-09`、`af3-08` 和 `drugr-09` 部分证据；
- 当前 RRF 保留更高的页级命中（0.905 对 0.810），而 CE-only 的 reference-context recall 更高。

结论是：CE-only 尚未证明整体优于当前 RRF；当前 fusion 的权重需要受控实验（例如加权 RRF 或
分题型策略）后再决定是否修改默认路径。

### 加权 RRF 受控对照

在相同 candidate-50、相同 cross-encoder revision 和 `rrf_k=60` 下，只提高 CE 排名列表的权重，
原始 candidate 列表权重固定为 1。这里的权重是 RRF 贡献的缩放，不是把 cross-encoder 分数与 BM25
或 dense 分数直接相加。

| 最终排序 | CE 权重 | reference recall | target doc | page hit | Table N hit | fact macro | fact micro | 完整覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE + 原始排名 RRF | 1（等权） | 0.657 | 0.981 | 0.905 | 0.944 | 0.794 | 0.782 | 0.717 |
| weighted RRF | 2 | 0.676 | 1.000 | 0.881 | 0.944 | 0.800 | 0.789 | 0.717 |
| weighted RRF | 4 | 0.676 | 1.000 | 0.833 | 0.944 | 0.788 | 0.776 | 0.698 |
| weighted RRF | 8 | 0.676 | 1.000 | 0.857 | 0.944 | 0.788 | 0.776 | 0.698 |

权重 2 只改善了 target-document、reference 和事实 macro/micro 代理，完整覆盖仍为
`0.717`，页级命中从 `0.905` 降至 `0.881`；权重 4/8 进一步降低完整覆盖和页级命中。因此当前证据
不足以把 weighted RRF 设为默认，甚至不足以证明“CE 权重越高越好”。权重 2 可保留为后续分题型或
邻接扩展实验的候选控制组，但不能替代 candidate recall 和 provenance 修复。

### 同小节扩展安全对照

应用原有的同小节扩展在单论文场景中可以补回邻接块，但直接把一个多论文集合中“最匹配小节”的
所有块前置会让通用标题和错误论文挤占 top-10。一次五论文对照（`..._sectionexp_safe4_20260829.json`）
的 `@10` fact macro/micro/完整覆盖为 `0.780/0.776/0.698`，低于 alias 修正后的当前控制组
`0.794/0.782/0.717`；这不是候选召回改善的证据。

因此当前实现增加了三道边界：只看上下文前 `context_k` 个块的来源、要求目标小节已有锚点、最多加入
6 个邻接块；当集合包含多个来源时直接跳过扩展。这样保留单论文网页回归所需的补充能力，同时不把
多论文混合集合的 section expansion 宣称为已验证的检索提升。多论文 document routing 和更细的
邻接策略留到后续阶段。

### candidate-k 50 与 80 对照

在同一 alias 修正、同一 Hybrid、同一 cross-encoder 和等权 RRF 条件下，把候选池从 50 提高到 80：

| candidate-k | reference recall | target doc | page hit | Table N hit | fact macro | fact micro | 完整覆盖 | rerank mean | rerank p95 | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50（控制组） | 0.657 | 0.981 | 0.905 | 0.944 | 0.794 | 0.782 | 0.717 | 约 2.91s | 约 3.59s | 约 2.59GB |
| 80 | 0.657 | 1.000 | 0.905 | 0.944 | 0.800 | 0.789 | 0.717 | 4.453s | 5.231s | 2.685GB |

candidate-k=80 没有提升完整覆盖或页级命中，只改善了少量事实覆盖代理；同时 mean 延迟约增加一半、
P95 超过 5 秒。因此不提高默认 candidate-k，保留 50 作为网页控制组；80 仅作为后续 document
routing 或特定题型的 opt-in 实验。

### Phase 4.2 provenance 诊断

required-fact 的词面命中本身不能证明证据来自正确论文或正确页码。现在每个离线 JSON 报告的
`overall_case_details[*].provenance[<k>]` 会逐事实保存命中块的位置：`benchmark_document_id`、
`source`、`page`、`headers`、`chunk_type`，并标记以下风险：

- `wrong_document`：命中块明确属于另一篇论文；
- `outside_gold_page`：命中块不在该用例人工标注的 `source_pages`；
- `reference_section`：命中块的 heading 属于 References/Bibliography/参考文献；
- `figure_or_caption`：块被解析为图或图注；
- `missing_page`：用例要求页码，但块没有合法页码 metadata。

这些字段是审计信息，不参与排序、不删除上下文，也不会把缺失 metadata 默认为可信。聚合字段同时
统计 `wrong_document_only_fact_count`、`outside_gold_page_only_fact_count` 和参考文献命中，便于
把“目标论文内的页码偏差”和“仅在错误论文中出现”分开。一次不带外部模型的 BM25 smoke run
（`/private/tmp/phase42_provenance_bm25.json`）在 @10 发现 88 个事实词面命中，其中 6 个只落在
标注页之外、2 个出现在 References heading；该结果用于验证诊断字段，不作为新的检索质量分数。
在同一 alias 修正后的 Hybrid+CE+等权 RRF 控制组上，@10 共有 115 个事实词面命中，6 个只落在
标注页之外，未出现“仅在错误论文中命中”或 References-only 的事实；完整诊断保存在
`/private/tmp/phase42_provenance_hybrid_ce_rrf.json`。这说明当前失败主要仍是候选召回/融合和页级
证据定位问题，而不是本次 provenance 代码改变了检索排名。

### Phase 4.3 document routing opt-in 对照

在 provenance 之后增加了保守的、离线可选 document routing：只从各来源自身的文本/metadata
建立 source-level 术语集合；只有问题中的高信号 ASCII 标识符（长度至少 4，或包含数字/特殊
标记）只属于一个来源时才路由，否则回退全库。路由器不读取 benchmark 的目标 `document_id`，
也不参与默认网页路径。

在五篇论文、53 道问题上，BM25 routing 触发 `39/53` 题且全部路由正确，指标与 BM25 全库
控制组相同。Hybrid+CE+等权 RRF routing 触发同样的 `39/53` 题，@10 为：

| 设置 | fact macro | fact micro | 完整覆盖 | 页级命中 | Table N 命中 |
|---|---:|---:|---:|---:|---:|
| 当前控制组 | 0.794 | 0.782 | 0.717 | 0.905 | 0.944 |
| routing opt-in | 0.805 | 0.796 | 0.717 | 0.929 | 0.944 |

虽然聚合代理不下降，但个别题目出现双向变化：`mgno-11`、`scidqa-07`、`table-llm-09`
得到补回，`af3-08`、`scidqa-09` 等题出现事实覆盖回退。这说明“来源路由正确”不等于“块级
证据完整”，当前不满足把 routing 设为网页默认的严格提升条件；它暂保留为后续 document
profile、查询意图和 page-aware 召回实验的控制开关。完整输出为
`/private/tmp/phase42_document_routing_hybrid_ce_rrf.json`。

### 1. Gold matcher 有独立缺口

至少 5 个案例包含现行解析文本中确实存在、但当前 required-fact 表面形式没有覆盖的事实：

| canonical fact | 原文表面 | 类型 |
|---|---|---|
| `第三人称` | `third-person` | 连字符 |
| `初始化` | `Initialization` / `initialize` | 词形 |
| `stride=2` | `stride of 2` | 介词短语 |
| `原子重叠` | `overlapping (clashing) atoms` | 括号短语 |
| `d²n²` | `d² n²` / HTML `d` + `sup` + `n` | 数学空格和上标 |

这些问题应先通过逐题 alias 或数学规范化修正，并单独报告修正前后指标；不能把修正后的数字解释成
检索器提升。

### 2. 最终 RRF 可能抵消 reranker 的局部提升

`scidqa-06`、`mgno-01`、`mgno-02` 以及 `table-llm-09` 都出现“证据块已进入 candidate、CE 排名很靠前，
但再与原始 candidate 等权 RRF 后掉出 top-10”。这说明当前 fusion 不是单纯的 reranker 质量问题，
需要比较（加权 RRF 的第一轮已完成）：

- CE-only top-10；
- 只在 candidate-50 内对原 rank 加权的 RRF；
- CE 与原始 rank 的可调权重（权重 2/4/8 已完成）；
- 维持当前实现作为控制组。

在完成这组对照前，不应把 CE+RRF 直接宣称为当前最优默认策略。

### 3. Candidate-50 仍是主要上限

多篇论文的 limitation、reproducibility、方法细节和多事实问题在 top-50 就未出现。reranker 不可能恢复
未进入候选池的事实；盲目增大 `n_results` 或只调整 CE 不能替代候选召回诊断。

### 4. 词法覆盖存在来源假阳性

AlphaFold 3 的 `去噪` 可在参考文献标题中出现，而真正回答 inference 过程的 p3 段落没有进入候选。这说明
required-fact coverage 只能作为诊断代理；Phase 4.2 已把 section/source/page 位置和风险标记写入离线
报告，但这些标记仍需人工复核或更严格的自动门槛，不能自动改写检索排名。

## 后续验收门槛

### P0：先修评估自洽性，再动检索

1. 已为上述 5 类表面形式补充经过 PDF 核对的 alias/数学形式；修正结果已单独记录，并明确标注这是
   matcher 变化，不是检索变化。
2. 为每个 failure 保存 candidate-50、CE-50、final-10 的 chunk ID、页码、section/type，避免只看最终
   `missing_facts`。
3. 已完成最小 provenance 检查：逐事实保存 source/page/section/type，并隔离跨论文、页码偏差、参考文献、
   图注和缺失页码风险；该字段只用于审计，不改变检索排名。

### P1：受控比较融合和候选池

1. CE-only 与当前 RRF 的同条件对照已完成；weighted RRF 权重 2/4/8 也已完成，均未达到“整体
   严格优于当前 RRF”的门槛，不把 CE-only 或 weighted RRF 设为默认。
2. candidate-k=80 的同条件对照已完成：完整覆盖不变而 P95 达到 `5.231s`，不满足默认切换条件；
   保留 50，并继续记录 CPU 延迟和峰值内存。
3. 对单来源多事实方法题加入同小节邻接块扩展，但继续保留 Table N 的跨表保护；多来源集合已有
   保守 document routing 对照，但由于存在题目级回退，仍不能把 routing 或扩展切换为网页默认。

最低回归门槛（相对 alias 修正后的当前结果不得下降）：

- `@10` fact macro `0.794`；
- `@10` fact micro `0.782`；
- 完整覆盖率 `0.717`；
- 页级命中 `0.905`；
- Table N 命中 `0.944`。

建议目标是在修正 matcher 和融合策略后将完整覆盖率提高到至少 `0.75`，但在 16 个案例完成根因修正
前不把该目标当作已证明的必然结果。

### P2：暂缓先进方向

在上述失败闭环完成前，不推进多模态 OCR/VLM、工具调用或 Graph-RAG。它们不能解决当前已经明确的
candidate recall、表面规范化和 RRF 融合问题。

## 停止条件

- 任一 Table N 问题出现跨表取数或来源 metadata 不一致；
- CE-only 看似提升，但加权融合导致引用/表号或跨论文污染；
- candidate-k 扩大后 CPU p95 高于 candidate-50 控制组约 `3.59s` 且没有完整覆盖收益（candidate-k=80
  实测 `5.231s`），或峰值内存超过约 `2.69GB`；
- 经过两种以上受控融合策略仍无法解释失败来源，此时停止调参，转入解析/标注复核。

本报告只覆盖检索上下文代理，不证明生成答案正确率、引用正确性、RAGAS 指标或跨论文泛化。
