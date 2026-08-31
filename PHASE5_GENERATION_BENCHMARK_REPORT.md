# Phase 5.4 生成答案基准与人工复核报告

日期：2026-08-30  
分支：`develop`  
范围：5 篇论文、53 道固定用例；本报告不覆盖原始 11 题 RAGAS 历史报告。

## 1. 运行边界与可复现输入

- 使用与网页相同的 `app.py` 查询入口和同一批 5 篇 PDF；在仓库外的临时 Chroma
  数据库中重建，共 479 个块（101 + 112 + 65 + 94 + 107），没有改动项目原有
  `chroma_db`。
- Dense 与 Hybrid+Rerank 使用完全相同的 53 道题、同一临时数据库和同一 DeepSeek
  生成配置；每次回答保存为仓库外 JSONL。两次均覆盖 53/53 题。
- Hybrid+Rerank 使用本地缓存的 `BAAI/bge-reranker-base` 固定 revision；没有下载新模型。
- 结果文件（均在 `/tmp`，不应提交到 Git）：
  - `/tmp/sci_rag_answers_dense_53.jsonl`
  - `/tmp/sci_rag_answers_hybrid_reranker_53.jsonl`
  - `/tmp/sci_rag_answer_audit_dense_53.json`
  - `/tmp/sci_rag_answer_audit_hybrid_reranker_53.json`
  - `/tmp/sci_rag_answer_compare_53.json`

## 2. 确定性词面审计

`required_facts` 只检查答案是否出现人工声明的事实表面形式，不判断语义、数值映射、
单位、表号、因果关系或引用是否正确。

| 模式 | fact macro | fact micro | 完整 | 部分 | 零 | 平均延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.5818 | 0.5411 | 45.28% | 28.30% | 26.42% | 1.60 s | 3.61 s |
| Hybrid+Rerank | 0.6211 | 0.6027 | 49.06% | 28.30% | 22.64% | 4.32 s | 6.27 s |

Hybrid 相对 Dense 的词面变化为：macro +3.93 个百分点、micro +6.16 个百分点、
完整覆盖 +3.77 个百分点；10 题覆盖提升、5 题退化、38 题不变。它说明候选证据
覆盖有所改善，但不是答案正确率提升的证据；重排带来约 2.7 倍平均延迟。

## 3. 人工语义复核

复核依据为 `cases.jsonl` 中的金标准、gold contexts 和逐题回答；复核记录保存在：

- `evaluation/benchmark/reviews_dense_53.jsonl`
- `evaluation/benchmark/reviews_hybrid_reranker_53.jsonl`

| 模式 | 正确 | 部分正确 | 错误 | 语义完全正确率 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 22 | 9 | 22 | 41.51% |
| Hybrid+Rerank | 22 | 13 | 18 | 41.51% |

这次复核把“资料未提供”但 gold context 明确存在的回答判为错误，把遗漏关键事实、
表号/单位不确定或夹带未经证实扩展的回答判为部分正确。它是项目内一次人工复核，
不是独立双人标注或统计显著性检验。

### 按论文的语义结果

| 文档 | Dense（正确/部分/错误） | Hybrid+Rerank（正确/部分/错误） | 主要观察 |
| --- | --- | --- | --- |
| DrugR | 8/2/1 | 9/1/1 | Hybrid 修复 GRPO 全称和显式推理数据集样本数的大部分信息；目标主导/目标饥饿仍答错或拒答。 |
| SciDQA | 4/2/4 | 4/4/2 | 仍漏 7,000、85%、80%、25% 和四种配置；full-text Avg 常漏 54.03。 |
| 科学表格理解 | 2/0/8 | 2/1/7 | 表格编码、模型架构、多个数值单元格和 SciGen 目标仍是主要失败点。 |
| MgNO | 2/3/6 | 2/5/4 | Hybrid 改善了离散系统和部分网格描述，但仍有 3×3、残差、循环、Helmholtz 和多层误差错答。 |
| AlphaFold 3 | 6/2/3 | 5/2/4 | Hybrid 修复 confidence 指标，但出现样本数顺序颠倒，仍漏立体化学失败与静态/动力学限制。 |

### 必须保留的逐题问题

- `drugr-09` Hybrid 只可判为部分正确：4,855 和闭环管道基本正确，但遗漏
  fingerprint 阈值、ADMETLab 等关键细节。
- `table-llm-05` Hybrid 的 METEOR/ROUGE-1 为 0.14/0.23，不是金标准的 0.15/0.24；
  不能因 BertS=0.85 或词面覆盖高而判对。
- `mgno-10` Hybrid 把问题所需的基线/六层值答成 2.10；`mgno-07` 还应保留
  `×10^-2` 单位。
- `af3-06` Dense 的 25、38、8 顺序正确；Hybrid 将蛋白质—RNA、dsDNA、CASP15
  RNA 误排为 38、8、25，属于语义错误而非格式差异。
- 多数拒答题（例如 `scidqa-06/07`、`table-llm-09`、`mgno-03/09/11`、
  `af3-10/11`）的事实在 gold contexts 中存在，不能解释为知识库缺失。

## 4. RAGAS 辅助结果（不等同正确率）

在保存完整 generation/evaluated context trace 后，对两次固定答案运行分别执行了
RAGAS 0.4.3 的 `ContextRelevance`、`Faithfulness` 和 `AnswerRelevancy`。报告只写在
仓库外：

- `/tmp/sci_rag_ragas_dense_53.json`、`/tmp/sci_rag_ragas_dense_53.md`
- `/tmp/sci_rag_ragas_hybrid_reranker_53.json`、`/tmp/sci_rag_ragas_hybrid_reranker_53.md`

| 指标 | Dense | Hybrid+Rerank | 有效样本 |
| --- | ---: | ---: | --- |
| Context Relevance | 0.5053 | 0.7402 | 47/53；51/53 |
| Faithfulness | 0.7545 | 0.7458 | 50/53；49/53 |
| Answer Relevancy | 0.5546 | 0.5447 | 53/53；53/53 |
| Gold context recall（字符串包含代理） | 0.0566 | 0.0566 | 53/53 |
| Gold fact coverage（确定性事实） | 0.5818 | 0.6211 | 53/53 |

两份报告均通过 `evaluation/ragas_preflight.py --require-trace` 的结构预检（0 error、
2 warnings）；逐题 `evaluated_contexts` 均为 generation contexts 的稳定前缀。报告还
记录了 RAGAS 0.4.3 三项指标的 declared input columns：`reference` 不在
`ContextRelevance`、`Faithfulness` 或 `AnswerRelevancy` 的输入集合中，因此这三项
分数实际上不使用 `ground_truth`；`ground_truth` 仅作为报告字段和本地事实对照保留。
预检的第二个警告是不能仅凭字段证明生成与评估上下文在运行时完全相同（本次 trace
比较本身是相同的）。评估过程中还出现
`LLM returned 1 generations instead of requested 3` 和少量超时/IncompleteOutput，故有效
样本数必须随均值一起报告。

Context Relevance 的 +0.2349 不能单独解释为检索质量提升：它来自同一 DeepSeek 模型的
随机/重试评判，且有效样本数不同；Faithfulness、Answer Relevancy 反而略降。`Gold context
recall=0.0566` 是严格字符串包含代理，不能直接当作语义召回率。RAGAS 结果只能作为
辅助诊断，必须与确定性事实覆盖和人工语义标签并列解读。

## 5. 结论与下一步门槛

1. 当前生成器仍是研究型原型，不足以宣称“53 题正确”或“Hybrid 已改善答案正确性”。
2. Hybrid+Rerank 可作为受控对照保留，但不应仅凭词面覆盖切换为默认模式；复合问题的
   多证据拼接、拒答误判和答案校验仍未解决。表格定位/单元格映射已在 Phase 5.6 增加
   通用确定性保护；第 2 节的历史统计仍来自修复前的生成批次，修复后批次见第 7 节。
3. 任何新的 RAGAS 分数必须同时保存生成上下文、实际评估上下文、reference contexts、
   ground truth 和生成/评判模型；否则只能报告为不可审计的历史分数。
4. RAGAS 已作为辅助信号完成；后续任何重跑都必须与本报告的确定性事实审计和人工语义
   标签并列展示，不能把 RAGAS 当作正确率。

## 6. Phase 5.6 后处理验证（不回写本报告统计）

针对人工复核中暴露的表格问题，已在 `sci_rag_core.py` 增加通用的确定性表格定位保护，
并使用同一份 479 块隔离数据库做了离线定向验证。此前失败的查询现在得到：

| 查询类型 | 验证结果 |
| --- | --- |
| Table 1 FlanT5-xl 的 METEOR/ROUGE-1/BertS | `0.08 / 0.10 / 0.78` |
| Table 2 Test(C&L) WikiTQ+SQA+SciGen | `0.15(+0.07) / 0.24(+0.14) / 0.85(+0.07)` |
| Table 2 Test(Other) 同一复合行 | `0.14(+0.06) / 0.23(+0.14) / 0.85(+0.07)` |
| Table 3 表格表示的 MSE/F1 | `2.30 / 0.38` |
| Darcy rough 的 L2/H1 | `0.339 / 1.380` |
| GPT-4o RAG/FT | `46.63 / 54.03` |
| MgNO 基线/六层 L2 Error | `1.63 / 1.47` |

这一步没有重新调用 DeepSeek 或重跑 53 题，因此第 2 节的答案统计、人工标签和 RAGAS
数字保持原样；新的离线测试只证明定位函数和应用分支的回归行为，不等于 53 题
生成正确率。

## 7. Phase 5.7 网页回归与修复后生成批次

网页回归使用 `127.0.0.1:7861` 和仓库外隔离数据库（479 个块）。页面加载后，上传页
显示 479，智能问答页可打开；通过同一 ChatInterface 的公开 Gradio 入口验证了 Table 2
两种设置、Darcy rough、GPT-4o RAG/FT、MgNO 基线/六层，以及 DrugR Table 2 `DrugR*`
两题。所有确定性表格输出的表号、实体、列名、数值和来源与金标准一致，服务随后已停止。

修复后重新采集的完整答案和审计文件在仓库外：

| 模式 | 答案文件 | fact macro/micro | full/partial/zero | 平均延迟 | P95 | 调用错误 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | `/private/tmp/sci_rag_answers_dense_53_phase56.jsonl` | 0.6903/0.6438 | 56.60%/24.53%/18.87% | 1.716 s | 3.889 s | 0/53 |
| Hybrid+Rerank | `/private/tmp/sci_rag_answers_hybrid_reranker_53_phase56.jsonl` | 0.7201/0.6918 | 62.26%/20.75%/16.98% | 4.219 s | 5.541 s | 0/53 |

相对 Phase 5.4，Dense 覆盖提升 8 题、退化 0 题；Hybrid+Rerank 提升 8 题、退化 2 题。
这些是确定性词面审计，不是语义正确率。一个直接反例是 Hybrid 的 `af3-06` 把
`CASP15 RNA` 的样本数答成 28，不能只按“25/38/8 都出现在文本”判为正确。

对新批次做了逐题人工语义复核（保守地将遗漏关键事实、错误拒答、数值映射错误判为
partial/incorrect），复核 JSONL 仍在仓库外：

- Dense：`/private/tmp/reviews_dense_53_phase56.jsonl`，正确/部分/错误=`28/8/17`
  （完全正确率 `52.83%`）。
- Hybrid+Rerank：`/private/tmp/reviews_hybrid_reranker_53_phase56.jsonl`，正确/部分/错误
  =`30/10/13`（完全正确率 `56.60%`）。

表号维度两批均为 17/17 个表格问题正确；需单位的 MgNO 表格题中 2/3 正确，Darcy
rough 答案虽数值正确但未明确写统一的 `×10^-2` 单位。公式维度中 Dense 的 `mgno-02`
仍错误、Hybrid 正确。该复核由单人完成，不能替代独立双人标注或显著性检验。

同批次的 retrieval trace 为 `/private/tmp/sci_rag_generation_trace_dense_53_phase56.json`
和 `/private/tmp/sci_rag_generation_trace_hybrid_reranker_53_phase56.json`，两者均为
53/53 非空且 evaluated contexts 是 generation contexts 的稳定前缀。RAGAS 重跑因首个
任务超过 180 秒并出现 IncompleteOutput/Timeout 已中止，未产生 Phase 5.7 RAGAS 报告；
第 4 节的 RAGAS 仍是 Phase 5.4/5.5 历史批次，不能与本节新答案混用。

## 8. Phase 5.8 运行时 document routing（默认关闭）

为降低多论文知识库中不同来源候选混杂的风险，`app.py` 现在支持
`SCI_RAG_DOCUMENT_ROUTING=true`。运行时按 `source` 聚合当前 Chroma 的文本、表题和
标题元数据，只有唯一高信号 ASCII 标识符能确定一个来源时才添加 Chroma source 过滤；
歧义或跨论文问题回退全库。Hybrid 的 BM25 候选同步限制在该 source；同小节扩展仍只在
整个集合只有单一 source 时执行，routing 不放宽这一边界。默认值为 `false`，所以不改变
既有单论文 Dense/Hybrid 行为。

新增回归覆盖唯一来源过滤、来源隔离、通用词过滤和歧义回退；离线测试为 98/98。该开关是受控
多论文实验，不是答案正确性或泛化证明；正式启用前仍需用固定 benchmark 记录 source/page/
table 命中、跨来源污染和人工答案标签。

在同一五论文解析块和固定 cross-encoder 条件下运行 `--document-routing` 后，39/53 题触发
路由且 39/39 正确、0 次误路由。@10 目标论文命中=`1.000`、页级命中=`0.929`、事实
macro/micro=`0.805/0.796`，相对 alias-corrected 未路由控制组的
`0.981/0.905/0.794/0.782` 有来源定位诊断收益；完整事实覆盖仍为 `0.717`，Table N
命中仍为 `0.944`，仍有 15/53 题不完整。原始 JSON 位于
`/private/tmp/sci_rag_benchmark_hybrid_reranker_routing_phase58_genericstop.json`。这证明路由减少了
来源混杂，但没有证明答案正确率或泛化能力，故不切换默认设置。

作为组合控制，另运行了 routing+section-expansion；其全局 @10 指标仍为
`0.805/0.796/0.717`，但 DrugR、MgNO 分文档完整率下降，未形成净收益。结果位于
`/private/tmp/sci_rag_benchmark_hybrid_reranker_routing_section_phase58.json`，因此应用
实现保持多来源集合不放宽同小节扩展的边界。

## 9. 本阶段 Git 边界

新增的复核 JSONL 和本报告是可审查的项目证据；实际答案、延迟和 RAGAS 中间产物仍在
仓库外 `/tmp`。本阶段未执行 `git add`、`commit` 或 `push`。
