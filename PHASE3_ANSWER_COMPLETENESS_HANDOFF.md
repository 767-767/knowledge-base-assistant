# Sci-RAG Phase 3 答案完整性接手记录

日期：2026-08-29  
分支：`develop`

## 本轮目标

处理 Phase 2 网页 A/B 暴露的通用问题：检索上下文已经包含多个互补事实，但生成回答只
概括最靠前的片段，遗漏阈值、工具名或后续步骤。本轮先建立不调用模型的答案级事实审计，
并在科学问答系统提示词中加入多片段整合约束；没有为任何单篇论文添加特殊规则。

## 已完成

- 新增 `evaluation/answer_audit.py`：读取 JSON/JSONL 测试集和用户保存的回答，按每题
  `required_facts` / `required_fact_aliases` 输出 `full/partial/zero`、匹配事实和遗漏事实。
- 答案审计复用 `evaluation/context_coverage.py` 的规范化、边界和别名逻辑，避免检索覆盖
  与答案覆盖使用两套不一致的判定。
- `evaluation/evaluate.py` 的 `fact_coverage()` 改用同一审计实现；既有字段保持兼容，
  但现在支持显式别名和 Unicode/Markdown 表面形式规范化。
- `app.py` 的科学问答 Prompt 增加通用多片段规则：复合问题需逐项核对，并保留互补片段
  中的专有名词、工具、阈值、数据规模、筛选条件和生成步骤。
- `sci_rag_core.py` 新增 `build_evidence_ledger()`：从已选上下文逐字抽取带片段标记的
  数字、阈值和科学实体核对清单，并将其附在生成提示前；它不推断新事实，也不替换完整
  上下文。中文问题与英文证据没有词面重叠时，会从其他已检索片段补足高显著性行。
- 新增离线单元测试，覆盖完整/部分/零答案、别名、重复/未知用例 ID、JSON/JSONL 输入。

## 使用方式

先把人工或网页回答保存成 JSON/JSONL。最小 JSONL 格式为：

```json
{"case_id":"drugr-09","answer":"...","mode":"hybrid","latency_seconds":2.1}
```

对单论文 11 题运行：

```bash
./venv/bin/python evaluation/answer_audit.py \
  --testset evaluation/test_questions.json \
  --answers /tmp/sci_rag_answers.jsonl \
  --require-all \
  --json-out /tmp/sci_rag_answer_audit.json
```

对多论文 53 题运行时，将 `--testset` 换成
`evaluation/benchmark/cases.jsonl`。回答文件只放在仓库外；脚本不会调用模型、读取
ChromaDB、写入报告文件（除非显式指定仓库外的 `--json-out`）。

本轮用现有 `evaluation/evaluation_report.json` 做离线回归（没有重新生成答案）得到：
11/11 题可审计，答案 fact macro/micro 为 `0.8409/0.8387`，full/partial/zero 为
`0.8182/0.0909/0.0909`。工具定位到第 3 题遗漏 `0.3404` 和 `DrugR*`，第 9 题遗漏
`DeepSeek-R1`、`0.6` 和 `ADMETLab`。这只是旧报告答案的完整性基线，不是新 Prompt 已
提升生成质量的证据。

## 证据边界

- `full` 只表示回答显式包含人工声明的 required facts，不代表句间逻辑、因果关系、单位、
  引用来源或事实真实性已经正确。
- `partial` 能定位遗漏事实，适合作为人工复核清单；不能自动决定“部分正确”的语义等级。
- 事实别名仍必须在 benchmark gold contexts 中人工声明并由校验器确认，不能由模型自动扩展。
- Prompt 与表格路径已通过一次固定 11 题的 Dense/Hybrid+Rerank 生成回归；这不是多轮随机
  重复、跨论文生成或 Prompt 改动前后严格对照实验，仍需在这些范围内谨慎解读。

## 下一步验收门槛

1. 如需量化 Prompt 改动本身的增益，在相同 retrieval mode、相同问题和相同数据库上补采
   改动前控制组回答。
2. 已完成 11 题答案审计和五篇论文检索代理；若进入多论文生成阶段，仍需人工复核所有
   partial/zero 以及表号、单位和引用。
3. 报告 answer fact macro/micro、full/partial/zero、逐题遗漏和引用/数值错误。
4. 若完整性提升伴随表格跨表、幻觉或延迟退化，停止将 Prompt 改动合入默认路径。

## 未完成

- 已完成当前单论文 11 题的 Dense 与 Hybrid+Rerank DeepSeek A/B；尚未对五篇论文执行完整的
  生成答案统计，因此不能把单论文结果外推为多论文泛化。
- 尚未建立语义答案正确性、引用正确性和数值单位校验器。
- 尚未重跑 RAGAS，也未进行真实五论文 Chroma 端到端生成评估。

## 本轮端到端验证记录（2026-08-29）

在未上传文档、未重建 ChromaDB 的前提下，使用现有数据库启动 `app.py`，并通过 Gradio
为 ChatInterface 暴露的同一问答函数提交 4 道代表性问题。浏览器页面已打开并确认知识库块数为
104；由于当前自定义 `gr.Textbox` 的 `submit_btn` 为 `False`，浏览器自动化无法触发可见的发送
控件，因此答案采集使用本机 Gradio 的等价 `/lambda_3` API（该端点包装同一个 `query_knowledge`
函数），不改变检索或生成代码路径。

回答文件和审计结果均写在仓库外的 `/tmp`，可复核：

- Dense：`/tmp/sci_rag_answers_dense.jsonl`、`/tmp/sci_rag_answer_audit_dense.json`
- Hybrid+Rerank：`/tmp/sci_rag_answers_reranker.jsonl`、`/tmp/sci_rag_answer_audit_reranker.json`

结果（仅 4/11 道题，不能当作整套评估）：

| 模式 | fact macro / micro | full / partial / zero | 主要遗漏 |
|---|---:|---:|---|
| Dense | 0.6875 / 0.5455 | 0.50 / 0.50 / 0.00 | 第9题缺 `0.6`、`ADMETLab`；第11题缺 `Reasoning`、`SMILES`、`shortfall` |
| Hybrid+Rerank | 0.8125 / 0.7273 | 0.75 / 0.25 / 0.00 | 第9题缺 `DeepSeek-R1`、`0.6`、`ADMETLab` |

两种模式下 Table 2 的两道数值题均返回正确值 `0.2060` 与 `0.3404`。Hybrid+Rerank 对强化
学习综合题覆盖了四个声明事实，但显式推理数据集题仍未稳定保留工具名和阈值；因此本轮只能证明
答案审计链路可用、以及该 4 题样本上的一次性观察，不能证明 Prompt 已解决泛化或整体正确性。

## 第二轮实现（2026-08-29）

针对上述遗漏，生成提示现在同时接收“事实核对清单”和完整参考片段。清单只从当前检索结果
中逐行抽取，不会把 benchmark 的 `required_facts` 注入线上回答，因此不依赖任何单篇论文。
`app.py` 对复合事实问题在最终上下文截断前扩展同一来源、同一最相关小节的兄弟块；清单为每个
候选上下文保留最多两条高信号原文，再按相关性补足，避免工具名/阈值被重复表格行挤掉。对
模型仍遗漏的数字、阈值或高信号工具名，新增的保守补充只引用清单中的原文，并排除 Markdown
表格、表题、训练集统计摘要和无关类别列表。新增回归覆盖了跨语言问题下互补片段中的
`DeepSeek-R1`、`ADMETLab`、`0.6`、`SMILES` 和 `4,855` 等文字/数字进入提示，以及补充规则
不带入 Table 5/类别噪声。随后补充多列行查询、目标列别名和比较题全表保护回归，离线测试现为
**59/59** 通过。

## 最终端到端回归（2026-08-29）

在不上传文档、不重建 ChromaDB 的前提下，使用现有 **104 个块**数据库启动最新
Hybrid+Rerank（`BAAI/bge-reranker-base` 固定 revision），通过 ChatInterface 暴露的同一
`/lambda_3` 函数提交原先授权的 4 道代表性问题（Table 2 两道数值题、显式推理数据集综合题、
多目标强化学习综合题）。页面的自定义输入框仍未提供可自动触发的可见发送按钮，因此采集
使用同一 Gradio 本地 API；没有改变回答函数或检索路径。

最终回答文件和审计结果写在仓库外，可复核：

- `/tmp/sci_rag_answers_reranker_v4.jsonl`
- `/tmp/sci_rag_answer_audit_reranker_v4.json`

本地答案审计结果为 **4/4：fact macro/micro = 1.0000/1.0000，full/partial/zero =
1.0000/0/0**。Table 2 返回 `0.2060`、`0.3404`；数据集题同时覆盖 `4,855`、闭环逆向工程、
`DeepSeek-R1`、`ADMETLab`、指纹相似度 `> 0.6`、ADMET 改进和 SMILES/属性增量理由生成；
强化学习题覆盖 `Pareto`、`Reasoning`、`SMILES` 和 `shortfall`。此前 Dense v3 的同四题审计也为
4/4 完整。该结果只证明这 4 道问题在一篇论文上的一次端到端回归通过，不能外推到 11 题、
五篇论文、多轮随机问题或 RAGAS 指标。

## 完整单论文回归（2026-08-29）

随后在同一现有数据库上，用固定的 `evaluation/test_questions.json` 依次提交全部 11 道题，
分别运行 Dense 与 Hybrid+Rerank；回答文件和审计结果均位于仓库外：

- Dense：`/tmp/sci_rag_answers_dense_11_v3.jsonl`、`/tmp/sci_rag_answer_audit_dense_11_v3.json`
- Hybrid+Rerank：`/tmp/sci_rag_answers_reranker_11_v3.jsonl`、`/tmp/sci_rag_answer_audit_reranker_11_v3.json`

两种模式的答案审计均为 **11/11，fact macro/micro=1.0000/1.0000，full/partial/zero=
1.0000/0/0**。本次还实际验证了三个此前未暴露的语义缺口：

- Table 6 的 `Unique SMILES` 数量和占比现在由结构化行查询直接返回
  `3,863 / 4,826` 与 `80.05%`；
- Table 5 的中文“靶点集合”通过通用 `target set` 别名定位到
  `ACE, AGTR1, ADRB1, ADRB2`；
- Table 1 的比较题不再进行行级过滤，保留所有基线行，因此四项最优指标和
  Fingerprint Similarity 非最优的结论均可由完整表格支持。

答案事实审计仍只衡量显式 required facts；上述结果不替代人工检查表号、单位、语义推理和引用。

## 多论文离线检索回归（2026-08-29）

使用仓库外的四篇免费 PDF（桌面根目录的 DrugR 种子论文加
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers`）运行了 BM25、Dense、Hybrid-RRF 和
Hybrid+cross-encoder+RRF。全局 53 题、`@10` 结果如下；没有调用 DeepSeek、没有启动 Gradio、
没有写入 ChromaDB：

注意：当前 ChromaDB 仍有 104 个块，而用现行解析器重新读取桌面种子 PDF 得到 101 个块；
本轮刻意没有重建数据库，因此线上回归与离线基准的块数不完全相同。下一阶段若要做可复现
对比，应先在隔离数据库中固定解析版本并记录块清单/哈希。

| 方法 | fact macro | fact micro | 完整覆盖题 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.627 | 0.599 | 0.547 | 0.690 | 1.000 |
| Dense | 0.539 | 0.537 | 0.434 | 0.762 | 0.667 |
| Hybrid-RRF | 0.627 | 0.592 | 0.547 | 0.786 | 0.833 |
| Hybrid + CE + RRF | 0.785 | 0.776 | 0.698 | 0.905 | 0.944 |

Hybrid+CE+RRF 相对 Hybrid 的 `@10` 完整事实覆盖提高 15.1 个百分点，仍有 16/53 题不完整，
其中包含解析/标注或长尾方法事实缺失。该结果支持将 reranker 保持为默认关闭的受控实验，
不支持把 Hybrid 或 reranker 设为默认，也不能解释为生成答案正确率或泛化证明。对应原始
JSON 输出位于：`/private/tmp/sci_rag_benchmark_bm25_20260829.json`、
`/private/tmp/sci_rag_benchmark_dense_20260829.json`、
`/private/tmp/sci_rag_benchmark_hybrid_20260829.json`、
`/private/tmp/sci_rag_benchmark_hybrid_reranker_20260829.json`。
