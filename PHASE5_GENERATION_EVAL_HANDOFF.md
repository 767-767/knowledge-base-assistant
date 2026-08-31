# Phase 5.0 生成评估输入契约

日期：2026-08-30  
分支：`develop`

## 本阶段目标

在调用任何模型 API 之前，先保证多论文答案采集和离线审计使用同一套、可复核的用例
契约。该阶段不生成答案、不启动 Gradio、不读取 ChromaDB，也不运行 RAGAS。

## 已完成

- `evaluation/answer_audit.py` 现在可以直接读取
  `evaluation/benchmark/cases.jsonl`；遇到 `source_testset/source_case_id` 指针时，会
  通过同目录 `manifest.json` 展开完整用例。
- `answer_required_facts` 可选，用于声明答案必须显式出现的事实子集；默认仍使用
  `required_facts`。这样“检索需要验证实体”与“答案无需重复问题中已给出的实体”可以分开。
- `answer_fact_aliases` 可选，只用于答案表面形式（例如中文术语与英文事实的对应），
  不会改变检索 required-fact coverage。
- 中文事实匹配不再套用 ASCII `\w` 边界，因此 `块`、`参考`、`初始化` 等词可在中文
  自然句中正确命中；ASCII 数字/标识符仍保留边界，避免 `42` 命中 `420`。
- 测试集重复 `case_id` 会被拒绝，避免答案审计静默覆盖记录。

## 验收证据

```text
evaluation/benchmark/cases.jsonl 展开为 53 道题
使用每题 ground_truth 作为占位答案时：53/53 full
离线测试：71/71
```

仓库外已有的 11 题历史答案文件也用新审计器复核通过：Dense 和 Hybrid+Rerank 均为
`11/11 full`（`fact macro/micro=1.0000/1.0000`）。这只是输入契约兼容性检查；没有重新
调用模型，也不代表五篇论文的生成质量已验证。

新增 `evaluation/compare_answer_runs.py` 作为后续 A/B 采集的离线对照入口。它要求 Dense
与候选模式覆盖完全相同的 case ID，输出逐题 coverage delta、遗漏事实和
`full/partial/zero` 状态转移，避免用不同题目子集制造虚假的聚合提升。该工具不调用模型。

新增 `evaluation/review_answers.py` 作为人工复核契约：它可从已保存回答生成空白 JSONL 模板，
并校验/汇总 `correct`、`partial`、`incorrect`、`unanswerable` 判断，以及表号、单位、公式、
引用四个独立维度。人工标签与 lexical fact coverage 并列保存，不会把词面命中自动提升为
语义正确。

新增 `evaluation/ragas_preflight.py` 作为 RAGAS 报告预检器：它只检查报告与测试集的 case ID、
答案、ground truth、上下文 trace、指标数组和均值是否自洽，并显式标注无法从 artifact 证明的
ground-truth 使用、生成/评估上下文一致性和生成模型记录。旧报告缺少这些 trace 时仅作警告，
使用 `--require-trace` 才会将其提升为阻塞错误。

同时，`evaluation/evaluate.py` 现在在未来的评估报告中保存完整的 generation context trace、
RAGAS 实际评估的 top-N 前缀、reference contexts 和 generation model。预检器会验证评估前缀
与生成 trace 的顺序一致性；本次没有重新运行 RAGAS，也没有覆盖现有历史报告。

占位答案只验证标注契约自洽，不代表模型已经回答正确。真实答案仍需人工或网页采集，
并应保存在仓库外的 JSONL 文件中。

## Phase 5.4 已完成：53 题生成采集与人工复核

在得到用户明确授权后，使用同一临时 Chroma 数据库和同一 53 题基准，通过网页
Gradio API 完成了 Dense 与 Hybrid+Rerank 两次完整采集（各 53/53 题）。原始回答和
审计 JSON 保存在仓库外 `/tmp`，没有覆盖原始数据库或历史 RAGAS 报告。

离线词面审计结果：Dense 的 fact macro/micro 为 `0.5818/0.5411`，Hybrid+Rerank 为
`0.6211/0.6027`；完整覆盖率分别为 `0.4528/0.4906`。Hybrid 有 10 题覆盖提升、
5 题退化，平均延迟从约 `1.60s` 增至 `4.32s`。

逐题人工语义复核结果：Dense `22 正确 / 9 部分 / 22 错误`，Hybrid+Rerank
`22 正确 / 13 部分 / 18 错误`。因此 Hybrid 主要减少了明显遗漏，但没有提高本次
复核的完全正确率；表格数值映射、单位、复合问题和“资料未提供”的误拒答仍需修复。
逐题标签见 `evaluation/benchmark/reviews_dense_53.jsonl` 和
`evaluation/benchmark/reviews_hybrid_reranker_53.jsonl`，完整分析见
`PHASE5_GENERATION_BENCHMARK_REPORT.md`。

## Phase 5.5 已完成：RAGAS 辅助对照

在用户授权外部模型调用后，使用已保存答案和同一临时数据库重建的完整上下文 trace，
分别运行了 53 题 Dense 与 Hybrid+Rerank 的 RAGAS 0.4.3 对照。两份报告均为 53/53
题，并通过 `evaluation/ragas_preflight.py --require-trace` 的结构预检（0 error、2
warnings）；预检明确记录三项指标的 input columns 不含 `reference`，因此它们不使用
ground truth，另一个警告是无法仅凭字段证明运行时上下文完全相同。逐题 trace 比较确认 evaluated contexts 是
generation contexts 的稳定前缀。

| 指标 | Dense | Hybrid+Rerank |
| --- | ---: | ---: |
| Context Relevance（有效样本） | 0.5053（47/53） | 0.7402（51/53） |
| Faithfulness（有效样本） | 0.7545（50/53） | 0.7458（49/53） |
| Answer Relevancy | 0.5546（53/53） | 0.5447（53/53） |

RAGAS 0.4.3 的三项 declared input columns 均不含 `reference`，因此这些指标实际上不使用
`ground_truth`；ground truth 仅在报告和本地事实审计中保留。评估期间出现少量 timeout/IncompleteOutput 及“只返回 1 个 generation”兼容性
警告，因此有效样本数与评判模型必须随均值保留。Context Relevance 的提升不能单独
证明检索质量提升；同一 DeepSeek 同时承担生成与评判，Faithfulness/Answer Relevancy
未提升，且确定性事实和人工语义结果仍是主要结论。临时报告位于 `/tmp`，未覆盖历史
`evaluation/evaluation_report.*`。

## 采集记录的最小格式

```json
{"case_id":"scidqa-01","answer":"...","mode":"hybrid-rerank","latency_seconds":2.1}
```

答案文件应覆盖每个 case 一次，不得重复 case ID。完成后运行：

```bash
./venv/bin/python evaluation/answer_audit.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/sci_rag_answers.jsonl \
  --require-all \
  --json-out /tmp/sci_rag_answer_audit.json
```

## 证据边界

- `full` 只表示答案包含声明的答案事实或显式答案别名，不证明语义、因果、单位、表号、
  引用或图片理解正确。
- `answer_fact_aliases` 必须由人工维护；不能让模型自行扩展别名。
- 只有在固定检索配置、固定问题和相同数据库下采集 Dense/Hybrid 对照后，才可比较答案
  完整性；仍需人工复核 partial/zero、表格和引用，再决定是否运行 RAGAS。
- A/B 对照报告中的 aggregate delta 只表示声明事实词面覆盖变化；它不证明答案语义、表号、
  单位、引用或忠实度提升。
- 人工复核应在固定答案文件上完成，并对所有 partial/zero、表格、单位、公式、引用和无法回答
  的情况填写 notes；未完成人工复核前，不应把 RAGAS 结果解释为最终正确率。
- RAGAS 预检通过只表示报告 artifact 结构自洽，不表示指标本身有效或具备泛化能力；任何
  ground-truth 使用和上下文一致性结论都必须有运行时 trace 或源代码/配置证据。

## Phase 5.6 已完成：通用表格定位回归

针对 Phase 5.4 中发现的表格映射错误，`sci_rag_core.py` 增加了不依赖论文名称的结构化
定位保护：

- 实体匹配覆盖整行非数值单元格，兼容 PDF 转 Markdown 后的空首列和模型位于后续列；
- 复合实体的 `+`/`/` 空格归一化，并拒绝用短组件（如 `WikiTQ`）匹配复合实体；
- `<br>` 多值保留分隔，问题明确要求“表格表示”或“题注”时选择对应变体；
- 识别重复行之间的 `Test(C&L)`/`Test(Other)` 等分组标记；问题指定分组时只在对应
  分组取值，指定分组但当前表没有该标记时不跨表猜测。

在仓库外的 479 块隔离数据库上，定向复核了 Table 1/2/3 的此前失败问题，并额外验证
了 Test(Other) 的重复行选择、Darcy rough 限定的 L2/H1、多列 RAG/FT 别名和 MgNO
baseline/六层多行查询；没有调用外部模型，也没有重建或修改原始 `chroma_db`。
离线测试为 94/94。Phase 5.4 已保存的 53 题生成答案未因本次代码修复而自动重跑，
因此旧报告中的统计仍表示修复前的生成批次。

## Phase 5.7 已完成：网页回归与修复后生成重跑

在本机 `127.0.0.1:7861` 启动 Gradio，使用仓库外隔离 Chroma 数据库（479 个块）完成
页面加载检查；上传页显示文本块数 479，智能问答页可正常打开。通过与 ChatInterface
相同的公开 Gradio 问答入口（`/lambda_3`）验证了五类表格查询：Table 2 的
`Test(C&L)`/`Test(Other)`、Darcy rough 的 L2/H1、GPT-4o 的 RAG/FT、MgNO 基线/六层，
以及此前报告中的 DrugR Table 2 `DrugR*` 两题。返回值的表号、行、列、数值和来源均与
金标准一致；没有上传文件、重建原始数据库或修改项目数据库。网页进程已停止。

随后在同一隔离数据库重跑了 53 道题的 Dense 与 Hybrid+Rerank 生成。原始回答、上下文
trace、词面审计和 A/B 对照均保存在仓库外 `/private/tmp`：

- Dense：`sci_rag_answers_dense_53_phase56.jsonl`、`sci_rag_answer_audit_dense_53_phase56.json`；
  fact macro/micro=`0.6903/0.6438`，full/partial/zero=`56.60%/24.53%/18.87%`，
  平均延迟 `1.716s`，P95 `3.889s`，53/53 无调用错误。
- Hybrid+Rerank：`sci_rag_answers_hybrid_reranker_53_phase56.jsonl`、
  `sci_rag_answer_audit_hybrid_reranker_53_phase56.json`；fact macro/micro=`0.7201/0.6918`，
  full/partial/zero=`62.26%/20.75%/16.98%`，平均延迟 `4.219s`，P95 `5.541s`，
  53/53 无调用错误。

相对 Phase 5.4，Dense 覆盖提升 8 题、退化 0 题；Hybrid+Rerank 提升 8 题、退化 2 题。
这仍是词面覆盖，不等于语义正确率；例如 Hybrid 的 AF3-06 将 `CASP15 RNA` 错答为 28，
说明必须继续人工复核。修复后 RAGAS 重跑曾启动，但第一个评估任务超过 180 秒并出现
超时/不完整输出，已主动中止，未生成新的 RAGAS 报告，也未覆盖历史报告。历史 RAGAS
数字仍只能作为 Phase 5.5 的辅助证据。

对两批新答案已完成单人逐题复核：Dense 正确/部分/错误=`28/8/17`，Hybrid+Rerank
=`30/10/13`。表号维度各 17/17 个表格问题正确；MgNO 需单位的题目为 2/3 正确；
`mgno-02` 公式在 Dense 错误、Hybrid 正确。复核原始 JSONL 为
`/private/tmp/reviews_dense_53_phase56.jsonl` 和
`/private/tmp/reviews_hybrid_reranker_53_phase56.jsonl`，完整汇总分别为
`/private/tmp/sci_rag_human_review_dense_53_phase56.json` 与
`/private/tmp/sci_rag_human_review_hybrid_reranker_53_phase56.json`。该复核是单人保守
判断，不是独立双人标注或统计显著性检验。

## 下一阶段门槛

1. Phase 5.7 两批新回答的完整单人复核已完成；仍不要把词面覆盖提升当作语义正确率，
   若要把结果用于论文或发布，应再做独立复核并报告一致性。
2. 若确有必要重跑 RAGAS，应先缩小批次或降低并发/超时，并保留完整 trace；否则沿用
   Phase 5.5 的历史 RAGAS 辅助结果，不再为同一批答案重复付费评判。
3. 运行时 document routing 已作为默认关闭的受控实验接入；在 source/page/table 命中、
   跨来源污染和人工答案标签完成对照前，不切换 Dense 默认，也不推进多模态、工具调用
   或 Graph-RAG。

## Phase 5.8 已完成：运行时 document routing（默认关闭）

为闭合多论文集合中的来源混杂风险，网页运行时现在可以通过
`SCI_RAG_DOCUMENT_ROUTING=true` 启用保守的 source 路由。路由复用已有
`DocumentRouter`：按当前 Chroma 中每个 `source` 聚合文本和表格元数据，仅当问题中的
高信号 ASCII 标识符唯一属于一个来源时才添加 `where={"source": ...}`；没有唯一标识符、
或问题同时指向多个来源时回退全库。Hybrid 的 BM25 候选也同步限制到该来源，复合问题的
同小节扩展仍仅在整个集合只有单一 source 时启用，routing 不放宽这一安全边界。默认值
保持 `false`，因此原有单论文 Dense/Hybrid 行为不变。

新增离线回归覆盖：唯一来源过滤、Hybrid 词法候选不重新引入其他来源，以及歧义问题不
添加过滤。全套离线测试为 98/98 通过；本轮未重建数据库、未
调用外部模型、未启动网页，也未执行 git add/commit/push。该开关仍是受控实验，必须用
固定多论文基准报告 source/page/table 命中和人工答案正确性，不能仅凭路由触发率宣称
泛化提升。

对项目当前 `chroma_db/chroma.sqlite3` 的只读检查显示集合仍为 104 个块，`source` 全部是
`2602.08213v1.pdf`（96 个 text、8 个 table）。因此 routing 在当前单论文数据库中不会
产生可观察的跨来源效果；只有在用户明确建立多论文索引后，才应按上述 benchmark 方式验证。

使用同一套外部五论文 PDF、同一解析器和固定 cross-encoder 的离线对照结果已保存于
`/private/tmp/sci_rag_benchmark_hybrid_reranker_routing_phase58_genericstop.json`：39/53 题触发路由，
39/39 路由正确，0 次误路由；@10 目标论文命中=`1.000`、页级命中=`0.929`、事实
macro/micro=`0.805/0.796`。相对未路由的 alias-corrected 控制组
`0.981/0.905/0.794/0.782` 有诊断改善，但完整事实覆盖仍为 `0.717`（15/53 题不完整），
Table N 命中仍为 `0.944`。因此 routing 只能作为来源隔离控制，不能宣称提升答案正确率或
泛化能力，也不应切换为默认。

另外运行了 routing+section-expansion 组合对照
`/private/tmp/sci_rag_benchmark_hybrid_reranker_routing_section_phase58.json`；全局指标仍为
`0.805/0.796/0.717`，但 DrugR、MgNO 的分文档完整率下降。该结果支持继续保留“多来源
集合不启用同小节扩展”的安全边界。

## Phase 5.9 已完成：复合问题子查询对照（默认关闭）

新增 `query_variants()` 和 `SCI_RAG_QUERY_DECOMPOSITION`/`--query-decomposition`。
原问题始终保留，标点或保守并列词拆出的子句在同一 source route 内检索，再以 RRF 融合。
该路径不翻译问题、不读取 benchmark 金标准；离线测试现为 103/103。

固定五论文/53 题的 routing + Hybrid + cross-encoder 对照为：@10 fact macro/micro/full
`0.843/0.830/0.774`，目标论文/Table N 命中 `1.000/0.944`，页级命中 `0.905`。
相对 routing 控制 `0.805/0.796/0.717` 有改善，但页级命中低于控制的 `0.929`，仍有
12/53 题不完整。因此只达到网页 A/B 门槛，不切换默认；完整报告为仓库外
`/private/tmp/sci_rag_benchmark_query_decomposition_phase59.json`，详见
`PHASE5_9_QUERY_DECOMPOSITION_HANDOFF.md`。
