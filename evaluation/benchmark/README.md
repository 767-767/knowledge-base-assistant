# Sci-RAG 多论文基准集

这个目录保存不含原始论文文件的基准集清单、用例引用和离线校验工具。
论文 PDF 保留在仓库之外；`manifest.json` 记录文件名、内容 SHA-256、领域和版式标签，
因此可以在不提交版权/大型文件的情况下检查评估使用的文档是否一致。

## 当前状态

当前清单包含 5 篇论文、53 道问题：现有 DrugR 种子论文的 11 道问题，以及
SciDQA、科学表格理解、MgNO 和 AlphaFold 3 四篇论文的独立问题。新增论文的
问题包含文本、表格、公式、图注、限制和复现性等证据类型。42 个新增用例已完成
一次 PDF 逐题核对，并补充了表头、单位、caption 和公式语义；这只确认标注证据自洽，
不代表检索器或生成器已经通过评估。

## 文件格式

`manifest.json`：

- `schema_version`：清单格式版本；
- `documents`：文档 ID、文件名、SHA-256、领域和版式标签；
- `cases_path`：相对于本目录的 JSONL 用例文件；
- `minimum_documents`：进入多论文基线的最低文档数。

`manifest_expanded.json` 是一个默认关闭的扩展清单，使用 `base_manifest` 继承稳定的
`manifest.json`，并追加已核对的 Findings of EACL 2026 THINKNOTE 及 13 道题；它不会改变
默认五论文/53 题基线。

`manifest_challenge.json` 继续继承扩展清单，但只追加用于方向门控的 35 道挑战题：10 道
需要图像像素或空间关系的 `image_only` 题、20 道要求根据已召回数值执行计算的
`computation` 题，以及 5 道要求同时召回两篇论文的 `cross_document` 题。计算题的
`calculation.expected_result` 不写入 gold context，只作为后续答案/工具审计目标；跨文档题的
`additional_document_ids` 必须全部命中才算目标文档命中。

`manifest_generalization.json` 继承扩展清单并加入 TACL 2025 TANQ、Findings of EMNLP 2025 FigEx
及 16 道留出题，共 8 篇论文、82 道题。它默认关闭，只用于换论文后的表格、Figure 空间关系和跨文档
检索验证；PDF 保存在仓库外，未提交到 Git。

16 道新增题在隔离 904 块数据库上各生成两轮，最终 32/32 次调用成功；16 个 case 的上下文与
provenance 均稳定。针对 TANQ 续块/跨文档证据和 Figure 空间坐标完成通用修复，并对受影响题目
聚焦复测；人工复核记录现为 `16 correct`，详情见 `reviews_generalization_16.jsonl`。这仍不能
外推为生产正确率或通用泛化结论。

`cases.jsonl` 每行一个用例。当前用例使用引用形式：

```json
{"case_id":"drugr-01","document_id":"drugr-2602-08213-v1","source_testset":"../test_questions.json","source_case_id":1}
```

新论文可以使用内联形式，至少包含：

```json
{"case_id":"paper-b-01","document_id":"paper-b","question":"...","ground_truth":"...","required_facts":["..."],"contexts":["..."]}
```

如果标注事实与英文论文证据使用不同表面形式，只能逐事实声明显式、可复核的别名：

```json
{"required_facts":["蛋白质"],"required_fact_aliases":{"蛋白质":["proteins"]}}
```

校验器会拒绝不属于 `required_facts` 的别名键，并要求事实本身或其声明别名确实出现在
gold contexts。事实等价性不由 embedding 或模型裁判。

## 离线校验

只校验清单和用例引用，不加载 embedding、Chroma、Gradio 或 API：

```bash
./venv/bin/python evaluation/validate_benchmark.py
```

校验外部论文文件是否存在且 SHA-256 一致：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop
```

论文可以分散在多个仓库外目录，重复传入 `--papers-dir` 即可。例如现有种子论文在
桌面根目录、新论文在单独目录时：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

原始论文不应复制到项目目录或提交到 Git。

扩展清单校验：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --manifest evaluation/benchmark/manifest_expanded.json \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

挑战清单校验与离线检索：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --manifest evaluation/benchmark/manifest_challenge.json \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete

./venv/bin/python evaluation/benchmark_retrieval.py \
  --manifest evaluation/benchmark/manifest_challenge.json \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --top-k 10,50 --show-failures
```

2026-09-01 的挑战集首轮 Hybrid 对照（证据 guard、路由、查询分解和 parent-window 开启）显示：
`image_only` 在文字层为 `0/10` 完整覆盖，`computation` 的输入事实在 @50 为 `18/20` 完整，
`cross_document` 在 @10 为 `4/5` 完整、@50 为 `5/5` 完整。它们是方向门控的检索/证据信号，
尚未证明图片理解、计算答案正确性或 Graph-RAG 必要性。

2026-09-01 的派生数值路由复测使用隔离 577 块数据库和 hybrid、文档路由、查询分解、parent-window
及空间图证据：20 道 computation 题两轮共 40 次调用全部成功；逐题复核每轮 `18/20` 正确、`2/20`
因多表题未召回第二张表而缺少操作数，实际完整操作数 `36/40`。无稳定纯算术/生成失败（`0`），
因此未达到计算器门槛；两道缺口另计为 routing/row-selection 问题，下一步转 image-only 实验。

2026-09-01 的只读 Hybrid 对照（路由、查询分解、结构化表格/图/公式/限制证据及
parent-window 开启）在扩展清单 @50 达到 fact macro/micro `0.968/0.973`、完整覆盖 `62/66`；
这只是检索代理，不代表 13 道新增题的生成或答案语义正确率。

## 离线检索基线

在比较 Hybrid、RRF 或 reranker 之前，可运行标准库实现的 BM25-lite 基线：

```bash
./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --top-k 1,3,5,10
```

该命令在内存中解析四篇外部 PDF 和桌面根目录的种子 PDF，并对五篇论文建立一个全局
词法索引；默认只打印摘要，不写入项目。需要逐题 JSON 时显式增加
`--json-out /tmp/sci_rag_baseline.json`。指标含义如下：

- `target_document_hit_rate`：目标论文是否进入 top-k；这是多论文路由指标。
- `reference_context_recall`：top-k 是否覆盖人工整理的英文参考片段（词元重叠代理），不是答案正确率。
- `source_page_hit_rate`：是否命中标注页码，是解析/检索的页级代理指标。
- `table_number_hit_rate`：显式询问 Table N 时是否命中正确表号，不证明行列单元格正确。
- `required_fact_coverage_macro`：先计算每题已覆盖事实数/应覆盖事实数，再按题平均。
- `required_fact_coverage_micro`：跨题汇总已覆盖事实数/全部事实数，事实多的题权重更高。
- `full/partial/zero_fact_coverage_rate`：全部、部分、完全没有覆盖 required facts 的题占比。

使用 `--show-failures` 可列出最大 k 下未完整覆盖的题和具体遗漏事实。事实覆盖只检查目标
论文的已检索上下文，其他论文即使含相同词也不能算作证据。它仍只说明 Prompt 候选中
是否出现人工声明的事实表面形式，不说明模型是否正确理解、组合或引用这些事实。

2026-08-28 在修订 DrugR GRPO/RL 金标准片段后的 5 篇/53 题全局 BM25 基线为：

| k | fact macro | fact micro | full cases | reference context | target document | source page | table number |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.308 | 0.279 | 0.283 | 0.252 | 0.811 | 0.310 | 0.667 |
| 3 | 0.431 | 0.422 | 0.377 | 0.336 | 0.868 | 0.429 | 0.833 |
| 5 | 0.495 | 0.483 | 0.434 | 0.365 | 0.868 | 0.429 | 0.833 |
| 10 | 0.627 | 0.599 | 0.547 | 0.497 | 0.906 | 0.690 | 1.000 |

这些数字只作为后续方法比较的固定基线；其中参考片段使用人工标注，页码使用标注的
`source_pages`，不能据此声称 RAG 生成答案正确或具备泛化能力。

## 本地 dense 与 Hybrid/RRF 对比

如果项目缓存中已有 `BAAI/bge-small-zh-v1.5`，可在严格离线模式下运行：

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --top-k 1,3,5,10
```

`--retriever dense` 只跑本地向量排名，`--retriever hybrid` 用 BM25 与 dense 的 top-50
候选做 RRF（默认 `rrf_k=60`）。脚本使用 `local_files_only=True` 和
`HF_HUB_OFFLINE=1`；模型不在本地时直接失败，不会下载。

Hybrid 还包含弱词法信号保护：如果 CJK 问题在语料中没有命中任何 CJK 词元，且只命中
少于两个 ASCII 词元，则跳过该问题的 BM25 列表并保留 dense 排名。这避免中文问题对
英文论文只命中一个高频方法名时，低信息量的词法顺序反而挤掉跨语言 dense 证据。

同一基准的全局 required-fact 结果（仅作检索比较）如下：

| 方法 / k | fact macro | fact micro | 完整覆盖题 | 目标论文命中 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 @1 | 0.308 | 0.279 | 0.283 | 0.811 | 0.310 | 0.667 |
| BM25 @5 | 0.495 | 0.483 | 0.434 | 0.868 | 0.429 | 0.833 |
| BM25 @10 | 0.627 | 0.599 | 0.547 | 0.906 | 0.690 | 1.000 |
| dense @1 | 0.157 | 0.163 | 0.132 | 0.774 | 0.286 | 0.222 |
| dense @5 | 0.368 | 0.367 | 0.302 | 0.943 | 0.548 | 0.444 |
| dense @10 | 0.539 | 0.537 | 0.434 | 0.981 | 0.762 | 0.667 |
| hybrid-RRF @1 | 0.233 | 0.245 | 0.208 | 0.830 | 0.286 | 0.500 |
| hybrid-RRF @5 | 0.516 | 0.469 | 0.472 | 0.943 | 0.619 | 0.778 |
| hybrid-RRF @10 | 0.627 | 0.592 | 0.547 | 0.981 | 0.786 | 0.833 |

本次结果仍不支持直接把 Hybrid 切换为线上默认：它在 @5 的完整事实覆盖率比 BM25 高
3.8 个百分点，但 @10 两者同为 0.547，Hybrid 的 fact micro 还略低；`Table N` 命中也
低于 BM25。Hybrid @50 的候选池达到 fact macro 0.849、fact micro 0.844、完整覆盖率
0.792，说明排序仍有可利用空间；但 11/53 题即使 @50 也不完整，其中 3 题为零覆盖，
这些不能靠 reranker 单独解决。因此本地 reranker 只应作为受控实验，同时必须保留显式
表格保护，并另行处理切分/解析或查询扩展问题。该实验结果见下节。

## 可选 document routing 对照

可以在 BM25 或 Hybrid 基准命令中加入 `--document-routing`。路由器只使用每篇论文自身的
文本和 metadata：当问题中的高信号 ASCII 标识符（例如 `DrugR`、`MgNO`、`SciDQA`）只
出现在一个来源时，才把候选池限制到该来源；没有唯一标识符、或不同标识符指向不同来源时，
自动回退全库。它不读取用例的目标 `document_id`，也不改变默认网页检索。

在当前五篇论文/53 题上，BM25 routing 触发 39/53 题且无误路由，整体指标与 BM25 控制组
相同。Hybrid+CE+等权 RRF routing 的 @10 结果为 fact macro/micro `0.805/0.796`、
完整覆盖 `0.717`、页级命中 `0.929`、Table N 命中 `0.944`；相对当前控制组
`0.794/0.782/0.717/0.905/0.944`，只改善事实/页级代理，仍有个别题目回退或退化。
因此 document routing 目前保留为 opt-in benchmark 控制，不切换网页默认，也不能替代
更严格的查询意图、页码和答案正确性验证。

## 可选复合问题子查询对照

对于同时询问多个事实的长问题，可以加入 `--query-decomposition` 做离线对照：原问题
始终保留，并按中文/英文标点及保守的“与/以及/和”规则生成最多三个子查询；各子查询
在原始 document route 范围内检索，再用 RRF 融合。该过程不翻译问题、不读取
`required_facts`，也不把金标准注入查询。默认不开启，因为子查询会增加向量查询次数，
且可能改变页级命中或把不同子句的证据混合。

示例：

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --document-routing --query-decomposition \
  --top-k 1,3,5,10 --json-out /tmp/sci_rag_query_decomposition.json
```

只有在与关闭该开关的同条件报告并列比较、且 `@10` 事实覆盖、目标论文/页级命中和
Table N 命中均不下降时，才可进入网页 A/B；该代理仍不等于答案正确率或泛化能力。

Phase 6 修复了“只有句末问号不同也生成第二个查询”的问题；没有发生真实子句拆分时，
现在只运行原问题。需要比较网页实际结构化表格路径时，可增加
`--structured-table-guard`。该开关会在 normal retrieval/rerank 之后扫描 canonical table
chunks，并在已存在 document route 时保持同一来源范围；它必须与关闭开关的 raw retrieval
报告分开解释。

同条件的五论文 @10 对照达到 fact macro/micro/full `0.881/0.871/0.811`、目标论文/页级/
Table N 命中 `1.000/0.929/1.000`，相对旧查询分解只有 `scidqa-09` 和
`table-llm-10` 从 zero 变为 full，没有逐题退化。完整命令可在上例基础上增加：

```bash
--reranker-model BAAI/bge-reranker-base \
--reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
--reranker-fusion rrf --document-routing --query-decomposition \
--structured-table-guard
```

`--adjacent-context` 仅保留为负对照：它把同来源同页邻块插入前两个锚点之后，虽修复 4 题，
却使 3 道完整题退化并把页级命中降到 `0.881`，因此没有接入网页。下一步若继续处理 PDF
段落边界，应测试不占用额外 top-k 槽位的 parent/window context enrichment，而不是直接
前置邻块。

`--parent-window` 是不挤占 top-k 槽位的后续对照：只在前两个文本锚点的有效 context 内拼接
同来源、同页相邻正文，并记录 `window_chunk_indices` 与字符开销；表格、参考文献、跨页/
跨来源和已经入选的邻块均跳过。它与 retrieval ranking 指标的区别是，事实覆盖按生成器实际
可见的扩展文字计算，而 `top_results` 仍保存原锚点排名。

固定五论文 @10 从 Phase 6.0 的 `0.881/0.871/0.811` 提高到
`0.936/0.932/0.887`（macro/micro/full），来源页与 Table N 保持 `0.929/1.000`；
四题改善、零题退化。53 题共增加 62,032 个字符，因此进入默认配置前仍必须做端到端生成和
延迟 A/B，不能把该词面覆盖提升解释为答案正确率。

答案采集完成后，可直接把本文件的 `cases.jsonl` 作为
`evaluation/answer_audit.py` 的 `--testset`；审计器会通过同目录 `manifest.json` 展开
其中的指针式 DrugR 用例，最终按完整 53 题计算答案事实覆盖。答案文件仍应放在仓库外，
并由人工或网页采集产生。

## 本地 cross-encoder 重排结果

使用已缓存且固定 revision 的 `BAAI/bge-reranker-base`，对 Hybrid top-50 候选逐题打分。
为了减少纯 cross-encoder 对个别题的回退，最终策略将 cross-encoder 排名与原 Hybrid
排名再做一次 RRF：

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --top-k 1,3,5,10 \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --reranker-candidate-k 50 --reranker-batch-size 8 \
  --reranker-max-length 512 --reranker-device cpu \
  --reranker-fusion rrf --reranker-fusion-rrf-k 60 --show-failures
```

| 方法 / k | fact macro | fact micro | 完整 | 部分 | 零 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid @5 | 0.516 | 0.469 | 0.472 | 0.113 | 0.415 |
| Hybrid @10 | 0.627 | 0.592 | 0.547 | 0.132 | 0.321 |
| Hybrid + CE + RRF @5 | 0.634 | 0.599 | 0.509 | 0.245 | 0.245 |
| Hybrid + CE + RRF @10 | 0.785 | 0.776 | 0.698 | 0.170 | 0.132 |

@10 完整覆盖率绝对提高 15.1 个百分点，超过预设的 5 个百分点门槛；目标论文、页级和
Table N 命中分别为 `0.981/0.905/0.944`。CPU 对每题 50 个候选的 mean/median/p95/max
为 `2.734/2.737/3.312/3.489` 秒，进程峰值 RSS 约 `2201 MB`。

纯 cross-encoder @10 也达到 0.698 完整覆盖率，但相对原 Hybrid 有 4 题回退；保守 RRF
只剩 `drugr-11` 从 0.75 降至 0.50，并把 17 个 table 类型用例的事实完整覆盖恢复到
1.000，因此应用接线采用 RRF 版本。离线 `Table N` 命中仍为 0.944，唯一未命中结构化
表号的是 `scidqa-09`，但其事实覆盖完整；应用仍会在重排之后加载并检查所有结构化表，
因此确定性表格保护不能移除。

该结果达到“可以默认关闭地接线”的门槛，不支持将 Hybrid 或 reranker 改为默认，更不
证明生成答案准确率。@10 仍有 16/53 题不完整，@50 候选本身不完整的问题也不会被重排
解决。

线上原型已提供默认关闭的 Hybrid 接线，用于受控网页对比：

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

该模式复用本页同一 BM25/RRF 实现，并在融合后继续执行显式 `Table N` 过滤和确定性
单元格定位；默认 `dense` 行为不变。设置 `SCI_RAG_RERANKER_MODEL` 和固定 revision 后
才会额外加载本地 cross-encoder。当前基准已经完成一次 53 题 Dense/Hybrid+Rerank 答案
采集和 RAGAS 辅助对照；阶段结果与逐题复核结论统一记录在项目根目录
`MODIFICATION_LOG.md`。上述检索代理和 RAGAS 分数都不能直接解释为
答案正确率。

## Phase 6.2 实验性 Figure 坐标证据

`--spatial-figure-evidence` 会从 born-digital PDF 的文字层抽取 Figure caption 上方的
短坐标文字，并只对显式 `Figure N`/`图N` 查询注入精确图号证据。普通问题仍排除这些
图块，以免改变基线候选池。该开关不运行 OCR、不读取图片像素，也不能覆盖没有文字层的图；
它是与网页 `SCI_RAG_SPATIAL_FIGURE_EVIDENCE=true` 对应的默认关闭对照。

## Phase 6.3 最小 image-only vision 实验

`evaluation/vision_experiment.py` 仅运行 `manifest_challenge.json` 中的 10 道
`requires_image=true` 题。它按题目解析的 Figure 编号定位 PDF 题注，裁剪整页宽度的图区，
并把临时 PNG 以内存 data URL 发送给 `deepseek-v4-flash-vision-exp`；不接入网页或运行时，
也不做 OCR、图片持久化、Graph-RAG 或工具执行。

2026-09-01 完成 20/20 次调用（两轮各 10 题），人工复核每轮为 `7 correct / 0 partial /
3 incorrect`，低于预设每轮至少 8 题正确的门槛。失败类型为复杂化学结构图、DockQ 矩阵图的
空答案，以及一轮 SCIDQA 流程图空答案；因此本步停止在实验记录，不进入 UI/默认运行时集成。

随后进行了一个限定的裁剪 A/B：full+detail 模式在同一条消息中发送完整图裁剪和统一规则
生成的下半部居中 detail crop（左右各收缩 10%、从完整裁剪区域的 35% 高度开始）。10 道题
各重复两轮，得到 9/10 和 8/10 正确；相同条件下已有单图基线为 7/10、7/10。
因此该输入策略达到本实验门槛，但仍只作为 opt-in 实验，不接入网页默认路径；剩余失败是
SciDQA Figure 6(a) 的上下半部判断（两轮）及一次 DrugR 复杂结构图空答。该结果不能证明
通用视觉问答能力，后续若集成仍需 UI 回归和更多论文验证。

## Phase 6.4 opt-in PDF Figure 视觉路径

`SCI_RAG_VISION_ENABLED` 默认关闭，且需要同时开启 `SCI_RAG_DOCUMENT_ROUTING`。开启后，上传 PDF
会按 SHA-256 保存到数据库目录的 `source_pdfs/`；只有明确的 Figure/Extended Data Figure 问题被路由器
唯一定位到一个 PDF 来源且 hash 文件存在时，才发送 full+detail 两张图。普通问题、表格问题、非 PDF、
来源不明确和旧 DB 缺 PDF 时继续走文本 RAG；视觉 API 异常会回退文本路径并提示原因。

当前实现是受控 opt-in 实验，不改变默认网页行为，也不代表已证明通用视觉问答能力；未接入 OCR、图片
向量、图片索引、Graph-RAG 或工具执行。

2026-09-02 复测修正后的 Figure 页定位，并为视觉 API 空内容增加一次有界重试。10 道题各重复两轮，
20/20 次调用返回非空答案；人工复核第一轮 `10/10 correct`、第二轮 `9/10 correct + 1 partial`，
10/10 个来源页与基准一致。该结果仍仅作为 opt-in 门控证据。

随后复核原始 PDF 后更正 `img-scidqa-02` 的空间 gold：Figure 6(a) 嵌入页面中的绿色高亮位于下半部，
不是上半部。使用修正标签在同一 full+detail 条件下追加一轮，三轮均为 `9/10 correct`（30/30 调用成功，
合计 27/30）；唯一不稳定项是 DrugR Figure 2(A) 的箱线图中位线比较。此结果不扩大为通用视觉能力结论。

## Phase 6.5 多表派生数值题复测

多表题此前可能只按问题中的第一张表筛选，导致第二个操作数在生成前丢失。修复后，`Table N` 集合按问题中全部显式表号匹配；派生数值题继续跳过单元格快捷回答并保留完整表格证据。

六论文挑战集离线检索中，`calc-table-02` 与 `calc-table-04` 在 @10/@50 均达到两项操作数全覆盖；20 道 computation 题在 @50 为 `20/20` 完整覆盖。两道目标题各进行两次真实生成烟测，答案分别稳定为 `0.14` 与 `0.07`，每次均返回 Table 1、Table 2。该结果是定向修复证据，不代表全部生成答案已证明正确。

交叉核对原 PDF 时发现 `cross-04` 的初始标注把 MgNO Darcy rough 的 `1,280` 个训练样本误写成 `1,000`；现已更正为 `7,000 / 1,280 ≈ 5.5` 倍，并同步更新 gold facts、contexts 和 calculation 元数据。该题此前的拒答是对矛盾证据的合理反应，不计入 Graph-RAG 失败样本。

## Phase 6.6 跨论文来源证据补全

多来源查询开启 `SCI_RAG_DOCUMENT_ROUTING=true` 与
`SCI_RAG_QUERY_DECOMPOSITION=true` 时，应用现在对每个路由来源保留一个有界的
lexical/同节/显式数字证据候选，并一次性构造来源覆盖前缀；这修复了“后插入来源挤掉先插入来源”
以及 dense top-k 漏掉远端数字或算法段落的问题。默认 dense、单来源问题和网页默认配置不变。

离线挑战检索的 `cross_document` 在 @50 为 `5/5` 完整覆盖。六论文 696 块隔离库上的真实生成复测
覆盖 5 道题、每题两轮（10/10 调用成功、10/10 人工复核正确），上下文和运行配置两轮 100% 稳定；
这证明了当前路由补全在该挑战集上的效果，不等同于跨论文泛化，也不构成 Graph-RAG 或工具执行器的
必要性证据。

## Phase 6.7 computation 全量生成门禁

在同一 696 块隔离库上，20 道 computation 题各重复两轮，40/40 次调用成功；19 道题两轮均正确，
`calc-af3-02` 两轮拒答。该题所需的 Figure 3(a) 两个指标只在图像像素中，文字/坐标证据没有
`87.7` 和 `86.9`，所以应归入 image-only 证据边界，而不是算术执行失败。上下文、metadata 和
运行配置在 20 个重复 case 上均稳定，不能据此宣称 20 题的通用答案正确率。

因此当前没有 5 道可由本地白名单工具稳定修复的真实运算失败，calculator/tool executor 仍按门槛
暂缓；图像证据继续由默认关闭的 opt-in vision 路径单独验证。
