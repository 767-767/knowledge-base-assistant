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

`manifest_blind_holdout_draft.json` 最初是下一阶段的候选盲测清单，继承上述 8 篇论文并追加桌面上尚未进入
基准的 SciDC 与 NaviRAG 两篇开放 PDF，共 10 篇论文、98 道题。新增 16 题覆盖约束解码公式、层级导航、
复杂表格和效率/复现性信息。由于这批题已经被用于检索诊断和定向修复，清单现标记为
`development-regression`，不能再作为未见盲测或正式泛化证据。两篇原始 PDF 仍保存在仓库外，不应提交到 Git。

`manifest_blind_holdout_fresh_draft.json` 曾是已冻结的 fresh evaluation 清单，包含 NeurIPS 2024 的 SPIQA、UDA
和 ICLR 2025 的 LiveXiv，共 20 道文本/表格/计算题。三篇 PDF 均为官方免费版本，已完成 SHA-256、页码、表格、
图注和代表页视觉核对。冻结前已经运行过 E1 离线检索诊断，但没有用于 Chroma 生成、答案生成或模型调参；
因此它不是严格意义上的 blind holdout。E2b 已经使用这批题生成并查看结果，清单现标记为
`development-regression`；它可以用于回归修复，但不能再作为未见评估或泛化证据。原冻结版本的
SHA-256 和 E2b 结果保留在 `PAPER_AUDIT.md`，后续正式验证必须另建未见评估集。

Phase E2b 已在包含全部 13 篇论文的 1,594 块隔离 Chroma 上完成一次生成对照：Dense 与 Gated
Hybrid + 固定 revision 的 BGE reranker 均为 20/20 请求成功。候选配置的 required-fact 上下文
完整覆盖为 15/20，逐题人工语义复核为 14 correct / 3 partial / 3 incorrect，尚未达到预设的
17/20 正确门槛，因此不切换线上默认检索，也不围绕这 20 题继续调参。完整生成 trace 保存在
仓库外 `/private/tmp`，不提交到 Git。

Phase E3b 已修复三类通用缺口：重复 `@N` 表头与拆分行实体、平均绝对变化等语义列的确定性匹配、以及唯一来源
流程/工具/过滤/样本问题的来源内证据补全。UDA Table 6、LiveXiv Table 2 的真实 PDF 导出形态有离线回归覆盖；
这 20 题仍只用于开发回归，不能重新充当未见泛化集。

`manifest_phaseF1_unseen.json` 是 E3b 之后新建的最小未见留出集：TableRAG（EMNLP 2025）和
CURIE（ICLR 2025）两篇此前未进入任何清单的论文，共 12 道题。题目覆盖复杂表格、跨页方法、长上下文
统计和 2 道 Figure 视觉题；两篇 PDF 已完成页数、SHA-256、文字层和关键页面视觉核验。F2 诊断后因修复了
表格 caption 证据对齐并补充 PDF 表头导出别名，清单已改标为 `development-regression`；它不再作为未见
泛化证据，后续正式生成应另建确认集。

Phase F3 在两篇论文的 402 块隔离库上完成受控生成：10 道文本/表格题各两轮均请求成功，人工复核每轮
8 correct / 2 partial / 0 incorrect；两道 Figure 31 图片题各两轮请求成功，但坐标题稳定误读一个数字，
因此视觉路径仍默认关闭。结果仅用于开发回归和能力边界记录，不代表生产正确率或泛化。

Phase F4A 已补齐两个通用表格答案缺口：IoU 多指标选择，以及问题明确指定的 backbone/表注指标限定词。
隔离应用路径复测已完整返回 TableRAG Table 4 和 CURIE Table 7 的请求项；全套离线测试为 `241/241`。

Phase F4B 的 4× 高分辨率 full+detail 对照覆盖 12 道现有图像题，结果为 `9 correct / 2 incorrect / 1 empty`，
没有超过已有输入策略的整体表现；高分辨率不接入正式路径，视觉仍默认关闭。

`manifest_phaseG_confirmation.json` / `cases_phaseG_confirmation.jsonl` 原为一次性未见确认集：3 篇 ACL Anthology
论文、20 道题（MT-RAIG 8、WikiMixQA 6、TableEval 6），PDF 保存在仓库外。题目和金标准仍保持冻结，但因生成
trace 已被审计，清单现标为 `development-regression`，不再作为未见确认集。隔离库 472 块上的 Gated Hybrid
离线候选诊断 @10/@50 完整事实覆盖为 `13/20`、`17/20`；这不是送入模型的最终上下文覆盖率。对真实 trace
中的 `contexts` 重算后，@10 等价覆盖为 `11/20 full、7/20 partial、2/20 zero`（macro/micro=`0.7592/0.7157`）。
两轮 DeepSeek 生成最终 `40/40` 成功，人工语义复核为每轮 `14 correct / 4 partial / 2 incorrect`，未达到
`≥17/20` 且错误 `≤1` 的确认门槛。因此该结果不支持泛化结论或切换默认 reranker；后续修复只能使用该开发回归集，
并在完成后另建全新的未见确认集。

Phase H1 的首个通用修复已在同一隔离库、假客户端的真实应用路径完成上下文检查：复合问题现在按问号/句号拆分，
并使用数量、占比、标注、代理模型、表格集合和人类引导等通用中英文别名。`mtraig-g-03` 的真实上下文事实覆盖
从 `0/12` 提升到 `12/12`，`mtraig-g-05` 从 `0/5` 提升到 `5/5`；20 题真实上下文覆盖由原来的
`11/20 full、7/20 partial、2/20 zero` 变为 `13/20 full、7/20 partial、0/20 zero`（macro/micro=`0.8592/0.8824`）。
这只是检索上下文检查，尚未重新调用 DeepSeek，不代表答案正确率或泛化结论。

Phase A 已完成独立 PDF 审核：16/16 题的页码、表号、公式、数值/单位和答案片段均与原文一致，关键页面另做
视觉核对。该结果只证明题库自洽；两篇论文都是免费 arXiv 预印本，所以清单仍是候选压力测试，不单独支撑
权威论文的泛化结论。是否转为正式盲测集，须先确认来源级别，再进行隔离库检索和生成。

## Phase B–D：候选盲测的检索与生成审计

Phase B 在仓库外临时结果 `/private/tmp/scirag_phaseB6.seARYm/retrieval.json` 上完成：Hybrid
候选池为 50、最终上下文上限为 10。98 题的 @10/@50 结果分别为：目标论文命中
`0.990/0.990`、页级命中 `0.862/0.989`、Table N 命中 `1.000/1.000`、required-fact
macro/micro `0.791/0.945` 与 `0.794/0.948`，完整覆盖 `0.724/0.908`，零覆盖
`0.143/0.020`。SciDC 8 题与 NaviRAG 8 题的 fact macro 在 @10/@50 分别为
`0.667/0.875` 和 `0.906/0.906`。@50 是候选池覆盖，@10 才接近最终生成证据；这些都是
检索覆盖代理，不是答案正确率。

Phase C 使用 10 篇论文重建的 1126 块临时 Chroma（`/private/tmp/scirag_phaseCD_db.y1f4be`），
对 16 道新增题各生成两轮：`32/32` 成功、来源越界 `0`、运行配置/上下文/metadata 稳定 `16/16`；
答案逐字稳定率为 `7/16`，说明生成措辞仍有随机性。逐题人工复核为 `14 correct / 2 partial / 0 incorrect`：
SciDC bottom-layer 公式有 `\\notin` 换行格式问题，NaviRAG 节点决策题漏写显式
`π(n) ∈ {absorb, expand}`；详细记录见 [`reviews_blind_holdout_16.jsonl`](reviews_blind_holdout_16.jsonl)，
精简回答快照见 [`answers_blind_holdout_16_repeat1.jsonl`](answers_blind_holdout_16_repeat1.jsonl)。
现已增加通用公式输出边界修复和原文核对项回退；两个 partial 题及一个控制题各运行两轮均成功，partial 症状已消失，
但未对其余 13 题重新生成，因此原有 14/2 统计仍是整批 v5 的人工结果。

随后使用 runner 的 `--no-source-filter` 对16道新增题各运行一轮：`16/16` 成功，首个上下文目标来源命中 `16/16`，
且 trace provenance 完整。该结果只说明当前路由路径在这批题上的行为；无过滤词面审计 macro/micro 为
`0.5833/0.5600`，因中英文和公式格式差异不作为答案准确率。

效率题最初漏掉第 7 页的具体 token、再生成和时间数字；新增“效率/代价/开销”来源内证据路由后，
两轮均覆盖 `3.6k→4.2k`、`0.8`、`1.9k→2.3k`、`2.5` 和约 `3×`。这次只验证了隔离库和受控
生成路径，未重建项目数据库、未运行 RAGAS，也不把 arXiv 候选题当作正式泛化证明。

此前 `manifest_generalization.json` 中的 TANQ/FigEx 16 道新增题曾在隔离 904 块数据库上各生成两轮，
结果为 32/32 次调用成功；详情见 `reviews_generalization_16.jsonl`。这段历史结果不属于当前
`manifest_blind_holdout_draft.json` 的 16 道新题，也不能外推为生产正确率或通用泛化结论。

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

## Phase 6.8 82 题生成语义审计

在已有仓库外临时结果 `answers_generalization_final.jsonl` 上完成 82 题×2 轮的离线人工语义复核，
不重复调用 API。复核记录为 `reviews_generalization_82.jsonl`：按两轮中较保守的结论统计为
`63 correct / 9 partial / 10 incorrect`。主要缺口是 SciDQA 的人工标注数字、重叠阈值和 RAG 配置，
科学表格行列标记/未来方向，MgNO 离散公式与随机种子复现说明，AF3 幻觉抑制，以及 THINKNOTE 的
少数公式、指标、图表题。两轮词面事实代理约 `0.80`，不能替代上述语义结果；该审计也不等同于
RAGAS、生产正确率或跨论文泛化证明。

由于上述结果生成时使用的是全局多论文库，现将稳定性 runner 改为按每个 case 的
`document_id` 与 `additional_document_ids` 传入 `source_filter`，只在目标论文范围内检索；普通网页
调用不传该参数，默认行为不变。修复后的 82×2 结果需在 API 可用时重新生成，旧结果不能直接复用。
同一路径下若输出文件中没有匹配的 `source_filter`，runner 会自动把该 case 视为待重跑，不会误把旧追踪续跑为新结果。

2026-09-03 在复现的 904 块隔离库上完成 82 题×2 轮生成（`164/164` 成功）。两轮答案词面事实
macro/micro 为 `0.8187/0.8333` 与 `0.8370/0.8542`，gold-context recall 均为 `0.6707`；配置、上下文
和 metadata 均稳定，source 越界为 `0`。该结果仍需对新答案独立语义复核，不能沿用修复前的人工标签，
也不能作为 RAGAS 或生产正确率证明。

## Phase 6.9 来源隔离批次差异复核

旧批次与新批次使用同一 8 篇论文、82 道题和两轮生成；差异仅在于新批次强制按目标论文过滤检索。
逐 case 对比后，22/82 道题两轮答案签名完全不变，60 道题有变化（多数是措辞变化）。对答案变化、
旧批次非正确或两轮词面事实不一致的 22 道高风险题回查 PDF、题目金标准和新上下文；其余题沿用旧语义
标签并检查未出现事实回退。新批次的保守语义统计为 **66 correct / 11 partial / 5 incorrect**，完整
复核记录见 [`reviews_generalization_82_sourcefiltered.jsonl`](reviews_generalization_82_sourcefiltered.jsonl)。

相对旧批次，TableLLM 行列标记与 future work、MgNO 离散系统、AlphaFold-Multimer cross-distillation、
THINKNOTE 500 题和 Figure 3 数值等题目恢复正确；MgNO 平滑公式、THINKNOTE 输出公式以及 TANQ 平均列数
因两轮中较差答案而按保守规则降为 partial。该统计是单次来源隔离批次的人工证据，不是 RAGAS、生产正确率
或跨论文泛化证明；后续修复只针对已确认的错误簇，并先做受影响题目回归。

## Phase 6.10 来源内证据补全（定向修复）

来源隔离批次暴露出一类可泛化的召回缺口：唯一来源已知但中文问题没有论文专名时，路由器无法提供
独特 token，方法段、阈值、指标和图示解释句可能不进入上下文。当前实现增加中英术语别名，并在
`source_filter` 只有一个来源时启用有界的来源内 lexical/同节/数字补全；普通网页调用不传过滤器，
因此默认检索路径不变。

显式 Figure 查询现在还会在精确图块之后补入同来源同页的 `text/formula` 解释块，排除图片文本，
以覆盖 PDF 双栏切分把解释句误标为 formula 的情况。未增加 OCR、图片索引、Graph-RAG 或工具执行器。

213 项离线 unittest 与 `git diff --check` 通过。使用 904 块隔离库先对 9 个受影响 case 各重复两次，
再对新增“检索/第一人称”别名影响的 `scidqa-08`、`scidqa-10` 各补测两次：最终
`scidqa-05/06/07/08`、`table-llm-03`、`mgno-11` 和 `thinknote-06` 两轮均覆盖声明事实；
`scidqa-10` 两轮均说明同行评审、作者和第三人称改写，但漏写金标准中的 OpenReview 平台，保持 partial。
随后对 `thinknote-13` 单独重复两次，均召回同页 Jackie Robinson 解释句并正确作答。该结果是定向回归
证据，不是 82 题全量重评、RAGAS 或生产准确率证明；完整 82 题来源隔离审计基线仍保留在
`reviews_generalization_82_sourcefiltered.jsonl`。

## Phase 6.11 来源补全影响审计

先用离线快照比较来源内补全对同一 82 题的实际上下文影响。未加门控的来源内回退会改变 `61/82`
题上下文，并使 `5` 题的事实覆盖代理下降；因此没有保留该范围。收窄为方法/阈值/配置等显式证据
意图，并对 Figure 查询排除路由证据后，最终仅改变 `24/82` 题上下文，检索事实覆盖代理由
`65/11/6`（full/partial/zero）变为 `75/6/1`，无覆盖降级、无来源越界。该代理只检查检索证据，
不等同于答案正确率。

在收窄门控版本上对 `20` 个受影响 case 各生成两轮，`40/40` 次 API 调用成功；结果与既有语义
复核总体一致。Figure 同页解释块只对包含“示例/正确答案”的显式问题启用，避免把数值 Figure 题
的邻近正文混入上下文；`thinknote-13` 两轮仍正确，`thinknote-09` 保留为文字层无法可靠映射
图像分组的未决题。尚未运行 82 题修复后全量重评、RAGAS 或生产流量测试。

## Phase 6.12 表格结构与跨页续句修复

对真实 PDF 页面和旧隔离批次的上下文回查后，修复三类通用结构问题：Setting 列现在可作为复合数据集
限定，避免同一模型的基线行抢先命中；PDF 导出的重复多层表头会合并为完整的场景/数据集列名，并识别
被拆到两个单元格的分组标签；parent-window 仅在前块以连接词结尾时允许跨页拼接续句。对应回归覆盖
`table-llm-05`、`thinknote-12` 和 `tanq-holdout-02`，全套离线 unittest 为 `216` 项。

旧 904 块隔离库上对三题各生成两轮，`6/6` 次调用成功；其中 `table-llm-05` 和 TANQ 的答案两轮均
覆盖金标准，`thinknote-12` 的 API 结果仍来自修复前生成的表格块，因此只能作为旧行为记录。新解析器的
双层表头和分组选择由真实 PDF 导出形态的纯函数回归测试证明；尚未在重新切分的隔离库上完成端到端 API
复测，原因是本地 Sentence-Transformers 加载触发 Hugging Face DNS 重试。项目数据库未重建。

## Phase 6.13 公式证据抽取与排序修复

公式问题门控现在覆盖形式化定义、最终输出和激活函数等通用问法；PDF 公式恢复改用 PyMuPDF 坐标行合并，保留双栏页面中的括号、运算顺序和参数顺序；公式候选按问题变量的等式左侧优先排序。系统提示要求公式逐字转录，不得交换参数或自行补写。

新增公式候选、变量左侧排序和 PDF 版面合并回归测试；全套离线 unittest `219` 项通过。旧 904 块隔离库上 MgNO 两题 `4/4` 次正确；THINKNOTE 旧库仍含历史乱码公式块。用当前解析器生成的临时内存证据验证 `T = M(Ika, q, D)` 与 `y = M(Ita, q, T, R)` 均可被正确转录。未重建项目数据库或运行 RAGAS。

## Phase 6.14 新鲜隔离库门禁与 82 题复测

当前源码从 8 篇清单 PDF 新建 `/private/tmp` 隔离库为 898 块（720 text / 48 table / 130 formula），未改项目
`chroma_db`。离线 Hybrid @10 的 82 题事实完整覆盖为 69.5%，表格题为 96.4%，文档路由为 62/62；均为
检索代理。7 道公式/表格目标题各两轮共 14/14 调用成功且人工核对正确；Table 3 拆分模型分组的过滤路径
随后修复并由 220 项离线测试覆盖。新鲜库上的 82 题×2 生成共 164/164 成功，来源越界为 0，运行配置和上下文
均 82/82 稳定；原子事实覆盖两轮为 macro/micro `0.9010/0.9167`、`0.9183/0.9292`，完整覆盖 69/82、70/82，
逐字稳定 24/82。该代理不替代独立语义复核或 RAGAS，结果未写入仓库。

## Phase 6.15 新鲜批次独立语义复核与定向回归

对上述 82 题×2 新答案逐题回查 gold、PDF 和上下文；结合两道受影响题的修复后两轮定向结果，保守状态为 **79 correct / 1 partial / 2 incorrect**。`scidqa-10` 仍未明确写出 OpenReview；`thinknote-09` 和 `figex-holdout-07` 的缺口是图像像素映射，不把文字层拒答误判为检索或计算器失败。该统计不等同于 RAGAS、生产准确率或修复后 82 题全量重新生成。

本轮通用修复覆盖：PDF 丢失模型名标点的实体匹配；模型/方法/数据集表述下的第一列方法分组继承；带 `(acc)` 等后缀的命名列匹配；以及架构替换问题的中英来源内证据召回。`thinknote-07`（69.8/71.0）和 `af3-02`（pairformer/evoformer、diffusion/structure module）各两轮均正确；全套离线 unittest 为 `223` 项。原始 JSONL 仍只保存在仓库外临时目录，未改项目数据库、未运行 RAGAS、未提交或推送。

## Phase 6.16 未决图像题视觉定向门禁

沿用 Phase 6.4 的默认关闭 full+detail 视觉路径，仅对文字层无法可靠映射的 `thinknote-09` 和
`figex-holdout-07` 各执行两轮 DeepSeek vision 调用，`4/4` 次成功且人工核对正确：Figure 3(a)
的 PopQA Top-K retrieval accuracy 在 K=1/K=5 为 `53.20/68.80`，Figure 7 的红色 O 位于右下角
一致性表面板。该结果是两个已知边界题的定向门禁，不是修复后 82 题全量视觉评估，也不改变默认网页
配置；`scidqa-10` 仍为文字路径部分正确。唯一 case×repeat 结果只保存在仓库外 `/private/tmp`，未改数据库、未运行
RAGAS、未提交或推送。

## Phase 6.17 网页状态与跨论文公式复测

上传回调已同步刷新知识库块数显示；隔离网页上传三篇论文后计数从 `117` 更新至 `224`、再到 `316`。多文档复合问句现在按公式子问句分别路由，避免第一篇论文的路由屏蔽后续论文公式；真实网页复测同时得到 DrugR `4,855` 和 `T = M(I_{ka}, q, D)`。新增回归测试后离线 unittest 为 `224` 项，未改项目数据库、未运行 RAGAS、未提交或推送。

## Phase 6.18 来源内复合问题的子句证据排序

来源过滤与查询分解同时开启时，来源内 lexical/同节/数字回退现在对带有疑问词的子句使用该子句本身；纯来源标识才复用完整问题。这样可避免一个复合问题的泛化同节结果遮蔽另一个子问题的直接证据，且普通网页路径不变。

新增来源内子句回归测试；全套离线 unittest 为 `225` 项，编译检查和 `git diff --check` 通过。使用 8 篇论文隔离库做 Hybrid @10 诊断，文档路由保持 `62/62` 正确；来源过滤下的 SciDQA 受控生成复测已包含 OpenReview 来源与第三人称改写。该结果仍是定向修复证据，不代表全量生成正确率、RAGAS 或生产泛化能力。

## Phase 6.19 当前修复批次收尾

`scidqa-10` 独立生成两轮均明确回答 OpenReview 上的同行评审—作者来源，并说明第三人称改写和上下文补充；DrugR Table 6、TANQ/FigEx 跨论文模型题和 THINKNOTE Table 1 控制题均通过。现有 8 篇论文、82 道题冻结为开发回归集，不再据此继续调参或宣称泛化能力；后续应建立未参与修复的盲测留出集。

## Phase H3 新论文确认集

`manifest_phaseH3_confirmation.json` / `cases_phaseH3_confirmation.jsonl` 曾冻结两篇新的 2025 ACL 论文、12 道题；
隔离库 304 块上两轮生成均成功，但人工复核每轮仅 `8 correct / 3 partial / 1 incorrect`，因 MEBench Table 3
跨列表头拆分导致的错误结构化答案未通过 H3。清单现标为 `development-regression`；题目和金标准保持冻结，修复后必须
另建未见确认集，不能把 H3 结果当作泛化证据。逐题记录见 `reviews_phaseH3_confirmation_v1.jsonl`。

## Phase H4 修复后开发回归

H3 失败样本仅用于回归，题目和金标准保持冻结。修复了跨单元格表头、重复数据集/Prompting 分组、中文多值限定，
并增加“选项”中英别名；全套离线 unittest `248/248`，解析、编译和 diff 检查通过。

两篇论文重建的 304 块隔离库上，Table 3 `GPT-4 + RAG` 四列、Table 2 X→Y 下 GPT-4 的
`78.63/60.71/67.32` 和 Medbullets 的 `five answer choices` 均进入真实送模上下文。两轮 DeepSeek `24/24` 成功，
来源/配置/上下文/metadata 稳定；按 case 人工复核 `11 correct / 1 partial / 0 incorrect`，达到 H4 开发回归门槛。
唯一 partial 是 MEBench Table 2 表头拆分展示，三类总数正确。trace 仅在 `/private/tmp/scirag_phaseH3_answers_h4_v1.jsonl`
（SHA-256 `39dd5acf493aea92728de4f15a9f3c1a7a15bdadf3d780f118a7c77710a7a561`），复核记录见
`reviews_phaseH3_confirmation_h4_v1.jsonl`。该结果不证明未见论文泛化、不改变线上默认配置；下一步建立新的 H5 未见确认集。

## Phase H5 新论文确认集（现为开发回归）

H5 使用两篇此前未进入基准的 ACL Anthology 免费论文：SciAssess（Findings of NAACL 2025，23 页）和 YESciEval（ACL 2025 Long
Papers，35 页），共 12 道题。PDF 只保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`；详情、哈希和题目清单见
`manifest_phaseH5_confirmation.json` / `cases_phaseH5_confirmation.jsonl`。清单状态为 `development-regression`，因为生成前已用该批次
做离线检索诊断和通用修复，不能再作为严格未见集。

在仓库外 432 块隔离 Chroma 上，真实应用路径来源路由 `12/12`，送模上下文 required-fact 覆盖 `12/12 full`（macro/micro=`1.0000/1.0000`）。
两轮 DeepSeek `24/24` 成功，context/config/provenance 按 case 稳定。严格词面答案审计 macro/micro=`0.6606/0.7073`，仅作表面覆盖信号；人工复核
第一轮 `12 correct / 0 partial / 0 incorrect`，第二轮 `10 correct / 2 partial / 0 incorrect`。两个 partial 是第二轮对两个 rubric/层级分组问题的保守拒答，
不是上下文缺失。按每个 case 两轮均完整的保守口径为 `10/12 correct、2/12 partial、0/12 incorrect`，未达到 `≥85%` 未见确认门槛；不宣称泛化或切换线上默认检索。
逐题记录见 `reviews_phaseH5_confirmation_v2.jsonl`，trace 只保存在仓库外 `/private/tmp/scirag_phaseH5_answers_v2.jsonl`。

## Phase H6 开发回归集（初始冻结，后续降级）

H6 为避免在 H5 上继续调参，选用两篇此前未进入任何基准的开放论文：CAQA（ACL 2025，23 页）和 MiMoTable（COLING 2025，13 页）。
PDF 均来自 ACL Anthology 官方免费链接，保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。

`manifest_phaseH6_confirmation.json` / `cases_phaseH6_confirmation.jsonl` 共 12 道题（每篇 6 题），覆盖归因类别与复杂度、KG 构建流程、
复杂表格统计、meta operations 和人工质量控制。清单初始在检索、生成和调参前冻结为 `frozen-unseen`，但隔离检索 gate 暴露通用质量控制别名缺口，现已按协议改标为 `development-regression`；当前 manifest SHA-256=
`14e036d7b28c71bd9a43556da03208a545653a189ea7d3069597f09a15ef135c`，cases SHA-256=
`c8976fc51e4fae8d7ae4f0504b01cded8e8caca9c98b67e978e571a558ef8a5a`。

通用别名和换行断词修复后，隔离 256 块 Chroma 的真实应用路径 required-fact gate 为 `12/12 full`。两轮 DeepSeek `24/24` 成功，人工复核为 `20 correct / 2 partial / 2 incorrect`，按 case 两轮均完整 `10/12`；错误集中在 MiMoTable Table 4 Simple=33.6% 遗漏和 Figure 6 数值错配。不宣称泛化；trace 仅保存在仓库外 `/private/tmp/scirag_phaseH6_answers_regression_v1.jsonl`，逐题记录见 `reviews_phaseH6_regression_v1.jsonl`。下一步建立真正未参与诊断的新 H7 集。

## Phase H7 开发回归集（初始冻结，后续降级，2026-09-06）

H7 选用两篇此前未进入任何基准或诊断的 ACL 2025 Long Papers：TC–RAG（27 页）和 ChartCoder（16 页），分别覆盖 RAG 状态/记忆/公式/效率与图表多模态/代码生成。PDF 来自 ACL Anthology 官方免费链接，保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。

`manifest_phaseH7_confirmation.json` / `cases_phaseH7_confirmation.jsonl` 共 12 道题（每篇 6 题），覆盖表格、公式、方法流程、效率指标和图表统计。清单初始在任何检索、生成或调参前冻结为 `frozen-unseen`，但 context gate 发现通用表头、整行多列、公式证据和多事实召回缺口，现按协议改标为 `development-regression`；当前 manifest SHA-256=`f92c5082083b1c736dfb40deac3c1c80508ebf23011e3fa12710e565b4e390b7`，cases SHA-256=`e3ad5c19ab528ba8bb1b2d607e3b69acb479947c0def20b20a6e0574c8a6d290`。

隔离 370 块 Chroma 的真实应用路径 required-fact gate 为 `5/12 full`，fact macro/micro=`0.6457/0.6705`；ChartCoder 六题 full，TC–RAG 失败集中在跨列拆分表头、整行多列查询、公式证据和多事实实验设置。H7 不作为未见泛化证据，修复完成后另建 H8。

随后补充通用三层/居中表头合并、中文平均指标别名和“分别”多列行抽取；TC–RAG Table 2 已能直接返回四个平均值。H7 fix2 离线 gate 为 `5/12 full`、fact macro/micro=`0.6576/0.6818`，仅作开发回归诊断，未调用 DeepSeek，H7 仍不作为未见证据，后续另建 H8。

## Phase H8 开发回归集（初始冻结，后续降级，2026-09-06）

H8 选用 ACL 2025 Long Paper ChainRAG 与 ACL 相关 MAGMaR 2025 两篇免费 PDF，新增 12 道题，覆盖多跳实体改写、句图检索、表格指标及 PDF/视频多模态管道。清单初始在检索前冻结为 `frozen-unseen`，随后隔离 gate 暴露多事实、跨列统计表和模态流程细节的通用覆盖缺口，现改标为 `development-regression`。

隔离 158 块 Chroma 的真实应用路径 required-fact gate 为 `6/12 full`，fact macro/micro=`0.7449/0.7500`；未调用 DeepSeek，不作为未见泛化证据，后续另建 H9。

H8 回归修复补齐多级表头续行、嵌套方法行外层模型、阈值括号/小数空格及来源内量化/设置/模态分布别名；同一 158 块隔离库的真实 `query_knowledge` 路径（假客户端、来源过滤）复测为 `12/12 full`，fact macro/micro=`1.0000/1.0000`。ChainRAG Table 1 和 MAGMaR 多模态阈值/统计均可核验；未调用 DeepSeek。H8 仍是开发回归，不证明未见泛化，下一步建立 H9。

当前 H8 文件 SHA-256：manifest=`fb1a4b69d1ceda09f56203269fe5d367ff450ad05b54429955914c87d936019b`，cases=`ecc61faad0c583e9debd2133935cfc0a280e4b8b77e3c514836cc277f9ff3d66`。

## H7 最后一轮通用修复（2026-09-07）

表格行过滤现仅接受实际 `Dataset` 单元格匹配的 `在/on` 限定词，避免将 CMB、MMCU 等横向表头误当作数据集行；同时补充“评价指标/加速策略”及英文检索别名。全套离线 unittest 为 `259/259`。

在同一 370 块 H7 隔离库、Hybrid、文档路由、查询分解、可选证据保护和假客户端路径复测，送模上下文 required-fact gate 为 `8/12 full、4/12 partial、0/12 zero`，fact macro/micro=`0.9000/0.8750`。TC–RAG Table 1 已完整，ChartCoder 六题保持完整；这仍是开发回归的上下文覆盖，不证明答案语义正确或未见论文泛化。H7 达到停损点，下一步建立不参与修复的 H9 确认集。

## H3–H8 配置 A/B 验收（2026-09-07）

A 为默认 Dense（文档路由、查询分解关闭），B 为 Dense+文档路由+查询分解；固定各批次隔离库、离线嵌入模型和 `context_k=10`，A/B 各重复 3 次且结果一致。B 的 full 案例数相对 A：H3 `9→11`、H5 `7→9`、H6 `8→11`、H7 `7→8`、H8 `7→8`；各批次 required-fact macro/micro 均提高，目标论文和显式 Table N 命中均保持 `100%`。

页级命中 H3/H5/H6 提高、H7 持平，但 H8 从 `12/12` 降为 `10/12`，未通过严格的全指标无回退门槛。因此线上默认仍为 Dense，B 仅作为 opt-in；Hybrid 不因本次验收切换默认。

## H7 两轮生成语义闸门（2026-09-07）

H7 在 370 块隔离库上完成 12 题×2 轮生成，`24/24` 成功。provenance、配置、context ID 和 metadata 均按 case 稳定；答案逐字稳定 `5/12`。人工复核结果为第 1 轮 `5 correct / 6 partial / 1 incorrect`、第 2 轮 `5 correct / 5 partial / 2 incorrect`，未通过 `≥10/12 correct 且 ≤1 incorrect` 的生成闸门。逐题记录见 `reviews_phaseH7_generation_v2.jsonl`，trace 只保存在仓库外 `/private/tmp/scirag_phaseH7_answers_v2.jsonl`。

H7 继续作为 `development-regression`，不证明泛化，也不改变线上默认；下一步应建立真正未参与修复的 H9，且在下载任何新 PDF 前先审阅候选及免费来源。

## Phase H9 新论文确认集（首轮后降级为开发回归，2026-09-07）

经用户批准，从 ACL Anthology 下载三篇此前未进入基准的免费 2025 PDF，保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：LongTableBench（Findings of EMNLP，39 页）、Table-R1（EMNLP，20 页）和
Query-Driven Multimodal GraphRAG（Findings of ACL，21 页）。三篇 PDF 的文字层逐页可读，代表页及关键表格/公式完成视觉核对。
`manifest_phaseH9_confirmation.json` / `cases_phaseH9_confirmation.jsonl` 共 12 道题，清单校验通过。

475 块仓库外隔离 Chroma 的真实 `app.query_knowledge` 门禁（Hybrid、路由、查询分解、parent-window、公式/图形证据保护、来源隔离、假客户端）为
`6/12 full、5/12 partial、1/12 zero`，required-fact macro/micro=`0.7593/0.8052`；全库不传 `source_filter` 的结果相同。
失败集中在共享的多列整行表格截断/单元格截断、无题注编号表格的显式 Table N 定位、跨块公式尾部条件和多事实段落召回，不能用论文特例解释。
按冻结协议，H9 manifest 已改标为 `development-regression`，不作为未见泛化证据，尚未调用 DeepSeek；下一步只做一次通用根因修复并在 H9 回归。

## H9 通用修复回归（2026-09-07）

本轮修复堆叠/远距离表头关联、`<br>` 并行单元格、显式多列限定词、PDF 公式续行和结构化行内有序事实审计；新增回归后全套离线 unittest `264/264`，`py_compile` 与 `git diff --check` 通过。

在仓库外 `/private/tmp/scirag_phaseH9_db_fix2_20260907` 重建 `476` 块，Hybrid+路由+查询分解+parent-window+公式/图形证据保护的真实 `app.query_knowledge`（来源过滤及全库不传过滤器）均为 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`。H9 仍是 `development-regression`，不作为泛化或答案正确率证据；未调用 DeepSeek，未修改项目 `chroma_db`。

之后在同一隔离库完成两轮 DeepSeek 生成，`24/24` 成功；trace 保存在仓库外 `/private/tmp/scirag_phaseH9_answers_regression_20260907.jsonl`（SHA-256=`7890f6bff824764016ade60cede8be8dea37883bead21fb1e3de31107809b097`）。人工 PDF/gold/context 复核为 `11 correct / 1 partial / 0 incorrect`；词面答案审计 macro/micro=`0.8536/0.8571`，真实送模 contexts=`1.0000/1.0000`。该集已参与修复定位，仍不能证明未见泛化或默认配置切换。

## Phase H10 新论文确认集（首轮后降级为开发回归，2026-09-07）

H10 新增三篇 ACL Anthology 2025 免费 PDF：SCITAT、REAL-MM-RAG 和 RealHiTBench，PDF 均保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`；关键表格和代表页面完成文字/视觉核对。对应
`manifest_phaseH10_unseen.json` / `cases_phaseH10_unseen.jsonl` 共 12 道题，清单初始冻结后进行一次隔离 Hybrid 检索诊断。

@50 目标文档命中为 `12/12`、文档路由为 `10/10`，但 required-fact 仅 `8/12 full`，macro/micro=`0.694/0.658`；缺口集中在宽表统计和多列值覆盖。清单按冻结协议改标为 `development-regression`，不作为未见泛化证据；诊断结果仅保存于仓库外 `/private/tmp/scirag_phaseH10_retrieval_20260907.json`，未调用 DeepSeek、未修改项目 Chroma。

## H10 表格标签与并排表解析修复（2026-09-07）

公共解析器新增 `Table S1` 等字母数字表号支持，并将 `pymupdf4llm` 合并的横向宽表按重复首级表头和对齐列通用拆分。SCITAT 第 4 页现在产生独立 Table 3/Table 4，REAL-MM-RAG 第 15 页的补充表元数据为 `table_label=S1`；原有整数表号保持兼容。

新增回归测试后全量离线 unittest 为 `266/266`，`py_compile` 与 `git diff --check` 通过。临时隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 的真实 `app.query_knowledge` 为 `9/12 full、1/12 partial、2/12 zero`；人工检查确认目标表块均已进入上下文，剩余缺口是事实审计字符串没有表达表格的“行标签—列值”关系。H10 继续是开发回归，不证明泛化或答案正确率。

随后将事实审计改为在单个 Markdown 行内关联行标签、列标题和单元格值，并兼容 PDF 千位逗号缺失及 `Table`/`Tables` 形式。全量离线 unittest 为 `268/268`；同一临时隔离库的真实应用路径复测为 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`。该结果仍仅属于 H10 开发回归，不证明未见论文泛化或答案语义正确。

当前离线 Hybrid 检索诊断在 `@50` 的目标文档、来源页、表号和 required-fact 均为 `12/12`，macro/micro=`1.0000/1.0000`；`@10` 为 `9/12 full`，用于保留低召回预算的边界观察。

H10 定向生成复核中，SCITAT Table 4 与 REAL-MM-RAG Table S1 两题答案正确。H9 GraphRAG Definition 4 公式题在上下文已完整的情况下仍出现不必要拒答和公式混排；通用相邻公式提示及多字符标签识别的单次重试未消除该问题。该项记录为生成/公式 provenance 边界，不再追加论文特例；H10 尚未进行全量生成。

H10 单轮生成回归（2026-09-08）：12/12 次 DeepSeek 调用成功；人工复核为 `10 correct / 1 partial / 1 incorrect`。partial 是 SCITAT 四类 reasoning type 漏列三个总类，incorrect 是 RealHiTBench Table 2 的模型/行列定位错误。逐题记录见 `reviews_phaseH10_generation_v1.jsonl`；该结果只达到开发回归门槛，不证明未见泛化或默认检索切换。
