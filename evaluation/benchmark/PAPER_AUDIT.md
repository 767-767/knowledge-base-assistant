# 多论文基准文件审计（Phase 2）

本记录只描述本地 PDF 文件和离线解析结果，不代表 RAG 检索或生成已经通过。
解析使用与 `app.py` 相同的 `pymupdf4llm.to_markdown` 参数（`page_chunks=True`、
`table_output="markdown"`、不写入图片），然后调用 `sci_rag_core.split_to_chunks`。
没有启动 Gradio、调用外部模型、写入 ChromaDB 或运行 RAGAS。

## 文件核验

| document_id | 本地文件 | 页数 | SHA-256 | 来源/定位 |
| --- | --- | ---: | --- | --- |
| `drugr-2602-08213-v1` | `2602.08213v1.pdf` | 已由原基准登记 | `15c08759ae28ac10b329528b20cc234c5046618a074dd59fd862d3af4cd0976f` | Desktop 根目录的现有种子 |
| `scidqa-emnlp-2024` | `2024.emnlp-main.1163.pdf` | 18 | `e098264a70fdb6a1e9d05daade8015141ed54018eeb196abc1478bfe344b5a20` | EMNLP 2024，ACL Anthology |
| `table-llm-sdp-2024` | `2024.sdp-1.28.pdf` | 14 | `e0d13f79aee19df2b09f6dc24a5479786536348efc3124020f599b08f66808b8` | SDP 2024 Workshop，ACL Anthology |
| `mgno-iclr-2024` | `ICLR-2024-mgno-efficient-parameterization-of-linear-operators-via-multigrid-Paper-Conference.pdf` | 20 | `743d8119f044f35a02729c7f70e121062ca1b44f1a795ad09bf9c059361df71a` | ICLR 2024 官方 proceedings |
| `alphafold3-nature-2024` | `s41586-024-07487-w.pdf` | 24 | `aba3109f2892454c9512570001598a069aaf422adb5aa0f3879414cb29a258fb` | Nature 2024，Open Access |

新增文件位于仓库外的 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`；种子文件仍位于
`/Users/qinleqi/Desktop/`。校验器支持重复传入多个 `--papers-dir`。

### Phase E0 新候选文件（2026-09-05）

| document_id | 本地文件 | 页数 | SHA-256 | 来源/定位 |
| --- | --- | ---: | --- | --- |
| `spiqa-neurips-2024` | `2024.neurips-datasets.spiqa.pdf` | 27 | `45eaa54208b42bf419643c87188adda1c244bea0c38a3b6eae8e7a7e5982f7b2` | NeurIPS 2024 Datasets and Benchmarks |
| `uda-neurips-2024` | `2024.neurips-datasets.uda.pdf` | 18 | `956f18c4450718fd5f6b98b51bd52eb5191ac7302f14778569ed415de5c95bf0` | NeurIPS 2024 Datasets and Benchmarks |
| `livexiv-iclr-2025` | `2025.iclr.livexiv.pdf` | 33 | `cdc0f5d49e6d299a949c7d714100acb5066e6f1c319c9185b91fde661a88ef5f` | ICLR 2025 |

三篇 PDF 均保存在仓库外目录。代表性标题页、表格、流程图和附录表格已渲染核对，文字与版面可读；
未启动网页、未写入数据库、未调用模型。新清单的 20 道题 required facts 均能在原 PDF 文字层找到，
其中 6 道为表格题、1 道为算术题，其余覆盖数据集统计、流程、限制和评估设置。该批次仍是候选草稿，
尚未进行生成或答案语义评估。

### Phase E1 离线检索门禁（2026-09-05）

在同一 3 篇候选 PDF 的临时内存索引上，BM25-lite @10 的 required-fact macro/micro 为
`0.750/0.780`，完整覆盖 `14/20`；默认 Hybrid-RRF 降为 `0.582/0.627`、`10/20`。
单独 dense @10 更低，为 `0.279/0.305`、`5/20`，其中 LiveXiv 仅 `1/7` 题完整覆盖。
开启现有的结构化表格、限制证据、文档路由、查询分解和 parent-window 后，Hybrid 为
`0.738/0.780`、`14/20`。再用已缓存且固定 revision 的 `BAAI/bge-reranker-base` 做
Hybrid top-50 重排并使用 RRF 融合，达到 `0.838/0.847`、`15/20`；平均单题重排约
5.86 秒、P95 11.01 秒、峰值内存约 1.75 GB。

这些都是候选上下文的词面覆盖代理，不是答案正确率；Hybrid/reranker 仍有 5 道题未完整覆盖，
集中在 LiveXiv 的视觉/人工复核/效率段以及个别跨页或段落证据。结果只用于判断下一阶段的
隔离生成门禁，不能据此切换线上默认检索或宣称泛化。

### Phase E2a 评估协议冻结（2026-09-05）

20 道题和三篇论文清单现已冻结为 `frozen-fresh-evaluation`。冻结前已经查看过 E1 离线检索诊断，
所以本集合不再宣称是严格 blind holdout；它只能用于固定的 fresh evaluation。后续不得根据这 20 题
的失败结果修改题目或调参；若修复过程使用了本集合，必须改标为 `development-regression`，并另建
未见评估集。正式生成验收应把完整多论文语料放入隔离库，但只评分这 20 道题。

E2b 已对这 20 题完成一次生成验收并查看逐题失败，因此从 E3 起清单状态转为
`development-regression`。E2a 冻结版本仍由下列 SHA-256 标识；后续修复结果不得与该一次性
验收结果混为新的未见评估证据。

最终文件完整性（冻结记录）：

- `manifest_blind_holdout_fresh_draft.json` SHA-256：`cb256f0084301d2aaefdad3502d94bbd473a9d1ad280daf5fd24c44d5518f352`；
- `cases_blind_holdout_fresh_draft.jsonl` SHA-256：`c103d2ff14e55a31cdd858b39f4ccd083e86784ffcaa0bd9fdaffb5847eaf01c`。

### Phase E2b 隔离库生成验收（2026-09-05）

在仓库外重建的全量 13 篇论文隔离库上，共 1,594 个块；只评分冻结的 20 道题，未写入项目
`chroma_db`。Dense 基线和 Gated Hybrid + `BAAI/bge-reranker-base`（revision
`2cfc18c9415c912f9d8155881c133215df768a70`）均为 `20/20` 请求成功，provenance 为 `20/20`
完整，目标论文在上下文中命中 `20/20`，但首个上下文目标论文仅 `18/20`。

| 配置 | 上下文 required-fact 完整覆盖 | 人工语义复核 | 平均延迟 |
| --- | ---: | ---: | ---: |
| Dense | `12/20` | `9 correct / 7 partial / 4 incorrect` | 1.37 s |
| Gated Hybrid + Reranker | `15/20` | `14 correct / 3 partial / 3 incorrect` | 4.41 s |

候选配置虽有明显提升，但未达到预设 `≥17/20` 正确、`≤1` 错误门槛，不能作为线上默认或
泛化证明。主要失败包括：`uda-fresh-04` 表格列结构被切碎，`uda-fresh-07` 的限制题受到
跨论文上下文干扰而拒答，`livexiv-fresh-02` 未召回 DeepSearch/过滤模型细节，
`livexiv-fresh-04` 取成 Table 2 的原始均值而非平均绝对变化，`livexiv-fresh-06` 漏掉
人工复核样本数。上述问题属于检索/证据排序或回答完整性，不证明需要视觉或工具调用。

生成 trace（仓库外，仅作可复核证据）：

- Dense：`/private/tmp/scirag_phaseE2b_dense_20260905.jsonl`，SHA-256
  `725472a4aedbbdd1edae1ec1f546e349111bb0eb7da068bbe840e1221695f145`；
- Gated Hybrid + Reranker：`/private/tmp/scirag_phaseE2b_hybrid_reranker_20260905.jsonl`，SHA-256
  `9fa02de6cf9f563fe2b45f4073a5d36ae57c674a2fde2f4d4e29149a5ea777f2`。

### Phase E3b 通用修复（2026-09-05）

E2b 暴露的失败簇已在共享路径修复，未针对论文或题号增加分支：

- 解析器识别以 `@1/@5/@10/@20` 为内容的第二表头，并按指标周期恢复被拆开的分组标签；跨相邻非数字单元格的行实体也会合并。
- 表格列匹配加入 `@N` 指标和 `average absolute change` 语义优先级，避免把 LiveXiv/Verified Subset 等比较列当作目标列。
- 路由器保守支持三字母全大写来源缩写；唯一来源的流程/工具/过滤/样本/变化问题启用来源内 lexical 与同节补证据，
  不改变无明确证据意图的普通检索。

UDA Table 6 与 LiveXiv Table 2 的真实 PDF 导出形态均已由离线回归测试验证；全套 unittest `236/236` 通过。
E2b 的 20 题已被消费，不能把本次修复后的结果作为未见泛化证据；下一步应使用未参与修复的新留出集复验。

### Phase F1 未见留出集冻结（2026-09-05）

新增 `manifest_phaseF1_unseen.json` 与 `cases_phaseF1_unseen.jsonl`，包含此前未进入基准的 TableRAG（EMNLP
2025）和 CURIE（ICLR 2025）两篇开放 PDF、12 道题（TableRAG 5 题、CURIE 7 题）。题目覆盖方法流程、
复杂表格、长上下文统计、跨页证据和两道 Figure 视觉题；每道题的页码、表号、单位、公式/坐标和
required facts 已回查 PDF，关键表格和 Figure 31 已做视觉核对。

PDF 外部文件核验：TableRAG 20 页，SHA-256=`f388e3af1397055b2c5fda06831f373c23826cda4b4a395a38070bcf3edee4b1`；
CURIE 48 页，SHA-256=`4416483d8380052f6fae90657c36a0857f33d3a2933162edaa007d4072841a07`。校验命令为：
`./venv/bin/python evaluation/validate_benchmark.py --manifest evaluation/benchmark/manifest_phaseF1_unseen.json
--papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers --verify-files --require-complete`，结果为 2 篇、12 题、
required facts 全部由 gold contexts 或显式别名支持。

本清单在任何 Chroma 重建、DeepSeek 生成或检索调参前冻结；F2 诊断后因证据对齐修复已改标为
`development-regression`，不再作为未见泛化证据。原始冻结内容仍由文件历史和本节记录保留。

### Phase F2 离线检索闸门（2026-09-05）

在仓库外临时内存索引上运行 2 篇 PDF、12 道题；未重建项目 `chroma_db`、未启动网页、未调用 DeepSeek。
BM25、Dense、基础 Hybrid 和开启全部受控证据开关的 Gated Hybrid 均实现目标论文路由 `12/12`、错误路由
`0`；图像题另计，不要求文字层覆盖。修正评估器使 fact/provenance 与 reranker/网页的表格 caption/header
证据一致，并为 CURIE PDF 的 `PV-speciific` 导出拼写增加用例级别别名后，结果如下：

| 配置 | @10 非图像完整覆盖 | @50 非图像完整覆盖 | @10 fact macro / micro | 备注 |
| --- | ---: | ---: | ---: | --- |
| BM25-lite | `9/10` | `10/10` | `0.771 / 0.849` | 词法基线 |
| Dense | `8/10` | `10/10` | `0.688 / 0.755` | 本地 `bge-small-zh-v1.5` |
| Hybrid-RRF | `9/10` | `10/10` | `0.771 / 0.849` | 未显示出相对 BM25 的提升 |
| Gated Hybrid | `9/10` | `10/10` | `0.812 / 0.887` | 路由、表格、图、公式、限制、分解和窗口均开启 |
| Gated Hybrid + BGE reranker | `9/10` | `10/10` | `0.812 / 0.887` | 平均单题约 `6.21s`，峰值约 `1.85GB` |

两道 CURIE Figure 31 `image_only` 题在文字层检索中均为 `0/2`；开启空间图证据只能正确定位 Figure 31，
不能读取图中坐标或判断蓝色框距离，必须留到视觉生成实验。F2 因此通过“文本/表格检索可用性”门槛，
但不支持把 Hybrid 或 reranker 改成线上默认，也不证明答案正确率；下一步是隔离生成烟测，并把图像题单独
交给视觉路径。

诊断 JSON 保存在仓库外：`/private/tmp/scirag_phaseF1_bm25_v2.json`、`scirag_phaseF1_dense_v2.json`、
`scirag_phaseF1_hybrid_v2.json`、`scirag_phaseF1_hybrid_gated_v2.json`、
`scirag_phaseF1_hybrid_reranker_gated_v2.json`。本步未运行 RAGAS。

### Phase F3 受控生成与视觉路径验收（2026-09-05）

F2 的表格问题促成了三处通用解析修复：组合延迟表头、跨单元格/继承行实体，以及按问题中命名的
任务分组筛选 Precision/Recall/F1 列；同时修正 PDF 导出中的 `PV-speciific` 表头断词。新增回归后，
核心、解析和基准测试共 `167/167` 项通过。修复使用了 F1 开发回归集，因此不能重新宣称未见泛化。

在仓库外的隔离 Chroma（两篇论文、`402` 块，含持久化 PDF）上，以 Hybrid、文档路由、查询分解、
parent-window、空间图证据和来源过滤固定配置运行 DeepSeek 两轮。10 道文本/表格题共 `20/20` 次
请求成功、`0` 次 API 错误；输出为 `/private/tmp/scirag_phaseF1_hybrid_answers_v2.jsonl`（SHA-256
`cac3d116043fdaeae46576c42f9036e4f2ac197be8fd0f1b6b8686648af35365`）。人工逐题复核为每轮
`8 correct / 2 partial / 0 incorrect`：表格数值和方法/统计题的核心事实均正确；一题漏写 Qwen/Exact
Match 限定词，另一题漏答 BIOGR 的 IoU `0.49`。词法金标准审计的低分（重复答案分别
`0.8267/0.8333` 与 `0.8017/0.8125`）受中文回答、英文 required facts 和表面别名限制，不能替代上述
人工语义判断；证据校验器将 8 个确定性表格答案标为 `structured_table_path`，说明该工具不是答案裁判。
文本答案的逐字稳定率为 `4/10`，但变体只出现在自然语言解释题；表格路径两轮稳定。

随后以 `SCI_RAG_VISION_ENABLED=true` 对两道 `image_only` Figure 31 题各运行两轮，输出为
`/private/tmp/scirag_phaseF1_vision_answers_v1.jsonl`（SHA-256
`7af3ada3a221144a4c22fd740dbbd1fac6b045163d865c5545f2cde16e925a56`）。视觉 API `4/4` 成功并正确
定位 `2025.iclr.curie.pdf` 第 44 页 Figure 31；模型识别“哪个预测框最接近”题为 `2/2` 正确，
但坐标题两轮均将绿色框的 `W=103.2` 误读为 `W=183.2`，所以 image-only 为 `1/2` 正确，不能
通过当前视觉闸门。该失败是像素/OCR 可靠性问题，不通过增加文本检索候选解决；后续若要启用视觉，
应先用新的图像题验证高分辨率/局部裁剪，并设定坐标容差与拒答标准。当前视觉功能继续默认关闭。

人工核对还发现 F1 用例 `tablerag-f1-03` 曾把论文原文 “more than 20 rows” 写成“至少 20 行”；现已
按 PDF 第 12 页纠正为“超过 20 行”，并同步把 required fact 改为 `more than 20 rows`。

本轮没有运行 RAGAS、没有启动 Gradio、没有写入项目 `chroma_db`，也没有提交或推送。F3 结果只证明
隔离环境中的受控生成和失败边界，不证明生产答案正确率或多论文泛化。

### Phase F4A 表格答案完整性修复（2026-09-06）

针对 F3 的两个 partial，完成两项通用修复：将 `IoU` 纳入多指标列选择；对问题中明确命名的非数值行单元格
（如 `Backbone=Qwen-2.5-72b`）保留为结构化答案限定，并在问题明确要求时从表注保留 `exact match` 指标。
没有加入论文或数值特判。

使用同一 402 块隔离库走真实 `app.query_knowledge` 路径复测：TableRAG Table 4 完整返回 ReAct/TableRAG
的 HeteQA 值、Qwen backbone 和 exact-match 指标；CURIE Table 7 完整返回 `IoU=0.49`、`Normalized
Distance Error=3.03`、`Relative Box Size=3.05`。新增回归后定向测试 `169/169`、全套测试 `241/241`，
基准清单校验、编译检查和 `git diff --check` 均通过。

本步仍未启动网页、未写入项目 `chroma_db`、未运行 RAGAS、未提交或推送。下一步是视觉实验或在代码冻结后
建立真正未参与修复的最终确认集，不能继续用 F1/F2 证明泛化。

### Phase F4B 视觉高分辨率对照（2026-09-06）

在不改正式视觉代码的前提下，用现有 10 道 image-only 挑战题和 F1 的 2 道 Figure 31 题，比较了 4×
渲染的 full+detail 输入。单轮 `12/12` 请求中，人工复核为 `9 correct / 2 incorrect / 1 empty`；结果为
`/private/tmp/scirag_phaseF4B_highres_r1.jsonl`（SHA-256
`7dcadb3d62d82a931fd7a94c6078c9e785678b598f1a18b0e067fd206b1275a9`）。高分辨率确实修正了 Figure 31
坐标题的 `W=103.2`，但同时使 DrugR 复杂结构图和 SciDQA 空间判断出错，并使另一轮 Figure 31 题返回
空答案；相对已有 full+detail 基线没有整体稳定收益。

因此不把高分辨率方案接入正式路径，也不启动第二轮 API 消耗。视觉继续保持 opt-in/默认关闭；后续只有在
新的图像题集合上验证局部裁剪、数字容差和失败回退均改善时，才重新考虑实现。该实验不证明通用视觉问答能力。

## 离线解析观察

| 文档 | 有文本页 | 总 chunks | `type=table` chunks | 自动识别 `table_number` | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| SciDQA | 18 | 112 | 6 | 6/6 | 当前解析器已关联表格后的 caption，识别 Table 1–5、7；论文自身没有 Table 6 主表块。 |
| Scientific Table LLM | 14 | 65 | 3 | 3/3 | 当前解析器已关联表格后的 caption，识别 Table 1–3。 |
| MgNO | 20 | 94 | 7 | 7/7 | Table 1–7 均已编号；Table 1 的分组表头和 Table 4 的独立单位列已规范化。 |
| AlphaFold 3 | 21 | 107 | 0 | 0/0 | 页面 1 的 DOI/作者元数据不再误建成 table chunk；仍有 3 页无可提取文本，Nature 版式中的 Extended Data/图形表格没有稳定转成 Markdown 表格。 |

这些数字是解析观察，不是数据库块数，也不是检索召回率。特别是“表格块存在”不等于
“表号、caption、行列结构都可供检索”。在实现 Hybrid/Rerank 或多模态前，应先为这些
差异建立可重复的解析验收测试。

## Phase 2 解析回归（2026-08-28）

本轮在 `sci_rag_core.py` 增加了离线回归覆盖：caption 位于表格前后、HTML/Markdown
装饰、跨列分组表头、PDF 断词（例如 `Darcy s` + `mooth`）、独立单位列（例如
`L2 Error (×10−2)`）、加粗标记与实体相邻、普通表首行保护，以及 DOI/作者元数据布局表的
排除。`tests/test_parser_regression.py` 的 8 个 fixture 测试和现有核心测试均通过。

随后用 `app.load_and_split_document()` 对四篇外部 PDF 做了只读冒烟：SciDQA 为
6 个表格块（编号 1、2、3、4、5、7），Scientific Table LLM 为 3 个（1、2、3），
MgNO 为 7 个（1–7），AlphaFold 3 为 0 个误识别的表格块。MgNO 的两个确定性单元格
查询分别返回 Table 1 的 `Darcy rough L2 = 0.339` 和 Table 4 的 `L2 Error (×10−2)
= 1.63`。这证明的是本地解析/单元格定位回归，不是检索召回、生成正确性或 RAGAS
泛化能力。

## 用例覆盖

`cases.jsonl` 当前有 53 个用例：DrugR 11 题、SciDQA 10 题、Scientific Table LLM 10 题、
MgNO 11 题、AlphaFold 3 11 题。新增用例覆盖：

- 文本事实、方法流程、训练/评估设置和限制条件；
- Markdown 表格中的行列数值与指标；
- 公式、网格/边界条件、模型架构和复杂度；
- 图注、Extended Data 相关样本量和多模态证据；
- 随机种子、数据规模和复现性声明。

`cases.jsonl` 中的 42 个新增用例已完成一次基于 PDF 的逐题人工核对。核心数值、表号、
页码和公式语义均与指定页面一致。本次修订补充了表格列名、单位和表注，统一了
`caption` 的中文术语，并移除了 `mgno-03` 中原文未直接声明的“最细层”推断。
这证明的是 gold 标注和参考证据的自洽性，不是检索召回、生成答案或 RAGAS 质量已经通过。

为避免中文 required facts 与英文论文证据被机械字符串匹配误判，当前用例允许逐事实
声明 `required_fact_aliases`。别名必须由人工写入，且校验器要求事实本身或其别名确实
出现在 gold contexts；没有使用模型或 embedding 充当事实等价裁判。DrugR 的 GRPO/RL
用例还补入了论文第 2 页实际证据，使 53/53 题的 required facts 都能由金标准核验。

表格上下文现在尽量包含 caption、表头、目标行和单位，但 PDF 解析器仍可能在真实入库时
丢失表号或列结构；因此正式基线前仍需建立解析回归测试和离线检索诊断。Nature PDF 的
3 个 Reporting Summary 页面没有文字层，只能通过渲染视觉读取；它们不属于当前 42 个用例
的证据页。

## 离线词法检索基线（2026-08-28）

新增 `evaluation/benchmark_retrieval.py`，用标准库 BM25-lite 在五篇论文的全局内存索引上
排名，不加载 Chroma、Embedding、Gradio 或外部 API。全局 top-k 结果如下：

| k | 参考片段覆盖代理 | 目标论文命中 | source page 命中 | Table N 命中 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.252 | 0.811 | 0.310 | 0.667 |
| 3 | 0.336 | 0.868 | 0.429 | 0.833 |
| 5 | 0.365 | 0.868 | 0.429 | 0.833 |
| 10 | 0.497 | 0.906 | 0.690 | 1.000 |

目标论文命中使用 `document_id` 只作评分过滤，不加入查询；参考片段覆盖使用英文词元
重叠阈值 0.6；`source_pages` 和显式 Table N 均来自标注。这些指标只能作为解析/检索
比较的基线，不能替代人工答案核验或 RAGAS。

使用本地缓存的 `BAAI/bge-small-zh-v1.5`（`HF_HUB_OFFLINE=1`）做对比后，dense-local
和 Hybrid-RRF 的全局结果为：

| 方法 / k | 参考片段覆盖 | 目标论文命中 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: |
| dense @1 | 0.132 | 0.774 | 0.286 | 0.222 |
| dense @5 | 0.330 | 0.943 | 0.548 | 0.444 |
| dense @10 | 0.443 | 0.981 | 0.762 | 0.667 |
| hybrid-RRF @1 | 0.208 | 0.830 | 0.286 | 0.500 |
| hybrid-RRF @5 | 0.481 | 0.943 | 0.619 | 0.778 |
| hybrid-RRF @10 | 0.525 | 0.981 | 0.786 | 0.833 |

上述 dense/Hybrid 参考片段数字是 required-fact 标注修订前的历史代理；当前精确结果见
下节。

## Required-fact 上下文覆盖（2026-08-28）

新增确定性事实覆盖统计后，BM25 / dense / Hybrid 的全局 @10 fact macro 分别为
`0.627/0.539/0.627`，fact micro 为 `0.599/0.537/0.592`，完整覆盖率为
`0.547/0.434/0.547`。Hybrid @5 的完整覆盖率 `0.472` 高于 BM25 的 `0.434`，但到 @10
两者相同；Hybrid 的目标论文和页级命中更高并没有自动转化成更多完整多事实上下文。

Hybrid @50 候选池的 fact macro/micro/完整覆盖率为 `0.849/0.844/0.792`。这支持下一步
做本地 reranker 的受控实验，因为候选中存在尚未排进前十的完整证据；但 @50 仍有
11/53 题不完整、其中 3 题零覆盖，所以切分/解析或查询扩展仍是独立前置问题。任何
reranker 都必须继续保留表号过滤和确定性单元格查找，且不能仅凭本文指标宣称答案正确。

## Cross-encoder 门槛实验（2026-08-28）

固定 Hybrid top-50 后，`BAAI/bge-reranker-base` 纯重排与“cross-encoder 排名 + 原 Hybrid
排名再次 RRF”两种策略的 @10 完整事实覆盖率均为 `0.698`。选择后者是因为它相对原
Hybrid 只让 `drugr-11` 一题下降，而纯重排有 4 题下降；其 table 类型事实完整覆盖也为
`1.000`。最终 RRF 策略 @10 fact macro/micro 为 `0.785/0.776`，比原 Hybrid
`0.627/0.592` 高。

该结果只达到“允许默认关闭地接入应用做人工 A/B”的门槛。53 题中仍有 16 题在 @10 未
完整覆盖；CPU 平均重排延迟约 2.73 秒、P95 3.31 秒、进程峰值约 2.20 GB，尚未验证并发
资源、真实五论文 Chroma 或生成答案质量。

## Phase 6.2 坐标图形文字观察（2026-08-30）

默认解析结果仍保持本文前面的表格不变。仅在 `include_spatial_figures=True` 时，解析器
额外读取 `Page.get_text("blocks", sort=True)` 的
`(x0, y0, x1, y1, text, block_no, block_type)` 字段，并把 Figure caption 上方的短文字
按 x 中心排序、以归一化坐标写入 `type=figure` 证据块。五篇论文分别得到 DrugR 5、
SciDQA 7、Scientific Table LLM 1、MgNO 5、AlphaFold 3 主图 5，共 23 个图块；总语料
由 479 增至 502 块。

AlphaFold 3 第 2 页 Figure 1 的实际文字层把 `PDB / protein–RNA` 与 `n = 25`、
`PDB / protein–dsDNA` 与 `n = 38`、`CASP15`/`RNA` 与 `n = 8` 放在相互重叠的 x 区间；
相邻的 `Glycosylation`/`n = 28` 位于另一 x 区间。原扁平 picture text 丢失了这种空间关系。
新证据仅在显式图号查询中旁路注入，并移除同一回答中的扁平 picture-text 干扰；普通检索
候选不包含 figure chunks。

## 扩展清单（默认关闭）

桌面已有的 `2026.findings-eacl.12.pdf` 为 Findings of EACL 2026 的开放获取论文
THINKNOTE（19 页，SHA-256 为 `befaff6facc1b9776c9a095b03ecfb0fe78ee5a07d6454201f14de4d55c77b30`）。
`manifest_expanded.json` 通过 `base_manifest` 继承五论文基线，并追加 13 道方法、公式、表格、
图和限制题；可用 `validate_benchmark.py --manifest ... --verify-files` 复核文件与标注。
它用于扩大领域覆盖，不改写默认五论文/53 题的历史指标，也不代表已完成生成或语义评估。

这不是图片理解：PDF 图片仍未持久化，纯扫描页或只存在于像素中的 Extended Data 图不会
生成坐标证据。当前结果只支持把该功能作为默认关闭的 born-digital PDF 实验路径。

## Phase G 原未见确认集（现为开发回归集，2026-09-06）

经用户批准，从 ACL Anthology 下载三篇此前未进入本项目基准的免费 PDF，保存于仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：MT-RAIG（ACL 2025，31 页，SHA-256
`a260e15eb654f9bd4757e2a31a1bc08d6a1759a22cdf75eea4b58f26ef8504ac`）、WikiMixQA（Findings ACL
2025，18 页，SHA-256 `43c62d2956fa641d0a5e3add15ab1eefa24c4d2e00a5390650772d92094a3a50`）和
TableEval（TRL 2025，34 页，SHA-256 `033f4e0d1d59d8ec19003c883db106c5f35910497950f12c9b829544068dea9d`）。
TableEval 的 `pdfinfo` 报告交叉引用警告，但项目使用的 PyMuPDF 成功读取 34 页和 98,692 个字符；未将该警告
误判为解析失败。

新增 `manifest_phaseG_confirmation.json` 与 `cases_phaseG_confirmation.jsonl`，原先冻结为 3 篇论文、20 道题：
MT-RAIG 8 题、WikiMixQA 6 题、TableEval 6 题。题目覆盖任务流程、数据集统计、复杂表格、检索指标、评估协议
和表格表示；原始冻结 manifest SHA-256=`0f374928b07e2400a300ebccdc3c8882889451e174fd022f2d75e5491b2389a2`，
现行 development-regression manifest SHA-256=`376ca12ad291b1c7678f4245162ebe77189f7a45cf7da2670f2c5a3318e0ae72`，
cases SHA-256=`92a7f2d9223a9872ae673833208e5dc2b740e3e5a93f35be1fba018e1e27d57d`。离线校验命令通过：
3 篇、20 题、required facts 全部由 gold contexts 或显式别名支持。

Phase G 生成 trace 已被查看并用于本轮通用修复，因此 manifest 现标为 `development-regression`；原题目、答案、
required facts 和别名不变，不再把它作为未见确认证据。重新计算真实送入模型的 `contexts` 后，20 题的事实覆盖为
`11/20 full、7/20 partial、2/20 zero`，macro/micro=`0.7592/0.7157`。此前的离线候选诊断 `13/20`（@10）和
`17/20`（@50）仍保留，但只能解释候选池覆盖，不能代替应用生成路径的上下文覆盖。

在仓库外隔离 Chroma `/private/tmp/scirag_phaseG_db.IDhuze` 中重建 472 个块（MT-RAIG 211、WikiMixQA 89、
TableEval 172）。Gated Hybrid（文档路由、查询分解、结构化表格、公式/限制证据、parent-window）检索结果：
@10 完整事实覆盖 `13/20`，@50 为 `17/20`；目标文档路由 `20/20`，错误路由为 0。已缓存 BGE reranker
（candidate-k=20、max-length=256、CPU）@10 完整覆盖 `12/20`，没有超过 Gated Hybrid；不切换线上默认。
这些是检索覆盖代理，不是答案正确率。

使用当前源码和固定隔离配置对 20 题生成两轮。首次沙箱运行出现 34 次 `Connection error`，联网权限续跑后仅重试
失败行，最终 `40/40` 成功；配置、上下文和 metadata 两轮均 `20/20` 稳定，答案逐字稳定 `3/20`，说明措辞有明显
随机性。trace 仅保存在仓库外 `/private/tmp/scirag_phaseG_answers_hybrid.jsonl`（SHA-256
`f6ac441fee9bd6bfd8c6e84ca177fd21e2763ced6151282775901a35e5550c27`），未写入项目数据库、未运行 RAGAS。

逐题人工语义复核（两轮结论一致）为 `14 correct / 4 partial / 2 incorrect`：

- correct：`mtraig-g-01/02/06/07/08`、`wikimix-g-01/02/03/06`、`tableeval-g-02/03/04/05/06`；
- partial：`mtraig-g-04`、`wikimix-g-04/05`、`tableeval-g-01`；
- incorrect：`mtraig-g-03`（误以为四类问题统计未提供）和 `mtraig-g-05`（拒答代理模型、问题数量及三种方法）。

原确认集未达到预设门槛（至少 `17/20` 正确且不超过 1 个错误），因此当前系统仍只能称为可用研究原型，不能
宣称已证明泛化或切换默认 reranker。它现仅可用于开发回归；完成修复后必须另建全新的未见确认集，否则应停止
扩展功能并接受当前能力边界。

Phase H1 的首个通用修复已在同一隔离库、假客户端的真实应用路径完成上下文检查：复合问题现在按问号/句号拆分，
并使用数量、占比、标注、代理模型、表格集合和人类引导等通用中英文别名。`mtraig-g-03` 的真实上下文事实覆盖
从 `0/12` 提升到 `12/12`，`mtraig-g-05` 从 `0/5` 提升到 `5/5`；20 题真实上下文覆盖由原来的
`11/20 full、7/20 partial、2/20 zero` 变为 `13/20 full、7/20 partial、0/20 zero`（macro/micro=`0.8592/0.8824`）。
这只是检索上下文检查，尚未重新调用 DeepSeek，不代表答案正确率或泛化结论。

## Phase H3 新论文确认集（现为开发回归集，2026-09-06）

经用户批准，从 ACL Anthology 下载两篇此前未进入任何基准的免费 PDF，保存于仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：MEBench（EMNLP 2025，14 页，SHA-256
`399951b507c5cdd7de37c169fbff9e8d36b4d3d9b37d9fd911d5e388ad92d7af`）和医学问答解释基准（NAACL 2025，37 页，
SHA-256 `7a136b45c53d1b4d36696c5525ec57051e0e0245f677825755635c38bdacabbe`）。

新增 `manifest_phaseH3_confirmation.json` 与 `cases_phaseH3_confirmation.jsonl`，共 12 道冻结题（每篇 6 题），
覆盖数据集统计、问答构建流程、复杂表格、人工评估和污染审计。校验器通过 2 篇、12 题，required facts 全部由
gold contexts 或显式别名支持。该集在生成后按协议改标为 `development-regression`，题目和金标准不因失败修改。

在仅含两篇新论文的隔离 Chroma `/private/tmp/scirag_phaseH3_db_20260906` 中重建 304 个块。默认 Hybrid、文档路由、
查询分解、parent-window、空间图证据和来源过滤的真实应用路径，离线前置覆盖为 `11/12 full、1/12 partial、0/12 zero`，
来源路由 `12/12`；唯一缺口是 Medbullets“五个答案选项”未进入送模上下文。

两轮 DeepSeek 生成 `24/24` 成功，provenance、配置、上下文和 metadata 均按 case 稳定。逐题人工复核每轮为
`8 correct / 3 partial / 1 incorrect`：MEBench Table 3 的 `GPT-4 + RAG` 行因跨列表头拆分返回错误结构化答案；
Medbullets 题漏答“五个选项”，Table 2 题漏答 GPT-4 的 MedQA-4=`78.63`。因此 H3 未达到 `≥85% correct 且 ≤1 incorrect`
门槛，不能宣称新论文泛化或切换线上默认检索。trace `/private/tmp/scirag_phaseH3_answers_v1.jsonl` 的 SHA-256 为
`34fc443917860acde0f8583ec56fcf72f04cf22bc027abec79edc1d6cdc00914`；逐题记录见
`reviews_phaseH3_confirmation_v1.jsonl`，结果未写项目 Chroma、未运行 RAGAS。

## Phase H4 修复后开发回归（2026-09-06）

H3 失败样本仅用于开发回归，题目和金标准未改动。修复覆盖跨单元格表头、重复数据集/Prompting 分组、中文多值限定
和“选项”来源内证据别名；没有加入论文专用规则。全套离线 unittest `248/248`，`py_compile` 与 `git diff --check` 通过。

在重新解析两篇论文得到的 304 块隔离 Chroma 上，Table 3 的 `GPT-4 + RAG` 四列、Table 2 X→Y 下 GPT-4 的
`78.63/60.71/67.32` 以及 Medbullets 的 `five answer choices` 均进入真实送模上下文；12/12 题来源过滤正确，
上下文事实覆盖 `12/12 full`。两轮 DeepSeek 共 `24/24` 成功，配置/上下文/metadata 稳定；人工复核按 case 为
`11 correct / 1 partial / 0 incorrect`。partial 是 MEBench Table 2 的拆分表头展示问题，三类总数本身正确。

trace 只保存在仓库外 `/private/tmp/scirag_phaseH3_answers_h4_v1.jsonl`（SHA-256
`39dd5acf493aea92728de4f15a9f3c1a7a15bdadf3d780f118a7c77710a7a561`），复核记录为
`reviews_phaseH3_confirmation_h4_v1.jsonl`。该结果只证明 H3 开发回归通过，不证明未见论文泛化，也不改变线上默认配置；
下一步需建立全新的 H5 未见确认集。

## Phase H5 新论文确认集（现为开发回归集，2026-09-06）

经用户批准，从 ACL Anthology 下载两篇此前未进入任何基准的免费 PDF，保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：SciAssess（Findings of NAACL 2025，23 页，SHA-256
`acab525af0ce2576e58658a76b56354e3f15e94aa507e0b1f2ad32b0207c67df`）和 YESciEval（ACL 2025 Long Papers，35 页，SHA-256
`40f201887a3f052be7afde16547ff29d2d4ab8d5cb2e2f2029b4c6f842d23058`）。新增清单和 12 道题（每篇 6 题）均通过离线校验。

H5 在生成前已用于离线检索诊断和通用表格/路由修复，因此按冻结协议现标为 `development-regression`，题目和金标准未改动。
两篇 PDF 在仓库外隔离 Chroma 中重建为 432 块；真实应用路径的来源路由为 `12/12`，实际送模上下文 required-fact 覆盖为
`12/12 full`（macro/micro=`1.0000/1.0000`）。

当前修复后两轮 DeepSeek 生成 `24/24` 成功，context IDs、运行配置和 provenance 按 case 稳定。严格词面答案审计仅为
macro/micro=`0.6606/0.7073`，不作为语义准确率；人工逐轮复核为第一轮 `12 correct / 0 partial / 0 incorrect`、第二轮
`10 correct / 2 partial / 0 incorrect`。两个 partial 是 SciAssess 层级定义和 YESciEval rubric 分组在第二轮被保守拒答，
不是检索缺失；逐题记录见 `reviews_phaseH5_confirmation_v2.jsonl`，trace 仅保存在仓库外
`/private/tmp/scirag_phaseH5_answers_v2.jsonl`（SHA-256 `9df5a445fce088b5d7d09a0a2d9647d5e5bbfa9f0f5bca851d6486d3ac35ffc9`）。

按“每个 case 两轮都完整”保守统计为 `10/12 correct、2/12 partial、0/12 incorrect`，低于 `≥85%` 的未见确认门槛；因此不宣称泛化，
不切换线上默认 reranker，也不再用 H5 调参。下一步应建立真正未参与诊断的新 H6 确认集；H5 仅保留作开发回归。

## Phase H6 开发回归集（初始冻结，2026-09-06；后续降级）

为避免继续在 H5 上调试，新增两篇此前未进入任何基准的开放论文：CAQA（ACL 2025，23 页，SHA-256
`0e7ec39f1afc80b90692b66b32ddd16fe394902affd163ed61351366eb42ea25`）和 MiMoTable（COLING 2025，13 页，SHA-256
`c3b61245bcea99174ad81ef1113a6e157b68b6b3cec75e8a59d7331a075e798a`）。两篇 PDF 均来自 ACL Anthology 官方免费链接，保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。

`manifest_phaseH6_confirmation.json` / `cases_phaseH6_confirmation.jsonl` 共 12 道题（每篇 6 题），覆盖归因类别与复杂度、KG 构建流程、
表格统计、meta operations 和人工质量控制。清单初始在检索、生成和调参前冻结为 `frozen-unseen`；随后隔离检索 gate 暴露通用质量控制别名缺口，按协议改标为 `development-regression`，当前 manifest SHA-256=
`14e036d7b28c71bd9a43556da03208a545653a189ea7d3069597f09a15ef135c`，cases SHA-256=
`c8976fc51e4fae8d7ae4f0504b01cded8e8caca9c98b67e978e571a558ef8a5a`。

初始真实应用路径 gate 为 `10/12 full`；增加通用“质检/质量控制”来源局部别名并兼容换行断开的 `independent - facts` 后，在仓库外隔离 256 块 Chroma 上复跑为 `12/12 full`。随后两轮 DeepSeek `24/24` 成功，人工复核为 `20 correct / 2 partial / 2 incorrect`，按每个 case 两轮均完整为 `10/12`。两个 partial/incorrect 均来自 MiMoTable：Table 4 遗漏 Simple=33.6%，Figure 6 错配 Lookup/Compare/Visualize 数值；不宣称泛化或切换线上默认检索。trace 仅保存在仓库外 `/private/tmp/scirag_phaseH6_answers_regression_v1.jsonl`（SHA-256 `8e024cb9b6ad0ad59e08a976ef5bc5da9bbebba897d13de497cd77c8cc8c178a`），逐题记录见 `reviews_phaseH6_regression_v1.jsonl`。

## Phase H7 开发回归集（初始冻结，后续降级，2026-09-06）

H7 选用两篇此前未进入任何基准或诊断的 ACL 2025 Long Papers：TC–RAG（27 页，RAG 状态/记忆/公式/效率）和 ChartCoder（16 页，图表多模态与代码生成）。PDF 均来自 ACL Anthology 官方免费链接，保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。

`manifest_phaseH7_confirmation.json` / `cases_phaseH7_confirmation.jsonl` 共 12 道题（每篇 6 题），覆盖表格、公式、方法流程、效率指标和图表多模态统计。清单初始在任何检索、生成或调参前冻结为 `frozen-unseen`，随后 context gate 发现通用表头、整行多列、公式证据和多事实召回缺口，按协议改标为 `development-regression`；当前 manifest SHA-256=`f92c5082083b1c736dfb40deac3c1c80508ebf23011e3fa12710e565b4e390b7`，cases SHA-256=`e3ad5c19ab528ba8bb1b2d607e3b69acb479947c0def20b20a6e0574c8a6d290`。

在仓库外隔离 370 块 Chroma 上，真实应用路径 required-fact gate 为 `5/12 full`，fact macro/micro=`0.6457/0.6705`。失败集中在 TC–RAG 的跨列拆分表头、整行多列查询只回传单元格、公式恢复和实验设置的多事实召回；ChartCoder 六题均 full。H7 不作为未见泛化证据，修复完成后需另建 H8。

随后补充通用三层/居中表头合并、中文平均指标别名和“分别”多列行抽取；TC–RAG Table 2 已能直接返回四个平均值。H7 fix2 离线 gate 为 `5/12 full`、fact macro/micro=`0.6576/0.6818`，仅作开发回归诊断，未调用 DeepSeek，H7 仍不作为未见证据，后续另建 H8。

## Phase H8 开发回归集（初始冻结，后续降级，2026-09-06）

H8 选用 ACL 2025 Long Paper ChainRAG 与 ACL 相关 MAGMaR 2025 两篇免费 PDF，新增 12 道题，覆盖多跳实体改写、句图检索、表格指标及 PDF/视频多模态管道。清单初始在检索前冻结为 `frozen-unseen`，随后隔离 gate 暴露多事实、跨列统计表和模态流程细节的通用覆盖缺口，现改标为 `development-regression`。

隔离 158 块 Chroma 的真实应用路径 required-fact gate 为 `6/12 full`，fact macro/micro=`0.7449/0.7500`；未调用 DeepSeek，不作为未见泛化证据，后续另建 H9。

H8 回归修复覆盖通用多级表头续行、嵌套方法行外层模型、阈值括号/小数空格和来源内量化/设置/模态分布别名；用例事实改为可由论文原文逐项核验的原子事实。在同一 158 块隔离库、假客户端和来源过滤下复测，真实送模上下文 required-fact gate 为 `12/12 full`，macro/micro=`1.0000/1.0000`。ChainRAG Table 1 保留 `GPT4o-mini / Ours (CxtInt)` 限定，MAGMaR 阈值与模态统计均进入上下文；未调用 DeepSeek。H8 仍仅作开发回归，不证明未见泛化，下一步建立全新的 H9 确认集。

当前 H8 文件 SHA-256：manifest=`fb1a4b69d1ceda09f56203269fe5d367ff450ad05b54429955914c87d936019b`，cases=`ecc61faad0c583e9debd2133935cfc0a280e4b8b77e3c514836cc277f9ff3d66`。

## H7 最后一轮通用修复（2026-09-07）

表格行过滤现仅接受实际 `Dataset` 单元格匹配的 `在/on` 限定词，避免将 CMB、MMCU 等横向表头误当作数据集行；同时补充“评价指标/加速策略”及英文检索别名。全套离线 unittest 为 `259/259`。

同一 370 块 H7 隔离库的真实 `query_knowledge`（Hybrid、路由、查询分解、可选证据保护、假客户端）送模上下文 required-fact gate 为 `8/12 full、4/12 partial、0/12 zero`，fact macro/micro=`0.9000/0.8750`。TC–RAG Table 1 已完整，ChartCoder 六题保持完整；该结果仍只是开发回归上下文覆盖，不证明答案语义正确或未见论文泛化。H7 达到停损点，下一步建立新的 H9 确认集。

## H3–H8 配置 A/B 验收（2026-09-07）

A 为默认 Dense（文档路由、查询分解关闭），B 为 Dense+文档路由+查询分解；固定各批次隔离库、离线嵌入模型和 `context_k=10`，A/B 各重复 3 次且结果一致。B 的 full 案例数相对 A：H3 `9→11`、H5 `7→9`、H6 `8→11`、H7 `7→8`、H8 `7→8`；各批次 required-fact macro/micro 均提高，目标论文和显式 Table N 命中均保持 `100%`。

页级命中 H3/H5/H6 提高、H7 持平，但 H8 从 `12/12` 降为 `10/12`，未通过严格的全指标无回退门槛。因此线上默认仍为 Dense，B 仅作为 opt-in；Hybrid 不因本次验收切换默认。

## H7 两轮生成语义闸门（2026-09-07）

同一 370 块隔离库上完成 H7 12 题×2 轮真实生成，最终 `24/24` 成功。trace 的 provenance 完整率为 `12/12`；重复 case 的 config、context ID、metadata 和 source fingerprint 稳定率均为 `1.0`，答案逐字稳定率为 `5/12`（7 个 case 仅措辞变化）。严格词面事实审计 macro/micro=`0.6733/0.6534`，不作为语义准确率。

人工按 PDF、gold 和实际上下文逐题复核（记录见 `reviews_phaseH7_generation_v2.jsonl`）：第 1 轮 `5 correct / 6 partial / 1 incorrect`，第 2 轮 `5 correct / 5 partial / 2 incorrect`。因此未达到每轮 `≥10/12 correct` 且 `≤1 incorrect` 的生成闸门；H7 保持 `development-regression`，不证明泛化，不切换默认配置。H9 应使用未参与 H7–H8 修复的新论文，下载前先完成候选审阅。

## Phase H9 新论文确认集（首轮后降级为开发回归，2026-09-07）

经用户批准，从 ACL Anthology 下载三篇此前未进入基准的免费 2025 PDF，保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：LongTableBench（Findings of EMNLP，39 页，SHA-256
`0de0b2f87eefff392a19ceb19bbe0b52124db13d79c2664fe9615586cc9a95a7`）、Table-R1（EMNLP，20 页，SHA-256
`ac314de535c21e9f31e0016267499ec2cac6308556d14e86a9fa68f61acedf6a`）和 Query-Driven Multimodal GraphRAG（Findings of ACL，21 页，SHA-256
`4b5c4f44d5bb9dd32f0246b1c1b2e03ba73442edfd7b931d3db07900c0720932`）。三篇 PDF 的文字层逐页可读，代表页及关键表格/公式完成视觉核对。

新增 `manifest_phaseH9_confirmation.json` / `cases_phaseH9_confirmation.jsonl` 共 12 道题，覆盖长上下文表格、表格推理训练和多模态 Graph-RAG；初始清单校验通过。
在 475 块仓库外隔离 Chroma 上，真实 `app.query_knowledge`（Hybrid、文档路由、查询分解、parent-window、公式/图形证据保护、来源隔离；假客户端）送模上下文 gate 为 `6/12 full、5/12 partial、1/12 zero`，required-fact macro/micro=`0.7593/0.8052`。全库不传 `source_filter` 的结果相同。

失败是共享实现缺口，不是三篇论文专用事实缺失：多列整行表格被截断或只抽取单元格；无题注编号的 Table 1 无法稳定按显式表号定位；GraphRAG Definition 4 公式跨块缺少尾部条件；Table-R1 的数据集计数和多任务奖励段落未同时进入有效上下文。按冻结协议，H9 已改标为 `development-regression`，不再作为未见泛化证据，也不调用 DeepSeek 生成；下一步只做一次通用根因修复与 H9 回归，不写论文特例。

## H9 通用修复回归（2026-09-07）

修复仅针对共享解析/审计问题：堆叠表头与远距离单题注关联、`<br>` 并行单元格对齐、显式多列查询的限定词裁剪、PDF 公式续行恢复，以及同一结构化行内的有序事实匹配（不跨 Markdown 行）。新增回归测试后，全套离线 unittest 为 `264/264`，`py_compile` 与 `git diff --check` 通过。

在新的仓库外隔离 Chroma `/private/tmp/scirag_phaseH9_db_fix2_20260907` 中重建 `476` 个块，使用 Hybrid、文档路由、查询分解、parent-window、公式/图形证据保护和假客户端复测；来源过滤与不传 `source_filter` 两种真实 `app.query_knowledge` 路径均为 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`。GraphRAG Definition 4 的跨行公式尾部、Table-R1 四个数据集计数及 LongTableBench 四列 Table 2 数值均进入最终上下文。

H9 仍按协议保留为 `development-regression`：本结果证明当前开发回归的送模上下文完整，不证明答案语义正确、未见论文泛化或线上默认配置应切换；本轮未调用 DeepSeek，未修改项目 `chroma_db`，隔离库和 trace 均在仓库外。

随后在同一隔离库完成两轮 DeepSeek 生成，`24/24` 成功；trace 仅保存在仓库外 `/private/tmp/scirag_phaseH9_answers_regression_20260907.jsonl`（SHA-256=`7890f6bff824764016ade60cede8be8dea37883bead21fb1e3de31107809b097`）。逐题 PDF/gold/context 复核记录为 `11 correct / 1 partial / 0 incorrect`（两轮合计 `22/24` correct、`2/24` partial），唯一 partial 是 Definition 4 题的公式完整性与不必要拒答。词面答案审计 macro/micro=`0.8536/0.8571`，只作表面信号；送模 contexts 仍为 `1.0000/1.0000`。H9 未达到未见确认条件，因为修复前已使用该集定位缺口。

## Phase H10 新论文确认集（首轮后降级为开发回归，2026-09-07）

经用户批准，从 ACL Anthology 下载三篇此前未进入基准的免费 2025 PDF，保存在仓库外
`/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`：SCITAT（Findings of ACL，23 页，SHA-256
`8609be38bbcc0a9224429578c3fcf49143635865d5ed317135681c39f87a4001`）、REAL-MM-RAG（ACL Long Papers，24 页，SHA-256
`6876cc59861da690da83978d5897a44bbf87bbc1f8a6a8145966002c41bb98ad`）和 RealHiTBench（Findings of ACL，33 页，SHA-256
`3ec01a3071f2331566460101f97b7ad32062c54ff4e5d263352948e40587f0d2`）。三篇 PDF 文字层逐页可读，关键表格和代表页完成视觉核对；SCITAT 第 2 页的示例表格属于图像内容，未作为正文表格证据使用。

新增 `manifest_phaseH10_unseen.json` / `cases_phaseH10_unseen.jsonl` 共 12 道题（每篇 4 题），初始冻结后进行一次隔离 Hybrid 检索诊断。当前清单 SHA-256：manifest=`72bf4b43d0505a6846cc1113a27dff2abc4ac77d8e6cbd71c924307dcddbdafe`，cases=`2f5a40f313554951976a4a8bc0153c70114dd3d5af27aaa119c9b5c04f829575`。

仓库外隔离诊断 `/private/tmp/scirag_phaseH10_retrieval_20260907.json` 使用 Hybrid、文档路由、查询分解、结构化表格/公式/图形证据保护和 parent-window；@10 为 `5/12 full`，@50 为 `8/12 full`，@50 required-fact macro/micro=`0.694/0.658`。目标文档命中 `12/12`，文档路由 `10/10` 正确；失败集中在 SCITAT/REAL-MM-RAG 宽表统计和多列值覆盖。

按冻结协议，H10 已改标为 `development-regression`，不再作为未见泛化证据；本步未调用 DeepSeek、未修改项目 `chroma_db`、未提交或推送。后续如继续修复，应先另建 H11 未见确认集，不能围绕 H10 反复调参。

### H10 表格标签与并排表解析修复（2026-09-07）

公共解析器现在支持 `Table S1` 等字母数字表号，并为 `pymupdf4llm` 合并的横向宽表按重复表头和列宽通用拆分。SCITAT 第 4 页已生成独立 Table 3/Table 4，REAL-MM-RAG 第 15 页的补充表元数据为 `table_label=S1`；普通整数表号行为保持兼容。

新增回归测试后全量离线 unittest 为 `266/266`，`py_compile` 与 `git diff --check` 通过。临时隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 的真实 `app.query_knowledge`（Hybrid、路由、查询分解、parent-window、假客户端）为 `9/12 full、1/12 partial、2/12 zero`。人工检查确认 Table 3、Table 4 和 Table S1 已进入上下文；剩余缺口来自 H10 事实审计字符串没有表达表格的“行标签—列值”关系，并非目标表块缺失。

H10 继续保持 `development-regression`，不用于泛化或答案正确率结论；本轮未调用 DeepSeek、未修改项目 `chroma_db`、未提交或推送。下一步应先改进通用表格事实审计口径，再进行生成验证。

### H10 表格事实审计口径修正（2026-09-07）

事实覆盖审计现在能在单个 Markdown 表格行内关联行标签、列标题和单元格值，并兼容 PDF 去掉千位逗号以及 `Table`/`Tables` 形式差异；不会跨行拼接事实。全量离线 unittest 为 `268/268`，`py_compile` 与 `git diff --check` 通过。

复用同一临时隔离库重跑真实 `app.query_knowledge`（Hybrid、路由、查询分解、parent-window、假客户端）后，H10 为 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`，无 partial/zero。H10 仍是 `development-regression`，不证明答案语义正确或未见泛化；尚未调用 DeepSeek，也未修改项目 `chroma_db`。

同一批次的离线 Hybrid 检索诊断在 `@50` 达到目标文档、来源页、表号和 required-fact `12/12`，macro/micro=`1.0000/1.0000`；`@10` 为 `9/12 full`，保留低召回预算下的真实边界。该诊断 JSON 仅保存于仓库外 `/private/tmp/scirag_phaseH10_retrieval_parserfix_20260907.json`。

H10 定向生成复核中，SCITAT Table 4 与 REAL-MM-RAG Table S1 两题答案正确。H9 GraphRAG Definition 4 公式题在上下文已完整的情况下仍出现不必要拒答和公式混排；通用相邻公式提示及多字符标签识别的单次重试未消除该问题。该项记录为生成/公式 provenance 边界，不再追加论文特例；H10 尚未进行全量生成。

### H10 单轮生成回归（2026-09-08）

在仓库外隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 上完成 12 题单轮 DeepSeek 生成，`12/12` 调用成功、无 API 错误。人工对照 PDF、gold 和实际 contexts 为 `10 correct / 1 partial / 1 incorrect`：partial 是 SCITAT 四类 reasoning type 漏列三个总类，incorrect 是 RealHiTBench Table 2 的模型/行列定位错误。逐题记录见 `reviews_phaseH10_generation_v1.jsonl`，trace 位于仓库外 `/private/tmp/scirag_phaseH10_answers_20260908.jsonl`（SHA-256=`935ae355f9c58194aab24f0208cb5dc54b944a1528cb2fe9d2abb8b4fc0d1e07`）。该结果仅达到开发回归门槛，不证明 H10 泛化或默认检索切换。
