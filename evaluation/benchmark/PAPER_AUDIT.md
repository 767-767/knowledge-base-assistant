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
