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

`cases.jsonl` 每行一个用例。当前用例使用引用形式：

```json
{"case_id":"drugr-01","document_id":"drugr-2602-08213-v1","source_testset":"../test_questions.json","source_case_id":1}
```

新论文可以使用内联形式，至少包含：

```json
{"case_id":"paper-b-01","document_id":"paper-b","question":"...","ground_truth":"...","required_facts":["..."],"contexts":["..."]}
```

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

2026-08-28 在当前 5 篇/53 题基准上的全局基线为：

| k | reference context | target document | source page | table number |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.252 | 0.811 | 0.310 | 0.667 |
| 3 | 0.336 | 0.868 | 0.429 | 0.833 |
| 5 | 0.365 | 0.868 | 0.429 | 0.833 |
| 10 | 0.506 | 0.906 | 0.690 | 1.000 |

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

同一基准的全局结果（仅作检索比较）如下：

| 方法 / k | 参考片段覆盖 | 目标论文命中 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: |
| BM25 @1 | 0.252 | 0.811 | 0.310 | 0.667 |
| BM25 @5 | 0.365 | 0.868 | 0.429 | 0.833 |
| BM25 @10 | 0.506 | 0.906 | 0.690 | 1.000 |
| dense @1 | 0.132 | 0.774 | 0.286 | 0.222 |
| dense @5 | 0.330 | 0.943 | 0.548 | 0.444 |
| dense @10 | 0.443 | 0.981 | 0.762 | 0.667 |
| hybrid-RRF @1 | 0.208 | 0.830 | 0.286 | 0.500 |
| hybrid-RRF @5 | 0.443 | 0.943 | 0.667 | 0.778 |
| hybrid-RRF @10 | 0.525 | 0.981 | 0.810 | 0.833 |

本次结果不支持直接把 Hybrid 切换为线上默认：它提升了参考片段和页级代理，但在
`Table N` 命中上仍低于 BM25 的 top-10；还需要保留显式表格的确定性过滤，并在更大
测试集上确认收益。
