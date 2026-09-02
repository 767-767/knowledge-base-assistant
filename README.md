# Sci-RAG

面向科学论文的 RAG 原型，支持 PDF/TXT/DOCX 解析、Markdown 表格结构化、ChromaDB
检索、DeepSeek 生成、Gradio UI，以及离线检索/答案审计和可选 RAGAS 评估。

当前定位是研究型可用原型，不是生产系统。默认检索仍为 dense；Hybrid、cross-encoder、
文档路由和窗口扩展均是可控开关；PDF Figure 的 VLM 路径仅作为默认关闭的 opt-in 实验。
OCR、图片向量索引、通用工具调用和 Graph-RAG 尚未实现。

## 快速开始

```bash
cd /Users/qinleqi/Desktop/knowledge-base-assistant
source venv/bin/activate
python3 test_setup.py
python3 -m unittest discover -s tests -v
python3 app.py
```

首次配置时复制 `.env.example` 为 `.env`，仅在启动 UI 或调用生成模型时填写
`DEEPSEEK_API_KEY`。`.env`、本地数据库、PDF、模型缓存和 `/tmp` 评估产物都不应提交。

`app.py` 的导入本身无副作用；模型、API 客户端、ChromaDB 和 Gradio 只在
`create_runtime()` 或 UI 入口中初始化。

## 文档与表格能力

- PDF 使用 `pymupdf4llm` 读取文字层，不写出或持久化图片。
- TXT 和 DOCX 走本地解析；DOCX 表格会转为 Markdown。
- 表格独立保存为 canonical table chunks，并保留表号、caption、页码和来源 metadata。
- 明确给出 `Table N + 行 + 列` 的问题优先执行确定性单元格查找，不调用生成模型猜值。
- 显式表号无法匹配时不会借用其他表格。
- born-digital Figure 坐标文字可选启用，但这不是 OCR 或像素级图片理解。
- 原始 PDF 文字层中被 Markdown 漏掉的等式会作为独立 `formula` 证据块保存；普通 dense/Hybrid
  候选不会使用它们，只有明确公式问题才在同源范围内补充。

上传后如需验证新 metadata，应使用新建的隔离数据库。旧 ChromaDB 不会自动迁移或重建。

## 网页回归

一键使用临时 ChromaDB 导入默认论文并启动正常 UI：

```bash
bash scripts/launch_phase1_ui_test.sh
```

使用其他 PDF：

```bash
bash scripts/launch_phase1_ui_test.sh /absolute/path/to/paper.pdf
```

只验证现有数据库兼容性（脚本会先复制数据库，不写原库）：

```bash
bash scripts/launch_phase1_ui_test.sh --existing
```

完成后在终端按 `Ctrl+C`。默认论文的最小验收项：

- 上传、问答、导图、测验四个页面可用；
- Table 2 / DrugR* / Overall Optimization Score 返回 `0.2060`；
- Table 2 / DrugR* / Target property F1 返回 `0.3404`；
- Table 1 / DrugR / Overall Optimization Score 返回 `0.2712`；
- 普通正文问题和第二篇文档上传不受表格路径影响。

## 检索配置

环境变量的完整默认值和说明见 `.env.example`。常用配置如下：

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `SCI_RAG_RETRIEVAL_MODE` | `dense` | `dense` 或 BM25+dense 的 `hybrid` |
| `SCI_RAG_RETRIEVAL_K` | `12` | dense 初始候选数 |
| `SCI_RAG_CONTEXT_K` | `10` | 实际送入生成模型的上下文槽位 |
| `SCI_RAG_DOCUMENT_ROUTING` | `false` | 唯一高信号标识符命中时限制来源 |
| `SCI_RAG_VISION_ENABLED` | `false` | PDF Figure 问题的 opt-in 视觉路径；需同时开启文档路由 |
| `SCI_RAG_VISION_MODEL` | `deepseek-v4-flash-vision-exp` | 视觉路径使用的模型名 |
| `SCI_RAG_QUERY_DECOMPOSITION` | `false` | 对复合问题生成有界子查询 |
| `SCI_RAG_PARENT_WINDOW` | `false` | 为前两个正文锚点拼接同页邻块 |
| `SCI_RAG_SPATIAL_FIGURE_EVIDENCE` | `false` | 读取 PDF 文字层中的 Figure 坐标证据 |
| `SCI_RAG_FORMULA_EVIDENCE_AUTO` | `true` | 显式公式/算法问题自动启用窄证据通道 |
| `SCI_RAG_FORMULA_EVIDENCE` | `false` | 全局公式证据实验开关 |
| `SCI_RAG_ANSWER_VALIDATION` | `false` | 返回只读证据核对提示，不改写或重试答案 |

Hybrid 运行示例：

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid python3 app.py
```

本地 cross-encoder 必须已缓存且固定 revision；运行时使用
`local_files_only=True`，缺失时直接失败，不会隐式下载：

```bash
HF_HUB_OFFLINE=1 \
SCI_RAG_RETRIEVAL_MODE=hybrid \
SCI_RAG_RERANKER_MODEL=BAAI/bge-reranker-base \
SCI_RAG_RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70 \
python3 app.py
```

Hybrid 首次查询会从当前 collection 构建内存 BM25 快照；上传文档后快照自动失效。
路由只在来源唯一时生效；开启查询分解后，跨论文问题的各子句分别限定到各自来源，并保留有界
来源内 lexical/同节/数字证据。歧义问题仍回退全库。所有实验检索之后仍执行表号保护和确定性单元格查找。

视觉路径默认关闭。开启 `SCI_RAG_VISION_ENABLED=true` 并同时开启
`SCI_RAG_DOCUMENT_ROUTING=true` 后，上传的 PDF 会按 SHA-256 保存到
`<SCI_RAG_DB_PATH>/source_pdfs/`；仅明确包含 Figure/Extended Data Figure 且能唯一定位来源的
问题会发送完整图和局部图。普通问题、表格问题、来源不明确的问题继续使用文本 RAG。
该路径目前是 opt-in 实验，尚未达到默认推广标准。

## 五论文离线基准

`evaluation/benchmark/` 包含 5 篇论文、53 道题的 manifest、人工 gold contexts、
required facts、别名和版本化复核标签。PDF 不进入 Git，只记录文件名与 SHA-256。

校验标注：

```bash
python3 evaluation/validate_benchmark.py
```

连同仓库外 PDF 一起核验：

```bash
python3 evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

不加载 Chroma、模型或 API 的 BM25 基线：

```bash
python3 evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --top-k 1,3,5,10 --show-failures
```

使用已缓存 embedding/reranker 的固定实验路径：

```bash
HF_HUB_OFFLINE=1 python3 evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --top-k 1,3,5,10 \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --reranker-fusion rrf --document-routing --query-decomposition \
  --structured-table-guard --parent-window --formula-evidence \
  --limitation-evidence --show-failures
```

如需将已核对的 THINKNOTE（Findings of EACL 2026）加入对照，可显式使用扩展清单；它通过
`base_manifest` 继承五论文基线，不改变默认 53 题：

```bash
python3 evaluation/validate_benchmark.py \
  --manifest evaluation/benchmark/manifest_expanded.json \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

扩展清单当前为 6 篇论文、66 题；13 道 THINKNOTE 用例已逐题对照本地 PDF，结果只作为
额外基准，不覆盖五论文历史报告。

泛化留出清单 `evaluation/benchmark/manifest_generalization.json` 在此基础上加入 TACL 2025
TANQ 与 Findings of EMNLP 2025 FigEx，共 8 篇论文、82 题；默认关闭，不改变 53/66 题基线。
PDF 仍保存在仓库外，校验命令为：

```bash
python3 evaluation/validate_benchmark.py \
  --manifest evaluation/benchmark/manifest_generalization.json \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

这些结果测量上下文中的词面事实覆盖和 provenance，不等于答案正确率。详细标注边界见
`evaluation/benchmark/README.md` 与 `evaluation/benchmark/PAPER_AUDIT.md`。

## 答案与生成审计

答案文件使用仓库外 JSONL，每行至少包含 `case_id` 和 `answer`：

```json
{"case_id":"drugr-09","answer":"...","mode":"hybrid"}
```

词面完整性审计：

```bash
python3 evaluation/answer_audit.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/sci_rag_answers.jsonl --require-all \
  --json-out /tmp/sci_rag_answer_audit.json
```

比较两个固定题集的回答：

```bash
python3 evaluation/compare_answer_runs.py \
  --testset evaluation/benchmark/cases.jsonl \
  --baseline /tmp/dense.jsonl --candidate /tmp/hybrid.jsonl \
  --baseline-name dense --candidate-name hybrid \
  --require-all --json-out /tmp/sci_rag_answer_compare.json
```

生成并校验人工复核模板：

```bash
python3 evaluation/review_answers.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/hybrid.jsonl --require-all \
  --template-out /tmp/sci_rag_review.jsonl

python3 evaluation/review_answers.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/hybrid.jsonl --reviews /tmp/sci_rag_review.jsonl \
  --require-all --json-out /tmp/sci_rag_review_summary.json
```

重复生成器 `evaluation/generation_stability.py` 只应连接隔离 ChromaDB。它按
`(repeat, case_id)` 安全续跑，并记录无密钥的配置、源码指纹、上下文 ID 和 metadata。
对应 trace 可用 `evaluation/audit_generation_trace.py` 与
`evaluation/validate_answer_evidence.py` 离线检查。

答案词面覆盖、拒答风险和证据提示都是诊断信号，不能代替逐题语义复核。

金标准答案审计（不调用模型）：

```bash
python3 evaluation/ground_truth_audit.py \
  --testset evaluation/benchmark/cases.jsonl \
  --answers /tmp/sci_rag_generation_trace.jsonl \
  --require-all --json-out /tmp/sci_rag_ground_truth_audit.json
```

该报告分别输出 required-fact 词面覆盖、人工整理上下文召回和规范化文本一致性；只有附带
`--reviews` 的人工判断才计入语义正确性，不能把任一自动指标直接称为答案正确率。

## RAGAS 边界

只有明确授权外部调用时才运行：

```bash
SCI_RAG_DB_PATH=/absolute/path/to/isolated-chroma \
python3 evaluation/evaluate.py \
  --testset evaluation/benchmark/cases.jsonl \
  --report-json /tmp/sci_rag_ragas.json \
  --report-md /tmp/sci_rag_ragas.md
```

审计已有报告：

```bash
python3 evaluation/ragas_preflight.py \
  --report-json evaluation/evaluation_report.json \
  --testset evaluation/test_questions.json --require-complete
```

现有三个 RAGAS 指标不使用 `ground_truth/reference`，因此不能证明答案正确性。高
Answer Relevance 只说明回答与问题相关；Faithfulness 只检查回答是否由送入评判的上下文
支持；Context Relevance 只评价上下文与问题的相关程度。单论文 11 题或五论文 53 题也不能
证明跨领域泛化。

## 当前证据与边界

- 当前公式隔离源码在全新五论文数据库中产生 577 块：455 个正文、24 个表格、23 个 Figure
  坐标文字和 75 个独立公式块；普通检索语料仍为 502 块。
- 公式隔离版本完成两轮 53 题生成，106/106 次 API 调用成功，provenance、源码指纹和运行配置完整且
  一致。两轮 top-1/3/5 上下文均为 `53/53` 相同；完整 top-10 为 `51/53` 相同，两处变化只发生在
  低位候选，目标证据和答案未受影响。随后仅修改了引用补充门控，未改变检索路径。
- 两轮词面事实审计分别为 `50/53 full`（macro/micro=`0.9811/0.9795`）和 `52/53 full`
  （`0.9937/0.9932`）。逐题语义复核两轮均为 `52 correct / 1 partial`；`mgno-04` 的目标事实正确，
  但附加的循环方向描述存在混淆。
- 两轮规范化答案文本只有 `18/53` 完全一致，说明生成措辞仍有随机性；引用补充门控修复后，针对
  `scidqa-05/06` 和 `table-llm-07` 的 3 题定向复测均不再附加无关证据。
- evidence-only 检查第一轮为 30 `ok`、23 `not_applicable`，第二轮为 29 `ok`、
  23 `not_applicable`、1 `review`；这些诊断与人工复核都不能外推为跨领域泛化或生产可靠性。
- 当前没有图片持久化/OCR、通用工具注册与执行器、图抽取或图数据库。
- `evaluation/benchmark/manifest_challenge.json` 提供默认关闭的 35 道定向挑战题：10 道
  image-only、20 道 computation、5 道 cross-document；它们只用于采集缺口，不改变默认基准。
- `evaluation/benchmark/manifest_generalization.json` 提供默认关闭的 16 道留出题，覆盖新论文的
  表格、图像空间关系和跨文档证据；最终两轮生成 32/32 行成功，16 个 case 的 context 与 provenance
  均稳定。针对暴露的四类缺口完成通用修复并聚焦复测；人工语义复核记录现为 `16 correct`，详见
  `evaluation/benchmark/reviews_generalization_16.jsonl`。这不是生产正确率或 RAGAS 结论。表格题的
  模型/数据集与 setting 消歧、同节续块排序、跨来源共享谓词补证据和空间坐标方向均有回归测试。
- 多模态至少需要 10 道人工核对的 image-only 失败题；Graph-RAG 至少需要 5 道稳定的
  跨文档多跳失败题；通用工具调用至少需要 20 道真实运算题和 5 道可被本地白名单工具
  稳定修复的失败题。未满足门槛前不增加子系统。

完整而简明的修改历史统一维护在 `MODIFICATION_LOG.md`，不再新增按 Phase 拆分的交接文档。

## 主要文件

- `app.py`：无副作用入口、运行时、检索编排和 Gradio UI。
- `sci_rag_core.py`：解析、切分、表格/公式/限制证据与答案核对。
- `sci_rag_retrieval.py`：BM25、文档路由、query variants 和 RRF。
- `sci_rag_reranking.py`：本地 cross-encoder 封装。
- `evaluation/`：基准加载、检索、生成、RAGAS 和审计工具。
- `tests/`：离线回归测试。
- `MODIFICATION_LOG.md`：唯一的阶段修改记录。
