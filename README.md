# Sci-RAG

面向科学论文的 RAG 原型，支持 PDF/TXT/DOCX 解析、Markdown 表格结构化、ChromaDB
检索、OpenAI 兼容模型生成、Gradio UI，以及基准标注和答案审计。

当前定位是研究型可用原型，不是生产系统。默认检索仍为 dense；Hybrid、cross-encoder、
文档路由和窗口扩展均是可控开关；PDF Figure 的 VLM 路径仅作为默认关闭的 opt-in 实验。
OCR、图片向量索引、通用工具调用和 Graph-RAG 尚未实现。

## 快速开始

macOS / Linux 可直接运行：

```bash
./start.sh
```

首次运行会在项目目录创建 Git 已忽略的 `.venv`、安装依赖，并在应用启动时下载中文
嵌入模型，因此耗时取决于网络。安装中断时重新运行 `./start.sh` 即可继续补全；已有完整
`venv` 或 `.venv` 时直接复用。也可以手动启动：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 test_setup.py
python3 app.py
```

应用可以在没有 API Key 时启动、上传和管理本地文档。生成问答前可在“设置”页选择
Ollama、DeepSeek、Gemini 或自定义 OpenAI 兼容服务；云端 API Key 只保留在当前进程。
也可以复制 `.env.example` 为 `.env` 做本机持久配置。`.env`、本地数据库、PDF、模型缓存
和 `/tmp` 评估产物都不应提交。

`app.py` 的导入本身无副作用；模型、API 客户端、ChromaDB 和 Gradio 只在
`create_runtime()` 或 UI 入口中初始化。

## 模型服务

推荐不产生 API 费用且文档不离开本机的 Ollama：

```bash
ollama pull qwen3:4b-instruct
```

确保 Ollama 正在运行，然后在“设置”页选择“Ollama（本地）”并应用；API Key 留空即可。
DeepSeek、Gemini 和其他远程服务使用用户自己的 Key。环境变量统一为 `LLM_API_KEY`、
`LLM_BASE_URL` 和 `LLM_MODEL`；已有的 `DEEPSEEK_*` 配置仍可继续使用。

## macOS 桌面构建（开发者）

`desktop.py` 会自行启动打包在应用内的 `llama.cpp`，使用随机本机令牌和动态端口，
并把知识库保存在用户的 `Library/Application Support/Sci-RAG`。生成的应用不再要求最终
用户安装 Python、Ollama 或填写 API Key；页面底部的“退出 Sci-RAG”会同时关闭网页服务
和本地模型进程。

当前构建目标是 Apple Silicon 的内部未签名验收包：

```bash
venv/bin/python -m pip install pyinstaller
SCI_RAG_LLAMA_SERVER=/path/to/llama-server \
SCI_RAG_LOCAL_MODEL=/path/to/qwen3-4b-instruct-q4_k_m.gguf \
SCI_RAG_EMBEDDING_MODEL=/path/to/bge-small-zh-v1.5 \
./scripts/build_macos_app.sh
```

产物位于 `dist/Sci-RAG.app`。对外分发前仍需补齐第三方许可证文件、应用图标、Developer ID
签名与 Apple 公证；这些只属于正式发布边界，不影响当前本机功能验收。

## 文档与表格能力

- PDF 使用 `pymupdf4llm` 读取文字层，不写出或持久化图片。
- TXT 和 DOCX 走本地解析；DOCX 表格会转为 Markdown。
- 上传页显示已入库文档及文本块数；删除文档需显式确认，并同步清理其本地视觉 PDF。
- 问答页可选择一篇或多篇文档限定检索范围；不选择时检索整个知识库。
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
- 文档清单、问答范围选择和确认删除可用；
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
| `SCI_RAG_CONTEXT_K` | `4` | 实际送入问答、大纲和测验模型的上下文槽位；可按模型窗口调高 |
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
`(repeat, case_id)` 安全续跑，并记录无密钥的配置、上下文 ID 和 metadata。
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

## 当前证据与边界

- Phase H15 已冻结最终一次性 release-candidate 留出集：3 篇此前未进入任何清单的论文、18 道题，
  固定使用 Hybrid、文档路由、查询分解、parent-window、表格/公式/空间 Figure 证据且不使用 reranker；
  真实 PDF 解析门禁为 `18/18 full`。一次性 `@10` 检索为 `14/18 full`、fact macro/micro=`0.831/0.843`，
  未达到 `18/18`，因此停止在检索层，不进入真实送模上下文或生成；当前线上默认仍为 Dense。
- 当前公式隔离源码在全新五论文数据库中产生 577 块：455 个正文、24 个表格、23 个 Figure
  坐标文字和 75 个独立公式块；普通检索语料仍为 502 块。
- 公式隔离版本完成两轮 53 题生成，106/106 次 API 调用成功，provenance 和运行配置完整且
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
  `evaluation/benchmark/reviews_generalization_16.jsonl`。这不是生产正确率结论。表格题的
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
- `evaluation/`：基准校验、真实应用上下文模拟、生成和审计工具。
- `tests/`：离线回归测试。
- `MODIFICATION_LOG.md`：唯一的阶段修改记录。
