# Sci-RAG Phase 0 / Phase 1 接手记录

日期：2026-08-28

## 本次目标

完成 Phase 0（可测试、可复现、无导入副作用）和 Phase 1（表格规范化、跨表隔离、页/块元数据、评估契约）的代码改造。没有执行 Git push，也没有调用 DeepSeek 或其他付费模型 API。

## 已完成

- 新增 `sci_rag_core.py`，将表格抽取、Markdown/HTML/Unicode 标记规范化、行实体选择、表格过滤和切分逻辑放入无 API/模型/Chroma/Gradio 副作用的模块。
- 重构 `app.py`：导入不再加载嵌入模型、创建 Chroma、读取 API key 或构建 UI；资源通过 `create_runtime()` 显式初始化。
- 增加 `RuntimeConfig`，支持 DB 路径、模型、endpoint、候选检索数量和最终上下文数量的环境变量配置。
- PDF 使用 `pymupdf4llm` 的 page chunks，并写入页码 metadata；DOCX 改用已声明的 `python-docx` 读取，避免旧 `Docx2txtLoader` 的隐式依赖问题。
- 表格被从正文 text 流移除，只保留规范 table 块；增加 `table_id`、`table_number`、`document_sha256`、`chunk_index`、`chunk_id` 等字段。
- 显式 Table N 问题额外加载 table 块，并过滤旧库中仍含 GFM 表格的 legacy text 块；行实体从表格第一列识别，不再取最长英文短语。
- 检索默认候选数从 60 降为 12，最终上下文默认最多 10 个；`query_knowledge(return_contexts=True)` 返回的 contexts 与生成 Prompt 使用的上下文一致，并附带 ID/metadata。
- `evaluation/evaluate.py` 改为惰性导入 RAGAS/模型；增加 deterministic `gold_fact_coverage` 和 `gold_context_recall`，并保存评估上下文、ID 和 metadata trace。
- `evaluation/test_questions.json` 为 11 题增加了人工声明的 `required_facts`，用于透明的事实覆盖 smoke check。
- 补齐并锁定当前 venv 验证过的直接依赖，新增 `.env.example`，重写 `test_setup.py` 为离线依赖检查，补充 `tests/`。
- 更新 README，说明 API、UI、离线测试和评估边界。

## 验收证据

以下命令均已执行成功：

```text
./venv/bin/python test_setup.py
./venv/bin/python -m unittest discover -s tests -v
python3 -c 'import app; assert app._runtime is None'
```

测试结果：9 个 unittest 全部通过；系统 Python 导入 `app.py` 成功且 runtime 仍为 `None`；临时 TXT、DOCX、PDF fixture 均能解析，DOCX 表格能形成独立 table chunk。

本次没有运行：完整 RAGAS、Gradio、真实文档上传、DeepSeek API、ChromaDB 重建。现有 `evaluation/evaluation_report.*` 仍是旧代码/旧数据库运行结果，不应被解释为本次改造后的新分数。

## 重要迁移注意

当前 `chroma_db/` 中已有旧的 104 块未被修改。新代码对显式 Table N 查询提供了 legacy text 保护，但旧库仍缺少新的稳定 ID、页码和部分 table metadata。正式采用 Phase 1 数据结构前，需要在用户授权后用原始论文重新建立临时/新版本索引，并通过回归集验证后再切换；本次没有执行该操作。

## 2026-08-28 表格单元格查询修复

使用附件 `2602.08213v1.pdf` 做了只读核验：第 3 页的 Table 2 确实包含
`DrugR*` 行，其中 Overall Optimization Score 为 `0.2060`，Target property
F1 score 为 `0.3404`。旧 ChromaDB 的 104 个块中也存在对应的独立 Table 2 块。

此前的错误原因是：表格检索虽然能够把 Table 2 放入候选，但同时还允许 Table 1
的叙述性/旧式文本块进入生成上下文；最终答案由模型在相似表头和数字之间自行选择，
所以会把 Table 1 的 `0.2712` 或 Table 1 的 `0.4364` 误认为 Table 2 的答案。

本次追加的通用修复：

- 解析 Markdown 表格的表头、第一列和单元格；支持 HTML/sup、Markdown 加粗和 Unicode 星号规范化。
- 对明确包含 `Table N + 行实体 + 列名称` 的问题执行确定性单元格查找，不再让模型决定行列交叉值。
- 支持常见中文表头别名，例如“整体优化得分”和“目标属性 F1”。英文表头仍优先采用字面匹配。
- 明确指定的 Table N 不存在时直接报告缺失，不再用其他表格回退。
- 误把“Table 1 also presents ...”这类交叉引用识别成表格的 fallback 解析规则已收紧。
- `.gitignore` 增加本地环境变体、SQLite 临时文件、PyMuPDF 会话文件和 macOS 元数据忽略规则，保留 `.env.example`。

新增/更新的离线回归测试共 11 项，覆盖上述单元格定位、跨表拒答和旧库兼容场景；未调用
DeepSeek，未重建 ChromaDB，未运行 Gradio 或完整 RAGAS。

## 一键网页验收

新增 `scripts/launch_phase1_ui_test.sh`。在项目根目录执行：

```bash
bash scripts/launch_phase1_ui_test.sh
```

脚本会在 `/tmp` 创建临时空数据库，导入默认的
`../2602.08213v1.pdf`，然后启动正常的 Gradio UI。脚本读取项目根目录的
`.env`，但不会复制或修改密钥；按 `Ctrl+C` 后临时数据库自动清理，当前
`chroma_db/` 不会被写入。

如需只测试旧数据库兼容性：

```bash
bash scripts/launch_phase1_ui_test.sh --existing
```

如附件 PDF 不在默认路径，可传绝对路径作为第一个参数。完整验收清单已同步到
`README.md`，包括三项 Table 1/Table 2 数值、上传、问答、大纲和出题功能。

## Phase 2 基准集第一步

在 `phase2/multipaper-benchmark` 分支新增了 `evaluation/benchmark/`、
`evaluation/benchmark_loader.py` 和 `evaluation/validate_benchmark.py`。用户已将四篇
免费 PDF 放在仓库外的 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`；当前清单
登记现有种子加这四篇论文，共 5 篇、53 道问题。校验器只读取 JSON/JSONL 和可选的外部
PDF SHA-256，不会加载模型、Chroma、Gradio 或 API；`--papers-dir` 可重复传入多个目录。

新增论文应放在仓库外，登记文件名、SHA-256、领域和版式标签，并为每篇准备独立问题。
当前 53 道题中的 42 个新增用例已完成一次 PDF 逐题人工核对，且已补充表头、单位、
caption 和公式语义。`evaluation/benchmark/PAPER_AUDIT.md` 记录了同一解析管线下的
页数、块数和表格识别风险。标注核对不等于检索或生成验证；表号未稳定进入 metadata 的
PDF，仍需先完成解析回归测试和离线检索诊断，再比较 Hybrid/RRF 的收益。

## Phase 2 解析回归（本轮）

`sci_rag_core.py` 现在兼容表格前后 caption、HTML/Markdown 装饰、跨列分组表头、PDF
断词、独立单位列和加粗标记相邻实体，并排除 DOI/作者元数据布局表。新增
`tests/test_parser_regression.py`，覆盖 8 个离线 fixture。四篇外部 PDF 的只读冒烟结果
为：SciDQA 6 个编号表格块（1、2、3、4、5、7）、Scientific Table LLM 3 个（1–3）、
MgNO 7 个（1–7）、AlphaFold 3 0 个误识别表格块；MgNO Table 1/4 的两个单元格查询
分别得到 `0.339` 和 `1.63`。这些结果只证明解析和确定性单元格定位，不证明检索、生成或
RAGAS 质量。

## Phase 2 离线检索基线（本轮）

新增 `evaluation/benchmark_retrieval.py`，以标准库 BM25-lite 在五篇论文上建立一个全局
内存词法索引；它不加载 Chroma、Embedding、Gradio 或外部 API。运行：

```bash
./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --top-k 1,3,5,10
```

当前全局结果（53 题）为：top-1/3/5/10 的参考片段覆盖代理分别为
`0.252/0.336/0.365/0.497`，目标论文命中率为 `0.811/0.868/0.868/0.906`，页级命中率为
`0.310/0.429/0.429/0.690`，显式表号命中率为 `0.667/0.833/0.833/1.000`。这些是后续
Hybrid/RRF/reranker 的比较基线，不是答案正确率；参考片段、页码和表号均来自人工标注。

进一步使用本地缓存模型并设置 `HF_HUB_OFFLINE=1` 后，当前带弱跨语言词法保护的
Hybrid-RRF 全局 top-1/3/5/10 参考片段覆盖代理为 `0.208/0.340/0.462/0.506`，目标论文
命中率为 `0.830/0.868/0.943/0.981`，页级命中率为 `0.286/0.476/0.619/0.786`，显式表号命中率为
`0.500/0.667/0.778/0.833`。Hybrid 提升了部分证据/页级代理，但 Table N top-10 仍低于
BM25，因此尚不切换线上默认检索；下一步应做显式表格保护与 reranker 的组合实验。

验证当前外部文件：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

## 后续 Phase 2 状态更新

BM25/RRF 已在 `phase2/hybrid-rerank` 分支以默认关闭的实验模式接入应用；完整实现边界、
回归证据和网页验收步骤见 `PHASE2_HYBRID_HANDOFF.md`。以下“未包含”列表按当前最新
状态更新，不再把 BM25/RRF 本身列为未实现。

## 仍未包含

- learned cross-encoder 已以默认关闭方式完成门槛实验和应用接线；当前仍缺真实网页 A/B、
  并发资源和答案质量验证，详见 `PHASE2_RERANKER_HANDOFF.md`。
- 多模态图片资产持久化、OCR/VLM。
- 通用 Agent 工具调用。
- Graph-RAG。
- 多论文答案生成/人工复核、RAGAS 重跑及生产部署配置。

## 建议提交信息

建议先在本地创建分支，例如：

```text
codex/phase0-phase1
```

建议 commit 标题：

```text
feat: establish side-effect-free Sci-RAG baseline
```

建议 commit body：

```text
- separate runtime initialization from parsing and evaluation
- canonicalize table chunks and protect explicit cross-table queries
- add page/provenance metadata and deterministic local evaluation checks
- pin direct dependencies and add offline regression tests

Tests: ./venv/bin/python -m unittest discover -s tests -v
Tests: ./venv/bin/python test_setup.py
Not run: DeepSeek/RAGAS, Gradio, Chroma rebuild, git push
Migration: existing chroma_db is intentionally untouched; reindex required
```

不建议在本轮直接提交旧报告的新分数，因为还没有在新索引和新上下文契约上重新执行 RAGAS。
