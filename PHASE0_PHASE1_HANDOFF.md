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

## 未包含在本轮

- Hybrid/BM25/RRF/cross-encoder reranker。
- 多模态图片资产持久化、OCR/VLM。
- 通用 Agent 工具调用。
- Graph-RAG。
- 完整多论文 benchmark、RAGAS 重跑及生产部署配置。

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
