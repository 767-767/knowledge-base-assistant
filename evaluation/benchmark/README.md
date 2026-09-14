# Sci-RAG 评估数据

本目录保存论文清单、问题、人工事实标注和复核结果。原始 PDF、模型缓存、ChromaDB
和运行 trace 均保存在仓库外。

## 文件

- `manifest.json` / `cases.jsonl`：默认五论文、53 题数据集。
- `manifest_expanded.json`、`manifest_challenge.json`、`manifest_generalization.json`：
  默认数据集的扩展与定向挑战。
- `manifest_phase*.json` / `cases_phase*.jsonl`：各阶段曾使用的开发回归数据。
- `reviews_*.jsonl`：人工语义复核结果。
- `PAPER_AUDIT.md`：论文来源、标注核对和历史实验记录。

清单可以通过 `base_manifest` 继承另一份清单。文档条目记录 `document_id`、文件名、
已有的文件摘要、领域与版式标签；问题条目记录问题、目标文档、gold contexts、
required facts、页码和可选别名。阶段清单是否仍可作为未见数据，以清单内的
`evaluation_status` 为准，不根据文件名推断。

## 校验

只校验清单结构和引用，不加载模型、数据库、Gradio 或 API：

```bash
./venv/bin/python evaluation/validate_benchmark.py
```

同时检查仓库外 PDF 是否齐全并与清单中已有摘要一致：

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /absolute/path/to/papers \
  --verify-files --require-complete
```

PDF 分散在多个目录时可以重复传入 `--papers-dir`。对需要核对实际解析文字层的清单，
再增加 `--verify-extracted-evidence`。原始论文不应复制进项目或提交到 Git。

## 真实应用路径

检索行为只通过 `app.query_knowledge` 验证，不再维护一套平行的 benchmark 检索器。
准备好仓库外的隔离 ChromaDB 后，可以使用假客户端检查实际送模上下文；该命令不调用
生成 API，也不修改数据库：

```bash
./venv/bin/python evaluation/app_context_gate.py \
  --manifest evaluation/benchmark/manifest.json \
  --db-path /absolute/path/to/isolated_chroma \
  --json-out /tmp/sci_rag_app_context.json
```

需要采集真实答案时使用 `evaluation/generation_stability.py` 连接隔离数据库，并把 trace
写到仓库外。随后按需运行：

```bash
./venv/bin/python evaluation/audit_generation_trace.py --help
./venv/bin/python evaluation/answer_audit.py --help
./venv/bin/python evaluation/review_answers.py --help
./venv/bin/python evaluation/validate_answer_evidence.py --help
```

自动词面覆盖只用于定位缺失证据，不能当作答案正确率。答案质量结论以逐题人工语义复核
为准；开发中已经查看或据此修复过的问题不能再次作为未见泛化证据。
