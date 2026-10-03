# 个人知识库助手评估数据

本目录保存论文清单、问题、人工事实标注和复核结果。原始 PDF、模型缓存、ChromaDB
和运行 trace 均保存在仓库外。

## 文件

- `manifest.json` / `cases.jsonl`：默认五论文、53 题数据集。
- `manifest_expanded.json`、`manifest_challenge.json`、`manifest_generalization.json`：
  默认数据集的扩展与定向挑战。
- `manifest_phase*.json` / `cases_phase*.jsonl`：各阶段曾使用的开发回归数据。
- `reviews_*.jsonl`：人工语义复核结果。
- `paper_question_check_cases.jsonl`：三篇已开发论文上的 9 道新问题；
  `paper_question_check_reviews_codex.jsonl` 是 Codex 对首轮回答的非独立逐题复核。
- `paper_question_followup_cases.jsonl`：随后预先写好的 3 道新问题；
  `paper_question_followup_reviews_codex.jsonl` 记录一次未采用的检索候选的非独立复核。
- `paper_current_development_cases.jsonl`：当前 19 道开发题快照，含 5 道新鲜重导入后仍通过的回归对照和
  14 道仍在定位的失败题；它只用于逐题诊断，不用于报告准确率。
- `paper_selection_signal_check_cases.jsonl`：检索前冻结的 6 道新问题；
  `paper_selection_signal_check_contexts.jsonl` 记录首次真实应用路径的上下文覆盖，不含模型答案。
- `paper_selection_signal_confirmation_cases.jsonl`：检索前冻结的 4 道 PIER-QA 独立确认题；
  `paper_selection_signal_confirmation_contexts.jsonl` 记录首次上下文核对结果。
- `paper_candidate_recall_confirmation_cases.jsonl`：检索前冻结的 4 道 IRPAPERS 候选召回确认题；
  `paper_candidate_recall_confirmation_contexts.jsonl` 记录首次上下文与断点分类。
- `paper_source_neighbor_confirmation_cases.jsonl`：来源局部邻块候选后、检索前冻结的 4 道 MT-RAIG 未见确认题；
  `paper_source_neighbor_confirmation_contexts.jsonl` 记录首次上下文和候选撤回依据。
- `paper_embedding_model_confirmation_cases.jsonl`：检索与模型比较前冻结的 4 道 REAL-MM-RAG 中英科学术语题；
  后续同名 `contexts` 文件记录默认检索上下文和嵌入排名比较。
- `paper_final_selection_confirmation_cases.jsonl`：检索前冻结的 4 道 SciAssess 最终四段选择确认题；
  后续同名 `contexts` 文件记录首次上下文与断点分类。
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

当前开发题也可重复传入论文目录，直接复用论文试验 runner：

```bash
./venv/bin/python evaluation/paper_pilot.py \
  --papers-dir /absolute/path/to/sci-rag-benchmark-papers \
  --papers-dir tmp/pdfs/uaeval.Ub4QQ4 \
  --cases-file evaluation/benchmark/paper_current_development_cases.jsonl \
  --use-case-sources
```

## 真实应用路径

检索行为只通过 `app.query_knowledge` 验证，不再维护一套平行的 benchmark 检索器。
准备好仓库外的隔离 ChromaDB 后，可以使用假客户端检查实际送模上下文；该命令不调用
生成 API，也不修改数据库：

```bash
./venv/bin/python evaluation/app_context_audit.py \
  --manifest evaluation/benchmark/manifest.json \
  --db-path /absolute/path/to/isolated_chroma \
  --json-out /tmp/sci_rag_app_context.json
```

覆盖不足会列出缺失事实，不阻断后续实验；读取或运行失败仍返回错误。

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

## 下一轮研究：论文问答的证据可靠性

研究问题：错误主要来自未找到证据，还是模型没有忠实使用已经找到的证据？

先使用已有题目熟悉标注和工具，再选择自己能完整阅读的 3 篇论文，人工整理约 20 道题，
覆盖单篇事实、方法理解、跨段整合、跨论文比较和文中无答案的问题。
逐题记录原文依据、页码、预期答案或拒答理由；跨论文题注明各篇分别支持什么结论。
题目在调试前分为开发组和未参与调试的检查组；已有开发题不作为新的泛化结果。

比较当前默认检索与一个针对主要错误的改进，保持模型、提示词和上下文预算一致。
论文全文能放入模型时，再与同一模型直接阅读全文的结果比较，并报告其输入长度。
复用现有生成与人工复核工具，不增加自动通过门槛。先逐题检查：

- 参考片段是否包含回答所需证据；
- 回答是否正确、完整，有没有原文不支持的断言；
- 来源与页码是否真正支持对应结论；
- 无答案时是否明确说明资料不足；
- 生成耗时及可获取的 token 使用量。

只根据开发组的主要失败原因修改一个机制，再运行检查组。报告同时保留改善和退化的案例，
区分检索失败、生成失败和标注歧义；小样本结果不外推为普遍准确率。
上述是完整研究方向。2026-09-21 已完成三篇论文阅读、12 道开发题及默认检索/文档路由的
真实本地模型比较，结果与失败原因见 [论文阅读与本地问答诊断](../PAPER_PILOT.md)。
2026-09-24 又完成同三篇论文上的 9 道新问题检查；题目及首轮答案尚无独立人工复核，
结果只支持“已知论文上的新问题”诊断，不代表新论文泛化。这 9 题的答案已经查看，
以后不再作为未见检查组。具体逐题结果和复跑命令也记录在上述诊断文档中。
后续 3 题也已查看答案，只能作开发诊断，不能再作为未见检查组。
另有 CAQA 论文的 [T01 单题](paper_question_transfer_cases.jsonl)，运行前核对原文、运行后已查看答案；
它不再是未见题，具体限制与结果见上述诊断文档。
2026-09-25 在同一文件预先写下并运行了 T02；它同样已成为开发题，不能作为后续未见检查组。
同日另用 YESciEval PDF 做 [三道新问题](paper_question_yescieval_cases.jsonl) 的一次性检查；
[非独立语义复核](paper_question_yescieval_reviews_codex.jsonl) 与检索失败分析见诊断文档。
这些题运行后均成为开发材料，不得继续算作未见问题。
随后预写并运行 [TANQ/SPIQA 四道忠实性检查题](paper_answer_fidelity_cases.jsonl)；
首次答案已审读，检索漏证、无据附加及原始页码见[论文诊断记录](../PAPER_PILOT.md)。
四题现均为开发材料，不能据此声称独立泛化率。

2026-09-26 增加 [参数问法配对](paper_parameter_language_pairs.jsonl)，包含三篇论文的
中文／原文术语窄问，以及 ChainRAG 完整方法题。固定证据各重复两轮；完整方法题即使
词面覆盖全部必要事实，仍会错误归属参数阶段。答案审读及完整 trace 位置见
[论文诊断记录](../PAPER_PILOT.md)。这些题也属于已见开发材料，不是独立准确率检查组。

2026-09-27 的 [方法参数关系题](paper_parameter_relation_cases.jsonl)检查参数的阶段、
对象和条件，而不仅是数值是否出现。[Codex 非独立复核](paper_parameter_relation_reviews_codex.jsonl)
保留未取到证据、手工选段的输入不足及无据扩写；表格意图修正只恢复部分子项。
运行后四题也均为开发题，具体原始记录与局限仍见论文诊断记录。

同日的 [方法词触发诊断](paper_method_trigger_cases.jsonl)检查单独询问筛选、改写、技能
及提前停止时的漏证据路径。[Codex 非独立复核](paper_method_trigger_reviews_codex.jsonl)
区分事实恢复与过强保证措辞；中文筛选题正确，英文筛选原题恢复证据后仍错误拒答。
另记录两种英文变体、HeteQA行数边界冲突，以及已撤回候选的结果，不将候选成绩算入最终产品。
另保留答案表平均尺寸的回归及PDF小数装饰残留问题，不只报告有利结果。
这些均为已见论文的开发材料，不是独立检查组，完整过程见论文诊断记录。

2026-10-03 在运行检索前冻结了 [SciAssess最终四段选择确认题](paper_final_selection_confirmation_cases.jsonl)，
题单 SHA-256 为 `428d9bdfc9cb49032a28ea6d3381c0813a8301c1c914c850798d9e38321385ef`。
首次真实应用上下文四题均为部分覆盖；[逐题上下文记录](paper_final_selection_confirmation_contexts.jsonl)
将其中两题归为候选召回缺失、两题归为最终选择缺失。section整体提前产生新退化；limitation同源词面补证只改善一题，
未达到预定的跨两道独立失败采用门槛，因此均未进入产品。
