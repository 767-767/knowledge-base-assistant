# Phase 7.1 当前版本整体验证交接

日期：2026-08-31  
分支：`develop`  
状态：完成当前版本生成与离线审计；未提交或 push

## 目标与边界

本阶段验证的是当前源码在五篇论文、53 道基准题上的真实生成链路。使用了隔离的 502 块
ChromaDB，只读读取其内容；没有重建数据库、启动 Gradio 或修改生产数据库。生成调用使用
用户已授权的 DeepSeek API，所有回答和审计 JSON 均写在 `/private/tmp`，不进入仓库。

## 正式运行配置

```text
retrieval_mode=hybrid
embedding_model=BAAI/bge-small-zh-v1.5
reranker_model=BAAI/bge-reranker-base
reranker_revision=2cfc18c9415c912f9d8155881c133215df768a70
reranker_device=cpu
retrieval_k=12, hybrid_candidate_k=50, hybrid_rrf_k=60, context_k=10
document_routing/query_decomposition/parent_window/spatial_figure_evidence=true
formula_evidence=false, answer_validation=false
db_chunks=502
```

正式命令如下（输出路径可按需替换）：

```bash
./venv/bin/python evaluation/generation_stability.py \
  --db-path /private/tmp/sci-rag-phase62.st4cV2/chroma_db \
  --output /private/tmp/sci_rag_current_phase71_hybrid_reranker_answers.jsonl \
  --repeats 1 --expected-chunks 502 --retrieval-mode hybrid \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --reranker-batch-size 8 --reranker-max-length 512 \
  --reranker-device cpu --reranker-rrf-k 60
```

运行 53/53 题成功，0 个调用错误。平均延迟约 4.62 秒/题，中位数约 4.72 秒，最大约
7.99 秒。每行都记录了无密钥运行配置和源码指纹；53/53 行 provenance 完整，重复 case 为
0。由于只运行一轮，本阶段不能估计生成文字的重复稳定率。

## 离线审计结果

正式回答的词面事实覆盖为：

| 指标 | 结果 |
| --- | ---: |
| required-fact macro | 0.9261 |
| required-fact micro | 0.8973 |
| 完整覆盖 | 46/53（86.79%） |
| 部分覆盖 | 6/53（11.32%） |
| 零覆盖 | 1/53（1.89%） |

上述数字是**词面事实审计**，不是人工语义正确率，也不是 RAGAS 分数。当前 7 个非完整
题（加入安全的格式/译法别名后）为：

- `drugr-09`：遗漏 `0.6`、`ADMETLab`；回答给出了 4,855 和 DeepSeek-R1，但没有明确写出
  这两个工具/阈值事实。
- `drugr-11`：10-context 回答遗漏 `Pareto`、`shortfall` 的明确术语；这是候选池问题，
  不是 50-context 版本的生成能力问题。
- `scidqa-07`：10-context 未给出四个配置；属于上下文尾部证据未进入默认前缀。
- `mgno-01`：回答写出 `Ω=(0,1)²`，原始审计曾因 `(0,1)` 表面形式而误报；本轮保留该项作为
  格式审计边界，不把它当成检索缺失。
- `mgno-02`：10-context 没有给出 `3×3`；50-context 回答给出 `3 × 3`。基准新增了等价
  书写别名，避免后续把乘号空格误报为错误。
- `mgno-04`：10-context 没有明确写出 stride=2、反卷积和两种循环；50-context 能恢复主要
  内容，但中文“步长为 2”已加入等价别名。
- `af3-11`：回答给出了闭合构象示例；基准新增了“闭合/闭合构象”别名，仍需明确静态结构
  与溶液动力学行为的限制。

证据一致性校验结果为 `not_applicable=23`、`ok=29`、`review=1`，唯一 review 为
`scidqa-08` 的高信号行，需要人工确认 BM25、切块和 top-ranked chunks 是否均被完整表达。
该校验不读取 gold，不能判断事实是否正确。

正式报告文件：

- `/private/tmp/sci_rag_current_phase71_hybrid_reranker_answers.jsonl`
- `/private/tmp/sci_rag_current_phase71_hybrid_reranker_audit_aliases.json`
- `/private/tmp/sci_rag_current_phase71_hybrid_reranker_trace_audit.json`
- `/private/tmp/sci_rag_current_phase71_hybrid_reranker_evidence_validation.json`

## context-k=50 限定对照

对上述 8 个边界题在相同 reranker、同一隔离库下将 `SCI_RAG_CONTEXT_K=50`，单独生成一轮：

- 8/8 成功、0 错误；词面完整覆盖 6/8（加入别名后），部分 2/8，零 0/8。
- `drugr-11`、`scidqa-07`、`mgno-01`、`mgno-02`、`af3-02` 的缺失事实被恢复或已被
  格式别名吸收；`drugr-09` 仍未明确列出 ADMETLab/0.6，说明仅扩大上下文不能保证生成
  覆盖；`mgno-04` 与 `af3-11` 仍有表达层面的遗漏。
- 该对照只证明尾部候选包含有用证据，不证明将 50 块交给生成模型会提升整体答案正确性。
  50-context 预计增加提示长度、延迟和费用，因此没有改网页默认值。

## 与历史结果的关系

与历史 Phase 5.7 Hybrid+Reranker 输出相比，本轮词面事实 macro/micro 从
`0.7201/0.6918` 提升至 `0.9261/0.8973`，完整覆盖从 `33/53` 提升至 `46/53`；
但历史运行的源码/配置指纹不完整，比较只能作为诊断，不能作为严格受控实验结论。

之前曾误跑过一轮没有 reranker 的 53 题输出；该文件只作为“无 reranker 诊断对照”，不应
与本正式结果混用。正式结果首行已核验 reranker model 和 revision 非空。

## 结论与下一步门槛

1. 当前 Hybrid+固定 reranker+路由/子查询/parent/window/空间文字证据链路可以稳定跑通五篇
   论文的 53 道题，且 provenance 完整。
2. 仍不能宣称 86.79% 是语义正确率；至少 `drugr-09`、`mgno-04`、`af3-11` 需要人工
   语义复核，`scidqa-08` 需要证据行复核。
3. 不把 `context_k` 提高到 50 作为默认修复；下一步应在真实失败样本驱动下，先评估受限的
   确定性单位/算术工具是否有需求。若没有 ≥5 个可由工具修复的真实题，Phase 7.2 保持
   设计审计而不引入 Agent 工具。
4. 多模态只在新增并人工核对至少 10 道 image-only/复杂图题后触发；Graph-RAG 继续等待
   明确的跨文档多跳需求。
