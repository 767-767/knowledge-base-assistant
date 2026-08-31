# Phase 6.5 受控二次生成门控交接

日期：2026-08-30  
分支：`develop`  
状态：完成一题受控 A/B；未将自动重试接入默认路径；尚未提交或 push

## 本阶段做了什么

`sci_rag_core.build_evidence_retry_prompt()` 生成一个不含 benchmark gold 的二次提示：
保留原答案中已有的有证据内容，只把 `validate_answer_against_evidence()` 标记的高信号
证据行放入有限窗口，要求模型补齐可由该证据逐字支持的数字、工具/模型名和流程步骤。
该函数不执行网络请求，是否二次调用由上层成本/风险策略决定。

用当时声明为 Phase 6.3 条件的隔离库和检索条件，对 `drugr-09` 做了一次原答案 + 一次 retry：

- Hybrid + document routing + query decomposition + parent/window + spatial figure 开启；
- cross-encoder `BAAI/bge-reranker-base` 固定 revision
  `2cfc18c9415c912f9d8155881c133215df768a70`；
- 两次 DeepSeek 调用均成功，原答案约 6.72 秒，retry 约 1.77 秒；
- 该次检索上下文没有包含明确的 `4,855` 数据集规模段落，因此原答案和 retry 都正确地
  拒答“资料未提供”，答案事实审计均为 `partial`（仅命中 `DeepSeek-R1`）；
- retry 没有改善结果，证明“答案重试”无法恢复检索阶段没有返回的证据块。

需要补充一个 provenance 限定：该历史输出生成于运行配置/源码指纹加入之前，JSON 中没有
记录完整 candidate-k、context-k、reranker 参数或代码版本。因此它是“当时固定上下文下
retry 无法补回缺失证据”的有效局部观察，但不能单独证明当前源码仍会漏掉 4,855。Phase 6.6
使用当前代码复测同一问题时，@10 已召回包含 4,855 的 H4 段落，且与离线 benchmark 的
DrugR @10 排名一致；当前答案无需触发 retry。

完整输入/输出和离线审计位于仓库外：

- `/private/tmp/sci_rag_phase65_drugr09_retry_reranker.json`
- `/private/tmp/sci_rag_phase65_drugr09_retry_audit.json`

## 门控结论

1. `review` 只适合在证据已经出现、答案疑似漏掉证据时触发；不应把它当成“答案错误”
   或触发无条件重试。
2. 如果 validator 返回 `ok` 但候选上下文缺少回答问题所需的证据，应该回到检索/查询
   扩展或安全拒答，不应重复生成。
3. 当前没有足够样本证明自动 retry 的净收益。若后续继续，至少需要固定 5–10 个已知
   “证据在 context、答案漏答”的困难题，记录一次调用与二次调用的完整答案审计、延迟、
   token/费用和退化情况；只有在二次调用改善率明显、零严重退化且成本上限可接受时，才
   考虑把 retry 作为显式 opt-in。

本阶段没有运行 RAGAS、没有启动 Gradio、没有改生产 ChromaDB，也没有执行 git 操作。
