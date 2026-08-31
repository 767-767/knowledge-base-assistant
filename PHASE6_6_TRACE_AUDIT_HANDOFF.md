# Phase 6.6 生成 trace 与检索条件审计交接

日期：2026-08-30  
分支：`develop`  
状态：完成离线审计基础设施；未提交或 push

## 本阶段完成

1. 新 trace 行记录 secret-free `runtime_config`，覆盖 embedding、检索模式、candidate-k、
   context-k、query decomposition、parent/window、空间图形、答案校验、reranker 模型与
   revision 等条件。
2. 新 trace 行记录 `source_fingerprint`，由检索、生成和稳定性 runner 关键源码组成；它是
   版本标记，不是安全签名。
3. 新增 `evaluation/audit_generation_trace.py`，只读 JSONL 即可报告：
   - provenance 是否完整；
   - 同一 case 的 context ID、metadata 和配置是否稳定；
   - 答案是否仅措辞变化；
   - 重复 key 和 API 错误行。
4. `supplement_answer_with_evidence()` 跳过训练语料、corpus/token 统计等高数字密度但与
   当前问题无关的行，避免将 ChemicalQA/MoleculeNet/UltraChat 等统计附加到答案。

## 对旧结果的解释

Phase 6.3 的 106 行 trace 在新增字段前生成，因此离线审计会标记 106 行 provenance
不完整。这不是旧结果被删除或判为失败，而是明确其无法证明使用了当前源码和完整参数。
实时复测使用当前代码、502 块隔离库和固定 `BAAI/bge-reranker-base` revision，已稳定召回
DrugR 的 4,855 样本段落；该单题结果不应直接回写旧 106 行统计。

## 新 trace 验收标准

- 每行同时含 `runtime_config` 和 `source_fingerprint`，不得出现 `DEEPSEEK_API_KEY`；
- 同一 `(case_id, repeat)` 不重复；
- 若要宣称 context 稳定，必须先通过 `--require-provenance`；
- 答案措辞变化只能说明 generation variability，不能直接转化为事实正确率。

## 下一步

在拥有完整 provenance 的新 trace 之前，不做自动 retry 默认化，也不把不同 reranker
fusion 或不同源码版本的结果合并。下一步可在离线 benchmark 上比较 candidate pool、最终
context-k 与 required-fact coverage，只有确认是检索缺失而非生成遗漏后，才设计受控候选扩展。
