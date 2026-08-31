# Phase 6.3 生成稳定性复测交接

日期：2026-08-30  
分支：`develop`  
状态：已完成两轮全量复测；默认网页配置未切换；尚未提交或 push

## 目标与协议

Phase 6.2 已证明检索上下文可以稳定定位 Figure 1 的空间证据，但这不等于生成答案稳定。
本阶段固定同一个 502 块隔离 Chroma、Hybrid + cross-encoder、document routing、query
decomposition、parent/window 和空间 Figure 证据开关，对完整 53 题各生成两次。每次记录
答案、上下文 ID、上下文 metadata、延迟、重试次数和错误状态；输出全部位于仓库外。

可复用命令：

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/generation_stability.py \
  --db-path /private/tmp/sci-rag-phase62.st4cV2/chroma_db \
  --output /private/tmp/sci_rag_phase63_stability_53x2.jsonl \
  --repeats 2 --expected-chunks 502
```

脚本支持断点续跑；API 错误行会在下一次运行中重试并覆盖同一个 `(repeat, case_id)`，避免
重复 case ID。它不运行 RAGAS，也不修改项目或生产数据库。

## 结果

- 106/106 次生成完成，API 错误 0；平均延迟 1.633 秒，中位数 1.579 秒，最大 5.015 秒。
- 两轮 53/53 的 context ID 和 metadata 完全一致，证明该配置下检索上下文可复现。
- 答案逐字（去除空白差异）完全一致 16/53；37 题存在措辞变化，不能据此判为事实错误。
- 事实自动审计：
  - 第 1 轮 macro/micro/full = `0.8003/0.7740/0.7170`；
  - 第 2 轮 macro/micro/full = `0.8019/0.7740/0.7170`。
- 只有两道题发生事实状态变化：`mgno-02` full→partial、`af3-02` partial→full；没有净改善，
  也没有出现 API 错误导致的伪结果。
- `af3-06` 两轮都正确回答蛋白质–RNA `25`、蛋白质–dsDNA `38`、CASP15 RNA `8`，且返回
  上下文不含扁平 picture-text。

逐轮答案审计和稳定性比较：

- `/private/tmp/sci_rag_phase63_stability_repeat1.jsonl`
- `/private/tmp/sci_rag_phase63_stability_repeat2.jsonl`
- `/private/tmp/sci_rag_phase63_stability_audit.json`
- 原始两轮：`/private/tmp/sci_rag_phase63_stability_53x2.jsonl`

## 结论边界

同一上下文稳定而答案表述变化，说明当前需要区分 retrieval determinism 与 generation
reliability。事实状态只有 2/53 发生迁移，表明大多数变化是措辞层；但完整事实覆盖只有约
71.7%，仍有固定困难题，不能宣称 53 题答案正确率为 71.7%。自动审计不判断语义等价、引用
正确性或图像像素理解；本阶段也没有运行 RAGAS。

因此不把 temperature 降低或增加重试直接改成默认行为。下一步应优先针对未完整覆盖题做
人工语义复核和可解释的答案校验/重试策略，再决定是否需要第二次生成调用；真正 OCR/多模态
仍是独立路线。

## 证据答案校验（已完成设计验证）

新增 `sci_rag_core.validate_answer_against_evidence()` 与离线命令
`evaluation/validate_answer_evidence.py`。它只读取生成 trace 中的 question、answer、
contexts 和 context metadata，重建与 prompt 相同的事实核对清单，并以保守规则检查：

- 数字问题是否漏掉与问题意图邻近的数字；
- 流程/复合问题是否只覆盖了高信号命名实体的一部分；
- 空答案/API 错误是否需要重试或人工查看。

它不读取 `ground_truth`、`required_facts` 或论文外部标注，`review` 不是“答案错误”结论。
显式表格单元格、带坐标图形和公式/算子问题分别返回 `not_applicable`，避免把邻近单元格、
图轴/子图数字或 superscript 解析伪影误判为遗漏。运行时通过
`SCI_RAG_ANSWER_VALIDATION=true` 可选开启提示；默认关闭，且不会自动重写或再次调用 API。

对本阶段 106 条稳定性记录的离线审计输出为：36 条结构化表格路径、4 条空间图形路径、
6 条公式/算子语义路径、58 条普通 `ok`，以及 2 条 `review`。两条均为同一个
`drugr-09`（显式推理数据集的流程答案漏掉 `Deepseek-R1` 的一致信号），不是新的事实正确率。
最终报告位于仓库外：`/private/tmp/sci_rag_phase63_evidence_validation_v14.json`。下一步若要
加入自动二次生成，必须先对这些 `review` 行做人工语义复核，并以固定的“原答案不退化、
额外 API 次数受控”的 A/B 验收；本阶段没有自动重试。

## 运行配置追踪（Phase 6.6 前置修正）

旧 trace 只保存了功能开关，没有保存 embedding、candidate-k、context-k、reranker
revision 或源码指纹。因此“开关相同”不足以证明两次运行是同一实验条件；旧 trace 的
结论应保留为历史记录，不能与后续源码变更后的单题结果直接拼接。

`evaluation/generation_stability.py` 现在为新行记录无密钥的 `runtime_config` 和
`source_fingerprint`。可用以下离线命令检查一份 trace 的条件完整性、context ID 稳定性
和答案措辞变化；命令不加载模型、不调用 API：

```bash
./venv/bin/python evaluation/audit_generation_trace.py \
  --trace /path/to/generation_stability.jsonl \
  --json-out /private/tmp/generation_trace_audit.json
```

历史的 Phase 6.3 文件缺少这些字段，运行审计时应预期出现 provenance incomplete，而不是
把它当成当前代码的回归。
