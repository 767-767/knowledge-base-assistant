# Phase 12 全量生成门禁交接

日期：2026-08-31  
分支：`develop`  
状态：完成两轮固定配置全量门禁；未提交或 push

## 门禁目的

验证 Phase 11 的唯一来源同节扩展接入后，五论文 53 题是否仍能完整运行、是否出现
明显答案事实退化，以及此前的 `scidqa-07` 方法段拒答是否被修复。该门禁使用与
Phase 7.1 相同的隔离数据库和主要检索参数；不把单轮生成结果当作稳定性或语义准确率
证明，也不因此切换网页默认开关。

## 固定条件

- 数据库：`/private/tmp/sci-rag-phase62.st4cV2/chroma_db`，502 个块；生产
  `chroma_db/` 未读取写入操作之外的状态，也未重建。
- 检索：Hybrid + `BAAI/bge-reranker-base`，revision
  `2cfc18c9415c912f9d8155881c133215df768a70`，CPU，reranker batch 8，最大长度 512，
  RRF-k 60。
- 运行开关：document routing、query decomposition、parent/window、spatial figure
  evidence 均为 true；formula evidence 和 answer validation 为 false；`context_k=10`。
- 生成：DeepSeek `deepseek-chat`，通过现有 `generation_stability.py` 路径调用。

## 运行命令和产物

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/generation_stability.py \
  --db-path /private/tmp/sci-rag-phase62.st4cV2/chroma_db \
  --output /private/tmp/sci_rag_phase12_current_route_section_53.jsonl \
  --repeats 1 --expected-chunks 502 --retrieval-mode hybrid \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --reranker-batch-size 8 --reranker-max-length 512 \
  --reranker-device cpu --reranker-rrf-k 60 --no-resume
```

随后复用同一输出文件补跑第二轮（不带 `--no-resume`）：

```bash
HF_HUB_OFFLINE=1 ./venv/bin/python evaluation/generation_stability.py \
  --db-path /private/tmp/sci-rag-phase62.st4cV2/chroma_db \
  --output /private/tmp/sci_rag_phase12_current_route_section_53.jsonl \
  --repeats 2 --expected-chunks 502 --retrieval-mode hybrid \
  --reranker-model BAAI/bge-reranker-base \
  --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --reranker-batch-size 8 --reranker-max-length 512 \
  --reranker-device cpu --reranker-rrf-k 60
```

完整回答和两轮 context trace 只保存在仓库外的
`/private/tmp/sci_rag_phase12_current_route_section_53.jsonl`（106 行）。离线派生报告为：

- `/private/tmp/sci_rag_phase12_answer_audit.json`（repeat 1；repeat 2 的同口径摘要记录在本交接文档）
- `/private/tmp/sci_rag_phase12_answer_compare.json`
- `/private/tmp/sci_rag_phase12_trace_audit.json`

## 全量结果

两轮共完成 `106/106`，API 错误 `0`，每行均有 `runtime_config`、source fingerprint、
context IDs 和 metadata，因此 provenance 完整率为 `53/53`。两轮的每轮词面结果为：

| repeat | fact macro | fact micro | 完整覆盖 | partial | zero |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9450 | 0.9315 | 47/53 | 6/53 | 0/53 |
| 2 | 0.9434 | 0.9247 | 47/53 | 6/53 | 0/53 |

与 Phase 7.1 固定配置的已有一轮结果做词面 answer-fact 对照（以下以 repeat 1 为主）：

| 指标 | Phase 7.1 | Phase 12 | 变化 |
| --- | ---: | ---: | ---: |
| fact macro | 0.9261 | 0.9450 | +0.0189 |
| fact micro | 0.8973 | 0.9315 | +0.0342 |
| 完整覆盖 | 46/53 (0.8679) | 47/53 (0.8868) | +1 |
| partial | 6/53 | 6/53 | 0 |
| zero | 1/53 | 0/53 | -1 |

逐题状态转移为：`zero→full` 1 题（`scidqa-07`）、`partial→full` 1 题
（`mgno-02`）、`full→partial` 1 题（`mgno-03`），其余 50 题状态不变。另有
`drugr-11` 在 partial 状态下词面覆盖由 0.50 增至 0.75，但仍遗漏 `shortfall`。

## 逐题人工抽查

本轮重点变化的四题已对照 benchmark gold/context 做人工检查：

- `scidqa-07`：正确。答案列出 closed-book、title-abs、RAG、full-text；三个方法段
  context 均来自 `2024.emnlp-main.1163.pdf` 的 `B Experiments > B.1 Experimental Setup`，
  没有其他论文块。
- `mgno-02`：正确。答案给出 `A∗u=f` 和 `3×3` 核，符合 gold；这次改善来自生成
  变体/证据呈现，不能单独归因于 Phase 11 的方法段扩展。
- `mgno-03`：部分正确。当前回答仍说明初始化未明确，且没有明确写出残差表达式；
  这是单轮 DeepSeek 生成波动/措辞问题，不是 `scidqa-07` 路由扩展的目标事实。
- `drugr-11`：部分正确。当前回答增加了 Pareto 重加权细节，但没有完整说明 batch-level
  shortfall boost；不能标为完全正确。

第二轮相对第一轮的 context/metadata 稳定率为 `50/53=94.34%`，答案逐字稳定率为
`19/53=35.85%`。只有 `af3-04` 发生 full→partial 状态变化，`mgno-03` 发生
partial→full；`scidqa-07` 和 `mgno-02` 两轮均为 full。由此可见检索大体稳定，
但生成表述和事实覆盖仍会波动。

这是一轮内部单人抽查，不是独立双人盲审；其作用是发现明显回退，不替代正式语义标注。

## 结论与默认策略

1. Phase 11 的目标题得到修复，且没有发现跨论文 context 污染。
2. 全量运行能力和词面覆盖没有明显退化；但 `full→partial` 的单题变化说明生成层
   仍有随机性，不能把聚合提升解释为检索改进的因果证明。
3. `SCI_RAG_DOCUMENT_ROUTING`、`SCI_RAG_FORMULA_EVIDENCE`、parent/window、空间图形
   证据以及 Hybrid/reranker 仍不改网页默认值；`context_k` 仍为 10。
4. 本轮没有运行 RAGAS、没有上传文档、没有启动 Gradio，也没有改变生产 ChromaDB、
   `.env` 或依赖。

若要把这条路径提升为默认能力，仍需对所有非完整题进行独立语义复核，并检查表号/单位/
公式/引用和延迟回归；两轮已经显示 context 稳定但答案逐字和个别事实状态会波动。当前
更稳妥的结论是：保留实现和测试，继续以显式 opt-in/唯一 source route 使用，不切换默认
开关。
