# Phase 6.8 公式证据通道交接

日期：2026-08-30  
分支：`develop`  
状态：离线实现与 A/B 完成；未提交或 push

## 目标

处理多论文候选池审计中 MGNO `mgno-02` 的 `3×3` 表示缺口，同时避免把 PDF 解析产生的
公式符号噪声提升到普通边界条件、架构或表格问题前面。

## 实现范围

- `sci_rag_core.py`：`formula_evidence_indices()` 只响应明确公式意图；要求至少两个语义
  词命中，公式符号命中设上限；跳过 figure/image 和 references/bibliography 块；支持
  `allowed_indices` 来源白名单。
- `app.py`：增加 `SCI_RAG_FORMULA_EVIDENCE`（默认 `false`）。已有唯一 source route 时
  使用该来源；单来源集合可使用全库；多来源无路由时传入空白白名单，不注入公式候选。
- `evaluation/benchmark_retrieval.py`：增加 `--formula-evidence`，与网页路径使用同一语义
  选择和来源范围规则，报告记录开关状态。
- `tests/test_core.py`、`tests/test_benchmark_retrieval.py`：覆盖语义门槛、来源隔离与歧义
  多论文不注入。

## 离线证据

固定五论文、502 块、Hybrid、固定 cross-encoder revision、document routing、query
decomposition、structured table/figure guard、parent/window：

| 范围 | 配置 | fact macro | fact micro | 完整覆盖 |
| --- | --- | ---: | ---: | ---: |
| 全局 @10 | 无公式 / 有公式 | 0.936 / 0.936 | 0.932 / 0.932 | 47/53 / 47/53 |
| 全局 @50 | 无公式 / 有公式 | 0.995 / 0.995 | 0.993 / 0.993 | 52/53 / 52/53 |
| MGNO 单文档 @10 | 无公式 / 有公式 | 0.841 / 0.864 | 0.862 / 0.897 | 8/11 / 9/11 |

MGNO 单文档的提升来自 `mgno-02`：`3×3` 进入候选；全局未限定 `mgno-02` 仍为 partial，
因为没有可靠的唯一来源路由，通道刻意不跨论文猜测。完整报告：
`/private/tmp/sci_rag_phase68_formula_guard_scoped_v2.json`；基线：
`/private/tmp/sci_rag_phase66_candidate_pool_10_50_fixed.json`。

## 验收与限制

1. 142 项离线单元测试全部通过，模块编译和 `git diff --check` 通过。
2. 全局五论文指标不能因公式通道变化；本次 A/B 满足。
3. 该通道只做候选提升，不做 OCR、符号归一化、数学校验或答案正确性判断。
4. `3×3` 仍未被全局未限定问题覆盖；下一阶段需决定是否增加来源选择交互、公式表示归一化
   或更受控的单文档上下文策略。不得仅把默认 context-k 改为 50。

## 默认策略

生产网页保持 `SCI_RAG_FORMULA_EVIDENCE=false`。若要手动试验单论文或已明确论文名称的问题，
可在隔离环境设置为 `true`，并记录运行配置与来源；本阶段没有运行 DeepSeek 生成或修改
生产 ChromaDB。
