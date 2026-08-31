# Phase 6.7 候选池与多-k 口径交接

日期：2026-08-30  
分支：`develop`  
状态：完成离线候选池审计及公式候选 A/B；未提交或 push

## 评估口径修正

旧实现先按最大请求 k 应用结构化表格、图形和 parent/window 后处理，再用前缀计算较小 k。
这样同一命令的 @10 会依赖是否同时请求 @50。现在检索和 reranker 仍只执行一次，但每个实际
`top_k` 独立应用后处理并独立构造有效 context；新增测试
`test_multi_k_metrics_do_not_depend_on_larger_requested_k` 防止回归。

## 固定条件与结果

命令使用五篇外部论文、502 块隔离解析、Hybrid、`BAAI/bge-reranker-base` 固定 revision、
document routing、query decomposition、structured table/figure guard 和 parent/window；
完整 JSON：`/private/tmp/sci_rag_phase66_candidate_pool_10_50_fixed.json`。

| context k | fact macro | fact micro | 完整覆盖 | reference context | source page |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.936 | 0.932 | 47/53 = 0.887 | 0.846 | 0.929 |
| 50 | 0.995 | 0.993 | 52/53 = 0.981 | 0.899 | 1.000 |

@10 不完整的 6 题中，`drugr-11`、`scidqa-07`、`mgno-03`、`mgno-05`、`af3-08` 在 @50
召回缺失事实；`mgno-02` 在 @50 仍缺少 `3×3`。因此 @50 的改善主要是排序前缀扩大带来的
尾部证据，而不是生成质量提升。

## 解释与边界

- 该实验只测 required-fact 是否出现在检索 context，不能证明答案正确、公式语义正确或
  泛化能力。
- 50 个 context 会显著增加提示长度；本次 CPU reranker 单题约 2.8 秒、峰值 RSS 约 2.5 GB，
  不适合无条件作为网页默认。
- `mgno-02` 的金标准句在 PDF 解析出的 page 5 text chunk 中包含 `3 × 3`，但 @50 候选仍
  未召回该 chunk；这是公式/版面表示的检索缺失，不是答案 retry 可以修复的生成遗漏。
- 因此暂不把 `context_k` 改为 50，也不自动把所有 @10 未命中题重试；应先做公式归一化/查询
  别名的受控 A/B，并对被 @50 救回的五题再进行有限生成验证。

## 公式候选 A/B（Phase 6.8）

为处理 `mgno-02` 的 `3×3` 表示缺口，加入了默认关闭的
`SCI_RAG_FORMULA_EVIDENCE`/`--formula-evidence`。该通道只在问题明确询问公式、方程、卷积核
尺寸或离散系统形式时触发；候选必须有至少两个语义词命中，且符号命中贡献设上限，避免
PDF-to-Markdown 的大量 `=`、`*` 或残缺标记压过真正相关段落。图形块和参考文献块不会进入
公式候选。

公式候选还受来源范围约束：已有唯一 source route 时只扫描该来源；无路由但全库只有一个
来源时可扫描全库；无路由且存在多个来源时不注入公式候选。这样避免一个未指明论文的公式题
从其他论文借用方程。该逻辑不读取 gold，也不是符号求解器。

修正后的离线报告：
`/private/tmp/sci_rag_phase68_formula_guard_scoped_v2.json`。与无公式通道基线
`/private/tmp/sci_rag_phase66_candidate_pool_10_50_fixed.json` 对比：

- 全局五论文 @10/@50 指标完全不变（fact macro/micro/full 为
  `0.936/0.932/0.887` 与 `0.995/0.993/0.981`）。
- 单文档 MGNO @10 的 fact macro 从 `0.841` 提升到 `0.864`，完整覆盖从 `8/11` 提升到
  `9/11`；`mgno-02` 的 `3×3` 被候选覆盖。全局未限定的 `mgno-02` 仍为 partial，因其
  没有可靠来源路由，保守策略不会跨论文猜测。
- 全局 @50 仍有 1/53 题缺 `3×3`；这说明下一步需要查询/表示归一化或明确来源交互，而
  不是把 50 个块交给生成模型。

新增回归覆盖：来源路由内的公式提升、歧义多来源不注入，以及公式候选语义门槛。当前测试
共 142 项全部通过；生产默认仍为关闭。

## 下一步验收门槛

候选扩展若要进入网页 opt-in，至少需要：

1. 在同一源码指纹和固定模型下，@10 基线不退化；
2. 对明确“证据在候选尾部”的 5–10 题，context 扩展后答案事实覆盖提升，且无严重退化；
3. 记录额外 context 字符数、DeepSeek 调用次数、延迟和费用上限；
4. 对公式、表格和图形问题继续使用各自结构化路径，不让尾部证据跨表/跨图污染。
