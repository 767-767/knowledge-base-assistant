# Phase 6.0 检索路径收敛交接

日期：2026-08-30  
分支：`develop`  
状态：已实现并完成离线验证；默认检索配置未切换；尚未提交或 push

## 本阶段目标

Phase 5.9 的五论文/53 题对照有两个必须先解释的问题：`scidqa-09` 的 Table 3
结构化证据没有进入离线 top-10，以及 `table-llm-10` 在开启查询分解后从完整覆盖退化为
零覆盖。本阶段只处理可复现的通用机制，不增加论文名、题目答案或 gold-fact 特例。

## 已接受的修正

1. `query_variants()` 不再把一个不可拆分的问题生成为“原问题”和“仅去掉句末问号的问题”。
   旧行为会对两个略有差异的 embedding 排名再次 RRF，`table-llm-10` 的结论块因此从
   top-10 被挤出。现在只有发生真实标点/并列子句拆分时才产生额外变体。
2. 表格意图从“出现 table/表格一词”收紧为明确表号或指代表达，例如 `Table 3`、`表2`、
   `下表`、`该表格中`。`表格理解训练`、`科学表格表示学习` 等研究主题不再触发全表扫描。
3. 网页在 document routing 已选择来源时，结构化表格扫描使用同一来源过滤；这避免同表号、
   同行列名的另一篇论文先被确定性查找命中。ChromaDB 1.5.9 的实际 `validate_where()` 已确认
   `$and` 过滤格式有效。
4. 离线 benchmark 新增 `--structured-table-guard`，显式区分原始检索排名与网页应用路径。
   该开关按网页逻辑扫描 canonical table chunks，并在可确定行列时只评估实际返回的结构化表格。

以上改动没有修改默认 Dense/Hybrid、routing、query-decomposition 或 reranker 开关，也没有
写入当前 ChromaDB。

## 正式离线对照

固定输入：5 篇 PDF、53 题、479 个解析块、`BAAI/bge-small-zh-v1.5`、
`BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`、candidate-k=50、
CE 与原 Hybrid 排名等权 RRF、document routing、query decomposition。模型均以
`HF_HUB_OFFLINE=1` 从本地缓存加载；没有调用 DeepSeek、RAGAS 或其他外部 API。

| 配置 | fact macro | fact micro | 完整覆盖 | 目标论文 | 来源页 | Table N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5.8 routing 控制 | 0.805 | 0.796 | 0.717（38/53） | 1.000 | 0.929 | 0.944 |
| Phase 5.9 旧查询分解 | 0.843 | 0.830 | 0.774（41/53） | 1.000 | 0.905 | 0.944 |
| Phase 6 修正 + 结构化表格保护 | **0.881** | **0.871** | **0.811（43/53）** | **1.000** | **0.929** | **1.000** |

Phase 5.9 到 Phase 6 只有两道状态迁移，且均为 `zero -> full`：

- `scidqa-09`：确定性 Table 3 单元格保护补回 `46.63` 与 `54.03`；
- `table-llm-10`：去除伪重复查询后，结论证据恢复到第 9 位。

没有题目从 full/partial 向下退化。全局 reranker mean/P95 为 `2.746/3.166s`，与
Phase 5.8 控制基本持平。正式 JSON 保存在仓库外：

`/private/tmp/sci_rag_benchmark_phase60_queryfix_tableintent_v2.json`

## 被拒绝的邻接块方案

`--adjacent-context` 是一个仅用于离线复现的负对照：围绕前两个文本锚点插入同来源、同页
相邻块，并跳过表格和参考文献。它把完整覆盖提高到 `44/53`，但存在不可接受的双向迁移：

- 改善：`af3-04`、`af3-10`、`mgno-01`、`table-llm-03`；
- 退化：`af3-11`（full -> partial）、`scidqa-08`（full -> zero）、
  `table-llm-10`（full -> zero）。

来源页命中同时从 `0.929` 降至 `0.881`。因此该方案不接入网页、不成为默认，只保留为
“直接插入邻块会挤掉尾部证据”的可复现停止证据。JSON 位于：

`/private/tmp/sci_rag_benchmark_phase60_adjacent_control.json`

## 剩余 10 道未完整案例

- zero：`af3-04`、`af3-10`、`mgno-03`；
- partial：`scidqa-07`、`af3-08`、`table-llm-03`、`mgno-05`、`drugr-11`、
  `mgno-01`、`mgno-02`。

多道题的正确证据位于高排名块的同页邻段，但本阶段已经证明“把邻段作为额外 top-10 项插入”
会造成尾部证据回退。下一步应比较不占用额外排名槽位的 parent/window context enrichment，
即把安全邻段附加到已选锚点的生成上下文中，同时保留原 top-10 排名和 context ID 追踪；在接入
网页前必须先离线证明来源页、`table-llm-10`、`scidqa-08` 和 `af3-11` 不退化。

## 验证

- `python -m unittest discover -s tests -v`：113/113 通过（含 Phase 6.1 parent/window 与图像/OCR 隔离回归）；
- 五论文完整离线对照：53/53 执行完成；
- 本地模型强制离线，无隐式下载；
- 未启动 Gradio，未调用外部模型 API；
- 未重建、写入或删除当前 ChromaDB；
- 未执行 git add、commit、push 或分支操作。

## 结论边界

`0.881/0.871/0.811` 是目标论文检索上下文中的确定性事实词面覆盖，不是答案正确率、
Faithfulness、引用正确性或泛化能力。新的路由/查询分解组合仍需在隔离五论文数据库上完成
53 题端到端生成与人工语义复核，才能讨论是否改变网页默认配置。

## Phase 6.1 parent/window 上下文

直接把相邻块插入 top-10 的负对照会挤掉尾部证据，因此新增另一条默认关闭的路径：排名、
context 槽位和锚点 ID 均保持不变，只在前两个文本锚点内部拼接同来源、同页、相邻的正文块。
表格、References/Bibliography、图像/OCR 块、跨页、跨来源和已经位于 top-10 的邻块均不重复加入；每个有效
context 的 metadata 记录 `window_chunk_indices`/`window_chunk_ids` 和新增字符数。

离线 benchmark 通过 `--parent-window` 评估**实际扩展后的上下文文字**，而不是把邻块当作
额外排名项。相同五论文配置的 @10 结果为：

| 配置 | fact macro | fact micro | 完整覆盖 | 来源页 | Table N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 6.0 控制 | 0.881 | 0.871 | 0.811（43/53） | 0.929 | 1.000 |
| + parent/window | **0.936** | **0.932** | **0.887（47/53）** | **0.929** | **1.000** |

只有四道状态迁移，均为改善：`af3-04`、`af3-10` 从 zero 到 full，`mgno-01`、
`table-llm-03` 从 partial 到 full。没有逐题退化，目标论文命中保持 `1.000`。53 题中
36 题实际扩展，共 61 个锚点、78 个邻块、60,388 个字符，平均约 1,139 字符/题；这只是
字符开销，不是 token/API 成本测量。完整 JSON 位于：

`/private/tmp/sci_rag_benchmark_phase61_parent_window_visual_guard.json`

网页通过 `SCI_RAG_PARENT_WINDOW=true` 显式开启，默认仍为 `false`。真实隔离 Chroma smoke
使用新解析的 DrugR PDF 建立 101 块索引，101/101 块均保存 `source/page/chunk_index`；一次
查询保持 10 个 context ID，扩展 2 个锚点、增加 1,469 字符，且返回 contexts 全部逐字进入
生成 prompt。该 smoke 使用 fake client，没有调用 DeepSeek。

当前生产 `chroma_db` 的 SQLite 只读核验显示 104/104 块有 `source/type`，但 0 块有
`page/chunk_index`。为避免猜测 Chroma 返回顺序，parent/window 在该旧索引上安全 no-op；
不要删除或重建生产库来做验证。仓库外既有五论文临时索引的 479/479 块元数据完整，可用于
下一阶段受控生成 A/B。

检索层仍未完整的 6 题是 `mgno-03`、`scidqa-07`、`af3-08`、`mgno-05`、`drugr-11`、
`mgno-02`；同一配置的 parent/window 端到端答案、拒答、引用和延迟对照已在下节完成。

## Phase 6.1 端到端生成 A/B（已完成）

在 479 块隔离五论文索引上，以相同 Hybrid+固定 cross-encoder、document routing、corrected
query decomposition、structured table guard 和 top-10 配置交替运行控制组与 parent/window
组。共完成 53×2 次，API 错误为 0；其中 `af3-03`、`af3-05`、`af3-06` 在图块隔离代码完成后
按最终实现各强制重跑一次。两组 53 题的 context ID 完全一致，parent/window 只改变锚点
文字；36 题扩展，共增加 59,130 字符。

自动答案事实覆盖从 macro/micro `0.8459/0.8151` 提升到 `0.9088/0.8836`，完整覆盖从
40/53 提升到 45/53，状态提升 5、退化 0。逐题人工语义复核将控制组判为
`40 correct / 11 partial / 2 incorrect`，parent/window 判为
`43 / 8 / 2`；真实改善为 `table-llm-03`、`mgno-05`、`af3-04`、`af3-10`。
`table-llm-10` 是严格中文别名未命中的词面假改善；`af3-06` 两组均把图中一个样本量与
数据集关联错，说明图像/OCR 证据仍需人工或多模态核验，但没有形成 parent/window 相对退化。

控制/window 单次均值延迟约为 `4.389/4.317s`；这是交替单次运行，不能据此声称窗口降低
延迟。开关继续默认关闭：当前结果证明了可追踪的正向 A/B 信号，但不是多随机种子、独立
标注或 RAGAS 证明，也不足以覆盖图像表格的语义正确性。外部工件：

- 完整 53 行：
  `/private/tmp/sci_rag_answers_phase61_control_53.jsonl`、
  `/private/tmp/sci_rag_answers_phase61_parent_window_53.jsonl`；
- 自动对照：`/private/tmp/sci_rag_phase61_answer_compare_53.json`；
- 人工复核：`/private/tmp/sci_rag_phase61_control_human_review_53.json[l]`、
  `/private/tmp/sci_rag_phase61_parent_window_human_review_53.json[l]`；
- 断点/强制重跑脚本：`/private/tmp/sci_rag_phase61_generation_ab.py`。

本轮未运行 RAGAS。人工复核是内部单人复核，不是独立双盲标注；自动事实覆盖是词面信号，
不能单独替代答案正确率。下一步应优先处理 figure/OCR 的结构化证据或重复随机种子，而不是
继续扩大 parent/window。

## Phase 6.2 born-digital Figure 坐标证据（已实现，默认关闭）

针对 `af3-06` 的 `25/38/8` 被扁平 picture text 错配为 `25/38/28`，本阶段先逐页渲染并
核对 PDF，再读取同页 `Page.get_text("blocks", sort=True)`。Figure 1 中三个目标标签和值
分别处于重叠的水平 x 区间，而 `Glycosylation / n=28` 位于独立区间，证明问题不是原 PDF
缺字，而是线性化丢失二维关联。

新增 `SCI_RAG_SPATIAL_FIGURE_EVIDENCE=false` / `--spatial-figure-evidence`：仅在显式开启并
重新入库时生成带 `figure_kind/figure_number/page/x/y` 的 `type=figure` 块；显式
`Figure N`/`图N` 查询在既有 source route 内扫描精确图号。figure chunks 不参与普通
dense、BM25、路由或 cross-encoder 候选；命中坐标证据时也移除扁平 picture-text 和复合
答案的自动补充段，防止 `n=28` 重新进入生成上下文。

五篇固定论文共新增 23 个图块，隔离库为 502 块。完整 53 题离线结果保持 Phase 6.1 的
@10 fact macro/micro/full `0.936/0.932/0.887`（47/53），目标论文/来源页/Table N 为
`1.000/0.929/1.000`，没有非目标问题的状态退化。一个 502 块隔离 Chroma 上的真实
DeepSeek 针对性复测把首个 context 定位到 AlphaFold 3 第 2 页 Figure 1，并正确回答
蛋白质–RNA `25`、蛋白质–dsDNA `38`、CASP15 RNA `8`。

边界：这不是 OCR 或多模态；图片不保存、不进入模型，纯像素/扫描图不会生成证据。单题
修复也不能证明所有图表泛化，因此开关继续默认关闭，生产 `chroma_db` 未修改。正式 JSON：
`/private/tmp/sci_rag_benchmark_phase62_spatial_figure_final.json`；定向生成：
`/private/tmp/sci_rag_phase62_af3_06_final.json`。

最终离线回归为 120/120，通过完整 benchmark 清单及五个 PDF SHA-256 校验；生产 SQLite
使用只读模式复核仍为 104 个 embeddings。未启动 Gradio、未运行 RAGAS、未执行 git
add/commit/push。
