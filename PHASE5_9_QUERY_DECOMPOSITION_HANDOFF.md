# Phase 5.9 复合问题子查询对照

日期：2026-08-30  
分支：`develop`  
状态：默认关闭；尚未提交或 push

> 后续状态：Phase 6.0 已发现并修复“不可拆分问题仅去掉句末问号也生成第二变体”的缺陷，
> 并用应用等价的结构化表格保护重新验证。下文数字保留为 Phase 5.9 历史对照；当前结果与
> 结论见 `PHASE6_RETRIEVAL_CONVERGENCE_HANDOFF.md`。

## 目标

Phase 5.8 的失败审计显示，多篇论文集合中有一类通用问题：一个问题同时询问两个或
更多事实，但单个查询向量/词法查询只把其中一部分证据送入 cross-encoder。这里增加
一个可审计的多查询对照，不依赖论文名称、gold facts 或外部模型。

## 实现

- `sci_rag_retrieval.py` 新增 `query_variants()`：原问题始终保留；按中英文标点，或
  保守的“与/以及/和”规则拆出最多三个子句（总变体最多四个）。不翻译、不改写、不
  注入金标准。
- `evaluation/benchmark_retrieval.py` 的 `HybridRetriever` 增加
  `query_decomposition` 开关。各变体使用**原问题选出的同一 source route**，分别检索后
  用 RRF 融合；默认 `False`，原单查询排名不变。
- `app.py` 增加 `SCI_RAG_QUERY_DECOMPOSITION` 环境开关。开启时 dense 查询和 Hybrid
  词法查询都对同一组变体执行，保留原表号/行列确定性保护；默认 `False`。
- `.env.example`、benchmark README、单元测试同步更新。

## 离线正式对照

输入为固定五篇免费 PDF、53 道问题、同一解析器、同一 `BAAI/bge-small-zh-v1.5` 与
`BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70`，启用
document routing、candidate-k=50、等权 CE+原排名 RRF。完整 JSON 在仓库外：

`/private/tmp/sci_rag_benchmark_query_decomposition_phase59.json`

| 设置 | @10 fact macro | fact micro | 完整覆盖 | 目标论文 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| routing（Phase 5.8 控制） | 0.805 | 0.796 | 0.717 | 1.000 | 0.929 | 0.944 |
| routing + 子查询 RRF | 0.843 | 0.830 | 0.774 | 1.000 | 0.905 | 0.944 |

子查询 RRF 使完整覆盖提高 3/53 题（15→12 个未完整案例），但页级命中回到控制组
的 `0.905`，没有超过 routing 单独的 `0.929`。@10 路由仍为 `39/53`，`39/39` 正确，
无错误来源路由。剩余零覆盖/部分覆盖仍包括 AlphaFold 3 的 diffusion/立体化学段落、
MgNO 初始化、SciDQA Table 3 数值等候选召回或版面解析问题。

该实验因此达到“可以在网页中做受控 A/B”的门槛，但没有达到“改成默认”的门槛：页面
命中下降，且仍不能证明答案语义正确率。cross-encoder 全局单题重排 mean/P95 为
`2.811/3.420s`，峰值 RSS `2151.2 MB`；额外子查询只增加向量/BM25 检索，不改变
cross-encoder 候选数，仍需网页实测端到端延迟。

## 网页 A/B（下一步由用户执行或授权）

保持当前数据库不变，先运行默认路径作为控制，再仅设置一个环境开关：

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid \
./venv/bin/python app.py
```

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid \
SCI_RAG_QUERY_DECOMPOSITION=true \
./venv/bin/python app.py
```

若同时测试 reranker，再沿用已固定的 `SCI_RAG_RERANKER_MODEL` 与 revision。至少固定
记录 11 道种子题中的复合题、两道 Table 2 题和一个单事实题，比较：答案、contexts、
表号/单位、来源、首次/重复延迟。任一跨表、跨来源、表号命中或单事实题回退，都停止
把开关用于网页默认。

## 已完成的单论文网页回归

在现有 104 块 DrugR 数据库上，使用相同的 Hybrid + 固定 cross-encoder 配置，分别关闭
和开启该开关，通过 Gradio 的同一 `lambda_3` ChatInterface 端点各提交 11 道种子题。两组
均为 11/11 无调用错误；答案事实审计均为 `fact macro/micro=1.0000/1.0000`、
`full/partial/zero=1.0000/0/0`，逐题状态全部 `full→full`。结果文件仅在仓库外：

- `/private/tmp/sci_rag_answers_seed11_hybrid_reranker_default_phase59.jsonl`
- `/private/tmp/sci_rag_answers_seed11_hybrid_reranker_query_decomposition_phase59.jsonl`
- `/private/tmp/sci_rag_answer_compare_seed11_phase59.json`

这只是单论文、11 道固定问题的事实词面回归；不能证明多论文答案语义正确率、引用正确性
或泛化能力。两组延迟也受 DeepSeek 服务和本机负载影响，不能替代独立性能压测。

## 验收与停止条件

- 离线：`@10` fact macro/micro、完整覆盖、目标论文命中、页级命中、Table N 命中均与
  routing 控制并列；不能只看 fact macro。
- 线上：默认关闭路径回归不变；开启后复合题必须保留互补上下文，表格确定性单元格
  查询必须逐字一致；单事实题不能因为拆分而丢失主证据。
- 任何语义回答、单位、引用或延迟退化都只保留为 opt-in，不切换默认，不推进多模态、
  工具调用或 Graph-RAG。
