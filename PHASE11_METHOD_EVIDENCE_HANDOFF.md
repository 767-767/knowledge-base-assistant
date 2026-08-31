# Phase 11 方法段证据修复交接

日期：2026-08-31  
分支：`develop`  
状态：完成定向修复与单题在线复核；未提交或 push

## 目标

Phase 9/10 的当前版本复核中，`scidqa-07` 在 `context_k=10` 下错误拒答。问题不是
论文没有这组事实，而是唯一已路由到 SciDQA 的首个检索块只包含被截断的
`Experimental Setup` 导语；包含四种配置完整清单的附录 `B.1 Experimental Setup`
位于候选池尾部。目标是在不把全局上下文扩大到 50、不过度放宽多论文检索边界的
前提下补回这类方法段落证据。

## 通用代码改动

### 1. 唯一来源下的同节扩展

`app.py` 的 `_section_expansion_result()` 现在接受已有的 `DocumentRoute`（约
`app.py:746-813`）。原有安全条件仍然保留：

- 只对复合/列表型问题启用；
- 必须已有检索锚点和可匹配的 `headers`；
- 只选择一个 `source`，最多加入 `MAX_SECTION_EXPANSION_CHUNKS=6`；
- 没有唯一 route 的多论文问题仍跳过扩展，避免同名标题跨论文污染；
- route 指向的 source 不存在时安全回退原始结果。

`query_knowledge()` 在完成路由和重排后把 route 传入该扩展（约 `app.py:1044`）。
因此它不改变 dense/Hybrid/reranker 的原始排名，也不改变 `context_k` 上限，只在
明确选出一个论文来源时从该来源的同一标题路径补充邻近块。

### 2. 方法配置问题的标题词和列表识别

`app.py:684-705` 增加了量化列表问题（例如“四种……分别是什么”）的复合问题识别，
并将“实验配置/配置”映射到 `experimental`、`setup`、`configuration(s)` 等标题词。
这不是 SciDQA 特例；同样适用于其他论文中询问若干实验设置、步骤或管道的复合问题。

### 3. 离线 benchmark 镜像

`evaluation/benchmark_retrieval.py:244-295` 的 `_section_expansion_indices()` 采用同样的
`route_source` 规则，供不调用 API 的检索对照使用。新增回归覆盖多来源语料下的唯一
route；无 route 的多来源负例仍不扩展。

## 验证证据

### 自动回归

以下命令均在仓库目录执行：

```bash
./venv/bin/python -m unittest discover -s tests -q
./venv/bin/python -m py_compile app.py evaluation/benchmark_retrieval.py
git diff --check
```

结果：`148` 项单元测试全部通过，编译检查和 diff 空白检查通过。新增/加强的测试位于
`tests/test_benchmark_retrieval.py:424-436` 和 `tests/test_core.py:1452-1458`，分别验证
唯一 route 的 benchmark 扩展和运行时不会注入其他论文、但会保留同源邻块。

### 受控 DeepSeek 单题复核

使用固定的隔离数据库 `/private/tmp/sci-rag-phase62.st4cV2/chroma_db`（502 块）、
Hybrid、固定 `BAAI/bge-reranker-base` revision、`context_k=10`，只跑 `scidqa-07` 一题。
生成输出和完整 context trace 保存在仓库外：

`/private/tmp/sci_rag_phase11_method_section_scidqa07.jsonl`

结果：`1/1` API 调用成功；上下文包含同一来源的 `B Experiments > B.1 Experimental
Setup` 两个块，列出 `closed-book`、`title-abs`、`RAG`、`full-text` 四种配置；不含
其他论文 source。离线 answer-fact 审计为：

```text
required_fact_coverage_macro = 1.0
required_fact_coverage_micro = 1.0
full_fact_coverage_rate      = 1.0 (1/1)
```

人工检查回答没有把题目错误标为“资料未提供”，并且四个配置与 PDF 附录清单一致。

## 与旧结果的差异

Phase 7.1/10 的正式 `context_k=10` 结果中，`scidqa-07` 只有截断导语，属于错误拒答。
本轮仍使用同一 `context_k=10` 和同一 reranker，只改变了唯一来源下的受限同节证据补充，
因此该单题结果支持“候选池尾部方法证据可通过安全扩展恢复”，不支持把单题改善外推为
53 题语义准确率或泛化能力提升。

## 默认行为和边界

- `SCI_RAG_DOCUMENT_ROUTING` 仍默认 `false`；未显式启用路由时，多论文同节扩展仍然
  保守跳过。
- 没有把 `SCI_RAG_CONTEXT_K` 改为 50，也没有改 `hybrid_candidate_k`、reranker 或
  公式证据默认开关。
- 没有引入 OCR/VLM、工具调用或 Graph-RAG；图片和生产 `chroma_db/` 未修改。
- 尚未重跑完整 53 题生成或重新做全量人工语义复核，不能宣称整体准确率改善。
- 该扩展依赖解析块中的 `source` 和 `headers`；旧库缺少这些 metadata 时应安全回退，
  不应根据 Chroma 返回顺序猜测邻接关系。

## 下一步门槛

若要改变默认开关或把该策略作为正式能力，需在同一固定配置下重跑完整 53 题，并至少
对所有原有非完整题做独立人工语义复核，确认没有跨论文污染、表号/单位/公式退化和
延迟回归。若只继续积累证据，优先收集 5--10 个不同论文的“唯一来源 + 方法/列表题”
失败样本，再决定是否扩大标题别名或引入更复杂的层级检索。
