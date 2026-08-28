# Sci-RAG Phase 2 多事实上下文覆盖接手记录

日期：2026-08-28  
分支：`phase2/context-coverage`

## 本轮结论

本轮把“检索到了某个相关片段”细化为“一个问题要求的原子事实是否都进入了目标论文的
top-k 上下文”。53 道题现在都能以人工金标准片段或显式别名核验 required facts；报告
可以区分完整、部分和零覆盖，并定位每题遗漏事实。该指标不调用模型，也不等同于答案
正确率。

Hybrid 在 top-5 有小幅事实完整度收益，但到 top-10 与 BM25 持平，因此仍不应替换默认
dense。另一方面，Hybrid top-50 候选池明显高于 top-10，证明 reranker 有受控实验价值；
但候选池仍有 11 题不完整，reranker 不能代替解析、切分或查询扩展。

## 实现内容

- 新增 `evaluation/context_coverage.py`：确定性的事实表面匹配、full/partial/zero 分类、
  macro/micro 聚合和金标准自洽检查。
- `evaluation/benchmark_loader.py` 支持并严格校验逐事实 `required_fact_aliases`。
- `evaluation/validate_benchmark.py` 要求每个 required fact 或其显式别名出现在 gold
  contexts；当前 53/53 题通过。
- `evaluation/benchmark_retrieval.py` 的报告 schema 升到 2，增加 top-k 事实覆盖、分论文、
  分题型和逐题失败列表；`--show-failures` 可直接打印遗漏事实。
- 补全 DrugR GRPO/RL 的真实论文证据，并为中文标注与英文证据的表面差异增加人工别名。
- 新增事实匹配边界、特殊 token、别名、聚合、跨论文证据隔离和多事实 top-k 回归测试。

## 固定基准结果

同一组 5 篇论文、53 题、全局索引，外部 PDF SHA-256 均与清单一致：

| 方法 / k | fact macro | fact micro | 完整 | 部分 | 零 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 @1 | 0.308 | 0.279 | 0.283 | 0.075 | 0.642 |
| BM25 @5 | 0.495 | 0.483 | 0.434 | 0.151 | 0.415 |
| BM25 @10 | 0.627 | 0.599 | 0.547 | 0.170 | 0.283 |
| dense @1 | 0.157 | 0.163 | 0.132 | 0.057 | 0.811 |
| dense @5 | 0.368 | 0.367 | 0.302 | 0.151 | 0.547 |
| dense @10 | 0.539 | 0.537 | 0.434 | 0.189 | 0.377 |
| Hybrid @1 | 0.233 | 0.245 | 0.208 | 0.057 | 0.736 |
| Hybrid @5 | 0.516 | 0.469 | 0.472 | 0.113 | 0.415 |
| Hybrid @10 | 0.627 | 0.592 | 0.547 | 0.132 | 0.321 |
| Hybrid @50 | 0.849 | 0.844 | 0.792 | 0.151 | 0.057 |

Hybrid @10 仍有 24 题未完整覆盖；@50 降到 11 题。@50 完全零覆盖的是 `af3-10`、
`mgno-03`、`mgno-11`；其余 8 题为部分覆盖。完整逐题列表可通过本页命令复现，不将临时
JSON 报告提交仓库。

## 验收命令

```bash
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --retriever hybrid --top-k 1,3,5,10 --show-failures
```

Dense/Hybrid 只加载已经缓存的 `BAAI/bge-small-zh-v1.5`，并设置
`local_files_only=True`；没有下载模型。BM25 不加载 embedding。所有 PDF 只在内存中解析，
没有启动 Gradio、调用 DeepSeek/RAGAS、写 ChromaDB 或修改现有数据库。

## 下一步：reranker 实验门槛

建议下一分支只做本地、默认关闭的 cross-encoder reranker 对比，不立即接入线上默认。
固定 Hybrid top-50 候选池，至少报告 top-5/top-10 required-fact macro、micro、完整覆盖率、
Table N 命中和每题耗时。建议验收门槛：top-10 完整覆盖率比当前 0.547 至少提高 5 个百分点，
Table N 和现有确定性单元格回归不下降，并记录模型大小、峰值内存及冷/热延迟。若本地没有
合适缓存模型、收益未达门槛或延迟不可接受，就停止接线，转向对 @50 仍不完整的 11 题做
解析/切分和查询扩展诊断。

## 未证明

- 没有运行答案生成、人工答案复核或新 RAGAS，不能声称答案正确率提高。
- 事实表面匹配不会识别未声明的同义改写，也不会验证事实之间的逻辑关系。
- 5 篇/53 题仍是小型研究基准，不证明跨领域生产泛化。
- 没有测试真实五论文 Chroma、网页并发、首次索引资源和长时间缓存一致性。

## 建议提交信息

建议 commit 标题：

```text
feat: add deterministic multi-fact retrieval coverage
```

建议 commit body：

```text
- validate required facts and explicit cross-language aliases against gold contexts
- report macro/micro and full/partial/zero fact coverage by k, paper, and case type
- expose per-case missing facts for BM25, dense, and Hybrid diagnostics
- add boundary, aggregation, and cross-document evidence tests

Tests: 43 unittest passed
Tests: benchmark manifest 5 papers / 53 cases, all SHA-256 matched
Tests: BM25, dense, Hybrid top-1/3/5/10 and Hybrid top-50 reproduced offline
Not run: Gradio, DeepSeek/RAGAS, Chroma rebuild/write, git push
```
