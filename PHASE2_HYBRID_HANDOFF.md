# Sci-RAG Phase 2 Hybrid/RRF 接手记录

日期：2026-08-28  
分支：`phase2/hybrid-rerank`

> 后续说明：本文件保留该分支提交时的历史代理分数。`phase2/context-coverage` 补全了
> DrugR GRPO/RL 金标准证据并加入 required-fact 覆盖后，当前可比结果以
> `PHASE2_CONTEXT_COVERAGE_HANDOFF.md` 和 `evaluation/benchmark/README.md` 为准。

## 本轮目标和结论

本轮将已验证的 BM25/RRF 离线实现接入应用，但保持为显式启用的实验路径。默认
`dense` 检索未改变，当前多论文证据也不足以把 Hybrid 设为默认。没有加入 learned
cross-encoder reranker，因为现有基准只证明 RRF 在部分检索代理上有收益，尚未证明额外
模型的质量收益能够抵消延迟、内存和依赖成本。

## 代码变更

- 新增 `sci_rag_retrieval.py`：无 Chroma、Sentence-Transformers、Gradio 或 API 导入的
  BM25、`RankedItem` 和 Reciprocal Rank Fusion 通用实现。
- `evaluation/benchmark_retrieval.py` 改为复用同一实现，避免离线基准和线上逻辑漂移。
- `app.py` 新增 `SCI_RAG_RETRIEVAL_MODE`；仅值为 `hybrid` 时才将 Chroma dense 候选与
  全库 BM25 候选做 RRF。未设置、设为 `dense` 或无效值时均走原 dense 路径。
- Hybrid 的词法快照在第一次问题时从当前 collection 读取文本和 metadata，在同一
  runtime 内复用；通过 `add_document_to_db()` 上传后会失效并在下次问题重建。
- 显式 `Table N` 的全表加载、跨表拒绝和确定性行列单元格定位保留在融合之后。
- `.env.example`、README、benchmark 文档和回归测试同步更新；没有增加依赖。

## 配置和行为边界

```text
SCI_RAG_RETRIEVAL_MODE=dense       # 默认，保持既有行为
SCI_RAG_HYBRID_CANDIDATE_K=50      # 每种排名的候选上限
SCI_RAG_HYBRID_RRF_K=60            # RRF 常数
SCI_RAG_CONTEXT_K=10               # 最终进入生成 Prompt 的上下文上限
```

临时启用网页对比：

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

Hybrid 首次提问会扫描当前 collection 并在进程内建立 BM25，因此知识库越大，首次延迟和
内存占用越高。缓存以 collection 数量作快速一致性检查，并由本应用上传路径主动失效；
同数量的外部原地修改、多进程写入或绕过 `add_document_to_db()` 的写入不能被可靠发现。
生产化前需要显式版本号或持久化词法索引，不能依赖当前原型缓存策略。

## 已执行验收

以下命令均已在本分支执行成功：

```bash
./venv/bin/python -m unittest discover -s tests -v
./venv/bin/python -m py_compile app.py sci_rag_core.py sci_rag_retrieval.py \
  evaluation/benchmark_retrieval.py evaluation/benchmark_loader.py \
  evaluation/validate_benchmark.py evaluation/evaluate.py test_setup.py
./venv/bin/python test_setup.py
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

结果：35/35 项 unittest 通过；5 篇论文、53 道用例完整且外部 PDF SHA-256 全部匹配。
测试使用 fake Chroma、fake embedding 和 fake model 验证：

- 默认 dense 的普通问题不读取全库、不创建词法快照；
- Hybrid 能引入 dense 未返回的词法候选，并复用快照；
- 返回的 `contexts` 与实际 Prompt 上下文一致；
- 上传完成后快照失效；
- Hybrid 下 Table 2 / DrugR* / Target property F1 仍确定性返回 `0.3404`，不调用模型；
- 明确指定但不存在的表格仍不会回退到其他表格。

网页回归还发现，旧表格意图规则会把任何含“多少/样本量/比率/数值”的正文问题误判为
表格题，从而加载并置顶所有 table 块。当前规则已收紧为只有明确出现 `Table`、`表2`、
`表格`、`下表`、`表中`等指代才走全表保护。一般数量题继续使用正常检索结果；新增测试
覆盖“显式推理数据集包含多少个样本、推理标注管道是什么”，保证它不会扫描全表。

同一问题还暴露了跨语言融合风险：中文问题在英文语料中可能只让 BM25 命中一个高频
ASCII 方法名，此时等权 RRF 会把正确 dense 证据挤出 top-k。当前 Hybrid 会在“没有
命中 CJK 词元且匹配 ASCII 词元少于两个”时跳过弱 BM25 列表、退回 dense 排名；明确
包含多个术语的 Table N 等查询仍正常融合。

## 离线多论文复测

BM25 和当前带弱词法信号保护的 Hybrid/RRF 均使用 5 篇/53 题的一个全局索引：

| 方法 / k | 参考片段覆盖 | 目标论文命中 | 页级命中 | Table N 命中 |
| --- | ---: | ---: | ---: | ---: |
| BM25 @1 | 0.252 | 0.811 | 0.310 | 0.667 |
| BM25 @5 | 0.365 | 0.868 | 0.429 | 0.833 |
| BM25 @10 | 0.506 | 0.906 | 0.690 | 1.000 |
| Hybrid/RRF @1 | 0.208 | 0.830 | 0.286 | 0.500 |
| Hybrid/RRF @5 | 0.481 | 0.943 | 0.619 | 0.778 |
| Hybrid/RRF @10 | 0.525 | 0.981 | 0.786 | 0.833 |

这些都是检索代理，不是答案准确率。Hybrid 在 @10 的参考片段、论文路由和页级代理上
高于 BM25，但 Table N 代理低于 BM25；应用的确定性表格保护补的是已识别结构化表格的
安全边界，不会把离线 Hybrid 的 Table N 指标改写成 1.000。

## 本轮仍未执行和未证明

- 用户已在当前单论文 Chroma 上运行 Gradio/DeepSeek 做上述 A/B；本代理未自行调用
  DeepSeek。尚未在包含五篇论文的真实 Chroma 上做网页 A/B，也未上传或重建数据库。
- 未运行完整 RAGAS；没有新的系统性答案正确率、忠实度或相关性分数。
- 未证明大知识库首次建索引的性能、长时间运行的缓存一致性或多用户并发安全。
- 未证明 Hybrid 优于 dense 默认路径的端到端答案质量，也未实现 learned reranker。

## 人工网页 A/B 验收结果

用户在当前知识库上分别启动默认 dense 和实验 Hybrid，固定表格题及其他功能均正常。
最初以下正文问题在两种模式下曾错误回答“资料未提供”：

```text
DrugR 的显式推理数据集包含多少个样本？推理标注是通过什么管道构建的？
```

根因是旧表格意图规则把“多少”误判为表格题，全库 table 块占据最终上下文。修复表格
意图和弱跨语言 BM25 保护后，用户重新验证得到：

- dense：正确返回 `4,855`，并说明从 DrugBank 收集超过 10,000 个种子、生成相似结构、
  模拟器评估、正负样本配对，再由 DeepSeek-R1 标注从负样本到正样本的推理过程；
- Hybrid：返回相同核心事实，并补充 85%/10%/5%（4,126/485/244）的数据划分；
- 两种模式来源均为 `2602.08213v1.pdf`，不再错误把 table 块作为主要证据。

这可判为“事实正确、网页回归通过”，但不是严格金标准满分：两份回答使用论文前部的
reverse data engineering 概述，没有穷尽 §4.4.1 中的指纹相似度 `>0.6`、ADMETLab 验证、
以及基于性质增量和两条 SMILES 生成机理理由等细节。后续答案质量评估应把本题标为
部分事实覆盖，不能因没有错误事实就记为完整正确。

## 后续复测命令

在 Git 工作区干净且 `.env` 已配置的前提下，可分别启动两个进程复测同一组问题；
一次只启动一个，停止后再切换模式：

```bash
./venv/bin/python app.py
```

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

每种模式至少记录：首次问题耗时、第二次同问题耗时、回答、来源和是否命中正确论文/页/表。
必须包含普通正文题、术语精确匹配题、跨论文相似术语题，以及以下表格回归题：

```text
Table 2 中 DrugR* 的整体优化得分是多少？
Table 2 中 DrugR* 的 Target property F1 score 是多少？
Table 1 中 DrugR 的整体优化得分是多少？
```

停止条件：任一明确 Table N 问题跨表取数、Hybrid 返回上下文与引用不一致、首次建索引
导致不可接受的资源占用，或 Hybrid 在固定问题集上没有可复现收益，都不得改成默认。

## 建议提交信息

建议 commit 标题：

```text
feat: add opt-in hybrid retrieval with table safeguards
```

建议 commit body：

```text
- share side-effect-free BM25 and RRF across app and benchmark
- keep dense retrieval as default and gate hybrid mode by environment
- cache lexical collection snapshots and invalidate them after uploads
- preserve deterministic Table N filtering after rank fusion
- add offline runtime contract and retrieval regression tests

Tests: 35 unittest passed
Tests: benchmark manifest 5 papers / 53 cases, all SHA-256 matched
Tests: BM25 and Hybrid/RRF proxy metrics reproduced
Not run: Gradio, DeepSeek/RAGAS, Chroma rebuild, git push
```
