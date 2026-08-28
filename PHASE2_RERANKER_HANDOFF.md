# Sci-RAG Phase 2 本地 Reranker 接手记录

日期：2026-08-28  
分支：`develop`

## 结论

固定 5 篇论文、53 题和 Hybrid top-50 候选后，本地 cross-encoder 达到预设门槛，可以
作为默认关闭的网页 A/B 实验路径接入，但不能改成默认检索。最终采用：

```text
Hybrid top-50 → BAAI/bge-reranker-base → 与原 Hybrid 排名再次 RRF → top-10
→ 显式 Table N / 确定性单元格保护 → Prompt
```

选择再次 RRF 而非纯 cross-encoder，是因为两者 @10 完整事实覆盖率都为 `0.698`，但 RRF
版本相对原 Hybrid 只回退 1 题，纯重排回退 4 题；RRF 版本的 17 个 table 类型用例事实
完整覆盖率也是 `1.000`。

## 模型与安全边界

- 模型：`BAAI/bge-reranker-base`
- revision：`2cfc18c9415c912f9d8155881c133215df768a70`
- 权重：`model.safetensors`，约 1.11 GB
- 位置：Hugging Face 用户缓存，仓库之外，不进入 Git
- 运行：`local_files_only=True`；缺少缓存时失败，不隐式联网
- 设备：当前 Mac CPU；batch 8；max length 512
- 没有安装或升级依赖；复用现有 `sentence-transformers==6.0.0`

只缓存安全权重和 tokenizer，没有下载重复的 `pytorch_model.bin` 或 ONNX。下载命令见
README；模型文件不应复制到项目目录或提交 Git。

## 代码实现

- 新增 `sci_rag_reranking.py`：无导入副作用的 cross-encoder 包装、稳定 tie-break、可选
  benchmark score cache 和统一 query-passage 文本。
- `evaluation/benchmark_retrieval.py` 报告 schema 升为 3；支持显式模型/revision、候选数、
  batch、max length、device、纯重排或 RRF 融合，并输出原检索分数、cross-encoder 分数、
  单题 mean/median/P95/max 延迟及进程峰值 RSS。
- `app.py` 只在 `hybrid + SCI_RAG_RERANKER_MODEL` 时加载模型；默认 dense 和原 Hybrid
  路径不变。重排之后仍执行全表加载、跨表拒答和确定性单元格定位。
- `.env.example`、README、benchmark 审计和回归测试同步更新；没有新增依赖。

## 固定基准结果

| 方法 / k | fact macro | fact micro | 完整 | 部分 | 零 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid @5 | 0.516 | 0.469 | 0.472 | 0.113 | 0.415 |
| Hybrid @10 | 0.627 | 0.592 | 0.547 | 0.132 | 0.321 |
| 纯 CE @5 | 0.755 | 0.735 | 0.660 | 0.208 | 0.132 |
| 纯 CE @10 | 0.783 | 0.776 | 0.698 | 0.189 | 0.113 |
| CE + 原排名 RRF @5 | 0.634 | 0.599 | 0.509 | 0.245 | 0.245 |
| CE + 原排名 RRF @10 | 0.785 | 0.776 | 0.698 | 0.170 | 0.132 |

最终 RRF 策略 @10 相对 Hybrid：完整覆盖率增加 15.1 个百分点；目标论文、页级、Table N
命中为 `0.981/0.905/0.944`。`scidqa-09` 没有在 top-10 命中结构化 Table 3 metadata，
但 required facts 完整；网页会在重排后另行加载所有结构化 table 块，因此必须保留现有
表格保护。

逐题对比中，最终 RRF 策略相对原 Hybrid 只有 `drugr-11` 从事实覆盖 0.75 降到 0.50，
遗漏 `Pareto` 和 `shortfall`。这说明整体门槛通过不等于逐题单调改善，网页 A/B 必须包含
该题。@10 总计仍有 16/53 题未完整覆盖，重排不能修复候选池本身缺失的证据。

## 资源实测

全局 53 题、每题 50 候选、CPU：

| 指标 | 秒 |
| --- | ---: |
| mean | 2.734 |
| median | 2.737 |
| P95 | 3.312 |
| max | 3.489 |

进程峰值 RSS 约 `2201 MB`。这是单进程离线值，尚未证明多用户并发、长时间运行或更大
知识库资源可接受。

## 已完成验收

- 50/50 项 unittest 通过。
- app、评估和 reranker 模块静态编译通过。
- 固定模型 revision 在 `HF_HUB_OFFLINE=1` 下加载成功。
- 真实 cross-encoder 对 DrugR 相关/无关片段正确排序。
- 真实 cross-encoder + fake collection/client 走通 `query_knowledge()`，返回 contexts 与
  Prompt 一致，不访问 Chroma 或 DeepSeek。
- 纯 cross-encoder 和保守 RRF 两组 5 篇/53 题报告均已复现；JSON 只写 `/tmp`。
- 未启动 Gradio、未调用 DeepSeek/RAGAS、未写或重建 ChromaDB、未执行 Git push。

## 待用户网页 A/B

一次只启动一个模式，使用现有数据库即可。先运行默认 Hybrid：

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

停止后运行 Hybrid + reranker：

```bash
HF_HUB_OFFLINE=1 \
SCI_RAG_RETRIEVAL_MODE=hybrid \
SCI_RAG_RERANKER_MODEL=BAAI/bge-reranker-base \
SCI_RAG_RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70 \
./venv/bin/python app.py
```

两种模式固定提问并记录回答、来源、首次/第二次延迟：

1. `Table 2 中 DrugR* 的整体优化得分是多少？`
2. `Table 2 中 DrugR* 的 Target property F1 score 是多少？`
3. `DrugR 的显式推理数据集（explicit-reasoning dataset）包含多少个样本？推理标注是通过什么管道构建的？`
4. `DrugR 的强化学习阶段如何解决多目标训练中的目标主导（objective domination）与目标饥饿（starvation）问题？`

停止条件：表格题跨表或不再确定性返回、上下文/来源与回答不一致、问题 4 明显丢失更多核心
机制、单题额外延迟不可接受、内存压力影响网页稳定性。网页 A/B 通过前不建议合并 `main`。

## 用户网页 A/B 结果

用户分别运行 Hybrid 与 Hybrid + reranker 后，报告问题 3、问题 4 在两种模式下答案一致：

- 问题 3 正确返回 `4,855`，也正确概括闭环逆向工程、候选生成和 ADMET 验证，但没有明确
  给出严格金标准中的 `DeepSeek-R1`、指纹相似度 `> 0.6`、`ADMETLab`，也没有说明用性质
  增量和两条 SMILES 生成机理性理由，因此记为“事实方向正确、答案部分完整”，不能记满分。
- 问题 4 正确覆盖样本级 Pareto 感知重加权、Reasoning/SMILES 组级缩放和带上限的通道级
  shortfall boost，达到该题严格金标准。

离线 reranker 报告中，问题 3 的 top-10 上下文已经覆盖上述四个必需事实。这说明该遗漏主要
是生成阶段没有综合多个互补片段，而不是候选池缺少证据；两种网页模式答案一致，也没有证据
表明该遗漏由 reranker 引入。后续应通过通用的多片段答案完整性约束和答案级评估处理，不能
针对 DrugR 写特殊规则。

本次回传尚未包含两道 Table 2 题的具体值、两种模式延迟和网页资源表现；最终提交前仍应确认
它们分别稳定返回 `0.2060`、`0.3404`，且没有跨表、崩溃或不可接受的额外延迟。

## 建议提交信息（网页 A/B 通过后）

```text
feat: add opt-in local cross-encoder reranking
```

```text
- rerank Hybrid top-50 with a pinned local cross-encoder
- conservatively fuse cross-encoder and original Hybrid ranks
- keep dense and Hybrid defaults unchanged and preserve table safeguards
- report fact coverage, per-case regressions, latency, and peak memory

Tests: 50 unittest passed
Tests: 5 papers / 53 cases, pure CE and CE+RRF reproduced offline
Not run by agent: Gradio, DeepSeek/RAGAS, Chroma write/rebuild, git push
```

## 仍未证明

- 没有系统性生成答案正确率或 RAGAS 新分数。
- 5 篇/53 题仍是小型基准，不能证明生产泛化。
- 未验证真实五论文 Chroma、多用户并发、服务超时和内存上限。
- 模型缓存不随 Git 分发；新环境必须显式下载并核验固定 revision。
