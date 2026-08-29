# 修改记录（Modification Log）

> 记录本项目从"通用学习工作台"改造为 Sci-RAG 的全过程。按时间倒序排列。

> 2026-08-29 Phase 3 答案完整性基础：新增无模型的 `evaluation/answer_audit.py`，对已保存
> 的 JSON/JSONL 回答按 required facts 输出 full/partial/zero 和遗漏事实；`evaluate.py` 的
> 答案事实检查改为复用同一套规范化/别名逻辑。科学问答 Prompt 增加通用多片段互补事实
> 整合约束，未写入 DrugR 特例。新增 `PHASE3_ANSWER_COMPLETENESS_HANDOFF.md` 和离线测试；
> 用旧 `evaluation/evaluation_report.json` 离线回归得到答案 fact macro/micro `0.8409/0.8387`，
> full/partial/zero `0.8182/0.0909/0.0909`，定位第 3、9 题遗漏事实；随后完成一次 4 题
> Dense 与 Hybrid+Rerank 的 DeepSeek A/B，并将结果写入 `PHASE3_ANSWER_COMPLETENESS_HANDOFF.md`。
> 第二轮新增通用 `build_evidence_ledger()`，并让复合问题在最终截断前扩展同来源、同小节的
> 兄弟块；清单按上下文保留高信号行，补充规则排除表格/表题/统计摘要噪声。随后补充多列
> 行查询、`target set` 别名和比较题全表保护，离线测试达到 `58/58`。使用现有 104 块数据库
> 对完整 11 题做 Dense 与 Hybrid+Rerank 端到端回归，两种模式答案 fact macro/micro 与 full
> 覆盖均为 `1.0000`；Q5/Q6/Q10 的语义路径也通过网页函数验证。五篇论文的离线检索回归显示
> Hybrid+CE+RRF @10 fact macro/micro=`0.785/0.776`、完整覆盖=`0.698`，仍有 16/53 题不完整。
> 尚未重跑 RAGAS 或完成五篇论文生成答案统计，不能据此证明整体泛化。

> 2026-08-28 Phase 2 reranker 网页 A/B 回传：用户报告 Hybrid 与 Hybrid + reranker 对
> DrugR 显式推理数据集和多目标平衡问题给出相同答案。多目标题完整覆盖 Pareto、
> Reasoning/SMILES 分组和 shortfall boost；数据集题正确返回 `4,855` 及管道方向，但遗漏
> `DeepSeek-R1`、`>0.6`、`ADMETLab` 和性质增量/SMILES 理由生成细节，严格记为部分完整。
> 离线 top-10 已含四项必需事实，故记录为通用多片段生成完整性缺口，而非 reranker 候选池
> 缺失或 DrugR 特例修复项。Table 2 精确值、网页延迟和资源表现仍待最终回传确认。

> 2026-08-28 Phase 2 本地 reranker：经用户授权，将固定 revision 的
> `BAAI/bge-reranker-base` safetensors 缓存到仓库外，不增加依赖或提交模型。新增无导入
> 副作用的 cross-encoder 重排、延迟/峰值内存报告和 fake-model 测试；应用只在 Hybrid
> 且显式配置模型时本地加载，并把重排结果与原 Hybrid 排名再次 RRF，之后继续执行 Table N
> 和确定性单元格保护。5 篇/53 题 @10 fact macro/micro/完整覆盖率从 Hybrid 的
> `0.627/0.592/0.547` 提升到 `0.785/0.776/0.698`；CPU mean/P95 为 `2.73/3.31` 秒，
> 峰值约 2.20 GB。仍有 16 题不完整，未运行 Gradio、DeepSeek/RAGAS 或写 ChromaDB。

> 2026-08-28 Phase 2 多事实上下文覆盖：新增纯本地、确定性的 `required_facts` 覆盖统计，
> 输出 top-1/3/5/10 的 macro/micro、full/partial/zero、分论文/分题型和逐题遗漏事实。
> 跨语言或表面形式差异只允许逐题声明 `required_fact_aliases`，校验器保证别名出现在人工
> gold contexts；DrugR 的 GRPO/RL 金标准片段同步补全。53/53 题标注自洽，43 项测试通过。
> 当前 BM25 / dense / Hybrid @10 完整事实覆盖率为 `0.547/0.434/0.547`；Hybrid @50
> 候选池为 `0.792`，支持把本地 reranker 作为下一项受控实验，但 11 题在 @50 仍不完整，
> 不能只靠重排。未调用 DeepSeek/RAGAS、未启动 Gradio、未写 ChromaDB。

> 2026-08-28 Phase 2 网页 A/B 复测：用户分别运行 dense 与 Hybrid，表格题和其他既有
> 功能正常。修复后的显式推理数据集问题在两种模式下均返回 `4,855`、DrugBank 正负
> 样本构造和 DeepSeek-R1 标注流程，来源为同一 PDF，不再由 table 块挤占证据。该回答
> 事实正确但只覆盖论文概述，未穷尽 §4.4.1 的 `>0.6`、ADMETLab 和性质增量/SMILES
> 理由生成细节，因此记录为网页回归通过、严格金标准部分覆盖，而非完整答案满分。

> 2026-08-28 Phase 2 数量型正文问题修复：此前 `is_table_question()` 把单独出现的
> “多少/样本量/比率/数值”也判为表格意图，会加载并置顶全库 table 块。对“显式推理
> 数据集包含多少个样本、标注管道是什么”这类正文题，8 个表格块因此占据最终 10 个
> 上下文，包含 `4,855` 和 reverse-engineering pipeline 的真实正文被挤出。现在只有
> 明确出现 `Table`、`表2`、`表格`、`下表`、`表中`等指代才启用全表保护；一般数量题
> 保持正常 dense/Hybrid 排名。对于中文问题只在英文语料命中少于两个 ASCII 词元的
> 弱 BM25 场景，Hybrid 现在跳过词法列表并保留 dense 排名。当前 Hybrid top-1/3/5/10
> 参考片段覆盖代理为 `0.208/0.349/0.481/0.525`，页级代理为
> `0.286/0.476/0.619/0.786`。新增意图、弱词法信号和运行时契约回归，完整离线测试为
> 35/35；当前 SQLite 仅只读核验，未调用 DeepSeek、未写 ChromaDB。

> 2026-08-28 Phase 2 可选 Hybrid 线上接入：新增无副作用的 `sci_rag_retrieval.py`，
> 让离线基准和 `app.py` 共用 BM25 与 RRF；`SCI_RAG_RETRIEVAL_MODE=hybrid` 才启用
> Chroma dense + 内存 BM25 融合，默认仍为 `dense`。词法快照首次查询构建、后续复用，
> 通过当前 runtime 上传文档后失效；显式 Table N 过滤和确定性单元格定位继续在融合后
> 执行。35 项离线测试、5 篇/53 题 SHA-256 校验、BM25 与 Hybrid 基准复测通过；未启动
> Gradio、未调用 DeepSeek/RAGAS、未写 ChromaDB。当前证据不支持把 Hybrid 设为默认，
> 也未加入 learned cross-encoder reranker；详情见 `PHASE2_HYBRID_HANDOFF.md`。

> 2026-08-28 Phase 2 解析回归：`sci_rag_core.py` 兼容表格前后 caption、跨列分组表头、
> PDF 断词、独立单位列和加粗标记相邻实体，并排除 DOI/作者元数据布局表。新增
> `tests/test_parser_regression.py` 的 8 个离线 fixture；四篇外部 PDF 只读冒烟确认
> SciDQA 6/6、Scientific Table LLM 3/3、MgNO 7/7 的表号识别，AlphaFold 3 不再误建
> 元数据表格块。MgNO Table 1/4 的确定性单元格回归返回 `0.339`/`1.63`。未调用外部模型、
> 未启动 UI、未写入 ChromaDB。

> 2026-08-28 Phase 2 离线检索基线：新增 `evaluation/benchmark_retrieval.py` 和
> `tests/test_benchmark_retrieval.py`。BM25-lite 在五篇论文的全局内存索引上输出文档、
> 参考片段、页码和 Table N 命中代理；全局 top-1/3/5/10 目标论文命中率为
> `0.811/0.868/0.868/0.906`，显式表号命中率为 `0.667/0.833/0.833/1.000`。
> 该诊断不调用 Embedding、Chroma、DeepSeek、Gradio 或 RAGAS，结果不能解释为答案正确率。

> 2026-08-28 Phase 2 初始 Hybrid/RRF 离线对比：`evaluation/benchmark_retrieval.py` 增加本地
> dense（Sentence-Transformers，`HF_HUB_OFFLINE=1`）和 BM25+dense 的 RRF 模式，并复用
> 单一模型实例；新增 RRF 去重、稳定排序和跨论文隔离测试。Hybrid 全局 top-1/3/5/10
> 当时尚未加入弱跨语言词法保护，参考片段覆盖代理为 `0.208/0.330/0.443/0.525`，Table N 命中率为
> `0.500/0.667/0.778/0.833`。仅证明离线实验可运行，未替换线上 app 默认 dense 检索，
> 未调用网络模型、未修改 ChromaDB。

> 2026-08-28 Phase 0/1 的代码、离线验收和迁移注意事项见
> [PHASE0_PHASE1_HANDOFF.md](PHASE0_PHASE1_HANDOFF.md)。本次未重建现有 ChromaDB、未运行 RAGAS、未调用外部模型。

> 2026-08-28 追加修复：明确 Table N/行/列的问题现在走通用的确定性 Markdown
> 单元格查找；缺失表号不再回退到其他表格；`.gitignore` 补充本地运行时文件规则。
> 附件 PDF 和现有 104 块 ChromaDB 均只读核验，未调用外部模型或重建数据库。

> 2026-08-28 Phase 2 多论文基准：用户提供了四篇仓库外的免费论文 PDF。新增
> `evaluation/benchmark/PAPER_AUDIT.md`，并在清单中登记 SciDQA（EMNLP 2024）、
> Scientific Table LLM（SDP 2024 Workshop）、MgNO（ICLR 2024）和 AlphaFold 3
>（Nature 2024）的文件名、SHA-256、来源、领域和版式标签；`cases.jsonl` 增加每篇
> 10–11 道独立问题，覆盖文本、表格、公式、图注、限制和复现性。校验器支持多个
> `--papers-dir`，已通过 5 篇/53 题的离线 SHA-256 校验。解析审计发现部分 PDF 的
> 表号/caption 未稳定进入 table metadata，AlphaFold 3 还有 3 个无文本页；这些是
> 后续通用解析改进的输入，不是已解决的检索证据。

> 2026-08-28 Phase 2 标注核对：对新增 42 道问题逐题对照本地 PDF 的文本层和渲染页面，
> 未发现核心数值或表号冲突。修订 `cases.jsonl`：为表格证据补充完整列名、单位和表注，
> 在 SDP Table 2 答案中保留括号内变化量，将 `caption` 明确为“题注”，并把 MgNO 公式
> 用例改为只陈述原文直接给出的初始化和残差公式。此次仅证明基准标注自洽，不证明
> 当前解析器、检索器或生成器已经正确。

---

## 第九次修改：2026-08-24 —— 接入 RAGAS 评估框架（evaluation/ + return_contexts）

### 背景

为量化 Sci-RAG 的检索与生成质量，引入 RAGAS 框架计算三个核心指标：
Context Relevance（上下文相关性）、Answer Faithfulness（答案忠实度）、
Answer Relevance（答案相关性）。评判 LLM 复用项目自带的 DeepSeek
客户端（deepseek-chat），AnswerRelevancy 所需 embedding 复用本地缓存的
BAAI/bge-small-zh-v1.5，均不新增外部 API 依赖。

### 修改内容

| 位置 | 改动 |
| --- | --- |
| `requirements.txt` | 新增 `ragas`、`datasets` 两行 |
| `app.py` `query_knowledge()` | 签名改为 `(message, history=None, return_contexts=True)`；`return_contexts=True` 时返回 `{"answer": 原始回答（不含参考来源页脚）, "contexts": 实际进入提示词的上下文列表（重排序/过滤后）}`，所有早退分支（空问题/空知识库/无检索结果）与异常分支同样返回 dict 形态，契约统一；`False` 时保持原字符串行为不变 |
| `app.py` 新增 `chat_respond()` | Gradio 包装：`query_knowledge(message, history, return_contexts=False)`，ChatInterface 的 `fn` 改接它（默认值变 True 后 UI 必须显式关闭，否则聊天会拿到 dict） |
| `evaluation/test_questions.json` | **新建**。11 题测试集（6 表格数值 + 3 方法 + 2 综合），全部基于 2602.08213v1.pdf 真实数据：Table 1/2/5/6 数值、GRPO、三阶段训练、4,855 样本 SFT 数据集、Pareto 自平衡机制；每题含 `question`/`ground_truth`/`contexts`（金标准上下文） |
| `evaluation/evaluate.py` | **新建**。加载测试集 → 逐题 `query_knowledge(return_contexts=True)` → RAGAS 三指标打分（每题取重排序后 top-10 上下文，`--max-contexts` 可调）→ 输出 `evaluation_report.json` / `evaluation_report.md`；支持 `--limit N` 冒烟测试 |

### 依赖兼容性踩坑（ragas 0.4.3 + langchain 1.x 生态，均已绕过）

1. **vertexai 模块缺失**：ragas 0.4.3 顶层导入 `langchain_community.chat_models.vertexai`，
   但 langchain-community 0.4.2（最新版）已移除该模块（集成迁至 langchain-classic）。
   evaluate.py 在 import ragas 之前向 `sys.modules` 注册占位模块（桩类不可实例化，
   评估只用 OpenAI 兼容接口，不受影响）。
2. **evaluate() 拒绝新指标**：`ragas.metrics.collections` 的新指标（BaseMetric 体系）
   过不了 evaluate() 的 `isinstance(m, Metric)` 校验（上游 0.4.3 内部不一致），
   改用 `ragas.metrics` 经典指标类，经 `evaluate(llm=..., embeddings=...)` 注入。
3. **旧式接口缺口**：经典 `ContextRelevance`（_nv_metrics）调用旧式
   `llm.agenerate_text()`，`llm_factory` 返回的 InstructorLLM 没有该方法 →
   在实例上补方法（内部走 DeepSeek 客户端 + `asyncio.to_thread`，返回
   `.generations[0][0].text`）；经典 `AnswerRelevancy` 调用 Langchain 风格
   `embed_query`/`embed_documents`，新版 HuggingFaceEmbeddings 只有 `embed_texts` →
   同样在实例上补两个方法。
4. **结果列名**：经典 ContextRelevance 的指标名为 `nv_context_relevance`
   （而非 context_relevance），报告列名映射已按此处理。
5. **openai 降级**：pip 安装 ragas 时 openai 3.3.1 → 1.109.1（ragas 依赖要求），
   app.py 的 `client.chat.completions.create` 用法在 1.x 下完全兼容，已验证。

### 验证（冒烟测试 --limit 2 全链路 ✅）

- 2 题查询均返回 dict 形态：`answer` 正确（0.2712 / 0.2060），contexts 56/55 个
- 三指标均产出有效分数：Context Relevance 1.0000、Faithfulness 0.5000、
  Answer Relevance 0.7103；报告 JSON/MD 正常生成
- 全量 11 题评估已运行，见 `evaluation/evaluation_report.json`

### 运行方式

```bash
./venv/bin/python evaluation/evaluate.py               # 全量（11 题，约 20-40 分钟）
./venv/bin/python evaluation/evaluate.py --limit 3     # 冒烟测试
./venv/bin/python evaluation/evaluate.py --max-contexts 5
```

### 已知行为（未修）

- DeepSeek 评判模型对 `AnswerRelevancy` 的 n=3 请求只返回 1 个生成
  （ragas 打印 "LLM returned 1 generations instead of requested 3"，自动降级不报错）；
- 答案中"根据参考片段 [X] 所示"的编号在 RAGAS 拿到的 raw contexts 中不存在
  （编号是 query_knowledge 组装提示词时加的），Faithfulness 判定可能因此偏低；
- ragas 0.4.3 的三处兼容补丁集中在 `evaluate.py` 头部，上游修复后可按注释
  切换回 `ragas.metrics.collections` 并删除补丁。

---

## 第八次修改：2026-08-24 —— query_knowledge 新增 Table N 二次过滤（caption 级）+ 检索提示

### 背景

`_rerank_table_first` 已保证命中时只保留匹配的表格块，但存在两个缺口：
（1）表格块若靠内容/headers 命中而 caption 不含 "Table N"（caption 为空或
非标准写法），重排序仍会放行；（2）"Table 1" 会被 `Table\s*1` 误匹配
"Table 12" 的 caption。本次在 `query_knowledge` 中、构建 context_parts 之前
增加一道 caption 级二次过滤，并在上下文开头提示用户只检索了指定表格的数据。

### 修改内容

| 位置 | 改动 |
| --- | --- |
| `query_knowledge()` 重排序之后 | 新增修改 8 块：`re.search(r'Table\s*(\d+)', message, re.IGNORECASE)` 提取表号；未指定表号则不过滤 |
| 过滤条件 | 保留所有 `type != 'table'` 的文本块 + `table_caption` 命中 `Table\s*N(?!\d)`（IGNORECASE）的表格块——`\s*` 同时覆盖 "Table 2"/"Table2" 两种写法，`(?!\d)` 防止 "Table 1" 误匹配 "Table 12" |
| 回退保护 | 过滤后若一个表格块都不剩，保持原 ordered 列表不变（回退到全部块，防止无答案） |
| 提示语 | 过滤生效时在上下文开头附加 `【检索提示】已根据您的要求只检索 Table N 的数据。`；与 rerank 的"未找到 Table N"提示互斥（matched 为空时二次过滤必然回退），不会同时出现两条矛盾提示 |

### 验证（6 个场景全部 ✅）

1. "Table 2 中 DrugR* 的整体优化得分是多少？" → 提示语存在，Table 1/3 块被排除，行级过滤只剩 DrugR* 行 ✅
2. 表格块靠内容命中 Table 2 但 caption 不含表号 → 被二次过滤剔除，正确表格块保留 ✅
3. 未指定表号（"表中 DrugR*…"）→ 不过滤、无提示语 ✅
4. "Table 5…"（无命中）→ 回退全部块，保留"未找到 Table 5"提示，不出现矛盾提示 ✅
5. 非表格提问 → 无提示、保持原序 ✅
6. "Table 1" 提问 vs caption "**Table 12**" → Table 12 块被剔除（(?!\d) 边界生效）✅

---

## 第七次修改：2026-08-24 —— 明确最终 order 构造：命中 Table N 时彻底排除其他表格块

### 背景

第五次修改已引入 `table_idx = matched`（命中时只保留命中的表格块），
但"其他表格块被排除"这一保证分散在两处代码里，且最终 order 构造的注释
只写"表格块置顶"，容易被误读为仍会混入全部表格块。本次把该保证显式化：
最终 order 由 `table_idx + other_idx` 唯一决定——命中时 table_idx 即 matched、
other_idx 只含非表格块，未被命中的 type="table" 块既不在 table_idx、也不在
other_idx 中，被彻底排除在上下文之外，避免大模型同时看到多张表格
（如 Table 1 与 Table 2）而取错数据。

### 修改内容

| 位置 | 改动 |
| --- | --- |
| `_rerank_table_first()` 的 `if matched:` 分支 | 注释明确：其他表格块（如 Table 1 的块）不得进入最终 order，否则大模型会同时看到多张表格而混淆 |
| `_rerank_table_first()` 末尾 order 构造 | `text_idx` 更名 `other_idx`；注释写明关键保证——matched 非空时未被命中的 type="table" 块被彻底排除；matched 为空时 table_idx 保持为全部表格块（原有回退行为） |
| 函数 docstring 规则 2 | 补充"其他 type="table" 块从最终 order 中彻底排除，非表格文本块不受影响" |

### 验证（4 个场景全部 ✅）

1. "Table 2 中 DrugR* 的整体优化得分是多少？" → order = [Table 2 块, 正文块…]，Table 1/3 的块不在 order 中；行级过滤后块内容只剩 DrugR* 行 ✅
2. "Table 5 的样本量是多少？"（无命中）→ 全部表格块置顶 + 提示语 ✅
3. "表中 DrugR* 的得分是多少？"（未指定表号）→ 全部表格块置顶、无提示语 ✅
4. "这篇论文讲了什么？" → 保持向量检索原序 ✅

---

## 第六次修改：2026-08-24 —— if matched 分支新增行级实体过滤，缩小表格上下文

### 背景

命中 Table N 后，整个表格块（含全部变体行）进入上下文，LLM 在多行相近的
数据里可能取错行。本次在检索侧配合：把与问题实体无关的表格行过滤掉，
让上下文只保留目标行（与用户自行新增的 Prompt 强制规则 4/5 及
n_results=60 配合，检索侧先精确、生成侧再约束）。

### 修改内容

| 位置 | 改动 |
| --- | --- |
| 新增常量 `_ENTITY_RE` | `[A-Za-z0-9_*+\-]+(?:[-\s][A-Za-z0-9_*+\-]+)*`：从中文问题中提取 ASCII 实体记号（如 "DrugR*"、"CO2 methanation"、"Ni-Fe"），中文被自然截断 |
| 新增函数 `_filter_table_rows_by_entity()` | 对表格块逐行查找（不区分大小写），只保留"表头行（GFM 含 \| --- \| 分隔行）+ 命中实体名的数据行"；无任何行命中时返回 None（调用方回退原块，宁可多给上下文也不误删） |
| `_rerank_table_first()` 的 `if matched:` 分支 | 提取实体（排除 "Table N" 表号本身与纯数字，剩余取最长者）→ 对每个命中的表格块做行级过滤 → **原地替换 `texts[i]` 内容**；无实体或未命中行时内容不变 |
| `query_knowledge()` | 无需改动——它在 `_rerank_table_first` 返回后才按 order 重建上下文，自然读到被替换后的内容 |

### 验证（8 个场景全部 ✅）

1. "Table 2 中 DrugR* 的整体优化得分是多少？" → 块内容过滤为 表头 + `| DrugR* | 0.874 |`，其他变体行（0.812 / Baseline）被移除 ✅
2. 实体（XYZ）不在表格中 → 回退原块，一行不删 ✅
3. 问题只有表号没有实体 → 不做行级过滤 ✅
4. 未指定表号（"表中 DrugR*…"）→ 不走 if matched 分支，不过滤 ✅
5. 表格里是小写 drugr*、问题是大写 DrugR* → 不区分大小写命中 ✅
6. 无 \| --- \| 的暴力提取块 → 首行作表头，命中行保留 ✅
7. 多词实体 "CO2 methanation" → 空格连接整体匹配 ✅
8. 非表格提问 → 原序且内容不变 ✅

### 已知行为（未修）

- 实体取"排除表号后最长的候选"，问题含多个实体（如 "A 与 B"）时只取最长的一个；
- 单字母实体（如 "N"）不区分大小写会命中很多行（几乎每个英文行都含 n），
  此时过滤近似无操作、等价于回退原块，无数据损失风险。

---

## 第五次修改：2026-08-23 —— Table N 匹配改用正则（IGNORECASE），修复加粗标题 "**Table 2**" 失配

### 问题

诊断发现：Table 2 的编号只存在于 metadata 的 `table_caption` 字段
（形如 "**Table 2** Out-of-Distribution..."——pymupdf4llm 把论文的加粗标题
转成了 Markdown 加粗标记），表格内容的 Markdown 源码（| 行）中
不包含 "Table 2"，导致匹配逻辑失配、系统误报"未找到精确匹配"。

### 修改内容（仅 `_rerank_table_first()` 一个函数）

- 匹配 "Table N" 从子串比对改为正则
  `re.search(rf'Table\s*{N}', 文本, re.IGNORECASE)`：
  `\s*` 兼容 "**Table 2**" 加粗标记、全角/多空格等格式差异，
  IGNORECASE 替代手动小写化；
- 命中检查顺序：① metadata 的 `table_caption`（表格标题只存在这里，
  块内容仅含 | 行、几乎不含 "Table N" 字样，是最主要命中来源）→
  ② 块内容（Markdown 表格源码）→ ③ metadata 的 `headers`
  （保守第三处，延续第四次修改）；任一命中即视为命中；
- 无命中时不丢弃表格块：保留全部 `type="table"` 的块置顶，并在上下文开头
  提示 `未找到 Table N 的精确匹配，以下是知识库中所有表格数据供参考。`
  （复用第四次修改的 note 机制，`query_knowledge()` 无需改动）；
- 未指定 Table 编号（table_num 为 None）：所有表格块置顶，行为不变；
- 函数 docstring 改为 raw string（`r"""`），消除 `\s` 无效转义警告。

### 验证（7 个场景全部 ✅）

1. "Table 2" 只存在于 caption（"**Table 2** ..." 加粗形态）→ 正则命中，仅保留 Table 2 块 ✅
2. caption 为 "Table 2"（不换行空格）→ `\s*` 兼容命中 ✅
3. "Table 3" 只存在于块内容 → 命中 ✅
4. "Table 2" 只存在于 headers → 命中（保守第三处）✅
5. 问 Table 5 无任何命中 → 全部表格块保留置顶 + 提示语 ✅
6. 未指定表号（"样本量是多少？"）→ 全部表格置顶、无提示语 ✅
7. 非数值提问 → 保持向量检索原序、无提示语 ✅

### 已知行为（未修）

- 正则 `Table\s*N` 是前缀匹配："Table 20" 的标题也会被 "Table 2" 的提问命中。
  若论文表格编号达到两位数，可在数字后加 `\b` 收紧（当前论文表格少，暂不处理）。

---

## 第四次修改：2026-08-23 —— Table N 匹配改为三处检查 + 无命中时保留全部表格块并提示

### 问题

问 "Table 2 中 DrugR* 的整体优化得分是多少？" 时，系统明明检索到了 Table 2 的
表格块（已在 Chroma 中），但重排序函数匹配 "Table 2" 失败，把该块丢弃了，
导致回答"未提供"。

### 修改内容

| 函数 | 改动 |
| --- | --- |
| `_rerank_table_first()` | ① 匹配 "Table N" 时**三处同时检查**（任一命中即视为命中）：metadata 的 `table_caption` 字段、块内容（Markdown 表格源码）、metadata 的 `headers` 字段；② 指定了 Table N 但**没有任何块命中时，不再丢弃表格块**——保留全部 `type="table"` 的块置顶，并返回提示语 `未找到 Table N 的精确匹配，以下是知识库中所有表格数据供参考。`；③ 未指定 Table 编号（table_num 为 None）时所有表格块置顶（原有行为不变）。返回值从 `order` 改为 `(order, note)` |
| `query_knowledge()` | 解包 `(order, note)`；note 非空时在上下文开头附加 `【检索提示】未找到 Table N 的精确匹配，以下是知识库中所有表格数据供参考。` |

### 验证（6 个场景全部 ✅）

1. 问 Table 2 但 caption/内容/headers 均无 "Table 2" 字样 → 全部表格块置顶不丢弃 + 返回提示语（DrugR* 表格块仍在上下文中，可正常回答）
2. `table_caption` 精确命中 → 仅保留 Table 2 块，无提示语
3. `headers` 字段命中 → 命中生效
4. 块内容（documents 源码）命中 → 命中生效
5. 未指定表号（"表中数值是多少？"）→ 全部表格置顶，无提示语
6. 非数值提问 → 保持原序，无提示语

---

## 第三次修改：2026-08-23 —— 修复 Table N 筛选未检查 table_caption 导致匹配失败

### 问题

`_rerank_table_first()` 匹配 "Table N" 时只检查了块内容（documents），
但表格抽块时标题只存进了 metadata 的 `table_caption` 字段（块内容只有 `|` 行），
导致"明确问了 Table 2"的筛选总是匹配失败，退化为保留全部表格块。

### 修改内容（仅 `_rerank_table_first()` 一个函数）

- 匹配 "Table N" 时**同时检查** metadata 的 `table_caption` 字段和块内容
  （两者均小写化比对，任一处命中即视为命中）；
- 指定了 Table N 但没有任何命中时，**保留全部表格块**（不丢弃），避免上下文为空；
- 重排序顺序不变：表格块置顶、非表格块随后（保持原相对顺序）；
- 同步更新了函数 docstring 中规则 2 / 规则 3 的说明。

### 验证（3 个场景全部 ✅）

1. 标题只存在于 `table_caption`、块内容无 "Table 2" 字样 → 正确命中并置顶，
   其他表格块（Table 1）被忽略，顺序 `[表格, 正文A, 正文B]`
2. 指定 "Table 5" 但无任何命中 → 保留全部表格块且置顶，无块被丢弃
3. 非数值/表格类提问 → 保持向量检索原序，不干预

---

## 第二次修改：2026-08-23 —— app.py 直接改造，修复"表格增强召回"完全失灵

### 问题背景

系统能回答"用了哪种强化学习算法？"这类文本抽取问题，但问
"Table 2 中 DrugR* 的整体优化得分是多少？"时完全检索不到表格块。

根因：app.py 此前仍在用 PyPDFLoader + 一刀切 512 字符分块，
表格被切碎成普通文本、没有任何 `type="table"` 标记、召回数只有 3，
重排序也不存在，所以表格类提问必然失败。

### 修改的函数与内容

| 函数 | 改动 |
| --- | --- |
| 顶部导入区 | 新增 `re` / `tempfile` / `shutil` / `Document` / `MarkdownHeaderTextSplitter`；**移除 PyPDFLoader**（乱码根源），PDF 改用 pymupdf4llm（函数内延迟导入） |
| `extract_tables()` | **新建**。修改 2：方案 A 用正则解析标准 GFM 表格（`\| --- \|` 分隔行 → 表头 + 数据行整表抽出）；方案 B 若无任何 GFM 表格，用 `\bTable\s*\d+.*?(?=\n\n|\Z)` 暴力提取所有 "Table N" 连续段落。每个表格块 metadata 强制带 `{"type": "table", "source": filename, "table_caption": ...}`。开关 `_ALWAYS_BRUTE_FORCE` 可让两种方案同时生效（混合情况） |
| `_split_to_chunks()` | **新建**（修改 3）。最终分块循环：`type="table"` 的块跳过 RecursiveCharacterTextSplitter，整表作为一个 chunk 直接入库，长表格不再被切碎；非表格文本保持 MarkdownHeaderTextSplitter 优先（#/##/###/#### 标题切分），超 1024 字符回退 RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128, separators=["\n\n", "\n", "。", "；"]) |
| `load_and_split_document()` | **改写**。修改 1：pymupdf4llm 转完 Markdown 后、切分之前，打印前 3000 字符 + 是否含 GFM 分隔行的统计到终端；PDF→Markdown（表格保留 \| --- \|、公式转 LaTeX 行内格式）+ 图片占位符提取 `[Image: xxx.png]` + 调用 extract_tables / _split_to_chunks |
| `add_document_to_db()` | **微调**。不再覆盖 metadata，完整保留每个块的 `source` / `headers` / `type` / `table_caption` 存入 Chroma（重排序依赖 `type` 字段） |
| `query_knowledge()` | **改写**（原 `respond` 更名，ChatInterface 接线同步更新）。修改 4：n_results 3→6；新增 `_rerank_table_first()` 重排序——提问含 `["Table", "表", "数值", "多少", "样本量", "比率", "n="]` 关键词时把 `type="table"` 的块强行置顶（不看相似度得分）；若明确问了 "Table N"（正则提取表号），只保留包含 "Table N" 字样的表格块、忽略其他表格块（无命中时回退不筛选，避免上下文为空）；上下文片段带编号【片段 X】+[表格] 标注 |
| `SCIENTIFIC_SYSTEM_PROMPT` | **新增常量**。科学严谨模式三规则：①数值原样引用并指明"根据参考片段 [X] 所示"；②趋势判断必须有明确对比依据，否则回复"资料未提供该趋势的明确依据，无法推测。"；③实验步骤按"第一、第二、第三"时间顺序重组。 |

### 本轮验证结果（已实测通过）

用临时脚本（运行后已删除）做了 5 项测试，全部 ✅：

1. GFM 表格提取：正确抽出 1 个表格块，caption 为 "Table 2 Overall optimization scores of DrugR* variants."
2. 备用暴力提取：无 GFM 分隔行时，成功提取 "Table N" 段落并打 type="table"
3. 长表格防切碎：4000+ 字符的表格保持为 1 个完整块；超长正文正常回退切分
4. 重排序："Table 2 中 DrugR*…" → Table 2 块置顶、Table 1 块被忽略；非数值提问不干预
5. 端到端（真实 bge-small-zh embedding + 临时 ChromaDB）："Table 2 中 DrugR* 的整体优化得分是多少？" → 表格块置顶，`| DrugR* | 0.874 |` 在上下文中

### 运行前必读

- **安装依赖**：`pip install pymupdf4llm pillow`（venv 中尚未安装，PDF 上传会提示）
- **清空旧库**：chroma_db 里旧的 512 一刀切块没有 type 标记，会稀释检索效果，建议删除 `chroma_db` 目录后重新上传论文
- **调试打印**：上传 PDF 时终端会打印前 3000 字符 + GFM 分隔行统计，用于确认 pymupdf4llm 是否把表格转成了 `| --- |` 格式；确认无误后可删除 `load_and_split_document` 中的 `[DEBUG]` 打印块
- **已知行为**：表格在"正文章节块"中还会以一份 HTML 形态存在（MarkdownHeaderTextSplitter 内部转换所致），检索与答案以独立的 `type="table"` 块为准，不影响正确性

---

## 第一次修改：2026-08-23 —— 新建 sci_rag_core.py，完成 4 项改造任务（模块级交付）

### 背景

原 app.py 使用 PyPDFLoader（科学论文乱码）+ RecursiveCharacterTextSplitter(512)
（上下文割裂）+ n_results=3 + 通用 Prompt（易幻觉），无法胜任 Nature 论文。

### 交付内容

| 文件 | 内容 |
| --- | --- |
| `sci_rag_core.py` | **新建**。任务 1：`load_documents()` 用 pymupdf4llm 把 PDF 转 Markdown（保留 \| --- \| 表格、LaTeX 行内公式），图片提取为 PNG 并打 `[Image: xxx.png]` 占位符；任务 2：`split_documents()` MarkdownHeaderTextSplitter 标题级切分 + 超长块回退 RecursiveCharacterTextSplitter(1024/128)；任务 3：`_extract_tables_from_markdown()` 表格独立成块（type="table"）+ `rerank_with_table_priority()` 关键词重排序；任务 4：`SCIENTIFIC_SYSTEM_PROMPT` 防幻觉三规则。另含 `retrieve_for_question()` 统一检索入口（n_results=6） |
| `test_sci_rag.py` | **新建**。验证脚本：标题切分、表格提取、重排序、端到端检索（真实 embedding + 临时 ChromaDB），全部通过 |

### 当时遗留的问题

- 上次只交付了独立模块 + app.py 接线说明，**未直接修改 app.py**（本次修改完成接线并强化了表格召回）
- 旧版切分产物残留在 chroma_db 中，需要清空重建

### 当前文件状态

- `app.py`：**本次改造后的唯一权威实现**（自包含全部逻辑）
- `sci_rag_core.py`：第一次修改的模块实现，现作为参考保留；确认 app.py 工作正常后可删除（`test_sci_rag.py` 依赖它，如需保留回归测试则一并保留）
