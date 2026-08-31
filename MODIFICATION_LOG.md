# Sci-RAG 修改记录

本文件是项目唯一的阶段修改记录。以后只在这里追加简短的“改了什么、如何验证、仍未证明什么”，
不再新增 `PHASE*.md` 交接文件。运行命令见 `README.md`，基准标注细节见
`evaluation/benchmark/`。

## 证据口径

- “已实现”表示代码路径存在，不代表运行或结果正确。
- “测试通过”仅表示对应离线/定向用例通过。
- 检索 required-fact 覆盖、答案词面覆盖、人工语义复核和 RAGAS 是不同证据，不能互相替代。
- 临时数据库与评估输出保存在 `/tmp` 或 `/private/tmp`，不提交到仓库。

## 当前状态（2026-08-31）

- 支持 PDF/TXT/DOCX、canonical Markdown 表格、Chroma dense 检索、DeepSeek 生成和 Gradio。
- `app.py` 可安全导入；模型、API、数据库和 UI 均延迟初始化。
- 默认 dense；Hybrid、固定本地 reranker、文档路由、query decomposition、parent/window、
  Figure 坐标证据和答案核对均可独立控制。
- 显式表号/行/列使用确定性单元格路径；显式公式/算法问题默认使用窄同源证据门控；显式限制问题
  使用有界同源证据门控。
- 五论文基准为 5 篇、53 题；外部 PDF 只记录 SHA-256，不进入 Git。
- 最终 53 题×2 轮生成 106/106 成功，配置、context IDs 和 metadata 为 53/53 稳定；人工核验
  安全别名后两轮词面事实覆盖均为 53/53 full，答案文本完全一致仅 18/53。
- 以上不是语义正确率。较早一轮独立语义复核为 43 correct / 7 partial / 3 incorrect，尚无
  对最终批次的独立双人复核。
- 当前无 OCR/VLM、通用工具执行器或 Graph-RAG。

## 阶段记录

### 2026-08-23 至 2026-08-24：初始 Sci-RAG 改造

- 从单体应用中提取 `sci_rag_core.py`，建立表格抽取、切分和检索前处理基础。
- 修复表格 caption/表号识别、加粗标题、行级实体过滤和显式 Table N 约束，避免跨表取值。
- 接入 `return_contexts` 与 RAGAS 评估脚本。仓库报告实际为 Context Relevance
  `0.8864 (11/11)`、Faithfulness `0.5833 (8/11)`、Answer Relevancy `0.8543 (11/11)`；
  三个指标均不使用 ground truth，不能解释为答案正确率。

### Phase 0 / 1：安全基线与可复现入口

- 移除全局 SSL 关闭、硬编码镜像和导入时初始化；补齐 `.env.example`、依赖与离线 setup 检查。
- PDF/TXT/DOCX 统一走 page-aware chunks，表格与正文分离，图片不持久化。
- 增加隔离数据库 UI 启动脚本和 Table 1/Table 2 回归题，保护原 `chroma_db`。

### Phase 2：多论文基准、Hybrid 与 reranker

- 建立 5 篇论文、53 题的 manifest/cases/gold-context 基准，并逐题核对 42 道新增题。
- 增加 PDF caption 前后、分组表头、单位列、断词和布局表误识别回归。
- 建立 BM25、dense、Hybrid/RRF 离线对照和 required-fact/provenance 指标。
- 增加默认关闭的 Hybrid runtime 与固定 revision 的本地 `bge-reranker-base`。
- 初始 Hybrid+CE+RRF @10 fact macro/micro/full 为 `0.785/0.776/0.698`；CPU 重排约
  2.73 秒/题、峰值约 2.20 GB。这是检索代理，不是答案准确率。

### Phase 3：答案完整性

- 增加无模型 `answer_audit.py`，按人工 required facts 计算 full/partial/zero 与 macro/micro。
- 增加 Dense 与 Hybrid 回答 A/B 契约；确认单论文 11 题通过不能证明多论文泛化。

### Phase 4：检索失败分型

- 将失败分为 gold 表面差异、候选池未召回、cross-encoder 降序和最终 RRF 稀释。
- 增加逐事实 provenance、保守文档路由和 candidate-k/融合权重对照。
- 拒绝了无界同节扩展和盲目增大候选池：收益不足且会增加污染、延迟和内存。

### Phase 5：生成评估链路

- 统一多论文 JSONL 加载、答案 A/B、人工复核模板和 RAGAS preflight。
- 首次五论文生成中，Dense/Hybrid 词面 full 为 `0.4528/0.4906`；内部语义复核均仅
  `22/53 correct`，暴露表格、单位、复合问题和错误拒答。
- 修复通用表格单元格定位：合并首列、任意实体列、`<br>` 变体、组标记和 caption 单位。
- 增加默认关闭的文档路由和有界 query decomposition，并通过隔离网页回归。

### Phase 6：检索收敛与证据可追踪

- 修复无真实拆分却重复查询的问题；结构化表格 guard 后 @10 达到
  fact macro/micro/full `0.881/0.871/0.811`。
- parent/window 在不占用额外 top-k 槽位下将其提高到 `0.936/0.932/0.887`，但增加约
  6.2 万字符，因此保持可选。
- 增加 born-digital Figure 坐标文字证据；不读取像素、不做 OCR，总块数由 479 增至 502。
- 增加重复生成、证据校验、受控 retry prompt 和 trace provenance 工具。自动 retry 未启用，
  因为缺失证据不能靠重复生成修复。
- 修正多-k 评估污染；@10/@50 full 为 `47/53` 与 `52/53`，未把 50 个上下文直接送入生成。
- 增加默认关闭的同源公式证据候选，作为后续窄门控基础。

### Phase 7：方向门控与当前版本生成

- 审计多模态、工具调用和 Graph-RAG：当前基准无足够 image-only、多跳或真实运算失败，三个方向
  均暂缓；已有确定性表格查找不扩展为通用 agent。
- 一轮 53 题生成全部成功，词面 fact macro/micro=`0.9261/0.8973`、full=`46/53`；
  `context_k=50` 未稳定改善回答，因此默认仍为 10。
- 表格 caption 单位保护确保共享 `×10⁻²` 等比例信息进入答案与 trace。

### Phase 8：整合回归

- 合并当时已通过的检索、表格、Figure 和评估工具；确认默认开关不因实验代码变化。
- 建立编译、完整单测、benchmark 校验和 `git diff --check` 的统一交接门槛。

### Phase 9：语义复核

- 对当时 53 题 Hybrid+reranker 批次逐题复核：`43 correct / 7 partial / 3 incorrect`。
- 证明词面覆盖会高估正确性；主要问题为表格归属矛盾、错误拒答和方法细节遗漏。

### Phase 10：定向生成修复

- 提示词禁止“先给出表格值、随后又称资料未提供”，并要求拒答前检查全部方法/附录片段。
- 表格上下文显示表号 metadata；公式/多重网格算法候选以实验开关接入。
- 定向修复 SciDQA 表格矛盾和 MGNO 初始化/循环证据，不据此宣称全量提升。

### Phase 11：方法段证据

- 让唯一文档路由下的复合问题可使用有界同节扩展，仍限制同源、最多六块且不处理歧义查询。
- 修复 SciDQA 四种实验配置的截断拒答；定向单题通过，不外推为 53 题正确率。

### Phase 12：首次完整生成门禁

- 隔离 502 块数据库上完成 53 题×2 轮，106/106 成功，两轮均 `47/53` 词面 full。
- context/metadata 稳定 `50/53`，答案文本稳定 `19/53`；确认生成波动仍需单独审计。

### Phase 13.1：拒答风险

- 答案审计新增 `answer_refusal_detected`、`refused_required_facts` 和 `answer_risk_flags`。
- 风险信号不改变 full/partial/zero，也不自动重写或重试。

### Phase 13.2：公式/算法 A/B

- `mgno-03/04` 两轮均 full；全量两轮均为 `48/53` full，目标问题有收益。
- 全局公式开关仍不默认启用，因为无关题仍受生成波动影响。

### Phase 13.3：公式意图自动门控

- 增加默认开启的 `SCI_RAG_FORMULA_EVIDENCE_AUTO`，仅对明确公式、PDE 或多重网格算法意图
  复用有界同源候选；手动全局开关仍默认关闭。
- 严格 53×2 运行达到 `51/53` 与 `52/53` 词面 full，context/metadata 稳定 53/53。

### Phase 14：剩余检索失败分型

- 修复方法标题平分导致正确 RL 小节被截断，并安全续接无 header 的同源连续正文。
- 增加 PDE 区域/边界条件意图；恢复 DrugR、MGNO 遗漏证据，不跨表格、图形或来源。

### Phase 15：无标题续文的答案完整性

- 为章节续文增加派生 `section_context`，不覆盖原 metadata，让证据清单能检查续文工具名、阈值
  和数字。
- `drugr-09` 两次定向回答均包含 `4,855`、`DeepSeek-R1`、`ADMETLab` 和 `0.6`。

### Phase 16：限制问题证据

- 增加通用限制/失败意图与同源证据门控，区分机制性限制和示例现象，并避免把多重网格
  restriction 误判为模型局限。
- 五论文离线 retrieval 保持 fact macro/micro=`0.948/0.939`、full=`48/53`、路由 `39/39`；
  AlphaFold 限制定向回答两轮稳定通过。

### Phase 17：最终完整生成门禁

- 最终源码完成 53 题×2 轮，106/106 API 调用成功，provenance 完整，配置/context/metadata
  稳定 53/53。
- repeat 1 初始 `53/53` full，repeat 2 初始 `51/53`；核对 PDF 后仅增加等价词面 alias
  （`3 \\times 3`、中文“图和方程/更多样化”），复审两轮均为 `53/53` full。模型输出未改。
- evidence-only 校验为 58 `ok`、2 `review`，两项均为已知 `scidqa-08` 风险提示。
- 结论只支持当前固定基准的生成回归，不支持 RAGAS、跨领域泛化或生产可靠性声明。

### 2026-08-31：文档与重复代码整理

- 将 31 份、约 3389 行 `PHASE*.md` 合并为本文件；README 只保留当前使用和验收说明。
- 复用核心 `file_sha256()`，删除应用与 benchmark loader 中重复的文件哈希实现，并移除两个
  无调用方的旧 helper。
- 保留具有不同输入/错误契约的 JSONL 读取器和所有安全/验证逻辑，未进行激进重构。
