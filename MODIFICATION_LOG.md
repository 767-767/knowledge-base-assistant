# 个人知识库助手修改记录

本文件是项目唯一的阶段修改记录。以后只在这里追加简短的“改了什么、如何验证、仍未证明什么”，
不再新增 `PHASE*.md` 交接文件。运行命令见 `README.md`，基准标注细节见
`evaluation/benchmark/`。

## 证据口径

- “已实现”表示代码路径存在，不代表运行或结果正确。
- “测试通过”仅表示对应离线/定向用例通过。
- 检索 required-fact 覆盖、答案词面覆盖、人工语义复核和 RAGAS 是不同证据，不能互相替代。
- 临时数据库与评估输出保存在 `/tmp` 或 `/private/tmp`，不提交到仓库。

## 当前状态（2026-09-02）

- 支持 PDF/TXT/DOCX、canonical Markdown 表格、Chroma dense 检索、DeepSeek 生成和 Gradio。
- `app.py` 可安全导入；模型、API、数据库和 UI 均延迟初始化。
- 默认 dense；Hybrid、固定本地 reranker、文档路由、query decomposition、parent/window、
  Figure 坐标证据和答案核对均可独立控制。
- 显式表号/行/列使用确定性单元格路径；显式公式/算法问题默认使用窄同源证据门控；显式限制问题
  使用有界同源证据门控。
- 默认五论文基准为 5 篇、53 题；另有默认关闭的六论文挑战集（101 题）；外部 PDF 只记录 SHA-256，不进入 Git。
- 最终 53 题×2 轮生成 106/106 成功，配置、context IDs 和 metadata 为 53/53 稳定；人工核验
  安全别名后两轮词面事实覆盖均为 53/53 full，答案文本完全一致仅 18/53。
- 以上不是语义正确率。较早一轮独立语义复核为 43 correct / 7 partial / 3 incorrect，尚无
  对最终批次的独立双人复核。
- 当前无 OCR/图片向量索引、通用工具执行器或 Graph-RAG；VLM 仅有默认关闭的 PDF Figure opt-in 路径。

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

### 2026-08-31：公式证据隔离

- 将 PDF 原始文字层中 Markdown 漏掉的等式保存为独立 `formula` 块；普通 dense、BM25、Hybrid、
  路由和 parent-window 均排除它，只在明确公式问题的同源门控中补充。
- 五篇论文普通块逐项未变；53 题候选及 48 道非公式题的有效上下文 A/B 均为 `changed=[]`。
  完整本地检索回归 @10 为 macro/micro=`0.954/0.946`、full=`48/53`、路由 `39/39`。
- 隔离库中两道 MgNO 公式题经 DeepSeek 端到端复测，均返回 `A * u = f` / `3 × 3` 或正确的
  初始化与残差更新。未重跑 53 题完整生成，仍不构成新的语义正确率或泛化结论。

### 2026-09-01：公式隔离源码完整复测

- 使用全新仓库外数据库导入五篇论文，共 577 块（普通检索语料 502、独立公式块 75）；SQLite
  完整性检查通过，未读取或修改日常数据库。
- 当前源码完成 53 题×2 轮，106/106 次 DeepSeek 调用成功，provenance、源码指纹和运行配置一致；
  top-1/3/5 上下文稳定 `53/53`，完整 top-10 稳定 `51/53`，两处差异仅在低位候选。
- 两轮词面 full 为 `50/53` 和 `52/53`；逐题语义复核均为 `52 correct / 1 partial`，唯一 partial
  是目标事实正确但附加方向描述混淆的 `mgno-04`。答案文本完全一致 `18/53`。
- `scidqa-05/06` 与 `table-llm-07` 两轮均出现无关补充证据；记录为后续通用引用降噪问题，
  未在本次公式隔离改动中加入基准特异规则。

### 2026-09-01：复合数字题引用降噪

- 非流程型数字题不再因相邻段落的实体名触发“补充原文核对项”；应用层只对明确的管道、流程、步骤
  或构建问题启用补充，保留流程题的工具名/阈值补全能力。
- 新增回归测试后共 170 项通过；`scidqa-05`、`scidqa-06`、`table-llm-07` 定向端到端复测 3/3
  成功且均无无关补充。未修改检索排序、数据库结构或基准特异别名。

### 2026-09-01：网页集成与金标准审计

- 修复 Gradio 自定义问答输入未显示提交按钮的问题；隔离网页验证表格题和流程题均能提交并返回答案，
  原数据库未写入。
- 新增离线 `ground_truth_audit.py` 及单测，分开报告金标准事实覆盖、金标准上下文召回、文本完全一致和可选
  人工语义判断；这些指标仍不等同于自动答案正确率。
- 金标准上下文召回改用有界 token overlap 兼容更大的邻接块和 Markdown 间距变化，避免把格式差异误报为未召回。
- 当前 53 题最新人工复核为 `52 correct / 1 partial`（`mgno-04` 的额外方向描述有矛盾）；5 道
  figure-evidence 题均可由文字/坐标证据回答，未观察到 image-only 失败；基准不含真实工具运算或跨文档多跳失败。
- 因此暂不增加多模态、通用工具执行器或 Graph-RAG，继续沿用“先积累达到门槛的失败样例再扩展子系统”的策略。

### 2026-09-01：可选第六篇论文扩展

- 增加 `manifest_expanded.json` 的 `base_manifest` 继承能力，保留默认五论文/53 题基线不变。
- 纳入桌面已有的开放获取 Findings of EACL 2026 THINKNOTE（13 道逐题核对用例），扩展清单为
  6 篇论文、66 题；文件 SHA-256 与 required facts 离线校验通过。
- 未下载新 PDF、未调用外部模型；扩展题尚未进行完整生成、RAGAS 或独立语义复核。

### 2026-09-01：扩展清单离线检索复测

- 六篇/66 题的 Hybrid + 路由 + 查询分解 + 结构化证据 + parent-window 在 @50 达到
  fact macro/micro=`0.968/0.973`、full=`62/66`；路由 `50/50` 正确，仍有 4 题为已知
  的公式/示例或跨段落缺口，未据此接入网页默认。
- 修复 born-digital 图形文字层把相邻两位小数标签拼接（如 `67.4068.80`）的问题，并增加
  回归测试；Figure 3 的图形数值恢复。固定五论文默认路径未改变。

### 2026-09-01：并排图题注解析修复

- 空间图证据不再用同页任意上一题注的底部截断候选，只对水平方向重叠的题注应用边界，
  从而保留左右并排的多个 Figure（如 Figure 2/3）。
- 新增并排题注回归测试；全套离线单测为 173 项通过。扩展清单 @50 事实覆盖仍为
  macro/micro=`0.968/0.973`、full=`62/66`，未将图块加入默认检索。

### 2026-09-01：定向缺口挑战集

- 新增默认关闭的 `manifest_challenge.json`，继承 6 篇论文并追加 10 道 image-only、20 道
  computation、5 道 cross-document 题；原 53/66 题清单不变。
- 计算题单独保存 operation/expected result，避免把推导答案混入检索 gold context；跨文档题
  支持 `additional_document_ids`，只有全部目标来源命中才计为目标文档命中。
- PDF 图像人工核对后，10 道 image-only 的文字层事实覆盖为 `0/10`；Hybrid @10/@50 的
  cross-document 完整覆盖为 `4/5`、`5/5`，computation 输入事实为 `18/20`（@50）。
  这些结果仅用于缺口采样，尚未运行完整生成、工具执行、VLM 或 Graph-RAG。

### 2026-09-01：派生数值问题路由修复与稳定复测

- 新增通用派生数值意图判断；明确求差、合计、平均、比例、倍数和相对变化的问题跳过结构化
  单元格快捷回答，保留完整表格进入生成上下文。普通表格列查询仍走确定性路径；提示词仅对
  明确计算请求允许使用参考片段原始操作数并按题目精度计算。
- 修正 `calc-drugr-04` 的歧义为 `3,863 - 1,117 = 2,746`，并要求 `calc-scidqa-04` 保留两位小数。
- 隔离 577 块数据库上 20 道计算题×2 轮共 40 次调用全部成功；每轮 18/20 正确、2/20 因多表题
  未召回第二张表而缺少操作数，实际操作数完整 `36/40`。未发现稳定纯算术/生成失败；上述两题
  同时记为 routing/row-selection 缺口，不能作为计算器门槛证据。
- 因稳定纯计算失败为 `0 < 5`，本步不增加 calculator/tool executor、VLM 或 Graph-RAG；下一方向
  转 image-only 挑战。

### 2026-09-01：最小 image-only vision 实验

- 新增仓库外运行脚本 `evaluation/vision_experiment.py`：复用 challenge loader 和 Figure 解析，
  从目标题注上方最多 420pt 的整页宽度区域生成内存 PNG data URL，直接调用
  `deepseek-v4-flash-vision-exp`；不接入 UI、运行时、OCR、图片持久化或本地 VLM。
- 新增 3 项最小测试；全套 181 项单测、三个 manifest 文件校验及 `git diff --check` 通过。
  10 张临时 crop 均人工核验包含对应 Figure 且可读。
- 真实调用 20/20 成功（10 题×2 轮）。逐题语义复核：每轮 `7 correct / 0 partial / 3 incorrect`；
  两轮均未达到预设 `>=8/10 correct` 门槛。失败集中在复杂结构图/矩阵图的空答案：
  `img-drugr-02` 两轮、`img-af3-01` 两轮，以及 `img-scidqa-01` 或 `img-scidqa-02` 各一轮。
- 按停止条件仅记录失败类型，不实施网页接入、OCR、图片存储、本地 VLM、Graph-RAG 或 tool executor。

### 2026-09-01：限定 full+detail 图像输入 A/B

- evaluation/vision_experiment.py 增加可选 full+detail 模式：同一 user message 发送完整图裁剪和一个通用下半部居中 detail crop；默认 full 单图模式保持不变。请求不包含
  ground_truth、contexts 或 required_facts，未接入网页、ChromaDB 或运行时。
- 在同一 10 道 image-only challenge 上重复两轮，20 个有效调用均使用已授权 DeepSeek；
  full+detail 人工复核为 9/10、8/10，相同条件下已有 full 基线为 7/10、7/10。
  因此达到本实验每轮至少 8 题正确的门槛，但只保留为 opt-in 实验；不据此宣称通用视觉
  能力或直接改网页默认行为。失败为两轮 SciDQA 上下半部判断、一次 DrugR 复杂结构图空答。

### 2026-09-01：opt-in PDF Figure 视觉路径

- `RuntimeConfig` 增加默认关闭的 `SCI_RAG_VISION_ENABLED` 和视觉模型名；开启时，PDF 按 SHA-256 保存到
  `<SCI_RAG_DB_PATH>/source_pdfs/`，非 PDF 和默认关闭状态不复制。
- `query_knowledge` 仅在视觉开关、文档路由、明确 Figure 问题、唯一来源和 hash PDF 均满足时发送 full+detail
  图片；其他情况保持文本 RAG。视觉异常或旧 DB 缺 PDF 时回退文本路径并保留简短提示。
- 新增运行时配置、PDF 持久化与视觉触发回归测试；全套离线单测为 188 项通过。未调用真实 API。

### 2026-09-01：跨页 Figure 题注定位修复

- Figure 页定位优先选择题注上方存在图像块的候选，避免把正文中的“Fig. N”引用误当题注；修复双栏论文中
  Figure 3 正文引用与下一页真实图注分离时的错误裁剪。
- 增加跨页题注回归测试；Figure 3 重新定位到第 5 页。网页视觉调用仍存在外部模型偶发空响应，未据此宣称
  image-only 网页挑战通过。

### 2026-09-02：视觉空响应重试与修正后复测

- 视觉调用对空内容增加一次有界重试；运行时和离线视觉实验共用同一实现，不改变默认关闭和文本回退策略。
- 修正后的 10 道 image-only 题×2 轮共 20 次调用均返回非空答案；人工复核第一轮 `10/10 correct`、第二轮
  `9/10 correct + 1 partial`，唯一 partial 为 AF3 矩阵答案省略对称的 D-C 表述。10/10 页码与基准来源一致。
- 该结果只证明当前六论文挑战集上的 opt-in 路径，不改变网页默认配置，也不代表通用视觉能力。

### 2026-09-02：多表计算操作数召回修复与生成复测

- `matching_table_indices` 现识别问题中的全部显式 `Table N`，派生数值题保留所有命名表，避免只保留第一张表导致第二个操作数丢失。
- 新增多表匹配、完整操作数传入生成层的回归测试；全套离线单测 `194` 项通过，语法检查与 `git diff --check` 通过。
- 六论文挑战集离线检索中，`calc-table-02`/`calc-table-04` 在 @10、@50 均达到两项事实全覆盖；计算题 @50 为 `20/20` 完整覆盖。
- 真实生成烟测各重复两次：`calc-table-02` 均计算为 `0.14`，`calc-table-04` 均计算为 `0.07`，且两次均返回 Table 1 与 Table 2 上下文。该证据仅覆盖这两道修复目标题，不等同于全量答案正确率。

### 2026-09-02：跨论文挑战题事实校正

- 复核原 PDF 后修正 `cross-04`：MgNO Darcy rough 训练集为 `1,280`（不是 `1,000`），SciDQA 人工审阅候选实例为 `7,000`；期望计算结果更新为约 `5.5` 倍。
- 原挑战题的事实错误会把检索/生成拒绝真实矛盾误判为失败，已同步更新问题、gold facts、contexts 和 calculation 元数据；其他四道 cross-document 题不变。

### 2026-09-02：跨论文按来源补证据与真实生成复测

- 修正来源平衡排序：多个路由来源的首选证据现在一次性构造上下文前缀，避免后处理来源把先前来源挤出 `context_k`。
- 对显式多来源问题增加有界的来源内 lexical、同节和大数字证据补全；保留默认 dense 行为，补全仅在文档路由与查询分解同时开启且检测到多个来源时生效。千位分隔符和 `top-3` 不再被拆成无效子句/数字。
- 在 696 块六论文隔离库上，5 道 `cross_document` 题各重复两次，10/10 调用成功；人工逐题复核为 10/10 正确，关键事实（GRPO、5 levels、top-3、500、428、200、7,000、1,280 及名称计数）均进入上下文。重复运行配置、上下文和 metadata 100% 稳定，答案措辞按模型采样有变化。
- 离线 Hybrid 路由检索的 cross-document 在 @50 仍为 `5/5` 完整覆盖；未因此接入 Graph-RAG、通用工具执行器或改变默认 dense 配置。

### 2026-09-02：computation 全量门禁复测

- 六论文 696 块隔离库上完成 20 道 computation 题各两轮，40/40 调用成功；除 `calc-af3-02` 外，19 道题两轮均给出正确结果（38/38）。
- `calc-af3-02` 两轮均因 Figure 3(a) 的 87.7/86.9 只存在于图像像素、未进入文字/坐标证据而拒答；这是 image-only 证据边界，不用工具执行器修复。上下文、metadata 和运行配置两轮均稳定，答案措辞逐字稳定率为 3/20。
- 该结果不满足“工具调用至少 5 道稳定可修复运算失败”的门槛；暂不接入通用 calculator/tool executor。保留现有 opt-in vision 路径处理图像证据。

### 2026-09-02：泛化留出基准（默认关闭）

- 在仓库外下载并核验 ACL Anthology 的 TACL 2025 TANQ 与 Findings of EMNLP 2025 FigEx，SHA-256 分别为
  `584f672c01c03087fe8bdfc1624bfa82e22af949cc2087dfd0da47c39a5609f9` 与
  `241a7ab35f8d695c16608dd39ed5459d6a13d4e3d840b7f49cfefe7194ec919c`；未将 PDF 放入 Git。
- 新增 `manifest_generalization.json` 与 16 道人工核验用例，继承六论文/66 题扩展清单，形成 8 篇/82 题留出集，覆盖复杂表格、图像空间关系、数值指标和跨文档证据；默认清单不变。
- 离线 Hybrid + 路由 + 查询分解 + 结构化表格保护 + parent-window 的 @50 结果为 fact macro/micro=`0.950/0.959`、完整覆盖 `75/82`，路由 `62/62` 正确。TANQ 新题为 `7/9` 完整，FigEx 新题为 `6/7` 完整；唯一 image-only 题按设计未由文字检索覆盖。
- 该结果只证明留出集的解析/检索代理；尚未调用生成模型、运行 RAGAS 或完成独立语义复核。下一步先人工复核 16 道题，再决定是否进行两轮生成。

### 2026-09-02：泛化表格行选择修复

- 修复通用表格确定性路径：模型与数据集同时出现时优先识别模型，继承跨行/跨空白数据集单元格，识别 Oracle/Closed book 等 setting 分组，按覆盖全部实体的 section 选择重复技能行，并支持同题多行与 `overall F1` 列消歧；未绑定具体论文名称。
- 新增三项核心回归测试；全套离线单测 `205` 项通过，语法检查与 `git diff --check` 通过。
- 在 8 篇论文隔离库上复核 TANQ/FigEx 四类受影响表格题，确定性答案已分别命中 MedICaT/BioSci-Fig、FigEx-7B 和 Gemini Flash/Human/PaLM-2 的正确行列；未重建原 `chroma_db`。
- 已授权的 16 题×2 轮生成共写入仓库外临时 JSONL；限定新增 16 题重试后，32 次中 10 次成功、22 次为 `Connection error`。成功样本在修复前暴露上述表格行误选；端点 `models.list()` 可达，但本轮不把不完整生成结果计作质量结论，也未运行 RAGAS。
- 随后单独发送一次最小 `chat.completions` 请求成功返回 `2+2=4`，但耗时约 4 分钟；这只能说明生成端点偶尔可达，不能替代 16 题稳定性测试，也未据此重跑整批。

### 2026-09-02：泛化留出生成恢复与人工复核

- DeepSeek 连接恢复后，16 道新增留出题各生成两轮；最终 32/32 行均成功，结果仅保存在仓库外临时 JSONL，未写入原数据库。
- 16 个 case 的 runtime、源码指纹、context IDs 和 metadata 两轮均稳定；答案逐字稳定为 `5/16`，其余为措辞变化。
- 人工核对两轮答案与 PDF/金标准：`12 correct / 3 partial / 1 incorrect`。部分正确为 TANQ 平均列数、TANQ+FigEx 跨文档数据集规模、TANQ/ FigEx caption 模型；错误为 FigEx Figure 7 的 O 标签区域。
- 该结果证明生成链路本轮可达，并暴露了留出集中的解析/跨文档/图像空间缺口；不等同于生产正确率，也未运行 RAGAS。复核标签保存于 `evaluation/benchmark/reviews_generalization_16.jsonl`。

### 2026-09-02：泛化留出检索与空间证据修复（未提交）

- 同节扩展允许相邻的重复 section header，并在上下文上限前优先放置锚点及连续续块；新增统计、表格行列、模型、caption 切分和证据评估的通用中英别名。跨来源拆出的短名称只负责路由，来源内补证据复用共享问题谓词。
- 图形坐标证据统一说明 PDF 左上原点和 y 向下，并在生成上下文中为已有坐标行补充 page-level quadrant；不改数据库、不启用图像识别。
- 新增/更新回归测试后全套离线单测 `208` 项通过。904 块隔离库上 4 道受影响留出题完成 8/8 次无错误生成，另对 TANQ 平均列数补测 2 次；最终人工复核记录更新为 `16 correct`。结果和原始 API 追踪仍只保存在仓库外临时目录，未运行 RAGAS。

### 2026-09-02：泛化留出最终全量稳定性复测

- 使用当前源码指纹和 904 块隔离库，对合并清单 82 题各生成两轮，共 164 次 DeepSeek 调用；`164/164` 无 API 错误。
- 两轮 runtime 配置、源码指纹、context IDs 和 metadata 均 `100%` 稳定；答案逐字一致 `25/82`，其余为措辞变化。
- 词面事实代理两轮分别为 macro/micro=`0.7913/0.8042`、`0.7882/0.8000`，完整覆盖 `57/82`、`56/82`；该代理不替代语义判断。新增 16 题沿用 PDF 人工复核为 `16/16 correct`。
- 完成网页冒烟：当前数据库显示 104 块；上传、问答、大纲、出题页签均可加载；原有 DrugR Table 2 问题返回 `0.3404`，普通 GRPO 问题返回正确算法名称。所有追踪与审计 JSONL 均保存在仓库外临时目录，未运行 RAGAS、未改数据库、未提交或推送。

### 2026-09-02：视觉基准空间标签复核

- 直接核对 SciDQA PDF Figure 6(a) 的原始图片：嵌入修订稿页面中的绿色高亮框位于页面中线以下，故将 `img-scidqa-02` 的 gold、required fact 和 context 从“上半部”更正为“下半部”；仅修正基准标注，不改变视觉代码。
- 当前 full+detail 视觉实验 10 道题×3 轮共 30/30 调用成功；按修正后的标签每轮均 9/10（合计 27/30）正确，剩余一次错误为 DrugR 箱线图中位线判断，其他两轮该题正确。该结果仍只是六论文 opt-in 门控证据。
- 隔离临时数据库完成一次真实网页 opt-in 烟测：Figure 3 anti-diabetic 结构题正确回答原始 2 个 F、优化 1 个 F，并返回第 5 页；此前同一网页测试的 Figure 2(A) 来源页正确但答案出现一次 DrugR 随机误判，GRPO 与 Table 2 对照题均正确。
- 视觉路径、普通文本和表格路径均保持默认行为边界；本轮未改代码、未重建项目数据库、未提交或推送。

### 2026-09-03：82 题生成结果语义审计

- 使用前一轮仓库外临时结果 `answers_generalization_final.jsonl` 完成 82 题×2 轮离线人工语义复核；未重复调用 API、未运行 RAGAS、未重建数据库。
- 新增 `evaluation/benchmark/reviews_generalization_82.jsonl`，逐题记录答案、表号、单位、公式和引用维度；按两轮较保守结论为 `63 correct / 9 partial / 10 incorrect`。
- 主要稳定缺口集中在 SciDQA 的 7,000/85%、80%/25%、full-text、BM25 分块，科学表格的 `<R>/<C>` 与未来方向，MgNO 的 `A*u=f`/3×3 与随机种子复现，AF3 的 cross-distillation，以及 THINKNOTE 的 500 题、Jackie Robinson、公式顺序、指标完整性和图表稳定性。
- 两轮词面事实代理约 `0.80` 仅作提示，不替代语义审计；暂不据此新增 Graph-RAG、工具执行器或改动默认检索路径。当前工作区未提交、未推送。

### 2026-09-03：生成基准按目标论文隔离检索

- 发现旧的 82 题生成追踪虽记录了 `document_id`，但实际调用仍使用全局多论文 ChromaDB；通用问题的上下文可能混入其他论文，不能公平归因生成错误。
- `query_knowledge` 新增可选 `source_filter`，贯穿 dense/Hybrid、结构化表格、公式、限制、图形和最终上下文；网页默认不传，原有行为不变；跨文档用例允许主文档和 `additional_document_ids`。
- `evaluation/generation_stability.py` 现在按 manifest 将目标文档 ID 映射为 source filename 后传入过滤，并在追踪中记录 `source_filter`。
- 新增来源隔离回归测试；已有输出若缺少匹配的 `source_filter` 会自动视为待重跑，避免把旧追踪误当成新结果。全套离线 unittest `210` 项通过，并用临时 Chroma collection 验证单来源 `$eq` 与跨来源 `$in` 过滤；本轮未调用 API、未重建项目数据库、未提交或推送。

### 2026-09-03：隔离库复现与生成重试

- 从桌面 8 篇已核验 PDF 重建一次性 `/tmp` 隔离库，块数为 `904`；真实 Chroma 检查 82/82 case 无来源越界，82/82 有上下文，跨文档 2/2 命中允许来源。
- 按新 `source_filter` 启动 82 题×2 轮生成；前 13 行中 6 次成功、7 次 `Connection error`，连续失败后手动停止，追踪仅保留在 `/tmp/scirag_generalization_sourcefilter.EiU7As/`，不作为质量结论。项目数据库和 Git 均未被生成过程写入。

### 2026-09-03：来源隔离生成复测完成

- 网络权限恢复后从断点续跑，最终 82 题×2 轮 `164/164` 成功；904 块隔离库、runtime 配置、context IDs、metadata 和 source filter 均可追溯，来源越界 `0`。
- 两轮答案词面事实代理分别为 macro/micro=`0.8187/0.8333` 与 `0.8370/0.8542`，完整覆盖 `59/82` 与 `62/82`；gold-context recall 均为 `0.6707`。这些是筛查指标，不是语义正确率或 RAGAS。
- 证据校验为 `ok=66/164、review=22/164`；重复答案逐字稳定 `25/82`，配置/上下文稳定 `82/82`。新结果保存在仓库外 `/tmp/scirag_generalization_sourcefilter.EiU7As/`，尚未用旧人工复核文件直接套标，待下一步独立语义复核。

### 2026-09-03：来源隔离批次差异语义复核

- 对同一 82 题的旧全局检索批次与新来源隔离批次做离线差异比较；22/82 道题答案签名两轮均不变，60 道题有变化。
- 回查 22 道高风险题的 PDF、金标准和新上下文，其余题沿用旧语义标签并检查无事实回退；新批次按两轮较差结果统计为 `66 correct / 11 partial / 5 incorrect`。
- 新增 `evaluation/benchmark/reviews_generalization_82_sourcefiltered.jsonl`。改进集中在 TableLLM、MgNO、AF3 和 THINKNOTE 的来源证据召回；未据此接入 Graph-RAG、工具执行器或默认 OCR/视觉路径。

### 2026-09-03：来源内证据补全定向修复

- 针对来源隔离批次暴露的通用召回缺口，增加中英方法术语别名；单一 `source_filter` 时启用有界来源内
  lexical/同节/数字证据补全，普通网页无过滤参数时保持原检索路径。
- 显式 Figure 查询在精确图块后补入同来源同页的 `text/formula` 解释块，排除图片文本，覆盖双栏 PDF
  将解释句误标为 formula 的情况；未增加 OCR、图片索引、Graph-RAG 或工具执行器。
- 全套离线 unittest `213` 项通过，`git diff --check` 通过。9 个受影响 case 各两次定向生成均无 API
  错误，随后对新增“检索/第一人称”别名影响的 `scidqa-08`、`scidqa-10` 各补测两次；最终
  `scidqa-05/06/07/08`、`table-llm-03`、`mgno-11` 和 `thinknote-06` 两轮均覆盖声明事实，
  `scidqa-10` 两轮均说明同行评审、作者和第三人称改写但漏写 OpenReview 平台；`thinknote-13` 追加两次
  均正确召回并回答 Jackie Robinson。该证据不替代 82 题全量重评、RAGAS 或生产准确率。

### 2026-09-04：来源补全影响审计与门控收窄

- 离线比较显示无门控来源内回退会改变 `61/82` 题上下文，并造成 `5` 题事实覆盖代理下降；收窄为
  显式方法/阈值/配置等证据意图，并对 Figure 查询排除路由证据后，最终改变 `24/82` 题，检索事实
  覆盖代理由 `65/11/6` 变为 `75/6/1`（full/partial/zero），无覆盖降级、无来源越界。
- 在收窄门控版本上完成 `20` 个 case × 两轮、`40/40` 次成功调用；这是定向生成和检索代理证据，
  不是 82 题修复后全量重评、RAGAS 或生产准确率证明。Figure 同页解释补全仅保留“示例/正确答案”
  问题；`thinknote-09` 仍因图像分组与文字层映射不可靠而未解决。

### 2026-09-04：表格结构与跨页续句修复

- Setting 列可作为复合数据集限定，避免同一模型的基线行抢先命中；多层 PDF 表头在存在重复文本列时
  合并为完整的场景/数据集列名，并识别拆到两个单元格的分组标签（如 `Llama-3-Ins-70B`）。
- parent-window 仅在前块以英文/中文连接词结尾时允许跨页续句，覆盖 TANQ 的 `6.7 rows and / 4 columns`
  形态，不开放任意跨页邻块。
- 新增三题结构回归，离线 unittest `216` 项通过。旧 904 块隔离库三题各两轮共 `6/6` 成功；表格解析
  修复尚未在重新切分的隔离库上端到端复测，因本地 Sentence-Transformers 加载触发 Hugging Face DNS
  重试，项目数据库未重建。

### 2026-09-04：公式证据抽取与排序修复

- 公式问题门控补充形式化定义、最终输出和激活函数等通用表达；恢复 PDF 公式时改用 PyMuPDF 坐标行合并，保留双栏页面中的括号、运算顺序和参数顺序；公式候选按问题变量的等式左侧优先排序。系统提示新增公式逐字转录规则。
- 新增公式候选、变量左侧排序和 PDF 版面合并回归测试；全套离线 unittest `219` 项通过，编译检查与 `git diff --check` 通过。
- 旧 904 块隔离库上 MgNO 两题 `4/4` 次生成正确；THINKNOTE 公式题仍受历史乱码公式块影响。用当前解析器对原 PDF 的临时内存证据验证两题均转录为 `T = M(Ika, q, D)` 与 `y = M(Ita, q, T, R)`。未重建项目数据库、未运行 RAGAS、未提交或推送。

### 2026-09-04：新鲜隔离库 Phase 1 门禁与全量复测

- 用当前解析器从 8 篇已核验 PDF 新建 `/private/tmp` 隔离库，共 `898` 块（720 text / 48 table / 130 formula）；未触碰项目 `chroma_db`。离线 Hybrid + 路由/查询分解/表格/公式/限制门控的 82 题 @10 事实完整覆盖为 `69.5%`，表格题 `96.4%`，路由 `62/62` 正确；这只是检索代理。
- 公式/表格目标 7 题各重复两轮，`14/14` 调用成功且人工核对正确。随后发现新 PDF 的 Table 3 将模型分组拆为两格；修复表行过滤保留分组上下文，新增回归后全套离线 unittest 为 `220` 项。
- 在同一 898 块库上完成 82 题×2 轮生成，`164/164` 无 API 错误；来源越界 `0`，上下文/metadata/运行配置两轮 `82/82` 稳定，答案逐字稳定 `24/82`。两轮原子事实 macro/micro 为 `0.9010/0.9167` 与 `0.9183/0.9292`，完整覆盖 `69/82` 与 `70/82`；不把该代理当作语义正确率，未运行 RAGAS，也未更新旧人工复核标签。

### 2026-09-04：新鲜批次独立语义复核与定向回归

- 对上述 82 题×2 新答案逐题回查 gold、PDF 和上下文；结合两道受影响题的修复后两轮定向结果，保守状态为 **79 correct / 1 partial / 2 incorrect**。`scidqa-10` 仍缺少 OpenReview 平台名；`thinknote-09` 与 `figex-holdout-07` 需要图像像素映射，文字路径继续拒答。该统计不是 RAGAS、生产正确率，也不是修复后 82 题再次全量生成。
- 修复通用表格实体标点、模型/方法的第一列分组继承及带指标后缀的命名列匹配；为架构替换问题补充中英来源内证据召回。`thinknote-07` 与 `af3-02` 各两轮均返回 gold 值。
- 全套离线 unittest `223` 项通过；生成追踪仍在仓库外 `/private/tmp`，未改项目数据库、未运行 RAGAS、未提交或推送。

### 2026-09-04：未决图像题的视觉路径定向门禁

- 复用默认关闭的 `SCI_RAG_VISION_ENABLED` full+detail 路径，在仓库外临时目录对 `thinknote-09` 和
  `figex-holdout-07` 各执行两轮视觉调用；`4/4` 次成功且人工核对正确：Figure 3(a) 为 `53.20/68.80`，
  Figure 7 的 O 位于右下角一致性表面板。
- 这只证明两个已知图像边界题在 opt-in 路径上的定向可用性；文字基线的 `79/1/2` 统计不改写为全量
  视觉准确率，也不据此开启默认视觉、OCR、图片索引或 Graph-RAG。结果 JSONL 仍保存在
  `/private/tmp/scirag_phase1_formula_6cya35wz/vision_targeted_phase615_unique.jsonl`。
- 未改项目数据库；未运行 RAGAS；未提交或推送。

### 2026-09-04：网页状态与跨论文公式复测

- 修复上传成功后知识库块数仍显示旧值的问题：上传回调现在同步更新状态和可见计数；隔离网页中计数由 `117 → 224 → 316` 正确更新，未触碰项目数据库。
- 修复多文档复合问题的公式证据丢失：对每个公式子问句独立执行文档路由和公式候选检索，并让来源内 lexical 证据优先于泛化同节回退。隔离网页复测同时正确回答 DrugR 显式推理数据集为 `4,855` 个样本，以及 `T = M(I_{ka}, q, D)`。
- 新增跨论文公式路由回归测试；全套离线 unittest `224` 项通过，编译检查和 `git diff --check` 通过。网页复测使用 `/private/tmp` 隔离库，未运行 RAGAS、未提交或推送。

### 2026-09-04：来源内复合问题的子句证据排序

- 修复来源过滤 + 查询分解下的通用证据回退：有实际疑问词的子句现在使用自身检索，只有纯来源标识才复用整句，避免 SciDQA 的 OpenReview 来源段落被图注挤出上下文。
- 新增来源内子句回归测试；8 篇论文 82 题离线 Hybrid 诊断的文档路由保持 `62/62` 正确；来源过滤下的 SciDQA 复测上下文包含 OpenReview，受控 DeepSeek 生成明确回答同行评审来源和第三人称改写。
- 全套离线 unittest `225` 项、编译检查和 `git diff --check` 通过；未重建项目数据库、未运行 RAGAS、未提交或推送。

### 2026-09-04：当前修复批次收尾

- `scidqa-10` 独立生成两轮均明确写出 OpenReview 的同行评审—作者来源、第三人称改写和上下文补充。
- DrugR Table 6、TANQ/FigEx 跨论文模型题和 THINKNOTE Table 1 控制题均通过；上下文来源均符合预期。
- 现有 8 篇论文、82 道题冻结为开发回归集，不再用其继续调参证明泛化；下一步转向未参与修复的盲测留出集。

### 2026-09-04：盲测留出集候选草稿

- 新增 `evaluation/benchmark/manifest_blind_holdout_draft.json` 与 `cases_blind_holdout_draft.jsonl`，追加桌面上未进入现有基准的 SciDC、NaviRAG 两篇 PDF 和 16 道公式、表格、导航及效率题；清单仍标记为 `draft-candidate`。
- 已核对 SHA-256、关键表格/公式页、页码和 required facts；离线基准校验显示 10 篇论文、98 道题、所有金标准事实均可由片段或显式别名核验。
- 候选题尚未上传 Chroma、尚未运行生成或用于调参；待题目与论文适用性确认后再转为正式留出集。

### 2026-09-05：Phase A 候选题独立审核

- 逐题回查新增 16 道题的 PDF 页面、表号、数值/单位、公式和答案片段；SciDC 第 3/6/7 页、NaviRAG 第 5/7/8 页另做视觉核对，未发现需要修正的事实错误。
- 结构校验保持通过：10 篇论文、98 道题，16 道新增题的页面范围和 required facts 均可由原文或显式别名核验；这证明题库自洽，不证明模型回答质量。
- 两篇来源均为免费 arXiv 预印本，清单继续保持 `draft-candidate`：可作为未参与修复的技术压力测试，但在加入正式泛化结论前仍应补充或获得用户认可的同行评审/顶会顶刊开放论文。
- 本阶段未上传 Chroma、未调用 DeepSeek、未运行 RAGAS、未调参；下一步须先决定是否接受这两篇作为正式盲测来源，再进行隔离库检索与生成。

### 2026-09-05：Phase B–D 候选盲测检索与生成审计

- Phase B 在 10 篇论文、98 道题上完成隔离 Hybrid 检索：候选池 50、上下文上限 10；目标论文命中
  `0.990`、页级命中 `0.989`、Table N 命中 `1.000`、required-fact macro/micro `0.945/0.948`，
  完整/部分/零覆盖 `0.908/0.071/0.020`。SciDC 与 NaviRAG 新题 fact macro 为 `0.875/0.906`；
  均为检索覆盖代理，不代表答案正确率。
- Phase C 在仓库外 `/private/tmp/scirag_phaseCD_db.y1f4be` 的 1126 块隔离库上，对新增 16 题各生成两轮，
  `32/32` 成功，来源越界 `0`，运行配置/context/metadata 稳定 `16/16`，答案逐字稳定 `7/16`。
  独立人工核对为 `14 correct / 2 partial / 0 incorrect`；partial 为 SciDC 公式 `\\notin` 换行和
  NaviRAG 节点决策题未显式写出 `π(n) ∈ {absorb, expand}`，记录见
  `evaluation/benchmark/reviews_blind_holdout_16.jsonl`。
- 修复效率问题的通用来源内证据门控：效率/代价/开销问题现在补入同来源 lexical 证据；scidc-holdout-08
  两轮均覆盖 `3.6k→4.2k`、`0.8`、`1.9k→2.3k`、`2.5` 和约 `3×`。离线 unittest `227/227` 通过。
- Phase D 仅记录限制与后续门槛：候选来源为免费 arXiv 预印本，尚不构成正式泛化证明；未重建项目数据库、
  未运行 RAGAS、未启用默认视觉/Graph-RAG/工具执行器，未提交或推送。生成 trace、回答和审计 JSONL 均保留在
  `/private/tmp`，不作为仓库测试输入。

### 2026-09-05：Phase B–D 验收收口

- 将原候选盲测清单标记为 `development-regression`：这 16 道题已参与诊断/定向修复，不再作为未见盲测。
- 人工复核记录按现有 schema 归一化：未审计 citation 标为 `uncertain`，两个公式缺口的维度标为 `incorrect`，
  整体判断仍为 `14 correct / 2 partial / 0 incorrect`；新增 16 行精简回答快照供离线复核。
- README 同时记录 @10 最终上下文与 @50 候选池，明确来源过滤生成只能证明隔离诊断，不能证明网页端路由或泛化。
- 本轮未调用 API、未启动网页、未重建项目数据库；下一步先运行 `review_answers.py --require-all`，再处理两个
  partial 的通用公式输出问题。

### 2026-09-05：公式输出边界修复

- 在公共公式路径恢复被 JSON 控制字符吞掉的常见 LaTeX 命令（例如 `\\notin`），并在公式证据已检索到但答案
  漏掉显式等式/运算符时附加带标签的原文核对项；未加入论文专用分支。
- 新增 3 个离线回归测试；全套 unittest `230/230` 通过。随后对两个历史 partial 题和一个正常公式控制题各跑
  两轮，`6/6` 成功，配置/context/metadata/provenance 均稳定；两个 partial 的通用症状已消失，控制题未附加无关
  公式行。结果写于仓库外 `/private/tmp/scirag_phase_formula_fix_v2.jsonl`（SHA-256：
  `482989d6aae7a1dc3f2624da8f89099443fd8523dcdacf32bebacedbacf1ec1b`），不代表 16 题全量重新生成。
- 随后收窄 LaTeX JSON 控制字符映射，仅保留常见且可由该故障解释的命令；unittest 仍为 `231/231`，未改变默认
  网页路径。
- 为区分诊断与真实路由，runner 增加 `--no-source-filter`；16 道新增题无过滤各运行一轮，`16/16` 成功且首个
  上下文命中目标来源 `16/16`，trace provenance `16/16` 完整。输出为仓库外
  `/private/tmp/scirag_phase_router_v1.jsonl`（SHA-256：
  `6540ea666a6e1170f344387bc14df9ada050a90817cc2341f75bc81abf63b376`）。词面审计
  macro/micro=`0.5833/0.5600` 仅作筛查信号，不等于语义准确率；未对这批题再次人工复核。

### 2026-09-05：Phase 0–1 网页冒烟验收

- 使用项目现有 `chroma_db`（页面显示 104 个文本块）启动原网页，未上传、删除或重建数据库。
- 依次验证 Table 2 精确单元格、显式推理数据集管道、GRPO 方法和公式参数四类问题；4/4 均返回与金标准一致的核心事实，Table 2 题明确返回 `0.2060` 和 `Table 2`，未复现此前误答为 Table 1 的问题。
- 网页启动需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 避免已缓存模型的 Hugging Face 元数据联网检查；该环境变量不改变代码默认行为。测试结束后已停止网页，未调用 RAGAS、未提交或推送。

### 2026-09-05：公式转义误改普通答案的边界修复

- 验收发现公式修复此前在公式意图判断前作用于所有答案，且无词边界替换可能把普通行首 `equation` 改成 `\\neq` 片段。
- 现改为仅在公式问题中执行修复，并对控制字符序列增加字母边界；新增普通换行和 `equation` 回归测试。未改变非公式网页默认路径，也未提交或推送。
- 修复后离线 unittest `233/233` 通过；三个公式相关用例各运行两轮，`6/6` 成功，两个目标公式均保留显式运算符，控制题未附加无关公式。trace provenance、context 和 metadata 均为 `3/3` 稳定；结果保存在仓库外 `/private/tmp/scirag_phase_formula_fix_v3.jsonl`（SHA-256：`901e0a65c655d6bcf57338dd9920fc17a6ac32dda8052fa45de0e16f6065e26e`）。

### 2026-09-05：Phase E0 新候选论文与题目草稿

- 用户批准下载并在仓库外保存三篇官方免费 PDF：NeurIPS 2024 的 SPIQA/UDA，以及 ICLR 2025 的 LiveXiv；登记文件名、页数和 SHA-256。
- 完成标题页、代表性表格/流程图和附录表格的视觉复核；20 道新题逐题检查表号、数值、单位、题型和 required facts，均能回溯到 PDF 原文。修正 UDA Table 5 题目，区分 `GPT-4-Omni=72.4` 与 `Well Parsed=71.9`，避免把列错位当成答案。
- 新增 `manifest_blind_holdout_fresh_draft.json` 与 `cases_blind_holdout_fresh_draft.jsonl`，清单状态为 `phase-e0-audited-candidate`。离线校验为 3 篇、20 题、required facts 全部受 gold contexts 支持；BM25 @10 检索代理为 fact macro/micro=`0.750/0.780`、完整覆盖 `14/20`。
- 该候选集尚未上传 Chroma、未调用 DeepSeek、未用于调参、未运行 RAGAS；上述检索数字只用于发现语言/路由缺口，不证明答案正确或泛化。

### 2026-09-05：Phase E2a 评估协议冻结

- 将三篇论文、20 道题的清单状态改为 `frozen-fresh-evaluation`，明确冻结日期和“不得因失败结果修改或调参”的协议。
- 补充事实：E1 离线检索诊断发生在冻结前，因此该集合不再称为严格 blind holdout；若后续修复使用它，必须降级为 `development-regression` 并另建未见评估集。
- 冻结文件 SHA-256：manifest=`cb256f0084301d2aaefdad3502d94bbd473a9d1ad280daf5fd24c44d5518f352`，cases=`c103d2ff14e55a31cdd858b39f4ccd083e86784ffcaa0bd9fdaffb5847eaf01c`。
- 本步只修改基准元数据和说明文档；未修改检索/生成代码、未启动网页、未调用 DeepSeek、未写入 Chroma、未运行 RAGAS。

### 2026-09-05：Phase E2b 隔离库生成验收

- 在仓库外重建 13 篇论文、1,594 块隔离 Chroma，仅评分冻结的20题；Dense 与 Gated Hybrid + 固定 BGE reranker 均 `20/20` 请求成功，未触碰项目 `chroma_db`。
- Dense 上下文 required-fact 完整覆盖 `12/20`，人工语义复核 `9 correct / 7 partial / 4 incorrect`；Gated Hybrid + Reranker 提升至 `15/20` 和 `14 correct / 3 partial / 3 incorrect`，但未达到 `≥17/20` 正确门槛，不切换线上默认。
- 两批 trace 保存在仓库外并记录于 `evaluation/benchmark/PAPER_AUDIT.md`；本步未修改应用/检索代码、未运行 RAGAS。
- 发现 `uda-fresh-06/07` 在完整语料路由中混入其他论文上下文；其中 `uda-fresh-07` 因此错误拒答。不得直接用冻结集失败样本调参；后续修复应使用开发回归集并另建未见评估集。

### 2026-09-05：Phase E3a 评估集合协议转换

- E2b 已经使用并查看这20道题的生成结果，清单现改标为 `development-regression`；允许用于回归修复，但不再作为未见评估证据。
- E2a 冻结版本的 manifest SHA-256=`cb256f0084301d2aaefdad3502d94bbd473a9d1ad280daf5fd24c44d5518f352`、cases SHA-256=`c103d2ff14e55a31cdd858b39f4ccd083e86784ffcaa0bd9fdaffb5847eaf01c` 保留在 `evaluation/benchmark/PAPER_AUDIT.md`。
- 本步只调整评估协议元数据和说明；尚未修改检索/解析代码。

### 2026-09-05：Phase E1 候选集离线检索对照

- 同一 3 篇候选 PDF 的 BM25-lite @10 required-fact macro/micro=`0.750/0.780`，完整覆盖 `14/20`；单独 dense 为 `0.279/0.305`、`5/20`；默认 Hybrid-RRF 为 `0.582/0.627`、`10/20`。
- 开启结构化表格、限制证据、文档路由、查询分解和 parent-window 后，Hybrid 为 `0.738/0.780`、`14/20`；固定缓存的 `BAAI/bge-reranker-base` top-50 重排并 RRF 融合后为 `0.838/0.847`、`15/20`。
- reranker 平均单题约 5.86 秒、P95 11.01 秒、峰值内存约 1.75 GB；5 道题仍未完整覆盖，主要是 LiveXiv 视觉/人工复核/效率段。以上仅为检索词面代理，不代表答案正确或泛化，也不支持直接切换线上默认。
- 诊断 JSONL 保存在仓库外 `/private/tmp/sci-rag-fresh-bm25.json`、`/private/tmp/sci-rag-fresh-hybrid.json`、`/private/tmp/sci-rag-fresh-hybrid-gated.json` 和 `/private/tmp/sci-rag-fresh-reranker-gated.json`；未上传 Chroma、未调用 DeepSeek、未运行 RAGAS。

### 2026-09-05：Phase E3b 通用表格与来源证据修复

- 修复 PDF 导出表格的重复 `@1/@5/@10/@20` 指标行、跨单元格分组名和拆分行实体；UDA Table 6 现在可稳定定位
  Sparse BM-25 的 FinHybrid/PaperTab `@10` 值 `87.4/90.0`。同时让“平均绝对变化”等语义列优先于比较上下文列，
  LiveXiv Table 2 可定位 VQA/TQA 的 `2.336/2.105`。
- 文档路由支持三字母全大写来源缩写（如 UDA），并在唯一来源且明确询问流程、工具、过滤、样本或变化时补入同来源
  lexical/同节证据；普通无证据意图的网页问题仍不触发该回退。
- 新增表格与路由回归测试；全套离线 unittest `236/236`、`py_compile` 和 `git diff --check` 通过。用真实 PDF 经
  `load_and_split_document` 接入内存假 Chroma 做无 API 应用路径检查，UDA Table 6 和 LiveXiv Table 2 均返回预期值。
  未调用 API、未启动网页、未重建项目数据库、未提交或推送。

### 2026-09-05：Phase F1 新未见留出集冻结

- 经用户同意下载两篇此前未使用的官方开放论文：TableRAG（EMNLP 2025）和 CURIE（ICLR 2025；使用公开
  arXiv 版本），保存于仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。TableRAG 20 页、SHA-256
  `f388e3af1397055b2c5fda06831f373c23826cda4b4a395a38070bcf3edee4b1`；CURIE 48 页、SHA-256
  `4416483d8380052f6fae90657c36a0857f33d3a2933162edaa007d4072841a07`。
- 完成两篇 PDF 文字层、关键表格、流程图和 Figure 31 地图坐标的视觉核对；新增 12 道题，覆盖复杂表格、
  长上下文/跨页方法统计和两道 image-only 题。`manifest_phaseF1_unseen.json` 状态为 `frozen-unseen`，
  未运行 Chroma、DeepSeek、RAGAS 或任何调参。
- 基准校验命令通过：2 篇、12 题、required facts 全部由 gold contexts 或显式别名支持。若后续失败指导代码修改，
  本集必须降级为 `development-regression` 并另建确认集；本步未提交或推送。

### 2026-09-05：Phase F2 新论文离线检索闸门

- 在仓库外临时内存索引上完成 BM25、Dense、Hybrid、Gated Hybrid 和固定 BGE reranker 对照；未重建项目
  `chroma_db`、未启动网页、未调用 DeepSeek。目标论文路由均为 `12/12`，错误路由为 `0`。
- 表格事实覆盖改为包含 retrieval/reranker/网页路径实际可见的 caption/header；CURIE Table 2 的 PDF 导出
  拼写 `PV-speciific` 通过用例别名承接。相关单元测试 50 项和基准校验通过。
- F2 @10 非图像题完整覆盖：BM25 `9/10`、Dense `8/10`、Hybrid `9/10`、Gated Hybrid `9/10`；@50
  各文本/表格配置均 `10/10`。两道 image-only 题文字层均不计入文本闸门，视觉路径后续单独验证。
- Gated Hybrid + reranker 没有超过 Gated Hybrid，平均单题约 `6.21s`、峰值约 `1.85GB`；不切换线上默认。
  F1 清单因本步修复已改标为 `development-regression`，不再作为未见泛化证据。诊断 JSON 仅保存在
  `/private/tmp`，本步未运行 RAGAS、未提交或推送。

### 2026-09-05：Phase F3 受控生成与视觉路径验收

- 在 402 块隔离 Chroma 上对 F1 的 10 道文本/表格题各生成两轮，DeepSeek `20/20` 成功、无 API 错误；人工复核每轮 `8 correct / 2 partial / 0 incorrect`。表格/方法核心事实正确，但两道题分别漏写限定词或一个请求值。
- 对两道 Figure 31 `image_only` 题各生成两轮，视觉请求 `4/4` 成功；模型识别题 `2/2` 正确，坐标题 `0/2` 正确（`103.2` 被读成 `183.2`）。视觉能力暂不通过闸门，继续默认关闭。
- 按 PDF 原文将 `tablerag-f1-03` 的筛选条件从“至少 20 行”纠正为“超过 20 行”，并将 required fact 改为 `more than 20 rows`。F1 已是开发回归集，不能作为未见泛化证据。
- 结果 JSON 只保存在仓库外 `/private/tmp`；本步未运行 RAGAS、未启动网页、未重建项目数据库、未提交或推送。

### 2026-09-06：Phase F4A 表格答案完整性修复

- 将 `IoU` 纳入通用多指标表格列选择；结构化表格答案保留问题中明确命名的非数值限定词，并在问题要求时保留表注中的 `exact match` 指标。
- 402 块隔离库真实 `app.query_knowledge` 复测：TableRAG Table 4 返回 Qwen backbone、exact match 与两行 HeteQA 值；CURIE Table 7 返回 `0.49/3.03/3.05`。
- 定向测试 `169/169`、全套测试 `241/241`，基准校验、编译和 diff 检查通过。未启动网页、未修改项目数据库、未运行 RAGAS、未提交或推送。

### 2026-09-06：Phase F4B 视觉高分辨率对照

- 以现有 10 道 image-only 挑战题和 F1 Figure 31 两题做 4× full+detail 隔离实验：`9 correct / 2 incorrect / 1 empty`，输出仅保存在 `/private/tmp/scirag_phaseF4B_highres_r1.jsonl`。
- 高分辨率修正了 `W=103.2`，但造成其他图像题退化，没有整体稳定收益；不接入正式代码，视觉继续默认关闭，不进行第二轮 API 调用。
- 未修改项目数据库、未运行 RAGAS、未提交或推送。

### 2026-09-06：Phase G 最终未见确认集

- 经用户批准下载 ACL Anthology 的 MT-RAIG、WikiMixQA、TableEval 三篇免费 PDF；均保存在仓库外
  `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`，PyMuPDF 可读取，文件哈希和页数写入新的确认清单。
- 新增 `evaluation/benchmark/manifest_phaseG_confirmation.json` 与 `cases_phaseG_confirmation.jsonl`，冻结 3 篇论文、20 道题；
  离线校验通过，required facts 全部有 gold context 或显式别名支持。
- 隔离 Chroma 共 472 块。Gated Hybrid @10/@50 完整事实覆盖为 `13/20`、`17/20`；reranker @10 为 `12/20`，不切换线上默认。
- DeepSeek 首次沙箱运行有 34 次连接错误；联网权限续跑后 40/40 成功。配置与上下文两轮均稳定，答案逐字稳定 3/20。
  人工语义复核两轮均为 `14 correct / 4 partial / 2 incorrect`，未达到 `≥17/20` 且错误 `≤1` 门槛。
- 生成 trace 和审计 JSON 仅保存在 `/private/tmp`；未运行 RAGAS、未写项目 `chroma_db`、未提交或推送。该确认集失败后不再直接调参，后续如修复需另建未见确认集。

### 2026-09-06：Phase H0/H1 评估口径与复合问题分解修复

- 重新审计 Phase G 的真实生成 trace：离线候选诊断 @10 的 `13/20` 不能代表应用实际送模上下文；按 trace `contexts`
  重算为 `11/20 full、7/20 partial、2/20 zero`，macro/micro=`0.7592/0.7157`。Phase G manifest 现标为
  `development-regression`，题目和金标准未改动，后续不再用它证明未见泛化。
- `evaluation/answer_audit.py` 现在分别报告答案事实覆盖和 trace `contexts` 覆盖，避免把候选池指标与最终生成证据混用。
- `query_variants` 支持按 `。！？?!` 拆分多个复合问题子句；保留原问题并限制变体数量，不增加依赖或论文专用规则。
- 为中文统计/标注复合问题补充通用英文检索别名和来源内证据触发词；在 472 块隔离库、假客户端的真实应用路径检查中，
  `mtraig-g-03` 从 `0/12` 到 `12/12`、`mtraig-g-05` 从 `0/5` 到 `5/5`；20 题上下文覆盖由 `11/20 full、7/20 partial、2/20 zero`
  变为 `13/20 full、7/20 partial、0/20 zero`（macro/micro=`0.8592/0.8824`），未宣称答案语义已重新验证。
- 答案审计允许带不同 `repeat` 值的生成 trace 直接输入，同时仍拒绝无 repeat 的重复 case。
- 新增对应回归测试；全套离线 unittest `244/244`、基准校验、`py_compile` 和 `git diff --check` 通过。本步未重建项目
  Chroma、未启动网页、未调用 DeepSeek、未提交或推送。

### 2026-09-06：Phase H2 定向生成闸门

- 在同一 472 块隔离库、默认 Hybrid、来源过滤和 H1 代码上，对 6 道历史失败/部分题及 2 道正确控制题各生成两轮；`16/16` 成功，provenance 完整，配置/上下文/metadata 按 case 均稳定。
- 逐题语义复核（两轮结论一致）：目标题 `2 correct / 3 partial / 1 incorrect`；控制题 `2/2 correct`。`mtraig-g-03`、`mtraig-g-05` 已恢复正确，但 `mtraig-g-04`、`wikimix-g-04/05` 仍部分缺失，`tableeval-g-01` 仍错误。
- 因未达到至少 `5/6` 个目标题正确的 H2 门槛，H3 未启动；trace 仅保存在仓库外 `/private/tmp/scirag_phaseH2_answers_v1.jsonl`（SHA-256 `ed43b0b575d728c7a8e4247fbf2a8318bf505c6eadb91e0c39ff825e90876d87`）。本步未运行 RAGAS、未写项目 Chroma、未提交或推送。

### 2026-09-06：Phase H2b 修复与 H3 确认闸门

- 为普通表格上下文附加 caption，并补充构建、来源、标注、数据子集等通用中英文检索别名；H2 目标题复测为 `5 correct / 1 partial`，控制题 `2/2 correct`，无错误答案。
- 新增两篇此前未进入任何基准的 2025 ACL 免费论文（MEBench、医学问答解释基准）及 12 道冻结题；PDF 清单校验通过，隔离 Chroma 共 304 块。
- H3 离线前置覆盖为 `11/12 full、1/12 partial、0/12 zero`，来源路由 `12/12`；两轮 DeepSeek `24/24` 成功，配置、上下文和 metadata 稳定。
- H3 人工复核每轮 `8 correct / 3 partial / 1 incorrect`，未达到 `≥85% correct 且 ≤1 incorrect`。失败集中在 MEBench Table 3 跨列表头拆分、Medbullets 选项数证据未召回和 Table 2 多列返回遗漏 GPT-4/MedQA-4 值；新增通用修复前不得改 H3 题目。
- 逐题复核写入 `evaluation/benchmark/reviews_phaseH3_confirmation_v1.jsonl`；生成 trace 仅保存在仓库外 `/private/tmp/scirag_phaseH3_answers_v1.jsonl`（SHA-256 `34fc443917860acde0f8583ec56fcf72f04cf22bc027abec79edc1d6cdc00914`）。本步未运行 RAGAS、未写项目 `chroma_db`、未提交或推送。

### 2026-09-06：Phase H4 通用表格与选项证据修复

- 修复跨单元格/空首列表头、重复数据集行、`X→Y/X→RY` 垂直分组和“在 A、B、C 上分别”限定解析；新增“选项”中英检索别名。
- 两篇 H3 论文重建的 304 块临时库中，Table 3 的 `GPT-4 + RAG` 四列、Table 2 X→Y 的 GPT-4 三个数值和 Medbullets“五个选项”证据均进入实际送模上下文；全套 unittest `248/248`、编译和 diff 检查通过。
- 两轮 DeepSeek `24/24` 成功、来源与配置稳定；独立复核为 `11 correct / 1 partial / 0 incorrect`，达到 H4 开发回归门槛。trace SHA-256：`39dd5acf493aea92728de4f15a9f3c1a7a15bdadf3d780f118a7c77710a7a561`；复核记录见 `evaluation/benchmark/reviews_phaseH3_confirmation_h4_v1.jsonl`。
- H3 仍是开发回归集，不能证明未见泛化；下一步应新建未参与修复的 H5 确认集。未修改项目 Chroma、未运行 RAGAS、未提交或推送。

### 2026-09-06：Phase H5 隔离生成与验收

- 经用户批准使用 ACL Anthology 的 SciAssess（Findings of NAACL 2025）和 YESciEval（ACL 2025）两篇免费 PDF；新增 12 道题，清单校验通过。
- H5 生成前已用于离线检索诊断和通用表格/路由修复，故按冻结协议改标为 `development-regression`；题目和金标准未改动。
- 两篇论文在仓库外隔离 Chroma 共 432 块；真实应用路径来源路由 `12/12`，送模上下文 required-fact 覆盖 `12/12 full`（macro/micro=`1.0000/1.0000`）。
- 两轮 DeepSeek `24/24` 成功，配置/context/provenance 按 case 稳定。人工复核第一轮 `12 correct / 0 partial / 0 incorrect`，第二轮 `10 correct / 2 partial / 0 incorrect`；按 case 两轮均完整为 `10/12`，未达到 `≥85%` 未见确认门槛。严格词面答案审计 macro/micro=`0.6606/0.7073` 仅作表面信号。
- 新增 `reviews_phaseH5_confirmation_v2.jsonl`；trace 只保存在仓库外 `/private/tmp/scirag_phaseH5_answers_v2.jsonl`（SHA-256 `9df5a445fce088b5d7d09a0a2d9647d5e5bbfa9f0f5bca851d6486d3ac35ffc9`）。不宣称泛化、不切换线上默认检索，下一步另建真正未参与诊断的 H6 确认集。

### 2026-09-06：Phase H6 初始冻结，后转开发回归

- 选用此前未进入任何基准的 ACL 2025 CAQA 和 COLING 2025 MiMoTable 两篇免费 PDF；完成页数、文字层、关键表格/图和方法段落核对。
- 新增 `manifest_phaseH6_confirmation.json` 与 `cases_phaseH6_confirmation.jsonl`，共 12 道题；离线清单校验通过，required facts 均由 gold contexts 或显式别名支持。
- H6 初始在任何检索、生成或调参前冻结为 `frozen-unseen`；cases SHA-256=`c8976fc51e4fae8d7ae4f0504b01cded8e8caca9c98b67e978e571a558ef8a5a`。随后隔离检索闸门暴露通用质量控制别名缺口，按冻结协议将 manifest 改标为 `development-regression`（当前 SHA-256=`14e036d7b28c71bd9a43556da03208a545653a189ea7d3069597f09a15ef135c`），另建 H7 才能再次做未见确认。
- 修复两处通用问题：来源局部质量控制别名增加“质检/质量控制”等中英文术语；required-fact 规范化兼容换行断开的 `independent - facts`。隔离 256 块 Chroma 的真实应用路径 gate 从初始 `10/12 full` 提升为 `12/12 full`，未修改项目 Chroma。
- H6 两轮 DeepSeek 回归 `24/24` 成功，来源、配置和 provenance 稳定；人工复核为 `20 correct / 2 partial / 2 incorrect`（按 case 两轮均完整 `10/12`）。错误集中在 MiMoTable Table 4 遗漏 Simple=33.6% 和 Figure 6 将 Lookup/Compare/Visualize 数值错配；不宣称泛化、不切换线上默认检索。trace 仅保存在仓库外 `/private/tmp/scirag_phaseH6_answers_regression_v1.jsonl`（SHA-256 `8e024cb9b6ad0ad59e08a976ef5bc5da9bbebba897d13de497cd77c8cc8c178a`），复核记录见 `reviews_phaseH6_regression_v1.jsonl`。

### 2026-09-06：Phase H7 初始冻结，后转开发回归

- 选用 ACL 2025 Long Papers 的 TC–RAG（[官方页面](https://aclanthology.org/2025.acl-long.558/)）和 ChartCoder（[官方页面](https://aclanthology.org/2025.acl-long.363/)）两篇此前未进入基准或诊断的免费 PDF；前者覆盖 RAG 状态/记忆/公式/效率，后者覆盖图表多模态、代码生成和图表统计。
- 新增 `manifest_phaseH7_confirmation.json` 与 `cases_phaseH7_confirmation.jsonl`，共 12 道题；PDF 页数、文字层、Table 1/2/3/4/7、Figure 2 及方法公式已完成离线核对，清单校验通过。
- H7 初始在任何检索、生成或调参前冻结为 `frozen-unseen`；随后 context gate 发现通用表头、整行多列、公式证据和多事实召回缺口，按协议改标为 `development-regression`（当前 manifest SHA-256=`f92c5082083b1c736dfb40deac3c1c80508ebf23011e3fa12710e565b4e390b7`）。cases SHA-256=`e3ad5c19ab528ba8bb1b2d607e3b69acb479947c0def20b20a6e0574c8a6d290`。论文仅保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`。
- H7 隔离库共 370 块；真实应用路径 required-fact gate 为 `5/12 full`，fact macro/micro=`0.6457/0.6705`。尚未调用 DeepSeek；修复完成后另建 H8 才能重新进行未见确认。
- 随后补充通用三层/居中表头合并、中文平均指标别名和“分别”多列行抽取；TC–RAG Table 2 已能直接返回四个平均值。H7 fix2 离线 gate 为 `5/12 full`、fact macro/micro=`0.6576/0.6818`，仅作开发回归诊断，未调用 DeepSeek，后续仍需另建 H8。

### 2026-09-06：Phase H8 初始冻结，后转开发回归

- 选用 ACL 2025 Long Paper ChainRAG 与 ACL 相关 MAGMaR 2025 两篇免费 PDF，均保存于仓库外；新增 `manifest_phaseH8_confirmation.json` 与 `cases_phaseH8_confirmation.jsonl`，共 12 道题。PDF 页数、关键表格和流程段落完成文字与视觉核对，清单校验通过。
- H8 初始在检索前冻结为 `frozen-unseen`；隔离 158 块 Chroma 的真实应用路径 gate 为 `6/12 full`，fact macro/micro=`0.7449/0.7500`。缺口集中在多事实段落、跨列统计表和模态流程细节，按协议现改为 `development-regression`；未调用 DeepSeek、未修改项目 Chroma，后续需另建 H9 才能重新获得未见确认。

### 2026-09-06：Phase H8 通用缺口修复回归

- 修复多级表头续行被当作实体、嵌套方法行丢失外层模型、PDF 阈值括号/小数空格，以及来源内量化/设置/模态分布别名；H8 required facts 同步改为可由论文原文逐项核验的原子事实。
- 本次 H8 文件 SHA-256：manifest=`fb1a4b69d1ceda09f56203269fe5d367ff450ad05b54429955914c87d936019b`，cases=`ecc61faad0c583e9debd2133935cfc0a280e4b8b77e3c514836cc277f9ff3d66`。
- 在同一 158 块隔离库的真实 `query_knowledge` 路径（假客户端、来源过滤、未调用 DeepSeek）复测，12/12 题事实覆盖完整，macro/micro=`1.0000/1.0000`；ChainRAG Table 1 现在保留 `GPT4o-mini / Ours (CxtInt)` 外层行限定，MAGMaR 阈值与模态数量均可核验。
- 本次 H8 仍是开发回归，不证明未见泛化，也不改变线上默认检索；下一步建立不参与修复的新 H9 确认集。未修改项目 `chroma_db`、未提交或推送。

### 2026-09-07：H3–H8 当前版本全库回归矩阵

- 使用当前代码、统一 Hybrid+RRF、文档路由、查询分解、表格/图形/公式/限制证据保护、相邻块和 parent-window，在所有论文共同索引、`@10`、不传 `source_filter` 的路径上重跑 H3–H8；未调用 DeepSeek，未写项目 `chroma_db`。完整结果仅保存在仓库外 `/private/tmp/scirag_phaseH3_H8_regression_matrix_20260907.json`。
- 结果为：H3 `10/12 full`（fact macro/micro=`0.9167/0.9180`）；H5 `8/12`（`0.7222/0.7561`）；H6 `9/12`（`0.9352/0.9348`）；H7 `4/12`（`0.6122/0.6364`）；H8 `7/12`（`0.8354/0.8646`）。所有批次目标文档命中率均为 `1.0000`，路由无错误；缺口主要是未指明论文时的多事实、复杂表格和公式证据分散。
- 该矩阵与 H5/H8 记录的来源过滤结果不是同一指标：后者验证“已知目标论文内的隔离检索”，本矩阵验证“普通多论文全库检索”。后续报告必须同时注明是否使用 `source_filter`，不得把来源过滤结果当作普通网页泛化证据。
- 全量离线 unittest `255/255`、`py_compile` 和 `git diff --check` 继续通过。下一步先固定两条评估口径并对 H7/H8 的缺口做一次根因分类，再决定是否建立 H9；不因单个新论文失败立即添加论文专用规则。
- 为核对上述原始检索代理与网页实际上下文的差异，使用当前 `app.query_knowledge`、同一批隔离库、全库检索且不传 `source_filter`，以假客户端禁用生成后复测：H3 `12/12 full`、H5 `12/12 full`、H6 `11/12 full`（1 partial）、H7 `5/12 full`（5 partial、2 zero）、H8 `12/12 full`。结果仅保存于仓库外 `/private/tmp/scirag_phaseH3_H8_app_global_context_matrix_20260907.json`。
- 因此 `benchmark_retrieval.py` 的较低数字是“原始 top-k 片段事实覆盖”代理，不应直接当作网页送模上下文；但 H7 的复杂表格/公式缺口在真实应用路径同样复现，不能归咎于评估器。下一步只分析 H7 的共享根因并设定停损点，H9 暂缓。

### 2026-09-07：H7 conditional-perplexity 公式意图修复

- H7 的 `tcrag-h7-04` 问法使用“定义 conditional perplexity 和 uncertainty”，原公式意图识别未触发公式证据通道；补充通用中英术语与 `cppl/uct` 别名，并提高被问题明确点名的公式缩写排序权重。未加入 TC–RAG 专用分支。
- 新增公式闸门回归测试；全量离线 unittest `256/256`、`py_compile` 和 `git diff --check` 通过。使用 H7 临时库、假客户端重测后，`tcrag-h7-04` 从 zero 提升为 partial，已召回 cppl/uct 公式片段；由于 PDF 公式被拆成多个文字块，仍缺少部分公式行，暂不继续在 H7 上追加特例。
- H7 当前应用路径汇总为 `5 full / 6 partial / 1 zero`，fact macro=`0.6854`；ChartCoder 六题中五题 full，TC–RAG 的复杂表格和多事实题仍是主要缺口。
- 本步未调用 DeepSeek、未写项目 `chroma_db`、未提交或推送；H7 仍为开发回归集。

### 2026-09-07：覆盖审计识别结构化/Markdown 表格行

- `evaluation/context_coverage.py` 现在在不合并不同表行的前提下识别逐行 `字段=值`、冒号/分号分隔和 Markdown `|` 单行；同时兼容 `Avg. Time (s) 50.91` 这类单位位于数值前的 PDF 表格形式。
- 新增结构化行、Markdown 行及跨行防串配回归测试；全套离线 unittest `257/257`、`py_compile` 和 `git diff --check` 通过。
- H7 原始 Hybrid 诊断（全库、全部保护、@10）由此前 `4/12 full`、macro/micro=`0.6122/0.6364` 变为 `5/12 full`、`7/12 partial`、`0/12 zero`，macro/micro=`0.7420/0.7500`。这仍是原始 top-k 片段代理，不是答案正确率。
- 同一 H7 临时库的真实 `app.query_knowledge` 有效上下文复测为 `7/12 full`、`5/12 partial`、`0/12 zero`；此前被误记的 TC–RAG Table 2 四个值和 ChartCoder Table 1 五行现可被覆盖审计识别。TC–RAG 多事实/公式题仍为 partial，未添加论文专用规则。
- 本步未调用 DeepSeek、未写项目 `chroma_db`、未提交或推送；H7 仍为开发回归集。下一步应冻结当前 H3–H8 代码口径，另建 H9 未见确认集，不再继续为 H7 追逐局部提升。

### 2026-09-07：H7 最后一轮通用表格与多事实路由修复

- 表格限定词现在只有在匹配实际 `Dataset` 单元格时才参与行过滤，避免把 `CMB/MMCU/CMB-Clin` 这类横向表头误当成数据集行；补充通用“评价指标/加速策略”及 `metrics/acceleration strategies` 检索别名。
- 新增两条通用回归测试；全套离线 unittest `259/259`、`py_compile` 和 `git diff --check` 通过。
- H7 同一隔离库、Hybrid+路由+查询分解+可选证据保护、假客户端的真实 `query_knowledge` gate 为 `8/12 full、4/12 partial、0/12 zero`，fact macro/micro=`0.9000/0.8750`；TC–RAG Table 1 已恢复完整，ChartCoder 六题保持完整。该结果仅证明送模上下文事实覆盖，不是答案语义准确率或泛化证据。
- H7 仍保留为 `development-regression`；达到停损点后不再加论文专用规则。下一步冻结 H3–H8 配置对照，并建立未参与修复的 H9 确认集。

### 2026-09-07：H3–H8 配置 A/B 验收

- 固定各批次现有隔离库、同一离线 `bge-small-zh-v1.5`、`context_k=10` 和假客户端，A 为默认 Dense（路由/查询分解关闭），B 为 Dense+文档路由+查询分解；A/B 各重复 3 次，逐次结果一致。未调用 DeepSeek、未写项目 `chroma_db`。
- B 的 full 案例数相对 A：H3 `9→11`、H5 `7→9`、H6 `8→11`、H7 `7→8`、H8 `7→8`；每个批次 required-fact macro/micro 均提升，目标论文命中均 `12/12`，显式 Table N 命中均保持 `100%`。
- 页级命中 H3/H5/H6 改善，H7 持平，但 H8 从 `12/12` 降至 `10/12`。因此严格“事实、页级、表号均不下降”的默认切换门槛未通过；线上默认继续 Dense，路由+查询分解仅保留 opt-in。Hybrid 另作参考，不因本次 A/B 切换默认。

### 2026-09-07：H7 两轮生成语义闸门与 provenance 审计

- 在同一 370 块 H7 隔离库、当前 Hybrid+路由+查询分解配置下完成 12 题×2 轮 DeepSeek 生成；首次沙箱网络中断的失败行在获准联网后续跑，最终 `24/24` 成功。trace 仅保存在仓库外 `/private/tmp/scirag_phaseH7_answers_v2.jsonl`，SHA-256=`01d39aedb506314e3bfa9f5c0a1981ae2c47c030d8c64eb845023340f70cee53`。
- provenance 审计为 12/12 case 完整；两轮 runtime config、context IDs、metadata 和 source fingerprint 均稳定，7/12 case 答案措辞发生变化。严格词面事实审计 macro/micro=`0.6733/0.6534`，仅作表面信号。
- 逐题人工核对写入 `evaluation/benchmark/reviews_phaseH7_generation_v2.jsonl`：第 1 轮 `5 correct / 6 partial / 1 incorrect`，第 2 轮 `5 correct / 5 partial / 2 incorrect`；保守按 case 为 `5 correct / 5 partial / 2 incorrect`，未达到每轮至少 `10/12 correct、错误不超过 1` 的 H7 生成闸门。失败集中在 TC–RAG 公式/状态语义、多数据集指标完整性，以及 ChartCoder 多行/多列表格答案截断。
- H7 继续标为 `development-regression`，不宣称泛化，不切换线上默认，不再追加论文专用修复。下一步仅建立真正未参与修复的 H9；下载新论文前需先审阅候选及其免费来源。

### 2026-09-07：Phase H9 首轮确认门禁

- 经用户批准，从 ACL Anthology 下载 LongTableBench、Table-R1 和 Query-Driven Multimodal GraphRAG 三篇免费 2025 PDF；PDF 只保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`，逐页文字层和代表页视觉核验完成。三篇文件 SHA-256 分别为 `0de0b2f87eefff392a19ceb19bbe0b52124db13d79c2664fe9615586cc9a95a7`、`ac314de535c21e9f31e0016267499ec2cac6308556d14e86a9fa68f61acedf6a`、`4b5c4f44d5bb9dd32f0246b1c1b2e03ba73442edfd7b931d3db07900c0720932`。
- 新增 `evaluation/benchmark/manifest_phaseH9_confirmation.json` 与 `cases_phaseH9_confirmation.jsonl`，共 12 道题；初始 required-fact、来源页和上下文校验通过。
- 在仓库外重建 475 块隔离 Chroma，使用当前真实 `app.query_knowledge`（Hybrid、文档路由、查询分解、parent-window、公式/图形证据保护、单来源过滤、假客户端）复测；送模上下文 gate 为 `6/12 full、5/12 partial、1/12 zero`，required-fact macro/micro=`0.7593/0.8052`。不传 `source_filter` 的全库结果相同；未调用 DeepSeek，未写项目 `chroma_db`。
- 失败归因于共享缺口：多列整行表格和确定性单元格抽取截断、无题注编号表格的 Table N 定位、跨块公式尾部条件、多事实段落召回；不是三篇论文专用事实问题。按冻结协议将 H9 manifest 改标为 `development-regression`，不再宣称未见泛化，不对 H9 追加特例修复。

### 2026-09-07：H9 通用解析与覆盖审计回归

- 修复堆叠/远距离表头关联、`<br>` 并行单元格对齐、显式多列查询的限定词裁剪、PDF 公式续行恢复，以及同一结构化行内的有序事实匹配；未加入论文专用规则。
- 新增回归测试；全套离线 unittest `264/264`、`py_compile` 和 `git diff --check` 通过。
- 在仓库外 `/private/tmp/scirag_phaseH9_db_fix2_20260907` 重建 `476` 块，以假客户端运行当前真实 `app.query_knowledge`；来源过滤和全库路径均为 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`。
- H9 仍保留为 `development-regression`，结果只证明送模上下文事实覆盖，不证明答案语义正确或未见泛化；本步未调用 DeepSeek、未修改项目 `chroma_db`、未提交或推送。

### 2026-09-07：H9 两轮生成回归

- 在 `/private/tmp/scirag_phaseH9_db_fix2_20260907` 上完成两轮 DeepSeek 生成，`24/24` 成功；trace 位于仓库外 `/private/tmp/scirag_phaseH9_answers_regression_20260907.jsonl`，SHA-256=`7890f6bff824764016ade60cede8be8dea37883bead21fb1e3de31107809b097`。
- 人工对照 PDF、gold 和 contexts 为 `11 correct / 1 partial / 0 incorrect`；partial 是 GraphRAG Definition 4 题的不必要拒答及 G2 公式不完整。词面答案审计 macro/micro=`0.8536/0.8571`，仅作表面信号；送模 contexts 保持 `1.0000/1.0000`。
- H9 已是 `development-regression`，不用于未见泛化结论；本步未修改项目 `chroma_db`、未提交或推送。

### 2026-09-07：Phase H10 新论文检索门禁

- 经用户批准下载三篇 ACL Anthology 免费 2025 PDF：SCITAT、REAL-MM-RAG、RealHiTBench；完成页数、SHA-256、文字层和关键表格视觉核对。新增 H10 manifest/cases，共 12 道题，初始冻结为 `frozen-unseen`。
- 在仓库外隔离库运行当前 Hybrid+文档路由+查询分解+结构化表格/公式/图形证据保护+parent-window；@50 目标文档命中 `12/12`、路由 `10/10`，但 required-fact `8/12 full`，macro/micro=`0.694/0.658`。失败集中在宽表统计和多列值覆盖。
- 按冻结协议将 H10 改标为 `development-regression`；本步未调用 DeepSeek、未写项目 `chroma_db`、未提交或推送。后续若继续修复，先建立新的 H11 未见确认集。

### 2026-09-07：H10 表格标签与并排表解析修复

- 公共解析器新增字母数字表号支持（例如 `Table S1`），保留原有整数 `table_number` 并增加 `table_label`；显式表号匹配和评测表号命中均改为按标签比较。
- 对 `pymupdf4llm` 将两个横向表合并为一个宽 GFM 表的情况，按重复首级表头和对齐列通用拆分；H10 SCITAT 第 4 页现在生成独立 Table 3 与 Table 4，且表格不再重复进入普通文本块。
- 新增两条解析回归测试；全量离线 unittest 当前为 `266/266`，`py_compile` 与 `git diff --check` 通过。三篇 H10 PDF 的公共加载路径均能识别独立表块，REAL-MM-RAG 的 `Table S1` 元数据标签为 `S1`。
- 在仓库外临时隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 复测真实 `app.query_knowledge`（Hybrid、路由、查询分解、parent-window、假客户端）为 `9/12 full、1/12 partial、2/12 zero`。SCITAT Table 3/4 与 REAL-MM-RAG Table S1 已进入上下文；剩余缺口是 H10 用例的事实字符串与表格“行标签/列值”顺序不一致，以及 RealHiTBench 统计题缺少表格结构别名，属于评测匹配口径问题，尚未调用 DeepSeek。
- 本步未写项目 `chroma_db`、未提交或推送。下一步先修正 H10 事实审计的通用表格行列匹配，再决定是否进行生成验证；不添加论文专用规则。

### 2026-09-07：H10 表格事实审计口径修正

- 事实覆盖审计新增通用 Markdown 表格关系匹配：在单行内关联行标签、列标题和单元格值，兼容 PDF 去掉千位逗号及 `Table`/`Tables` 单复数差异；不跨表行拼接，也不修改任何 H10 用例。
- 新增两条审计回归测试；全量离线 unittest 当前为 `268/268`，`py_compile` 与 `git diff --check` 通过。
- 复用临时隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 重跑实际 `app.query_knowledge` 上下文路径后，H10 `12/12 full`，required-fact macro/micro=`1.0000/1.0000`，无 partial/zero。该结果仍属于开发回归，不证明未见泛化；本轮未调用 DeepSeek、未写项目 `chroma_db`、未提交或推送。
- 同一批次使用 `evaluation/benchmark_retrieval.py` 的当前 Hybrid、路由、查询分解、表格/公式/限制证据保护和 parent-window 配置重跑离线检索，@50 目标文档、来源页、表号和 required-fact 均为 `12/12`，macro/micro=`1.0000/1.0000`；@10 仍为 `9/12 full`，不把低 k 结果隐藏。

随后对 H10 的两道表格题和 H9 的一道公式题进行了定向 DeepSeek 生成复核。SCITAT Table 4 与 REAL-MM-RAG Table S1 答案正确；GraphRAG Definition 4 题虽已检索到跨块公式证据，模型仍不稳定地拒答并混排不完整公式。已补充通用的相邻公式拼接提示和多字符公式标签识别，单次重试仍未达到完整、可核验的答案，因此将其归类为当前生成/公式 provenance 边界，不继续增加论文专用规则或重复消耗 API。H10 尚未进行全量生成闸门。

### 2026-09-08：H10 单轮生成回归

- 在仓库外隔离库 `/private/tmp/scirag_phaseH10_parserfix_m060cn06` 上完成 H10 12 题单轮 DeepSeek 生成，`12/12` 调用成功、无 API 错误；trace 位于 `/private/tmp/scirag_phaseH10_answers_20260908.jsonl`，SHA-256=`935ae355f9c58194aab24f0208cb5dc54b944a1528cb2fe9d2abb8b4fc0d1e07`。
- 人工对照 PDF、gold 和实际 contexts：`10 correct / 1 partial / 1 incorrect`。partial 为 SCITAT 四类 reasoning type 漏列三个总类；incorrect 为 RealHiTBench Table 2 的模型/行列定位错误。逐题记录见 `evaluation/benchmark/reviews_phaseH10_generation_v1.jsonl`。
- 该结果达到“单轮至少 10/12 正确且 incorrect 不超过 1”的开发回归门槛，但 H10 已参与修复，仍不构成未见泛化证据；H10 保持 `development-regression`，不切换线上默认检索。

### 2026-09-08：H10 RealHiTBench 表格行列定位修复

- 修复共享 Markdown 表格解析与查询路径：稀疏分组行不再误判为叶表头，拆分的 `Numerical Reasoning` 表头可正确合并；同一行的模型与 `Input/Modality/Mode` 限定词不再互相误选，并在请求单一 `F1` 指标时过滤 sibling metric 列。
- 新增 H10 风格回归测试；直接解析外部 RealHiTBench PDF Table 2 可稳定返回 `GPT4o(TreeThinker)` 的 `Image+Text` 行及 `73.32/64.28/77.42` 三个 F1 值。
- 全量离线 unittest `270/270`、`py_compile`、`git diff --check`、benchmark manifest 校验和 Gradio Blocks 构建冒烟通过；仅有 Gradio `theme` 参数迁移 warning。
- H10 继续标为 `development-regression`；未调用新 API、未修改项目 `chroma_db`、未切换线上默认，也未开始 H11。

### 2026-09-08：Phase H11 新论文确认集首轮门禁

- 按冻结口径新增两篇此前未进入任何基准或修复的公开论文：IRPAPERS（arXiv:2602.17687，24 页）和 PDF Retrieval Augmented Question Answering（arXiv:2506.18027，14 页）。PDF 保存在仓库外 `/Users/qinleqi/Desktop/sci-rag-benchmark-papers/`，SHA-256 已写入 `evaluation/benchmark/manifest_phaseH11_unseen.json`。
- 新增 12 道 H11 用例（每篇 6 题），覆盖数据集统计、检索对照、宽表、多模态 PDF 管线和结果表；冻结前通过 `validate_benchmark.py --verify-files --require-complete`。
- 首轮隔离 Hybrid（文档路由、查询分解、结构化表格/公式/限制/图形证据保护、parent-window）目标文档/来源页/表号均为 `12/12`；但 @50 required-fact 仅 `4/12 full`，macro/micro=`0.350/0.300`，失败集中在宽表和表格字段覆盖。诊断 JSON 仅保存于仓库外 `/private/tmp/scirag_phaseH11_retrieval_20260908.json`。
- 根因分析发现事实审计遗漏“行标签+列头+值”顺序；在 `evaluation/context_coverage.py` 增加一条通用组合并补充回归测试，全量离线 unittest 为 `271/271`。同一配置重跑后 H11 @50 提升至 `5/12` 完整、macro/micro=`0.517/0.457`，但仍有 7/12 题不完整，未达到确认门槛；结果保存于 `/private/tmp/scirag_phaseH11_retrieval_fix1_20260908.json`。
- 按冻结协议 H11 保持 `development-regression`；不在 H11 上继续调参、不调用 DeepSeek 生成、不修改项目 `chroma_db`。下一步另建 H12 未见确认集。

### 2026-09-09：H11 事实审计有效性修复与复测

- 修复 `evaluation/context_coverage.py` 的通用 Markdown 表格事实匹配：不再丢弃含数字的行标签，并按当前值之前的单元格前缀匹配复合行标识；同时将等价的小数尾零（例如 `0.1910` 与 PDF 提取的 `0.191`）纳入可审计形式。
- 新增数字模型名、参数化行、多级模型行和小数精度差异的回归测试；全量离线 unittest `275/275`，`py_compile` 与 `git diff --check` 通过。
- 使用与 H11 首轮完全相同的 Hybrid、路由、查询分解、结构化表格/公式/限制/图形证据保护和 parent-window 配置重跑：@50 目标文档、来源页、表号和路由均为 `12/12`；required-fact `11/12 full`，macro/micro=`0.933/0.943`，7/7 表格题全部完整，唯一缺口为 PIER-QA 多步骤 pipeline 证据题。
- 结果保存于仓库外 `/private/tmp/scirag_phaseH11_retrieval_fix3_20260909.json`。H11 继续标记为 `development-regression`；不在该集合上继续调参或调用 DeepSeek，下一步进入回归检查点和新的 H12 未见确认集设计。

### 2026-09-09：Phase H12 未见集冻结与首轮诊断

- 新增两篇此前未出现在任何基准清单的公开论文：T²-RAGBench（arXiv:2506.12071，27 页）和 mmRAG（arXiv:2505.11180，19 页），PDF 保存在仓库外论文目录，SHA-256 已写入 `evaluation/benchmark/manifest_phaseH12_unseen.json`。
- 新增 12 道冻结用例（每篇 6 题），覆盖数据统计、表格行列值、检索配置、标注协议和评估设置；冻结前通过 `validate_benchmark.py --verify-files --require-complete`，未运行 H12 检索或生成诊断。
- 使用与 H11 相同的 Hybrid、路由、查询分解、结构化表格/公式/限制/图形证据保护和 parent-window 配置完成首轮隔离检索：@50 目标文档、来源页、表号均为 `12/12`，但 required-fact `7/12 full`，macro/micro=`0.764/0.763`，full/partial/zero=`0.583/0.333/0.083`。
- 失败集中在表格行标签/列值和自然语言事实顺序的表面对齐，另有一处 pipeline 证据排名问题；结果保存于仓库外 `/private/tmp/scirag_phaseH12_retrieval_v1_20260909.json`。按冻结协议 H12 已改标为 `development-regression`，不在该集合上调参、改题或调用 DeepSeek。

### 2026-09-09：新增 PDF 解析证据前置门禁

- `evaluation/validate_benchmark.py` 新增 `--verify-extracted-evidence`：按每道题的 `source_pages` 加载真实 PDF 解析输出，并用同一事实审计器检查 required facts；缺失时在检索前失败。
- 新增两条门禁单元测试；全量离线 unittest `277/277` 通过。
- 对 H12 回放该门禁时，6 道题在检索前暴露实际解析表面与 gold fact 不一致，验证了 H12 的失败主要混入了标注/审计对齐问题；H12 不重写、不调参，保留为开发回归。

### 2026-09-09：Phase H13 解析门禁通过与一次性检索验收

- 新增两篇此前未进入任何基准或修复的公开论文：BRIGHT（arXiv:2407.12883，51 页）和 G-Retriever（arXiv:2402.07630，23 页），共 12 道表格事实题；真实 PDF 解析证据前置门禁全部通过后，manifest 冻结为 `frozen-unseen`。
- 使用与 H11/H12 相同的 Hybrid、文档路由、查询分解、结构化表格/公式/限制/图形证据保护和 parent-window 配置完成一次性隔离检索：@10、@50 均为目标文档/来源页/表号 `12/12`，required-fact macro/micro=`1.000/1.000`，`12/12 full`，无 partial/zero，路由 `12/12` 正确。
- H13 不再修改题目、解析器或检索参数；结果仅证明当前检索证据链在该未见集上通过，不自动扩展到生成语义正确率。诊断 JSON 保存在仓库外 `/private/tmp/scirag_phaseH13_retrieval_v1_20260909.json`；未调用 DeepSeek、未写项目 `chroma_db`、未提交或推送。

### 2026-09-09：Phase H13 单轮生成闸门

- 在仓库外隔离库 `/private/tmp/scirag_phaseH13_db_20260909`（420 块）上完成 H13 单轮生成；最初 11/12 成功，唯一失败为 `bright-h13-05` 网络连接错误，联网环境下仅补跑该 case 后达到 `12/12` 成功。
- 生成 trace 位于 `/private/tmp/scirag_phaseH13_answers_20260909.jsonl`，SHA-256=`f91250551eb55905f6f32df43aee3d745cdceb68ed94d332e66fd104c44464aa`；provenance 完整 `12/12`，配置和上下文稳定，无重复 case。
- 独立人工复核为 `10 correct / 2 partial / 0 incorrect`，达到预设生成闸门（至少 10 correct 且 incorrect 不超过 1）。partial 为 BRIGHT Table 2 一题遗漏两个请求值，以及 G-Retriever Table 11 一题遗漏 KAPING 值；未出现错误答案。逐题记录见 `evaluation/benchmark/reviews_phaseH13_generation_v1.jsonl`。
- H13 的生成语义结果达到开发闸门，但不单独宣称未见泛化；本步未修改项目 `chroma_db`，未提交或推送。

### 2026-09-09：H13 真实送模上下文差异归类

- 对 H13 生成 trace 的真实 `contexts` 做独立审计后，发现 `gretriever-h13-05` 只送入了 G-Retriever=70.49 的 Table 11 行，缺少题目要求的 KAPING=60.81；因此实际送模上下文是 `11/12 full`，macro/micro=`0.9583/0.9688`。
- 该差异没有触发题目或检索参数修改；按冻结协议，H13 manifest 已从 `frozen-unseen` 改为 `development-regression`。离线原始检索 12/12 与真实应用路径不一致，下一步只做一次通用根因分析，暂不继续生成或调参。

### 2026-09-09：表格行实体引用标记的通用修复

- 根因是表格行实体识别只接受 `实体名(...)`，无法识别 `KAPING [1] (top-k triple retrieval)` 这类“实体名 + 引用 + 括号说明”；真实 `query_knowledge` 因此只返回 G-Retriever 行。
- 将括号说明前的引用标记改为可选通用语法，并新增回归测试；在 H13 隔离库同一路径复测已同时返回 KAPING=60.81 与 G-Retriever=70.49。
- 全量离线 unittest `278/278`、`py_compile` 和 `git diff --check` 已通过；H13 继续保留为 `development-regression`，不因该修复重新宣称未见泛化，也不立即重跑生成。

### 2026-09-09：新增真实送模上下文门禁

- 新增 `evaluation/app_context_gate.py`：连接已构建的隔离 Chroma，以本地假客户端运行真实 `app.query_knowledge` 路径，并使用同一 required-fact 审计器检查最终 `contexts`；不会调用 DeepSeek。
- 新增最小缺失行回归测试和 README 命令。H13 修复后的隔离库门禁为 `12/12 full`、fact micro=`1.0000`，完整报告位于仓库外 `/private/tmp/scirag_phaseH13_app_context_gate_fix1_20260909.json`。
- 新的固定顺序为 PDF 解析门禁 → 检索诊断 → 真实送模上下文门禁 → 单轮生成 → 人工复核；任何前置门禁失败即停止，不消费生成 API。全量离线 unittest `279/279`、`py_compile` 和 `git diff --check` 通过。

### 2026-09-09：Phase H14 完整门禁与回答验收

- 新增此前未进入任何 manifest 的 BERGEN（Findings of EMNLP 2024，24 页）和 Benchmarking Retrieval-Augmented Generation for Medicine / MIRAGE（Findings of ACL 2024，19 页）；官方 PDF 保存在仓库外论文目录，SHA-256 已写入 H14 manifest。
- 12 道 H14 表格题在冻结前通过真实 PDF 解析证据门禁；冻结后的首轮 Hybrid 检索 @10/@50 均为目标文档、来源页、表号和 required-fact `12/12`，路由 `12/12` 正确。诊断保存于 `/private/tmp/scirag_phaseH14_retrieval_v1_20260909.json`。
- 在 257 块隔离 Chroma 上运行真实送模上下文门禁，`12/12 full`、fact micro=`1.0000`；报告位于 `/private/tmp/scirag_phaseH14_app_context_gate_v1_20260909.json`。
- 首次应用回答 trace 为 `12/12` 成功、provenance 完整；所有题均由确定性结构化表格路径直接回答，因此未实际调用 DeepSeek。人工复核为 `7 correct / 5 partial / 0 incorrect`，未达到至少 10 correct 的门槛；partial 均为多行或多列请求被裁剪。
- H14 已改标为 `development-regression`，不修改冻结题目、不在本轮修复或重跑。trace 位于 `/private/tmp/scirag_phaseH14_answers_v1_20260909.jsonl`，SHA-256=`cec3db7603c4ae84fb1685257a0261d3afbb574cca0488c5ef17d963f96f96fe`；逐题记录见 `evaluation/benchmark/reviews_phaseH14_generation_v1.jsonl`。
- 只读归因将 5 个 partial 收敛为两类共享缺口：`LoRa_r_`、`PubMedQA*`/`BioASQ-Y/N` 等带装饰符行标签的多行匹配，以及中文“原始文档数/平均长度/是否开源”到 `#Doc./Avg. L/Open` 的列意图映射。后续只允许各做一个通用修复和最小回归，不在 H14 上调参。

### 2026-09-09：H14 多行/多列表格回答通用修复

- 表格实体规范化现在将字母数字之间的 PDF/Markdown 下划线保留为词边界，使 `LoRa_r_` 可与问题中的 `LoRA r` 匹配；多行提取不再把已经选中的 Dataset 行实体重复当作全局数据集筛选条件。
- 通用列别名增加中文“原始文档数/文档数”“平均长度”“是否开源/开源”到 `#Doc.`、`Avg. L`、`Open` 的映射；没有加入 BERGEN/MIRAGE 专用规则。
- 复用原 257 块隔离库运行真实送模上下文门禁仍为 `12/12 full`、fact micro=`1.0000`；随后单轮确定性表格回答及离线答案审计为 `12/12 full`，答案与 contexts 的 fact macro/micro 均为 `1.0000/1.0000`。回答 trace SHA-256=`2d95c4279edd5ba49b19764c82f19159957308b279927f6d65268f3c423f6243`。
- H14 继续保持 `development-regression`，不恢复未见集身份；本轮未实际调用 DeepSeek、未修改项目 `chroma_db`、未提交或推送。

### 2026-09-10：Phase H15 最终一次性留出集冻结

- 从 ACL Anthology 官方来源选取此前未进入任何 manifest 的 RAGTruth（17 页）、MedExQA（15 页）和 M-LongDoc（18 页）；PDF 仅保存在仓库外论文目录，页数、SHA-256、文字层及关键表格/公式页面已核对。
- 新增 `manifest_phaseH15_unseen.json` / `cases_phaseH15_unseen.jsonl`，共 18 道题（每篇 6 题）；冻结前真实 PDF 解析证据门禁为 `18/18 full`，随后清单改标为 `frozen-unseen`。
- release-candidate 配置固定为 Hybrid、`retrieval_k=12`、`context_k=10`、文档路由、查询分解、parent-window、表格/公式/空间 Figure 证据，不使用 reranker 或视觉模型；这不改变产品 Dense 默认值。
- 停止规则固定为：门禁通过即结束本轮开发周期；任一门禁失败则记录边界并进入 backlog，不针对 H15 修复、不立即建立 H16，除非同一失败在至少两篇无关论文上复现。本步未运行 H15 检索、真实送模上下文或生成，未修改解析器、项目 Chroma，也未提交或推送。

### 2026-09-10：H15 一次性检索门禁停止

- 在仓库外 `/private/tmp/scirag_phaseH15_db.kTMhzy` 用冻结配置重建 3 篇 PDF 的独立 Chroma，共 297 块；首次模型加载的联网探测已中止，随后使用 `HF_HUB_OFFLINE=1` 从本地缓存成功完成索引。
- 按固定 Hybrid、`@10`、文档路由、查询分解、结构化表格/限制/公式/空间 Figure 证据和 parent-window 配置完成一次性检索；目标文档命中 `18/18`，路由正确 `17/17`，但 required-fact 为 `14/18 full`、macro/micro=`0.831/0.843`，缺口为 `ragtruth-h15-01`、`ragtruth-h15-03`、`medexqa-h15-06`、`mlongdoc-h15-02`。
- 诊断报告位于仓库外 `/private/tmp/scirag_phaseH15_retrieval_20260910.json`。按 H15 停止规则，本轮不进入真实送模上下文或生成，不修改 H15 清单、解析器或检索参数；4 个缺口进入 backlog，未提交或推送。

### 2026-09-10：H15 缺口跨阶段收束审计

- 只读对照 H7、H11、H12 的历史失败报告：虽然“多事实题在有限 top-k 下覆盖不足”是共同表象，但 H15 的 RAGTruth 标注协议、M-LongDoc 训练语料/证据页和 MedExQA 限制未来工作分别属于不同证据形态；历史缺口也分别涉及流程块、统计表和表格/公式。
- 当前没有足够证据把它们归因到同一个可安全共享的解析器或检索根因，因此不启动共享修复，不重跑 H15，不建立 H16；H15 保持冻结留出结果并继续作为 backlog 边界样本。未提交或推送。

### 2026-09-10：评测 runner 配置契约硬化

- `evaluation/benchmark_retrieval.py` 新增严格的 `release_candidate_config` 校验和 `--enforce-manifest-config`；它会在加载 embedding 模型和解析 PDF 前拒绝缺少 runner 控制项或 CLI/manifest 漂移，并在 JSON 报告记录 manifest 配置及是否强制校验。
- 新增两条配置契约回归测试，针对性 benchmark retrieval 测试 `47/47`、`py_compile` 和 `git diff --check` 通过。H15 历史诊断未重跑；其 manifest 缺少新增的 runner-only 控制项，继续按原始报告解释。
- 本步未修改产品默认 Dense 配置、未修改冻结 H15 清单、未调用生成 API、未提交或推送。

### 2026-09-10：历史清单稳定性检查点

- 使用当前 `validate_benchmark.py --verify-files --verify-extracted-evidence --require-complete` 复核 H11–H15：H13、H14、H15 全部通过；H11/H12 按历史已知问题分别暴露旧 required facts 与当前 PDF 解析表面的不一致。
- H11/H12 继续保持 `development-regression`，不重写旧题、不为历史门禁失败追加解析器特例；今后只有新清单在冻结前通过真实 PDF 证据门禁，才进入一次性检索。

### 2026-09-12：默认 Dense UI 发布检查点

- 在仓库外临时目录复制项目 ChromaDB，关闭路由、查询分解、parent-window、空间 Figure、公式和视觉开关，以默认 Dense 配置运行 `app.create_runtime()` 与 `app.build_demo()`；结果为 `UI_SMOKE_OK`，集合 104 个块，Gradio `Blocks` 构建成功。
- 使用仅用于冒烟的占位 API key，未执行查询或生成 API，项目 `chroma_db` 未写入；仅出现 Gradio 主题参数迁移警告。工作区审计通过，未提交或推送。

### 2026-09-12：去除评测过度防御

- 删除未用于产品决策的 RAGAS runner、兼容层、报告预检、历史报告和回答 baseline 对比器；一并移除 `ragas`、`datasets` 以及代码未使用的 `langchain`、`langchain-community`、`docx2txt`、`pillow` 顶层依赖，保留实际检索、生成、人工复核和证据测量。
- 删除严格 manifest 配置 contract、源码 fingerprint hash 和固定 chunk 数前置阻断；生成追踪直接比较已有运行配置、上下文、metadata 和答案，真实应用上下文模拟继续保留在外部生成边界。
- 合并重复环境变量布尔解析并删除一个未调用函数；输入校验、原子写入、来源隔离和其他安全措施未变。全量离线测试 `264/264`、`py_compile`、依赖检查和 `git diff --check` 通过，未提交或推送。

### 2026-09-12：删除平行检索评测实现

- 删除约 2,000 行、会复制应用检索与证据补全行为的 `evaluation/benchmark_retrieval.py` 及其 runner 专用测试；评测改为直接运行 `app.query_knowledge` 的上下文模拟，避免产品和评测行为分叉。
- 将共享 BM25、路由、查询拆分、来源覆盖和 RRF 的 17 条原语测试迁入 `tests/test_retrieval.py`；没有删除这些产品能力或测试覆盖。
- 精简基准说明，保留清单/PDF 校验、已有文件摘要核验、隔离数据库、真实应用上下文模拟和人工答案复核；删除不可再执行的命令和重复阶段叙述。全量离线测试 `236/236`、`py_compile`、依赖检查和默认 5 篇/53 题清单校验通过；未修改安全校验、项目数据库或外部服务，未提交或推送。

### 2026-09-12：本地知识库管理闭环

- 上传页新增实际入库文档与文本块数清单；问答页可选择一篇或多篇文档作为现有 `source_filter`，不选择时仍检索全库。
- 新增单文档删除：必须先显式确认，再按实际 Chroma ID 删除对应文本块、失效内存检索快照，并清理不再被其他文档引用的本地视觉 PDF。
- 应用不再因缺少 API Key 而拒绝启动；设置页可为当前进程配置 DeepSeek，密钥不写入知识库或项目文件。未配置时仍可上传、查看和删除本地文档，生成操作返回明确提示。
- 兼容 Gradio 6 的字符串上传路径，并把主题参数移到正确的启动位置。使用真实临时 Chroma 验证删除与文件清理，无密钥首次启动 UI 验证通过，全量离线测试 `239/239` 通过；未调用外部 API、未修改项目数据库、未提交或推送。
- 新增 `start.sh` 作为 macOS/Linux 的最小本地启动入口：首次运行仅创建 `.venv` 并安装现有 `requirements.txt`，后续直接复用环境；强制离线模型、空 API Key 和临时数据库的真实启动烟测已在本机端口成功提供 UI，停止后已清理临时数据库。没有引入桌面框架、锁文件、摘要或额外发布门禁。

### 2026-09-13：全新目录首次运行与用户闭环

- 在不含 `.git`、虚拟环境、`.env`、数据库和模型缓存的仓库外副本运行 `start.sh`；脚本完成依赖安装、中文嵌入模型下载和无 API Key 启动，真实 Gradio 页面可由本机 HTTP 访问。
- 首次安装中断曾留下半成品 `.venv`；启动器现复用现有 `test_setup.py` 检查直接依赖，缺失时继续补装，而不是仅凭目录存在跳过安装。慢速网络还实测触发 ONNX 下载超时，故只增加 pip 自带的 60 秒超时和断点续传次数，没有新增自定义状态、锁、hash 或重试框架。
- 在同一全新环境完成 TXT 上传、2 个真实向量块入库、文档清单、限定来源的确定性表格问答（`0.91`）、删除确认、确认删除和运行时重建后的空库检查；完整环境再次启动时跳过依赖安装并正常提供 UI。未调用生成 API、未修改项目 `chroma_db`、未提交或推送。

### 2026-09-13：本地与自带 Key 的模型服务

- 生成配置由 DeepSeek 专用字段改为通用 OpenAI 兼容 Base URL、模型名和 API Key；设置页提供 Ollama、DeepSeek、Gemini 与自定义预设，复用现有 OpenAI SDK，没有新增 provider 层或依赖。
- Ollama 回环地址允许空 Key 并使用 SDK 所需的本地占位值；远程服务仍要求用户自己的 Key，设置只留在当前进程。新 `LLM_*` 环境变量优先，已有 `DEEPSEEK_*` 配置继续兼容。
- 使用本机临时 HTTP 端点实际接收一次 `/v1/chat/completions` 请求，确认所选模型和响应链路有效；无 Key UI 构建、全量离线测试和输入边界测试通过。未接入共享免费 Key、账号系统、密钥存储或新发布门禁。

### 2026-09-13：Ollama 真实本地生成验收

- 在 Apple M5 / 16 GB Mac 上安装并启动 Ollama 0.34.0；模型服务只监听 `127.0.0.1:11434`，测试使用仓库外临时 ChromaDB，未调用云端 API 或修改项目数据库。
- 实测 `qwen3:4b` 是 Thinking 标签：普通问答、大纲和测验分别约 53、73、70 秒，并因推理耗尽输出预算返回空正文；`reasoning_effort=none` 仍不能让该标签直接回答。因此将 Ollama 预设和 README 命令改为官方 `qwen3:4b-instruct`，没有增加 provider 分支或额外配置。
- 同一材料复测：问答 7.29 秒且 5/5 目标事实命中；未知成本问题 1.45 秒并拒绝编造；Markdown 大纲 7.22 秒；5 道单选题 14.01 秒且题目/答案数量完整。模型驻留约 3.18 GB Metal 内存；大纲和题目偶发将型号数字写成“等”，记录为 4B 小模型的文本质量边界。

### 2026-09-13：真实 PDF 全页面验收

- 通过 Gradio 页面在独立临时库上传 20 页 MgNO（ICLR 2024）PDF，成功生成 159 个文本块；页面文档清单、块数和来源筛选一致。全程仅连接本机 `qwen3:4b-instruct`，未调用云端 API 或修改项目知识库。
- 首次公式问答复现默认 Ollama 4096-token 窗口阻断：请求约 4605 tokens 并返回 400。将已有 `SCI_RAG_CONTEXT_K` 默认值从 10 调整为 4，并让问答、大纲、测验共同遵守该限制；不增加重试、模型分支或新依赖。修复后同一页面、同一 PDF 的公式问答成功。
- 页面验收结果：公式题返回 `A∗u=f` 与 `3×3`；Table 1 题返回 Darcy rough `L2=0.339`、`H1=1.380` 及 `×10⁻²` 表注；正文题返回 `(0,1)²` 与 Dirichlet/Neumann/periodic；不存在的咖啡品牌信息被明确拒答。大纲生成层级完整，测验生成 5 题、5 个答案和解析。
- 删除闭环通过：未勾选确认时页面返回“请先确认删除”；确认后删除 159 个文本块，块数变为 0、文档清单变为“暂无”。随后停止测试服务并清理独立临时库，项目知识库和原始 PDF 未受影响。
- 公式答案曾附加与问题符号无关的“原文核对项”；补全逻辑现在在问题明确命名符号时只接受同名左值或明确公式标签，同题页面复测不再出现该尾注。仍记录一个非阻断体验缺口：4B 模型的正文/公式回答明显偏长，偶尔在引文中产生 `<tool_call>` 字样，最终事实结论不受影响。
- 页面观测耗时约为：PDF 入库 4 秒内、公式问答 25 秒内、确定性表格问答 1 秒内、正文问答 15 秒内、缺失事实拒答 4 秒内、大纲 15 秒内、测验 13 秒内。模型驻留约 3.18 GB Metal 内存，上下文为 4096。全量离线测试 `242/242`、`py_compile`、依赖检查和 `git diff --check` 通过；未提交或推送。

### 2026-09-21：大纲与自测资料范围

- 大纲和自测复用现有来源过滤，支持单篇、多篇及未选择时的全库范围；上传、删除后同步更新选择列表。范围无内容时提示重新选择，不回退到其他资料。
- 页面按“选择资料、生成、查看结果”排列，并说明仅使用部分片段，不能保证全文覆盖。README 同步更新；后续证据可靠性研究计划保存在评测目录，尚未开展新论文比较实验。
- 251 项测试通过。临时真实 ChromaDB 与本机 Qwen 验证了上传、来源隔离问答与原文、大纲、自测 JSON 解析、未完成作答提示、评分、删除确认和列表刷新；新进程成功读取保留文档。
- 浏览器实际上传测试 TXT，选择来源并成功生成大纲；检查自测页初始按钮状态。1280 与 390 像素宽度的页面宽度均未超出视口。未调用云端模型，未更改用户资料库。Windows 与 DMG 未在本轮验证。

### 2026-09-14：内置 llama.cpp 与 macOS 应用打包

- 绕过 Ollama 服务，直接用独立 `llama-server` 加载现有 Qwen3 4B Instruct GGUF；短请求、159 块真实 PDF 入库、正文问答、知识大纲和 5 题测验均通过。4096 上下文会在真实公式题的 3821-token 输入后截断输出，改用 8192 后不再截断；实测模型服务 RSS 约 3.7 GB。
- 新增 `desktop.py`：从打包资源启动单槽、离线、仅回环监听的模型服务，使用随机 API Key 和系统分配端口，等待就绪后复用现有 OpenAI 客户端；退出时同步停止子进程。冻结程序入口调用标准库 `multiprocessing.freeze_support()`，避免资源追踪子进程递归启动应用。
- 新增最小 macOS 构建脚本，复用本机模型资源并收集 Gradio 源数据、Chroma 动态模块与 Rust 扩展。Apple Silicon 内部 `.app` 约 3.5 GB；打包产物实测仅启动一个模型实例，页面返回 HTTP 200，临时 TXT 上传为 1 个块，问答精确返回 `BLUE-482` 及来源。桌面模式隐藏外部模型设置、显示本地隐私说明和启动窗口，并提供“退出 Sci-RAG”按钮；退出请求正常返回后网页服务与模型进程均停止，无残留进程。当前仅为 ad-hoc 签名内部包，尚未完成图标、第三方许可证归档、Developer ID 签名或公证；244 项测试及静态检查通过，未提交或推送。

### 2026-09-21：论文阅读与本地问答开发实验

- 阅读 RAGTruth、SciDQA、ChainRAG 的方法、实验和局限，核对关键 PDF 页面，并参考 RAG 原论文与 Lost in the Middle 的相关章节。校正旧 ChainRAG 两题的证据页码；未修改历史答案或把开发题重新包装成未见数据。
- 新增最小本地实验入口 `evaluation/paper_pilot.py` 和 12 道有原文依据的开发题，复用实际上传与 `app.query_knowledge`；仅调用本机 Qwen3-4B Instruct，不读取云端 Key，不修改日常知识库。
- 完成默认 dense 与仅开启已有文档路由的 24 次问答；Codex 对照原文初评完整正确 1/12 → 4/12，但一题退化，两组各有 2 次输入超限、1 次输出截断，不能视为稳定提升。未改产品默认配置。逐题结论、局限、复跑方法和后续优先级见 `evaluation/PAPER_PILOT.md`。
- 原始请求、响应、usage 和证据留在忽略的 `tmp/pdfs/paper_reading/`；251 项测试、实验 trace 完整性检查、编译和 `git diff --check` 通过。尚无独立人工复核或未见检查组；没有新增 hash、冻结清单、gate、依赖，也没有执行任何 Git 写操作。

### 2026-09-22：输入预算、停止限制判别与表头修复

- 问答、大纲、自测复用完整片段预算与生成入口；新增可配置的实际窗口/输出上限，事实清单仅占剩余预算，证据面板与送模片段保持一致。输出截断会提示，截断自测 JSON 不载入；内置桌面窗口与其 8192-token 服务一致。长度是估算，非模型专用分词计数。
- 修复“停止限制”误触发局限性检索、宽泛章节扩展压过源内词面证据，以及重复指标组表头错位/命名组列选择。复用已有文档标识，在同一文件全部新片段写入成功后清理过期解析；导入失败或空结果不先清空旧资料。未操作日常知识库，实验库清理了 6 个过期解析片段并恢复为 304 块，原始 PDF 保留。
- 257 项测试通过，包含真实 Chroma 重导入成功/失败/空解析、预算与证据面板对应、输出截断、停止限制和分组表头回归；编译及差异空白检查通过。两组共 24 题终态复验（4 次确定性查表、20 次本机模型调用）无 API 输入超限，P10 两组仍输出截断且均显示提示。
- 路由组开发题由 Codex 初评完整 7/12，仍有 2 部分、1 错误且截断、2 拒答；多个因素同时修改且无独立人工复核，不是泛化准确率或单因素提升证明。默认 dense 仍有明显证据选择和生成质量问题，未默认开启路由。全部中间失败和终态记录见 `evaluation/PAPER_PILOT.md`。
- 同一隔离库、同一模型实际生成大纲 8.53 秒（323 输出 tokens）和自测 15.52 秒（615 输出 tokens），均正常结束，自测解析为 5 道完整题；只验证流程与格式，不据此宣称生成内容全部正确。未新增依赖、hash 或 gate，未执行暂存、提交、推送、合并；未重建 DMG 或进行 Windows 验收。

### 2026-09-22：漏证据修复与生成侧失败对照

- 补齐方法设置、图像和实验边界的源内检索触发词与别名；直接命中优先，不插入无关相邻方法段。复用源内候选，跟随首段明确引用的一段同源图注，修复 P05 top-3 图注未送模；P08/P09/P12 的关键方法和边界段已进入实际上下文。
- 压缩重复提示词以容纳证据，保留既有数值、公式、表格、图形安全约束。实际运行五题诊断、两次整组、一次固定上下文对照及四题默认检索对照。额外短答指令虽使 P05 写出 top-3，却使 P08/P09 漏掉 BM25/提前停止条件，已撤回，未挑选最优单题拼成结果。
- 最终保留代码对应 `routed-vu55t7oq` 整组，输入 1802–2058 tokens，无 API 超限；P09/P10 仍输出截断，P05 漏数、P10 概念误解、P12 错认实验输入仍未解决。258 项测试与差异检查通过；详细失败记录和下一步见 `evaluation/PAPER_PILOT.md`。未新增依赖或 Git 写操作。

### 2026-09-22：跨论文评测证据补齐

- 修复已有问题拆分在多句问题中漏拆配对来源；跨来源候选按来源交替选取，限定文档后去除来源标识词的重复排序权重，并在送模标签中附已有 H1 标题。补充评测/参考答案/忠实等英文别名；实验入口可显式开启已有的问题拆分。应用默认配置未改变。
- 表格、公式专门指令仅在问题或候选证据涉及对应类型时加入，仍保留全部相关约束及数值、图形坐标、来源安全规则；预算与真实请求使用同一提示词。没有恢复被否决的短答指令，也未添加依赖、自动重试或专用答案补丁。
- 261 项测试、编译及差异检查通过。完成三次跨论文诊断、五题固定输入对照、十二题路由+拆分整组及五题默认检索复核，全部原始记录保留。终态 P05 含 top-3；P10 实际收到两文四段评测证据，但生成仍过度归纳并截断；P09 事实齐全但重复截断，P12 仍误读否定证据及输入边界。默认检索仍有明显缺证据和错误，不宣称已解决全部漏答。
- 下一步转向固定证据的事实归属、明确否定与重复输出问题，不继续为已经到达的证据叠加检索机制。详情见 `evaluation/PAPER_PILOT.md`。未操作日常资料库，未暂存、提交、推送或合并。

### 2026-09-22：固定证据生成诊断（未采用失败方案）

- 完成 24 次本机固定请求对照：显式否定/推论规则、原文优先回答结构、去重压缩通用指令，以及分别调整 presence_penalty、温度。部分答案不再截断，但仍有错误拒答、实验输入边界混淆、方向误述、引用改写和过度归因。全部候选均未写入产品代码，不把“有输出”或“未截断”当作修复成功。
- 只读检查 Ollama 模板与采样参数，并参考官方 Qwen/Ollama 文档；没有证明模板有误，也不声称已证明模型规模是唯一原因。原始失败请求和响应全部保留，路径与逐组结论见 `evaluation/PAPER_PILOT.md`。
- 261 项现有测试通过。保留现有代码、安全约束和默认配置；本轮只追加实验记录，未执行 Git 写操作。下一步拟在用户允许额外下载后，用另一候选本地模型进行同证据对照；尚未下载或替换模型。

### 2026-09-22：8B 下载与同证据模型对照

- 经用户授权下载 `qwen3:8b`（5.2 GB），本机成功 GPU 运行，保留旧 4B 和应用默认配置。复用现有实验入口支持固定请求回放、指定模型和显式关闭思考，没有新增依赖或产品分支。
- 完成 11 次本机对照：8B 原提示词三题、已有候选提示词六题、原提示词思考模式两题。P09 重复截断改善，思考模式 P12 的主要否定边界改善，但 P10 仍出现错误归纳，未将候选模型或提示词设为默认。思考组改变了窗口和预算，不作单因素收益或泛化声明；详见 `evaluation/PAPER_PILOT.md`。
- 新增回放不改写原请求的最小测试；262 项测试、编译与差异空白检查通过。未操作日常资料库，未执行暂存、提交、推送或合并。

### 2026-09-22：拆问与判断对象诊断

- 完成 8 次本机对照：单文事实、跨文比较、单独综合解释、判断对象提问，以及原文定义句消融。单文事实和目标比较相对较好，但综合解释与证据判断对象仍误读；精简原文后 8B/4B 也均有错误。不采用自动拆问、多轮生成或失败提示词，不把正常结束当成语义通过。
- 修复实验回放文件不存在却以空结果正常退出的问题，保留共享读取函数的恢复运行行为，增加一项回归测试；263 项测试、编译及差异空白检查通过。详见 `evaluation/PAPER_PILOT.md`；未修改产品模型默认值、安全规则或日常资料库，未执行 Git 写操作。

### 2026-09-24：系统指令受控诊断

- 对 P10/P12 完成原指令、仅扩展角色职责、短版诊断指令三个条件共六次本机调用，固定问题、证据、模型及请求采样/输出设置。全部正常结束，但 P10 各条件均误解综合解释，P12 仍有输入边界混淆或遗漏。候选均未采用，未删除产品安全约束；详见 `evaluation/PAPER_PILOT.md`。
- 按用户要求使用 Luna 子模型进行机械回归与调用链核对：263 项测试、编译、差异空白检查通过；问答预算与实际发送的系统提示词一致。暂不继续对同一题叠加提示词，下一步转向默认检索的实际证据覆盖检查。本轮仅追加实验记录，未改产品代码、日常资料库或执行 Git 写操作。

### 2026-09-24：资料范围复核与中文方法术语检索

- 复现默认全库检索的跨论文混入与关键段漏取；已有网页单篇范围选择能让 P02/P05/P09/P12 取到关键证据，无需新增 UI 或自动开启路由。README 补充使用指导，校正代码中“UI 不传范围”的过时注释。
- 中文改写发现源内检索缺少“候选句、重排、种子、扩展、结束”的英文映射；复用已有词表和触发机制补齐，新增最小英文证据检索/来源隔离测试。V09 改写补回扩展条件和参数段，其他三种改写的选段保持；未声称不限定范围的默认模式已修好。
- 完成检索对照及五次真实 4B 生成；关键方法事实覆盖改善，但 P12 仍误解否定边界，V09 回答仍漏 3000 词。Luna 机械核对、264 项测试、编译及差异检查通过；全部原始记录与局限见 `evaluation/PAPER_PILOT.md`。未修改日常资料库，未执行 Git 写操作。

### 2026-09-24：多篇资料范围的证据覆盖

- 复用已有分句、文档识别、源内证据和来源轮换：用户明确选中多篇并点名论文时，检索分句限制到对应所选来源。没有全局默认启用文档路由，也不把未问到的第三篇强塞给模型。
- 多篇上下文标签加上路由器确实匹配到的问题词，使正式标题不含方法缩写的论文也能与用户提问对应。对同一道 SciDQA/ChainRAG 方法题，实测此前虽有两篇原文却被 4B 拒答，标注后两篇方法和各自的“3”均答出；P10 仍有生成错误及截断。网页和 README 增加比较多篇时点名文档的提示。
- 七个多篇/单篇/范围外检索条件修复前后成对复核，新增真实 Chroma 最小回归测试；原始 trace 与适用限制见 `evaluation/PAPER_PILOT.md`。未使用日常资料库，未执行 Git 写操作。
- 265 项测试、编译和差异空白检查通过。Luna 的机械检查因其额度用尽未完成，最终由主 agent 复核。

### 2026-09-26：全项目精简审计

- 删除无真实调用的证据重试提示构造器、表号包装、重排缓存及统计结果包装、旧 Phase1 UI 启动脚本。保留当前 PDF/TXT/DOCX、ZIP 启动器、桌面入口、研究评测与已有数据/安全保护。
- 统一模型配置为 `LLM_*`，同步视觉实验和本机配置变量名；删除旧 DeepSeek 别名。删除旧调用者的全局懒加载 runtime，当前入口与回调显式传入资源；移除无用问答 history 参数、旧文件对象适配和不符合当前 PDF/嵌入库返回形状的分支。
- 上下文覆盖工具改名为 `evaluation/app_context_audit.py`，保留逐题缺失事实报告，覆盖不足不再成为执行门禁。全局 Codex 规则加入禁止超前设计、过度工程、无事实依据的防御和旧路径胶水等前提；已有安全措施与当前支持范围仍必须保留。
- 276 项测试通过；新增真实 PDF 页码回归及审计非阻断检查。真实本地 BGE 与隔离 Chroma 验证三种格式导入、重复上传、来源限定、问答与学习流程；生成使用测试客户端，不证明模型答案质量。原生 Gradio 92 个组件、38 个事件构建成功，临时本地网页 HTTP 200，确认删除检查通过；未使用日常资料库，未调用云端模型，未执行 Git 写操作。

### 2026-09-26：问答忠实性续测

- 预写 TANQ/SPIQA 四道新开发题，原文视觉核对后在新隔离库运行本机 4B；核心事实及缺证边界均答出。另复现 AF02/AF04 并为六份回答保存逐条断言支持与无据附加的 Codex 非独立复核，不将小样本当作独立准确率。
- 固定证据进行八次“引用后解释”格式复核、三次已安装 8B 的显式推理预算检查；无据评价和明确限制漏答仍在，未修改产品提示、模型或预算。下一步为复合问题分项生成的同题对照，详见 `evaluation/PAPER_PILOT.md`。没有新框架、门禁、自动重试、云端调用或 Git 写操作。

### 2026-09-26：分项生成对照与复测

- 完成 27 次本机固定证据对照，记录完整输出、调用数、token 用量和耗时。增加输出预算、分项多调用及单次编号均未在 SPIQA/ChainRAG 稳定消除错误，不写入产品。RAGTruth 单次编号两问的两轮结果较好，但不能据单题推广。
- ChainRAG 直接询问原文 `word limit` 两次均答出 3000，原方法问法仍漏答；据此收窄下一步为跨论文参数关系与证据定位检查，而非继续叠加提示词。README 补充可选的编号提问及原文参数术语建议，同时保留核查提醒。详细结果与局限见 `evaluation/PAPER_PILOT.md`。
- 276 项测试、编译和差异检查通过。没有新依赖、门禁、自动重试、默认生成改动或 Git 写操作；未使用日常资料库。

### 2026-09-26：参数术语与关系归属对照

- 保存三篇论文的中文／原文术语窄问及完整方法题，共八个配对问题。完成 16 次固定证据本机调用；窄问两种表述均保留关键参数，没有支持增加英文术语自动改写的收益。
- 完整 ChainRAG 问法明确 `word limit` 后补出 3000，但仍把图构建实体筛选参数误归种子检索，且出现无据的全局／局部限制解释。现有词面审计对此两份错误答案均报告 full；记录具体断言与原文归属，不将它视为语义正确或新增门禁。
- 本轮未修改产品生成逻辑。276 项测试、编译、差异检查通过；未使用日常资料库或执行 Git 写操作。后续优先核对跨论文的方法参数关系，避免继续对同题叠加提示。

### 2026-09-27：方法关系取证与表格误路由修正

- 核对 TableRAG/TANQ 四个原始 PDF 方法页，预写四道参数阶段/对象关系题，在新 247 块隔离库完成实际问答及人工证据对照，共十次本机调用。保存非独立复核，区分检索漏证、输入上文不足和无据目的解释，不以数字覆盖代替关系正确性。
- 修正共享表格意图表达式：裸“答案表中”不再被当成现成表取值引用；编号表格及“该答案表/该表/下表”等明确指示保留。MR04 恢复证据评估正文并答出两个子项，映射正文仍漏取，不宣称整题修好。
- 新增先失败再通过的最小回归测试；同库十二题修正前后对照，除目标题外十一题结果保持，Table 4 明确单元格取值仍为23。277 项测试、编译、差异检查通过。下一步核对双栏页阅读顺序及标题归属；未新增模型调用机制、配置、门禁、答案补丁或执行 Git 写操作。

### 2026-09-27：严格双栏阅读顺序与章节归属

- 定位 TableRAG 标题/正文错配发生在 PDF 转 Markdown 阶段，复用原生布局坐标及文本区间，仅对明确分开的双栏页按列阅读，不改写内容、不切回旧解析器、不猜测跨栏布局。五篇 99 页中修改五页，逐页渲染核对；48 个表格块和 30 个补回公式块保持一致。
- 先复现再修复加载路径的标题归属回归；278 项测试、编译、差异检查通过。新隔离库四次真实问答只恢复 MR02 附录筛选条件，仍缺 HybridQA 比较证据，其余三题仍有漏取；十二题成对证据检查只有 MR02 改变，明确表格取值仍为23。
- 核对并注明论文正文 ≥20 行与附录 >20 行的边界不一致，修订 MR02 参考说明，不将原文冲突计作模型幻觉。保存候选排名与实际调用链，进一步定位 MR03 把参数“限制什么对象”误判为局限问题，局限候选挤掉已召回的方法正文；下步先修正这项意图混淆，再处理候选抢位/页脚噪声，不叠加生成补丁。详情见 `evaluation/PAPER_PILOT.md`；未修改日常资料库或执行 Git 写操作。

### 2026-09-27：参数约束误判及方法词映射

- 修正共享局限表达式中“限制什么/哪些”等动词问法的误触发，保留真正及混合局限问题的检索。移除条目/多少泛化成数据集规模/统计的两处词面扩展，补齐筛选、改写、技能的直接英文方法词；复用现有映射及真实调用，不新增分支、默认配置或生成补丁。
- 三次真实本机调用：仅意图修正仍拒答；词面候选与实际产品调用恢复单元格超过五条目时过滤、最多三种技能，但五轮改写及停止/丢弃条件仍漏取，存在从缺证据概括全文无约束的风险，不记完整忠实。更新 MR03 非独立复核为 partial。
- 最终同库十六题前后成对检查只有 MR03 改变，真实局限题、数据集数量及明确表格取值保持。句末英文标点候选虽通过字面测试，却挤掉已恢复的技能段，未采用；记录失败，不据单元测试宣称问答变好。278 项测试、编译、差异检查通过。下一步核对跨页改写正文续段与图注排序；未修改日常资料库或执行 Git 写操作。

### 2026-09-27：跨页续句与图注候选竞争

- 测量确认仅取消图注提权仍漏取改写条件；复用已有连续正文遍历和共享跨页续句识别，识别时忽略本块脚注/独立页码，只优先一条明确的下一页续句，原始证据不变，保留无续句时的图注跟随。没有开启默认父窗口、扩大预算或新增生成补丁。
- MR03 实际送模从37/25/96/28变为37/25/27/28，三项方法证据齐全，回答补出五次改写及停止/丢弃条件；但总结停止时机仍有歧义，不记完整忠实。派生停止边界题因中文改写未触发词面补充而缺证拒答，同一完整证据对照能正确区分五次上限和提前停止；共三次本机调用，记录全部失败与对照，不视作独立准确率。
- 回归先失败再通过；279 项测试、编译和差异检查通过，十六题前后成对检查其余十五题结果及请求不变。下一步检查已有方法词映射与词面补充触发规则的不一致；未修改日常资料库或执行 Git 写操作。详见 `evaluation/PAPER_PILOT.md`。

### 2026-09-27：独立改写/技能问法的触发遗漏

- 原有方法词回归依赖“步骤/阶段”而掩盖独立提问漏触发；改成中英文独立问法先失败再通过，现有触发列表仅补改写、技能、rephrasing、skills。筛选触发候选挤掉MR02已恢复条件且未修好筛选题，未采用；没有合并全部词面映射为触发词。
- 同库23题成对检查，原16题选段、答案及请求保持，其中Table 4仅增加词面证据元数据标记；5个目标问法取证改变，2个筛选问法仍失败。6次本机实际生成：技能中英文正确，停止边界在单篇及与上轮相同的两篇库均答出提前停止/上限丢弃；改写两题仍有过强语义保证措辞，保留partial。保存开发题及逐项非独立复核，不记总体准确率。
- 279项测试、编译、差异检查通过；下一步检查筛选条件与统计候选竞争。未改模型/提示/预算默认值、日常资料库或执行Git写操作，完整过程见 `evaluation/PAPER_PILOT.md`。

### 2026-09-27：答案表泛化统计导致筛选漏证

- 候选评分确认“答案表”无据扩展到行列统计，移除该泛化并保留answer/table字面含义；仅补来源限定的答案表概念触发，不启用全体筛选或英语词形补丁。保留明确行列/平均/规模问法、文档隔离及现有安全规则。
- 同库26题成对检查22题结果及请求保持；4个受影响问法各一次实际本机调用。中文筛选正确答单个单元格超过五条目时丢弃，平均尺寸证据仍在；但一份尺寸回答照抄6._7解析装饰，MR04仍缺关系映射，明确记录partial。英文筛选仍未修好，不将候选恢复等同回答通过。
- 新回归先失败后通过，280项测试、编译、差异检查通过；下一步核对英文方法条件与问句常用词的候选竞争，并保留小数呈现问题。未改分词器、默认生成配置、日常资料库或执行Git写操作。

### 2026-09-27：英文筛选取证及全库词频影响

- 最终仅补现有filtering→filtered映射及英文取证触发。疑问词删除虽恢复两个变体，却在同库实际对照出现oracle错答open，已撤回；不将候选成绩记为产品通过。既有回归先失败再通过，280项测试、编译、差异检查通过。
- 29题最终成对检查26题结果及请求保持；十三次本机调用记录候选、对照及最终结果。最终英文筛选见到条件仍拒答，HeteQA仍合并正文/附录行数边界，分别记incorrect/partial；单段证据对照能答对，但不证明干扰的唯一原因，不新增生成补丁。
- 进一步确认同一TANQ的108块内容不变，添加SPIQA后全库BM25统计仍改变来源限定排序，导致停止条件掉出上下文；临时文档局部索引能消除该词面漂移，尚未写入产品。下一步优先修正索引统计范围并验证多库实际效果；未修改日常库、默认模型/提示/预算或执行Git写操作。

### 2026-09-27：来源局部词面统计与复合问答覆盖复查

- 在共享来源词面补充中复用已有局部BM25构建方式，消除其它文档对所选来源文档频率/平均长度的影响；保留原块类型、来源边界和续段规则。新增先失败后通过的最小词频隔离回归，四个既有续段单元测试调整排名mock位置，不新增兼容路径或缓存框架。
- 两种隔离库各29题成对检查：修正后的29题词面结果、完整结果及模型请求跨库相同；同库前后不变分别19/29、17/29，明确表格取值23及真正局限控制保持。小样本热快照额外约5ms，不支持新缓存。281项测试、编译、差异检查通过。
- 九次实际本机调用：停止边界及英文筛选原题在两库各一次答对，但英文变体仍见证拒答；复合MR03技能段被实验段挤掉，实际漏答且输出截断，明确记录退化。论文行数冲突及PaLM-2漏答仍partial，不把统计隔离当作整体质量通过。下一步优先复合题子项覆盖/融合截断，详见 `evaluation/PAPER_PILOT.md`；未修改日常库、生成默认值或执行Git写操作。

### 2026-09-27：删除稠密首段强制提权，恢复复合题方法覆盖

- 删除单篇选择时将首个稠密结果重复包装并前置的提权块，复用原合并顺序；稠密结果仍正常召回和合并，类型/来源/表格/图形/公式等边界保持。新增先失败后通过的末尾方法段抢位回归，既有词面补充测试调整证据优先顺序，不增加新框架或默认配置。
- 两库29/33题成对检查及62条件实际产品复核一致，原29题修正后跨库输入一致；同库完整结果/请求不变9/29、12/33，不冒充其余题全部不变。MR03恢复25/27/53/28三项方法证据，明确表取值23和真正局限控制保持；MR04映射及MT04-en条件仍漏取。
- 十八次本机调用保留全部结果：MR03修正后答全筛选/停止/三种技能，消融及SPIQA人工筛选控制答对；英文变体仍拒答、oracle设置引用不精确、行数冲突仍合并、停止边界开头缺证话术仍不协调。282项测试、编译、差异检查通过。下一步分开诊断英文条件检索漏证与见证据仍漏答；没有扩大窗口、叠提示/重试、修改日常库或执行Git写操作，详见 `evaluation/PAPER_PILOT.md`。

### 2026-09-28：英文答案表词面召回与模型拒答分层

- 复用既有`答案表→answer/table`词汇映射，仅补`answer-table→answer/table`。MT04-en条件段由第5词面候选升至首位并进入模型输入；两套库29/33题成对检查均只有该题完整结果/请求变化。新增先失败后通过的普通词汇回归；未改全局分词器、窗口或生成提示。
- 实际本机调用中MT04-en先错误宣称缺证、后正确答五条目阈值，记partial；MT05-en仍见到25却整题拒答，单段25对照能答对但不能判定唯一干扰项。限于`How does`的疑问词候选虽把25排首位仍拒答，已撤回；未采纳额外`cell-entry`映射或新增答案补丁。
- 七次本机调用、现有两题非独立复核和隔离库记录详见`evaluation/PAPER_PILOT.md`。保留MT05模型漏答及其它未解决问题；无新框架、缓存、门禁、云端调用、日常库改动或Git写操作。

### 2026-09-28：MT05固定输入对照与小数转换残留

- MT05固定四种证据输入：两种四段均错误拒答，25+48两段及25单段均答对；两种四段去掉事实清单仍拒答。六次本机调用各一次，不能把长度、干扰段或模型随机性指认为唯一原因，未采用泛化删段或新提示。
- TANQ第6页原文为6.7，PDF转Markdown变成`6 _._ 7`；只在导入边界修正这一明确残留。回归先失败再通过，原PDF重新解析已见6.7；日常库旧索引未重建。283项测试通过，未执行Git写操作。详情见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：三元组直译词恢复MR04映射证据

- AF01逐项引用提示和MT06冲突边界提示各做原样/候选本机对照，均未稳定修好目标；候选未写入产品。MR04证实原四段缺答案表映射，仅补“三元组→triple/triples”使映射段25入模。
- 两套隔离库29/33题成对检查每库只有MR04请求改变；更宽的“关系”映射改变另外两题并被撤回。MR04原请求仍漏答映射、候选单次本机调用正确答关系对应列及单元格对应WD三元组。普通断言先失败再通过，283项测试通过。未改日常库、模型默认值或执行Git写操作；详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：文本块与迭代直译词恢复MR01实现证据

- 原MR01四段缺实现参数32及提前终止22；只补“文本块”或只补“迭代”各能恢复一项，合用后送22/32/19/115。两套隔离库29/33题成对检查均仅MR01请求变化；更宽的生成提示未采用。
- 原请求本机4B整题拒答，最小候选单次答对30/3候选、1000/200-token分块、五次上限及无需子查询时终止。产品请求与候选相同，现有回归先失败再通过；日常库未重建、无Git写操作。AF01引用与MT06阈值冲突仍待解决，详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：AF01/MT06固定证据生成对照（未采纳产品改动）

- 已安装8B对两题各做一次同证据调用：AF01引用正确，MT06仍未区分正文≥20与附录>20；耗时33.6/27.5秒。4B仅移除事实清单各调用一次：AF01本轮正确但原提示也曾正确，MT06仍合并边界。没有足够依据更换默认模型、删除清单或加冲突特例；产品代码未改，日常库和Git未写。详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：英文连字符复合词取证与新问法对照

- 在已有来源词面查询入口额外拆分纯字母连字符词，保留原词及含数字标识符；删去被通用规则覆盖的`answer-table`专属映射。两道已见论文的新问法先核对PDF原文，未据模型输出改题；其中TANQ的`context-length`原本导致`<4k`证据缺席。
- 两套隔离库对原29/33题及新两题成对检查，各仅TANQ新题与MT04-en请求改变。断词后TANQ同时送达`<4k`与3-shot依据；本机4B/8B各一次均答出两项。TableRAG新题两模型在原请求上也各答出ReAct过度拆分及至少10%提升。MT04-en单次对照由错误拒答变为正确阈值回答，不能据此宣称稳定改善或总体准确率。
- 现有回归增补断词及含数字标识符断言，283项测试通过。AF01引用波动、MT06正文/附录边界冲突及MT05见证拒答仍待处理；未改日常资料库、生成默认值或执行Git写操作。详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：新论文三题的一次性盲测（无产品改动）

- 从此前未用于项目调试的ACL 2025论文UAEval4RAG预先固定三题，核对PDF原文后在112块隔离库按默认检索与本机4B各运行一次：一题部分回答，两题因关键证据未入模而拒答。仅替换为对应原文片段的三次固定证据对照均答出；不是稳定性或总体正确率。
- 诊断确认来源局部词面帮助指标题却挤掉配置题的dense结论，简单全局开关不能同时解决；分类题另有跨页清单漏取。未为这三题新增词条、后处理、基准门禁或调整默认配置，下一步先在既有开发题上衡量候选争位策略，再用另一篇新论文复核。记录见`evaluation/PAPER_PILOT.md`；日常库与Git均未写。

### 2026-09-30：来源局部词面消融与跨页清单核查（无产品改动）

- 在原29/33道开发题的两套隔离库中，关闭来源局部词面分别改变20/29、21/33个实际请求；MR01、MR03及MT04-en等已恢复的必需证据会丢失。因此不能用全局关闭修单篇新论文的候选挤占。该数值仅为输入变化，不是正确率。
- 只读扫描两套旧库及新论文库的明显跨页清单形态，仅新论文段6→7匹配，尚无依据扩大跨页窗口或删页脚。下一步需以既有开发题逐事实覆盖和最终请求衡量候选仲裁，再另取新论文验收；未改产品、日常资料库或执行Git写操作。详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：dense固定配额候选被开发题否定（无产品改动）

- 用现有逐事实词面覆盖工具，在两套隔离库中模拟“来源局部前两段、dense前两段、其余候选”的最小交错。247块库22道有标注题的full由15降至10、事实命中由49/61降至41/61；358块库26题由17降至12、60/74降至52/74。两库各五题丢失已恢复事实，没有标注题补全，故撤回候选。
- 只保存离线诊断，不把输入变化或词面覆盖当答案正确率，不以新论文单题修复覆盖旧题退化；未改产品、日常库或执行Git写操作。下一步转查候选冗余/归属根因，详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：局部BM25仅正文候选同样撤回（无产品改动）

- 仅在内存中让来源局部BM25用正文而非正文加标题/来源计分，两库各13题请求变化；有标注题的full分别15→14、17→16，词面事实命中49/61→48/61、60/74→59/74。两库均使MT01-en的五条目阈值段25掉出四段，没有标注题新增覆盖；因此不改共享计分文本。
- 此前的dense固定配额及本候选都未达到“旧题不失证”的最小要求；不能继续围绕新论文三题堆排序规则。日志见`evaluation/PAPER_PILOT.md`，未改产品、日常库或执行Git写操作。

### 2026-09-30：LiveXiv新论文检索验收及公式自动触发修正

- 排查否定了“局部BM25元数据是主因”和删除“论文”泛化译词两种猜测，均未写入产品。另用预先固定的ICLR 2025 LiveXiv三题在176块隔离库验收，本机4B初次三题均因关键证据缺席而拒答，不据此评价模型总体准确率。
- LX02的“更新基准时如何避免重测”误触公式路径，四段全成公式；只收窄这一条`更新…如何/什么`识别。修后正文段38首位，单次回答正确给出重测3–5个模型及约75%–85%节省，但未解释IRT细节。LX01筛选流程和LX03 Claude筛选偏差仍缺证，未增加特例。现有基准问题的公式分类无变化，283项测试通过；未改日常库或执行Git写操作。详见`evaluation/PAPER_PILOT.md`。

### 2026-09-30：LX01固定证据与跨语言检索核查（无产品改动）

- 已核对第5页段28/29/30；原中文请求全缺，固定三段后本机4B单次答对两步筛选及30%、38.5%、6.15%。启用既有来源局部、问题分解或hybrid不能同时恢复三段；含英文原文术语的等义问法则能送29/30/28。已缓存bge-m3只做离线向量排序，中文问题的关键段列第2/5，仍不能保证四段完整覆盖。
- 因替换默认模型会增加体积且证据仍不齐，本轮不改产品、不新增逐词别名或翻译调用。LX03英文问法也未恢复目标段，不能把所有漏证归于同一语言问题。诊断及下一步见`evaluation/PAPER_PILOT.md`；未动日常库或执行Git写操作。

### 2026-09-30：LX03召回断点与固定证据归因（无产品改动）

- 目视复核LiveXiv第7–8页：偏差结论在段46。原中文题同源dense仅排第46，局部词面前六也没有它；hybrid候选50以及本机已有权重的重排器均未将46送入四段，不能靠简单跨页拼接或换模式修复。
- 固定给出段45/46后，本机4B一次说对Claude参与筛选，但无据写成也参与生成，并把段46内容错引为片段1；因此除检索漏证外仍有生成归因问题。没有加入单题拼接、提示或后处理。详细trace见`evaluation/PAPER_PILOT.md`；未改产品、日常库或执行Git写操作。

### 2026-09-30：跨题分层漏证与AF02真实回答复核（无产品改动）

- 复用29/33旧开发题和两套隔离库，逐层统计源库、合并候选、最终四段的标注词串覆盖。247块库为59/55/49（共61事实），358块库为71/66/60（共74事实）。这只是词面代理；R01/R02的“缺失”存在改写和跨块断句，不能直接按分数调排序。
- AF02摘要证据虽在候选第8位、未进最终四段，但本机4B原请求已正确回答四者角色；把摘要段提到首位再答一次也正确，未证明产品缺陷或排序收益。故不写产品补丁，不新增门禁/基准。详细片段与回答见`evaluation/PAPER_PILOT.md`；未动日常库或执行Git写操作。

### 2026-09-30：跨两篇论文排除直接换嵌入与分句优先候选（无产品改动）

- 在LiveXiv/UAEval4RAG共六题上只读比较当前小模型与本机BGE-M3的原文段余弦排名。M3改善LX01/LX03部分目标，却仍漏LX01第29段与UAE01第7段；UAE03当前小模型已召回目标却在最终候选丢失。不能以向量排名局部提升切换默认嵌入模型。
- 可选分句“整句优先”候选未增加六题最终目标证据，还改变跨论文P10的来源专属取证；候选代码和测试已撤回。保留既有功能，后续需将初始召回与最终选择分别验证。详见`evaluation/PAPER_PILOT.md`；未改日常库或执行Git写操作。

### 2026-10-01：收窄无关趋势拒答句；P12归属错误仍在

- 当前隔离库复测P08/P12各一次：P08三类边及α=60、m=10、`|i-j|≤3`均答对；P12虽有实验边界原文，仍把另行提供的Llama 3.2图像摘要误说成实验模型输入，且套用无关的“趋势依据不足”固定句。
- 仅修改系统提示中趋势规则的一行：只在询问升降/因果趋势时检查对比依据，其他问题不套固定拒答句。内存对照使P12无关模板消失，真实变化题Q07仍答对约30%且未外推；不宣称P12整题已修好。一条现有测试断言及283项普通测试通过。详见`evaluation/PAPER_PILOT.md`；未动日常库或执行Git写操作。

### 2026-10-01：P12脚注与4B/8B归属对照（无产品改动）

- 正式论文实验脚注已入隔离库第25块，明确实验用图注、不直接处理图像；默认P12送模61/58/91未含此块。固定补入25后，4B仍误称另行提供的图像摘要是实验输入；8B原三段仍暗示实验解答依赖摘要，8B加脚注的一次回答才完整守住范围。
- 这只是单题每条件一次的诊断，P12继续记部分正确；不据此全局增加脚注、改默认模型、删原文或加后处理。原文与四条件详见`evaluation/PAPER_PILOT.md`；未动日常库、云端或执行Git写操作。

### 2026-10-01：修正可选相邻窗口的反向跨页拼接

- 命中跨页句子后半段时，向前找前页邻块的续句判断传反了文本顺序；现按原文顺序判断，并在跨页窗口元数据中保留两页页码，避免只引用续页。新用例先失败再通过，原正向用例及全套284项测试通过。
- 固定证据将本机4B温度降至0的LX03、MT06-en、P12仍有归因或边界错误，AF01出现错引；不改默认温度或增加归因门禁。详情见`evaluation/PAPER_PILOT.md`。`parent_window`仍默认关闭，未动日常库、云端或执行Git写操作。

### 2026-10-01：补回中文“偏差”问题的英文原文取证

- 仅在现有来源局部补证中增加通用“偏差→bias”及触发词；LX03由缺段46变为送122/46/133/142，本机4B答对筛选偏差主因，但仍借无关附录扩写，记部分改善而非完全正确。SciDQA另一道偏差题四段不变。
- 全局去除BM25词尾句点虽修词元匹配，却让LX03再次丢掉46，已撤回，不扩展为分词重构。新增一条普通回归；全套285项测试通过。详见`evaluation/PAPER_PILOT.md`；未动日常库、云端或执行Git写操作。

### 2026-10-01：补回LiveXiv两步筛选和比例证据

- 只加入“盲测→blind test”、“错误答案→incorrect/ground-truth”及“盲测”的现有单篇补证触发。LX01由缺28/29/30变为送28/29/50/30；本机4B单次答出两步和30%、38.5%、6.15%。只加“筛选”触发或泛化筛选译词均无法取齐，未采用。
- 两套旧隔离库29/33题的最终片段ID无变化；这不是泛化证明。新增一条普通回归，全套286项测试通过。未扩大窗口、换模型、重建日常库或执行Git写操作。详见`evaluation/PAPER_PILOT.md`。

### 2026-10-01：Table-R1 独立论文问答核验（无产品改动）

- 目视核对PDF第3/14页后，用预写问题在一次性124块隔离库真实运行；第14页关键段进入上下文，本机4B单次正确回答RLVR 48,563与SFT 33,601、数据来源和筛选关系。问句未触发上一轮新增的“盲测/错误答案”词表，不能把这次正确当作该映射的迁移证明。
- 同题单另三题：Table 1/2两道结构化表格查值正确；奖励机制复合题仅答对三种准确性奖励，漏答第4页已入库的`<think>...</think> <answer>...</answer>`及JSON结构。默认四段缺相邻格式定义段。通用格式译词造成准确性奖励反向缺证，已撤回；窄译词加可选相邻窗口虽补两段，4B仍漏`<think>`，不足以升为默认修复。详见`evaluation/PAPER_PILOT.md`；日常库、云端和Git未写。

### 2026-10-01：修正显式双问句的局部词面补证顺序

- 单篇资料、多问号问题原先把整句BM25前六整体置于最终候选前，后一个问句的证据即使已检出也可能被四段上限挤掉。只对明确分开的问句轮流取局部BM25前两项，再由整句结果补足；新增“准确性奖励”“格式奖励”两处通用直译，不改单问句、上下文上限或模型调用数。
- Table-R1奖励题现将两个关键正文块24/25排在前两位；本机4B答对三类奖励及answer内JSON，但仍漏`<think>`，记部分正确。独立ChainRAG双问题的100/3/3000事实答对；TableEval双问题改前改后均缺证，不记作回归或收益。287项普通测试通过，详情见`evaluation/PAPER_PILOT.md`；未动日常库、云端或执行Git写操作。

### 2026-10-01：TableEval 第5页漏证定位（无产品改动）

- 原文第5页同段明确给出图像/文本差约1–13%、四种文本格式最大差约4%，入库块31无损；默认向量排第3，却被前置的六个局部词面候选挤出，模型缺证拒答。普通RRF融合和仅删来源名称均未使目标进入四段。内存中加单题词面映射虽使4B答对，但不作为共享修复提交；另一种通道仲裁候选也未在UAE03获得独立收益。
- 仅更新诊断记录，不加词条、门禁或答案清洗；详见`evaluation/PAPER_PILOT.md`。日常库、云端和Git未写。

### 2026-10-01：双问句共享证据避免被局部词面候选挤出

- 单篇、两个明确问句的dense检索中，仅当同一原文块同时进入两个子问句前三和整句前四，才把它置于局部词面补证之前。TableEval原缺失的第5页块31进入送模首位，本机4B单次答出1–13%及约4%；Table-R1、ChainRAG对照题送模顺序不变。Table-R1仍有`<think>`漏答，未在此轮修正。
- 新增交集/无交集回归测试，288项普通测试通过。没有单论文译词、固定通道配额、答案补丁或模型重试；隔离库在运行结束后清理，未动日常库，也未执行Git写操作。详情见`evaluation/PAPER_PILOT.md`。

### 2026-10-01：排除 PDF 页脚并保留跨页清单

- PDF 导入时仅排除解析器标记的`page-footer`文本；来源局部列表题若候选正文以冒号结束，且紧邻的下一页无标题块构成续句，就把引导段与清单段一同提前。目视确认UAEval4RAG原PDF第1–2页确为此结构。新隔离库UAE01送入6/7并由本机4B列出六类；UAE02/03仍缺证，Table-R1的`<think>`仍漏答，未做单题答案补丁。

### 2026-10-01：缩写断词与独立旧题回归

- 目视核对YESciEval原文的`LLM _gen_`/`LLM _eval_`与提问中的`LLMgen`/`LLMeval`不一致，在共享查询词函数中保留完整词并拆出大写前缀、小写后缀。新隔离库实测两段任务定义进入前两位，本机4B一次答出任务、1–5量表和JSON格式；RAGTruth三道旧题送模片段未变化，290项测试通过。
- UAE02可选分句单次答对双定义，默认仍缺acceptable ratio原文；UAE03两种设置仍漏通用最优配置结论。未把分句默认化，也未加入单题译词、答案补丁或新门禁；详见`evaluation/PAPER_PILOT.md`。未触及日常库、云端和Git写操作。

### 2026-10-01：修正来源内零分分句挤占与 ratio 触发

- 来源内BM25不再把未命中的分句零分块提前；单词`ratio`可触发已有来源局部补证，词边界避免误命中`generation`。新隔离库UAE02送到第4–5页两项指标定义，本机4B一次答出定义与区别；UAE01、UAE03及YESciEval旧题的送模片段顺序未变。UAE03仍缺结论，不加专属答案补丁。
- 292项普通测试通过；详细断点和单次实验局限见`evaluation/PAPER_PILOT.md`。未修改日常资料库、云端或执行Git写操作。
- TableEval、ChainRAG对照问答保持正确，289项普通测试通过。用户日常资料库未改；已有PDF需重新上传才会应用新解析。无Git写操作。详见`evaluation/PAPER_PILOT.md`。

### 2026-10-02：固化当前19道开发题快照

- 汇总9道已恢复回归对照和10道当前失败题，逐题记录当前判断、失败阶段及已知最终片段；该集合只用于诊断，不报告准确率。41道旧开发题的词面扫描与人工排假阳性确认Q02、F02、F03和UAE03均为dense已召回、最终四段漏证。
- 论文试验runner现在允许重复传入`--papers-dir`，直接从主论文目录和UAEval4RAG目录解析同一题单；未复制PDF、未新增runner或门禁。下一步比较四道同类失败的共享候选结构。
- 内存候选“词面前四与dense前四重合时dense优先”虽恢复F02，却使41题词面事实命中从114/141降为87/141，故撤回且不改产品；Q02、F02、F03、UAE03暂无一个共同且不伤旧题的简单排序信号。
- 固定请求的8B对照恢复MT05核心过滤条件但仍无据扩写，MT06仍合并正文`≥20`与附录`>20`。默认4B即使增加明确冲突保留指令也继续合并阈值；提示候选撤回，不更换模型或添加答案后处理。
- 8B在AF01单次正确引用，LX03仍无据扩写，Table-R1仍漏`<think>`；结合既有P12结果，模型放大没有形成跨六题的统一生成修复，默认4B与产品提示保持不变。
- 合并候选池的纯cross-encoder虽恢复四道选段失败，仍使LX01、UAE01、UAE02、YESciEval及P12丢关键证据；与原顺序RRF后仍退化LX01/UAE01等且漏F03。两种候选均撤回，不启用reranker或新增分数gate。
- 安装不同家族、同规模的`gemma3:4b`，固定六道已有请求完整运行两轮。MT05、LX03、AF01两轮均通过；Table-R1第一轮仍漏闭合`</answer>`、第二轮完整给出`<think>...</think> <answer>...</answer>`及JSON；P12仍混淆图像摘要与实验输入，MT06仍合并正文`≥20`和附录`>20`。该候选进入后续更大固定请求回归，但当前不更换默认模型、不改提示或加答案后处理。原始结果和逐题判断见`evaluation/PAPER_PILOT.md`。
- 用当前解析器重导入七篇PDF并重跑九道历史控制后，仅P08、TableEval、UAE01、UAE02、YESciEval仍完整；MR01、MR03、MR04、LX01分别缺当前块32、28、25、30中的必要证据。单篇隔离重建得到相同片段，说明旧快照沿用了过时请求。四个目标都在来源局部BM25前六但未进最终四段，已有问题拆分也不改变结果；19题快照已校正为5道新鲜控制和14道当前失败。
- 对九个Qwen请求原样回放Gemma 4B：五道完整题的核心事实保留，但P08、TableEval、YESciEval出现缺引用或错引；四道缺证题中又无据补出100次迭代、五种技能和30%/30%等错误。Gemma不再作为默认替换候选，产品模型与生成路径不改。完整配对trace见`evaluation/PAPER_PILOT.md`。
- 在同一当前解析库补入RAGTruth和SciDQA，统一审计5道新鲜控制与8道最终选段失败。八题必要块在完整合并候选中分布于第5–17位，来源局部BM25为第4–6位或未进前六；ChainRAG新解析送模序号校正为28/16/46/60，目标块仍为30。
- “问题要点覆盖加同页/邻接”只有借助金标准答案才能在离线模拟中补齐八题；去掉金标准条件会给控制题引入杂块并可能截断已有证据。该候选淘汰，不改排序、上下文上限或gate。后续停止在这13道已见题上继续调参，先建立新的预写检查组验证断点是否复现。完整名次和限制见`evaluation/PAPER_PILOT.md`。
- 在任何检索前冻结MiMoTable与MedExQA的6道新题，题单SHA-256为`c1bf60e918944453d5034fa9c008336c30b74a10ad9d5e36c61001ae023e12c3`；首次真实应用路径上下文完整3/6。两道MiMoTable题分别漏同页流程结尾和被独立页码隔开的跨页续段，MedExQA评分规则块在合并候选第13位而未进四段。
- 现有问题拆分不恢复缺口；parent-window会使一题已有覆盖从7/7降到1/7；hybrid只恢复评分题，却使另一题从6/7降到1/7。没有统一安全开关，未调用生成模型或修改产品。冻结题单与逐题上下文记录见`evaluation/benchmark/paper_selection_signal_check_cases.jsonl`和`paper_selection_signal_check_contexts.jsonl`。
- 临时去除MiMoTable页底纯数字页码使解析块数从91减为89，但最终选段仅序号平移，续段事实仍缺；放宽混合正文/图形块判定也不改变三题上下文。两项均无实际收益，未写入产品或测试。
- 在检索前冻结4道PIER-QA独立确认题（SHA-256 `0c330f3c44ed63aa93b26f112eb400a1fbc41835887c3e5d3fa9d14dddb616c9`）；新63块隔离库首次上下文人工核对4/4完整，没有复现合并后选段缺失。自动覆盖对RAG-aware题的partial为词面匹配假阴性，完整事实均在已送入的块26；因此不改产品排序，转向另一篇独立论文确认候选召回。
- 随后在检索前冻结4道IRPAPERS题（SHA-256 `41c040650bedab814bdae122ce830dc0576c64ae330c419d2305e6a89f4a5c24`）。首次默认路径人工核对得到3道合并后选择缺失、1道候选召回缺失；decomposition无变化，parent-window和hybrid均有退化，英文标识符临时补证只恢复t-SNE一题且扰动既有题，均不提交产品代码。下一步分别追踪同节候选生成与词面通道整组前置。
- 收窄并提交来源局部邻块信号：只对词面第2、3名，在命名同节或无标题同页补入显式英文标识符时插入一个相邻正文，候选总数仍为6。IRPAPERS的MUVERA与t-SNE题上下文均由partial恢复为full；5道当前控制、第一组6题及PIER-QA 4题无核心事实退化。LX01补到比例块却挤出盲测块，仍保留partial。新增一条共享词面回归，294项普通测试通过；下一步用新预写论文题做未见确认。
- 邻块候选的MT-RAIG未见确认组在检索前冻结（SHA-256 `e90fe5e2676b36167bf6960ba15690d6f8059062a6168ad14a485d674a1e49f4`）。首次上下文仅1/4完整；候选在四题最终上下文中均未触发，撤回后片段逐项相同，故按YAGNI撤回产品代码和专属测试。保留题单与上下文证据，下一步只读审计跨页表格隔断及正文/附录同名候选优先级。
- 后续只读审计中，词面锚定section expansion只找回`expand_facts`跨页块30；排除纯标题块也只使EVAL正文和噪声结论名次小幅前移，仍不进入候选。两项都未修复实际调用，未新增词条、扩大K或修改产品；全套293项普通测试通过。
- 在检索前冻结4道REAL-MM-RAG中英术语题（SHA-256 `bf1b722d6ef7e9dffa10992ff814185bf5e71fc01aaf18f954982870226421a0`）。默认104块隔离库人工上下文2/4完整，两道失败均为候选召回；M3把两题必要块送入合并候选，其中一题最终由partial变full，另一题转为四段选择缺失。
- M3重建当前5道回归控制的四篇论文后，五题最终块集合逐项不变；F02、UAE03也无改善。其本地缓存约4.3 GB、1024维，104块编码约20.9秒，而默认模型约92 MB、512维、1.0秒。收益尚不足以承担默认切换成本；保留已有环境变量作为实验入口，不改产品模型或新增开关。
- M3隔离库继续补齐当前19道开发题的全部10篇论文，共1282块；5道控制无必要事实退化，MR01与F03新增目标块并达到完整上下文，其余现有失败未形成可确认改善。这里只确认2/14上下文改善，未调用生成模型，不把它计为答案修复；默认模型仍不切换。
- 约477 MB、384维的multilingual MiniLM在4道冻结新题上达到与M3相同的候选收益且编码更快，但5道控制中P08丢SA定义与`|i-j|≤3`，TableEval丢1–13%和约4%结论。因2/5控制实质退化，候选淘汰，不扩大19题实验或写产品配置。

### 2026-10-03：SciAssess最终四段选择确认（无产品改动）

- 在任何检索前冻结4道SciAssess多事实题，SHA-256为`428d9bdfc9cb49032a28ea6d3381c0813a8301c1c914c850798d9e38321385ef`。新184块隔离库首次上下文四题均为partial；两题必要块未进完整候选，另两题目标块已在合并候选第5/12位却被四段上限截断。
- section expansion整体提前使两道新题降为zero并扰动TableEval控制，候选淘汰。仅让明确limitation问题复用现有同源BM25可把局限题从6/9恢复到9/9，五道当前控制和三道旧探针不变，但只有一项独立收益，未达到采用门槛。
- 新增冻结题单的逐题上下文、目标块名次和失败阶段记录；产品排序、上下文上限、默认嵌入和生成路径均未修改，也未调用生成模型。
- 继续测量两道候选召回缺失：默认dense/BM25均把至少一个必要块排到第23–94位；M3虽把部分目标前移，仍分别留下第17和第39位的必要块，无法进入12段候选。两题分别受跨栏混排和长段切分影响，没有共享的低风险修复；不扩大K、不恢复已撤回邻块规则，也不重建M3产品索引。
- 真实重解析把正文块上限从1024临时改为1280后，连续改进段虽成功合并，四题仍无新增完整覆盖，实验流程题还降为zero；候选与临时库已删除。代码差异审计确认新增helper均有调用，旧gate/启动脚本及对应测试已删除；进一步只清理仓库零引用的临时探针，不删除仍承担导入去重、文件校验或视觉记录的哈希与安全措施。

### 2026-10-03：当前界面真实启动冒烟（无产品改动）

- 用空隔离资料库实际构建普通模式界面，确认92个组件、38条事件依赖及资料库、问答、大纲、自测、模型设置五个页面均存在；桌面托管模式正确隐藏模型设置并保留退出按钮。
- 随机本机端口启动Gradio后，首页HTTP返回200且服务正常关闭。完整293项普通测试继续通过；未因冒烟结果新增门禁、baseline、包装层或重复测试。
- 另用临时TXT、真实默认嵌入和真实Chroma完成添加、限定来源问答与确认删除：唯一标记`BLUE-482`进入实际上下文并触发一次生成请求，删除后集合恢复为0；临时目录自动清理，未触碰日常知识库。

### 2026-10-03：重建并验证当前 macOS 内部应用包

- 用当前代码重建 Apple Silicon `个人知识库助手.app`，包内继续使用本地 Qwen3 4B GGUF、BGE 小模型和 `llama-server`；`Info.plist`、arm64 架构及 ad-hoc 深度签名校验均通过。
- 直接启动包内模型服务，`/health` 返回 200，真实聊天补全返回 `OK`，随后服务正常退出且退出码为 0。再从 `.app` 完整入口启动模型与界面，Gradio 首页返回 200（119,433 字节），发送正常终止信号后主进程、模型服务和资源跟踪子进程均无残留。未新增发布 gate、冻结清单或重复启动器。
- 验证后删除旧 `Sci-RAG` 两份应用副本、PyInstaller 中间目录、临时 spec 及同内容的无扩展分发目录，只保留可运行的 `dist/个人知识库助手.app`。当前仍是内部 ad-hoc 签名包，正式对外分发时再做 Developer ID 签名、公证和许可证归档。

### 2026-10-04：合并后首轮隔离内测与空文档反馈

- 从已合并的 `develop` 启动包内 Qwen3 4B 服务，在临时 Chroma 库真实导入合成 TXT、合成 DOCX 和 UAEval4RAG 原 PDF；三者分别生成1、2、106个文本块。TXT 唯一标记、DOCX 日期与表格预算、PDF 六类不可回答请求均进入正确来源上下文；跨 TXT/DOCX 问答同时引用两篇资料并答出两项事实。
- 真实生成了限定来源学习大纲和5道结构完整的自测题，正确答案提交得到5/5；重新创建运行时后库存和109个块保持不变。删除确认保护生效，删除后原资料不可检索，重新导入成功，最后确认删除全部资料后集合为0；临时目录自动清理，未触碰日常知识库。
- 空 TXT 原先不写入任何块，却返回“成功添加0个文本块”。共享导入入口现对零块文档抛出明确错误，页面返回“添加失败：文档中没有可导入的文本”；已有旧解析仍在失败后保留，原安全顺序不变。
- DOCX 原文和送模片段均为 `2031-04-17`，默认英文复合问法连续3次生成 `2031-04--17`；要求逐字保留、中文问法及仅问日期时均正确。该现象属于本机4B生成格式波动，当前没有跨案例证据支持全局提示、重试、答案清洗或新门禁，继续依赖原文面板核对。
