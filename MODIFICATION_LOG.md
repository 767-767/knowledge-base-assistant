# Sci-RAG 修改记录

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
