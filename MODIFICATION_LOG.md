# 修改记录（Modification Log）

> 记录本项目从"通用学习工作台"改造为 Sci-RAG 的全过程。按时间倒序排列。

> 2026-08-31 Phase 12 全量生成门禁：在 502 块隔离 Chroma、Hybrid+固定 reranker、
> routing/query decomposition/parent-window/spatial evidence 开启且 `context_k=10` 的
> 同一配置下完成五论文 53 题两轮共 106/106 道 DeepSeek 调用，0 API 错误，provenance
> 53/53 完整。两轮词面答案 fact macro/micro 分别为 `0.9450/0.9315`、`0.9434/0.9247`，
> 均为 `47/53` full；与 Phase 7.1 一轮结果对照时 `scidqa-07 zero→full`、`mgno-02
> partial→full`，但 `mgno-03` 有一轮 full→partial。重点抽查确认 `scidqa-07` 四配置
> 答案正确；两轮 context 稳定率 `50/53`、答案逐字稳定率 `19/53`，不能据此宣称语义
> 准确率或切换默认开关。完整回答/trace 仅写入 `/private/tmp`，新增
> `PHASE12_FULL_GENERATION_GATE_HANDOFF.md`；未运行 RAGAS、未改生产 ChromaDB 或依赖，
> 未执行 git add/commit/push。

> 2026-08-31 Phase 11 方法段证据修复：将已有同小节扩展限制性地接入唯一文档路由，允许
> 多论文库中明确指向单一 source 的复合/列表问题恢复同标题路径的邻近方法块；无唯一 route
> 的多来源问题仍跳过扩展，来源边界、6 块上限和 context-k 均不变。新增“实验配置/配置”
> 标题别名与量化列表问题识别，通用于方法设置、步骤和管道类问题。固定 502 块隔离库、
> Hybrid+固定 reranker、context-k=10 下，`scidqa-07` 定向 DeepSeek 复核 1/1 成功，
> 恢复 closed-book/title-abs/RAG/full-text 四项配置，answer-fact 审计 1/1 full，未混入
> 其他论文。离线测试 148/148、`py_compile`、`git diff --check` 通过；未重跑全量 53 题，
> 未改默认开关、生产 ChromaDB 或依赖，未执行 git add/commit/push。详见
> `PHASE11_METHOD_EVIDENCE_HANDOFF.md`。

> 2026-08-31 Phase 10 定向生成修复：为生成提示增加表格值与拒答一致性、拒答前扫描全部
> 片段和局部缺失不扩大为整题拒答的通用约束；表格块标签显示解析出的 `Table N` metadata。
> 在相同 502 块隔离库、Hybrid+固定 reranker、context-k=10 下，SciDQA `scidqa-03/04`
> 重跑 2/2 消除“给出值后又否认 Table 2”的矛盾。公式/算法证据开关仍默认关闭，但其
> 明确意图门控扩展到初始化、残差/平滑迭代、限制/延拓、stride 和 cycle；开启该开关后
> MGNO `mgno-03/04` 定向生成 2/2 full，分别恢复零初始化/残差和 stride=2/反卷积/
> Backslash-cycle/V-cycle。离线测试 147/147 通过；本轮未重跑全量 53 题、未改默认
> context-k、未改生产 ChromaDB，未执行 git add/commit/push。完整交接见
> `PHASE10_TARGETED_REPAIR_HANDOFF.md`。

> 2026-08-31 Phase 9 生成答案语义复核：对当前 Hybrid+固定 reranker 的五论文 53 题生成结果
> 做一轮内部逐题复核，结果为 `43 correct / 7 partial / 3 incorrect`（81.13%/13.21%/5.66%）。
> 词面 required-fact macro/micro=`0.9261/0.8973`、完整覆盖 `46/53`，与语义结果的差异确认
> 词面审计不能替代正确性判断。非完整案例包括 DrugR 工具/阈值与 Pareto 细节遗漏、SciDQA
> Table 2 自相矛盾和四配置错误拒答、MGNO 公式/方法拒答或域符号/单位遗漏、AlphaFold 3
> 静态结构限制遗漏。完整理由写入 `PHASE9_SEMANTIC_REVIEW_HANDOFF.md`，JSONL 仅保存在
> `/private/tmp`；本阶段未运行 RAGAS、未改生产 ChromaDB、未执行 git add/commit/push。

> 2026-08-31 Phase 7 direction audit：新增 `PHASE7_DIRECTION_AUDIT.md`，基于当前 PDF 解析、
> 生成调用、依赖和五论文语料审计多模态、工具调用与 Graph-RAG。确认图片未持久化、没有通用
> tool registry/执行器或图谱组件，但已有受限确定性表格查值路径；因此只有测得算术/单位缺口
> 才扩展白名单工具，只有 image-only 失败集达到门槛才做隔离 OCR/VLM，Graph-RAG 需先有跨
> 文档多跳需求。未新增依赖、未调用外部 API、未启动 Gradio、未改生产 ChromaDB、未执行 git 操作。

> 2026-08-31 Phase 7.1 当前版本整体验证：修正 `evaluation/generation_stability.py`，允许命令行
> 显式记录 reranker model/revision、批大小、长度、设备和 RRF 参数；在隔离 502 块 ChromaDB
> 上用 Hybrid+固定 `BAAI/bge-reranker-base` 跑完五论文 53 题，53/53 成功、0 错误。词面答案
> 审计 macro/micro=`0.9261/0.8973`，完整覆盖 `46/53`；53/53 行 provenance 完整，但单轮
> 不能估计答案文字稳定性。对 8 道边界题做 `context_k=50` 限定对照，6/8 词面完整，证明尾部
> 候选有价值但不能保证工具/阈值被生成，也不能据此修改网页默认 context-k。为避免格式误报，
> benchmark 为 `3×3`、stride=2、扩散模块和闭合构象加入人工确认的等价别名。新增
> `PHASE7_1_CURRENT_GENERATION_HANDOFF.md`；完整 JSON 仅写入 `/private/tmp`。本阶段未运行
> RAGAS、未启动 Gradio、未改生产 ChromaDB、未执行 git add/commit/push。

> 2026-08-31 Phase 7.2 确定性工具门控：审计五论文 53 题，未发现真实单位换算、算术或外部
> 查值缺口，不引入通用 Agent/tool registry。修复一个普遍的科学表格表达问题：结构化行/列
> 查值现在保留 caption 中明确声明的共享单位/尺度（如 `×10⁻²`），并把 table caption 放入
> 返回 trace；无单位提示的旧答案格式不变。隔离 502 块数据库 smoke 通过，MgNO Table 1
> 结果带出 `×10⁻²`，全套测试 145/145 通过。新增 `PHASE7_2_TOOL_AUDIT.md`，规定至少
> 20 道操作题和 5 个可由白名单工具修复的稳定失败才进入工具实验。本阶段未调用 DeepSeek、
> 未启动 Gradio、未改生产 ChromaDB、未执行 git add/commit/push。

> 2026-08-31 Phase 7.3 多模态/Graph-RAG 门控：结合五论文 53 题和当前 Figure/公式回归，未发现
> image-only 失败集或跨文档多跳需求。保留现有 born-digital 文字/坐标通道，不新增 OCR/VLM、
> 图片资产或图数据库依赖；新增 `PHASE7_3_DIRECTION_GATE.md`，规定多模态需至少 10 道人工核对
> image-only/复杂图题，Graph-RAG 需至少 5 道纯文本稳定失败的跨文档多跳题，且均须保留
> provenance。Phase 7.3 停止在门控审计层面，默认路径不变，未调用外部 API、未启动 Gradio、
> 未改生产 ChromaDB、未执行 git add/commit/push。

> 2026-08-31 Phase 8 整合回归：汇总 Phase 7.1–7.3 结果并新增 `PHASE8_INTEGRATION_HANDOFF.md`。
> 当前代码/测试/文档通过 145/145 单元测试、py_compile 和 `git diff --check`；隔离 502 块库的
> Hybrid+固定 reranker 53 题生成 53/53 成功、0 错误，provenance 53/53 完整。确认默认
> context-k、公式证据、多模态、工具调用和 Graph-RAG 均未切换；生产 ChromaDB 未改，完整
> 生成 JSON 留在 `/private/tmp`。交接文档列出用户 review 文件、检查命令和后续触发门槛；本轮
> 未执行 git add/commit/push。

> 2026-08-30 Phase 6.8 formula-evidence guard：将显式公式候选收紧为公式/方程/卷积核尺寸/
> 离散系统形式问题，要求至少两个语义词命中并限制 PDF 符号噪声权重；候选仅在唯一来源范围
> 内提升，未路由的多论文问题不跨来源注入。五论文全局 @10/@50 指标与基线完全一致；单文档
> MGNO @10 完整覆盖由 8/11 提升为 9/11，`mgno-02` 的 `3×3` 可被候选覆盖，但全局未限定
> 问题仍缺该事实，因此默认开关保持关闭。新增歧义来源回归测试，142 项测试全通过；未运行
> RAGAS、未调用外部 API、未改生产 ChromaDB、未执行 git 操作。

> 2026-08-30 Phase 6.6 trace provenance：`evaluation/generation_stability.py` 现在为新生成行记录
> 不含密钥的 embedding、检索、context-k、reranker revision、生成模型等运行配置，以及涵盖
> 关键源码文件的指纹；新增 `evaluation/audit_generation_trace.py` 离线审计重复 case 的
> config/context/metadata/答案稳定性。历史 trace 缺少这些字段时明确标记为 provenance 不完整，
> 防止将不同源码或模型条件的结果混合比较。另修正 composite 答案补充逻辑，跳过训练语料和
> corpus 统计等与问题无关的高数字密度行；未改变默认检索或自动 retry 行为。

> 2026-08-30 Phase 6.7 candidate-pool audit：修正 `evaluation/benchmark_retrieval.py` 的多-k
> 口径，使 table/figure/parent-window 后处理按每个实际 k 独立计算；新增回归测试防止请求 @50
> 改变 @10 指标。在固定五论文、Hybrid+固定 cross-encoder 条件下，修正后的 @10/@50 fact
> macro/micro/full 为 `0.936/0.932/0.887` 与 `0.995/0.993/0.981`；@50 只剩 `mgno-02`
> 的 `3×3` 公式表面未召回。结果说明候选池尾部有价值但不能直接把 50 块交给默认生成；未切换
> 默认 context-k，下一步优先处理公式表示/查询召回。

> 2026-08-30 Phase 6.4 证据答案校验：在 `sci_rag_core.py` 新增不读取 gold 的
> `validate_answer_against_evidence()`，并新增 `evaluation/validate_answer_evidence.py` 和
> 对应测试。规则只检查与问题意图邻近的文字证据标记，显式表格、空间图形、公式/算子语义
> 返回 `not_applicable`；`review` 仅是人工/显式重试提示，不代表答案错误。对 Phase 6.3 的
> 106 条稳定性记录得到 36 条结构化表格、4 条空间图形、6 条公式语义、58 条 `ok` 和 2 条
> 同一 `drugr-09` 流程遗漏提示。新增 `SCI_RAG_ANSWER_VALIDATION` 默认关闭的运行时诊断，
> 开启时只附加提示，不自动重写或重试；未运行 RAGAS、未改生产 ChromaDB、未执行 git 操作。

> 2026-08-30 Phase 6.5 受控二次生成门控：新增不执行网络请求的
> `build_evidence_retry_prompt()`，并在固定 Phase 6.3 reranker 条件下对 `drugr-09` 做一次
> 原答案/retry A/B。由于检索上下文缺失明确的 `4,855` 段落，二次生成没有改善，证明答案
> retry 不能恢复该固定上下文中的检索缺失证据；该历史 trace 未记录完整运行配置或源码指纹，
> 不能直接外推到当前代码。当前代码单题复测已召回 4,855 段落；自动 retry 仍保持关闭。完整记录见 `PHASE6_5_RETRY_GATE_HANDOFF.md`
> 和仓库外 `/private/tmp/sci_rag_phase65_drugr09_retry_audit.json`，未运行 RAGAS、未改生产
> ChromaDB、未执行 git 操作。

> 2026-08-30 Phase 6.3 生成稳定性：新增 `evaluation/generation_stability.py`，固定 502 块
> 隔离库和 Phase 6.2 检索配置，对 53 题各运行两轮，共 106/106 成功。两轮上下文 ID/metadata
> 53/53 完全一致；答案逐字一致 16/53；事实状态仅 `mgno-02` full→partial、`af3-02`
> partial→full 两题迁移。两轮答案事实 macro/micro/full 约为
> `0.80/0.774/0.717`。结果证明检索可复现但生成措辞有变异，不能将词面审计当作语义正确率，
> 默认 temperature 和 API 重试策略保持不变；完整交接见 `PHASE6_3_GENERATION_STABILITY_HANDOFF.md`。

> 2026-08-30 Phase 6.2 Figure 坐标证据：新增默认关闭的
> `SCI_RAG_SPATIAL_FIGURE_EVIDENCE`/`--spatial-figure-evidence`。它只读取 born-digital PDF
> 已有文字块坐标，不运行 OCR、不读取或持久化图片；显式 Figure N 查询按图号和既有 source
> route 注入坐标证据，普通 dense/BM25/reranker 候选排除 figure chunks。五篇论文新增
> 23 个图块，隔离库共 502 块；53 题 @10 fact macro/micro/full 保持
> `0.936/0.932/0.887`，目标论文/页/Table N 保持 `1.000/0.929/1.000`。AlphaFold 3 Figure 1
> 定向 DeepSeek 复测正确关联 `25/38/8`，不再把相邻 Glycosylation 的 `28` 当作 CASP15 RNA。
> 这只是单个 born-digital 图形回归，不证明通用多模态能力，开关仍默认关闭，生产 DB 未改。
> 最终离线回归 120/120 通过，五论文文件及 53 题 benchmark 校验通过。

> 2026-08-30 Phase 6.1 parent/window 有效上下文：新增默认关闭的
> `SCI_RAG_PARENT_WINDOW`/`--parent-window`。它不改变 top-k、锚点 ID 或来源页，只在前两个
> 文本锚点内拼接同来源同页邻块，跳过表格、参考文献、跨页/跨来源和已入选块，并记录 window
> chunk IDs/indices/字符数，并跳过图像/OCR 块的窗口扩展。五论文 53 题 @10 fact macro/micro/full 从
> `0.881/0.871/0.811` 提升到 `0.936/0.932/0.887`（47/53），改善 4 题且零退化，
> 目标论文/来源页/Table N 仍为 `1.000/0.929/1.000`；总字符增量 60,388。真实隔离 DrugR
> Chroma smoke 建立 101 块、元数据 101/101 完整，一次查询保持 10 个 ID 并扩展 2 个锚点；
> fake client 证明返回 context 与 prompt 一致，未调用 DeepSeek。SQLite 只读核验发现生产
> 104 块旧库没有 page/chunk_index，所以该开关在旧库安全 no-op，不重建生产库。随后在隔离
> 479 块五论文库完成 53×2 端到端生成；自动答案事实完整覆盖从 40/53 增至 45/53，内部
> 人工复核从 `40 correct / 11 partial / 2 incorrect` 变为 `43 / 8 / 2`，真实改善 4 题，
> 未发现 parent/window 相对语义回退；`table-llm-10` 为词面别名假改善，`af3-06` 两组均有
> 图像/OCR 数值关联错误。所有回答、对照和人工复核均在 `/private/tmp`，开关仍默认关闭。
> 本阶段离线回归 113/113 通过，未运行 RAGAS，
> 未启动 Gradio，未改生产 ChromaDB，未执行 git 操作。

> 2026-08-30 Phase 6.0 检索路径收敛：修复 query decomposition 把“原问题”和“仅去掉
> 句末问号的问题”当成两个变体的缺陷；收紧表格意图，使“表格理解/科学表格表示学习”不再
> 触发全表扫描；document routing 命中来源后，网页结构化表格扫描使用相同 source 过滤。
> benchmark 新增 `--structured-table-guard`，显式模拟网页确定性行列查找。固定五论文 53 题
> 的 routing+query-decomposition+table-guard @10 fact macro/micro/full 为
> `0.881/0.871/0.811`（43/53），目标论文/来源页/Table N 命中为
> `1.000/0.929/1.000`；相对 Phase 5.9 只有 `scidqa-09`、`table-llm-10` 从 zero 变为
> full，无逐题退化。另保留 `--adjacent-context` 离线负对照：虽修复 4 题，但使 3 道完整题
> 回退、页级命中降至 `0.881`，未接入网页。离线测试 107/107 通过；未调用外部 API、未启动
> Gradio、未修改 ChromaDB，未执行 git 操作。详见 `PHASE6_RETRIEVAL_CONVERGENCE_HANDOFF.md`。

> 2026-08-30 Phase 5.9 复合问题子查询对照（默认关闭）：新增
> `query_variants()`，按标点和保守的中英文并列词拆出最多三个子句，原问题始终保留；
> benchmark 与网页分别通过 `--query-decomposition`、`SCI_RAG_QUERY_DECOMPOSITION` 启用，
> 各变体在原 source route 范围内检索后 RRF 融合。新增 5 个离线/运行时回归测试，全部
> 103/103 通过。五论文 53 题 routing+CE+RRF 对照的 @10 fact macro/micro/full 为
> `0.843/0.830/0.774`，较 routing 控制 `0.805/0.796/0.717` 提升；目标论文命中仍
> `1.000`、Table N 命中仍 `0.944`，页级命中从 `0.929` 降至 `0.905`，仍有 12/53 题
> 不完整。完整报告仅写在仓库外 `/private/tmp/sci_rag_benchmark_query_decomposition_phase59.json`；
> 因页级回退和仍未证明答案语义正确率，该开关不改默认。未重建 ChromaDB、未调用
> DeepSeek/RAGAS、未执行 git add/commit/push。

> 同日网页回归：在现有 104 块 DrugR 数据库、相同 Hybrid+固定 cross-encoder 配置下，
> 关闭/开启子查询开关各通过 Gradio `lambda_3` 提交 11 道种子题，均 11/11 无调用错误；
> 两组答案事实审计均为 macro/micro=`1.0000/1.0000`、full=`11/11`，逐题状态均为
> `full→full`。答案 JSONL 和 A/B 对照仅写在仓库外 `/private/tmp`；这只是单论文事实词面
> 回归，不能证明多论文语义正确率或泛化能力。服务已停止，未修改 ChromaDB。

> 2026-08-30 Phase 5.8 运行时 document routing（默认关闭）：将已有保守
> `DocumentRouter` 接入网页运行时，新增 `SCI_RAG_DOCUMENT_ROUTING` 开关。仅当问题中的
> 高信号 ASCII 标识符唯一属于一个 `source` 时，才对 Chroma dense 查询和 Hybrid BM25
> 候选施加 source 过滤；歧义/跨论文问题回退全库。多来源集合仍不放宽同小节扩展边界，避免
> 混入同名 section。新增唯一来源、来源隔离、通用词过滤和歧义回退回归；离线测试 98/98
> 通过。默认 Dense、默认关闭 routing、原始数据库和依赖均未改变；本轮未启动网页、未调用
> 外部模型、未执行 git add/commit/push。随后在同一五论文离线 benchmark 上验证 `--document-routing`：
> 39/53 题触发且 39/39 正确、0 次误路由；@10 目标论文/页级命中=`1.000/0.929`，事实
> macro/micro=`0.805/0.796`，完整覆盖仍=`0.717`（15/53 题不完整），结果保存在仓库外
> `/private/tmp/sci_rag_benchmark_hybrid_reranker_routing_phase58_genericstop.json`。这只是来源隔离诊断收益，
> 不代表答案正确率或泛化提升。

> 2026-08-30 Phase 5.7 网页回归与修复后生成重跑：在 `127.0.0.1:7861` 使用仓库外
> 隔离 Chroma（479 块）打开 Gradio，页面显示块数并通过同一 ChatInterface 入口验证
> Table 2 两种设置、Darcy rough、GPT-4o RAG/FT、MgNO 基线/六层及 DrugR Table 2
> `DrugR*` 两题；表号、实体、列和值均与金标准一致，服务已停止。随后重跑五论文 53
> 题 Dense 与 Hybrid+Rerank：Dense fact macro/micro=`0.6903/0.6438`、full=`56.60%`、
> 平均延迟 `1.716s`；Hybrid+Rerank=`0.7201/0.6918`、full=`62.26%`、平均延迟
> `4.219s`，均 53/53 无调用错误。答案、trace、审计和 A/B 对照仅写在仓库外
> `/private/tmp`；相对 Phase 5.4，Dense 提升 8/退化 0，Hybrid 提升 8/退化 2。RAGAS
> 重跑因首个任务超过 180 秒并出现 IncompleteOutput/Timeout 主动中止，未生成新报告，
> 历史 RAGAS 数字不与本批次混用。未修改原始 ChromaDB、未执行 git add/commit/push。
> 单人逐题复核新批次为 Dense `28/8/17`、Hybrid+Rerank `30/10/13`（正确/部分/错误），
> 表号维度各 17/17，MgNO 单位题 2/3，`mgno-02` 公式 Dense 错误而 Hybrid 正确；复核
> JSONL 仅保存在 `/private/tmp`，不能替代独立双人标注。
> 回归期间发现离线缓存若用短 revision 下载、再用完整 SHA 启动，Hugging Face 可能无法解析
> 本地 ref；已将 README 下载示例改为与运行配置一致的完整 SHA，未修改模型或依赖。

> 2026-08-30 Phase 5.6 表格定位回归修复：针对五论文生成基准暴露的通用问题，更新
> `sci_rag_core.py` 的结构化表格查找逻辑，使实体可位于任意列（兼容 PDF 合并单元格导致
> 的空首列），保留 `<br>` 多值分隔并按问题中的“表格表示/题注”选择对应值；复合实体的
> `+`/`/` 空格统一归一化，避免把 `WikiTQ` 误当成 `WikiTQ+SQA+SciGen`。对重复实体增加
> 跨行分组标记（如 `Test(C&L)`/`Test(Other)`）过滤，并在显式分组缺失时阻止跨表误匹配；
> 对重复行的 baseline/层数变体按问题中的实体描述收集，避免把 `MgNO, 4 levels` 混入
> “基线 MgNO 和六层 MgNO”。多列指标按共享数据集限定词选择（如 Darcy rough 的 L2/H1），
> 并支持 `FT`/`full-text` 别名。真实隔离数据库定向验证了 Table 1/2/3 的表格问题，返回值
> 包括 `0.08/0.10/0.78`、`0.15(+0.07)/0.24(+0.14)/0.85(+0.07)`、`2.30/0.38`、
> Darcy rough `0.339/1.380`、GPT-4o RAG/FT `46.63/54.03` 和 MgNO `1.47/1.63`；另
> 验证 Test(Other) 重复行返回 `0.14/0.23/0.85`。新增 5 个表格定位回归，离线测试达到 94/94。
> 未调用 DeepSeek、未启动 Gradio、未写入原始 ChromaDB；尚未重跑五论文生成基准，故
> Phase 5.4 旧答案报告中的统计数字保持不变。

> 2026-08-30 Phase 5.4 五论文生成基准：经用户明确授权，使用同一临时 Chroma（479 个块）
> 和 53 道固定问题，通过网页 Gradio 查询路径分别完成 Dense、Hybrid+Rerank 各 53/53 题
> 的 DeepSeek 答案采集；原始数据库、旧 RAGAS 报告和 `.env` 均未改动。词面事实审计为
> Dense `0.5818/0.5411`、Hybrid+Rerank `0.6211/0.6027`（macro/micro），完整覆盖率
> `0.4528/0.4906`，平均延迟约 `1.60/4.32s`。逐题人工语义复核为 Dense `22/9/22`、
> Hybrid+Rerank `22/13/18`（正确/部分/错误），完全正确率均为 `41.51%`；Hybrid 主要
> 减少遗漏，未证明正确率提升。新增 `PHASE5_GENERATION_BENCHMARK_REPORT.md` 及两份
> 53 题复核标签 JSONL；本条记录形成时尚未运行 RAGAS，且未执行 git add、commit 或 push。

> 2026-08-30 Phase 5.5 RAGAS 辅助对照与报告契约：在完整生成/evaluation context trace
> 基础上完成 Dense、Hybrid+Rerank 各 53 题 RAGAS 0.4.3 对照，报告保存在仓库外 `/tmp`，
> 未覆盖历史 `evaluation/evaluation_report.*`。Dense/Hybrid 的 Context Relevance 为
> `0.5053`（47/53）/`0.7402`（51/53），Faithfulness 为 `0.7545`（50/53）/`0.7458`
> （49/53），Answer Relevancy 为 `0.5546`/`0.5447`（均 53/53）。发现三项 RAGAS 指标的
> declared input columns 均不含 `reference`，明确记录为“不使用 ground_truth”；新增
> `ragas_metric_input_columns` 元数据和预检分支。修复 `evaluate.py` 在多论文报告中保留
> `case_id`，新增回归测试；结构预检 0 error、2 warnings。
> 离线 unittest 总数更新为 87/87。

> 2026-08-30 Phase 5.0 生成评估输入契约：修复 `evaluation/answer_audit.py` 对多论文
> `cases.jsonl` 指针记录的解析，自动通过同目录 `manifest.json` 展开 11 个 DrugR 用例，
> 并增加重复 case ID 拒绝，避免将未解析的指针误计为无 required facts。直接读取 benchmark
> 现在得到完整 53 道题；离线测试达到 71/71。新增 `PHASE5_GENERATION_EVAL_HANDOFF.md`。
> 既有仓库外 Dense/Hybrid+Rerank 11 题答案在新契约下均复核为 11/11 full；这只是兼容性检查。
> 未调用 DeepSeek/RAGAS、未启动 Gradio、未写 ChromaDB。

> 2026-08-30 Phase 5.1 答案 A/B 对照工具：新增 `evaluation/compare_answer_runs.py`，复用
> 离线答案事实审计并强制两次运行使用相同 case ID，输出逐题覆盖差值、遗漏事实、状态转移
> 和聚合 macro/micro/full/partial/zero 差异；新增 2 项回归，离线测试达到 73/73。该工具
> 不调用模型；仍不把词面覆盖解释为语义正确性或 RAGAS 结果。

> 2026-08-30 Phase 5.2 人工复核契约：新增 `evaluation/review_answers.py`，可生成仓库外的
> 答案复核模板，并校验/汇总 correct/partial/incorrect/unanswerable 以及表号、单位、公式、
> 引用四个独立维度；新增 4 项回归，离线测试达到 77/77。人工标签与 lexical coverage 分开，
> 未调用 DeepSeek/RAGAS、未启动 Gradio、未写 ChromaDB。

> 2026-08-30 Phase 5.3 RAGAS 报告预检：新增 `evaluation/ragas_preflight.py`，离线校验报告与
> 测试集的 case/score 一致性、ground truth、生成与评估上下文 trace、reference contexts 及
> 模型元数据，并把无法由 artifact 证明的 ground-truth 使用和上下文一致性标为边界；新增
> 6 项回归，离线测试达到 84/84。与此同时 `evaluation/evaluate.py` 为未来报告保存完整生成
> trace、评估前缀、reference contexts 和 generation model；未覆盖旧报告。未运行 RAGAS、未调用
> DeepSeek、未启动 Gradio、未写 ChromaDB。

> 2026-08-30 Phase 4.2 provenance 诊断：在离线 benchmark 的每个 required fact 上记录命中块的
> `benchmark_document_id`、source、page、headers 和 chunk type，并标记跨论文、超出金标准页、
> 参考文献段、图注/图片块和缺失页码等风险。该诊断只增加报告字段，不改变检索排名或默认
> `equal RRF` / `candidate-k=50`；新增回归测试覆盖跨论文词面命中不能被误当作目标证据。
> 63 项离线测试更新为 64/64；随后在 alias 修正后的 Hybrid+CE+RRF 控制组完成五论文 smoke run，
> @10 记录 115 个事实词面命中，其中 6 个仅落在标注页之外。未写 ChromaDB、未调用 DeepSeek/RAGAS/Gradio。

> 2026-08-30 Phase 4.3 document routing 对照：新增保守的离线 `DocumentRouter` 和
> `--document-routing` 开关；只有唯一高信号 ASCII 术语才限制 source 候选池，歧义查询回退全库。
> 五篇论文/53 题中 39 题触发且全部路由正确。Hybrid+CE+等权 RRF @10 的 fact macro/micro
> 为 `0.805/0.796`，完整覆盖 `0.717`，页级/Table N 为 `0.929/0.944`；虽然聚合代理不降，
> 个别题目发生回退，因此不切换网页默认。离线测试达到 67/67；未写 ChromaDB、未调用
> DeepSeek/RAGAS/Gradio。

> 2026-08-29 Phase 4.0 检索失败审计：对五篇论文、53 道问题的 Hybrid+CE+RRF `@10` 结果中
> 16 道未完整案例逐题对照现行解析块、Hybrid top-50、cross-encoder top-50 和最终 RRF top-10。
> 分类出 gold/数学/连字符表面形式缺口、候选池未召回、cross-encoder 降权和最终 RRF 稀释四类
> 可重复现象，并通过 MgNO、SciDQA、AlphaFold 3 PDF 页面核对公式、初始化、立体化学和
> third-person 等原文。新增 `PHASE4_RETRIEVAL_FAILURE_AUDIT.md`；本阶段只写审计文档，未改
> 默认检索器、未重建 ChromaDB、未调用 DeepSeek/RAGAS/Gradio。

> 2026-08-29 Phase 4.0 P0 benchmark 表面形式修正：为 `third-person`、`Initialization/initialize`、
> `stride of 2`、`overlapping (clashing) atoms` 和 PDF 数学上标拆分后的 `d 2 n 2` 增加逐题 alias。
> 保持 Hybrid+CE+RRF 最终排名不变，离线 `@10` fact macro/micro 从 `0.785/0.776` 更新为
> `0.794/0.782`，完整覆盖从 `0.698` 更新为 `0.717`，未完整案例从 16 降为 15；该变化属于
> matcher 修正，不是检索收益。未写 ChromaDB、未调用 DeepSeek/RAGAS/Gradio。

> 2026-08-29 Phase 4.0 P1 融合对照：在同一五论文解析块、Hybrid candidate-50 和固定
> `BAAI/bge-reranker-base` revision 上，补跑 CE-only 与 CE 权重为 2/4/8 的 weighted RRF。
> CE-only 与当前等权 RRF 均为 `0.717` 完整覆盖但恢复不同案例；weighted RRF 权重 2 的
> fact macro/micro 为 `0.800/0.789`、完整覆盖仍为 `0.717`，页级命中降至 `0.881`；权重 4/8
> 完整覆盖降为 `0.698`。因此没有改变默认融合，下一步优先处理 candidate recall、邻接块和
> provenance，而不是继续盲目提高 CE 权重。离线测试达到 61/61；未写 ChromaDB、未调用
> DeepSeek/RAGAS/Gradio。

> 2026-08-29 Phase 4.0 P1 同小节扩展安全边界：离线对照发现多论文集合中将最匹配 section 的全部
> 块前置会把通用标题或错误来源推入 top-10，`@10` 完整覆盖降至 `0.698`。现在扩展仅在上下文前
> `context_k` 块单一来源、目标 section 已有锚点时启用，最多加入 6 个邻接块；多来源集合直接跳过。
> 单论文网页路径保留，未把该实验宣称为多论文检索收益。离线测试达到 63/63；未写 ChromaDB、
> 未调用 DeepSeek/RAGAS/Gradio。

> 2026-08-30 Phase 4.1 candidate-k 对照：在相同 alias、Hybrid、cross-encoder 和等权 RRF 条件下，
> 将候选池从 50 提高到 80。`@10` 完整覆盖仍为 `0.717`，fact macro/micro 为 `0.800/0.789`，
> 但全局 rerank mean/P95 增至 `4.453/5.231s`，峰值约 `2.685GB`。没有满足默认切换门槛，
> 保留 candidate-k=50；80 仅作为后续 opt-in 实验。未写 ChromaDB、未调用 DeepSeek/RAGAS/Gradio。

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
