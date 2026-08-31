# Phase 7.3 多模态与 Graph-RAG 触发门控

日期：2026-08-31  
分支：`develop`  
状态：完成静态/离线门控；两条方向均暂缓；未提交或 push

## 当前证据

固定 benchmark 共 53 题，其中 5 题标注为 figure、2 题为 equation，另有若干表格/方法题
引用图或公式。当前生成链路对这些题使用 PDF 文字层和可选的空间坐标证据；正式 10-context
运行中，AlphaFold 3 的 Figure 1 样本数（25/38/8）、Figure 2 的 pairformer 规模以及
表格/公式题均能取得可用文本证据。已有 `PHASE7_DIRECTION_AUDIT.md` 对代码和依赖的静态
审计确认：

- `pymupdf4llm` 使用 `write_images=False`、`embed_images=False`；没有图像像素持久化、OCR
  或 VLM 调用。
- 空间图形通道只消费 `Page.get_text("blocks")` 的 born-digital 文本和坐标；它不是通用图像
  理解，也不能覆盖扫描 PDF、图片内文字、曲线/箭头/化学结构和热图。
- 当前没有图谱实体/关系抽取、图数据库或图邻居检索；53 题也没有明确要求跨论文多跳推理。

## 多模态门控结论

当前没有 image-only 失败集，因此不新增 OCR/VLM 依赖、不保存图片资产、不改变默认解析路径。
进入多模态实验前必须：

1. 另建至少 10 道人工核对的 image-only 或复杂子图题，明确图号、面板、数字/实体和 gold；
2. 文本层和空间坐标基线在其中稳定缺失，且 OCR/VLM 能在隔离 A/B 中修复；
3. 图像证据与 text/table evidence 分开存储、排名和 trace，记录图像哈希、模型版本、延迟和
   费用；
4. 图中数字/实体规范化准确率至少 0.90，非图问题事实覆盖不低于当前基线，出现串图或错读
   即停止扩大范围。

## Graph-RAG 门控结论

当前没有引入 Graph-RAG 的必要性。论文数只有 5 篇，现有题目主要是同文档表格、方法、公式
和图形证据；近期缺口是候选前缀和答案表达，不是跨文档实体关系。进入 Graph-RAG 前必须：

1. 新增明确的跨文档多跳问题集，并人工标注节点、边、实体消歧和 provenance；
2. 证明纯文本 Hybrid+reranker 在至少 5 道题上稳定失败，而失败确实需要图邻居/关系扩展；
3. 先在隔离内存图中与原文检索并行 A/B，不替换原文证据；比较 hop recall、answer fact
   coverage、source correctness、图构建成本和查询延迟；
4. 任何自动抽取的关系都不能直接成为无来源证据，图边必须保留原文 chunk/page provenance。

## 阶段停止决定

在上述触发条件满足前，Phase 7.3 停止在审计层面：不新增 OCR/VLM 或图数据库依赖，不改变
默认开关，不启动付费多模态或图谱实验。下一阶段进入 Phase 8 的整合回归；若用户未来提供
image-only 或跨文档多跳目标，再按对应门控单独开实验。
