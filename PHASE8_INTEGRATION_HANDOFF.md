# Phase 8 整合回归与交接

日期：2026-08-31  
分支：`develop`  
状态：本轮整合回归完成；未提交或 push

## 本轮纳入的变更

1. `evaluation/generation_stability.py` 支持显式传入并记录 reranker model/revision、批大小、
   最大长度、设备、reranker RRF 参数和公式开关；避免评估命令“看似启用 reranker、实际使用
   null 配置”的审计风险。
2. `app.py` 的确定性表格回答保留 caption 中声明的共享单位/尺度，并在返回的 context trace
   中保留原始 table caption；无单位提示的旧答案格式不变。
3. `evaluation/benchmark/cases.jsonl` 增加经人工确认的安全表面别名：`3 × 3`、中文“步长为 2”、
   “扩散模块”和“闭合构象”。别名只修正词面格式误报，不改 gold 事实。
4. 新增 Phase 7.1–7.3 审计和门控文档；完整生成 JSON/审计报告继续存放在仓库外
   `/private/tmp`，不把答案或密钥写入 Git。

## 验收证据

- 单元测试：`145/145` 通过。
- 编译：`app.py`、`sci_rag_core.py`、`sci_rag_retrieval.py` 和生成 runner 均通过
  `py_compile`。
- 格式：`git diff --check` 通过。
- 隔离表格 smoke：MgNO Table 1 的 `0.339/1.380` 答案同时带出 `×10⁻²` 表注；没有调用
  DeepSeek，不改数据库。
- 当前版本生成：Hybrid + 固定 reranker 在 502 块隔离库完成 53/53 题，0 错误；词面事实
  macro/micro=`0.9261/0.8973`，完整覆盖 `46/53`。这是词面审计，不是语义正确率。
- 8 题 `context_k=50` 限定对照：6/8 词面完整；未因此改变网页默认 `context_k=10`。
- 运行 trace：53/53 行 provenance 完整；只跑一轮，不能估计答案文字稳定性。

## 当前默认行为

- 网页默认仍按 `.env`/`RuntimeConfig` 配置运行；没有强制切换到 context-k=50、公式证据、
  OCR/VLM、工具调用或 Graph-RAG。
- 生产 ChromaDB 未重建、未删除、未写入；所有生成实验使用隔离库。
- 没有运行完整 RAGAS，也没有把 53 题词面覆盖率写成语义正确率或泛化结论。

## 交接前检查命令

```bash
cd /Users/qinleqi/Desktop/knowledge-base-assistant
./venv/bin/python -m unittest discover -s tests -q
./venv/bin/python -m py_compile app.py sci_rag_core.py sci_rag_retrieval.py evaluation/generation_stability.py
git diff --check
git status --short --branch
```

`git status` 中的修改应仅包含本项目本轮已知源码、测试和文档；`/private/tmp` 下的生成结果
不应被 add。提交前应由用户 review `app.py`、`sci_rag_core.py`、`evaluation/generation_stability.py`、
`evaluation/benchmark/cases.jsonl` 及 Phase 7/8 文档，确认后再自行 `git add`、`commit` 和
`push`。本助手不会执行 push。

## 后续停止条件

如果没有新的用户目标，下一步不继续扩大架构。只有以下情况之一成立时才开新实验：

- 新增 ≥20 道真实单位/算术/一致性题，且 ≥5 道由白名单工具可修复；
- 新增 ≥10 道 image-only/复杂图题，且文字层基线稳定失败；
- 新增明确跨文档多跳题，并有 ≥5 道纯文本基线稳定失败且需要图关系。

否则保持当前文本/表格/空间文字 RAG 原型，先积累人工语义复核和可复现报告。
