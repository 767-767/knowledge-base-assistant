#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sci-RAG 项目 RAGAS 评估脚本

流程：
  1. 从 evaluation/test_questions.json 加载测试集（question / ground_truth / contexts）
  2. 对每个问题调用 app.query_knowledge(..., return_contexts=True) 获取回答与检索上下文
  3. 用 RAGAS 计算三个核心指标：
       - Context Relevance    （context_relevance，上下文相关性）
       - Answer Faithfulness  （faithfulness，答案忠实度）
       - Answer Relevance     （answer_relevancy，答案相关性）
  4. 生成 evaluation/evaluation_report.json 与 evaluation/evaluation_report.md

用法（务必在项目根目录执行，chroma_db 路径与 .env 均相对项目根）：
    ./venv/bin/python evaluation/evaluate.py                 # 全量评估（11 题，约 15-30 分钟）
    ./venv/bin/python evaluation/evaluate.py --limit 3       # 冒烟测试：只跑前 3 题
    ./venv/bin/python evaluation/evaluate.py --max-contexts 5   # 每题最多取前 5 个重排序上下文打分

说明：
  - 评判 LLM 复用 app.py 的 DeepSeek 客户端（deepseek-chat），不额外申请新密钥；
  - AnswerRelevancy 所需的 embedding 复用本地缓存的 BAAI/bge-small-zh-v1.5，
    与检索侧保持一致，不调用外部 API；
  - 每题检索最多返回 60 个上下文，RAGAS 打分默认只取重排序后的前 MAX_CONTEXTS 个
    （控制调用量与时长的常见做法），报告中会如实标注。
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# app.py 以 "./chroma_db" 相对路径打开知识库、load_dotenv 读取 .env，故必须先切到项目根
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


# ---------- 兼容性桩（必须在 import ragas 之前执行） ----------
# ragas 0.4.3 顶层导入 langchain_community.chat_models.vertexai，但 langchain-community
# 0.4.x 已移除该模块（相关集成迁至 langchain-classic / langchain-google-vertexai）。
# 本评估只使用 OpenAI 兼容接口（DeepSeek），注册占位模块避免 ImportError，
# 桩类不会被实例化。
def _install_vertexai_stub():
    import types as _types

    _mod_name = "langchain_community.chat_models.vertexai"
    if _mod_name in sys.modules:
        return
    _mod = _types.ModuleType(_mod_name)

    class _ChatVertexAIUnavailable:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "VertexAI 桩模块：Sci-RAG 评估仅使用 OpenAI 兼容接口（DeepSeek），"
                "ChatVertexAI 不可实例化。"
            )

    _mod.ChatVertexAI = _ChatVertexAIUnavailable
    sys.modules[_mod_name] = _mod


_install_vertexai_stub()

import warnings  # noqa: E402

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

# 注意：ragas 0.4.3 中 ragas.metrics.collections 的新指标（BaseMetric 体系）尚不能被
# evaluate() 接受（内部 isinstance(m, Metric) 校验不一致），故使用经典 Metric 类，
# 通过 evaluate(llm=..., embeddings=...) 注入 DeepSeek 评判模型与本地 embedding。
# 上游升级修复后可切换回 ragas.metrics.collections。
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.metrics")
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    ContextRelevance,
    Faithfulness,
)

import app  # noqa: E402  加载知识库、embedding 模型与 DeepSeek 客户端

# ---------- 配置 ----------
JUDGE_MODEL = "deepseek-chat"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 与 app.py 检索侧一致（本地缓存）
MAX_CONTEXTS_DEFAULT = 10  # RAGAS 打分用的重排序后 top-N 上下文数量
METRIC_LABELS = {
    # 注意：ragas 0.4.3 经典 ContextRelevance 指标（_nv_metrics）的结果列名为
    # nv_context_relevance，而非 context_relevance
    "nv_context_relevance": "Context Relevance（上下文相关性）",
    "faithfulness": "Answer Faithfulness（答案忠实度）",
    "answer_relevancy": "Answer Relevance（答案相关性）",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTSET = os.path.join(SCRIPT_DIR, "test_questions.json")
DEFAULT_REPORT_JSON = os.path.join(SCRIPT_DIR, "evaluation_report.json")
DEFAULT_REPORT_MD = os.path.join(SCRIPT_DIR, "evaluation_report.md")


# ---------- 步骤 1：加载测试集 ----------
def load_testset(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])
    if not cases:
        raise ValueError(f"测试集为空或缺少 test_cases 字段：{path}")
    for c in cases:
        for field in ("question", "ground_truth"):
            if field not in c:
                raise ValueError(f"测试用例缺少 {field} 字段：{c.get('id', '?')}")
    return data.get("meta", {}), cases


# ---------- 步骤 2：逐题调用 query_knowledge ----------
def run_pipeline(cases, max_contexts):
    """对每个测试问题调用 app.query_knowledge，收集 (question, answer, contexts)。"""
    records = []
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"\n[{i}/{len(cases)}] 提问：{question}")
        t0 = time.time()
        result = app.query_knowledge(question, None, return_contexts=True)
        elapsed = time.time() - t0
        if not isinstance(result, dict):
            raise RuntimeError(
                f"query_knowledge 未返回 dict（return_contexts=True 时预期为 dict）：{result!r}"
            )
        answer = result["answer"]
        contexts = list(result.get("contexts", []))[:max_contexts]
        if answer.startswith("❌"):
            print(f"  ⚠️ 调用出错：{answer[:100]}")
        print(f"  耗时 {elapsed:.1f}s，检索上下文 {len(result.get('contexts', []))} 个，"
              f"参与打分 {len(contexts)} 个")
        print(f"  回答预览：{answer[:120]}{'...' if len(answer) > 120 else ''}")
        records.append({
            "case": case,
            "answer": answer,
            "contexts": contexts,
            "num_retrieved": len(result.get("contexts", [])),
        })
    return records


# ---------- 步骤 3：RAGAS 打分 ----------
def build_judge_llm():
    """用 app 的 DeepSeek 客户端构造 RAGAS 评判 LLM（structured output 走 instructor 适配器）。"""
    return llm_factory(JUDGE_MODEL, client=app.client, temperature=0)


def _attach_legacy_llm_compat(judge_llm):
    """兼容补丁：ragas 0.4.3 经典指标 ContextRelevance（_nv_metrics）仍调用旧式
    llm.agenerate_text(StringPromptValue, n=1, temperature=...)，而 llm_factory 返回的
    新版 InstructorLLM 没有该方法。这里在实例上补齐旧接口（返回 .generations[0][0].text），
    底层走 DeepSeek 客户端；其余指标使用的新式 structured generate 不受影响。"""
    import asyncio
    import types as _types

    async def agenerate_text(prompt, n=1, temperature=0.1, **kwargs):
        text = prompt.text if hasattr(prompt, "text") else str(prompt)

        def _call():
            resp = app.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": text}],
                n=n,
                temperature=temperature,
            )
            return resp.choices[0].message.content

        # app.client 是同步客户端，用线程池避免阻塞 ragas 的事件循环
        content = await asyncio.to_thread(_call)
        generation = _types.SimpleNamespace(text=content)
        return _types.SimpleNamespace(generations=[[generation]])

    judge_llm.agenerate_text = agenerate_text
    return judge_llm


def _attach_legacy_embeddings_compat(embeddings):
    """兼容补丁：经典指标 AnswerRelevancy 调用 Langchain 风格的
    embeddings.embed_query / embed_documents，而新版 HuggingFaceEmbeddings
    只提供 embed_texts。这里在实例上补齐两个方法（同实例、同模型）。"""
    embeddings.embed_query = lambda text: embeddings.embed_texts([text])[0]
    embeddings.embed_documents = lambda texts: embeddings.embed_texts(list(texts))
    return embeddings


def evaluate_with_ragas(records):
    import ragas

    judge_llm = _attach_legacy_llm_compat(build_judge_llm())
    embeddings = _attach_legacy_embeddings_compat(
        HuggingFaceEmbeddings(EMBEDDING_MODEL_NAME)
    )
    # 经典 Metric 类实例化时不带参数，llm / embeddings 统一由 evaluate() 注入
    metrics = [ContextRelevance(), Faithfulness(), AnswerRelevancy()]

    samples = []
    for r in records:
        samples.append({
            "user_input": r["case"]["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["case"]["ground_truth"],
            "reference_contexts": r["case"].get("contexts", []),
        })
    dataset = EvaluationDataset.from_list(samples)

    print("\n开始 RAGAS 打分（3 个指标 × %d 题，评判模型 deepseek-chat）..." % len(samples))
    result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=embeddings,
                      show_progress=True, run_config=RunConfig(timeout=180))
    df = result.to_pandas()
    return ragas.__version__, df


# ---------- 步骤 4：报告生成 ----------
def _clean_score(v):
    """NaN → None，其余保留 4 位小数，保证 JSON 合法。"""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return round(float(v), 4)


def build_report(meta, cases, records, df, max_contexts, elapsed_total):
    metric_cols = [c for c in ("nv_context_relevance", "faithfulness", "answer_relevancy")
                   if c in df.columns]

    per_metric = {}
    for col in metric_cols:
        scores = [_clean_score(v) for v in df[col].tolist()]
        valid = [s for s in scores if s is not None]
        per_metric[col] = {
            "mean": round(sum(valid) / len(valid), 4) if valid else None,
            "count_valid": len(valid),
            "count_total": len(scores),
            "scores": scores,
        }

    results = []
    for idx, (case, record) in enumerate(zip(cases, records)):
        results.append({
            "id": case.get("id", idx + 1),
            "type": case.get("type", "unknown"),
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": record["answer"],
            "num_retrieved_contexts": record["num_retrieved"],
            "num_contexts_evaluated": len(record["contexts"]),
            "scores": {col: _clean_score(df[col].tolist()[idx]) for col in metric_cols},
        })

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paper": meta.get("paper", ""),
            "paper_title": meta.get("title", ""),
            "ragas_version": None,  # 由调用方填充
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "num_test_cases": len(cases),
            "max_contexts_per_question": max_contexts,
            "elapsed_seconds": round(elapsed_total, 1),
            "metric_labels": METRIC_LABELS,
        },
        "summary": per_metric,
        "results": results,
    }


def write_reports(report, json_path, md_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["meta"]
    lines = [
        "# Sci-RAG 评估报告（RAGAS）",
        "",
        f"- 论文：{m['paper_title']}（`{m['paper']}`）",
        f"- 生成时间：{m['generated_at']}",
        f"- RAGAS 版本：{m['ragas_version']}；评判模型：{m['judge_model']}；"
        f"Embedding：{m['embedding_model']}",
        f"- 测试题数：{m['num_test_cases']}；每题参与打分的检索上下文数："
        f"{m['max_contexts_per_question']}（重排序后 top-N）",
        f"- 总耗时：{m['elapsed_seconds']} 秒",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 平均分 | 有效样本 |",
        "| --- | --- | --- |",
    ]
    for col, label in METRIC_LABELS.items():
        s = report["summary"].get(col, {})
        mean = s.get("mean")
        mean_str = f"{mean:.4f}" if mean is not None else "N/A"
        lines.append(f"| {label} | **{mean_str}** | {s.get('count_valid', 0)}/{s.get('count_total', 0)} |")
    lines += ["", "## 逐题明细", ""]

    for r in report["results"]:
        lines.append(f"### {r['id']}. [{r['type']}] {r['question']}")
        lines.append("")
        lines.append(f"- 检索上下文：{r['num_retrieved_contexts']} 个（打分取前 {r['num_contexts_evaluated']} 个）")
        lines.append(f"- 标准答案：{r['ground_truth']}")
        lines.append(f"- 模型回答：{r['answer']}")
        score_parts = []
        for col, label in METRIC_LABELS.items():
            v = r["scores"].get(col)
            score_parts.append(f"{label}：{f'{v:.4f}' if v is not None else 'N/A'}")
        lines.append(f"- 得分：{'；'.join(score_parts)}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="Sci-RAG RAGAS 评估")
    parser.add_argument("--testset", default=DEFAULT_TESTSET, help="测试集 JSON 路径")
    parser.add_argument("--max-contexts", type=int, default=MAX_CONTEXTS_DEFAULT,
                        help="每题参与 RAGAS 打分的重排序后上下文数量（默认 10）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 题（冒烟测试用）")
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON, help="JSON 报告输出路径")
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD, help="Markdown 报告输出路径")
    args = parser.parse_args()

    t_start = time.time()
    meta, cases = load_testset(args.testset)
    if args.limit:
        cases = cases[:args.limit]
    print(f"加载测试集 {len(cases)} 题（{meta.get('paper', '未知论文')}）")

    records = run_pipeline(cases, args.max_contexts)
    ragas_version, df = evaluate_with_ragas(records)

    report = build_report(meta, cases, records, df, args.max_contexts,
                          elapsed_total=time.time() - t_start)
    report["meta"]["ragas_version"] = ragas_version
    write_reports(report, args.report_json, args.report_md)

    print("\n" + "=" * 60)
    print("评估完成，平均分：")
    for col, label in METRIC_LABELS.items():
        s = report["summary"].get(col, {})
        mean = s.get("mean")
        print(f"  {label}: {f'{mean:.4f}' if mean is not None else 'N/A'}")
    print(f"\n报告已保存：\n  {args.report_json}\n  {args.report_md}")


if __name__ == "__main__":
    main()
