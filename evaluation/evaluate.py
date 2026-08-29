#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sci-RAG evaluation utilities and optional RAGAS runner.

Importing this module is offline-safe.  RAGAS, the embedding model, ChromaDB,
and the DeepSeek client are loaded only by ``main``/``evaluate_with_ragas``.
The local fact/context checks use the test-set references directly and do not
call a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

from evaluation.answer_audit import audit_answer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTSET = os.path.join(SCRIPT_DIR, "test_questions.json")
DEFAULT_REPORT_JSON = os.path.join(SCRIPT_DIR, "evaluation_report.json")
DEFAULT_REPORT_MD = os.path.join(SCRIPT_DIR, "evaluation_report.md")

JUDGE_MODEL = "deepseek-chat"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MAX_CONTEXTS_DEFAULT = 10
METRIC_LABELS = {
    "nv_context_relevance": "Context Relevance（上下文相关性）",
    "faithfulness": "Answer Faithfulness（答案忠实度）",
    "answer_relevancy": "Answer Relevance（答案相关性）",
}
LOCAL_METRIC_LABELS = {
    "gold_context_recall": "Gold context recall（规范上下文召回）",
    "gold_fact_coverage": "Gold fact coverage（确定性事实覆盖）",
}


def load_testset(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data.get("test_cases", [])
    if not cases:
        raise ValueError(f"测试集为空或缺少 test_cases 字段：{path}")
    for case in cases:
        for field in ("question", "ground_truth"):
            if field not in case:
                raise ValueError(f"测试用例缺少 {field} 字段：{case.get('id', '?')}")
    return data.get("meta", {}), cases


def _normalise_for_evaluation(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fact_coverage(answer: str, case: dict[str, Any]) -> tuple[float | None, bool | None]:
    """Check manually declared atomic facts without an LLM.

    This is intentionally a transparent smoke check, not a replacement for
    semantic human review.  Cases without ``required_facts`` return ``None``.
    """

    audit = audit_answer(case, answer)
    score = audit["answer_fact_coverage"]
    if score is None:
        return None, None
    return float(score), audit["answer_fact_status"] == "full"


def gold_context_recall(contexts: list[str], case: dict[str, Any]) -> float | None:
    """Return the fraction of gold context snippets represented in retrieved text."""

    gold = case.get("contexts") or []
    if not gold:
        return None
    retrieved = [_normalise_for_evaluation(context) for context in contexts]
    hits = 0
    for reference in gold:
        target = _normalise_for_evaluation(reference)
        if any(target in candidate or candidate in target for candidate in retrieved):
            hits += 1
    return hits / len(gold)


def run_pipeline(cases: list[dict[str, Any]], max_contexts: int, runtime: Any) -> list[dict[str, Any]]:
    """Call the app once per question and retain the exact evaluation trace."""

    import app

    records = []
    for index, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"\n[{index}/{len(cases)}] 提问：{question}")
        started = time.time()
        result = app.query_knowledge(question, None, return_contexts=True, runtime=runtime)
        elapsed = time.time() - started
        if not isinstance(result, dict):
            raise RuntimeError("query_knowledge(return_contexts=True) 必须返回 dict")
        answer = str(result.get("answer", ""))
        all_contexts = list(result.get("contexts", []))
        all_ids = list(result.get("context_ids", []))
        all_metas = list(result.get("context_metadatas", []))
        contexts = all_contexts[:max_contexts]
        context_ids = all_ids[:max_contexts]
        context_metas = all_metas[:max_contexts]
        if answer.startswith("❌"):
            raise RuntimeError(f"第 {case.get('id', index)} 题模型调用失败：{answer[:200]}")
        print(f"  耗时 {elapsed:.1f}s，检索上下文 {len(all_contexts)} 个，参与打分 {len(contexts)} 个")
        print(f"  回答预览：{answer[:120]}{'...' if len(answer) > 120 else ''}")
        records.append(
            {
                "case": case,
                "answer": answer,
                "contexts": contexts,
                "context_ids": context_ids,
                "context_metadatas": context_metas,
                "num_retrieved": len(all_contexts),
            }
        )
    return records


def _install_vertexai_stub() -> None:
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class ChatVertexAIUnavailable:
        def __init__(self, *args: Any, **kwargs: Any):
            raise NotImplementedError("本项目评估仅使用 OpenAI 兼容接口（DeepSeek）")

    module.ChatVertexAI = ChatVertexAIUnavailable
    sys.modules[module_name] = module


def _import_ragas_components() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load the pinned RAGAS compatibility layer only when explicitly requested."""

    _install_vertexai_stub()
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.metrics")
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import AnswerRelevancy, ContextRelevance, Faithfulness
    from ragas.run_config import RunConfig

    return EvaluationDataset, evaluate, HuggingFaceEmbeddings, llm_factory, (ContextRelevance, Faithfulness, AnswerRelevancy), RunConfig


def build_judge_llm(client: Any, model: str = JUDGE_MODEL) -> Any:
    _, _, _, llm_factory, _, _ = _import_ragas_components()
    return llm_factory(model, client=client, temperature=0)


def _attach_legacy_llm_compat(judge_llm: Any, client: Any, model: str) -> Any:
    import asyncio
    import types

    async def agenerate_text(prompt: Any, n: int = 1, temperature: float = 0.1, **kwargs: Any) -> Any:
        text = prompt.text if hasattr(prompt, "text") else str(prompt)

        def call() -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                n=n,
                temperature=temperature,
            )
            return response.choices[0].message.content

        content = await asyncio.to_thread(call)
        return types.SimpleNamespace(generations=[[types.SimpleNamespace(text=content)]])

    judge_llm.agenerate_text = agenerate_text
    return judge_llm


def _attach_legacy_embeddings_compat(embeddings: Any) -> Any:
    embeddings.embed_query = lambda text: embeddings.embed_texts([text])[0]
    embeddings.embed_documents = lambda texts: embeddings.embed_texts(list(texts))
    return embeddings


def evaluate_with_ragas(records: list[dict[str, Any]], client: Any, model: str = JUDGE_MODEL) -> tuple[str, Any]:
    EvaluationDataset, evaluate, HuggingFaceEmbeddings, _, metric_classes, RunConfig = _import_ragas_components()
    import ragas
    ContextRelevance, Faithfulness, AnswerRelevancy = metric_classes
    judge_llm = _attach_legacy_llm_compat(build_judge_llm(client, model), client, model)
    embeddings = _attach_legacy_embeddings_compat(HuggingFaceEmbeddings(EMBEDDING_MODEL_NAME))
    samples = [
        {
            "user_input": record["case"]["question"],
            "response": record["answer"],
            "retrieved_contexts": record["contexts"],
            "reference": record["case"]["ground_truth"],
            "reference_contexts": record["case"].get("contexts", []),
        }
        for record in records
    ]
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[ContextRelevance(), Faithfulness(), AnswerRelevancy()],
        llm=judge_llm,
        embeddings=embeddings,
        show_progress=True,
        run_config=RunConfig(timeout=180),
    )
    return ragas.__version__, result.to_pandas()


def _clean_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return round(value, 4)


def build_report(
    meta: dict[str, Any],
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    df: Any | None,
    max_contexts: int,
    elapsed_total: float,
    ragas_version: str | None = None,
    judge_model: str = JUDGE_MODEL,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> dict[str, Any]:
    metric_cols = [column for column in METRIC_LABELS if df is not None and column in df.columns]
    per_metric: dict[str, Any] = {}
    for column in metric_cols:
        scores = [_clean_score(value) for value in df[column].tolist()]
        valid = [score for score in scores if score is not None]
        per_metric[column] = {
            "mean": round(sum(valid) / len(valid), 4) if valid else None,
            "count_valid": len(valid),
            "count_total": len(scores),
            "scores": scores,
        }

    local_scores = {"gold_context_recall": [], "gold_fact_coverage": []}
    results = []
    for index, (case, record) in enumerate(zip(cases, records), start=1):
        fact_score, fact_exact = fact_coverage(record["answer"], case)
        context_score = gold_context_recall(record["contexts"], case)
        if context_score is not None:
            local_scores["gold_context_recall"].append(round(context_score, 4))
        if fact_score is not None:
            local_scores["gold_fact_coverage"].append(round(fact_score, 4))
        ragas_scores = {
            column: _clean_score(df[column].tolist()[index - 1])
            for column in metric_cols
        }
        results.append(
            {
                "id": case.get("id", index),
                "type": case.get("type", "unknown"),
                "question": case["question"],
                "ground_truth": case["ground_truth"],
                "answer": record["answer"],
                "num_retrieved_contexts": record["num_retrieved"],
                "num_contexts_evaluated": len(record["contexts"]),
                "evaluated_contexts": record["contexts"],
                "context_ids": record.get("context_ids", []),
                "context_metadatas": record.get("context_metadatas", []),
                "gold_context_recall": context_score,
                "gold_fact_coverage": fact_score,
                "gold_fact_exact": fact_exact,
                "scores": ragas_scores,
            }
        )

    for key, values in local_scores.items():
        per_metric[key] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "count_valid": len(values),
            "count_total": len(cases),
            "scores": values,
        }

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paper": meta.get("paper", ""),
            "paper_title": meta.get("title", ""),
            "ragas_version": ragas_version,
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "num_test_cases": len(cases),
            "max_contexts_per_question": max_contexts,
            "elapsed_seconds": round(elapsed_total, 1),
            "metric_labels": {**METRIC_LABELS, **LOCAL_METRIC_LABELS},
        },
        "summary": per_metric,
        "results": results,
    }


def write_reports(report: dict[str, Any], json_path: str, md_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    metadata = report["meta"]
    lines = [
        "# Sci-RAG 评估报告（RAGAS）",
        "",
        f"- 论文：{metadata['paper_title']}（`{metadata['paper']}`）",
        f"- 生成时间：{metadata['generated_at']}",
        f"- RAGAS 版本：{metadata['ragas_version']}；评判模型：{metadata['judge_model']}；Embedding：{metadata['embedding_model']}",
        f"- 测试题数：{metadata['num_test_cases']}；每题参与打分的检索上下文数：{metadata['max_contexts_per_question']}（重排序后 top-N）",
        f"- 总耗时：{metadata['elapsed_seconds']} 秒",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 平均分 | 有效样本 |",
        "| --- | --- | --- |",
    ]
    for column, label in metadata["metric_labels"].items():
        summary = report["summary"].get(column, {})
        mean = summary.get("mean")
        mean_text = f"{mean:.4f}" if mean is not None else "N/A"
        lines.append(f"| {label} | **{mean_text}** | {summary.get('count_valid', 0)}/{summary.get('count_total', 0)} |")
    lines += ["", "## 逐题明细", ""]
    for result in report["results"]:
        lines += [
            f"### {result['id']}. [{result['type']}] {result['question']}",
            "",
            f"- 检索上下文：{result['num_retrieved_contexts']} 个（打分取前 {result['num_contexts_evaluated']} 个）",
            f"- 标准答案：{result['ground_truth']}",
            f"- 模型回答：{result['answer']}",
            f"- Gold context recall：{result['gold_context_recall'] if result['gold_context_recall'] is not None else 'N/A'}",
            f"- Gold fact coverage：{result['gold_fact_coverage'] if result['gold_fact_coverage'] is not None else 'N/A'}",
        ]
        score_parts = []
        for column, label in METRIC_LABELS.items():
            value = result["scores"].get(column)
            score_parts.append(f"{label}：{value:.4f}" if value is not None else f"{label}：N/A")
        lines += [f"- RAGAS 得分：{'；'.join(score_parts)}", ""]
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sci-RAG RAGAS 评估")
    parser.add_argument("--testset", default=DEFAULT_TESTSET)
    parser.add_argument("--max-contexts", type=int, default=MAX_CONTEXTS_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    if args.max_contexts <= 0:
        raise ValueError("--max-contexts 必须为正整数")

    started = time.time()
    meta, cases = load_testset(args.testset)
    if args.limit is not None:
        cases = cases[: args.limit]
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import app

    runtime = app.create_runtime()
    records = run_pipeline(cases, args.max_contexts, runtime)
    ragas_version, dataframe = evaluate_with_ragas(records, runtime.client, runtime.config.deepseek_model)
    report = build_report(
        meta,
        cases,
        records,
        dataframe,
        args.max_contexts,
        time.time() - started,
        ragas_version,
        runtime.config.deepseek_model,
        runtime.config.embedding_model,
    )
    write_reports(report, args.report_json, args.report_md)
    print("\n评估完成，报告已保存：", args.report_json, args.report_md)


if __name__ == "__main__":
    main()
