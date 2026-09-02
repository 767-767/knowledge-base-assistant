#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit saved answers against benchmark gold data without calling a model.

The report keeps three evidence levels separate:

* required-fact coverage is a deterministic lexical proxy derived from gold
  annotations;
* gold-context recall checks whether curated evidence snippets were retrieved;
* human judgments, when supplied, are the only semantic correctness labels.

No embedding model, ChromaDB, Gradio, RAGAS, or external API is loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.answer_audit import (  # noqa: E402
    _load_answers,
    _load_cases,
    audit_answer,
    aggregate_answer_audit,
)
from evaluation.evaluate import gold_context_recall  # noqa: E402
from evaluation.review_answers import load_reviews  # noqa: E402


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _trace_contexts(record: dict[str, Any]) -> list[str]:
    """Read generation contexts first, then common saved-answer fields."""

    for field in ("generation_contexts", "contexts", "evaluated_contexts"):
        value = record.get(field)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def audit_ground_truth(
    cases: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
    *,
    reviews: Iterable[dict[str, Any]] | None = None,
    require_all: bool = False,
) -> dict[str, Any]:
    """Join saved answers with gold facts, contexts, and optional reviews."""

    case_list = list(cases)
    case_map = {
        str(case.get("case_id", case.get("id"))): case
        for case in case_list
    }
    answer_list = list(answers)
    answer_map: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    for record in answer_list:
        case_id = str(record.get("case_id", record.get("id", "")))
        if case_id not in case_map:
            unknown.append(case_id)
            continue
        if case_id in answer_map:
            raise ValueError(f"回答文件存在重复 case id：{case_id}")
        answer_map[case_id] = record
    if unknown:
        raise ValueError("回答文件包含测试集之外的 case id：" + ", ".join(sorted(unknown)))
    missing = sorted(set(case_map) - set(answer_map))
    if require_all and missing:
        raise ValueError("回答文件缺少 case id：" + ", ".join(missing))

    review_map: dict[str, dict[str, Any]] = {}
    if reviews is not None:
        for review in reviews:
            case_id = str(review["case_id"])
            if case_id in review_map:
                raise ValueError(f"复核文件存在重复 case id：{case_id}")
            if case_id not in case_map:
                raise ValueError(f"复核文件包含测试集之外的 case id：{case_id}")
            if case_id not in answer_map:
                raise ValueError(f"复核文件包含没有对应回答的 case id：{case_id}")
            review_map[case_id] = review
    missing_reviews = sorted(set(answer_map) - set(review_map)) if reviews is not None else []
    if require_all and reviews is not None and missing_reviews:
        raise ValueError("人工复核缺少 case id：" + ", ".join(missing_reviews))

    rows: list[dict[str, Any]] = []
    context_scores: list[float] = []
    exact_matches = 0
    for case in case_list:
        case_id = str(case.get("case_id", case.get("id")))
        record = answer_map.get(case_id)
        if record is None:
            continue
        answer = str(record.get("answer", ""))
        lexical = audit_answer(case, answer)
        contexts = _trace_contexts(record)
        context_recall = gold_context_recall(contexts, case)
        if context_recall is not None:
            context_scores.append(float(context_recall))
        exact_match = _normalise(answer) == _normalise(case.get("ground_truth"))
        exact_matches += int(exact_match)
        review = review_map.get(case_id)
        rows.append(
            {
                "case_id": case_id,
                "document_id": case.get("document_id"),
                "type": case.get("type", "unknown"),
                "question": case.get("question", ""),
                "ground_truth": case.get("ground_truth", ""),
                "answer": answer,
                "answer_fact_status": lexical["answer_fact_status"],
                "answer_fact_coverage": lexical["answer_fact_coverage"],
                "answer_required_fact_count": lexical["required_fact_count"],
                "matched_facts": lexical["matched_facts"],
                "missing_facts": lexical["missing_facts"],
                "answer_refusal_detected": lexical["answer_refusal_detected"],
                "refused_required_facts": lexical["refused_required_facts"],
                "answer_risk_flags": lexical["answer_risk_flags"],
                "gold_context_recall": context_recall,
                "context_count": len(contexts),
                "ground_truth_exact_match": exact_match,
                "human_judgment": review.get("judgment") if review else None,
                "human_aspects": {
                    aspect: review.get(aspect, "")
                    for aspect in ("table_number", "units", "formula", "citation")
                }
                if review
                else None,
                "human_notes": review.get("notes", "") if review else None,
            }
        )

    lexical_summary = aggregate_answer_audit(
        [
            {
                **row,
                "required_fact_coverage": row["answer_fact_coverage"],
                "fact_coverage_status": row["answer_fact_status"],
                "required_fact_count": row["answer_required_fact_count"],
                "matched_required_fact_count": len(row["matched_facts"]),
            }
            for row in rows
        ]
    )
    human_summary = None
    if reviews is not None:
        human_summary = {
            "reviewed_case_count": len(review_map),
            "missing_case_ids": missing_reviews,
            "judgment_counts": dict(
                sorted(Counter(str(row.get("judgment", "")) for row in review_map.values()).items())
            ),
        }

    return {
        "schema": "sci-rag-ground-truth-audit-v1",
        "case_count": len(case_list),
        "answer_count": len(rows),
        "missing_case_ids": missing,
        "summary": {
            "answer_fact_coverage": lexical_summary,
            "gold_context_recall": {
                "mean": _mean(context_scores),
                "count_valid": len(context_scores),
                "count_total": len(rows),
            },
            "ground_truth_exact_match": {
                "count": exact_matches,
                "count_total": len(rows),
                "rate": round(exact_matches / len(rows), 4) if rows else None,
            },
            "human_review": human_summary,
        },
        "claim_boundaries": {
            "answer_fact_coverage": "金标准派生的词面代理，只能筛查事实是否显式出现，不能证明语义正确。",
            "gold_context_recall": "人工整理片段的重叠召回，只能说明证据是否被取回，不能证明答案相关性或忠实度。",
            "ground_truth_exact_match": "表面文本一致性诊断，改写或等价表达会被判不一致。",
            "human_review": "只有填写 human_judgment 后，才有逐题语义正确性证据。",
        },
        "results": rows,
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    facts = summary["answer_fact_coverage"]
    context = summary["gold_context_recall"]
    exact = summary["ground_truth_exact_match"]
    print(f"金标准审计：{report['answer_count']}/{report['case_count']} 题")
    print(
        "事实覆盖 macro/micro="
        f"{facts['required_fact_coverage_macro']!s}/{facts['required_fact_coverage_micro']!s}；"
        f"full={facts['full_fact_coverage_rate']!s}；"
        f"gold-context recall={context['mean']!s} ({context['count_valid']}/{context['count_total']})"
    )
    print(f"规范化答案完全一致：{exact['count']}/{exact['count_total']}（仅表面诊断）")
    if summary["human_review"] is not None:
        print(f"人工复核判断：{summary['human_review']['judgment_counts']}")
    if report["missing_case_ids"]:
        print("尚未回答：" + ", ".join(report["missing_case_ids"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="离线审计答案与 benchmark 金标准的关系")
    parser.add_argument("--testset", required=True, help="JSON/JSONL benchmark 测试集")
    parser.add_argument("--answers", required=True, help="生成 trace 或答案 JSON/JSONL")
    parser.add_argument("--reviews", help="可选：已填写的人工复核 JSON/JSONL")
    parser.add_argument("--require-all", action="store_true", help="要求回答（和复核）覆盖全部用例")
    parser.add_argument("--json-out", help="写入完整 JSON 报告")
    args = parser.parse_args()
    cases = _load_cases(args.testset)
    answers = _load_answers(args.answers)
    reviews = load_reviews(args.reviews) if args.reviews else None
    report = audit_ground_truth(cases, answers, reviews=reviews, require_all=args.require_all)
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
