#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline contract and summary for human review of saved answers.

The lexical audit can only tell whether declared fact surfaces occur in an
answer. This module keeps that signal separate from a human semantic review of
correctness, table number, units, formulas, and citations. It never calls a
model or imports the application runtime.
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
    audit_answers,
)


JUDGMENTS = {"correct", "partial", "incorrect", "unanswerable"}
ASPECT_VALUES = {"correct", "incorrect", "not_applicable", "uncertain"}
ASPECTS = ("table_number", "units", "formula", "citation")


def _case_id(record: dict[str, Any]) -> str:
    value = record.get("case_id", record.get("id"))
    if value is None:
        raise ValueError("复核记录缺少 case_id/id")
    return str(value)


def _validate_review(record: dict[str, Any], *, allow_blank_judgment: bool) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("人工复核记录必须是对象")
    case_id = _case_id(record)
    judgment = record.get("judgment", "")
    if not isinstance(judgment, str):
        raise ValueError(f"{case_id} 的 judgment 必须是字符串")
    judgment = judgment.strip().casefold()
    if judgment not in JUDGMENTS and not (allow_blank_judgment and not judgment):
        allowed = ", ".join(sorted(JUDGMENTS))
        raise ValueError(f"{case_id} 的 judgment 无效：{judgment!r}（允许：{allowed}）")
    normalized: dict[str, Any] = {**record, "case_id": case_id, "judgment": judgment}
    for aspect in ASPECTS:
        value = record.get(aspect, "")
        if not isinstance(value, str):
            raise ValueError(f"{case_id} 的 {aspect} 必须是字符串")
        value = value.strip().casefold()
        if value and value not in ASPECT_VALUES:
            allowed = ", ".join(sorted(ASPECT_VALUES))
            raise ValueError(f"{case_id} 的 {aspect} 无效：{value!r}（允许：{allowed}）")
        normalized[aspect] = value
    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError(f"{case_id} 的 notes 必须是字符串")
    normalized["notes"] = notes
    return normalized


def load_reviews(path: str | Path, *, allow_blank_judgment: bool = False) -> list[dict[str, Any]]:
    """Load and validate JSON/JSONL human review records."""

    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("reviews", payload.get("results"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"复核文件必须是非空数组、JSONL，或含 reviews/results 的对象：{source}")
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in payload:
        normalized = _validate_review(record, allow_blank_judgment=allow_blank_judgment)
        case_id = normalized["case_id"]
        if case_id in seen:
            raise ValueError(f"复核文件存在重复 case id：{case_id}")
        seen.add(case_id)
        reviews.append(normalized)
    return reviews


def build_review_template(
    cases: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
    *,
    require_all: bool = False,
) -> list[dict[str, Any]]:
    """Build blank human-review rows enriched with lexical audit evidence."""

    case_list = list(cases)
    answer_list = list(answers)
    audit = audit_answers(case_list, answer_list, require_all=require_all)
    template: list[dict[str, Any]] = []
    for row in audit["results"]:
        template.append(
            {
                "case_id": row["case_id"],
                "document_id": row.get("document_id"),
                "type": row.get("type"),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "mode": row.get("mode"),
                "latency_seconds": row.get("latency_seconds"),
                "answer_fact_status": row.get("answer_fact_status"),
                "answer_fact_coverage": row.get("answer_fact_coverage"),
                "missing_facts": row.get("missing_facts", []),
                "answer_refusal_detected": row.get("answer_refusal_detected", False),
                "refused_required_facts": row.get("refused_required_facts", []),
                "answer_risk_flags": row.get("answer_risk_flags", []),
                "judgment": "",
                "table_number": "",
                "units": "",
                "formula": "",
                "citation": "",
                "notes": "",
            }
        )
    return template


def summarize_reviews(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize human judgments and independently labeled answer aspects."""

    row_list = list(rows)
    judgments = Counter(str(row.get("judgment", "")) for row in row_list)
    aspects: dict[str, dict[str, int]] = {}
    for aspect in ASPECTS:
        counts = Counter(
            str(row.get(aspect, ""))
            for row in row_list
            if str(row.get(aspect, ""))
        )
        aspects[aspect] = dict(sorted(counts.items()))
    reviewed = sum(1 for row in row_list if row.get("judgment"))
    return {
        "reviewed_case_count": reviewed,
        "judgment_counts": dict(sorted(judgments.items())),
        "aspect_counts": aspects,
    }


def review_answers(
    cases: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Join lexical audit rows with human review labels without conflating them."""

    case_list = list(cases)
    answer_report = audit_answers(case_list, answers, require_all=require_all)
    answer_rows = {str(row["case_id"]): row for row in answer_report["results"]}
    case_ids = {
        str(case.get("case_id", case.get("id")))
        for case in case_list
    }
    normalized_reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for review in reviews:
        normalized = _validate_review(review, allow_blank_judgment=False)
        case_id = normalized["case_id"]
        if case_id in seen:
            raise ValueError(f"复核文件存在重复 case id：{case_id}")
        if case_id not in case_ids:
            raise ValueError(f"复核文件包含测试集之外的 case id：{case_id}")
        if case_id not in answer_rows:
            raise ValueError(f"复核文件包含没有对应回答的 case id：{case_id}")
        seen.add(case_id)
        normalized_reviews.append(normalized)
    missing_reviews = sorted(case_ids - seen)
    if require_all and missing_reviews:
        raise ValueError(f"人工复核缺少 case id：{', '.join(missing_reviews)}")

    review_map = {row["case_id"]: row for row in normalized_reviews}
    results: list[dict[str, Any]] = []
    for answer_row in answer_report["results"]:
        case_id = str(answer_row["case_id"])
        review = review_map.get(case_id)
        if review is None:
            continue
        results.append(
            {
                "case_id": case_id,
                "document_id": answer_row.get("document_id"),
                "type": answer_row.get("type"),
                "question": answer_row.get("question", ""),
                "answer_fact_status": answer_row.get("answer_fact_status"),
                "answer_fact_coverage": answer_row.get("answer_fact_coverage"),
                "missing_facts": answer_row.get("missing_facts", []),
                "answer_refusal_detected": answer_row.get("answer_refusal_detected", False),
                "refused_required_facts": answer_row.get("refused_required_facts", []),
                "answer_risk_flags": answer_row.get("answer_risk_flags", []),
                "judgment": review["judgment"],
                **{aspect: review[aspect] for aspect in ASPECTS},
                "notes": review["notes"],
            }
        )
    return {
        "schema": "sci-rag-human-review-v1",
        "case_count": len(case_ids),
        "answer_count": answer_report["answer_count"],
        "review_count": len(results),
        "missing_case_ids": missing_reviews,
        "answer_audit_summary": answer_report["summary"],
        "review_summary": summarize_reviews(results),
        "results": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"人工复核：{report['review_count']}/{report['case_count']} 题；"
        f"词面审计回答数：{report['answer_count']}"
    )
    print(f"判断统计：{report['review_summary']['judgment_counts']}")
    print(f"维度统计：{report['review_summary']['aspect_counts']}")
    if report["missing_case_ids"]:
        print(f"尚未复核：{', '.join(report['missing_case_ids'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线生成/校验 Sci-RAG 人工答案复核记录")
    parser.add_argument("--testset", required=True, help="JSON/JSONL 测试集路径")
    parser.add_argument("--answers", required=True, help="JSON/JSONL 回答路径")
    parser.add_argument("--reviews", help="已填写的 JSON/JSONL 人工复核路径")
    parser.add_argument("--template-out", help="生成空白复核模板的 JSONL 路径")
    parser.add_argument("--require-all", action="store_true", help="要求回答和复核都覆盖全部用例")
    parser.add_argument("--json-out", help="写入完整复核报告的 JSON 路径")
    args = parser.parse_args()
    if not args.reviews and not args.template_out:
        parser.error("必须指定 --reviews 或 --template-out")

    cases = _load_cases(args.testset)
    answers = _load_answers(args.answers)
    if args.template_out:
        template = build_review_template(cases, answers, require_all=args.require_all)
        output = Path(args.template_out)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in template),
            encoding="utf-8",
        )
        print(f"复核模板：{output}（{len(template)} 题）")
        if not args.reviews:
            return 0

    reviews = load_reviews(args.reviews)
    report = review_answers(cases, answers, reviews, require_all=args.require_all)
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
