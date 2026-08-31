#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline A/B comparison for two saved Sci-RAG answer runs.

The comparison deliberately reuses :mod:`evaluation.answer_audit` rather than
calling a model judge.  It is intended for fixed-question Dense/Hybrid or
other retrieval-mode comparisons: both answer files must contain the same case
IDs, and ``--require-all`` can require complete benchmark coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

# ``python evaluation/compare_answer_runs.py`` puts ``evaluation/`` at
# ``sys.path[0]``. Add the project root so package imports are deterministic.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.answer_audit import (  # noqa: E402
    _load_answers,
    _load_cases,
    audit_answers,
)


SUMMARY_DELTA_KEYS = (
    "required_fact_coverage_macro",
    "required_fact_coverage_micro",
    "full_fact_coverage_rate",
    "partial_fact_coverage_rate",
    "zero_fact_coverage_rate",
)


def _delta(candidate: Any, baseline: Any) -> float | None:
    """Return a numeric candidate-minus-baseline delta when both are scored."""

    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def compare_answer_runs(
    cases: Iterable[dict[str, Any]],
    baseline_answers: Iterable[dict[str, Any]],
    candidate_answers: Iterable[dict[str, Any]],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    require_all: bool = False,
) -> dict[str, Any]:
    """Compare two answer files using the same deterministic fact contract.

    A comparison is rejected when the runs contain different case IDs. This
    prevents an apparently improved aggregate from merely reflecting an easier
    subset of questions. ``require_all=True`` additionally requires every case
    in the supplied test set to appear in both runs.
    """

    case_list = list(cases)
    case_map: dict[str, dict[str, Any]] = {}
    for case in case_list:
        case_id = str(case.get("case_id", case.get("id")))
        if case_id in case_map:
            raise ValueError(f"测试集存在重复 case id：{case_id}")
        case_map[case_id] = case

    baseline_report = audit_answers(
        case_list, baseline_answers, require_all=require_all
    )
    candidate_report = audit_answers(
        case_list, candidate_answers, require_all=require_all
    )
    baseline_rows = {str(row["case_id"]): row for row in baseline_report["results"]}
    candidate_rows = {str(row["case_id"]): row for row in candidate_report["results"]}
    baseline_ids = set(baseline_rows)
    candidate_ids = set(candidate_rows)
    if baseline_ids != candidate_ids:
        missing_from_baseline = sorted(candidate_ids - baseline_ids)
        missing_from_candidate = sorted(baseline_ids - candidate_ids)
        details: list[str] = []
        if missing_from_baseline:
            details.append(f"baseline 缺少 {', '.join(missing_from_baseline)}")
        if missing_from_candidate:
            details.append(f"candidate 缺少 {', '.join(missing_from_candidate)}")
        raise ValueError("两次回答必须覆盖同一组 case id：" + "；".join(details))

    rows: list[dict[str, Any]] = []
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    transitions: Counter[str] = Counter()
    for case_id in (str(case.get("case_id", case.get("id"))) for case in case_list):
        if case_id not in baseline_rows:
            continue
        baseline = baseline_rows[case_id]
        candidate = candidate_rows[case_id]
        baseline_score = baseline.get("answer_fact_coverage")
        candidate_score = candidate.get("answer_fact_coverage")
        if baseline_score is not None and candidate_score is not None:
            if candidate_score > baseline_score:
                improved.append(case_id)
            elif candidate_score < baseline_score:
                regressed.append(case_id)
            else:
                unchanged.append(case_id)
        transition = f"{baseline['answer_fact_status']}->{candidate['answer_fact_status']}"
        transitions[transition] += 1
        rows.append(
            {
                "case_id": case_id,
                "document_id": case_map[case_id].get("document_id"),
                "type": case_map[case_id].get("type", "unknown"),
                "question": case_map[case_id].get("question", ""),
                "baseline": {
                    "name": baseline_name,
                    "answer": baseline["answer"],
                    "mode": baseline.get("mode"),
                    "latency_seconds": baseline.get("latency_seconds"),
                    "fact_coverage": baseline.get("answer_fact_coverage"),
                    "fact_status": baseline.get("answer_fact_status"),
                    "missing_facts": baseline.get("missing_facts", []),
                },
                "candidate": {
                    "name": candidate_name,
                    "answer": candidate["answer"],
                    "mode": candidate.get("mode"),
                    "latency_seconds": candidate.get("latency_seconds"),
                    "fact_coverage": candidate.get("answer_fact_coverage"),
                    "fact_status": candidate.get("answer_fact_status"),
                    "missing_facts": candidate.get("missing_facts", []),
                },
                "fact_coverage_delta": _delta(candidate_score, baseline_score),
            }
        )

    baseline_summary = baseline_report["summary"]
    candidate_summary = candidate_report["summary"]
    summary_delta = {
        key: _delta(candidate_summary.get(key), baseline_summary.get(key))
        for key in SUMMARY_DELTA_KEYS
    }
    return {
        "schema": "sci-rag-answer-compare-v1",
        "case_count": len(rows),
        "require_all": bool(require_all),
        "baseline": {
            "name": baseline_name,
            "answer_count": baseline_report["answer_count"],
            "missing_case_ids": baseline_report["missing_case_ids"],
            "summary": baseline_summary,
        },
        "candidate": {
            "name": candidate_name,
            "answer_count": candidate_report["answer_count"],
            "missing_case_ids": candidate_report["missing_case_ids"],
            "summary": candidate_summary,
        },
        "summary_delta_candidate_minus_baseline": summary_delta,
        "case_comparison": {
            "improved_case_ids": improved,
            "regressed_case_ids": regressed,
            "unchanged_case_ids": unchanged,
            "status_transitions": dict(sorted(transitions.items())),
        },
        "results": rows,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    return (
        f"fact macro/micro={fmt(summary.get('required_fact_coverage_macro'))}/"
        f"{fmt(summary.get('required_fact_coverage_micro'))}; "
        f"full/partial/zero={fmt(summary.get('full_fact_coverage_rate'))}/"
        f"{fmt(summary.get('partial_fact_coverage_rate'))}/"
        f"{fmt(summary.get('zero_fact_coverage_rate'))}"
    )


def _print_summary(report: dict[str, Any]) -> None:
    baseline = report["baseline"]
    candidate = report["candidate"]
    delta = report["summary_delta_candidate_minus_baseline"]
    changes = report["case_comparison"]
    print(f"答案 A/B 对照：{report['case_count']} 题")
    print(f"- {baseline['name']}: {_format_summary(baseline['summary'])}")
    print(f"- {candidate['name']}: {_format_summary(candidate['summary'])}")
    print(
        "- candidate-baseline: "
        f"macro={delta['required_fact_coverage_macro']!s}; "
        f"micro={delta['required_fact_coverage_micro']!s}; "
        f"full={delta['full_fact_coverage_rate']!s}"
    )
    print(
        "- 覆盖变化："
        f"提升 {len(changes['improved_case_ids'])}，"
        f"退化 {len(changes['regressed_case_ids'])}，"
        f"不变 {len(changes['unchanged_case_ids'])}"
    )
    if changes["status_transitions"]:
        print(f"- 状态转移：{changes['status_transitions']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线比较两次 Sci-RAG 答案运行")
    parser.add_argument("--testset", required=True, help="JSON/JSONL 测试集路径")
    parser.add_argument("--baseline", required=True, help="基线 JSON/JSONL 回答路径")
    parser.add_argument("--candidate", required=True, help="候选 JSON/JSONL 回答路径")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="要求两次回答都覆盖测试集全部用例",
    )
    parser.add_argument("--json-out", help="可选：写入完整 JSON 对照报告")
    args = parser.parse_args()

    report = compare_answer_runs(
        _load_cases(args.testset),
        _load_answers(args.baseline),
        _load_answers(args.candidate),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        require_all=args.require_all,
    )
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
