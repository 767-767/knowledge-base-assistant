#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline, auditable answer completeness checks.

This module checks whether an answer explicitly contains the atomic facts that
were declared for a benchmark case.  It intentionally does not judge semantic
equivalence, truth beyond the gold facts, or citation quality; aliases must be
declared in the case and are validated against gold contexts elsewhere.

The command-line entry point accepts a test set plus a user-produced JSON or
JSONL answer file.  It never imports an embedding model, ChromaDB, Gradio,
RAGAS, or an external API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

# ``python evaluation/answer_audit.py`` puts ``evaluation/`` (rather than the
# repository root) at ``sys.path[0]``. Add the root explicitly so the CLI and
# package import use the same path resolution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.context_coverage import (
    aggregate_fact_coverage,
    aliases_for_fact,
    case_fact_coverage,
    fact_is_present,
)


_REFUSAL_LANGUAGE_RE = re.compile(
    r"资料(?:中)?(?:未提供|没有提供|未明确|没有明确|未说明|没有说明)(?!其他)|"
    r"(?:无法|不能|不可|不确定).{0,24}(?:回答|确认|判断|推断|确定)|"
    r"(?:not provided|not specified|not available|unclear|cannot answer|"
    r"unable to answer|insufficient information|cannot determine)",
    re.IGNORECASE,
)
_ANSWER_SEGMENT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def _answer_refusal_risks(
    answer_case: dict[str, Any],
    answer: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Detect refusal language without changing lexical fact coverage.

    A model can mention a required fact while saying that the fact is not
    available.  The lexical audit must remain a transparent string matcher,
    so this helper emits an independent review signal instead of relabeling a
    ``full``/``partial`` result as semantically wrong.
    """

    text = str(answer or "")
    segments = [segment for segment in _ANSWER_SEGMENT_RE.split(text) if segment]
    refusal_segments = [
        (segment, list(_REFUSAL_LANGUAGE_RE.finditer(segment)))
        for segment in segments
    ]
    refusals = [match for _segment, matches in refusal_segments for match in matches]
    refused_facts: list[str] = []
    required = [
        str(fact)
        for fact in answer_case.get("required_facts") or []
    ]
    for segment, segment_refusals in refusal_segments:
        for match in segment_refusals:
            # A fact is considered refused only when it follows the refusal
            # phrase. Facts mentioned earlier in a valid answer should not be
            # marked merely because the answer adds a caveat about other details.
            window = segment[match.end() :]
            if re.search(r"其他|另一种|额外|other|another|additional", window, re.IGNORECASE):
                continue
            for fact in required:
                if fact in refused_facts:
                    continue
                if fact_is_present(
                    fact,
                    [window],
                    aliases_for_fact(answer_case, fact)[1:],
                ):
                    refused_facts.append(fact)
    flags: list[str] = []
    if refusals:
        flags.append("refusal_language_present")
    if refused_facts:
        flags.append("required_fact_mentioned_in_refusal")
    if refused_facts and coverage.get("fact_coverage_status") == "full":
        flags.append("full_coverage_with_refusal")
    return {
        "answer_refusal_detected": bool(refusals),
        "refused_required_facts": refused_facts,
        "answer_risk_flags": flags,
    }


def audit_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Return one answer-level fact audit row for a benchmark case."""

    answer_case = dict(case)
    answer_required = [
        str(fact)
        for fact in case.get("answer_required_facts", case.get("required_facts")) or []
    ]
    answer_case["required_facts"] = answer_required
    merged_aliases = {
        str(fact): [str(value) for value in values]
        for fact, values in (case.get("required_fact_aliases") or {}).items()
    }
    for fact, values in (case.get("answer_fact_aliases") or {}).items():
        merged_aliases.setdefault(str(fact), []).extend(str(value) for value in values)
        merged_aliases[str(fact)] = list(dict.fromkeys(merged_aliases[str(fact)]))
    answer_case["required_fact_aliases"] = merged_aliases
    coverage = case_fact_coverage(answer_case, [answer])
    refusal_risks = _answer_refusal_risks(answer_case, answer, coverage)
    return {
        "id": case.get("id", case.get("case_id")),
        "case_id": case.get("case_id", case.get("id")),
        "document_id": case.get("document_id"),
        "type": case.get("type", "unknown"),
        "question": case.get("question", ""),
        "answer": str(answer),
        "required_facts": answer_required,
        "retrieval_required_facts": [
            str(fact) for fact in case.get("required_facts") or []
        ],
        "required_fact_aliases": merged_aliases,
        "answer_required_facts": answer_required,
        "answer_fact_aliases": case.get("answer_fact_aliases") or {},
        "answer_fact_coverage": coverage["required_fact_coverage"],
        "answer_fact_status": coverage["fact_coverage_status"],
        **refusal_risks,
        "matched_facts": coverage["matched_required_facts"],
        "missing_facts": coverage["missing_required_facts"],
        # Keep the canonical field names used by context_coverage aggregation
        # so answer and retrieval reports remain directly comparable.
        "required_fact_coverage": coverage["required_fact_coverage"],
        "fact_coverage_status": coverage["fact_coverage_status"],
        "matched_required_facts": coverage["matched_required_facts"],
        "missing_required_facts": coverage["missing_required_facts"],
        "matched_required_fact_count": coverage["matched_required_fact_count"],
        "matched_fact_count": coverage["matched_required_fact_count"],
        "required_fact_count": coverage["required_fact_count"],
    }


def aggregate_answer_audit(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate answer rows with the same macro/micro semantics as retrieval."""

    row_list = list(rows)
    fact_summary = aggregate_fact_coverage(row_list)
    risk_rows = [row for row in row_list if row.get("answer_risk_flags")]
    refusal_rows = [row for row in row_list if row.get("answer_refusal_detected")]
    conflict_rows = [
        row
        for row in row_list
        if "required_fact_mentioned_in_refusal" in (row.get("answer_risk_flags") or [])
    ]
    full_refusal_rows = [
        row
        for row in row_list
        if "full_coverage_with_refusal" in (row.get("answer_risk_flags") or [])
    ]
    return {
        **fact_summary,
        "answer_risk_case_count": len(risk_rows),
        "answer_refusal_case_count": len(refusal_rows),
        "required_fact_refusal_conflict_case_count": len(conflict_rows),
        "full_coverage_with_refusal_case_count": len(full_refusal_rows),
    }


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        raw_cases = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The multi-paper benchmark stores the 11 DrugR cases as pointers to
        # the original test set. Resolve the complete manifest here so users
        # can pass ``evaluation/benchmark/cases.jsonl`` directly to this CLI.
        if any(
            isinstance(case, dict)
            and ("source_testset" in case or "source_case_id" in case)
            for case in raw_cases
        ):
            manifest_path = source.parent / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(
                    f"指针式 benchmark 用例缺少同目录 manifest.json：{source}"
                )
            try:
                from evaluation.benchmark_loader import load_benchmark

                cases = load_benchmark(manifest_path, verify_files=False)["cases"]
            except (OSError, ValueError) as exc:
                raise ValueError(f"无法解析 benchmark 用例指针：{source}: {exc}") from exc
        else:
            cases = raw_cases
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "documents" in payload and "cases_path" in payload:
            try:
                from evaluation.benchmark_loader import load_benchmark

                cases = load_benchmark(source, verify_files=False)["cases"]
            except (OSError, ValueError) as exc:
                raise ValueError(f"无法解析 benchmark manifest：{source}: {exc}") from exc
        else:
            cases = payload.get("test_cases") if isinstance(payload, dict) else payload
            if cases is None and isinstance(payload, dict):
                cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"测试集必须是非空 JSON 数组或含 test_cases/cases 的对象：{source}")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"测试集中的用例必须是对象：{source}")
        case_id = case.get("case_id", case.get("id"))
        if case_id is None:
            raise ValueError(f"测试用例缺少 id/case_id：{source}")
        normalized_id = str(case_id)
        if normalized_id in seen:
            raise ValueError(f"测试集存在重复 case id：{normalized_id}：{source}")
        seen.add(normalized_id)
        retrieval_facts = {
            str(fact) for fact in case.get("required_facts") or []
        }
        answer_required = case.get("answer_required_facts")
        if answer_required is not None:
            if (
                not isinstance(answer_required, list)
                or not answer_required
                or not all(isinstance(fact, str) and fact.strip() for fact in answer_required)
            ):
                raise ValueError(f"用例 {normalized_id} 的 answer_required_facts 必须是非空字符串数组：{source}")
            unknown = sorted(set(answer_required) - retrieval_facts)
            if unknown:
                raise ValueError(
                    f"用例 {normalized_id} 的 answer_required_facts 含未声明事实：{', '.join(unknown)}：{source}"
                )
        answer_aliases = case.get("answer_fact_aliases")
        if answer_aliases is not None:
            if not isinstance(answer_aliases, dict):
                raise ValueError(f"用例 {normalized_id} 的 answer_fact_aliases 必须是对象：{source}")
            allowed = set(answer_required or retrieval_facts)
            unknown = sorted(set(answer_aliases) - allowed)
            if unknown:
                raise ValueError(
                    f"用例 {normalized_id} 的 answer_fact_aliases 含未声明事实：{', '.join(unknown)}：{source}"
                )
            for fact, values in answer_aliases.items():
                if (
                    not isinstance(values, list)
                    or not values
                    or not all(isinstance(value, str) and value.strip() for value in values)
                ):
                    raise ValueError(
                        f"用例 {normalized_id} 的事实 {fact} answer 别名必须是非空字符串数组：{source}"
                    )
        normalized.append({**case, "case_id": normalized_id})
    return normalized


def _load_answers(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        payload: Any = [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("answers", payload.get("results"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"回答文件必须是非空数组、JSONL，或含 answers/results 的对象：{source}")

    answers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError(f"回答记录必须是对象：{source}")
        case_id = record.get("case_id", record.get("id"))
        if case_id is None or "answer" not in record:
            raise ValueError(f"回答记录必须含 id/case_id 和 answer：{source}")
        key = str(case_id)
        if key in seen:
            raise ValueError(f"回答文件存在重复 case id：{key}")
        seen.add(key)
        answers.append({**record, "case_id": key, "answer": str(record["answer"])})
    return answers


def audit_answers(
    cases: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Audit supplied answers against cases and return a serializable report."""

    case_map = {str(case.get("case_id", case.get("id"))): case for case in cases}
    answer_records = list(answers)
    rows: list[dict[str, Any]] = []
    unknown: list[str] = []
    for record in answer_records:
        case_id = str(record.get("case_id", record.get("id")))
        case = case_map.get(case_id)
        if case is None:
            unknown.append(case_id)
            continue
        row = audit_answer(case, record["answer"])
        row["mode"] = record.get("mode")
        row["latency_seconds"] = record.get("latency_seconds")
        rows.append(row)

    expected_ids = set(case_map)
    supplied_ids = {str(row["case_id"]) for row in rows}
    missing_ids = sorted(expected_ids - supplied_ids)
    if unknown:
        raise ValueError(f"回答文件包含测试集之外的 case id：{', '.join(sorted(unknown))}")
    if require_all and missing_ids:
        raise ValueError(f"回答文件缺少 case id：{', '.join(missing_ids)}")

    summary = aggregate_answer_audit(rows)
    return {
        "schema": "sci-rag-answer-audit-v1",
        "case_count": len(case_map),
        "answer_count": len(rows),
        "missing_case_ids": missing_ids,
        "summary": summary,
        "results": rows,
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    print(f"答案审计：{report['answer_count']}/{report['case_count']} 题")
    print(
        "fact macro/micro="
        f"{fmt(summary['required_fact_coverage_macro'])}/"
        f"{fmt(summary['required_fact_coverage_micro'])}; "
        "full/partial/zero="
        f"{fmt(summary['full_fact_coverage_rate'])}/"
        f"{fmt(summary['partial_fact_coverage_rate'])}/"
        f"{fmt(summary['zero_fact_coverage_rate'])}"
    )
    for row in report["results"]:
        if row["answer_fact_status"] != "full":
            print(
                f"- {row['case_id']}: {row['answer_fact_status']}; "
                f"遗漏={', '.join(row['missing_facts']) or '无'}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="离线审计 benchmark 答案是否覆盖声明的原子事实")
    parser.add_argument("--testset", required=True, help="JSON/JSONL 测试集路径")
    parser.add_argument("--answers", required=True, help="用户保存的 JSON/JSONL 回答路径")
    parser.add_argument("--require-all", action="store_true", help="要求回答文件覆盖测试集全部用例")
    parser.add_argument("--json-out", help="可选：写入完整 JSON 审计报告")
    args = parser.parse_args()

    report = audit_answers(
        _load_cases(args.testset),
        _load_answers(args.answers),
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
