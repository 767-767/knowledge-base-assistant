#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit repeat-generation traces without calling models or reading gold facts.

The report answers a narrow provenance question: for each case, were the
retrieval/generation settings and source fingerprint recorded, did the context
IDs stay stable, and did the answer text vary?  It deliberately does not
judge semantic correctness and does not load the application runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {line_number} 行无效：{source}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象：{source}")
        rows.append(value)
    return rows


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _context_signature(row: dict[str, Any]) -> str:
    ids = row.get("context_ids")
    if isinstance(ids, list) and ids:
        return _stable_json([str(value) for value in ids])
    contexts = row.get("contexts")
    if not isinstance(contexts, list):
        contexts = []
    digest = hashlib.sha256()
    for context in contexts:
        digest.update(str(context).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _metadata_signature(row: dict[str, Any]) -> str:
    metadata = row.get("context_metadatas")
    return _stable_json(metadata if isinstance(metadata, list) else [])


def _answer_signature(row: dict[str, Any]) -> str:
    explicit = row.get("answer_signature")
    if explicit is not None:
        return " ".join(str(explicit).split())
    return " ".join(str(row.get("answer", "") or "").split())


def audit_generation_trace(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeat consistency and provenance completeness."""

    row_list = list(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_keys: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in row_list:
        case_id = str(row.get("case_id", ""))
        repeat = str(row.get("repeat", ""))
        key = (case_id, repeat)
        if key in seen_keys:
            duplicate_keys.append(f"{case_id}@{repeat}")
        seen_keys.add(key)
        groups[case_id].append(row)

    case_reports: list[dict[str, Any]] = []
    for case_id in sorted(groups):
        case_rows = groups[case_id]
        config_signatures = {
            _stable_json(row.get("runtime_config"))
            for row in case_rows
            if isinstance(row.get("runtime_config"), dict)
        }
        source_fingerprints = {
            str(row.get("source_fingerprint"))
            for row in case_rows
            if row.get("source_fingerprint")
        }
        missing_provenance_rows = sum(
            int(not isinstance(row.get("runtime_config"), dict) or not row.get("source_fingerprint"))
            for row in case_rows
        )
        context_signatures = {_context_signature(row) for row in case_rows}
        metadata_signatures = {_metadata_signature(row) for row in case_rows}
        answer_signatures = {_answer_signature(row) for row in case_rows}
        errors = sum(int(bool(row.get("error"))) for row in case_rows)
        repeated = len(case_rows) > 1
        case_reports.append(
            {
                "case_id": case_id,
                "row_count": len(case_rows),
                "repeat_count": len(case_rows),
                "repeated": repeated,
                "provenance_complete": missing_provenance_rows == 0,
                "missing_provenance_rows": missing_provenance_rows,
                "runtime_config_count": len(config_signatures),
                "source_fingerprint_count": len(source_fingerprints),
                "context_signature_count": len(context_signatures),
                "metadata_signature_count": len(metadata_signatures),
                "answer_signature_count": len(answer_signatures),
                "context_stable": repeated and len(context_signatures) == 1,
                "metadata_stable": repeated and len(metadata_signatures) == 1,
                "answer_exactly_stable": repeated and len(answer_signatures) == 1,
                "configuration_stable": repeated and len(config_signatures) == 1 and len(source_fingerprints) == 1,
                "errors": errors,
            }
        )

    repeated_reports = [row for row in case_reports if row["repeated"]]
    count = len(repeated_reports)

    def rate(field: str) -> float | None:
        if not count:
            return None
        return sum(bool(row[field]) for row in repeated_reports) / count

    aggregate = {
        "rows": len(row_list),
        "cases": len(case_reports),
        "repeated_cases": count,
        "duplicate_repeat_keys": len(duplicate_keys),
        "provenance_complete_cases": sum(bool(row["provenance_complete"]) for row in case_reports),
        "provenance_incomplete_rows": sum(int(row["missing_provenance_rows"]) for row in case_reports),
        "configuration_mismatch_cases": sum(
            bool(row["repeated"] and not row["configuration_stable"]) for row in case_reports
        ),
        "context_changed_cases": sum(
            bool(row["repeated"] and not row["context_stable"]) for row in case_reports
        ),
        "metadata_changed_cases": sum(
            bool(row["repeated"] and not row["metadata_stable"]) for row in case_reports
        ),
        "answer_variation_cases": sum(
            bool(row["repeated"] and not row["answer_exactly_stable"]) for row in case_reports
        ),
        "error_rows": sum(int(row.get("errors", 0)) for row in case_reports),
        "context_stable_rate": rate("context_stable"),
        "metadata_stable_rate": rate("metadata_stable"),
        "answer_exactly_stable_rate": rate("answer_exactly_stable"),
        "configuration_stable_rate": rate("configuration_stable"),
    }
    return {
        "schema": "sci-rag-generation-trace-audit-v1",
        "aggregate": aggregate,
        "duplicate_repeat_keys": sorted(set(duplicate_keys)),
        "cases": case_reports,
    }


def _print_summary(report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    print(
        f"生成 trace 审计：{aggregate['rows']} 行、{aggregate['cases']} 个 case，"
        f"重复 case {aggregate['repeated_cases']} 个"
    )
    print(
        "- provenance 完整 case："
        f"{aggregate['provenance_complete_cases']}/{aggregate['cases']}；"
        f"缺失 provenance 行：{aggregate['provenance_incomplete_rows']}"
    )
    print(
        "- 重复 case 稳定率："
        f"config={aggregate['configuration_stable_rate']!s}，"
        f"context={aggregate['context_stable_rate']!s}，"
        f"metadata={aggregate['metadata_stable_rate']!s}，"
        f"答案逐字={aggregate['answer_exactly_stable_rate']!s}"
    )
    print(
        "- 变化 case："
        f"config={aggregate['configuration_mismatch_cases']}，"
        f"context={aggregate['context_changed_cases']}，"
        f"答案措辞={aggregate['answer_variation_cases']}"
    )
    if aggregate["duplicate_repeat_keys"]:
        print(f"- 重复 repeat key：{aggregate['duplicate_repeat_keys']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="generation_stability 输出 JSONL")
    parser.add_argument("--json-out", help="可选：写出完整审计 JSON")
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="若任何行缺少 runtime_config 或 source_fingerprint，则返回失败",
    )
    args = parser.parse_args()
    try:
        report = audit_generation_trace(_load_jsonl(args.trace))
    except (OSError, ValueError) as exc:
        print(f"❌ trace 审计失败：{exc}", file=sys.stderr)
        return 1
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    if args.require_provenance and report["aggregate"]["provenance_incomplete_rows"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
