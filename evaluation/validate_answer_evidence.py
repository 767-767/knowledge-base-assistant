#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline evidence-only validation for generated answer traces.

The validator never loads Chroma, an embedding model, Gradio, RAGAS, or an
external API. It consumes a JSONL trace containing ``question``, ``answer``,
``contexts`` and ``context_metadatas`` (as produced by
``evaluation/generation_stability.py``), rebuilds the same evidence ledger
used by the prompt, and emits review signals. It does not read benchmark
``ground_truth`` or ``required_facts`` and therefore cannot claim semantic
answer correctness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sci_rag_core import build_evidence_ledger, validate_answer_against_evidence  # noqa: E402


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是有效 JSON：{source}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象：{source}")
        for field in ("question", "answer", "contexts"):
            if field not in row:
                raise ValueError(f"第 {line_number} 行缺少字段 {field}：{source}")
        rows.append(row)
    if not rows:
        raise ValueError(f"答案 trace 不能为空：{source}")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for row in rows:
        contexts = [str(value or "") for value in row.get("contexts") or []]
        metadatas = [value if isinstance(value, dict) else {} for value in row.get("context_metadatas") or []]
        ledger = build_evidence_ledger(row["question"], contexts, metadatas)
        validation = validate_answer_against_evidence(row["question"], row["answer"], ledger)
        results.append(
            {
                "repeat": row.get("repeat"),
                "case_id": row.get("case_id", row.get("id")),
                "question": row["question"],
                "answer": row["answer"],
                "validation": validation,
            }
        )
    status_counts = Counter(result["validation"]["status"] for result in results)
    reason_counts = Counter(
        reason
        for result in results
        for reason in result["validation"].get("reasons", [])
    )
    review_ids = [
        result["case_id"]
        for result in results
        if result["validation"]["status"] == "review"
    ]
    return {
        "schema": "sci-rag-answer-evidence-validation-v1",
        "row_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_case_ids": review_ids,
        "results": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"证据答案校验：{report['row_count']} 条生成记录")
    print("状态：" + ", ".join(f"{key}={value}" for key, value in report["status_counts"].items()))
    if report["reason_counts"]:
        print("原因：" + ", ".join(f"{key}={value}" for key, value in report["reason_counts"].items()))
    print(f"需人工/显式重试复核：{len(report['review_case_ids'])} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, help="generation trace JSONL")
    parser.add_argument("--json-out", help="optional report path; otherwise only print summary")
    args = parser.parse_args()
    report = validate_rows(load_jsonl(args.answers))
    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
