#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit the exact contexts produced by ``app.query_knowledge`` without an API call."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from evaluation.benchmark_loader import load_benchmark  # noqa: E402
from evaluation.context_coverage import aliases_for_fact, fact_is_present  # noqa: E402
from evaluation.generation_stability import (  # noqa: E402
    build_runtime,
    runtime_config_trace,
)


class _OfflineClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="offline context gate"))]
        )


def audit_case(case: dict[str, Any], contexts: list[str]) -> dict[str, Any]:
    required = [str(fact) for fact in case.get("required_facts", [])]
    missing = [
        fact
        for fact in required
        if not fact_is_present(fact, contexts, aliases=aliases_for_fact(case, fact))
    ]
    return {
        "case_id": str(case["case_id"]),
        "document_id": str(case["document_id"]),
        "required_fact_count": len(required),
        "covered_fact_count": len(required) - len(missing),
        "missing_facts": missing,
        "status": "full" if not missing else ("partial" if len(missing) < len(required) else "zero"),
    }


def run_gate(
    manifest_path: str | Path,
    db_path: str | Path,
) -> dict[str, Any]:
    benchmark = load_benchmark(manifest_path, verify_files=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    runtime = build_runtime(
        db_path,
        retrieval_mode="hybrid",
        document_routing=True,
        query_decomposition=True,
        parent_window=True,
        spatial_figure_evidence=True,
        formula_evidence=True,
        formula_evidence_auto=True,
    )
    runtime.client = _OfflineClient()
    chunk_count = int(runtime.collection.count())

    filenames = {
        str(document["document_id"]): str(document["filename"])
        for document in benchmark["documents"]
    }
    rows: list[dict[str, Any]] = []
    total_facts = covered_facts = 0
    for case in benchmark["cases"]:
        sources = [
            filenames[document_id]
            for document_id in [
                str(case["document_id"]),
                *(str(value) for value in case.get("additional_document_ids", [])),
            ]
        ]
        result = app.query_knowledge(
            str(case["question"]),
            return_contexts=True,
            runtime=runtime,
            source_filter=sources,
        )
        contexts = [str(value or "") for value in result.get("contexts") or []]
        row = audit_case(case, contexts)
        row.update(
            {
                "source_filter": sources,
                "context_count": len(contexts),
                "context_ids": result.get("context_ids") or [],
                "context_metadatas": result.get("context_metadatas") or [],
                "contexts": contexts,
            }
        )
        rows.append(row)
        total_facts += row["required_fact_count"]
        covered_facts += row["covered_fact_count"]

    full = sum(row["status"] == "full" for row in rows)
    return {
        "schema": "sci-rag-app-context-gate-v1",
        "manifest": str(Path(manifest_path).resolve()),
        "db_chunks": chunk_count,
        "runtime_config": runtime_config_trace(runtime),
        "case_count": len(rows),
        "full_case_count": full,
        "fact_micro": covered_facts / total_facts if total_facts else 1.0,
        "failures": [row for row in rows if row["status"] != "full"],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--db-path", required=True, help="已构建的隔离 ChromaDB 路径")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    try:
        report = run_gate(
            args.manifest,
            args.db_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"❌ 真实送模上下文门禁失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"真实送模上下文：{report['full_case_count']}/{report['case_count']} full；"
        f"fact micro={report['fact_micro']:.4f}"
    )
    for row in report["failures"]:
        print(f"- {row['case_id']}: {row['status']}; 遗漏={', '.join(row['missing_facts'])}")
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整 JSON：{output}")
    return 0 if not report["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
