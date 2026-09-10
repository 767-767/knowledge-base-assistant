#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate the multi-paper benchmark manifest offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark_loader import (  # noqa: E402
    BenchmarkValidationError,
    DEFAULT_MANIFEST,
    benchmark_summary,
    load_benchmark,
)
from evaluation.context_coverage import unsupported_gold_facts  # noqa: E402
from evaluation.context_coverage import aliases_for_fact, fact_is_present  # noqa: E402


def _missing_extracted_facts(
    case: dict[str, object],
    contexts: list[str],
) -> list[str]:
    """Return facts not detectable in the actual PDF-derived contexts."""

    required_facts = case.get("required_facts") or []
    return [
        str(fact)
        for fact in required_facts
        if not fact_is_present(
            str(fact),
            contexts,
            aliases=aliases_for_fact(case, str(fact)),
        )
    ]


def _verify_extracted_evidence(
    benchmark: dict[str, object],
    papers_dirs: list[str],
) -> dict[str, list[str]]:
    """Check frozen facts against page-scoped PDF parser output."""

    from app import load_and_split_document

    directories = [Path(directory).resolve() for directory in papers_dirs]
    chunks_by_document: dict[str, list[object]] = {}
    for document in benchmark["documents"]:  # type: ignore[index]
        filename = str(document["filename"])
        path = next(
            (directory / filename for directory in directories if (directory / filename).is_file()),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"找不到论文文件：{filename}")
        chunks_by_document[str(document["document_id"])] = load_and_split_document(str(path))

    failures: dict[str, list[str]] = {}
    for case in benchmark["cases"]:  # type: ignore[index]
        chunks = chunks_by_document[str(case["document_id"])]
        source_pages = {int(page) for page in case.get("source_pages", []) or []}
        contexts = [
            str(chunk.page_content)
            for chunk in chunks
            if not source_pages or int(chunk.metadata.get("page", 0)) in source_pages
        ]
        missing = _missing_extracted_facts(case, contexts)
        if missing:
            failures[str(case["case_id"])] = missing
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--papers-dir",
        action="append",
        dest="papers_dirs",
        help="包含清单中文件名的外部论文目录；可重复传入多个目录",
    )
    parser.add_argument("--verify-files", action="store_true", help="同时校验论文文件 SHA-256")
    parser.add_argument(
        "--verify-extracted-evidence",
        action="store_true",
        help="将 required_facts 对照 source_pages 的实际 PDF 解析输出校验",
    )
    parser.add_argument("--require-complete", action="store_true", help="要求达到 manifest.minimum_documents")
    args = parser.parse_args()

    try:
        benchmark = load_benchmark(
            args.manifest,
            papers_dir=args.papers_dirs,
            verify_files=args.verify_files or args.verify_extracted_evidence,
        )
    except BenchmarkValidationError as exc:
        print(f"❌ 基准集校验失败：{exc}", file=sys.stderr)
        return 1

    summary = benchmark_summary(benchmark)
    unsupported = {
        case["case_id"]: missing
        for case in benchmark["cases"]
        if (missing := unsupported_gold_facts(case))
    }
    if unsupported:
        for case_id, missing in unsupported.items():
            print(
                f"❌ 金标准片段不支持 required_facts：{case_id}: {', '.join(missing)}",
                file=sys.stderr,
            )
        return 3
    if args.verify_extracted_evidence:
        if not args.papers_dirs:
            print("❌ --verify-extracted-evidence 需要同时提供 --papers-dir", file=sys.stderr)
            return 4
        try:
            extracted_failures = _verify_extracted_evidence(benchmark, args.papers_dirs)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"❌ PDF 解析证据校验失败：{exc}", file=sys.stderr)
            return 4
        if extracted_failures:
            for case_id, missing in extracted_failures.items():
                print(
                    f"❌ 实际 PDF 解析输出未支持 required_facts：{case_id}: {', '.join(missing)}",
                    file=sys.stderr,
                )
            return 4
        print("实际 PDF 解析输出：全部 required_facts 均可在 source_pages 中识别")
    status = "complete" if summary["complete"] else "pending"
    print(f"✅ 基准集格式有效：{summary['benchmark_id']}")
    print(f"文档数：{summary['documents']}；用例数：{summary['cases']}；状态：{status}")
    print("required_facts：全部可由金标准片段或显式别名核验")
    for document_id, count in summary["cases_per_document"].items():
        print(f"  - {document_id}: {count} 题")
    if not summary["complete"]:
        print(
            f"⚠️ 尚未达到多论文门槛：当前 {summary['documents']} 篇，"
            f"最低要求 {summary['minimum_documents']} 篇。"
        )
        if args.require_complete:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
