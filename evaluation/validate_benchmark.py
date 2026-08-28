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
    parser.add_argument("--require-complete", action="store_true", help="要求达到 manifest.minimum_documents")
    args = parser.parse_args()

    try:
        benchmark = load_benchmark(
            args.manifest,
            papers_dir=args.papers_dirs,
            verify_files=args.verify_files,
        )
    except BenchmarkValidationError as exc:
        print(f"❌ 基准集校验失败：{exc}", file=sys.stderr)
        return 1

    summary = benchmark_summary(benchmark)
    status = "complete" if summary["complete"] else "pending"
    print(f"✅ 基准集格式有效：{summary['benchmark_id']}")
    print(f"文档数：{summary['documents']}；用例数：{summary['cases']}；状态：{status}")
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
