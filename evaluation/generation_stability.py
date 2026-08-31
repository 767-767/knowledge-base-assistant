#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Repeat fixed Sci-RAG generations against an isolated database.

The runner is deliberately outside the normal UI path. It keeps one runtime
and one retrieval configuration for every repetition, writes one JSON object
per case/repetition outside the repository, and can resume after an API error
or an interrupted process. It does not run RAGAS or judge semantic truth.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import os
import sys
import time
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from evaluation.benchmark_loader import DEFAULT_MANIFEST, load_benchmark  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def select_cases(
    cases: Iterable[dict[str, Any]],
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select benchmark cases while preserving manifest order."""

    case_list = list(cases)
    requested = {str(value) for value in (case_ids or [])}
    if not requested:
        return case_list
    known = {str(case["case_id"]) for case in case_list}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError("未知 case id：" + ", ".join(unknown))
    return [case for case in case_list if str(case["case_id"]) in requested]


def answer_signature(answer: Any) -> str:
    """Normalize superficial whitespace for exact-repeat diagnostics."""

    return " ".join(str(answer or "").split())


def runtime_config_trace(runtime: app.Runtime) -> dict[str, Any]:
    """Return reproducibility metadata without secrets or live clients.

    Historical traces only recorded the feature switches.  That is not enough
    to prove that two runs used the same embedding/reranker, candidate pool,
    context size, or generation model.  Keep this payload JSON-safe and omit
    ``DEEPSEEK_API_KEY`` and the client/base URL deliberately.
    """

    config = runtime.config
    return {
        "embedding_model": config.embedding_model,
        "deepseek_model": config.deepseek_model,
        "db_path": str(Path(config.db_path).expanduser().resolve()),
        "retrieval_k": int(config.retrieval_k),
        "context_k": int(config.context_k),
        "retrieval_mode": config.retrieval_mode,
        "hybrid_candidate_k": int(config.hybrid_candidate_k),
        "hybrid_rrf_k": int(config.hybrid_rrf_k),
        "document_routing": bool(config.document_routing),
        "query_decomposition": bool(config.query_decomposition),
        "parent_window": bool(config.parent_window),
        "spatial_figure_evidence": bool(config.spatial_figure_evidence),
        "formula_evidence": bool(config.formula_evidence),
        "answer_validation": bool(config.answer_validation),
        "reranker_model": config.reranker_model,
        "reranker_revision": config.reranker_revision,
        "reranker_batch_size": int(config.reranker_batch_size),
        "reranker_max_length": int(config.reranker_max_length),
        "reranker_device": config.reranker_device,
        "reranker_rrf_k": int(config.reranker_rrf_k),
    }


def source_fingerprint() -> str:
    """Hash retrieval/generation source files used by this runner.

    This is a provenance marker, not a security signature.  It lets an audit
    distinguish traces generated before and after local code changes without
    recording the repository path or relying on an existing Git commit.
    """

    digest = hashlib.sha256()
    for relative in (
        "app.py",
        "sci_rag_core.py",
        "sci_rag_retrieval.py",
        "sci_rag_reranking.py",
        "evaluation/generation_stability.py",
    ):
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_runtime(
    db_path: str | Path,
    *,
    retrieval_mode: str,
    document_routing: bool,
    query_decomposition: bool,
    parent_window: bool,
    spatial_figure_evidence: bool,
    formula_evidence: bool | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    reranker_batch_size: int | None = None,
    reranker_max_length: int | None = None,
    reranker_device: str | None = None,
    reranker_rrf_k: int | None = None,
) -> app.Runtime:
    """Create the explicitly configured runtime used by every repeat."""

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    base = app.RuntimeConfig.from_env()
    overrides: dict[str, Any] = {
        "db_path": str(Path(db_path).expanduser().resolve()),
        "retrieval_mode": retrieval_mode,
        "document_routing": bool(document_routing),
        "query_decomposition": bool(query_decomposition),
        "parent_window": bool(parent_window),
        "spatial_figure_evidence": bool(spatial_figure_evidence),
    }
    optional_overrides = {
        "formula_evidence": formula_evidence,
        "reranker_model": reranker_model,
        "reranker_revision": reranker_revision,
        "reranker_batch_size": reranker_batch_size,
        "reranker_max_length": reranker_max_length,
        "reranker_device": reranker_device,
        "reranker_rrf_k": reranker_rrf_k,
    }
    overrides.update(
        {key: value for key, value in optional_overrides.items() if value is not None}
    )
    config = replace(base, **overrides)
    return app.create_runtime(config)


def run_stability(
    *,
    manifest_path: str | Path,
    db_path: str | Path,
    output_path: str | Path,
    repeats: int,
    case_ids: Iterable[str] | None = None,
    resume: bool = True,
    retrieval_mode: str = "hybrid",
    document_routing: bool = True,
    query_decomposition: bool = True,
    parent_window: bool = True,
    spatial_figure_evidence: bool = True,
    formula_evidence: bool | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    reranker_batch_size: int | None = None,
    reranker_max_length: int | None = None,
    reranker_device: str | None = None,
    reranker_rrf_k: int | None = None,
    expected_chunks: int | None = None,
) -> dict[str, Any]:
    """Run repeated generations and return a small execution summary."""

    if repeats <= 0:
        raise ValueError("repeats 必须为正数")
    output = Path(output_path).expanduser().resolve()
    benchmark = load_benchmark(manifest_path, verify_files=False)
    cases = select_cases(benchmark["cases"], case_ids)
    runtime = build_runtime(
        db_path,
        retrieval_mode=retrieval_mode,
        document_routing=document_routing,
        query_decomposition=query_decomposition,
        parent_window=parent_window,
        spatial_figure_evidence=spatial_figure_evidence,
        formula_evidence=formula_evidence,
        reranker_model=reranker_model,
        reranker_revision=reranker_revision,
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
        reranker_device=reranker_device,
        reranker_rrf_k=reranker_rrf_k,
    )
    chunk_count = int(runtime.collection.count())
    if expected_chunks is not None and chunk_count != int(expected_chunks):
        raise ValueError(
            f"隔离数据库块数不符：实际 {chunk_count}，期望 {expected_chunks}"
        )
    trace_config = runtime_config_trace(runtime)
    trace_source_fingerprint = source_fingerprint()

    if output.exists() and not resume:
        raise ValueError(f"输出文件已存在；如需续跑请不要使用 --no-resume：{output}")
    existing = _load_rows(output) if resume else []
    saved_rows = {
        (int(row.get("repeat", 0)), str(row.get("case_id"))): row
        for row in existing
        if row.get("repeat") and row.get("case_id")
    }
    completed = {
        (int(row.get("repeat", 0)), str(row.get("case_id")))
        for row in existing
        if row.get("repeat") and row.get("case_id") and not row.get("error")
    }
    requested_total = repeats * len(cases)
    attempted = len(existing)
    errors = 0
    for repeat in range(1, repeats + 1):
        for case_number, case in enumerate(cases, start=1):
            case_id = str(case["case_id"])
            if resume and (repeat, case_id) in completed:
                continue
            started = time.perf_counter()
            result: dict[str, Any] = {}
            attempts = 0
            for attempts in range(1, 4):
                result = app.query_knowledge(
                    str(case["question"]),
                    return_contexts=True,
                    runtime=runtime,
                )
                if not str(result.get("answer", "")).startswith("❌ 调用出错："):
                    break
                if attempts < 3:
                    time.sleep(2 * attempts)
            answer = str(result.get("answer", ""))
            is_error = answer.startswith("❌ 调用出错：")
            errors += int(is_error)
            row = {
                "repeat": repeat,
                "case_number": case_number,
                "case_id": case_id,
                "document_id": str(case.get("document_id", "")),
                "question": str(case["question"]),
                "answer": answer,
                "answer_signature": answer_signature(answer),
                "mode": "generation_stability",
                "runtime_config": trace_config,
                "source_fingerprint": trace_source_fingerprint,
                "retrieval_mode": retrieval_mode,
                "document_routing": bool(document_routing),
                "query_decomposition": bool(query_decomposition),
                "parent_window": bool(parent_window),
                "spatial_figure_evidence": bool(spatial_figure_evidence),
                "attempts": attempts,
                "error": is_error,
                "latency_seconds": time.perf_counter() - started,
                "contexts": result.get("contexts") or [],
                "context_ids": result.get("context_ids") or [],
                "context_metadatas": result.get("context_metadatas") or [],
                "context_count": len(result.get("contexts") or []),
                "db_chunks": chunk_count,
            }
            saved_rows[(repeat, case_id)] = row
            _write_rows(
                output,
                sorted(saved_rows.values(), key=lambda item: (int(item["repeat"]), int(item["case_number"]))),
            )
            attempted += 1
            print(
                f"{repeat}/{repeats} {case_number}/{len(cases)} {case_id} "
                f"{row['latency_seconds']:.2f}s error={is_error}",
                flush=True,
            )
    return {
        "output": str(output),
        "db_chunks": chunk_count,
        "cases": len(cases),
        "repeats": repeats,
        "requested_rows": requested_total,
        "rows_present_after_run": len(_load_rows(output)),
        "rows_attempted_this_run": attempted - len(existing),
        "errors_this_run": errors,
        "resumed": bool(resume),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--db-path", required=True, help="隔离 ChromaDB 路径")
    parser.add_argument("--output", required=True, help="仓库外 JSONL 输出路径")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--expected-chunks", type=int)
    parser.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--no-document-routing", action="store_true")
    parser.add_argument("--no-query-decomposition", action="store_true")
    parser.add_argument("--no-parent-window", action="store_true")
    parser.add_argument("--no-spatial-figure-evidence", action="store_true")
    parser.add_argument(
        "--formula-evidence",
        action="store_true",
        default=None,
        help="启用显式公式候选通道（默认沿用 .env）",
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--reranker-batch-size", type=int)
    parser.add_argument("--reranker-max-length", type=int)
    parser.add_argument("--reranker-device")
    parser.add_argument("--reranker-rrf-k", type=int)
    args = parser.parse_args()
    try:
        summary = run_stability(
            manifest_path=args.manifest,
            db_path=args.db_path,
            output_path=args.output,
            repeats=args.repeats,
            case_ids=args.case_ids,
            resume=not args.no_resume,
            retrieval_mode=args.retrieval_mode,
            document_routing=not args.no_document_routing,
            query_decomposition=not args.no_query_decomposition,
            parent_window=not args.no_parent_window,
            spatial_figure_evidence=not args.no_spatial_figure_evidence,
            formula_evidence=args.formula_evidence,
            reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            reranker_device=args.reranker_device,
            reranker_rrf_k=args.reranker_rrf_k,
            expected_chunks=args.expected_chunks,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"❌ 稳定性复测失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
