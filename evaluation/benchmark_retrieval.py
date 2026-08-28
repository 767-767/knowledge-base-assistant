#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline retrieval diagnostics for the multi-paper benchmark.

PDF parsing is delegated to the same side-effect-free ingestion path used by
the application. BM25 uses only local code; dense and Hybrid/RRF optionally use
an already-cached Sentence-Transformers model in forced offline mode. No
ChromaDB, UI, or external API is loaded. The report is a retrieval diagnostic,
not an answer-quality or RAGAS score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import load_and_split_document  # noqa: E402
from evaluation.benchmark_loader import DEFAULT_MANIFEST, load_benchmark  # noqa: E402
from evaluation.context_coverage import (  # noqa: E402
    aggregate_fact_coverage,
    case_fact_coverage,
)
from sci_rag_core import Chunk, normalize_for_match, table_number_from_question  # noqa: E402
from sci_rag_retrieval import BM25Index, RankedItem, reciprocal_rank_fusion  # noqa: E402


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*")


def evidence_tokens(value: Any) -> list[str]:
    """Return stable ASCII tokens for comparing English PDF evidence snippets."""

    return [token.casefold() for token in ASCII_TOKEN_RE.findall(normalize_for_match(value))]


def searchable_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    return "\n".join(
        [
            chunk.page_content,
            str(metadata.get("table_caption", "")),
            str(metadata.get("headers", "")),
        ]
    )


class DenseIndex:
    """Local-only Sentence-Transformers index used for the hybrid comparison."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model: Any | None = None,
    ):
        # Never allow a benchmark run to turn into an implicit model download.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.chunks = list(chunks)
        self.model_name = model_name
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, local_files_only=True)
        self.model = model
        self.embeddings = self.model.encode(
            [searchable_text(chunk) for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, question: str, k: int = 10) -> list[RankedItem]:
        limit = max(0, min(int(k), len(self.chunks)))
        if not self.chunks or not limit:
            return []
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_embedding
        ranked = [RankedItem(index, float(scores[index])) for index in range(len(self.chunks))]
        ranked.sort(key=lambda item: (-item.score, int(item.key)))
        return ranked[:limit]


class HybridRetriever:
    """BM25, local dense, or RRF-fused retrieval over one in-memory corpus."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        mode: str = "bm25",
        dense_model_name: str = "BAAI/bge-small-zh-v1.5",
        rrf_k: int = 60,
        dense_model: Any | None = None,
    ):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"不支持的 retriever：{mode}")
        self.chunks = list(chunks)
        self.mode = mode
        self.rrf_k = rrf_k
        self.bm25 = BM25Index(searchable_text(chunk) for chunk in self.chunks)
        self.dense = (
            DenseIndex(self.chunks, dense_model_name, model=dense_model)
            if mode in {"dense", "hybrid"}
            else None
        )

    def retrieve(self, question: str, k: int = 10) -> list[RankedItem]:
        if self.mode == "bm25":
            return self.bm25.retrieve(question, k)
        if self.mode == "dense":
            assert self.dense is not None
            return self.dense.retrieve(question, k)
        candidate_k = min(len(self.chunks), max(int(k) * 5, 50))
        assert self.dense is not None
        lexical = (
            self.bm25.retrieve(question, candidate_k)
            if self.bm25.has_lexical_signal(question)
            else []
        )
        return reciprocal_rank_fusion(
            [lexical, self.dense.retrieve(question, candidate_k)],
            rrf_k=self.rrf_k,
            limit=k,
        )


def _reference_context_match(reference: str, chunk_text: str, threshold: float = 0.6) -> bool:
    reference_set = set(evidence_tokens(reference))
    if not reference_set:
        return False
    chunk_set = set(evidence_tokens(chunk_text))
    overlap = len(reference_set & chunk_set) / len(reference_set)
    return overlap >= threshold


def _target_ranked_chunks(
    case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]
) -> list[Chunk]:
    """Keep evidence metrics scoped to the case's target document.

    Ranking may be global across all papers, but a similarly worded chunk from
    another paper must not count as evidence for this case.  The optional
    ``benchmark_document_id`` is attached by :func:`run_diagnostic`.
    """

    target_document = case.get("document_id")
    selected = [chunks[int(result.key)] for result in ranked]
    if not target_document:
        return selected
    marked = [
        chunk
        for chunk in selected
        if chunk.metadata.get("benchmark_document_id") is None
        or chunk.metadata.get("benchmark_document_id") == target_document
    ]
    return marked


def _case_context_recall(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]) -> float:
    references = [str(context) for context in case.get("contexts", [])]
    if not references:
        return 0.0
    retrieved = [chunk.page_content for chunk in _target_ranked_chunks(case, chunks, ranked)]
    matched = sum(
        any(_reference_context_match(reference, text) for text in retrieved)
        for reference in references
    )
    return matched / len(references)


def _case_page_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]) -> bool | None:
    source_pages = {int(page) for page in case.get("source_pages", []) if str(page).isdigit()}
    if not source_pages:
        return None
    return any(chunk.metadata.get("page") in source_pages for chunk in _target_ranked_chunks(case, chunks, ranked))


def _case_document_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]) -> bool | None:
    target_document = case.get("document_id")
    if not target_document:
        return None
    return any(
        chunks[int(result.key)].metadata.get("benchmark_document_id") == target_document
        for result in ranked
    )


def _case_table_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]) -> bool | None:
    table_number = table_number_from_question(str(case.get("question", "")))
    if table_number is None:
        return None
    target = int(table_number)
    return any(
        chunk.metadata.get("type") == "table" and chunk.metadata.get("table_number") == target
        for chunk in _target_ranked_chunks(case, chunks, ranked)
    )


def _case_required_fact_coverage(
    case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedItem]
) -> dict[str, Any]:
    contexts = [
        chunk.page_content for chunk in _target_ranked_chunks(case, chunks, ranked)
    ]
    return case_fact_coverage(case, contexts)


def _metric_mean(values: list[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def aggregate_case_results(
    case_results: list[dict[str, Any]], top_k_values: list[int]
) -> dict[str, dict[str, Any]]:
    """Aggregate case-level diagnostics without weighting documents equally."""

    aggregate: dict[str, dict[str, Any]] = {}
    for top_k in top_k_values:
        rows = [result["metrics"][str(top_k)] for result in case_results]
        aggregate[str(top_k)] = {
            "reference_context_recall": _metric_mean([row["reference_context_recall"] for row in rows]),
            "target_document_hit_rate": _metric_mean([row["target_document_hit"] for row in rows]),
            "source_page_hit_rate": _metric_mean([row["source_page_hit"] for row in rows]),
            "table_number_hit_rate": _metric_mean(
                [row["table_number_hit"] for row in rows if row["table_number_hit"] is not None]
            ),
            **aggregate_fact_coverage(rows),
        }
    return aggregate


def aggregate_fact_coverage_by(
    case_results: list[dict[str, Any]],
    top_k_values: list[int],
    field: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate global fact coverage by document or question type."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for result in case_results:
        groups.setdefault(str(result.get(field) or "unknown"), []).append(result)
    return {
        group: {
            "cases": len(results),
            "top_k": {
                str(top_k): aggregate_fact_coverage(
                    [result["metrics"][str(top_k)] for result in results]
                )
                for top_k in top_k_values
            },
        }
        for group, results in sorted(groups.items())
    }


def fact_failure_lists(
    case_results: list[dict[str, Any]], top_k_values: list[int]
) -> dict[str, list[dict[str, Any]]]:
    """Return every not-fully-covered case and its missing atomic facts."""

    failures: dict[str, list[dict[str, Any]]] = {}
    for top_k in top_k_values:
        rows = []
        for result in case_results:
            metrics = result["metrics"][str(top_k)]
            if metrics["fact_coverage_status"] in {"full", "not_scored"}:
                continue
            rows.append(
                {
                    "case_id": result["case_id"],
                    "document_id": result["document_id"],
                    "type": result["type"],
                    "question": result["question"],
                    "status": metrics["fact_coverage_status"],
                    "coverage": metrics["required_fact_coverage"],
                    "matched_facts": metrics["matched_required_facts"],
                    "missing_facts": metrics["missing_required_facts"],
                }
            )
        rows.sort(key=lambda row: (float(row["coverage"]), row["document_id"], row["case_id"]))
        failures[str(top_k)] = rows
    return failures


def evaluate_document(
    document_id: str,
    cases: list[dict[str, Any]],
    chunks: list[Chunk],
    top_k_values: list[int],
    retriever: str = "bm25",
    dense_model_name: str = "BAAI/bge-small-zh-v1.5",
    rrf_k: int = 60,
    dense_model: Any | None = None,
) -> dict[str, Any]:
    index = HybridRetriever(
        chunks,
        mode=retriever,
        dense_model_name=dense_model_name,
        rrf_k=rrf_k,
        dense_model=dense_model,
    )
    max_k = max(top_k_values, default=0)
    case_results: list[dict[str, Any]] = []
    for case in cases:
        ranked = index.retrieve(str(case["question"]), max_k)
        metrics: dict[str, dict[str, Any]] = {}
        for top_k in top_k_values:
            prefix = ranked[:top_k]
            metrics[str(top_k)] = {
                "reference_context_recall": _case_context_recall(case, chunks, prefix),
                "target_document_hit": _case_document_hit(case, chunks, prefix),
                "source_page_hit": _case_page_hit(case, chunks, prefix),
                "table_number_hit": _case_table_hit(case, chunks, prefix),
                **_case_required_fact_coverage(case, chunks, prefix),
            }
        case_results.append(
            {
                "case_id": case["case_id"],
                "document_id": case.get("document_id", document_id),
                "question": case["question"],
                "type": case.get("type", ""),
                "required_facts": case.get("required_facts", []),
                "required_fact_aliases": case.get("required_fact_aliases", {}),
                "top_results": [
                    {
                        "rank": rank,
                        "score": round(result.score, 6),
                        "chunk_index": int(result.key),
                        "page": chunks[int(result.key)].metadata.get("page"),
                        "chunk_type": chunks[int(result.key)].metadata.get("type", "text"),
                        "table_number": chunks[int(result.key)].metadata.get("table_number"),
                    }
                    for rank, result in enumerate(ranked, start=1)
                ],
                "metrics": metrics,
            }
        )

    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "cases": len(case_results),
        "aggregate": aggregate_case_results(case_results, top_k_values),
        "fact_coverage_by_type": aggregate_fact_coverage_by(
            case_results, top_k_values, "type"
        ),
        "cases_detail": case_results,
    }


def _find_pdf(filename: str, papers_dirs: list[Path]) -> Path:
    for directory in papers_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    searched = "、".join(str(directory / filename) for directory in papers_dirs)
    raise FileNotFoundError(f"找不到论文文件：{searched}")


def run_diagnostic(
    manifest_path: str | Path,
    papers_dirs: Iterable[str | Path],
    top_k_values: Iterable[int],
    retriever: str = "bm25",
    dense_model_name: str = "BAAI/bge-small-zh-v1.5",
    rrf_k: int = 60,
) -> dict[str, Any]:
    directories = [Path(directory).resolve() for directory in papers_dirs]
    if not directories:
        raise ValueError("至少需要一个 --papers-dir")
    benchmark = load_benchmark(manifest_path, papers_dir=directories, verify_files=True)
    cases_by_document: dict[str, list[dict[str, Any]]] = {}
    for case in benchmark["cases"]:
        cases_by_document.setdefault(case["document_id"], []).append(case)

    normalized_k = sorted({max(1, int(value)) for value in top_k_values})
    if not normalized_k:
        raise ValueError("至少需要一个 top-k")
    dense_model = None
    if retriever in {"dense", "hybrid"}:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        dense_model = SentenceTransformer(dense_model_name, local_files_only=True)
    documents: list[dict[str, Any]] = []
    all_chunks: list[Chunk] = []
    for document in benchmark["documents"]:
        path = _find_pdf(str(document["filename"]), directories)
        chunks = load_and_split_document(str(path))
        for chunk in chunks:
            chunk.metadata["benchmark_document_id"] = str(document["document_id"])
        all_chunks.extend(chunks)
        documents.append(
            evaluate_document(
                str(document["document_id"]),
                cases_by_document.get(str(document["document_id"]), []),
                chunks,
                normalized_k,
                retriever=retriever,
                dense_model_name=dense_model_name,
                rrf_k=rrf_k,
                dense_model=dense_model,
            )
        )

    global_result = evaluate_document(
        "all-documents",
        benchmark["cases"],
        all_chunks,
        normalized_k,
        retriever=retriever,
        dense_model_name=dense_model_name,
        rrf_k=rrf_k,
        dense_model=dense_model,
    )
    return {
        "schema_version": 2,
        "method": {"bm25": "bm25-lite", "dense": "dense-local", "hybrid": "hybrid-rrf"}[retriever],
        "retriever": retriever,
        "dense_model": dense_model_name if retriever in {"dense", "hybrid"} else None,
        "rrf_k": rrf_k if retriever == "hybrid" else None,
        "manifest": str(Path(manifest_path).resolve()),
        "top_k": normalized_k,
        "documents": documents,
        "overall": global_result["aggregate"],
        "fact_coverage_by_document": aggregate_fact_coverage_by(
            global_result["cases_detail"], normalized_k, "document_id"
        ),
        "fact_coverage_by_type": global_result["fact_coverage_by_type"],
        "fact_failures": fact_failure_lists(global_result["cases_detail"], normalized_k),
        "overall_case_details": global_result["cases_detail"],
        "notes": [
            "overall metrics rank one global index containing all benchmark documents; per-document metrics are an easier diagnostic and are not the multi-paper routing result.",
            "dense-local and hybrid-rrf use only a locally cached Sentence-Transformers model with HF_HUB_OFFLINE=1; a missing cache fails instead of downloading.",
            "hybrid-rrf skips the lexical list when a CJK query has no matching CJK token and fewer than two matching ASCII terms, preventing weak cross-language BM25 ranks from displacing dense evidence.",
            "reference_context_recall compares manually curated English evidence snippets with retrieved chunks by token overlap; it is not answer correctness.",
            "target_document_hit_rate measures whether the target paper enters top-k; the case's document_id is used only for scoring, never added to the query.",
            "source_page_hit_rate uses annotated source_pages and is a page-level diagnostic, not a retrieval gold standard.",
            "table_number_hit_rate is reported only for questions that explicitly name Table N.",
            "required_fact_coverage is deterministic lexical coverage over atomic required_facts; cross-language equivalents are accepted only through case-level required_fact_aliases that are validated against gold contexts.",
            "fact coverage measures whether retrieved target-document context contains annotated facts, not whether a generated answer uses them correctly.",
            "No ChromaDB, Gradio, RAGAS, or external API is used; dense-local and hybrid-rrf do use the locally cached embedding model described above.",
        ],
    }


def _print_summary(report: dict[str, Any], show_failures: bool = False) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"方法：{report['method']}；top-k：{','.join(map(str, report['top_k']))}")
    for document in report["documents"]:
        print(f"\n{document['document_id']}：{document['chunks']} chunks；{document['cases']} cases")
        for top_k, metrics in document["aggregate"].items():
            print(
                f"  @{top_k}: reference_context_recall={fmt(metrics['reference_context_recall'])}; "
                f"target_document_hit_rate={fmt(metrics['target_document_hit_rate'])}; "
                f"source_page_hit_rate={fmt(metrics['source_page_hit_rate'])}; "
                f"table_number_hit_rate={fmt(metrics['table_number_hit_rate'])}; "
                f"fact_macro={fmt(metrics['required_fact_coverage_macro'])}; "
                f"full_fact_cases={fmt(metrics['full_fact_coverage_rate'])}"
            )

    print("\n总体（按用例加权）：")
    for top_k, metrics in report["overall"].items():
        print(
            f"  @{top_k}: reference_context_recall={fmt(metrics['reference_context_recall'])}; "
            f"target_document_hit_rate={fmt(metrics['target_document_hit_rate'])}; "
            f"source_page_hit_rate={fmt(metrics['source_page_hit_rate'])}; "
            f"table_number_hit_rate={fmt(metrics['table_number_hit_rate'])}; "
            f"fact_macro={fmt(metrics['required_fact_coverage_macro'])}; "
            f"fact_micro={fmt(metrics['required_fact_coverage_micro'])}; "
            f"full/partial/zero={fmt(metrics['full_fact_coverage_rate'])}/"
            f"{fmt(metrics['partial_fact_coverage_rate'])}/"
            f"{fmt(metrics['zero_fact_coverage_rate'])}"
        )

    largest_k = str(max(report["top_k"]))
    print(f"\n分题型事实覆盖（@{largest_k}）：")
    for case_type, group in report["fact_coverage_by_type"].items():
        metrics = group["top_k"][largest_k]
        print(
            f"  {case_type}: cases={group['cases']}; "
            f"macro={fmt(metrics['required_fact_coverage_macro'])}; "
            f"full={fmt(metrics['full_fact_coverage_rate'])}"
        )

    failures = report["fact_failures"][largest_k]
    print(f"\n@{largest_k} 未完整覆盖：{len(failures)} / {report['overall'][largest_k]['fact_scored_cases']} 题")
    if show_failures:
        for failure in failures:
            print(
                f"  {failure['case_id']} [{failure['status']}] "
                f"missing={','.join(failure['missing_facts'])}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--papers-dir",
        action="append",
        dest="papers_dirs",
        required=True,
        help="包含清单中文件名的外部论文目录；可重复传入多个目录",
    )
    parser.add_argument(
        "--top-k",
        default="1,3,5,10",
        help="逗号分隔的 k 值，默认 1,3,5,10",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="bm25=词法基线；dense=本地向量；hybrid=BM25+dense 的 RRF",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="dense/hybrid 使用的本地模型名；始终以离线模式加载",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="hybrid 的 RRF 常数，默认 60")
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="打印最大 top-k 下所有未完整覆盖用例及遗漏事实",
    )
    parser.add_argument("--json-out", help="可选：将完整 JSON 诊断写入指定路径")
    args = parser.parse_args()

    try:
        top_k_values = [int(value.strip()) for value in args.top_k.split(",") if value.strip()]
        report = run_diagnostic(
            args.manifest,
            args.papers_dirs,
            top_k_values,
            retriever=args.retriever,
            dense_model_name=args.embedding_model,
            rrf_k=args.rrf_k,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"❌ 基线诊断失败：{exc}", file=sys.stderr)
        return 1

    _print_summary(report, show_failures=args.show_failures)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整 JSON：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
