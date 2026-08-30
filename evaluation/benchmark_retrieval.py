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
import resource
import statistics
import sys
from time import perf_counter
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    _header_match_score,
    _is_composite_fact_question,
    _section_query_terms,
    load_and_split_document,
)
from evaluation.benchmark_loader import DEFAULT_MANIFEST, load_benchmark  # noqa: E402
from evaluation.context_coverage import (  # noqa: E402
    aggregate_fact_coverage,
    case_fact_coverage,
)
from sci_rag_core import Chunk, normalize_for_match, table_number_from_question  # noqa: E402
from sci_rag_reranking import CrossEncoderReranker, reranker_document_text  # noqa: E402
from sci_rag_retrieval import BM25Index, RankedItem, reciprocal_rank_fusion  # noqa: E402


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*")
MAX_SECTION_EXPANSION_CHUNKS = 6


def evidence_tokens(value: Any) -> list[str]:
    """Return stable ASCII tokens for comparing English PDF evidence snippets."""

    return [token.casefold() for token in ASCII_TOKEN_RE.findall(normalize_for_match(value))]


def searchable_text(chunk: Chunk) -> str:
    return reranker_document_text(chunk.page_content, chunk.metadata)


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


def _section_expansion_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    anchor_ranked: list[RankedItem] | None = None,
) -> list[int]:
    """Return application-equivalent same-section additions for a query.

    The application expands only sections already represented by the retrieved
    context, only for composite questions, and only when the corpus is a single
    source. This benchmark helper mirrors that bounded rule over the in-memory
    parsed corpus so the effect can be measured without touching ChromaDB. It
    returns additions first, followed by the original ranking with duplicates
    removed.
    """

    if not _is_composite_fact_question(question):
        return [int(item.key) for item in ranked]
    anchor_items = anchor_ranked if anchor_ranked is not None else ranked
    anchor_sources = {
        str(chunks[int(item.key)].metadata.get("source", ""))
        for item in anchor_items
        if 0 <= int(item.key) < len(chunks)
        and chunks[int(item.key)].metadata.get("source")
    }
    if len(anchor_sources) > 1:
        return [int(item.key) for item in ranked]
    base_source = ""
    for item in anchor_items:
        index = int(item.key)
        if 0 <= index < len(chunks):
            base_source = str(chunks[index].metadata.get("source", ""))
            if base_source:
                break
    if not base_source:
        return [int(item.key) for item in ranked]
    corpus_sources = {
        str(chunk.metadata.get("source", ""))
        for chunk in chunks
        if chunk.metadata.get("source")
    }
    if len(corpus_sources) > 1:
        return [int(item.key) for item in ranked]
    query_terms = _section_query_terms(question)
    if not query_terms:
        return [int(item.key) for item in ranked]

    header_rows: list[tuple[int, int, str, str]] = []
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", ""))
        header = str(chunk.metadata.get("headers", ""))
        if source != base_source or not header:
            continue
        score = _header_match_score(header, query_terms)
        if score:
            header_rows.append((score, index, source, header))
    if not header_rows:
        return [int(item.key) for item in ranked]

    best_score = max(row[0] for row in header_rows)
    selected_headers = {(row[2], row[3]) for row in header_rows if row[0] == best_score}
    original = [int(item.key) for item in ranked]
    anchors = [
        index
        for item in anchor_items
        for index in [int(item.key)]
        if 0 <= index < len(chunks)
        and (
            str(chunks[index].metadata.get("source", "")),
            str(chunks[index].metadata.get("headers", "")),
        ) in selected_headers
    ]
    if not anchors:
        return original
    expanded = [
        index
        for index, chunk in enumerate(chunks)
        if (
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("headers", "")),
        ) in selected_headers
    ]
    expanded.sort(
        key=lambda index: (
            min((abs(index - anchor) for anchor in anchors), default=index),
            index,
        )
    )
    ordered: list[int] = []
    seen: set[int] = set()
    for index in [*expanded[:MAX_SECTION_EXPANSION_CHUNKS], *original]:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    return ordered


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


def _latency_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    usable = sorted(float(value) for value in values)
    if not usable:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    p95_index = max(0, min(len(usable) - 1, int(len(usable) * 0.95 + 0.999999) - 1))
    return {
        "count": len(usable),
        "mean": sum(usable) / len(usable),
        "median": statistics.median(usable),
        "p95": usable[p95_index],
        "max": usable[-1],
    }


def aggregate_case_latency(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval and optional reranker wall-clock latency."""

    return {
        field: _latency_statistics(
            result.get("timing", {}).get(field, 0.0) for result in case_results
        )
        for field in ("retrieval_seconds", "rerank_seconds", "total_seconds")
    }


def _process_peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def evaluate_document(
    document_id: str,
    cases: list[dict[str, Any]],
    chunks: list[Chunk],
    top_k_values: list[int],
    retriever: str = "bm25",
    dense_model_name: str = "BAAI/bge-small-zh-v1.5",
    rrf_k: int = 60,
    dense_model: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    reranker_candidate_k: int = 50,
    reranker_fusion: str = "none",
    reranker_fusion_rrf_k: int = 60,
    reranker_fusion_ce_weight: float = 1.0,
    section_expansion: bool = False,
) -> dict[str, Any]:
    if reranker_fusion not in {"none", "rrf", "weighted_rrf"}:
        raise ValueError(f"不支持的 reranker fusion：{reranker_fusion}")
    if not reranker_fusion_ce_weight > 0:
        raise ValueError("reranker fusion 的 CE 权重必须为正数")
    index = HybridRetriever(
        chunks,
        mode=retriever,
        dense_model_name=dense_model_name,
        rrf_k=rrf_k,
        dense_model=dense_model,
    )
    max_k = max(top_k_values, default=0)
    retrieval_k = max(max_k, int(reranker_candidate_k)) if reranker else max_k
    rerank_documents = [searchable_text(chunk) for chunk in chunks] if reranker else []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        total_started = perf_counter()
        retrieval_started = perf_counter()
        candidates = index.retrieve(question, retrieval_k)
        retrieval_seconds = perf_counter() - retrieval_started
        initial_scores = {candidate.key: candidate.score for candidate in candidates}
        reranker_scores: dict[Any, float] = {}
        rerank_seconds = 0.0
        scored_pairs = 0
        cache_hits = 0
        if reranker:
            rerank_result = reranker.rerank(question, candidates, rerank_documents)
            reranker_scores = {item.key: item.score for item in rerank_result.ranked}
            fusion_limit = retrieval_k if section_expansion else max_k
            if reranker_fusion == "rrf":
                ranked = reciprocal_rank_fusion(
                    [rerank_result.ranked, candidates],
                    rrf_k=reranker_fusion_rrf_k,
                    limit=fusion_limit,
                )
            elif reranker_fusion == "weighted_rrf":
                ranked = reciprocal_rank_fusion(
                    [rerank_result.ranked, candidates],
                    rrf_k=reranker_fusion_rrf_k,
                    limit=fusion_limit,
                    weights=[reranker_fusion_ce_weight, 1.0],
                )
            else:
                ranked = rerank_result.ranked[:fusion_limit]
            rerank_seconds = rerank_result.elapsed_seconds
            scored_pairs = rerank_result.scored_pairs
            cache_hits = rerank_result.cache_hits
        else:
            ranked = candidates[:max_k]
        if section_expansion:
            ranked = [
                RankedItem(index, 0.0)
                for index in _section_expansion_indices(
                    question,
                    ranked,
                    chunks,
                    anchor_ranked=ranked[:max_k],
                )[:max_k]
            ]
        else:
            ranked = ranked[:max_k]
        total_seconds = perf_counter() - total_started
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
                        "retrieval_score": (
                            round(initial_scores[result.key], 6)
                            if result.key in initial_scores
                            else None
                        ),
                        "rerank_score": (
                            round(reranker_scores[result.key], 6)
                            if reranker and result.key in reranker_scores
                            else None
                        ),
                        "chunk_index": int(result.key),
                        "page": chunks[int(result.key)].metadata.get("page"),
                        "chunk_type": chunks[int(result.key)].metadata.get("type", "text"),
                        "table_number": chunks[int(result.key)].metadata.get("table_number"),
                    }
                    for rank, result in enumerate(ranked, start=1)
                ],
                "candidate_count": len(candidates),
                "timing": {
                    "retrieval_seconds": retrieval_seconds,
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": total_seconds,
                    "reranker_scored_pairs": scored_pairs,
                    "reranker_cache_hits": cache_hits,
                },
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
        "latency": aggregate_case_latency(case_results),
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
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    reranker_candidate_k: int = 50,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 512,
    reranker_device: str = "cpu",
    reranker_fusion: str = "none",
    reranker_fusion_rrf_k: int = 60,
    reranker_fusion_ce_weight: float = 1.0,
    section_expansion: bool = False,
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
    if reranker_model and retriever != "hybrid":
        raise ValueError("cross-encoder 实验必须基于 --retriever hybrid")
    if reranker_candidate_k < max(normalized_k):
        raise ValueError("reranker candidate-k 不能小于最大 top-k")
    if reranker_fusion not in {"none", "rrf", "weighted_rrf"}:
        raise ValueError(f"不支持的 reranker fusion：{reranker_fusion}")
    if not reranker_fusion_ce_weight > 0:
        raise ValueError("reranker fusion 的 CE 权重必须为正数")
    dense_model = None
    if retriever in {"dense", "hybrid"}:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        dense_model = SentenceTransformer(dense_model_name, local_files_only=True)
    reranker = None
    reranker_load_seconds = None
    if reranker_model:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        load_started = perf_counter()
        reranker = CrossEncoderReranker(
            reranker_model,
            revision=reranker_revision,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            device=reranker_device,
            local_files_only=True,
            cache_scores=True,
        )
        reranker_load_seconds = perf_counter() - load_started
    parsed_documents: list[tuple[dict[str, Any], list[Chunk]]] = []
    all_chunks: list[Chunk] = []
    for document in benchmark["documents"]:
        path = _find_pdf(str(document["filename"]), directories)
        chunks = load_and_split_document(str(path))
        for chunk in chunks:
            chunk.metadata["benchmark_document_id"] = str(document["document_id"])
        all_chunks.extend(chunks)
        parsed_documents.append((document, chunks))

    # Evaluate the global multi-paper task first so latency is not reduced by
    # the benchmark-only score cache used by the easier per-document reports.
    global_result = evaluate_document(
        "all-documents",
        benchmark["cases"],
        all_chunks,
        normalized_k,
        retriever=retriever,
        dense_model_name=dense_model_name,
        rrf_k=rrf_k,
        dense_model=dense_model,
        reranker=reranker,
        reranker_candidate_k=reranker_candidate_k,
        reranker_fusion=reranker_fusion,
        reranker_fusion_rrf_k=reranker_fusion_rrf_k,
        reranker_fusion_ce_weight=reranker_fusion_ce_weight,
        section_expansion=section_expansion,
    )
    documents: list[dict[str, Any]] = []
    for document, chunks in parsed_documents:
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
                reranker=reranker,
                reranker_candidate_k=reranker_candidate_k,
                reranker_fusion=reranker_fusion,
                reranker_fusion_rrf_k=reranker_fusion_rrf_k,
                reranker_fusion_ce_weight=reranker_fusion_ce_weight,
                section_expansion=section_expansion,
            )
        )
    method = {"bm25": "bm25-lite", "dense": "dense-local", "hybrid": "hybrid-rrf"}[retriever]
    if reranker:
        method += "+cross-encoder"
    return {
        "schema_version": 3,
        "method": method,
        "retriever": retriever,
        "dense_model": dense_model_name if retriever in {"dense", "hybrid"} else None,
        "rrf_k": rrf_k if retriever == "hybrid" else None,
        "reranker": (
            {
                "model": reranker_model,
                "revision": reranker_revision,
                "candidate_k": reranker_candidate_k,
                "batch_size": reranker_batch_size,
                "max_length": reranker_max_length,
                "device": reranker.device,
                "fusion": reranker_fusion,
                "fusion_rrf_k": reranker_fusion_rrf_k if reranker_fusion == "rrf" else None,
                "fusion_ce_weight": (
                    reranker_fusion_ce_weight if reranker_fusion == "weighted_rrf" else None
                ),
                "section_expansion": section_expansion,
                "load_seconds": reranker_load_seconds,
            }
            if reranker
            else None
        ),
        "manifest": str(Path(manifest_path).resolve()),
        "top_k": normalized_k,
        "documents": documents,
        "overall": global_result["aggregate"],
        "fact_coverage_by_document": aggregate_fact_coverage_by(
            global_result["cases_detail"], normalized_k, "document_id"
        ),
        "fact_coverage_by_type": global_result["fact_coverage_by_type"],
        "fact_failures": fact_failure_lists(global_result["cases_detail"], normalized_k),
        "latency": global_result["latency"],
        "process_peak_rss_mb": _process_peak_rss_mb(),
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
            "cross-encoder reranking is opt-in, local-files-only, and applied only to the configured Hybrid candidate pool; weighted_rrf scales the CE list against the original candidate list while preserving the default equal-weight rrf path; report latency is measured on the global multi-paper run before easier per-document diagnostics.",
            "section_expansion is an opt-in benchmark control that mirrors the application's bounded same-section expansion for composite questions; it is disabled unless --section-expansion is passed.",
            "No ChromaDB, Gradio, RAGAS, or external API is used; dense-local and hybrid-rrf do use the locally cached embedding model described above.",
        ],
    }


def _print_summary(report: dict[str, Any], show_failures: bool = False) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"方法：{report['method']}；top-k：{','.join(map(str, report['top_k']))}")
    if report.get("reranker"):
        config = report["reranker"]
        print(
            f"reranker：{config['model']}@{config['revision'] or 'default'}；"
            f"candidate_k={config['candidate_k']}；batch={config['batch_size']}；"
            f"max_length={config['max_length']}；device={config['device']}；"
            f"fusion={config['fusion']}；"
            f"load={fmt(config['load_seconds'])}s"
        )
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
    if report.get("reranker"):
        rerank_latency = report["latency"]["rerank_seconds"]
        print(
            "\n全局 reranker 单题延迟："
            f"mean={fmt(rerank_latency['mean'])}s；"
            f"median={fmt(rerank_latency['median'])}s；"
            f"p95={fmt(rerank_latency['p95'])}s；"
            f"max={fmt(rerank_latency['max'])}s；"
            f"peak_rss={report['process_peak_rss_mb']:.1f} MB"
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
    parser.add_argument("--reranker-model", help="可选：本地已缓存的 cross-encoder 模型或路径")
    parser.add_argument("--reranker-revision", help="可选：固定模型 commit/revision")
    parser.add_argument("--reranker-candidate-k", type=int, default=50, help="重排候选数，默认 50")
    parser.add_argument("--reranker-batch-size", type=int, default=8, help="重排 batch size，默认 8")
    parser.add_argument("--reranker-max-length", type=int, default=512, help="query-passage 最大 token，默认 512")
    parser.add_argument("--reranker-device", default="cpu", help="重排设备，默认 cpu")
    parser.add_argument(
        "--reranker-fusion",
        choices=("none", "rrf", "weighted_rrf"),
        default="none",
        help="cross-encoder 与原候选排名的融合：none、等权 rrf 或加权 weighted_rrf；默认 none",
    )
    parser.add_argument(
        "--reranker-fusion-rrf-k",
        type=int,
        default=60,
        help="reranker fusion 的 RRF 常数，默认 60",
    )
    parser.add_argument(
        "--reranker-fusion-ce-weight",
        type=float,
        default=1.0,
        help="weighted_rrf 中 cross-encoder 列表的权重，原候选列表固定为 1；默认 1",
    )
    parser.add_argument(
        "--section-expansion",
        action="store_true",
        help="对复合问题加入应用同小节扩展的离线对照；默认关闭",
    )
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
            reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision,
            reranker_candidate_k=args.reranker_candidate_k,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            reranker_device=args.reranker_device,
            reranker_fusion=args.reranker_fusion,
            reranker_fusion_rrf_k=args.reranker_fusion_rrf_k,
            reranker_fusion_ce_weight=args.reranker_fusion_ce_weight,
            section_expansion=args.section_expansion,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
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
