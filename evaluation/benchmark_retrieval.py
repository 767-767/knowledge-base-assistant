#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline lexical retrieval diagnostics for the multi-paper benchmark.

This module deliberately uses only the standard library for ranking.  PDF
parsing is delegated to the same side-effect-free ingestion path used by the
application, but no ChromaDB, embedding model, UI, or external API is loaded.
The report is a retrieval diagnostic, not an answer-quality or RAGAS score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import load_and_split_document  # noqa: E402
from evaluation.benchmark_loader import DEFAULT_MANIFEST, load_benchmark  # noqa: E402
from sci_rag_core import Chunk, normalize_for_match, table_number_from_question  # noqa: E402


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*|[\u4e00-\u9fff]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*")


def tokenize(value: Any) -> list[str]:
    """Tokenize English identifiers, numbers, and individual CJK characters."""

    return [token.casefold() for token in TOKEN_RE.findall(normalize_for_match(value))]


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


@dataclass(frozen=True)
class RankedChunk:
    index: int
    score: float


class BM25Index:
    """Small deterministic BM25 implementation for offline comparisons."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(searchable_text(chunk)) for chunk in self.chunks]
        self._term_frequency = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    def _idf(self, token: str) -> float:
        documents = len(self._tokens)
        frequency = self._document_frequency.get(token, 0)
        if not documents or not frequency:
            return 0.0
        return math.log(1.0 + (documents - frequency + 0.5) / (frequency + 0.5))

    def score(self, question: str, index: int) -> float:
        if index < 0 or index >= len(self._tokens) or not self._average_length:
            return 0.0
        frequencies = self._term_frequency[index]
        length = len(self._tokens[index])
        score = 0.0
        for token in set(tokenize(question)):
            term_frequency = frequencies.get(token, 0)
            if not term_frequency:
                continue
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * length / self._average_length
            )
            score += self._idf(token) * term_frequency * (self.k1 + 1.0) / denominator
        return score

    def retrieve(self, question: str, k: int = 10) -> list[RankedChunk]:
        """Return stable score-descending results, preserving source order on ties."""

        limit = max(0, min(int(k), len(self.chunks)))
        ranked = [RankedChunk(index, self.score(question, index)) for index in range(len(self.chunks))]
        ranked.sort(key=lambda item: (-item.score, item.index))
        return ranked[:limit]


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

    def retrieve(self, question: str, k: int = 10) -> list[RankedChunk]:
        limit = max(0, min(int(k), len(self.chunks)))
        if not self.chunks or not limit:
            return []
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_embedding
        ranked = [RankedChunk(index, float(scores[index])) for index in range(len(self.chunks))]
        ranked.sort(key=lambda item: (-item.score, item.index))
        return ranked[:limit]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[int | RankedChunk]],
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[RankedChunk]:
    """Fuse ranked index lists while deduplicating each list and preserving ties."""

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须为正整数")
    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    seen_order = 0
    for ranking in rankings:
        list_seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            index = item.index if isinstance(item, RankedChunk) else int(item)
            if index in list_seen:
                continue
            list_seen.add(index)
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
            if index not in first_seen:
                first_seen[index] = seen_order
                seen_order += 1
    fused = [RankedChunk(index, score) for index, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, first_seen[item.index]))
    return fused if limit is None else fused[: max(0, int(limit))]


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
        self.bm25 = BM25Index(self.chunks)
        self.dense = (
            DenseIndex(self.chunks, dense_model_name, model=dense_model)
            if mode in {"dense", "hybrid"}
            else None
        )

    def retrieve(self, question: str, k: int = 10) -> list[RankedChunk]:
        if self.mode == "bm25":
            return self.bm25.retrieve(question, k)
        if self.mode == "dense":
            assert self.dense is not None
            return self.dense.retrieve(question, k)
        candidate_k = min(len(self.chunks), max(int(k) * 5, 50))
        assert self.dense is not None
        return reciprocal_rank_fusion(
            [self.bm25.retrieve(question, candidate_k), self.dense.retrieve(question, candidate_k)],
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
    case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedChunk]
) -> list[Chunk]:
    """Keep evidence metrics scoped to the case's target document.

    Ranking may be global across all papers, but a similarly worded chunk from
    another paper must not count as evidence for this case.  The optional
    ``benchmark_document_id`` is attached by :func:`run_diagnostic`.
    """

    target_document = case.get("document_id")
    selected = [chunks[result.index] for result in ranked]
    if not target_document:
        return selected
    marked = [
        chunk
        for chunk in selected
        if chunk.metadata.get("benchmark_document_id") is None
        or chunk.metadata.get("benchmark_document_id") == target_document
    ]
    return marked


def _case_context_recall(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedChunk]) -> float:
    references = [str(context) for context in case.get("contexts", [])]
    if not references:
        return 0.0
    retrieved = [chunk.page_content for chunk in _target_ranked_chunks(case, chunks, ranked)]
    matched = sum(
        any(_reference_context_match(reference, text) for text in retrieved)
        for reference in references
    )
    return matched / len(references)


def _case_page_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedChunk]) -> bool | None:
    source_pages = {int(page) for page in case.get("source_pages", []) if str(page).isdigit()}
    if not source_pages:
        return None
    return any(chunk.metadata.get("page") in source_pages for chunk in _target_ranked_chunks(case, chunks, ranked))


def _case_document_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedChunk]) -> bool | None:
    target_document = case.get("document_id")
    if not target_document:
        return None
    return any(
        chunks[result.index].metadata.get("benchmark_document_id") == target_document
        for result in ranked
    )


def _case_table_hit(case: dict[str, Any], chunks: list[Chunk], ranked: list[RankedChunk]) -> bool | None:
    table_number = table_number_from_question(str(case.get("question", "")))
    if table_number is None:
        return None
    target = int(table_number)
    return any(
        chunk.metadata.get("type") == "table" and chunk.metadata.get("table_number") == target
        for chunk in _target_ranked_chunks(case, chunks, ranked)
    )


def _metric_mean(values: list[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def aggregate_case_results(case_results: list[dict[str, Any]], top_k_values: list[int]) -> dict[str, dict[str, float | None]]:
    """Aggregate case-level diagnostics without weighting documents equally."""

    aggregate: dict[str, dict[str, float | None]] = {}
    for top_k in top_k_values:
        rows = [result["metrics"][str(top_k)] for result in case_results]
        aggregate[str(top_k)] = {
            "reference_context_recall": _metric_mean([row["reference_context_recall"] for row in rows]),
            "target_document_hit_rate": _metric_mean([row["target_document_hit"] for row in rows]),
            "source_page_hit_rate": _metric_mean([row["source_page_hit"] for row in rows]),
            "table_number_hit_rate": _metric_mean(
                [row["table_number_hit"] for row in rows if row["table_number_hit"] is not None]
            ),
        }
    return aggregate


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
        metrics: dict[str, dict[str, float | bool | None]] = {}
        for top_k in top_k_values:
            prefix = ranked[:top_k]
            metrics[str(top_k)] = {
                "reference_context_recall": _case_context_recall(case, chunks, prefix),
                "target_document_hit": _case_document_hit(case, chunks, prefix),
                "source_page_hit": _case_page_hit(case, chunks, prefix),
                "table_number_hit": _case_table_hit(case, chunks, prefix),
            }
        case_results.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "type": case.get("type", ""),
                "top_results": [
                    {
                        "rank": rank,
                        "score": round(result.score, 6),
                        "chunk_index": result.index,
                        "page": chunks[result.index].metadata.get("page"),
                        "chunk_type": chunks[result.index].metadata.get("type", "text"),
                        "table_number": chunks[result.index].metadata.get("table_number"),
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
        "schema_version": 1,
        "method": {"bm25": "bm25-lite", "dense": "dense-local", "hybrid": "hybrid-rrf"}[retriever],
        "retriever": retriever,
        "dense_model": dense_model_name if retriever in {"dense", "hybrid"} else None,
        "rrf_k": rrf_k if retriever == "hybrid" else None,
        "manifest": str(Path(manifest_path).resolve()),
        "top_k": normalized_k,
        "documents": documents,
        "overall": global_result["aggregate"],
        "overall_case_details": global_result["cases_detail"],
        "notes": [
            "overall metrics rank one global index containing all benchmark documents; per-document metrics are an easier diagnostic and are not the multi-paper routing result.",
            "dense-local and hybrid-rrf use only a locally cached Sentence-Transformers model with HF_HUB_OFFLINE=1; a missing cache fails instead of downloading.",
            "reference_context_recall compares manually curated English evidence snippets with retrieved chunks by token overlap; it is not answer correctness.",
            "target_document_hit_rate measures whether the target paper enters top-k; the case's document_id is used only for scoring, never added to the query.",
            "source_page_hit_rate uses annotated source_pages and is a page-level diagnostic, not a retrieval gold standard.",
            "table_number_hit_rate is reported only for questions that explicitly name Table N.",
            "No ChromaDB, embedding model, Gradio, RAGAS, or external API is used.",
        ],
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"方法：{report['method']}；top-k：{','.join(map(str, report['top_k']))}")
    for document in report["documents"]:
        print(f"\n{document['document_id']}：{document['chunks']} chunks；{document['cases']} cases")
        for top_k, metrics in document["aggregate"].items():
            def fmt(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.3f}"

            print(
                f"  @{top_k}: reference_context_recall={fmt(metrics['reference_context_recall'])}; "
                f"target_document_hit_rate={fmt(metrics['target_document_hit_rate'])}; "
                f"source_page_hit_rate={fmt(metrics['source_page_hit_rate'])}; "
                f"table_number_hit_rate={fmt(metrics['table_number_hit_rate'])}"
            )

    print("\n总体（按用例加权）：")
    for top_k, metrics in report["overall"].items():
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        print(
            f"  @{top_k}: reference_context_recall={fmt(metrics['reference_context_recall'])}; "
            f"target_document_hit_rate={fmt(metrics['target_document_hit_rate'])}; "
            f"source_page_hit_rate={fmt(metrics['source_page_hit_rate'])}; "
            f"table_number_hit_rate={fmt(metrics['table_number_hit_rate'])}"
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

    _print_summary(report)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整 JSON：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
