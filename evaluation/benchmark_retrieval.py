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
    _section_continuation_indices,
    _section_query_terms,
    load_and_split_document,
)
from evaluation.benchmark_loader import DEFAULT_MANIFEST, load_benchmark  # noqa: E402
from evaluation.context_coverage import (  # noqa: E402
    aggregate_fact_coverage,
    aliases_for_fact,
    case_fact_coverage,
    fact_is_present,
)
from sci_rag_core import (  # noqa: E402
    Chunk,
    formula_evidence_indices,
    find_table_cell_in_chunks,
    figure_reference_from_metadata,
    figure_reference_from_question,
    limitation_evidence_indices,
    is_limitation_question,
    is_table_question,
    normalize_for_match,
    rerank_table_first,
    table_number_from_question,
)
from sci_rag_reranking import CrossEncoderReranker, reranker_document_text  # noqa: E402
from sci_rag_retrieval import (  # noqa: E402
    BM25Index,
    DocumentRoute,
    DocumentRouter,
    RankedItem,
    query_variants,
    reciprocal_rank_fusion,
)


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*")
REFERENCE_HEADER_RE = re.compile(r"(?:references?|bibliography|参考文献)", re.IGNORECASE)
PICTURE_TEXT_MARKER_RE = re.compile(
    r"<!--\s*(?:start|end) of picture text\s*-->|<img\b",
    re.IGNORECASE,
)
MAX_SECTION_EXPANSION_CHUNKS = 6
MAX_ADJACENT_ANCHORS = 2


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
        indices: Iterable[int] | None = None,
    ):
        # Never allow a benchmark run to turn into an implicit model download.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.chunks = list(chunks)
        self.model_name = model_name
        self.indices = (
            list(range(len(self.chunks)))
            if indices is None
            else [int(index) for index in indices if 0 <= int(index) < len(self.chunks)]
        )
        self._embedding_positions = {
            index: position for position, index in enumerate(self.indices)
        }
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, local_files_only=True)
        self.model = model
        self.embeddings = self.model.encode(
            [searchable_text(self.chunks[index]) for index in self.indices],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        question: str,
        k: int = 10,
        indices: Iterable[int] | None = None,
    ) -> list[RankedItem]:
        candidate_indices = (
            list(self.indices)
            if indices is None
            else [
                int(index)
                for index in indices
                if int(index) in self._embedding_positions
            ]
        )
        limit = max(0, min(int(k), len(candidate_indices)))
        if not candidate_indices or not limit:
            return []
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        positions = [self._embedding_positions[index] for index in candidate_indices]
        scores = self.embeddings[positions] @ query_embedding
        ranked = [
            RankedItem(index, float(score))
            for index, score in zip(candidate_indices, scores)
        ]
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
        document_routing: bool = False,
        query_decomposition: bool = False,
        excluded_chunk_types: Iterable[str] = (),
    ):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"不支持的 retriever：{mode}")
        self.chunks = list(chunks)
        self.mode = mode
        self.rrf_k = rrf_k
        self.query_decomposition = bool(query_decomposition)
        self.excluded_chunk_types = {
            str(value).casefold() for value in excluded_chunk_types
        }
        self._retrieval_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if str(chunk.metadata.get("type", "text")).casefold()
            not in self.excluded_chunk_types
        ]
        self._retrieval_positions = {
            index: position for position, index in enumerate(self._retrieval_indices)
        }
        self.bm25 = BM25Index(
            searchable_text(self.chunks[index]) for index in self._retrieval_indices
        )
        self.document_routing = bool(document_routing)
        self.document_router: DocumentRouter | None = None
        self._document_indices: dict[str, list[int]] = {}
        if self.document_routing:
            profiles: dict[str, list[str]] = {}
            for index in self._retrieval_indices:
                chunk = self.chunks[index]
                metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                document_id = str(
                    metadata.get("benchmark_document_id")
                    or metadata.get("source")
                    or "unknown"
                )
                self._document_indices.setdefault(document_id, []).append(index)
                profiles.setdefault(document_id, []).append(searchable_text(chunk))
            document_ids = sorted(profiles)
            self.document_router = DocumentRouter(
                document_ids,
                ("\n".join(profiles[document_id]) for document_id in document_ids),
            )
        self.dense = (
            DenseIndex(
                self.chunks,
                dense_model_name,
                model=dense_model,
                indices=self._retrieval_indices,
            )
            if mode in {"dense", "hybrid"}
            else None
        )

    def route(self, question: str) -> DocumentRoute | None:
        """Return the optional source route chosen without benchmark labels."""

        return (
            self.document_router.route(question)
            if self.document_router is not None
            else None
        )

    def _bm25_retrieve(
        self,
        question: str,
        k: int,
        indices: Iterable[int] | None = None,
    ) -> list[RankedItem]:
        """Search the same eligible corpus used by dense retrieval."""

        candidate_indices = (
            self._retrieval_indices
            if indices is None
            else [int(index) for index in indices if int(index) in self._retrieval_positions]
        )
        positions = [self._retrieval_positions[index] for index in candidate_indices]
        return [
            RankedItem(self._retrieval_indices[int(item.key)], item.score)
            for item in self.bm25.retrieve(question, k, indices=positions)
        ]

    def _retrieve_single(
        self,
        question: str,
        k: int,
        eligible_indices: Iterable[int] | None = None,
    ) -> list[RankedItem]:
        """Retrieve one query variant within a fixed source scope."""

        if self.mode == "bm25":
            return self._bm25_retrieve(question, k, indices=eligible_indices)
        if self.mode == "dense":
            assert self.dense is not None
            return self.dense.retrieve(question, k, indices=eligible_indices)
        candidate_k = min(len(self._retrieval_indices), max(int(k) * 5, 50))
        assert self.dense is not None
        lexical = (
            self._bm25_retrieve(question, candidate_k, indices=eligible_indices)
            if self.bm25.has_lexical_signal(question)
            else []
        )
        return reciprocal_rank_fusion(
            [lexical, self.dense.retrieve(question, candidate_k, indices=eligible_indices)],
            rrf_k=self.rrf_k,
            limit=k,
        )

    def retrieve(self, question: str, k: int = 10) -> list[RankedItem]:
        route = self.route(question)
        eligible_indices = (
            self._document_indices.get(route.document_id, [])
            if route is not None
            else self._retrieval_indices
        )
        if not self.query_decomposition:
            return self._retrieve_single(question, k, eligible_indices)
        variants = query_variants(question)
        if len(variants) <= 1:
            return self._retrieve_single(question, k, eligible_indices)
        rankings = [
            self._retrieve_single(variant, k, eligible_indices)
            for variant in variants
        ]
        return reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k, limit=k)


def _section_expansion_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    anchor_ranked: list[RankedItem] | None = None,
    route_source: str | None = None,
) -> list[int]:
    """Return application-equivalent same-section additions for a query.

    The application expands only sections already represented by the retrieved
    context, only for composite/list questions, and only within one source. A
    multi-source corpus is allowed when the conservative document router has
    selected a unique ``route_source``; otherwise expansion is skipped. This
    benchmark helper mirrors that bounded rule over the in-memory parsed corpus
    so the effect can be measured without touching ChromaDB. It returns
    additions first, followed by the original ranking with duplicates removed.
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
    route_source = str(route_source or "").strip()
    if len(corpus_sources) > 1 and not route_source:
        return [int(item.key) for item in ranked]
    if route_source and route_source not in corpus_sources:
        return [int(item.key) for item in ranked]
    if route_source:
        base_source = route_source
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
    continuation_indices: list[int] = []
    for anchor in anchors:
        continuation_indices.extend(
            _section_continuation_indices(
                anchor,
                [chunk.metadata for chunk in chunks],
                [chunk.page_content for chunk in chunks],
                base_source,
                MAX_SECTION_EXPANSION_CHUNKS,
            )
        )
    ordered: list[int] = []
    seen: set[int] = set()
    for index in [
        *expanded[:MAX_SECTION_EXPANSION_CHUNKS],
        *continuation_indices,
        *original,
    ]:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    return ordered


def _structured_table_guard_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    route: DocumentRoute | None = None,
) -> list[int]:
    """Return the application-equivalent context order for table questions.

    The web application scans canonical table chunks after normal retrieval and
    performs an exact row/column lookup when possible.  This opt-in benchmark
    control mirrors that behavior without changing the raw retriever metrics.
    If document routing selected a source, the table scan remains in that same
    scope so a same-numbered table from another paper cannot answer first.
    """

    original = [int(item.key) for item in ranked]
    if not is_table_question(question):
        return original
    route_document = route.document_id if route is not None else None
    table_indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.metadata.get("type") == "table"
        and (
            route_document is None
            or str(chunk.metadata.get("benchmark_document_id", "")) == route_document
        )
    ]
    if not table_indices:
        return original

    table_texts = [chunks[index].page_content for index in table_indices]
    table_metas = [chunks[index].metadata for index in table_indices]
    cell_match = find_table_cell_in_chunks(question, table_texts, table_metas)
    if cell_match is not None:
        table_position, _ = cell_match
        return [table_indices[table_position]]

    combined = list(dict.fromkeys([*original, *table_indices]))
    combined_texts = [chunks[index].page_content for index in combined]
    combined_metas = [chunks[index].metadata for index in combined]
    order, _, _ = rerank_table_first(question, combined_texts, combined_metas)
    return [combined[position] for position in order]


def _structured_figure_guard_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    route: DocumentRoute | None = None,
) -> list[int]:
    """Put exact Figure N spatial evidence first inside the current route.

    This mirrors the application's opt-in collection scan. Non-matching figure
    chunks are removed for an explicit figure query so text from Figure 2 or
    Extended Data Figure 1 cannot be presented as Figure 1 evidence.
    """

    original = [int(item.key) for item in ranked]
    reference = figure_reference_from_question(question)
    if reference is None:
        return original
    route_document = route.document_id if route is not None else None
    matches = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.metadata.get("type") == "figure"
        and figure_reference_from_metadata(chunk.metadata) == reference
        and (
            route_document is None
            or str(chunk.metadata.get("benchmark_document_id", "")) == route_document
        )
    ]
    if not matches:
        return [
            index for index in original
            if chunks[index].metadata.get("type") != "figure"
        ]
    match_set = set(matches)
    return matches + [
        index
        for index in original
        if index not in match_set
        and chunks[index].metadata.get("type") != "figure"
        and not PICTURE_TEXT_MARKER_RE.search(chunks[index].page_content)
    ]


def _formula_guard_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    route: DocumentRoute | None = None,
) -> list[int]:
    """Promote a few formula-bearing chunks for explicit formula questions."""

    original = [int(item.key) for item in ranked]
    metas = [chunk.metadata for chunk in chunks]
    source_ids = {
        str(metadata.get("benchmark_document_id") or metadata.get("source") or "").strip()
        for metadata in metas
        if str(metadata.get("benchmark_document_id") or metadata.get("source") or "").strip()
    }
    allowed_formula_indices: list[int] | None
    if route is not None:
        allowed_formula_indices = [
            index
            for index in range(len(chunks))
            if str(metas[index].get("benchmark_document_id", "")) == route.document_id
            or str(metas[index].get("source", "")) == route.document_id
        ]
    elif len(source_ids) == 1:
        # Per-document diagnostics and a single-paper corpus are safe scopes.
        allowed_formula_indices = list(range(len(chunks)))
    else:
        # An unqualified multi-paper query must not receive formula text from
        # an arbitrary source merely because it contains equation markers.
        allowed_formula_indices = []
    formula_indices = formula_evidence_indices(
        question,
        [chunk.page_content for chunk in chunks],
        metas,
        max_results=8,
        allowed_indices=allowed_formula_indices,
    )
    return list(dict.fromkeys([*formula_indices, *original]))


def _limitation_guard_indices(
    question: str,
    ranked: list[RankedItem],
    chunks: list[Chunk],
    route: DocumentRoute | None = None,
) -> list[int]:
    """Promote literal limitation passages for explicit limitation questions."""

    original = [int(item.key) for item in ranked]
    if not is_limitation_question(question):
        return original
    metas = [chunk.metadata for chunk in chunks]
    source_ids = {
        str(metadata.get("benchmark_document_id") or metadata.get("source") or "").strip()
        for metadata in metas
        if str(metadata.get("benchmark_document_id") or metadata.get("source") or "").strip()
    }
    allowed_indices: list[int] | None
    if route is not None:
        allowed_indices = [
            index
            for index in range(len(chunks))
            if str(metas[index].get("benchmark_document_id", "")) == route.document_id
            or str(metas[index].get("source", "")) == route.document_id
        ]
    elif len(source_ids) == 1:
        allowed_indices = list(range(len(chunks)))
    else:
        allowed_indices = []
    limitation_indices = limitation_evidence_indices(
        question,
        [chunk.page_content for chunk in chunks],
        metas,
        max_results=6,
        allowed_indices=allowed_indices,
    )
    return list(dict.fromkeys([*limitation_indices, *original]))


def _non_formula_neighbor_index(
    chunks: list[Chunk],
    index: int,
    direction: int,
) -> int | None:
    """Return the adjacent ordinary block while treating formula evidence as out-of-band."""

    neighbor = index + direction
    while 0 <= neighbor < len(chunks):
        if chunks[neighbor].metadata.get("type") != "formula":
            return neighbor
        neighbor += direction
    return None


def _adjacent_context_indices(
    ranked: list[RankedItem],
    chunks: list[Chunk],
    anchor_count: int = MAX_ADJACENT_ANCHORS,
) -> list[int]:
    """Interleave same-page text neighbors around the strongest anchors.

    PDF paragraph boundaries often split a method statement from its numeric
    detail even though both remain on one page.  This bounded control expands
    only the first two ranked text chunks, never crosses a source or page, and
    never expands table/reference chunks.  It is deliberately opt-in until a
    full benchmark proves that displaced tail contexts do not regress.
    """

    original = [int(item.key) for item in ranked]
    ordered: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            ordered.append(index)

    for rank, index in enumerate(original):
        add(index)
        if rank >= max(0, int(anchor_count)) or not (0 <= index < len(chunks)):
            continue
        anchor = chunks[index]
        metadata = anchor.metadata if isinstance(anchor.metadata, dict) else {}
        if metadata.get("type", "text") != "text":
            continue
        source = str(metadata.get("source", ""))
        page = metadata.get("page")
        header = str(metadata.get("headers", ""))
        if not source or page is None or REFERENCE_HEADER_RE.search(header):
            continue
        for direction in (-1, 1):
            neighbor = _non_formula_neighbor_index(chunks, index, direction)
            if neighbor is None:
                continue
            candidate = chunks[neighbor]
            candidate_meta = (
                candidate.metadata if isinstance(candidate.metadata, dict) else {}
            )
            if candidate_meta.get("type", "text") != "text":
                continue
            if str(candidate_meta.get("source", "")) != source:
                continue
            if candidate_meta.get("page") != page:
                continue
            if REFERENCE_HEADER_RE.search(str(candidate_meta.get("headers", ""))):
                continue
            add(neighbor)
    return ordered


def _parent_window_scoring_chunks(
    ranked: list[RankedItem],
    chunks: list[Chunk],
    anchor_count: int = MAX_ADJACENT_ANCHORS,
) -> tuple[list[Chunk], dict[int, tuple[int, ...]]]:
    """Enrich top text anchors with same-page neighbors without changing rank.

    The returned list has the same length and indices as ``chunks``.  Only the
    copied anchor chunks receive concatenated text and ``window_chunk_indices``
    metadata, so existing ranking/provenance code can score the exact effective
    context while preserving the original top-k slots.  Neighbors already in
    the selected ranking are not duplicated inside an anchor window.
    """

    effective = list(chunks)
    selected = {int(item.key) for item in ranked}
    expansions: dict[int, tuple[int, ...]] = {}
    for item in ranked[: max(0, int(anchor_count))]:
        index = int(item.key)
        if not (0 <= index < len(chunks)):
            continue
        anchor = chunks[index]
        metadata = anchor.metadata if isinstance(anchor.metadata, dict) else {}
        if metadata.get("type", "text") != "text":
            continue
        # OCR/image blocks can place unrelated axes and sample counts in one
        # text span. Keep the anchor itself, but do not broaden that ambiguity
        # with same-page neighbours.
        if PICTURE_TEXT_MARKER_RE.search(anchor.page_content):
            continue
        source = str(metadata.get("source", ""))
        page = metadata.get("page")
        header = str(metadata.get("headers", ""))
        if not source or page is None or REFERENCE_HEADER_RE.search(header):
            continue
        included = [index]
        for direction in (-1, 1):
            neighbor = _non_formula_neighbor_index(chunks, index, direction)
            if neighbor is None or neighbor in selected:
                continue
            candidate = chunks[neighbor]
            candidate_meta = (
                candidate.metadata if isinstance(candidate.metadata, dict) else {}
            )
            if candidate_meta.get("type", "text") != "text":
                continue
            if PICTURE_TEXT_MARKER_RE.search(candidate.page_content):
                continue
            if str(candidate_meta.get("source", "")) != source:
                continue
            if candidate_meta.get("page") != page:
                continue
            if REFERENCE_HEADER_RE.search(str(candidate_meta.get("headers", ""))):
                continue
            included.append(neighbor)
        included.sort()
        if included == [index]:
            continue
        window_metadata = dict(metadata)
        window_metadata["window_chunk_indices"] = included
        effective[index] = Chunk(
            "\n\n".join(chunks[chunk_index].page_content for chunk_index in included),
            window_metadata,
        )
        expansions[index] = tuple(included)
    return effective, expansions


def _context_window_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [result.get("context_window", {}) for result in case_results]
    return {
        "cases": len(rows),
        "expanded_cases": sum(bool(row.get("expanded_anchor_count")) for row in rows),
        "expanded_anchor_count": sum(int(row.get("expanded_anchor_count", 0)) for row in rows),
        "added_chunk_count": sum(int(row.get("added_chunk_count", 0)) for row in rows),
        "added_character_count": sum(int(row.get("added_character_count", 0)) for row in rows),
    }


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


def _page_number(value: Any) -> int | None:
    """Return an integer page number without treating malformed metadata as page 0."""

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _provenance_location(
    case: dict[str, Any],
    chunk: Chunk,
    rank: int,
) -> dict[str, Any]:
    """Describe one fact-bearing chunk and flag provenance risks.

    This is deliberately diagnostic: it never removes a retrieved chunk or
    changes the retrieval score. ``benchmark_document_id`` is attached only
    by the benchmark runner; missing metadata is reported as ``unknown`` so
    legacy Chroma chunks are not silently treated as trusted evidence.
    """

    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    target_document = str(case.get("document_id") or "")
    actual_document = metadata.get("benchmark_document_id")
    if target_document and actual_document is not None:
        document_status = (
            "target" if str(actual_document) == target_document else "other"
        )
    else:
        document_status = "unknown"

    page = _page_number(metadata.get("page"))
    gold_pages = {
        _page_number(value)
        for value in case.get("source_pages", [])
        if _page_number(value) is not None
    }
    if not gold_pages:
        page_status = "not_scored"
    elif page is None:
        page_status = "missing"
    elif page in gold_pages:
        page_status = "match"
    else:
        page_status = "outside_gold_pages"

    headers = str(metadata.get("headers", ""))
    chunk_type = str(metadata.get("type", "text")).casefold()
    flags: list[str] = []
    if document_status == "other":
        flags.append("wrong_document")
    if page_status == "outside_gold_pages":
        flags.append("outside_gold_page")
    elif page_status == "missing":
        flags.append("missing_page")
    if REFERENCE_HEADER_RE.search(normalize_for_match(headers)):
        flags.append("reference_section")
    if chunk_type in {"figure", "image", "caption", "figure_caption"}:
        flags.append("figure_or_caption")

    return {
        "rank": rank,
        "chunk_index": None,
        "document_status": document_status,
        "benchmark_document_id": (
            str(actual_document) if actual_document is not None else None
        ),
        "source": metadata.get("source"),
        "page": page,
        "page_status": page_status,
        "headers": headers,
        "chunk_type": chunk_type,
        "flags": flags,
    }


def case_provenance(
    case: dict[str, Any],
    chunks: list[Chunk],
    ranked: list[RankedItem],
    top_k: int,
) -> dict[str, Any]:
    """Audit where each required fact appears in a retrieved prefix.

    Required-fact coverage remains the primary deterministic retrieval proxy.
    This companion audit exposes whether a lexical match is in the target
    paper/page or only appears in another paper, a reference section, a
    figure/caption, or a chunk without page metadata.
    """

    required = [str(fact) for fact in case.get("required_facts") or []]
    fact_matches: dict[str, list[dict[str, Any]]] = {}
    for fact in required:
        matches: list[dict[str, Any]] = []
        aliases = aliases_for_fact(case, fact)[1:]
        for rank, result in enumerate(ranked[:top_k], start=1):
            index = int(result.key)
            if index < 0 or index >= len(chunks):
                continue
            chunk = chunks[index]
            if not fact_is_present(fact, [chunk.page_content], aliases):
                continue
            location = _provenance_location(case, chunk, rank)
            location["chunk_index"] = index
            matches.append(location)
        fact_matches[fact] = matches

    matched_rows = [rows for rows in fact_matches.values() if rows]
    wrong_document_only = sum(
        bool(rows)
        and not any(row["document_status"] in {"target", "unknown"} for row in rows)
        for rows in fact_matches.values()
    )
    outside_page_only = sum(
        bool(rows)
        and not any(row["page_status"] in {"match", "not_scored"} for row in rows)
        for rows in fact_matches.values()
    )
    return {
        "required_fact_count": len(required),
        "matched_fact_count": len(matched_rows),
        "missing_fact_count": len(required) - len(matched_rows),
        "target_document_fact_count": sum(
            any(row["document_status"] == "target" for row in rows)
            for rows in fact_matches.values()
        ),
        "unknown_document_fact_count": sum(
            any(row["document_status"] == "unknown" for row in rows)
            for rows in fact_matches.values()
        ),
        "wrong_document_only_fact_count": wrong_document_only,
        "gold_page_fact_count": sum(
            any(row["page_status"] == "match" for row in rows)
            for rows in fact_matches.values()
        ),
        "outside_gold_page_only_fact_count": outside_page_only,
        "reference_section_fact_count": sum(
            any("reference_section" in row["flags"] for row in rows)
            for rows in fact_matches.values()
        ),
        "figure_or_caption_fact_count": sum(
            any("figure_or_caption" in row["flags"] for row in rows)
            for rows in fact_matches.values()
        ),
        "missing_page_fact_count": sum(
            any("missing_page" in row["flags"] for row in rows)
            for rows in fact_matches.values()
        ),
        "fact_matches": fact_matches,
    }


def aggregate_provenance(
    case_results: list[dict[str, Any]], top_k_values: list[int]
) -> dict[str, dict[str, int]]:
    """Aggregate provenance warning counts without hiding per-case locations."""

    fields = (
        "matched_fact_count",
        "target_document_fact_count",
        "unknown_document_fact_count",
        "wrong_document_only_fact_count",
        "gold_page_fact_count",
        "outside_gold_page_only_fact_count",
        "reference_section_fact_count",
        "figure_or_caption_fact_count",
        "missing_page_fact_count",
    )
    output: dict[str, dict[str, int]] = {}
    for top_k in top_k_values:
        rows = [
            result["provenance"][str(top_k)]
            for result in case_results
            if str(top_k) in result.get("provenance", {})
        ]
        output[str(top_k)] = {
            field: sum(int(row.get(field) or 0) for row in rows) for field in fields
        }
        output[str(top_k)]["cases_with_wrong_document_only_facts"] = sum(
            int(row.get("wrong_document_only_fact_count") or 0) > 0 for row in rows
        )
        output[str(top_k)]["cases_with_reference_section_matches"] = sum(
            int(row.get("reference_section_fact_count") or 0) > 0 for row in rows
        )
    return output


def summarize_document_routing(case_results: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize opt-in routes without using benchmark labels to choose them."""

    routed = [
        result.get("routing", {})
        for result in case_results
        if result.get("routing", {}).get("selected_document")
    ]
    correct = sum(
        route.get("selected_document") == result.get("document_id")
        for result in case_results
        for route in [result.get("routing", {})]
        if route.get("selected_document")
    )
    return {
        "cases": len(case_results),
        "routed_cases": len(routed),
        "unrouted_cases": len(case_results) - len(routed),
        "correct_routes": correct,
        "incorrect_routes": len(routed) - correct,
    }


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
    structured_table_guard: bool = False,
    structured_figure_guard: bool = False,
    formula_evidence: bool = False,
    limitation_evidence: bool = False,
    adjacent_context: bool = False,
    parent_window: bool = False,
    document_routing: bool = False,
    query_decomposition: bool = False,
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
        document_routing=document_routing,
        query_decomposition=query_decomposition,
        excluded_chunk_types=("formula", "figure")
        if structured_figure_guard
        else ("formula",),
    )
    max_k = max(top_k_values, default=0)
    retrieval_k = max(max_k, int(reranker_candidate_k)) if reranker else max_k
    rerank_documents = [searchable_text(chunk) for chunk in chunks] if reranker else []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        total_started = perf_counter()
        retrieval_started = perf_counter()
        route = index.route(question)
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
                base_ranked = reciprocal_rank_fusion(
                    [rerank_result.ranked, candidates],
                    rrf_k=reranker_fusion_rrf_k,
                    limit=fusion_limit,
                )
            elif reranker_fusion == "weighted_rrf":
                base_ranked = reciprocal_rank_fusion(
                    [rerank_result.ranked, candidates],
                    rrf_k=reranker_fusion_rrf_k,
                    limit=fusion_limit,
                    weights=[reranker_fusion_ce_weight, 1.0],
                )
            else:
                base_ranked = rerank_result.ranked[:fusion_limit]
            rerank_seconds = rerank_result.elapsed_seconds
            scored_pairs = rerank_result.scored_pairs
            cache_hits = rerank_result.cache_hits
        else:
            base_ranked = candidates[:max_k]

        def prepare_for_top_k(
            top_k: int,
        ) -> tuple[list[RankedItem], list[Chunk], dict[int, tuple[int, ...]]]:
            """Apply optional post-ranking controls independently for one k."""

            ranked_for_k = list(base_ranked[:top_k])
            if section_expansion:
                ranked_for_k = [
                    RankedItem(index, 0.0)
                    for index in _section_expansion_indices(
                        question,
                        ranked_for_k,
                        chunks,
                        anchor_ranked=ranked_for_k,
                        route_source=route.document_id if route else None,
                    )[:top_k]
                ]
            if adjacent_context:
                score_by_index = {int(item.key): item.score for item in ranked_for_k}
                ranked_for_k = [
                    RankedItem(index, score_by_index.get(index, 0.0))
                    for index in _adjacent_context_indices(ranked_for_k, chunks)[:top_k]
                ]
            if structured_table_guard:
                score_by_index = {int(item.key): item.score for item in ranked_for_k}
                ranked_for_k = [
                    RankedItem(index, score_by_index.get(index, 0.0))
                    for index in _structured_table_guard_indices(
                        question,
                        ranked_for_k,
                        chunks,
                        route=route,
                    )[:top_k]
                ]
            if structured_figure_guard:
                score_by_index = {int(item.key): item.score for item in ranked_for_k}
                ranked_for_k = [
                    RankedItem(index, score_by_index.get(index, 0.0))
                    for index in _structured_figure_guard_indices(
                        question,
                        ranked_for_k,
                        chunks,
                        route=route,
                    )[:top_k]
                ]
            if formula_evidence:
                score_by_index = {int(item.key): item.score for item in ranked_for_k}
                ranked_for_k = [
                    RankedItem(index, score_by_index.get(index, 0.0))
                    for index in _formula_guard_indices(
                        question,
                        ranked_for_k,
                        chunks,
                        route=route,
                    )[:top_k]
                ]
            if limitation_evidence:
                score_by_index = {int(item.key): item.score for item in ranked_for_k}
                ranked_for_k = [
                    RankedItem(index, score_by_index.get(index, 0.0))
                    for index in _limitation_guard_indices(
                        question,
                        ranked_for_k,
                        chunks,
                        route=route,
                    )[:top_k]
                ]
            scoring_chunks_for_k = chunks
            window_expansions_for_k: dict[int, tuple[int, ...]] = {}
            if parent_window:
                scoring_chunks_for_k, window_expansions_for_k = _parent_window_scoring_chunks(
                    ranked_for_k,
                    chunks,
                )
            return ranked_for_k, scoring_chunks_for_k, window_expansions_for_k

        prepared = {
            int(top_k): prepare_for_top_k(int(top_k))
            for top_k in top_k_values
        }
        display_ranked, _display_scoring_chunks, display_window_expansions = prepared[max_k]
        total_seconds = perf_counter() - total_started
        metrics: dict[str, dict[str, Any]] = {}
        provenance: dict[str, dict[str, Any]] = {}
        for top_k in top_k_values:
            ranked_for_k, scoring_chunks_for_k, _window_expansions_for_k = prepared[int(top_k)]
            prefix = ranked_for_k[:top_k]
            metrics[str(top_k)] = {
                "reference_context_recall": _case_context_recall(case, scoring_chunks_for_k, prefix),
                "target_document_hit": _case_document_hit(case, scoring_chunks_for_k, prefix),
                "source_page_hit": _case_page_hit(case, scoring_chunks_for_k, prefix),
                "table_number_hit": _case_table_hit(case, scoring_chunks_for_k, prefix),
                **_case_required_fact_coverage(case, scoring_chunks_for_k, prefix),
            }
            provenance[str(top_k)] = case_provenance(
                case, scoring_chunks_for_k, prefix, top_k
            )
        added_indices = {
            neighbor
            for anchor, included in display_window_expansions.items()
            for neighbor in included
            if neighbor != anchor
        }
        case_results.append(
            {
                "case_id": case["case_id"],
                "document_id": case.get("document_id", document_id),
                "question": case["question"],
                "type": case.get("type", ""),
                "required_facts": case.get("required_facts", []),
                "required_fact_aliases": case.get("required_fact_aliases", {}),
                "routing": {
                    "enabled": bool(document_routing),
                    "selected_document": route.document_id if route else None,
                    "distinctive_tokens": list(route.distinctive_tokens) if route else [],
                },
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
                        "figure_number": chunks[int(result.key)].metadata.get("figure_number"),
                        "window_chunk_indices": list(
                            display_window_expansions.get(int(result.key), (int(result.key),))
                        ),
                    }
                    for rank, result in enumerate(display_ranked, start=1)
                ],
                "candidate_count": len(candidates),
                "context_window": {
                    "enabled": bool(parent_window),
                    "expanded_anchor_count": len(display_window_expansions),
                    "added_chunk_count": len(added_indices),
                    "added_character_count": sum(
                        len(chunks[index].page_content) for index in added_indices
                    ),
                },
                "timing": {
                    "retrieval_seconds": retrieval_seconds,
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": total_seconds,
                    "reranker_scored_pairs": scored_pairs,
                    "reranker_cache_hits": cache_hits,
                },
                "metrics": metrics,
                "provenance": provenance,
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
        "context_window": _context_window_summary(case_results),
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
    structured_table_guard: bool = False,
    spatial_figure_evidence: bool = False,
    formula_evidence: bool = False,
    limitation_evidence: bool = False,
    adjacent_context: bool = False,
    parent_window: bool = False,
    document_routing: bool = False,
    query_decomposition: bool = False,
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
        chunks = load_and_split_document(
            str(path),
            include_spatial_figures=spatial_figure_evidence,
        )
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
        structured_table_guard=structured_table_guard,
        structured_figure_guard=spatial_figure_evidence,
        formula_evidence=formula_evidence,
        limitation_evidence=limitation_evidence,
        adjacent_context=adjacent_context,
        parent_window=parent_window,
        document_routing=document_routing,
        query_decomposition=query_decomposition,
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
                structured_table_guard=structured_table_guard,
                structured_figure_guard=spatial_figure_evidence,
                formula_evidence=formula_evidence,
                limitation_evidence=limitation_evidence,
                adjacent_context=adjacent_context,
                parent_window=parent_window,
                document_routing=document_routing,
                query_decomposition=query_decomposition,
            )
        )
    method = {"bm25": "bm25-lite", "dense": "dense-local", "hybrid": "hybrid-rrf"}[retriever]
    if reranker:
        method += "+cross-encoder"
    return {
        "schema_version": 8,
        "method": method,
        "retriever": retriever,
        "dense_model": dense_model_name if retriever in {"dense", "hybrid"} else None,
        "rrf_k": rrf_k if retriever == "hybrid" else None,
        "document_routing": bool(document_routing),
        "query_decomposition": bool(query_decomposition),
        "structured_table_guard": bool(structured_table_guard),
        "spatial_figure_evidence": bool(spatial_figure_evidence),
        "formula_evidence": bool(formula_evidence),
        "limitation_evidence": bool(limitation_evidence),
        "adjacent_context": bool(adjacent_context),
        "parent_window": bool(parent_window),
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
        "provenance": aggregate_provenance(global_result["cases_detail"], normalized_k),
        "routing": {
            "enabled": bool(document_routing),
            **summarize_document_routing(global_result["cases_detail"]),
        },
        "fact_failures": fact_failure_lists(global_result["cases_detail"], normalized_k),
        "latency": global_result["latency"],
        "context_window": global_result["context_window"],
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
            "structured_table_guard is an opt-in application-path control: explicit table questions scan canonical table chunks within an existing source route and use the deterministic row/column lookup before normal context ordering; raw retriever controls keep it disabled.",
            "spatial_figure_evidence is an opt-in born-digital-PDF control: it serializes short text-layer blocks above recognized Figure N captions with normalized coordinates and promotes an exact figure match within the existing source route. It performs no OCR or image-pixel understanding.",
            "adjacent_context is an opt-in bounded control that interleaves same-page text neighbors around the first two ranked anchors without crossing sources, pages, tables, or reference sections.",
            "parent_window is an opt-in effective-context control: it concatenates eligible same-page neighbors inside the first two text anchors without consuming additional top-k slots; window indices and character overhead are recorded per case.",
            "formula_evidence is an opt-in lexical guard for explicit equation/PDE questions: it promotes a small same-source set of formula-bearing text blocks and is not a symbolic solver.",
            "limitation_evidence is an opt-in lexical guard for explicit limitation/failure questions: it promotes a small same-source set of passages that state the limitation and its example; it is not a semantic truth judge.",
            "document_routing is an opt-in lexical source router: it narrows retrieval only when distinctive ASCII identifiers belong to exactly one source; ambiguous or generic queries use the global index.",
            "query_decomposition is an opt-in deterministic multi-query control: the original question is always retained and bounded punctuation/conjunction clauses are retrieved within the original route scope, then fused with RRF; it does not translate or inject benchmark facts.",
            "No ChromaDB, Gradio, RAGAS, or external API is used; dense-local and hybrid-rrf do use the locally cached embedding model described above.",
        ],
    }


def _print_summary(report: dict[str, Any], show_failures: bool = False) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(
        f"方法：{report['method']}；top-k：{','.join(map(str, report['top_k']))}；"
        f"query-decomposition={'on' if report.get('query_decomposition') else 'off'}；"
        f"structured-table-guard={'on' if report.get('structured_table_guard') else 'off'}；"
        f"spatial-figure-evidence={'on' if report.get('spatial_figure_evidence') else 'off'}；"
        f"formula-evidence={'on' if report.get('formula_evidence') else 'off'}；"
        f"limitation-evidence={'on' if report.get('limitation_evidence') else 'off'}；"
        f"adjacent-context={'on' if report.get('adjacent_context') else 'off'}；"
        f"parent-window={'on' if report.get('parent_window') else 'off'}"
    )
    if report.get("reranker"):
        config = report["reranker"]
        print(
            f"reranker：{config['model']}@{config['revision'] or 'default'}；"
            f"candidate_k={config['candidate_k']}；batch={config['batch_size']}；"
            f"max_length={config['max_length']}；device={config['device']}；"
            f"fusion={config['fusion']}；"
            f"load={fmt(config['load_seconds'])}s"
        )
    if report.get("parent_window"):
        window = report.get("context_window", {})
        print(
            "parent-window："
            f"expanded_cases={window.get('expanded_cases', 0)}/{window.get('cases', 0)}；"
            f"anchors={window.get('expanded_anchor_count', 0)}；"
            f"added_chunks={window.get('added_chunk_count', 0)}；"
            f"added_chars={window.get('added_character_count', 0)}"
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
    provenance = report.get("provenance", {}).get(largest_k)
    if provenance:
        print(
            f"\n来源风险汇总（@{largest_k}，按事实计）："
            f"wrong_document_only={provenance['wrong_document_only_fact_count']}；"
            f"outside_gold_page_only={provenance['outside_gold_page_only_fact_count']}；"
            f"reference_section={provenance['reference_section_fact_count']}；"
            f"missing_page={provenance['missing_page_fact_count']}"
        )
    routing = report.get("routing")
    if routing and routing.get("enabled"):
        print(
            "文档路由汇总："
            f"routed={routing['routed_cases']}/{routing['cases']}；"
            f"correct={routing['correct_routes']}；"
            f"incorrect={routing['incorrect_routes']}"
        )

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
        "--structured-table-guard",
        action="store_true",
        help="启用与网页一致的结构化表格扫描和确定性单元格保护；默认关闭",
    )
    parser.add_argument(
        "--spatial-figure-evidence",
        action="store_true",
        help="抽取 PDF 文字层的 Figure N 坐标证据并启用精确图号保护；无 OCR/像素理解，默认关闭",
    )
    parser.add_argument(
        "--formula-evidence",
        action="store_true",
        help="对显式公式/PDE问题加入同来源公式文字候选；默认关闭",
    )
    parser.add_argument(
        "--limitation-evidence",
        action="store_true",
        help="对显式限制/失败问题加入同来源限制文字候选；默认关闭",
    )
    parser.add_argument(
        "--adjacent-context",
        action="store_true",
        help="围绕前两个文本锚点加入同来源同页相邻块的受控对照；默认关闭",
    )
    parser.add_argument(
        "--parent-window",
        action="store_true",
        help="不改变 top-k 排名，在前两个文本锚点内附加同来源同页邻块；默认关闭",
    )
    parser.add_argument(
        "--document-routing",
        action="store_true",
        help="按唯一高信号术语启用保守的文档路由对照；歧义问题回退全库，默认关闭",
    )
    parser.add_argument(
        "--query-decomposition",
        action="store_true",
        help="对复合问题启用有界子查询 RRF 对照；原问题始终保留，默认关闭",
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
            structured_table_guard=args.structured_table_guard,
            spatial_figure_evidence=args.spatial_figure_evidence,
            formula_evidence=args.formula_evidence,
            limitation_evidence=args.limitation_evidence,
            adjacent_context=args.adjacent_context,
            parent_window=args.parent_window,
            document_routing=args.document_routing,
            query_decomposition=args.query_decomposition,
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
