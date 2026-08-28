"""Side-effect-free cross-encoder reranking helpers.

Importing this module does not load Sentence-Transformers or any model.  The
caller must explicitly construct :class:`CrossEncoderReranker`, which defaults
to local-files-only loading so an evaluation cannot download weights silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from sci_rag_retrieval import RankedItem


def reranker_document_text(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Build the same query-passage text for offline and runtime reranking."""

    metadata = metadata or {}
    return "\n".join(
        [
            str(text),
            str(metadata.get("table_caption", "")),
            str(metadata.get("headers", "")),
        ]
    )


@dataclass(frozen=True)
class RerankResult:
    ranked: list[RankedItem]
    elapsed_seconds: float
    scored_pairs: int
    cache_hits: int


class CrossEncoderReranker:
    """Rank query-passage pairs with an explicitly loaded cross-encoder."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        revision: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "cpu",
        local_files_only: bool = True,
        model: Any | None = None,
        cache_scores: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("reranker batch_size 必须为正整数")
        if max_length <= 0:
            raise ValueError("reranker max_length 必须为正整数")
        if model is None:
            if not model_name_or_path:
                raise ValueError("必须提供 reranker 模型或 model_name_or_path")
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                model_name_or_path,
                revision=revision,
                local_files_only=local_files_only,
                max_length=max_length,
                device=device,
            )
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = str(getattr(model, "device", device))
        self.cache_scores = bool(cache_scores)
        self._score_cache: dict[tuple[str, str], float] = {}

    def rerank(
        self,
        question: str,
        candidates: Sequence[RankedItem],
        documents: Sequence[str],
    ) -> RerankResult:
        """Score candidates and preserve their original order for exact ties."""

        started = perf_counter()
        if not candidates:
            return RerankResult([], perf_counter() - started, 0, 0)

        candidate_texts: list[str] = []
        for candidate in candidates:
            index = int(candidate.key)
            if index < 0 or index >= len(documents):
                raise IndexError(f"reranker candidate key 越界：{candidate.key}")
            candidate_texts.append(str(documents[index]))

        scores: list[float | None] = [None] * len(candidates)
        missing_pairs: list[list[str]] = []
        missing_positions: list[int] = []
        cache_hits = 0
        for position, text in enumerate(candidate_texts):
            cache_key = (question, text)
            if self.cache_scores and cache_key in self._score_cache:
                scores[position] = self._score_cache[cache_key]
                cache_hits += 1
            else:
                missing_pairs.append([question, text])
                missing_positions.append(position)

        if missing_pairs:
            predicted = self.model.predict(
                missing_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            predicted_values = [float(value) for value in predicted]
            if len(predicted_values) != len(missing_positions):
                raise ValueError("reranker 返回分数数量与候选数量不一致")
            for position, score in zip(missing_positions, predicted_values):
                scores[position] = score
                if self.cache_scores:
                    self._score_cache[(question, candidate_texts[position])] = score

        rescored = [
            (position, RankedItem(candidate.key, float(scores[position])))
            for position, candidate in enumerate(candidates)
        ]
        rescored.sort(key=lambda value: (-value[1].score, value[0]))
        return RerankResult(
            ranked=[item for _, item in rescored],
            elapsed_seconds=perf_counter() - started,
            scored_pairs=len(missing_pairs),
            cache_hits=cache_hits,
        )
