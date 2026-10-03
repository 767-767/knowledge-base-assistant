"""Side-effect-free cross-encoder reranking helpers.

Importing this module does not load Sentence-Transformers or any model. The
caller must explicitly construct :class:`CrossEncoderReranker`.
"""

from __future__ import annotations

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
        self.batch_size = int(batch_size)

    def rerank(
        self,
        question: str,
        candidates: Sequence[RankedItem],
        documents: Sequence[str],
    ) -> list[RankedItem]:
        """Score candidates and preserve their original order for exact ties."""

        if not candidates:
            return []

        candidate_texts: list[str] = []
        for candidate in candidates:
            index = int(candidate.key)
            if index < 0 or index >= len(documents):
                raise IndexError(f"reranker candidate key 越界：{candidate.key}")
            candidate_texts.append(str(documents[index]))

        scores = self.model.predict(
            [[question, text] for text in candidate_texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(candidates):
            raise ValueError("reranker 返回分数数量与候选数量不一致")
        return sorted(
            [RankedItem(candidate.key, float(score)) for candidate, score in zip(candidates, scores)],
            key=lambda item: -item.score,
        )
