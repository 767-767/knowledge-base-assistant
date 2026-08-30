"""Side-effect-free lexical ranking and rank fusion primitives.

The module does not import ChromaDB, Sentence-Transformers, Gradio, or an API
client.  Both the application runtime and offline benchmark can therefore use
the same BM25/RRF implementation without creating resources at import time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Hashable, Iterable
import math
import re
from typing import Any

from sci_rag_core import normalize_for_match


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9*._+\-]*|[\u4e00-\u9fff]")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(value: Any) -> list[str]:
    """Tokenize English identifiers, numbers, and individual CJK characters."""

    return [token.casefold() for token in TOKEN_RE.findall(normalize_for_match(value))]


@dataclass(frozen=True)
class RankedItem:
    key: Hashable
    score: float


class BM25Index:
    """Small deterministic BM25 implementation over caller-owned documents."""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(document) for document in self.documents]
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

    def matching_query_tokens(self, question: str) -> set[str]:
        """Return distinct query tokens that occur in at least one document."""

        return {
            token
            for token in set(tokenize(question))
            if self._document_frequency.get(token, 0) > 0
        }

    def has_lexical_signal(self, question: str) -> bool:
        """Whether BM25 has enough signal to participate in rank fusion.

        A Chinese question over an English paper may contain only one repeated
        ASCII identifier, such as a method name. Treating the resulting BM25
        order as equally informative as dense retrieval can displace relevant
        cross-lingual evidence. In that language-mismatch case, require at
        least two matching ASCII terms; same-language queries and queries with
        matching CJK terms retain normal lexical ranking.
        """

        query_tokens = set(tokenize(question))
        matching = self.matching_query_tokens(question)
        if not matching:
            return False
        contains_cjk = any(CJK_TOKEN_RE.fullmatch(token) for token in query_tokens)
        if not contains_cjk:
            return True
        if any(CJK_TOKEN_RE.fullmatch(token) for token in matching):
            return True
        return len(matching) >= 2

    def retrieve(self, question: str, k: int = 10) -> list[RankedItem]:
        """Return stable score-descending indices, preserving source order on ties."""

        limit = max(0, min(int(k), len(self.documents)))
        ranked = [RankedItem(index, self.score(question, index)) for index in range(len(self.documents))]
        ranked.sort(key=lambda item: (-item.score, int(item.key)))
        return ranked[:limit]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Hashable | RankedItem]],
    rrf_k: int = 60,
    limit: int | None = None,
    weights: Iterable[float] | None = None,
) -> list[RankedItem]:
    """Fuse ranked keys with optional per-list weights.

    ``weights`` is opt-in so existing callers retain ordinary equal-weight
    RRF. A weight scales a list's rank contribution without changing the
    within-list order.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须为正整数")
    ranking_lists = list(rankings)
    if weights is None:
        ranking_weights = [1.0] * len(ranking_lists)
    else:
        ranking_weights = [float(value) for value in weights]
        if len(ranking_weights) != len(ranking_lists):
            raise ValueError("weights 数量必须与 rankings 数量一致")
        if any(not math.isfinite(value) or value <= 0.0 for value in ranking_weights):
            raise ValueError("weights 必须为有限正数")
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    seen_order = 0
    for ranking, weight in zip(ranking_lists, ranking_weights):
        list_seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            key = item.key if isinstance(item, RankedItem) else item
            if key in list_seen:
                continue
            list_seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in first_seen:
                first_seen[key] = seen_order
                seen_order += 1
    fused = [RankedItem(key, score) for key, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, first_seen[item.key]))
    return fused if limit is None else fused[: max(0, int(limit))]
