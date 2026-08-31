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
_CLAUSE_SEPARATOR_RE = re.compile(r"[，,；;]+")
_CONJUNCTION_RE = re.compile(r"\s*(与|以及|和)\s*")
_QUESTION_CUE_RE = re.compile(r"什么|哪些|多少|如何|是否|分别|配置|指标|问题|条件|下")


def tokenize(value: Any) -> list[str]:
    """Tokenize English identifiers, numbers, and individual CJK characters."""

    return [token.casefold() for token in TOKEN_RE.findall(normalize_for_match(value))]


def query_variants(question: str, max_variants: int = 4) -> list[str]:
    """Return a bounded set of deterministic subqueries for composite questions.

    The original question is always first.  Clause variants are deliberately
    conservative: punctuation is preferred, while ``与/以及/和`` is used only
    when both sides look like independent clauses or paired ASCII identifiers.
    This helper does not translate or invent terms; callers can opt into
    multi-query retrieval without changing the default single-query path.
    """

    if max_variants <= 0:
        return []
    original = str(question or "").strip()
    if not original:
        return []
    base = original.rstrip("？?!。")
    clauses = [part.strip() for part in _CLAUSE_SEPARATOR_RE.split(base) if part.strip()]
    if len(clauses) == 1:
        clause = clauses[0]
        clauses = []
        for match in _CONJUNCTION_RE.finditer(clause):
            left = clause[: match.start()].strip()
            right = clause[match.end() :].strip()
            if len(left) < 4 or len(right) < 3:
                continue
            left_ascii = bool(re.search(r"[A-Za-z0-9]", left))
            right_ascii = bool(re.search(r"[A-Za-z0-9]", right))
            cue = bool(_QUESTION_CUE_RE.search(left + right))
            if (left_ascii and right_ascii) or cue:
                clauses = [left, right]
                break

    variants: list[str] = []
    for value in [original, *clauses]:
        value = value.strip()
        if len(value) < 3 or value in variants:
            continue
        variants.append(value)
        if len(variants) >= int(max_variants):
            break
    return variants


@dataclass(frozen=True)
class RankedItem:
    key: Hashable
    score: float


@dataclass(frozen=True)
class DocumentRoute:
    """A conservative route selected from source-level lexical signals."""

    document_id: str
    distinctive_tokens: tuple[str, ...]


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

    def retrieve(
        self,
        question: str,
        k: int = 10,
        indices: Iterable[int] | None = None,
    ) -> list[RankedItem]:
        """Return stable score-descending indices, preserving source order on ties."""

        limit = max(0, min(int(k), len(self.documents)))
        candidate_indices = (
            range(len(self.documents))
            if indices is None
            else [index for index in indices if 0 <= int(index) < len(self.documents)]
        )
        ranked = [
            RankedItem(int(index), self.score(question, int(index)))
            for index in candidate_indices
        ]
        ranked.sort(key=lambda item: (-item.score, int(item.key)))
        return ranked[:limit]


class DocumentRouter:
    """Route only when a distinctive ASCII identifier belongs to one source.

    The router intentionally refuses to guess from generic words such as
    ``table``, ``method`` or ``RAG``. It also refuses when distinctive terms
    point to different documents. A caller can therefore fall back to the
    global index whenever the query is ambiguous or cross-document.
    """

    _DISTINCTIVE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9*._+\-]{3,}$")
    # A token that happens to occur in only one paper is not necessarily a
    # document identifier.  These words are common in scientific prose and
    # must not route a query merely because of corpus imbalance.
    _GENERIC_TOKENS = frozenset(
        {
            "paper",
            "papers",
            "article",
            "method",
            "methods",
            "model",
            "models",
            "dataset",
            "datasets",
            "data",
            "table",
            "tables",
            "figure",
            "figures",
            "results",
            "result",
            "approach",
            "study",
            "work",
            "section",
            "training",
            "performance",
            "question",
            "questions",
            "answer",
            "answers",
            "pipeline",
            "evaluation",
            "experiment",
            "experiments",
            "scientific",
        }
    )

    def __init__(self, document_ids: Iterable[str], profiles: Iterable[str]):
        self.document_ids = [str(value) for value in document_ids]
        self.profiles = [str(value) for value in profiles]
        if len(self.document_ids) != len(self.profiles):
            raise ValueError("document_ids 与 profiles 数量必须一致")
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids 必须唯一")
        profile_tokens = [set(tokenize(profile)) for profile in self.profiles]
        token_documents: dict[str, set[int]] = {}
        for index, tokens in enumerate(profile_tokens):
            for token in tokens:
                token_documents.setdefault(token, set()).add(index)
        self._token_documents = token_documents

    def route(self, question: str) -> DocumentRoute | None:
        query_tokens = {
            token
            for token in tokenize(question)
            if self._DISTINCTIVE_TOKEN_RE.fullmatch(token)
            and token not in self._GENERIC_TOKENS
        }
        document_tokens: dict[int, list[str]] = {}
        for token in sorted(query_tokens):
            owners = self._token_documents.get(token, set())
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            document_tokens.setdefault(owner, []).append(token)
        if len(document_tokens) != 1:
            return None
        owner, tokens = next(iter(document_tokens.items()))
        return DocumentRoute(self.document_ids[owner], tuple(tokens))


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
