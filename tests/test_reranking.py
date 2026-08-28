import unittest

from sci_rag_reranking import CrossEncoderReranker
from sci_rag_retrieval import RankedItem


class FakeCrossEncoder:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append((list(pairs), batch_size, show_progress_bar))
        return [text.count("relevant") for _, text in pairs]


class RerankingTests(unittest.TestCase):
    def test_reranker_scores_candidates_and_preserves_input_order_on_ties(self):
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model, batch_size=2)
        candidates = [RankedItem(0, 0.9), RankedItem(1, 0.8), RankedItem(2, 0.7)]
        documents = ["not useful", "relevant relevant", "relevant once"]

        result = reranker.rerank("question", candidates, documents)

        self.assertEqual([item.key for item in result.ranked], [1, 2, 0])
        self.assertEqual([item.score for item in result.ranked], [2.0, 1.0, 0.0])
        self.assertEqual(result.scored_pairs, 3)
        self.assertEqual(result.cache_hits, 0)
        self.assertEqual(model.calls[0][1], 2)

    def test_benchmark_cache_reuses_identical_query_passage_scores(self):
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model, cache_scores=True)
        candidates = [RankedItem(0, 1.0), RankedItem(1, 0.5)]
        documents = ["relevant", "not useful"]

        first = reranker.rerank("question", candidates, documents)
        second = reranker.rerank("question", candidates, documents)

        self.assertEqual(first.scored_pairs, 2)
        self.assertEqual(second.scored_pairs, 0)
        self.assertEqual(second.cache_hits, 2)
        self.assertEqual(len(model.calls), 1)

    def test_reranker_rejects_invalid_configuration_and_candidate_key(self):
        with self.assertRaises(ValueError):
            CrossEncoderReranker(model=FakeCrossEncoder(), batch_size=0)
        with self.assertRaises(ValueError):
            CrossEncoderReranker(model=FakeCrossEncoder(), max_length=0)
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())
        with self.assertRaises(IndexError):
            reranker.rerank("question", [RankedItem(2, 1.0)], ["only one"])


if __name__ == "__main__":
    unittest.main()
