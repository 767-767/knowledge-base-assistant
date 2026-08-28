import unittest

from evaluation.benchmark_retrieval import BM25Index, RankedChunk, evaluate_document, reciprocal_rank_fusion
from sci_rag_core import Chunk


class BenchmarkRetrievalTests(unittest.TestCase):
    def test_rrf_deduplicates_each_ranked_list_and_preserves_fused_order(self):
        fused = reciprocal_rank_fusion(
            [[1, 2, 2, 3], [3, RankedChunk(1, 0.1), 4]],
            rrf_k=10,
        )

        self.assertEqual([item.index for item in fused], [1, 3, 2, 4])
        self.assertGreater(fused[0].score, fused[2].score)

    def test_rrf_rejects_non_positive_constant(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[1]], rrf_k=0)

    def test_bm25_prefers_matching_table_and_is_deterministic(self):
        chunks = [
            Chunk("Background about optimization.", {"page": 1, "type": "text"}),
            Chunk(
                "|Model|Score|\n|---|---|\n|DrugR*|0.2060|",
                {"page": 3, "type": "table", "table_number": 2, "table_caption": "Table 2: Results"},
            ),
            Chunk(
                "|Model|Score|\n|---|---|\n|DrugR|0.2712|",
                {"page": 2, "type": "table", "table_number": 1, "table_caption": "Table 1: Results"},
            ),
        ]
        index = BM25Index(chunks)
        first = index.retrieve("Table 2 DrugR* Score", 3)
        second = index.retrieve("Table 2 DrugR* Score", 3)

        self.assertEqual([item.index for item in first], [item.index for item in second])
        self.assertEqual(first[0].index, 1)

    def test_diagnostic_reports_page_and_table_proxies_separately(self):
        chunks = [
            Chunk("Introductory prose.", {"page": 1, "type": "text"}),
            Chunk(
                "|Model|Score|\n|---|---|\n|DrugR*|0.2060|",
                {"page": 3, "type": "table", "table_number": 2, "table_caption": "Table 2: Results"},
            ),
        ]
        cases = [
            {
                "case_id": "case-1",
                "question": "Table 2 中 DrugR* 的 Score 是多少？",
                "type": "table",
                "contexts": ["|Model|Score|\n|---|---|\n|DrugR*|0.2060|"],
                "source_pages": [3],
            }
        ]
        report = evaluate_document("paper", cases, chunks, [1, 2])

        self.assertEqual(report["aggregate"]["1"]["source_page_hit_rate"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["table_number_hit_rate"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["reference_context_recall"], 1.0)

    def test_report_has_case_weighted_overall_aggregate(self):
        chunks = [Chunk("A result 1.", {"page": 1, "type": "text"})]
        cases = [
            {
                "case_id": "case-1",
                "question": "result",
                "contexts": ["A result 1."],
                "source_pages": [1],
            },
            {
                "case_id": "case-2",
                "question": "missing",
                "contexts": ["Other evidence."],
                "source_pages": [2],
            },
        ]
        report = evaluate_document("paper", cases, chunks, [1])
        self.assertIn("aggregate", report)
        self.assertEqual(report["aggregate"]["1"]["reference_context_recall"], 0.5)

    def test_global_ranking_does_not_count_another_paper_as_evidence(self):
        chunks = [
            Chunk(
                "DrugR score 0.99 in paper B.",
                {"page": 1, "type": "text", "benchmark_document_id": "paper-b"},
            ),
            Chunk(
                "DrugR score 0.20 in paper A.",
                {"page": 2, "type": "text", "benchmark_document_id": "paper-a"},
            ),
        ]
        cases = [
            {
                "case_id": "case-a",
                "document_id": "paper-a",
                "question": "DrugR score",
                "contexts": ["DrugR score 0.20 in paper A."],
                "source_pages": [2],
            }
        ]
        report = evaluate_document("all-documents", cases, chunks, [1, 2])

        self.assertEqual(report["aggregate"]["1"]["reference_context_recall"], 0.0)
        self.assertEqual(report["aggregate"]["2"]["reference_context_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
