import unittest

from evaluation.benchmark_retrieval import (
    evaluate_document,
    fact_failure_lists,
    searchable_text,
)
from sci_rag_reranking import CrossEncoderReranker
from sci_rag_retrieval import BM25Index, RankedItem, reciprocal_rank_fusion
from sci_rag_core import Chunk


class BenchmarkRetrievalTests(unittest.TestCase):
    def test_cross_encoder_rrf_is_applied_before_fact_metrics(self):
        class Model:
            device = "cpu"

            def predict(self, pairs, batch_size, show_progress_bar):
                return [1.0 if "answer" in passage else 0.0 for _, passage in pairs]

        chunks = [
            Chunk("query but no evidence", {"page": 1, "type": "text"}),
            Chunk("The answer is 42.", {"page": 2, "type": "text"}),
        ]
        cases = [
            {
                "case_id": "case-1",
                "question": "query",
                "contexts": ["The answer is 42."],
                "required_facts": ["42"],
                "source_pages": [2],
            }
        ]
        reranker = CrossEncoderReranker(model=Model(), batch_size=2)

        report = evaluate_document(
            "paper",
            cases,
            chunks,
            [1],
            reranker=reranker,
            reranker_candidate_k=2,
            reranker_fusion="rrf",
        )

        detail = report["cases_detail"][0]
        self.assertEqual(detail["top_results"][0]["chunk_index"], 1)
        self.assertEqual(detail["metrics"]["1"]["fact_coverage_status"], "full")
        self.assertEqual(detail["timing"]["reranker_scored_pairs"], 2)

    def test_diagnostic_rejects_unknown_reranker_fusion(self):
        with self.assertRaises(ValueError):
            evaluate_document("paper", [], [], [1], reranker_fusion="unknown")

    def test_rrf_deduplicates_each_ranked_list_and_preserves_fused_order(self):
        fused = reciprocal_rank_fusion(
            [[1, 2, 2, 3], [3, RankedItem(1, 0.1), 4]],
            rrf_k=10,
        )

        self.assertEqual([item.key for item in fused], [1, 3, 2, 4])
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
        index = BM25Index(searchable_text(chunk) for chunk in chunks)
        first = index.retrieve("Table 2 DrugR* Score", 3)
        second = index.retrieve("Table 2 DrugR* Score", 3)

        self.assertEqual([item.key for item in first], [item.key for item in second])
        self.assertEqual(first[0].key, 1)

    def test_bm25_detects_weak_cross_language_signal(self):
        index = BM25Index(
            [
                "DrugR overview and evaluation.",
                "DrugR dataset contains 4,855 samples.",
                "Table 2 reports DrugR* scores.",
            ]
        )

        self.assertFalse(index.has_lexical_signal("DrugR 的显式推理数据集有多少样本？"))
        self.assertTrue(index.has_lexical_signal("Table 2 中 DrugR* 的得分是多少？"))
        self.assertTrue(index.has_lexical_signal("DrugR dataset samples"))

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
                "required_facts": ["DrugR*", "0.2060"],
            }
        ]
        report = evaluate_document("paper", cases, chunks, [1, 2])

        self.assertEqual(report["aggregate"]["1"]["source_page_hit_rate"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["table_number_hit_rate"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["reference_context_recall"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["required_fact_coverage_macro"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["full_fact_coverage_rate"], 1.0)

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
                "required_facts": ["0.20"],
            }
        ]
        report = evaluate_document("all-documents", cases, chunks, [1, 2])

        self.assertEqual(report["aggregate"]["1"]["reference_context_recall"], 0.0)
        self.assertEqual(report["aggregate"]["2"]["reference_context_recall"], 1.0)
        self.assertEqual(report["aggregate"]["1"]["required_fact_coverage_macro"], 0.0)
        self.assertEqual(report["aggregate"]["2"]["required_fact_coverage_macro"], 1.0)

    def test_multifact_coverage_improves_from_partial_to_full_with_larger_k(self):
        chunks = [
            Chunk("The dataset contains 4,855 samples.", {"page": 1, "type": "text"}),
            Chunk("The split is 85%/10%/5%.", {"page": 2, "type": "text"}),
        ]
        cases = [
            {
                "case_id": "case-1",
                "question": "dataset samples split",
                "type": "method",
                "contexts": ["4,855 samples", "85%/10%/5%"],
                "source_pages": [1, 2],
                "required_facts": ["4,855", "85%/10%/5%"],
            }
        ]
        report = evaluate_document("paper", cases, chunks, [1, 2])
        top_one = report["cases_detail"][0]["metrics"]["1"]
        top_two = report["cases_detail"][0]["metrics"]["2"]

        self.assertEqual(top_one["fact_coverage_status"], "partial")
        self.assertEqual(top_one["required_fact_coverage"], 0.5)
        self.assertEqual(top_two["fact_coverage_status"], "full")
        self.assertEqual(top_two["required_fact_coverage"], 1.0)

        failures = fact_failure_lists(report["cases_detail"], [1, 2])
        self.assertEqual(failures["1"][0]["missing_facts"], ["85%/10%/5%"])
        self.assertEqual(failures["2"], [])


if __name__ == "__main__":
    unittest.main()
