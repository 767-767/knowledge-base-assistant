import unittest

from sci_rag_reranking import reranker_document_text
from sci_rag_retrieval import (
    BM25Index,
    DocumentRouter,
    RankedItem,
    ensure_source_coverage,
    query_variants,
    reciprocal_rank_fusion,
)


class RetrievalTests(unittest.TestCase):
    def test_ensure_source_coverage_promotes_missing_routed_source(self):
        order = ensure_source_coverage(
            [0, 1, 2],
            [
                {"source": "paper-a"},
                {"source": "paper-a"},
                {"source": "paper-b"},
            ],
            ["paper-a", "paper-b"],
            2,
        )
        self.assertEqual(order[:2], [0, 2])

    def test_ensure_source_coverage_prefers_clause_candidate(self):
        order = ensure_source_coverage(
            [0, 1, 2],
            [
                {"source": "paper-a"},
                {"source": "paper-a"},
                {"source": "paper-b"},
            ],
            ["paper-a", "paper-b"],
            2,
            preferred_order=[1, 2],
        )
        self.assertEqual(order[:2], [1, 2])

    def test_ensure_source_coverage_keeps_each_preferred_source_in_prefix(self):
        order = ensure_source_coverage(
            [0, 1, 2, 3],
            [
                {"source": "paper-a"},
                {"source": "paper-a"},
                {"source": "paper-b"},
                {"source": "paper-b"},
            ],
            ["paper-a", "paper-b"],
            2,
            preferred_order=[1, 3],
        )
        self.assertEqual(order[:2], [1, 3])

    def test_query_variants_keep_original_and_split_composite_clauses(self):
        question = "MgNO 的二维椭圆 PDE 定义在哪个区域，并考虑哪些边界条件？"
        variants = query_variants(question)

        self.assertEqual(variants[0], question)
        self.assertIn("MgNO 的二维椭圆 PDE 定义在哪个区域", variants)
        self.assertIn("并考虑哪些边界条件", variants)

    def test_query_variants_do_not_split_short_or_empty_queries(self):
        self.assertEqual(query_variants("短题"), [])
        self.assertEqual(query_variants(""), [])
        self.assertEqual(query_variants("What is RAG?", max_variants=1), ["What is RAG?"])

    def test_query_variants_do_not_duplicate_question_without_real_clause_split(self):
        question = "论文提出的后续研究方向包括哪些内容？"
        self.assertEqual(query_variants(question), [question])

    def test_query_variants_keep_thousands_separators_inside_a_clause(self):
        variants = query_variants("MgNO 有 1,280 个样本，SciDQA 审阅了 7,000 个实例？")
        self.assertIn("MgNO 有 1,280 个样本", variants)
        self.assertIn("SciDQA 审阅了 7,000 个实例", variants)

    def test_query_variants_split_multiple_question_clauses(self):
        variants = query_variants(
            "使用什么代理模型？每个表格集合生成多少个问题？包含哪三种方法？"
        )
        self.assertEqual(len(variants), 4)
        self.assertIn("使用什么代理模型", variants)
        self.assertIn("每个表格集合生成多少个问题", variants)
        self.assertIn("包含哪三种方法", variants)

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

    def test_weighted_rrf_scales_a_ranked_list(self):
        equal = reciprocal_rank_fusion([["ce", "shared"], ["shared", "original"]], rrf_k=10)
        weighted = reciprocal_rank_fusion(
            [["ce", "shared"], ["shared", "original"]],
            rrf_k=10,
            weights=[20.0, 1.0],
        )

        self.assertEqual(equal[0].key, "shared")
        self.assertEqual(weighted[0].key, "ce")

    def test_weighted_rrf_validates_weights(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[1], [2]], weights=[1.0])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[1]], weights=[0.0])

    def test_bm25_prefers_matching_table_and_is_deterministic(self):
        documents = [
            reranker_document_text("Background about optimization."),
            reranker_document_text(
                "|Model|Score|\n|---|---|\n|DrugR*|0.2060|",
                {"table_caption": "Table 2: Results"},
            ),
            reranker_document_text(
                "|Model|Score|\n|---|---|\n|DrugR|0.2712|",
                {"table_caption": "Table 1: Results"},
            ),
        ]
        index = BM25Index(documents)

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

    def test_document_router_only_routes_unique_distinctive_terms(self):
        router = DocumentRouter(
            ["paper-a", "paper-b"],
            [
                "DrugR molecular optimization and explicit reasoning",
                "AlphaFold3 biomolecular structure prediction",
            ],
        )

        route = router.route("DrugR 的显式推理数据集包含多少样本？")
        self.assertIsNotNone(route)
        self.assertEqual(route.document_id, "paper-a")
        self.assertIn("drugr", route.distinctive_tokens)
        self.assertIsNone(router.route("方法的性能是多少？"))
        self.assertIsNone(router.route("DrugR AlphaFold3 的差异是什么？"))

    def test_document_router_recovers_three_letter_uppercase_source_id(self):
        router = DocumentRouter(
            ["uda.pdf", "other.pdf"],
            ["UDA dataset and table retrieval", "other paper results"],
        )

        route = router.route("UDA 在结论中明确指出哪些限制？")
        self.assertIsNotNone(route)
        self.assertEqual(route.document_id, "uda.pdf")
        self.assertIsNone(router.route("方法的性能是多少？"))

    def test_document_router_ignores_unique_generic_terms(self):
        router = DocumentRouter(
            ["paper-a", "paper-b"],
            ["method results and paper overview", "model results and paper overview"],
        )

        self.assertIsNone(router.route("What is the method?"))
        self.assertIsNone(router.route("Which paper reports the results?"))


if __name__ == "__main__":
    unittest.main()
