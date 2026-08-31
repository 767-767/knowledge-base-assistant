import unittest

from evaluation.benchmark_retrieval import (
    case_provenance,
    evaluate_document,
    fact_failure_lists,
    HybridRetriever,
    searchable_text,
    _adjacent_context_indices,
    _parent_window_scoring_chunks,
    _section_expansion_indices,
    _structured_figure_guard_indices,
    _formula_guard_indices,
    _structured_table_guard_indices,
)
from sci_rag_reranking import CrossEncoderReranker
from sci_rag_retrieval import (
    BM25Index,
    DocumentRoute,
    DocumentRouter,
    RankedItem,
    query_variants,
    reciprocal_rank_fusion,
)
from sci_rag_core import Chunk


class BenchmarkRetrievalTests(unittest.TestCase):
    def test_query_variants_keep_original_and_split_composite_clauses(self):
        variants = query_variants(
            "MgNO 的二维椭圆 PDE 定义在哪个区域，并考虑哪些边界条件？"
        )
        self.assertEqual(variants[0], "MgNO 的二维椭圆 PDE 定义在哪个区域，并考虑哪些边界条件？")
        self.assertIn("MgNO 的二维椭圆 PDE 定义在哪个区域", variants)
        self.assertIn("并考虑哪些边界条件", variants)

    def test_query_variants_do_not_split_short_or_empty_queries(self):
        self.assertEqual(query_variants("短题"), [])
        self.assertEqual(query_variants(""), [])
        self.assertEqual(query_variants("What is RAG?", max_variants=1), ["What is RAG?"])

    def test_query_variants_do_not_duplicate_question_without_real_clause_split(self):
        question = "论文提出的后续研究方向包括哪些内容？"
        self.assertEqual(query_variants(question), [question])

    def test_query_decomposition_fuses_variant_rankings_with_fixed_route(self):
        chunks = [
            Chunk("DrugR overview", {"source": "drugr.pdf"}),
            Chunk("DrugR explicit dataset has 4855 samples", {"source": "drugr.pdf"}),
            Chunk("AlphaFold structure", {"source": "af3.pdf"}),
        ]
        retriever = HybridRetriever(
            chunks,
            mode="bm25",
            document_routing=True,
            query_decomposition=True,
        )
        result = retriever.retrieve("DrugR dataset samples, pipeline", 2)
        self.assertEqual([item.key for item in result], [1, 0])

    def test_structured_table_guard_uses_exact_cell_inside_existing_route(self):
        other_table = """|Model|RAG|FT|
|---|---|---|
|GPT-4o|1.00|2.00|"""
        routed_table = """|Model|RAG|FT|
|---|---|---|
|GPT-4o|46.63|54.03|"""
        chunks = [
            Chunk(
                other_table,
                {
                    "type": "table",
                    "table_number": 3,
                    "benchmark_document_id": "other-paper",
                },
            ),
            Chunk("retrieved prose", {"benchmark_document_id": "scidqa-paper"}),
            Chunk(
                routed_table,
                {
                    "type": "table",
                    "table_number": 3,
                    "benchmark_document_id": "scidqa-paper",
                },
            ),
        ]

        guarded = _structured_table_guard_indices(
            "Table 3 中 GPT-4o 在 RAG 和 full-text 配置下的 Avg 分别是多少？",
            [RankedItem(1, 1.0), RankedItem(0, 0.5)],
            chunks,
            route=DocumentRoute("scidqa-paper", ("full-text",)),
        )

        self.assertEqual(guarded, [2])

    def test_structured_figure_guard_uses_exact_figure_inside_existing_route(self):
        chunks = [
            Chunk(
                "Other paper Figure 1 n = 99",
                {
                    "type": "figure",
                    "figure_kind": "figure",
                    "figure_number": 1,
                    "benchmark_document_id": "other-paper",
                },
            ),
            Chunk(
                "normal retrieved prose",
                {"type": "text", "benchmark_document_id": "alphafold3"},
            ),
            Chunk(
                "Figure 1 CASP15 RNA n = 8",
                {
                    "type": "figure",
                    "figure_kind": "figure",
                    "figure_number": 1,
                    "benchmark_document_id": "alphafold3",
                },
            ),
            Chunk(
                "Extended Data Figure 1",
                {
                    "type": "figure",
                    "figure_kind": "extended_data_figure",
                    "figure_number": 1,
                    "benchmark_document_id": "alphafold3",
                },
            ),
            Chunk(
                "<!-- Start of picture text --> Figure 1 n = 25 n = 38 n = 8 n = 28",
                {"type": "text", "benchmark_document_id": "alphafold3"},
            ),
        ]

        guarded = _structured_figure_guard_indices(
            "Figure 1 中 CASP15 RNA 的样本数是多少？",
            [
                RankedItem(0, 1.0),
                RankedItem(1, 0.9),
                RankedItem(3, 0.8),
                RankedItem(4, 0.7),
            ],
            chunks,
            route=DocumentRoute("alphafold3", ("casp15",)),
        )

        self.assertEqual(guarded, [2, 1])

    def test_formula_guard_promotes_formula_text_without_cross_source_leak(self):
        chunks = [
            Chunk(
                "A general paragraph",
                {"source": "paper.pdf", "benchmark_document_id": "paper-a", "type": "text"},
            ),
            Chunk(
                "With linear FEM, the elliptic PDE system is A*u=f; kernel dimensions 3 × 3.",
                {"source": "paper.pdf", "benchmark_document_id": "paper-a", "type": "text"},
            ),
            Chunk(
                "Other paper equation x = 99.",
                {"source": "other.pdf", "benchmark_document_id": "paper-b", "type": "text"},
            ),
        ]
        route = DocumentRoute("paper-a", ("mgno",))
        guarded = _formula_guard_indices(
            "线性有限元离散后的椭圆 PDE 系统写成什么形式，卷积核尺寸是多少？",
            [RankedItem(0, 1.0)],
            chunks,
            route=route,
        )
        self.assertEqual(guarded[0], 1)
        self.assertNotIn(2, guarded[:2])

    def test_formula_guard_skips_ambiguous_multi_source_query_without_route(self):
        chunks = [
            Chunk(
                "The elliptic PDE system is A*u=f; kernel dimensions 3 × 3.",
                {"source": "paper-a.pdf", "benchmark_document_id": "paper-a", "type": "text"},
            ),
            Chunk(
                "Another paper equation B*v=g; kernel dimensions 5 × 5.",
                {"source": "paper-b.pdf", "benchmark_document_id": "paper-b", "type": "text"},
            ),
        ]
        guarded = _formula_guard_indices(
            "线性有限元离散后的椭圆 PDE 系统写成什么形式，卷积核尺寸是多少？",
            [RankedItem(0, 1.0)],
            chunks,
            route=None,
        )
        self.assertEqual(guarded, [0])

    def test_spatial_figure_channel_does_not_compete_in_normal_retrieval(self):
        chunks = [
            Chunk(
                "ordinary query repeated ordinary query n = 99",
                {"type": "figure", "figure_kind": "figure", "figure_number": 1},
            ),
            Chunk("ordinary query answer is 42", {"type": "text"}),
        ]

        report = evaluate_document(
            "paper",
            [
                {
                    "case_id": "ordinary",
                    "question": "ordinary query",
                    "contexts": ["ordinary query answer is 42"],
                    "required_facts": ["42"],
                }
            ],
            chunks,
            [1],
            structured_figure_guard=True,
        )

        detail = report["cases_detail"][0]
        self.assertEqual(detail["top_results"][0]["chunk_index"], 1)
        self.assertEqual(detail["metrics"]["1"]["fact_coverage_status"], "full")

    def test_adjacent_context_stays_on_same_source_page_and_skips_tables(self):
        chunks = [
            Chunk("previous", {"source": "paper.pdf", "page": 2, "type": "text"}),
            Chunk("anchor", {"source": "paper.pdf", "page": 2, "type": "text"}),
            Chunk("table", {"source": "paper.pdf", "page": 2, "type": "table"}),
            Chunk("other source", {"source": "other.pdf", "page": 2, "type": "text"}),
            Chunk("second anchor", {"source": "paper.pdf", "page": 3, "type": "text"}),
            Chunk("next page peer", {"source": "paper.pdf", "page": 3, "type": "text"}),
        ]

        expanded = _adjacent_context_indices(
            [RankedItem(1, 1.0), RankedItem(4, 0.9), RankedItem(3, 0.8)],
            chunks,
        )

        self.assertEqual(expanded, [1, 0, 4, 5, 3])

    def test_parent_window_enriches_text_without_changing_rank_slots(self):
        chunks = [
            Chunk("numeric detail 42", {"source": "paper.pdf", "page": 2, "type": "text"}),
            Chunk("method anchor", {"source": "paper.pdf", "page": 2, "type": "text"}),
            Chunk("already selected", {"source": "paper.pdf", "page": 2, "type": "text"}),
            Chunk("other page", {"source": "paper.pdf", "page": 3, "type": "text"}),
        ]
        ranked = [RankedItem(1, 1.0), RankedItem(2, 0.9)]

        effective, expansions = _parent_window_scoring_chunks(ranked, chunks)

        self.assertEqual([item.key for item in ranked], [1, 2])
        self.assertEqual(expansions, {1: (0, 1)})
        self.assertIn("numeric detail 42", effective[1].page_content)
        self.assertEqual(effective[2].page_content, "already selected")

    def test_parent_window_fact_coverage_uses_effective_context(self):
        chunks = [
            Chunk("The kernel has size 3×3.", {"source": "paper.pdf", "page": 4, "type": "text"}),
            Chunk("The finite-element system is A*u=f.", {"source": "paper.pdf", "page": 4, "type": "text"}),
        ]
        cases = [
            {
                "case_id": "window-case",
                "question": "What is the finite-element system equation?",
                "contexts": ["The finite-element system is A*u=f. The kernel has size 3×3."],
                "required_facts": ["A", "u", "f", "3×3"],
                "source_pages": [4],
            }
        ]

        report = evaluate_document(
            "paper",
            cases,
            chunks,
            [1],
            parent_window=True,
        )

        detail = report["cases_detail"][0]
        self.assertEqual(detail["top_results"][0]["chunk_index"], 1)
        self.assertEqual(detail["top_results"][0]["window_chunk_indices"], [0, 1])
        self.assertEqual(detail["metrics"]["1"]["fact_coverage_status"], "full")

    def test_multi_k_metrics_do_not_depend_on_larger_requested_k(self):
        # The neighbor carries the fact but is ranked third.  At @2 it is
        # available through parent/window enrichment; it is already selected
        # at @3 and therefore must not be silently injected into the @2
        # metric merely because @3 was requested in the same command.
        chunks = [
            Chunk(
                "question anchor",
                {"source": "paper.pdf", "page": 1, "chunk_index": 0, "type": "text"},
            ),
            Chunk(
                "answer 42",
                {"source": "paper.pdf", "page": 1, "chunk_index": 1, "type": "text"},
            ),
            Chunk(
                "question distractor",
                {"source": "paper.pdf", "page": 1, "chunk_index": 2, "type": "text"},
            ),
        ]
        case = {
            "case_id": "multi-k",
            "question": "question anchor distractor",
            "contexts": ["answer 42"],
            "required_facts": ["42"],
        }
        single = evaluate_document(
            "paper", [case], chunks, [2], parent_window=True
        )["cases_detail"][0]["metrics"]["2"]
        combined = evaluate_document(
            "paper", [case], chunks, [2, 3], parent_window=True
        )["cases_detail"][0]["metrics"]["2"]
        self.assertEqual(combined, single)

    def test_parent_window_does_not_expand_picture_text_blocks(self):
        chunks = [
            Chunk(
                "<!-- Start of picture text --> Figure 1 n=25 n=38 n=8",
                {"source": "paper.pdf", "page": 2, "chunk_index": 1, "type": "text"},
            ),
            Chunk(
                "Adjacent prose with unrelated sample counts.",
                {"source": "paper.pdf", "page": 2, "chunk_index": 2, "type": "text"},
            ),
        ]
        effective, expansions = _parent_window_scoring_chunks(
            [RankedItem(0, 1.0)], chunks
        )

        self.assertEqual(effective[0].page_content, chunks[0].page_content)
        self.assertEqual(expansions, {})
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

    def test_section_expansion_adds_same_header_chunks_after_candidate(self):
        chunks = [
            Chunk("section overview", {"source": "paper.pdf", "headers": "H2: Pipeline"}),
            Chunk("missing sibling fact", {"source": "paper.pdf", "headers": "H2: Pipeline"}),
            Chunk("other section", {"source": "paper.pdf", "headers": "H2: Results"}),
        ]
        expanded = _section_expansion_indices(
            "What is the dataset pipeline and its steps?",
            [RankedItem(0, 1.0)],
            chunks,
        )

        self.assertEqual(expanded[:2], [0, 1])
        mixed = _section_expansion_indices(
            "What is the dataset pipeline and its steps?",
            [RankedItem(0, 1.0)],
            [*chunks, Chunk("same heading from another paper", {"source": "other.pdf", "headers": "H2: Pipeline"})],
        )
        self.assertEqual(mixed, [0])

    def test_section_expansion_can_use_a_unique_route_in_multi_source_corpus(self):
        chunks = [
            Chunk("experimental setup overview", {"source": "paper.pdf", "headers": "H2: Experimental Setup"}),
            Chunk("closed-book, title-abs, RAG, full-text", {"source": "paper.pdf", "headers": "H2: Experimental Setup"}),
            Chunk("same heading from another paper", {"source": "other.pdf", "headers": "H2: Experimental Setup"}),
        ]
        expanded = _section_expansion_indices(
            "SciDQA 的四种实验配置是什么？",
            [RankedItem(0, 1.0)],
            chunks,
            route_source="paper.pdf",
        )
        self.assertEqual(expanded[:2], [0, 1])

    def test_section_expansion_adds_headerless_text_continuations_until_heading(self):
        chunks = [
            Chunk("overview", {"source": "paper.pdf", "headers": "H2: Pipeline", "type": "text", "chunk_index": 10}),
            Chunk("table noise", {"source": "paper.pdf", "type": "table", "chunk_index": 11}),
            Chunk("tool=ADMETLab", {"source": "paper.pdf", "type": "text", "chunk_index": 12}),
            Chunk("similarity > 0.6", {"source": "paper.pdf", "type": "text", "chunk_index": 13}),
            Chunk("next section", {"source": "paper.pdf", "headers": "H2: Results", "type": "text", "chunk_index": 14}),
        ]
        expanded = _section_expansion_indices(
            "What is the dataset pipeline and its steps?",
            [RankedItem(0, 1.0)],
            chunks,
        )
        self.assertEqual(expanded[:3], [0, 2, 3])
        self.assertNotIn(1, expanded[:4])
        self.assertNotIn(4, expanded[:4])

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

    def test_document_router_ignores_unique_generic_terms(self):
        router = DocumentRouter(
            ["paper-a", "paper-b"],
            ["method results and paper overview", "model results and paper overview"],
        )

        self.assertIsNone(router.route("What is the method?"))
        self.assertIsNone(router.route("Which paper reports the results?"))

    def test_opt_in_document_routing_restricts_bm25_candidates(self):
        chunks = [
            Chunk("DrugR reports an explicit reasoning dataset.", {"source": "drugr.pdf"}),
            Chunk("AlphaFold3 predicts biomolecular structures.", {"source": "af3.pdf"}),
        ]
        retriever = HybridRetriever(chunks, mode="bm25", document_routing=True)
        route = retriever.route("DrugR dataset")
        self.assertIsNotNone(route)
        self.assertEqual(route.document_id, "drugr.pdf")
        self.assertEqual(retriever.retrieve("DrugR dataset", 2)[0].key, 0)

    def test_evaluate_reports_document_routing_decision(self):
        chunks = [
            Chunk(
                "DrugR reports an explicit reasoning dataset.",
                {"source": "drugr.pdf", "benchmark_document_id": "drugr"},
            ),
            Chunk(
                "AlphaFold3 predicts biomolecular structures.",
                {"source": "af3.pdf", "benchmark_document_id": "af3"},
            ),
        ]
        report = evaluate_document(
            "all-documents",
            [
                {
                    "case_id": "drugr-1",
                    "document_id": "drugr",
                    "question": "DrugR dataset",
                    "contexts": ["DrugR reports an explicit reasoning dataset."],
                    "required_facts": ["DrugR"],
                }
            ],
            chunks,
            [1],
            retriever="bm25",
            document_routing=True,
        )
        detail = report["cases_detail"][0]
        self.assertEqual(detail["routing"]["selected_document"], "drugr")
        self.assertEqual(detail["routing"]["distinctive_tokens"], ["drugr"])

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

    def test_provenance_separates_wrong_document_and_reference_matches(self):
        chunks = [
            Chunk(
                "The answer is 42 in a cited paper.",
                {
                    "source": "paper-b.pdf",
                    "page": 1,
                    "headers": "H1: References",
                    "benchmark_document_id": "paper-b",
                },
            ),
            Chunk(
                "The answer is 42 in the target results.",
                {
                    "source": "paper-a.pdf",
                    "page": 3,
                    "headers": "H1: Results",
                    "benchmark_document_id": "paper-a",
                },
            ),
        ]
        case = {
            "case_id": "case-a",
            "document_id": "paper-a",
            "required_facts": ["42"],
            "source_pages": [3],
        }

        top_one = case_provenance(case, chunks, [RankedItem(0, 1.0)], 1)
        self.assertEqual(top_one["wrong_document_only_fact_count"], 1)
        self.assertEqual(top_one["reference_section_fact_count"], 1)
        self.assertEqual(top_one["gold_page_fact_count"], 0)
        self.assertEqual(top_one["fact_matches"]["42"][0]["flags"], [
            "wrong_document",
            "outside_gold_page",
            "reference_section",
        ])

        top_two = case_provenance(
            case,
            chunks,
            [RankedItem(0, 1.0), RankedItem(1, 0.5)],
            2,
        )
        self.assertEqual(top_two["wrong_document_only_fact_count"], 0)
        self.assertEqual(top_two["target_document_fact_count"], 1)
        self.assertEqual(top_two["gold_page_fact_count"], 1)

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
