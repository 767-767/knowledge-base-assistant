import unittest
from unittest.mock import patch

import app
from evaluation.generation_stability import (
    answer_signature,
    build_runtime,
    completed_keys,
    runtime_config_trace,
    select_cases,
    source_fingerprint,
)


class GenerationStabilityTests(unittest.TestCase):
    def test_select_cases_preserves_manifest_order_and_rejects_unknown(self):
        cases = [
            {"case_id": "b", "question": "B"},
            {"case_id": "a", "question": "A"},
            {"case_id": "c", "question": "C"},
        ]

        selected = select_cases(cases, ["c", "a"])

        self.assertEqual([case["case_id"] for case in selected], ["a", "c"])
        with self.assertRaises(ValueError):
            select_cases(cases, ["missing"])

    def test_answer_signature_only_collapses_superficial_whitespace(self):
        self.assertEqual(answer_signature("  n = 25\n\n n = 38  "), "n = 25 n = 38")

    def test_completed_keys_rejects_legacy_rows_without_source_filter(self):
        sources = {"case-1": ["paper.pdf"]}
        rows = [
            {"repeat": 1, "case_id": "case-1", "error": False},
            {
                "repeat": 2,
                "case_id": "case-1",
                "error": False,
                "source_filter": ["paper.pdf"],
            },
        ]
        self.assertEqual(completed_keys(rows, sources), {(2, "case-1")})

    def test_completed_keys_accepts_no_source_filter_trace(self):
        rows = [
            {"repeat": 1, "case_id": "case-1", "error": False, "source_filter": None},
        ]
        self.assertEqual(completed_keys(rows, {"case-1": None}), {(1, "case-1")})

    def test_runtime_trace_contains_model_and_retrieval_settings_but_no_secret(self):
        config = app.RuntimeConfig(
            db_path="/private/tmp/example-db",
            retrieval_mode="hybrid",
            document_routing=True,
            query_decomposition=True,
            parent_window=True,
            spatial_figure_evidence=True,
            formula_evidence=True,
            hybrid_candidate_k=50,
            reranker_model="local-reranker",
            reranker_revision="fixed-revision",
        )
        runtime = type("RuntimeDouble", (), {"config": config})()
        trace = runtime_config_trace(runtime)
        self.assertEqual(trace["retrieval_mode"], "hybrid")
        self.assertEqual(trace["hybrid_candidate_k"], 50)
        self.assertEqual(trace["reranker_revision"], "fixed-revision")
        self.assertTrue(trace["formula_evidence"])
        self.assertTrue(trace["formula_evidence_auto"])
        self.assertNotIn("DEEPSEEK_API_KEY", trace)
        self.assertNotIn("deepseek_base_url", trace)

    def test_source_fingerprint_is_stable_hex(self):
        fingerprint = source_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        int(fingerprint, 16)

    def test_build_runtime_passes_explicit_reranker_and_formula_settings(self):
        base = app.RuntimeConfig(db_path="./base", retrieval_mode="dense")
        runtime = type("RuntimeDouble", (), {"config": base})()
        with patch.object(app.RuntimeConfig, "from_env", return_value=base), patch.object(
            app, "create_runtime", return_value=runtime
        ) as create_runtime:
            build_runtime(
                "/private/tmp/isolation",
                retrieval_mode="hybrid",
                document_routing=True,
                query_decomposition=True,
                parent_window=True,
                spatial_figure_evidence=True,
                formula_evidence=True,
                formula_evidence_auto=False,
                reranker_model="BAAI/bge-reranker-base",
                reranker_revision="fixed-revision",
                reranker_batch_size=4,
                reranker_max_length=256,
                reranker_device="cpu",
                reranker_rrf_k=30,
            )

        config = create_runtime.call_args.args[0]
        self.assertEqual(config.db_path, "/private/tmp/isolation")
        self.assertEqual(config.retrieval_mode, "hybrid")
        self.assertEqual(config.reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(config.reranker_revision, "fixed-revision")
        self.assertEqual(config.reranker_batch_size, 4)
        self.assertEqual(config.reranker_max_length, 256)
        self.assertEqual(config.reranker_rrf_k, 30)
        self.assertTrue(config.formula_evidence)
        self.assertFalse(config.formula_evidence_auto)


if __name__ == "__main__":
    unittest.main()
