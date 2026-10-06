import unittest
import contextlib
import io
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import app
from evaluation.paper_pilot import find_paper, main as paper_pilot_main, replay_request
from evaluation.generation_stability import (
    build_runtime,
    completed_keys,
    runtime_config_trace,
    select_cases,
)


class GenerationStabilityTests(unittest.TestCase):
    def test_find_paper_checks_supplied_directories_in_order(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            paper = Path(second) / "paper.pdf"
            paper.touch()
            self.assertEqual(find_paper("paper.pdf", [Path(first), Path(second)]), paper)
            with self.assertRaises(FileNotFoundError):
                find_paper("missing.pdf", [Path(first), Path(second)])

    def test_missing_pilot_cases_file_fails_before_model_or_database_setup(self):
        with TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.jsonl")
            stderr = io.StringIO()
            with patch("sys.argv", ["paper_pilot", "--papers-dir", directory,
                                    "--cases-file", missing]), \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                paper_pilot_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("cases file not found", stderr.getvalue())

    def test_missing_replay_trace_fails_instead_of_reporting_empty_success(self):
        with TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.jsonl")
            stderr = io.StringIO()
            with patch("sys.argv", ["paper_pilot", "--replay-trace", missing]), \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                paper_pilot_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("replay trace not found", stderr.getvalue())

    def test_replay_only_changes_model_and_explicit_thinking_control(self):
        original = {"model": "old", "messages": [{"role": "user", "content": "evidence"}],
                    "temperature": 0.3, "max_tokens": 1024}
        row = {"calls": [{"request": original}]}
        request = replay_request(row, "new", no_thinking=True)
        self.assertEqual(request, {**original, "model": "new", "reasoning_effort": "none"})
        self.assertEqual(original["model"], "old")
        self.assertNotIn("reasoning_effort", original)
        self.assertEqual(replay_request(row, "old"), original)

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
        self.assertEqual(trace["llm_model"], config.llm_model)
        self.assertNotIn("LLM_API_KEY", trace)
        self.assertNotIn("LLM_API_KEY", trace)
        self.assertNotIn("llm_base_url", trace)

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
