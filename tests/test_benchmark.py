import unittest
import hashlib
import json
from pathlib import Path
import tempfile

from evaluation.benchmark_loader import (
    BenchmarkValidationError,
    benchmark_summary,
    load_benchmark,
)
from evaluation.context_coverage import unsupported_gold_facts
from sci_rag_core import is_derived_value_question


class BenchmarkLoaderTests(unittest.TestCase):
    def test_multipaper_benchmark_resolves_seed_and_inline_cases(self):
        benchmark = load_benchmark("evaluation/benchmark/manifest.json")
        summary = benchmark_summary(benchmark)
        self.assertEqual(summary["documents"], 5)
        self.assertEqual(summary["cases"], 53)
        self.assertTrue(summary["complete"])
        case = next(case for case in benchmark["cases"] if case["case_id"] == "drugr-02")
        self.assertEqual(case["document_id"], "drugr-2602-08213-v1")
        self.assertIn("0.2060", case["ground_truth"])
        self.assertTrue(case["contexts"])
        case = next(case for case in benchmark["cases"] if case["case_id"] == "af3-08")
        self.assertEqual(case["document_id"], "alphafold3-nature-2024")
        self.assertEqual(case["required_facts"], ["pLDDT", "PAE", "PDE"])
        self.assertTrue(all(case.get("required_facts") for case in benchmark["cases"]))
        self.assertTrue(all(not unsupported_gold_facts(case) for case in benchmark["cases"]))

    def test_expanded_manifest_merges_base_benchmark_without_changing_default(self):
        base = load_benchmark("evaluation/benchmark/manifest.json")
        expanded = load_benchmark("evaluation/benchmark/manifest_expanded.json")
        self.assertEqual(benchmark_summary(base)["documents"], 5)
        self.assertEqual(benchmark_summary(base)["cases"], 53)
        self.assertEqual(benchmark_summary(expanded)["documents"], 6)
        self.assertEqual(benchmark_summary(expanded)["cases"], 66)
        self.assertEqual(
            benchmark_summary(expanded)["cases_per_document"]["thinknote-eacl-2026"],
            13,
        )
        self.assertTrue(
            all(not unsupported_gold_facts(case) for case in expanded["cases"])
        )

    def test_challenge_manifest_adds_targeted_missing_case_types(self):
        challenge = load_benchmark("evaluation/benchmark/manifest_challenge.json")
        summary = benchmark_summary(challenge)
        self.assertEqual(summary["documents"], 6)
        self.assertEqual(summary["cases"], 101)
        self.assertEqual(
            sum(case.get("challenge_type") == "image_only" for case in challenge["cases"]),
            10,
        )
        self.assertEqual(
            sum(case.get("challenge_type") == "computation" for case in challenge["cases"]),
            20,
        )
        cross_cases = [
            case for case in challenge["cases"] if case.get("challenge_type") == "cross_document"
        ]
        self.assertEqual(len(cross_cases), 5)
        self.assertTrue(all(case.get("additional_document_ids") for case in cross_cases))
        self.assertTrue(all(not unsupported_gold_facts(case) for case in challenge["cases"]))
        drugr = next(case for case in challenge["cases"] if case["case_id"] == "calc-drugr-04")
        self.assertEqual(drugr["calculation"]["operation"], "3863 - 1117")
        self.assertEqual(drugr["calculation"]["expected_result"], "2,746")
        self.assertNotIn("2,746", " ".join(drugr["contexts"]))
        scidqa = next(case for case in challenge["cases"] if case["case_id"] == "calc-scidqa-04")
        self.assertIn("保留两位小数", scidqa["question"])
        self.assertEqual(scidqa["ground_truth"], "41.96%。")
        for case in challenge["cases"]:
            if case.get("challenge_type") == "computation":
                expected = case["calculation"]["expected_result"]
                self.assertNotIn(expected, " ".join(case["contexts"]))
                self.assertTrue(is_derived_value_question(case["question"]))

    def test_generalization_manifest_merges_new_holdout_cases(self):
        benchmark = load_benchmark("evaluation/benchmark/manifest_generalization.json")
        summary = benchmark_summary(benchmark)
        self.assertEqual(summary["documents"], 8)
        self.assertEqual(summary["cases"], 82)
        self.assertEqual(summary["cases_per_document"]["tanq-tacl-2025"], 9)
        self.assertEqual(summary["cases_per_document"]["figex-emnlp-2025"], 7)
        cross_cases = [
            case for case in benchmark["cases"] if case.get("challenge_type") == "cross_document"
        ]
        self.assertEqual(len(cross_cases), 2)
        self.assertTrue(all(case.get("additional_document_ids") for case in cross_cases))
        image_cases = [
            case for case in benchmark["cases"] if case.get("challenge_type") == "image_only"
        ]
        self.assertEqual(len(image_cases), 1)
        self.assertTrue(all(not unsupported_gold_facts(case) for case in benchmark["cases"]))

    def test_loader_rejects_alias_for_an_unknown_required_fact(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            manifest = {
                "schema_version": 1,
                "benchmark_id": "test",
                "cases_path": "cases.jsonl",
                "documents": [
                    {"document_id": "paper", "filename": "paper.pdf", "sha256": "0" * 64}
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "document_id": "paper",
                        "question": "q",
                        "ground_truth": "a",
                        "contexts": ["alpha"],
                        "required_facts": ["alpha"],
                        "required_fact_aliases": {"beta": ["b"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(BenchmarkValidationError):
                load_benchmark(root / "manifest.json")

    def test_verify_files_accepts_multiple_external_directories(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            seed_dir = root / "seed"
            extra_dir = root / "extra"
            seed_dir.mkdir()
            extra_dir.mkdir()
            seed = b"seed pdf placeholder"
            extra = b"extra pdf placeholder"
            (seed_dir / "seed.pdf").write_bytes(seed)
            (extra_dir / "extra.pdf").write_bytes(extra)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            manifest = {
                "schema_version": 1,
                "benchmark_id": "test",
                "minimum_documents": 2,
                "cases_path": "cases.jsonl",
                "documents": [
                    {
                        "document_id": "seed",
                        "filename": "seed.pdf",
                        "sha256": hashlib.sha256(seed).hexdigest(),
                    },
                    {
                        "document_id": "extra",
                        "filename": "extra.pdf",
                        "sha256": hashlib.sha256(extra).hexdigest(),
                    },
                ],
            }
            (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (manifest_dir / "cases.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "document_id": "seed",
                        "question": "q",
                        "ground_truth": "a",
                        "contexts": ["a"],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "case_id": "case-2",
                        "document_id": "extra",
                        "question": "q",
                        "ground_truth": "a",
                        "contexts": ["a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark = load_benchmark(
                manifest_dir / "manifest.json",
                papers_dir=[seed_dir, extra_dir],
                verify_files=True,
            )
            self.assertTrue(benchmark_summary(benchmark)["complete"])


if __name__ == "__main__":
    unittest.main()
