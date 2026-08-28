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
