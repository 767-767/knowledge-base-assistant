import unittest

from evaluation.audit_generation_trace import audit_generation_trace


class GenerationTraceAuditTests(unittest.TestCase):
    def test_separates_stable_context_from_variable_answer(self):
        rows = [
            {
                "case_id": "case-1",
                "repeat": 1,
                "answer": "答案 A",
                "context_ids": ["c1", "c2"],
                "context_metadatas": [{"page": 1}],
                "runtime_config": {"retrieval_mode": "hybrid"},
                "source_fingerprint": "abc",
            },
            {
                "case_id": "case-1",
                "repeat": 2,
                "answer": "答案 B",
                "context_ids": ["c1", "c2"],
                "context_metadatas": [{"page": 1}],
                "runtime_config": {"retrieval_mode": "hybrid"},
                "source_fingerprint": "abc",
            },
        ]
        report = audit_generation_trace(rows)
        case = report["cases"][0]
        self.assertTrue(case["configuration_stable"])
        self.assertTrue(case["context_stable"])
        self.assertFalse(case["answer_exactly_stable"])
        self.assertEqual(report["aggregate"]["answer_variation_cases"], 1)

    def test_marks_legacy_trace_without_provenance(self):
        report = audit_generation_trace(
            [{"case_id": "legacy", "repeat": 1, "answer": "ok", "contexts": ["x"]}]
        )
        self.assertEqual(report["aggregate"]["provenance_incomplete_rows"], 1)
        self.assertFalse(report["cases"][0]["provenance_complete"])
        self.assertIsNone(report["aggregate"]["context_stable_rate"])


if __name__ == "__main__":
    unittest.main()
