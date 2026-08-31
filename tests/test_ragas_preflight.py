import unittest

from evaluation.ragas_preflight import preflight_report


class RagasPreflightTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"case_id": "case-1", "question": "q", "ground_truth": "a"},
        ]

    def _report(self):
        return {
            "meta": {"num_test_cases": 1, "judge_model": "judge", "generation_model": "generator", "embedding_model": "embed"},
            "summary": {
                "nv_context_relevance": {"mean": 1.0, "count_valid": 1, "count_total": 1, "scores": [1.0]},
                "faithfulness": {"mean": 0.5, "count_valid": 1, "count_total": 1, "scores": [0.5]},
                "answer_relevancy": {"mean": 0.8, "count_valid": 1, "count_total": 1, "scores": [0.8]},
            },
            "results": [
                {
                    "id": "case-1",
                    "question": "q",
                    "ground_truth": "a",
                    "answer": "a",
                    "num_contexts_evaluated": 1,
                    "generation_contexts": ["evidence", "extra"],
                    "generation_context_ids": ["chunk-1", "chunk-2"],
                    "generation_context_metadatas": [{"page": 1}, {"page": 2}],
                    "evaluated_contexts": ["evidence"],
                    "context_ids": ["chunk-1"],
                    "context_metadatas": [{"page": 1}],
                    "reference_contexts": ["gold"],
                    "scores": {"faithfulness": 0.5},
                }
            ],
        }

    def test_consistent_report_is_artifact_ready_but_claims_remain_bounded(self):
        result = preflight_report(self._report(), self.cases, require_complete=True, require_trace=True)
        self.assertTrue(result["ready_for_interpretation"])
        self.assertTrue(result["checks"]["trace_lengths_consistent"])
        self.assertEqual(result["claim_boundaries"]["ground_truth_metric_usage"], "not_provable_from_saved_report")

    def test_recorded_metric_inputs_show_ground_truth_is_not_used(self):
        report = self._report()
        report["meta"]["ragas_metric_input_columns"] = {
            "nv_context_relevance": ["user_input", "retrieved_contexts"],
            "faithfulness": ["user_input", "response", "retrieved_contexts"],
            "answer_relevancy": ["user_input", "response"],
        }
        result = preflight_report(report, self.cases)
        self.assertTrue(result["checks"]["ragas_metric_input_columns_recorded"])
        self.assertFalse(result["checks"]["ground_truth_required_by_any_ragas_metric"])
        self.assertEqual(
            result["claim_boundaries"]["ground_truth_metric_usage"],
            "not_used_by_declared_metrics",
        )

    def test_missing_trace_is_warning_unless_required(self):
        report = self._report()
        report["results"][0].pop("evaluated_contexts")
        result = preflight_report(report, self.cases, require_complete=True)
        self.assertTrue(result["ready_for_interpretation"])
        self.assertTrue(result["warnings"])
        strict = preflight_report(report, self.cases, require_complete=True, require_trace=True)
        self.assertFalse(strict["ready_for_interpretation"])

    def test_generation_and_evaluated_trace_must_align(self):
        report = self._report()
        report["results"][0]["generation_contexts"] = ["different"]
        result = preflight_report(report, self.cases)
        self.assertFalse(result["ready_for_interpretation"])
        self.assertTrue(any("前缀" in issue for issue in result["errors"]))

    def test_invalid_metric_mean_is_blocking(self):
        report = self._report()
        report["summary"]["faithfulness"]["mean"] = 0.9
        result = preflight_report(report, self.cases)
        self.assertFalse(result["ready_for_interpretation"])
        self.assertTrue(any("mean" in issue for issue in result["errors"]))

    def test_non_numeric_metric_values_are_reported_not_crashed(self):
        report = self._report()
        report["summary"]["faithfulness"]["scores"] = ["not-a-score"]
        result = preflight_report(report, self.cases)
        self.assertFalse(result["ready_for_interpretation"])
        self.assertTrue(any("不是数值" in issue for issue in result["errors"]))

    def test_id_mismatch_is_blocking(self):
        report = self._report()
        report["results"][0]["id"] = "other"
        result = preflight_report(report, self.cases, require_complete=True)
        self.assertFalse(result["ready_for_interpretation"])
        self.assertTrue(any("ID" in issue for issue in result["errors"]))


if __name__ == "__main__":
    unittest.main()
