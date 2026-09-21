import unittest

from evaluation.ground_truth_audit import audit_ground_truth


class GroundTruthAuditTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "case-1",
            "document_id": "paper-a",
            "type": "fact",
            "question": "How many samples and which model?",
            "ground_truth": "10 samples using Model-X.",
            "required_facts": ["10", "Model-X"],
            "contexts": ["The dataset contains 10 samples and uses Model-X."],
        }

    def test_report_separates_fact_context_and_surface_metrics(self):
        report = audit_ground_truth(
            [self.case],
            [
                {
                    "case_id": "case-1",
                    "answer": "There are 10 samples using Model-X.",
                    "contexts": ["The dataset contains 10 samples and uses Model-X."],
                }
            ],
            require_all=True,
        )
        self.assertEqual(report["summary"]["answer_fact_coverage"]["full_fact_coverage_rate"], 1.0)
        self.assertEqual(report["summary"]["gold_context_recall"]["mean"], 1.0)
        self.assertEqual(report["summary"]["ground_truth_exact_match"]["rate"], 0.0)
        self.assertIn("不能证明语义正确", report["claim_boundaries"]["answer_fact_coverage"])

    def test_missing_answers_are_explicit(self):
        report = audit_ground_truth([self.case], [], require_all=False)
        self.assertEqual(report["answer_count"], 0)
        self.assertEqual(report["missing_case_ids"], ["case-1"])
        with self.assertRaises(ValueError):
            audit_ground_truth([self.case], [], require_all=True)

    def test_generation_contexts_take_precedence(self):
        report = audit_ground_truth(
            [self.case],
            [
                {
                    "case_id": "case-1",
                    "answer": "10 samples using Model-X.",
                    "generation_contexts": ["The dataset contains 10 samples and uses Model-X."],
                    "contexts": ["unrelated"],
                }
            ],
            require_all=True,
        )
        self.assertEqual(report["summary"]["gold_context_recall"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
