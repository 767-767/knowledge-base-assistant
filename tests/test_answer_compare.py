import unittest

from evaluation.compare_answer_runs import compare_answer_runs


class AnswerCompareTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {
                "case_id": "case-1",
                "document_id": "paper-a",
                "question": "How many samples?",
                "required_facts": ["4,855", "DeepSeek-R1"],
            },
            {
                "case_id": "case-2",
                "document_id": "paper-a",
                "question": "What score?",
                "required_facts": ["0.3404"],
            },
        ]

    def test_compare_reports_fact_and_status_changes(self):
        report = compare_answer_runs(
            self.cases,
            [
                {"case_id": "case-1", "answer": "4,855"},
                {"case_id": "case-2", "answer": "0.3404"},
            ],
            [
                {"case_id": "case-1", "answer": "4,855; DeepSeek-R1"},
                {"case_id": "case-2", "answer": "0.3404"},
            ],
            baseline_name="dense",
            candidate_name="hybrid",
            require_all=True,
        )
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["baseline"]["summary"]["full_fact_coverage_rate"], 0.5)
        self.assertEqual(report["candidate"]["summary"]["full_fact_coverage_rate"], 1.0)
        self.assertEqual(report["summary_delta_candidate_minus_baseline"]["full_fact_coverage_rate"], 0.5)
        self.assertEqual(report["case_comparison"]["improved_case_ids"], ["case-1"])
        self.assertEqual(report["case_comparison"]["status_transitions"]["partial->full"], 1)

    def test_compare_rejects_different_case_sets(self):
        with self.assertRaises(ValueError):
            compare_answer_runs(
                self.cases,
                [{"case_id": "case-1", "answer": "4,855"}],
                [
                    {"case_id": "case-1", "answer": "4,855"},
                    {"case_id": "case-2", "answer": "0.3404"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
