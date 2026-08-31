import unittest

from evaluation.validate_answer_evidence import validate_rows


class AnswerEvidenceValidationTests(unittest.TestCase):
    def test_validation_uses_trace_contexts_and_not_gold_fields(self):
        report = validate_rows(
            [
                {
                    "case_id": "case-1",
                    "question": "How many samples are in the dataset?",
                    "answer": "The dataset contains 4,855 samples.",
                    "contexts": ["The dataset contains 4,855 samples."],
                    "context_metadatas": [{"source": "paper.pdf", "headers": "Dataset"}],
                    "ground_truth": "this field is intentionally ignored",
                    "required_facts": ["not read by validator"],
                }
            ]
        )
        self.assertEqual(report["status_counts"], {"ok": 1})
        self.assertEqual(report["review_case_ids"], [])

    def test_validation_reports_review_rows_without_claiming_incorrectness(self):
        report = validate_rows(
            [
                {
                    "case_id": "case-2",
                    "question": "How is the annotation pipeline constructed?",
                    "answer": "DeepSeek R1 provides rationales.",
                    "contexts": [
                        "The annotation pipeline uses DeepSeek-R1 and ADMETLab above 0.6."
                    ],
                    "context_metadatas": [{"source": "paper.pdf", "headers": "Dataset"}],
                }
            ]
        )
        self.assertEqual(report["status_counts"], {"review": 1})
        self.assertIn("partial_high_signal_line", report["reason_counts"])
        self.assertEqual(report["review_case_ids"], ["case-2"])


if __name__ == "__main__":
    unittest.main()
