import unittest

from evaluation.review_answers import (
    build_review_template,
    review_answers,
    summarize_reviews,
)


class ReviewAnswersTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {
                "case_id": "case-1",
                "document_id": "paper-a",
                "question": "What score?",
                "required_facts": ["0.3404"],
            },
            {
                "case_id": "case-2",
                "document_id": "paper-a",
                "question": "How many?",
                "required_facts": ["4,855", "DeepSeek-R1"],
            },
        ]
        self.answers = [
            {"case_id": "case-1", "answer": "0.3404"},
            {"case_id": "case-2", "answer": "4,855; DeepSeek-R1"},
        ]

    def test_template_keeps_lexical_signal_separate_from_blank_labels(self):
        template = build_review_template(self.cases, self.answers, require_all=True)
        self.assertEqual(len(template), 2)
        self.assertEqual(template[0]["answer_fact_status"], "full")
        self.assertEqual(template[0]["judgment"], "")
        self.assertEqual(template[0]["table_number"], "")

    def test_review_report_joins_labels_and_summarizes_aspects(self):
        report = review_answers(
            self.cases,
            self.answers,
            [
                {
                    "case_id": "case-1",
                    "judgment": "correct",
                    "table_number": "correct",
                    "units": "not_applicable",
                    "formula": "not_applicable",
                    "citation": "correct",
                    "notes": "exact cell",
                },
                {
                    "case_id": "case-2",
                    "judgment": "partial",
                    "table_number": "not_applicable",
                    "units": "uncertain",
                    "formula": "not_applicable",
                    "citation": "incorrect",
                    "notes": "citation needs review",
                },
            ],
            require_all=True,
        )
        self.assertEqual(report["review_count"], 2)
        self.assertEqual(report["review_summary"]["judgment_counts"], {"correct": 1, "partial": 1})
        self.assertEqual(report["review_summary"]["aspect_counts"]["citation"]["incorrect"], 1)
        self.assertEqual(report["results"][1]["answer_fact_status"], "full")

    def test_review_summary_ignores_unlabeled_aspects(self):
        summary = summarize_reviews([{"judgment": "correct", "table_number": ""}])
        self.assertEqual(summary["reviewed_case_count"], 1)
        self.assertEqual(summary["aspect_counts"]["table_number"], {})

    def test_review_requires_all_case_ids_when_requested(self):
        with self.assertRaises(ValueError):
            review_answers(
                self.cases,
                self.answers,
                [{"case_id": "case-1", "judgment": "correct"}],
                require_all=True,
            )


if __name__ == "__main__":
    unittest.main()
