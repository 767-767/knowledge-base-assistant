import unittest

from evaluation.evaluate import fact_coverage, gold_context_recall, load_testset


class EvaluationTests(unittest.TestCase):
    def test_testset_has_atomic_facts(self):
        _, cases = load_testset("evaluation/test_questions.json")
        self.assertEqual(len(cases), 11)
        self.assertTrue(all(case.get("required_facts") for case in cases))

    def test_fact_coverage_exposes_wrong_table_value(self):
        _, cases = load_testset("evaluation/test_questions.json")
        case = cases[2]
        wrong_score, wrong_exact = fact_coverage("Table 2 中 DrugR 的得分为 0.4364。", case)
        correct_score, correct_exact = fact_coverage("Table 2 中 DrugR* 的得分为 0.3404。", case)
        self.assertEqual((wrong_score, wrong_exact), (0.0, False))
        self.assertEqual((correct_score, correct_exact), (1.0, True))

    def test_gold_context_recall_is_reference_based(self):
        case = {"contexts": ["alpha beta", "gamma delta"]}
        self.assertEqual(gold_context_recall(["alpha beta", "gamma delta"], case), 1.0)
        self.assertEqual(gold_context_recall(["alpha beta"], case), 0.5)


if __name__ == "__main__":
    unittest.main()
