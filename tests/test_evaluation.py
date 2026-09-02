import unittest

from evaluation.evaluate import build_report, fact_coverage, gold_context_recall, load_testset


class EvaluationTests(unittest.TestCase):
    def test_testset_has_atomic_facts(self):
        _, cases = load_testset("evaluation/test_questions.json")
        self.assertEqual(len(cases), 11)
        self.assertTrue(all(case.get("required_facts") for case in cases))

    def test_load_testset_supports_multipaper_jsonl(self):
        meta, cases = load_testset("evaluation/benchmark/cases.jsonl")
        self.assertEqual(meta["paper"], "cases")
        self.assertEqual(len(cases), 53)
        self.assertEqual(cases[0]["case_id"], "drugr-01")

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

    def test_gold_context_recall_allows_larger_format_shifted_chunks(self):
        case = {"contexts": ["The dataset contains 10 samples and uses Model-X."]}
        retrieved = ["Before the sentence. The dataset contains 10 samples; it uses Model-X. After it."]
        self.assertEqual(gold_context_recall(retrieved, case), 1.0)

    def test_build_report_preserves_generation_and_evaluation_traces(self):
        case = {
            "id": "case-1",
            "question": "q",
            "ground_truth": "gold",
            "contexts": ["gold"],
            "required_facts": ["gold"],
        }
        report = build_report(
            {},
            [case],
            [
                {
                    "case": case,
                    "answer": "gold",
                    "generation_contexts": ["gold", "extra"],
                    "generation_context_ids": ["id-1", "id-2"],
                    "generation_context_metadatas": [{"page": 1}, {"page": 2}],
                    "contexts": ["gold"],
                    "context_ids": ["id-1"],
                    "context_metadatas": [{"page": 1}],
                    "num_retrieved": 2,
                }
            ],
            None,
            1,
            0.1,
            ragas_version="0.4.3",
            judge_model="judge",
            embedding_model="embed",
            generation_model="generator",
        )
        result = report["results"][0]
        self.assertEqual(result["generation_contexts"], ["gold", "extra"])
        self.assertEqual(result["evaluated_contexts"], ["gold"])
        self.assertEqual(result["reference_contexts"], ["gold"])
        self.assertEqual(report["meta"]["generation_model"], "generator")
        self.assertEqual(
            report["meta"]["ragas_metric_input_columns"]["faithfulness"],
            ["user_input", "response", "retrieved_contexts"],
        )

    def test_build_report_preserves_benchmark_case_id(self):
        case = {
            "case_id": "paper-a-01",
            "question": "问题",
            "ground_truth": "答案",
            "contexts": ["gold"],
            "required_facts": ["答案"],
        }
        record = {
            "case": case,
            "answer": "答案",
            "generation_contexts": ["retrieved"],
            "generation_context_ids": ["chunk-1"],
            "generation_context_metadatas": [{}],
            "contexts": ["retrieved"],
            "context_ids": ["chunk-1"],
            "context_metadatas": [{}],
            "num_retrieved": 1,
        }
        report = build_report({}, [case], [record], None, 10, 0.1)
        self.assertEqual(report["results"][0]["id"], "paper-a-01")


if __name__ == "__main__":
    unittest.main()
