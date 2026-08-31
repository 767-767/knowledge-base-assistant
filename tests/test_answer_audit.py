import tempfile
import unittest
from pathlib import Path

from evaluation.answer_audit import audit_answer, audit_answers


class AnswerAuditTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "drugr-09",
            "document_id": "drugr",
            "type": "method",
            "question": "How is the dataset built?",
            "required_facts": ["4,855", "DeepSeek-R1", "0.6", "ADMETLab"],
            "required_fact_aliases": {"ADMETLab": ["ADMET Lab"]},
        }

    def test_audit_distinguishes_full_partial_and_zero(self):
        full = audit_answer(
            self.case,
            "4,855 samples; DeepSeek-R1 proposes candidates with similarity 0.6; ADMET Lab evaluates them.",
        )
        partial = audit_answer(self.case, "The dataset contains 4,855 samples and uses ADMET evaluation.")
        zero = audit_answer(self.case, "The paper studies molecular optimization.")
        self.assertEqual(full["answer_fact_status"], "full")
        self.assertEqual(partial["answer_fact_status"], "partial")
        self.assertEqual(partial["missing_facts"], ["DeepSeek-R1", "0.6", "ADMETLab"])
        self.assertEqual(zero["answer_fact_status"], "zero")

    def test_audit_answers_requires_all_and_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            audit_answers([self.case], [{"case_id": "other", "answer": "x"}])
        with self.assertRaises(ValueError):
            audit_answers([self.case], [], require_all=True)

    def test_jsonl_and_json_answer_inputs_are_supported(self):
        import json
        from evaluation.answer_audit import _load_answers, _load_cases

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            answers_path = root / "answers.jsonl"
            cases_path.write_text(json.dumps({"test_cases": [self.case]}), encoding="utf-8")
            answers_path.write_text(json.dumps({"case_id": "drugr-09", "answer": "4,855"}) + "\n", encoding="utf-8")
            self.assertEqual(_load_cases(cases_path)[0]["case_id"], "drugr-09")
            self.assertEqual(_load_answers(answers_path)[0]["answer"], "4,855")

    def test_benchmark_pointer_cases_are_resolved_before_answer_audit(self):
        from evaluation.answer_audit import _load_cases

        cases_path = Path(__file__).resolve().parents[1] / "evaluation/benchmark/cases.jsonl"
        cases = _load_cases(cases_path)
        self.assertEqual(len(cases), 53)
        drugr = next(case for case in cases if case["case_id"] == "drugr-09")
        self.assertIn("4,855", drugr["required_facts"])
        self.assertIn("DeepSeek-R1", drugr["contexts"][0])

    def test_case_loader_rejects_duplicate_ids(self):
        import json
        from evaluation.answer_audit import _load_cases

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            record = {"case_id": "duplicate", "question": "q", "ground_truth": "a"}
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_cases(path)

    def test_cjk_fact_matches_inside_adjacent_chinese_prose(self):
        case = {
            "case_id": "cjk",
            "required_facts": ["块", "不具有对话性质"],
        }
        result = audit_answer(case, "论文被切分为较小的块，最终用于不具有对话性质的科学表格。")
        self.assertEqual(result["answer_fact_status"], "full")

    def test_benchmark_ground_truths_are_auditable(self):
        from evaluation.answer_audit import _load_cases

        cases_path = Path(__file__).resolve().parents[1] / "evaluation/benchmark/cases.jsonl"
        cases = _load_cases(cases_path)
        report = audit_answers(
            cases,
            [{"case_id": case["case_id"], "answer": case["ground_truth"]} for case in cases],
            require_all=True,
        )
        self.assertEqual(report["summary"]["full_fact_coverage_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
