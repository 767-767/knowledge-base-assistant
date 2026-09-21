import unittest

from evaluation.app_context_gate import audit_case


class AppContextGateTests(unittest.TestCase):
    def test_audit_case_reports_missing_table_row(self):
        case = {
            "case_id": "table-11",
            "document_id": "paper",
            "required_facts": [
                "KAPING Retrieval Accuracy 60.81",
                "G-Retriever Retrieval Accuracy 70.49",
            ],
        }
        result = audit_case(
            case,
            ["Table 11 结构化多行：G-Retriever：Retrieval Accuracy=70.49"],
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["missing_facts"], ["KAPING Retrieval Accuracy 60.81"])


if __name__ == "__main__":
    unittest.main()
