from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from evaluation import app_context_audit
from evaluation.app_context_audit import audit_case


class AppContextAuditTests(unittest.TestCase):
    def test_incomplete_coverage_is_reported_without_blocking(self):
        report = {
            "full_case_count": 0,
            "case_count": 1,
            "fact_micro": 0.5,
            "failures": [{"case_id": "example", "status": "partial", "missing_facts": ["42"]}],
        }
        output = StringIO()
        with (
            patch("sys.argv", ["app_context_audit", "--manifest", "cases.json", "--db-path", "isolated"]),
            patch.object(app_context_audit, "run_audit", return_value=report),
            redirect_stdout(output),
        ):
            self.assertEqual(app_context_audit.main(), 0)
        self.assertIn("example: partial; 遗漏=42", output.getvalue())

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
