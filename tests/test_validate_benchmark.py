import unittest

from evaluation.validate_benchmark import _missing_extracted_facts


class ExtractedEvidenceValidationTests(unittest.TestCase):
    def test_missing_extracted_facts_uses_declared_aliases(self):
        case = {
            "required_facts": ["0.1910", "GPT-4o"],
            "required_fact_aliases": {"0.1910": ["0.191"]},
        }
        self.assertEqual(_missing_extracted_facts(case, ["GPT-4o reports 0.191"]), [])

    def test_missing_extracted_facts_reports_only_unmatched_facts(self):
        case = {"required_facts": ["alpha", "beta"]}
        self.assertEqual(_missing_extracted_facts(case, ["alpha is present"]), ["beta"])


if __name__ == "__main__":
    unittest.main()
