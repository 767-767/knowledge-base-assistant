import unittest

from evaluation.context_coverage import (
    aggregate_fact_coverage,
    case_fact_coverage,
    fact_is_present,
    unsupported_gold_facts,
)


class ContextCoverageTests(unittest.TestCase):
    def test_numeric_facts_require_token_boundaries(self):
        self.assertTrue(fact_is_present("8", ["The model uses 8 layers."]))
        self.assertFalse(fact_is_present("8", ["The model uses 38 layers."]))
        self.assertTrue(fact_is_present("0.1", ["The threshold is 0.1."]))
        self.assertFalse(fact_is_present("0.1", ["The threshold is 10.1."]))

    def test_special_tokens_survive_markup_normalization(self):
        context = "We serialize rows, columns, and captions with <R>, <C>, and <CAP>."
        self.assertTrue(fact_is_present("<R>", [context]))
        self.assertTrue(fact_is_present("<CAP>", [context]))

    def test_punctuation_and_explicit_aliases_are_auditable(self):
        self.assertTrue(fact_is_present("(0, 1)", ["x is sampled from (0,1)."]))
        self.assertTrue(fact_is_present("entropy ≥4.5", ["entropy (≥ 4.5) is used."]))
        self.assertTrue(fact_is_present("independent facts", ["independent - facts from multiple citations"]))
        case = {
            "required_facts": ["蛋白质", "核酸"],
            "required_fact_aliases": {
                "蛋白质": ["proteins"],
                "核酸": ["nucleic acids"],
            },
        }
        result = case_fact_coverage(case, ["The model handles proteins and nucleic acids."])
        self.assertEqual(result["fact_coverage_status"], "full")
        self.assertEqual(result["required_fact_coverage"], 1.0)

    def test_structured_and_markdown_table_rows_match_without_cross_row_merge(self):
        structured = (
            "Table 2 结构化行：行=TC–RAG；Avg. Interactions=4.78；"
            "Avg. Retrievers=3.37；Avg. Time (s)=50.91；Avg. Token=458.82"
        )
        self.assertTrue(fact_is_present("Avg. Interactions 4.78", [structured]))
        self.assertTrue(fact_is_present("Avg. Time 50.91 s", [structured]))

        markdown = """| Dataset | Split | Types | Samples |
|---|---|---:|---:|
| ChartX | Eval | 18 | 6k |
| Plot2Code | Eval | 6 | 132 |"""
        self.assertTrue(fact_is_present("ChartX Eval 18 6k", [markdown]))
        self.assertTrue(fact_is_present("Plot2Code Eval 6 132", [markdown]))
        self.assertFalse(fact_is_present("ChartX Eval 6 132", [markdown]))

    def test_structured_row_label_and_value_match_in_order(self):
        context = "WTQ：Dataset=WTQ；Samples=13,706\nHiTab：Dataset=HiTab；Samples=6,793"
        self.assertTrue(fact_is_present("WTQ 13,706", [context]))
        self.assertTrue(fact_is_present("HiTab 6,793", [context]))
        self.assertFalse(fact_is_present("WTQ 6,793", [context]))

    def test_markdown_table_facts_match_row_label_and_column_value(self):
        context = """|Statistics|Table|Text|TaT|Total|
|---|---|---|---|---|
|Short-form answers|234|13|93|340|
|Free-form answers|308|67|238|613|
|Total|542|80|331|953|"""
        self.assertTrue(fact_is_present("340 short-form answers", [context]))
        self.assertTrue(fact_is_present("Table 542", [context]))
        self.assertFalse(fact_is_present("340 free-form answers", [context]))

    def test_markdown_table_facts_match_supplemental_statistics_headers(self):
        context = """|Benchmark|Benchmark # Pages|Queries # Queries|
|---|---|---|
|FinReport|2687|853|
|FinSlides|2280|1052|"""
        self.assertTrue(fact_is_present("FinReport 2,687 pages", [context]))
        self.assertTrue(fact_is_present("FinReport 853 queries", [context]))

    def test_case_statuses_distinguish_full_partial_zero_and_not_scored(self):
        case = {"required_facts": ["alpha", "beta"]}
        self.assertEqual(case_fact_coverage(case, ["alpha beta"])["fact_coverage_status"], "full")
        self.assertEqual(case_fact_coverage(case, ["alpha"])["fact_coverage_status"], "partial")
        self.assertEqual(case_fact_coverage(case, ["gamma"])["fact_coverage_status"], "zero")
        self.assertEqual(case_fact_coverage({}, ["alpha"])["fact_coverage_status"], "not_scored")

    def test_aggregate_reports_macro_micro_and_status_rates(self):
        rows = [
            case_fact_coverage({"required_facts": ["a", "b"]}, ["a"]),
            case_fact_coverage({"required_facts": ["c", "d"]}, ["c d"]),
        ]
        aggregate = aggregate_fact_coverage(rows)
        self.assertEqual(aggregate["required_fact_coverage_macro"], 0.75)
        self.assertEqual(aggregate["required_fact_coverage_micro"], 0.75)
        self.assertEqual(aggregate["full_fact_coverage_rate"], 0.5)
        self.assertEqual(aggregate["partial_fact_coverage_rate"], 0.5)
        self.assertEqual(aggregate["zero_fact_coverage_rate"], 0.0)
        self.assertEqual(aggregate["fact_scored_cases"], 2)
        self.assertEqual(aggregate["required_fact_count"], 4)

    def test_gold_support_accepts_only_declared_aliases(self):
        case = {
            "required_facts": ["蛋白质", "核酸"],
            "required_fact_aliases": {"蛋白质": ["proteins"]},
            "contexts": ["The model handles proteins and nucleic acids."],
        }
        self.assertEqual(unsupported_gold_facts(case), ["核酸"])


if __name__ == "__main__":
    unittest.main()
