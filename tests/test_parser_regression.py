import unittest

from sci_rag_core import (
    Chunk,
    extract_table_cell,
    extract_table_row_values,
    extract_tables,
    matching_table_indices,
    parse_markdown_table,
    split_to_chunks,
    table_number_from_question,
)


class ParserRegressionTests(unittest.TestCase):
    def test_caption_after_table_is_associated_and_removed_from_body(self):
        markdown = """|Metric|Value|
|---|---|
|A|1|


Table 2: Results on the held-out set.

Narrative text.
"""
        tables, body = extract_tables(markdown, "paper.pdf", {"page": 4})

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].metadata["table_number"], 2)
        self.assertEqual(tables[0].metadata["table_caption"], "Table 2: Results on the held-out set.")
        self.assertNotIn("Table 2:", body)
        self.assertIn("Narrative text.", body)

    def test_decorated_caption_before_table_is_detected(self):
        markdown = """<u>Table 3: Ablation results</u>

|Metric|Value|
|---|---|
|A|1|
"""
        tables, _ = extract_tables(markdown, "paper.pdf")

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].metadata["table_number"], 3)

    def test_grouped_headers_are_combined_without_word_fragment_space(self):
        markdown = """Table 1: Error comparison

| | |Darcy s|mooth|Darcy|rough||Darcy multiscale|
|---|---|---|---|---|---|---|---|
|Model|time|L2|H1|L2|H1|L2|H1|
|MgNO|1|0.20|0.10|0.339|0.40|0.50|0.60|
"""
        tables, _ = extract_tables(markdown, "paper.pdf")

        self.assertEqual(len(tables), 1)
        content = tables[0].page_content
        self.assertIn("Darcy smooth L2", content)
        self.assertNotIn("Darcy s mooth", content)
        cell = extract_table_cell(
            "Table 1 中 MgNO 的 Darcy rough L2 是多少？",
            content,
            tables[0].metadata,
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "0.339")

    def test_split_group_header_is_repaired_and_usage_backbone_keeps_both_rows(self):
        markdown = """Table 2: Latency evaluation, measured in seconds.

|Method|WikiTQ|H|eteQA|
|---|---|---|---|
||**Total Latency**<br>**Avg. Latency / Step**|**Total Latency**|**Avg. Latency / Step**|
|TableRAG|13.50<br>5.92|24.57|7.70|
|ReAct|17.70<br>7.93|45.65|10.79|
"""
        tables, _body = extract_tables(markdown, "paper.pdf")
        content = tables[0].page_content
        self.assertIn("HeteQA Total Latency", content)
        self.assertIn("HeteQA Avg. Latency / Step", content)
        latency_row = extract_table_row_values(
            "Table 2 中，TableRAG 和 ReAct 在 HeteQA 上的总延迟和平均每步延迟分别是多少？",
            content,
            tables[0].metadata,
        )
        self.assertEqual(
            [item["column"] for item in latency_row["rows"][0]["values"]],
            ["HeteQA Total Latency", "HeteQA Avg. Latency / Step"],
        )

        row = extract_table_row_values(
            "Table 2 中，使用 Qwen-2.5-72b 时，TableRAG 和 ReAct 在 HeteQA 上的总延迟和平均每步延迟分别是多少？",
            """|Method|Backbone|HeteQA|
|---|---|---|
|ReAct|Claude-3.5|19.4|
||Qwen-2.5-72b|12.5|
|TableRAG|Claude-3.5|27.8|
||Qwen-2.5-72b|25.5|""",
            {"type": "table", "table_number": 4},
        )
        self.assertEqual(
            [(item["row"], item["values"][0]["value"]) for item in row["rows"]],
            [("ReAct", "12.5"), ("TableRAG", "25.5")],
        )
        self.assertEqual(
            [item for item in row["rows"][0]["descriptors"]],
            [{"column": "Backbone", "value": "Qwen-2.5-72b"}],
        )

    def test_iou_is_selected_with_other_named_metrics(self):
        question = (
            "Table 7 中 Gemini 2.0 Flash 在 BIOGR 上的 IoU、"
            "Normalized Distance Error 和 Relative Box Size 分别是多少？"
        )
        table = """|Model|IoU|Normalized Distance Error|Relative Box Size|
|---|---|---|---|
|Gemini 2.0 Flash|0.49|3.03|3.05|"""
        row = extract_table_row_values(
            question, table, {"type": "table", "table_number": 7}
        )
        self.assertEqual(
            [(item["column"], item["value"]) for item in row["values"]],
            [
                ("IoU", "0.49"),
                ("Normalized Distance Error", "3.03"),
                ("Relative Box Size", "3.05"),
            ],
        )

    def test_split_specific_task_header_is_normalized_for_column_lookup(self):
        markdown = """Table 2: Retrieval performance.

|Model|Mi Pr.|PV-speciific Rec.|PV-speciific F1|
|---|---|---|---|
|Claude 3 (Opus)|32.18|47.06|31.48|
"""
        tables, _body = extract_tables(markdown, "paper.pdf")
        headers, _rows = parse_markdown_table(tables[0].page_content)
        self.assertEqual(
            headers[1:],
            ["MPV-specific Pr.", "MPV-specific Rec.", "MPV-specific F1"],
        )
        row = extract_table_row_values(
            "Table 2 中，Claude 3 (Opus) 在 MPV-specific 子任务上的 Precision、Recall 和 F1 分别是多少？",
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertEqual([item["value"] for item in row["values"]], ["32.18", "47.06", "31.48"])

    def test_normal_table_first_data_row_is_not_dropped_as_a_header(self):
        markdown = """Table 2: Model scores

|Model|Dataset|Score|
|---|---|---|
|Model A|Data X|0.91|
|Model B|Data Y|0.87|
"""
        tables, _ = extract_tables(markdown, "paper.pdf")

        self.assertEqual(tables[0].page_content.splitlines()[2:], [
            "|Model A|Data X|0.91|",
            "|Model B|Data Y|0.87|",
        ])

    def test_standalone_unit_column_is_folded_into_metric_header(self):
        markdown = """Table 4: Configuration errors

|Model Configuration|L2 Error|(×10<sup>−2</sup>)|
|---|---|---|
|Baseline MgNO||1.63|
"""
        tables, _ = extract_tables(markdown, "paper.pdf")

        self.assertEqual(len(tables), 1)
        self.assertIn("L2 Error (×10−2)", tables[0].page_content)
        cell = extract_table_cell(
            "Table 4 中 Baseline MgNO 的 L2 Error 是多少？",
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "1.63")

    def test_adjacent_bold_markup_preserves_row_entity_boundary(self):
        markdown = """Table 4: Configuration errors

|Model Configuration|L2 Error|(×10<sup>−2</sup>)|
|---|---|---|
|**Baseline**MgNO||1.63|
"""
        tables, _ = extract_tables(markdown, "paper.pdf")

        cell = extract_table_cell(
            "Table 4 中 Baseline MgNO 的 L2 Error 是多少？",
            tables[0].page_content,
            tables[0].metadata,
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["row"], "Baseline MgNO")
        self.assertEqual(cell["value"], "1.63")

    def test_url_metadata_layout_is_not_indexed_as_scientific_table(self):
        markdown = """|https://doi.org/10.1234/example|Author|
|---|---|
|Open access|A. Researcher|

Article text.
"""
        tables, body = extract_tables(markdown, "paper.pdf")

        self.assertEqual(tables, [])
        self.assertIn("https://doi.org/10.1234/example", body)
        self.assertIn("Article text.", body)

    def test_page_metadata_survives_canonical_table_split(self):
        markdown = """Table 2: Results

|Metric|Value|
|---|---|
|A|1|
"""
        chunks = split_to_chunks([Chunk(markdown, {"page": 7})], "paper.pdf")

        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("type") == "table"]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0].metadata["page"], 7)
        self.assertEqual(table_chunks[0].metadata["table_number"], 2)

    def test_side_by_side_tables_are_split_and_captions_are_matched(self):
        markdown = """|**Statistics**|**Long-context**|**Short-context**|**Statistics**|**Table**|**Text**|**TaT**|**Total**|
|---|---|---|---|---|---|---|---|
|Questions|953|953|Short-form answers|234|13|93|340|
|Papers|871|871|Free-form answers|308|67|238|613|
|Avg. Tables|5.2|1.1|Total|542|80|331|953|
|Avg. Cells|60.8|56.6||||||

Table 4: Question distribution over different answers and sources.

Table 3: The statistics of the benchmark, including average tables and cells.
"""
        tables, body = extract_tables(markdown, "paper.pdf", {"page": 4})

        self.assertEqual([table.metadata["table_label"] for table in tables], ["3", "4"])
        self.assertNotIn("953", body)
        question = "Table 3 中 Questions、Papers、Avg. Tables、Avg. Cells 分别是多少？"
        matched = matching_table_indices(
            question,
            [table.page_content for table in tables],
            [table.metadata for table in tables],
        )
        self.assertEqual(matched, [0])
        values = extract_table_row_values(question, tables[0].page_content, tables[0].metadata)
        self.assertEqual(values["rows"][0]["values"][0]["value"], "953")

    def test_supplemental_table_label_is_preserved_and_matchable(self):
        markdown = """Table S1: Benchmark statistics.

|Benchmark||Documents||Queries||Evidence Label||
|---|---|---|---|---|---|---|---|
|Benchmark|# Pages|# Docs|Avg. Len|# Queries|Text|Table|Visual|
|FinReport|2687|19|141|853|75%|24%|1%|
"""
        tables, _body = extract_tables(markdown, "paper.pdf")

        self.assertEqual(tables[0].metadata["table_label"], "S1")
        self.assertIsNone(tables[0].metadata["table_number"])
        question = "Table S1 中 FinReport 的 # Pages 是多少？"
        self.assertEqual(table_number_from_question(question), "S1")
        self.assertEqual(
            matching_table_indices(
                question,
                [tables[0].page_content],
                [tables[0].metadata],
            ),
            [0],
        )


if __name__ == "__main__":
    unittest.main()
