import unittest

from sci_rag_core import Chunk, extract_table_cell, extract_tables, split_to_chunks


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


if __name__ == "__main__":
    unittest.main()
