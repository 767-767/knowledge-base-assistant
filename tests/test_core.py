import unittest

import app
from sci_rag_core import (
    Chunk,
    extract_table_cell,
    extract_tables,
    filter_table_rows_by_entity,
    normalize_for_match,
    rerank_table_first,
    select_row_entity,
    split_to_chunks,
)


TABLE_1 = """|Baseline|Score|
|---|---|
|**DrugR**|**0.2712**|
|GPT5|0.1969|"""
TABLE_2 = """|Baseline|F1|
|---|---|
|**DrugR**<sup>_∗_</sup>|**0.3404**|
|_SFT_<sup>_∗_</sup>|0.2997|"""
TABLE_2_FULL = """|Baseline|Overall Optimization Score|Target property F1 score|
|---|---|---|
|**DrugR**<sup>_∗_</sup>|**0.2060**|**0.3404**|
|_SFT_<sup>_∗_</sup>|0.1949|0.2997|"""


class CoreTests(unittest.TestCase):
    def test_table_spans_are_removed_from_text_chunks(self):
        markdown = f"# Results\n\n**Table 1**\n{TABLE_1}\n\n**Table 2**\n{TABLE_2}\n\nNarrative."
        tables, body = extract_tables(markdown, "paper.pdf", {"page": 2})
        self.assertEqual([table.metadata["table_id"] for table in tables], ["table-1", "table-2"])
        self.assertNotIn("0.2712", body)
        self.assertNotIn("0.3404", body)
        chunks = split_to_chunks([Chunk(markdown, {"source": "paper.pdf", "page": 2})], "paper.pdf")
        self.assertEqual(sum(chunk.metadata.get("type") == "table" for chunk in chunks), 2)
        self.assertFalse(
            any(chunk.metadata.get("type") == "text" and "|---|---|" in chunk.page_content for chunk in chunks)
        )

    def test_row_entity_ignores_column_name_and_normalizes_sup(self):
        question = "Table 2 中 DrugR* 的 Target property F1 score 是多少？"
        entity = select_row_entity(question, TABLE_2)
        self.assertEqual(entity, "DrugR*")
        filtered = filter_table_rows_by_entity(TABLE_2, entity)
        self.assertIn("0.3404", filtered)
        self.assertNotIn("0.2997", filtered)
        self.assertEqual(normalize_for_match("**DrugR**"), "drugr")
        self.assertEqual(normalize_for_match("**DrugR**<sup>_∗_</sup>"), "drugr*")

    def test_structured_cell_lookup_supports_chinese_alias(self):
        cell = extract_table_cell(
            "Table 2 中 DrugR*（在少量新数据上微调后）的整体优化得分是多少？",
            TABLE_2_FULL,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "0.2060")

        cell = extract_table_cell(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            TABLE_2,
            {"type": "table", "table_number": 2},
        )
        self.assertIsNotNone(cell)
        self.assertEqual(cell["value"], "0.3404")

    def test_caption_detection_does_not_match_stable(self):
        markdown = "Figure 5 stable training dynamics.\n\n|x|y|\n|---|---|\n|0|1|"
        tables, _ = extract_tables(markdown, "paper.pdf")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].metadata["table_id"], "table-unnamed-1")

    def test_rerank_excludes_legacy_table_text_for_explicit_table(self):
        texts = [TABLE_1, "Narrative", TABLE_2]
        metas = [
            {"type": "text"},
            {"type": "text"},
            {"type": "table", "table_number": 2, "table_id": "table-2"},
        ]
        order, _, filtered = rerank_table_first(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？", texts, metas
        )
        self.assertEqual(order, [2, 1])
        self.assertIn("0.3404", filtered[2])
        self.assertNotIn("0.2712", [filtered[index] for index in order])


class RuntimeContractTests(unittest.TestCase):
    def test_app_import_has_no_runtime(self):
        self.assertIsNone(app._runtime)

    def test_query_uses_filtered_contexts_for_generation_and_return(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["legacy", "narrative"]],
                    "documents": [[TABLE_1, "Method narrative"]],
                    "metadatas": [[{"type": "text"}, {"type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table2"],
                    "documents": [TABLE_2],
                    "metadatas": [{"type": "table", "table_number": 2, "table_id": "table-2"}],
                }

        class Client:
            def __init__(self):
                self.prompt = None
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "0.3404"})()})()]},
                )()

        client = Client()
        runtime = app.Runtime(app.RuntimeConfig(retrieval_k=2, context_k=2), client, Embedding(), Collection())
        result = app.query_knowledge(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            return_contexts=True,
            runtime=runtime,
        )
        self.assertIn("Table 2 结构化单元格", result["contexts"][0])
        self.assertIn("值=0.3404", result["contexts"][0])
        self.assertEqual(result["answer"], "根据 Table 2 中“DrugR*”行的“F1”列，数值为 **0.3404**。")
        # Deterministic table lookups must not call the generation model.
        self.assertIsNone(client.prompt)

    def test_explicit_missing_table_never_falls_back_to_other_table(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 1

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[TABLE_1]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {"ids": ["table1"], "documents": [TABLE_1], "metadatas": [{"type": "table", "table_number": 1}]}

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                raise AssertionError("missing explicit table must not call the model")

        runtime = app.Runtime(app.RuntimeConfig(retrieval_k=1, context_k=1), Client(), Embedding(), Collection())
        result = app.query_knowledge("Table 2 中 DrugR* 的整体优化得分是多少？", runtime=runtime)
        self.assertIn("Table 2", result["answer"])
        self.assertIn("不能用其他表格替代", result["answer"])


if __name__ == "__main__":
    unittest.main()
