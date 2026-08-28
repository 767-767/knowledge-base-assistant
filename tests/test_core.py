import os
import tempfile
import unittest
from unittest.mock import patch

import app
from sci_rag_reranking import RerankResult
from sci_rag_retrieval import RankedItem
from sci_rag_core import (
    Chunk,
    extract_table_cell,
    extract_tables,
    filter_table_rows_by_entity,
    is_table_question,
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
    def test_table_intent_requires_an_explicit_table_reference(self):
        self.assertFalse(is_table_question("显式推理数据集包含多少个样本？"))
        self.assertFalse(is_table_question("训练样本量和成功比率是多少？"))
        self.assertFalse(is_table_question("该数值是否稳定？"))
        self.assertTrue(is_table_question("Table 2 中 DrugR* 的得分是多少？"))
        self.assertTrue(is_table_question("表2中哪个模型最好？"))
        self.assertTrue(is_table_question("下表给出了哪些结果？"))

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

    def test_runtime_config_defaults_dense_and_validates_hybrid_settings(self):
        with patch.dict(
            os.environ,
            {
                "SCI_RAG_RETRIEVAL_MODE": "hybrid",
                "SCI_RAG_HYBRID_RRF_K": "45",
                "SCI_RAG_RERANKER_MODEL": "BAAI/bge-reranker-base",
                "SCI_RAG_RERANKER_REVISION": "fixed-revision",
            },
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertEqual(config.retrieval_mode, "hybrid")
        self.assertEqual(config.hybrid_rrf_k, 45)
        self.assertEqual(config.reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(config.reranker_revision, "fixed-revision")

        with patch.dict(os.environ, {"SCI_RAG_RETRIEVAL_MODE": "unsupported"}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertEqual(config.retrieval_mode, "dense")
        self.assertIsNone(config.reranker_model)

    def test_runtime_rejects_reranker_outside_hybrid_mode(self):
        with self.assertRaises(ValueError):
            app.Runtime(app.RuntimeConfig(), None, None, None, reranker=object())

    def test_opt_in_reranker_changes_hybrid_order_and_prompt_context(self):
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
                    "ids": [["weak", "strong"]],
                    "documents": [["Weak context.", "The answer is 4,855 samples."]],
                    "metadatas": [[{"type": "text"}, {"type": "text"}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["weak", "strong", "other"],
                    "documents": [
                        "Weak context.",
                        "The answer is 4,855 samples.",
                        "Other context.",
                    ],
                    "metadatas": [{"type": "text"}] * 3,
                }

        class Reranker:
            def __init__(self):
                self.calls = 0

            def rerank(self, _question, candidates, documents):
                self.calls += 1
                order = sorted(
                    candidates,
                    key=lambda item: "4,855" not in documents[int(item.key)],
                )
                ranked = [
                    RankedItem(item.key, float(len(order) - index))
                    for index, item in enumerate(order)
                ]
                return RerankResult(ranked, 0.01, len(ranked), 0)

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        reranker = Reranker()
        client = Client()
        runtime = app.Runtime(
            app.RuntimeConfig(
                retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
            ),
            client,
            Embedding(),
            Collection(),
            reranker=reranker,
        )
        result = app.query_knowledge("How many samples?", runtime=runtime)

        self.assertEqual(reranker.calls, 1)
        self.assertIn("4,855", result["contexts"][0])
        self.assertTrue(all(context in client.prompt for context in result["contexts"]))

    def test_dense_narrative_quantity_question_does_not_scan_tables(self):
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
                    "ids": [["dense"]],
                    "documents": [["The explicit-reasoning dataset contains 4,855 samples and uses a reverse-engineering pipeline."]],
                    "metadatas": [[{"type": "text"}]],
                }

            def get(self, **_kwargs):
                raise AssertionError("a narrative quantity question must not scan all tables")

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        runtime = app.Runtime(app.RuntimeConfig(), Client(), Embedding(), Collection())
        result = app.query_knowledge(
            "DrugR 的显式推理数据集包含多少个样本？推理标注是通过什么管道构建的？",
            runtime=runtime,
        )
        self.assertIn("4,855 samples", result["contexts"][0])
        self.assertIsNone(runtime._lexical_snapshot)

    def test_hybrid_mode_adds_lexical_candidate_and_reuses_snapshot(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.get_calls = 0

            def count(self):
                return 3

            def query(self, **_kwargs):
                return {
                    "ids": [["dense"]],
                    "documents": [["Dense evidence"]],
                    "metadatas": [[{"type": "text"}]],
                }

            def get(self, **_kwargs):
                self.get_calls += 1
                return {
                    "ids": ["dense", "lexical", "other"],
                    "documents": [
                        "Dense evidence",
                        "Chlorophyll flux is controlled by the antenna complex.",
                        "Unrelated appendix.",
                    ],
                    "metadatas": [{"type": "text"}, {"type": "text"}, {"type": "text"}],
                }

        class Client:
            def __init__(self):
                self.prompt = ""
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.prompt = kwargs["messages"][1]["content"]
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        collection = Collection()
        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
        )
        client = Client()
        runtime = app.Runtime(config, client, Embedding(), collection)
        first = app.query_knowledge("What controls chlorophyll flux?", runtime=runtime)
        second = app.query_knowledge("What controls chlorophyll flux?", runtime=runtime)
        self.assertTrue(any("Chlorophyll flux" in context for context in first["contexts"]))
        self.assertEqual(first["contexts"], second["contexts"])
        self.assertTrue(all(context in client.prompt for context in second["contexts"]))
        self.assertEqual(collection.get_calls, 1)

    def test_hybrid_falls_back_to_dense_for_weak_cross_language_bm25(self):
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
                    "ids": [["stats", "pipeline", "overview"]],
                    "documents": [[
                        "The explicit-reasoning dataset contains 4,855 samples.",
                        "A closed-loop reverse-engineering pipeline constructs the annotations.",
                        "DrugR overview.",
                    ]],
                    "metadatas": [[{"type": "text"}] * 3],
                }

            def get(self, **kwargs):
                self.assert_no_table_filter(kwargs)
                return {
                    "ids": ["overview", "stats", "pipeline"],
                    "documents": [
                        "DrugR overview.",
                        "The explicit-reasoning dataset contains 4,855 samples.",
                        "A closed-loop reverse-engineering pipeline constructs the annotations.",
                    ],
                    "metadatas": [{"type": "text"}] * 3,
                }

            @staticmethod
            def assert_no_table_filter(kwargs):
                if "where" in kwargs:
                    raise AssertionError("narrative quantity question must not fetch all tables")

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]},
                )()

        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=3, context_k=2
        )
        runtime = app.Runtime(config, Client(), Embedding(), Collection())
        result = app.query_knowledge(
            "DrugR 的显式推理数据集包含多少个样本？推理标注是通过什么管道构建的？",
            runtime=runtime,
        )
        self.assertEqual(
            result["contexts"],
            [
                "The explicit-reasoning dataset contains 4,855 samples.",
                "A closed-loop reverse-engineering pipeline constructs the annotations.",
            ],
        )

    def test_upload_invalidates_hybrid_lexical_snapshot(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def __init__(self):
                self.total = 0

            def count(self):
                return self.total

            def upsert(self, **_kwargs):
                self.total += 1

        runtime = app.Runtime(app.RuntimeConfig(), None, Embedding(), Collection())
        runtime._lexical_snapshot = object()
        with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
            with patch.object(
                app,
                "load_and_split_document",
                return_value=[Chunk("New evidence", {"type": "text"})],
            ):
                app.add_document_to_db(handle.name, runtime=runtime)
        self.assertIsNone(runtime._lexical_snapshot)

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

    def test_hybrid_mode_preserves_explicit_table_cell_protection(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Embedding:
            def encode(self, _message):
                return Vector([0.1, 0.2])

        class Collection:
            def count(self):
                return 2

            def query(self, **_kwargs):
                return {
                    "ids": [["table1"]],
                    "documents": [[TABLE_1]],
                    "metadatas": [[{"type": "table", "table_number": 1}]],
                }

            def get(self, **_kwargs):
                return {
                    "ids": ["table1", "table2"],
                    "documents": [TABLE_1, TABLE_2],
                    "metadatas": [
                        {"type": "table", "table_number": 1, "table_caption": "Table 1"},
                        {"type": "table", "table_number": 2, "table_caption": "Table 2"},
                    ],
                }

        class Client:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **_kwargs):
                raise AssertionError("explicit table cell must not call the model")

        config = app.RuntimeConfig(
            retrieval_mode="hybrid", hybrid_candidate_k=2, context_k=2
        )
        class Reranker:
            def rerank(self, _question, candidates, _documents):
                ranked = [RankedItem(item.key, 1.0) for item in reversed(candidates)]
                return RerankResult(ranked, 0.01, len(ranked), 0)

        runtime = app.Runtime(
            config, Client(), Embedding(), Collection(), reranker=Reranker()
        )
        result = app.query_knowledge(
            "Table 2 中 DrugR* 的 Target property F1 score 是多少？",
            runtime=runtime,
        )
        self.assertEqual(
            result["answer"],
            "根据 Table 2 中“DrugR*”行的“F1”列，数值为 **0.3404**。",
        )
        self.assertEqual(result["context_metadatas"][0]["table_number"], 2)

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
