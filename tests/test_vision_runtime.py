import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

import app
from sci_rag_core import Chunk, file_sha256


class _Embedding:
    def encode(self, _text):
        return SimpleNamespace(tolist=lambda: [0.1, 0.2])


class _Collection:
    def __init__(self, records=None):
        self.records = records or []
        self.upserts = []

    def count(self):
        return len(self.records) + len(self.upserts)

    def get(self, **_kwargs):
        return {
            "ids": [record[0] for record in self.records],
            "documents": [record[1] for record in self.records],
            "metadatas": [record[2] for record in self.records],
        }

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class _Client:
    def __init__(self, content="vision answer", error=None):
        self.chat = self
        self.completions = self
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = self.content.pop(0) if isinstance(self.content, list) else self.content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _EmptyRetrievalCollection:
    def count(self):
        return 1

    def get(self, **_kwargs):
        return {
            "ids": ["existing"],
            "documents": ["existing text"],
            "metadatas": [{"source": "paper.pdf"}],
        }

    def query(self, **_kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}


class VisionRuntimeTests(unittest.TestCase):
    def test_vision_config_is_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = app.RuntimeConfig.from_env()
        self.assertFalse(config.vision_enabled)
        self.assertEqual(config.vision_model, "deepseek-v4-flash-vision-exp")

    def test_vision_config_reads_opt_in_values(self):
        with patch.dict(
            os.environ,
            {"SCI_RAG_VISION_ENABLED": "true", "SCI_RAG_VISION_MODEL": "vision-test"},
            clear=True,
        ):
            config = app.RuntimeConfig.from_env()
        self.assertTrue(config.vision_enabled)
        self.assertEqual(config.vision_model, "vision-test")

    def test_enabled_pdf_is_persisted_by_hash_and_disabled_pdf_is_not(self):
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "paper.pdf"
            pdf.write_bytes(b"pdf fixture")
            for enabled in (False, True):
                db_path = Path(directory) / ("db-on" if enabled else "db-off")
                runtime = app.Runtime(
                    app.RuntimeConfig(db_path=str(db_path), vision_enabled=enabled),
                    None,
                    _Embedding(),
                    _Collection(),
                )
                with patch.object(
                    app,
                    "load_and_split_document",
                    return_value=[Chunk("text", {"source": "paper.pdf"})],
                ):
                    app.add_document_to_db(str(pdf), runtime=runtime)
                target = db_path / "source_pdfs" / f"{file_sha256(pdf)}.pdf"
                self.assertEqual(target.is_file(), enabled)
                if enabled:
                    self.assertEqual(target.read_bytes(), pdf.read_bytes())

    def test_non_pdf_is_not_persisted(self):
        with TemporaryDirectory() as directory:
            text = Path(directory) / "notes.txt"
            text.write_text("text", encoding="utf-8")
            runtime = app.Runtime(
                app.RuntimeConfig(db_path=directory, vision_enabled=True),
                None,
                _Embedding(),
                _Collection(),
            )
            with patch.object(
                app,
                "load_and_split_document",
                return_value=[Chunk("text", {"source": "notes.txt"})],
            ):
                app.add_document_to_db(str(text), runtime=runtime)
            self.assertFalse((Path(directory) / "source_pdfs").exists())

    def test_routed_figure_calls_vision_once_with_two_images_and_no_gold(self):
        with TemporaryDirectory() as directory:
            source = "2602.08213v1.pdf"
            digest = "a" * 64
            collection = _Collection(
                [("1", "DrugR evidence", {"source": source, "document_sha256": digest})]
            )
            client = _Client()
            runtime = app.Runtime(
                app.RuntimeConfig(
                    db_path=directory,
                    vision_enabled=True,
                    document_routing=True,
                ),
                client,
                _Embedding(),
                collection,
            )
            source_pdf = Path(directory) / "source_pdfs" / f"{digest}.pdf"
            source_pdf.parent.mkdir()
            source_pdf.write_bytes(b"pdf")
            image = {
                "data_url": "data:image/png;base64,full",
                "page": 2,
                "detail": {"data_url": "data:image/png;base64,detail"},
            }
            with patch.object(app, "render_figure", return_value=image):
                answer, error, metadata = app._vision_answer(
                    f"{source} Figure 2 中显示什么？", runtime
                )
            self.assertIsNone(error)
            self.assertIn("第 2 页", answer)
            self.assertEqual(metadata["source"], source)
            self.assertEqual(len(client.calls), 1)
            payload = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
            self.assertIn("data:image/png;base64,full", payload)
            self.assertIn("data:image/png;base64,detail", payload)
            self.assertNotIn("ground_truth", payload)
            self.assertNotIn("contexts", payload)
            self.assertNotIn("required_facts", payload)

    def test_empty_vision_response_is_retried_once(self):
        with TemporaryDirectory() as directory:
            source = "2602.08213v1.pdf"
            digest = "c" * 64
            client = _Client(content=["", "retry answer"])
            runtime = app.Runtime(
                app.RuntimeConfig(
                    db_path=directory,
                    vision_enabled=True,
                    document_routing=True,
                ),
                client,
                _Embedding(),
                _Collection([("1", "DrugR evidence", {"source": source, "document_sha256": digest})]),
            )
            source_pdf = Path(directory) / "source_pdfs" / f"{digest}.pdf"
            source_pdf.parent.mkdir()
            source_pdf.write_bytes(b"pdf")
            image = {"data_url": "data:image/png;base64,full", "page": 2}
            with patch.object(app, "render_figure", return_value=image):
                answer, error, _ = app._vision_answer(f"{source} Figure 2 中显示什么？", runtime)
            self.assertIsNone(error)
            self.assertIn("retry answer", answer)
            self.assertEqual(len(client.calls), 2)

    def test_disabled_ambiguous_missing_pdf_and_table_do_not_call_vision(self):
        with TemporaryDirectory() as directory:
            source = "2602.08213v1.pdf"
            digest = "b" * 64
            client = _Client()
            records = [("1", "DrugR evidence", {"source": source, "document_sha256": digest})]
            runtime = app.Runtime(
                app.RuntimeConfig(db_path=directory, document_routing=True),
                client,
                _Embedding(),
                _Collection(records),
            )
            answer, error, metadata = app._vision_answer(
                f"{source} Figure 2 中显示什么？", runtime
            )
            self.assertIsNone(answer)
            self.assertIsNone(error)
            self.assertIsNone(metadata)
            runtime = app.Runtime(
                app.RuntimeConfig(
                    db_path=directory, vision_enabled=True, document_routing=True
                ),
                client,
                _Embedding(),
                _Collection(records),
            )
            answer, error, metadata = app._vision_answer(
                f"{source} Figure 2 中显示什么？", runtime
            )
            self.assertIsNone(answer)
            self.assertIn("回退文本检索", error)
            self.assertIsNone(metadata)
            runtime = app.Runtime(
                app.RuntimeConfig(
                    db_path=directory, vision_enabled=True, document_routing=True
                ),
                client,
                _Embedding(),
                _Collection(
                    records + [("2", "Other evidence", {"source": "other.pdf", "document_sha256": "c" * 64})]
                ),
            )
            self.assertEqual(
                app._vision_answer("Figure 2 中显示什么？", runtime),
                (None, None, None),
            )
            answer, error, metadata = app._vision_answer(
                f"{source} Table 2 的数值是什么？", runtime
            )
            self.assertIsNone(answer)
            self.assertIsNone(error)
            self.assertIsNone(metadata)
            self.assertEqual(client.calls, [])

    def test_api_error_returns_text_rag_fallback_note_without_secret(self):
        with TemporaryDirectory() as directory:
            source = "2602.08213v1.pdf"
            digest = "d" * 64
            source_pdf = Path(directory) / "source_pdfs" / f"{digest}.pdf"
            source_pdf.parent.mkdir()
            source_pdf.write_bytes(b"pdf")
            client = _Client(error=RuntimeError("secret should not be shown"))
            runtime = app.Runtime(
                app.RuntimeConfig(
                    db_path=directory,
                    vision_enabled=True,
                    document_routing=True,
                ),
                client,
                _Embedding(),
                _Collection(
                    [("1", "DrugR evidence", {"source": source, "document_sha256": digest})]
                ),
            )
            image = {
                "data_url": "data:image/png;base64,full",
                "page": 2,
                "detail": {"data_url": "data:image/png;base64,detail"},
            }
            with patch.object(app, "render_figure", return_value=image):
                answer, error, metadata = app._vision_answer(
                    f"{source} Figure 2 中显示什么？", runtime
                )
            self.assertIsNone(answer)
            self.assertIsNone(metadata)
            self.assertIn("回退文本检索", error)
            self.assertNotIn("secret should not be shown", error)

    def test_empty_text_retrieval_keeps_vision_fallback_notice(self):
        runtime = app.Runtime(
            app.RuntimeConfig(vision_enabled=True, document_routing=True),
            None,
            _Embedding(),
            _EmptyRetrievalCollection(),
        )
        with patch.object(
            app,
            "_vision_answer",
            return_value=(None, "视觉问答不可用，已回退文本检索（RuntimeError）。", None),
        ):
            result = app.query_knowledge("Figure 2 中显示什么？", runtime=runtime)
        self.assertIn("未找到相关内容", result["answer"])
        self.assertIn("回退文本检索", result["answer"])


if __name__ == "__main__":
    unittest.main()
