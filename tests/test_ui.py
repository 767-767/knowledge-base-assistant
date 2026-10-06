import unittest

import gradio as gr

import app


class UiTests(unittest.TestCase):
    def test_theme_can_be_compared_during_launch(self):
        theme = app.app_theme(gr)
        self.assertNotEqual(theme.to_dict(), gr.themes.Soft().to_dict())

    def test_library_delete_controls_support_batch_selection(self):
        class Collection:
            def count(self):
                return 2

            def get(self, **_kwargs):
                return {
                    "metadatas": [
                        {"source": "a.pdf"},
                        {"source": "b.pdf"},
                    ]
                }

        runtime = app.Runtime(app.RuntimeConfig(), None, None, Collection())
        config = app.build_demo(runtime).get_config_file()
        components = config["components"]

        delete_select = next(
            component
            for component in components
            if component.get("props", {}).get("label")
            == "选择一篇或多篇要删除的文档"
        )
        delete_confirm = next(
            component
            for component in components
            if component.get("props", {}).get("elem_id") == "kb-delete-confirm"
        )
        library_grid = next(
            component
            for component in components
            if "kb-library-grid"
            in component.get("props", {}).get("elem_classes", [])
        )
        upload_status = next(
            component
            for component in components
            if component.get("props", {}).get("label") == "上传状态"
        )
        upload_event = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "handle_upload"
        )

        self.assertTrue(delete_select["props"]["multiselect"])
        self.assertEqual(delete_select["props"]["value"], [])
        self.assertFalse(delete_confirm["props"]["value"])
        self.assertFalse(library_grid["props"]["equal_height"])
        self.assertEqual(upload_event["show_progress_on"], [upload_status["id"]])


if __name__ == "__main__":
    unittest.main()
