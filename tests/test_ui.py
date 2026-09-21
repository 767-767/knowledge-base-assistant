import unittest

import gradio as gr

import app


class UiTests(unittest.TestCase):
    def test_theme_can_be_compared_during_launch(self):
        theme = app.app_theme(gr)
        self.assertNotEqual(theme.to_dict(), gr.themes.Soft().to_dict())


if __name__ == "__main__":
    unittest.main()
