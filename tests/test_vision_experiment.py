import json
import unittest

from evaluation.benchmark_loader import load_benchmark
from evaluation.vision_experiment import detail_clip, figure_clip, vision_messages
from sci_rag_core import figure_reference_from_question
from sci_rag_vision import figure_page


class VisionExperimentTests(unittest.TestCase):
    def test_figure_page_prefers_caption_with_image_above_reference_text(self):
        class FakePage:
            def __init__(self, blocks):
                self.blocks = blocks

            def get_text(self, *_args, **_kwargs):
                return self.blocks

        class FakeDocument:
            def __iter__(self):
                return iter(
                    [
                        FakePage([(0, 100, 100, 120, "Fig. 3 presents a result", 0, 0)]),
                        FakePage(
                            [
                                (10, 20, 90, 80, "", 0, 1),
                                (0, 100, 100, 120, "Figure 3: actual caption", 0, 0),
                            ]
                        ),
                    ]
                )

        self.assertEqual(figure_page(FakeDocument(), ("figure", 3)), 2)

    def test_clip_uses_previous_overlapping_caption_and_target_caption(self):
        blocks = [
            (10, 100, 190, 120, "Figure 1. first", 0, 0),
            (10, 135, 190, 155, "Figure 2. target", 1, 0),
            (10, 60, 190, 80, "plot labels", 2, 0),
        ]
        self.assertEqual(
            figure_clip(blocks, 200, 500, ("figure", 2)),
            (0.0, 120.0, 200.0, 135.0, "Figure 2. target"),
        )

    def test_payload_contains_question_and_image_but_no_gold(self):
        messages = vision_messages(
            "Figure 2 中哪个更高？",
            "data:image/png;base64,abc",
            "data:image/png;base64,detail",
        )
        payload = json.dumps(messages, ensure_ascii=False)
        self.assertIn("Figure 2 中哪个更高？", payload)
        self.assertIn("data:image/png;base64,abc", payload)
        self.assertIn("data:image/png;base64,detail", payload)
        self.assertNotIn("ground_truth", payload)
        self.assertNotIn("required_facts", payload)
        self.assertNotIn("contexts", payload)

    def test_detail_crop_is_generic_lower_center(self):
        self.assertEqual(detail_clip((0, 100, 1000, 500, "Figure 1")), (100, 240, 900, 500))

    def test_challenge_selects_exactly_ten_parseable_figures(self):
        benchmark = load_benchmark("evaluation/benchmark/manifest_challenge.json")
        cases = [case for case in benchmark["cases"] if case.get("requires_image") is True]
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(figure_reference_from_question(case["question"]) for case in cases))
