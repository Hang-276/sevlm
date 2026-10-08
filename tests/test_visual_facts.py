"""Gold visual changes come from the rendered scene, not response keywords."""

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.visual_facts import (
    gold_changes_from_task,
    score_visual_changes,
)
from open_r1.self_evolve.rewards import parse_structured_answer, structured_exact_match_reward


class VisualFactsTests(unittest.TestCase):
    def setUp(self):
        self.objects = [
            {
                "index": 41,
                "original": {"color": "purple", "shape": "cube", "size": "small", "material": "metal"},
                "replacement": {"color": "brown", "shape": "sphere", "size": "small", "material": "metal"},
            },
            {
                "index": 7,
                "original": {"color": "green", "shape": "cylinder", "size": "small", "material": "metal"},
                "replacement": {"color": "red", "shape": "cube", "size": "small", "material": "metal"},
            },
        ]

    def test_fresh_scene_uses_all_replaced_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path = Path(directory) / "scene.json"
            scene_path.write_text(json.dumps({"modification": {"replaced_objects": self.objects}}))
            facts = gold_changes_from_task({"scene_path": str(scene_path)})
        self.assertEqual(facts, [
            {"attribute": "color", "before": "purple", "after": "brown"},
            {"attribute": "shape", "before": "cube", "after": "sphere"},
            {"attribute": "color", "before": "green", "after": "red"},
            {"attribute": "shape", "before": "cylinder", "after": "cube"},
        ])

    def test_variant_selects_list_position_and_supports_missing_file_fallback(self):
        task = {
            "scene_path": "/missing/scene.json",
            "metadata": {
                "variant": {"keep_indices": [1]},
                "comparison_data": {"replaced_objects": self.objects},
            },
        }
        self.assertEqual(gold_changes_from_task(task), [
            {"attribute": "color", "before": "green", "after": "red"},
            {"attribute": "shape", "before": "cylinder", "after": "cube"},
        ])
        self.assertEqual(gold_changes_from_task({"metadata": {"self_evolve_task": task}}),
                         gold_changes_from_task(task))
        # A raw trajectory may carry a stale outer scene_path. The wrapped
        # source task is authoritative for its edited-variant selection.
        self.assertEqual(gold_changes_from_task({
            "scene_path": "/stale/outer/scene.json",
            "metadata": {"self_evolve_task": task},
        }), gold_changes_from_task(task))

    def test_duplicate_and_invalid_tags_reduce_multiset_f1(self):
        gold = [{"attribute": "color", "before": "red", "after": "blue"}]
        correct = "<change>color:red->blue</change>"
        score, details = score_visual_changes(correct * 2, gold)
        self.assertAlmostEqual(score, 2 / 3)
        self.assertEqual((details["true_positive"], details["false_positive"]), (1, 1))

        score, details = score_visual_changes(correct * 2, gold * 2)
        self.assertEqual(score, 1.0)
        self.assertEqual(details["true_positive"], 2)

        score, details = score_visual_changes(
            correct + "<change>texture:smooth->rough</change><change>color:red->green", gold,
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(details["num_invalid"], 2)
        self.assertEqual(details["num_predicted"], 3)

    def test_wrong_before_after_pair_is_not_rewarded(self):
        gold = [
            {"attribute": "color", "before": "purple", "after": "brown"},
            {"attribute": "color", "before": "green", "after": "red"},
        ]
        wrong = "<change>color:purple->red</change><change>color:green->brown</change>"
        score, details = score_visual_changes(wrong, gold)
        self.assertEqual(score, 0.0)
        self.assertEqual((details["false_positive"], details["false_negative"]), (2, 2))

    def test_missing_or_bad_metadata_cannot_pay(self):
        self.assertEqual(gold_changes_from_task({}), [])
        bad_variant = {
            "metadata": {
                "variant": {"keep_indices": [4]},
                "comparison_data": {"replaced_objects": self.objects},
            },
        }
        self.assertEqual(gold_changes_from_task(bad_variant), [])
        score, details = score_visual_changes("<change>color:red->blue</change>", [])
        self.assertEqual(score, 0.0)
        self.assertFalse(details["available"])
        for invalid in (1, True, "color:red->blue", {"attribute": "color"}):
            with self.subTest(gold=invalid):
                self.assertEqual(score_visual_changes("<change>color:red->blue</change>", invalid)[0], 0.0)

    def test_inconsistent_scene_label_and_duplicate_variant_indices_fail_closed(self):
        objects = self.objects
        base = {"metadata": {"comparison_data": {"replaced_objects": objects}}}
        wrong_label = {**base, "solution": "<answer>spy=2; changed_attributes=9</answer>"}
        self.assertEqual(gold_changes_from_task(wrong_label), [])
        duplicate_keep = {
            "metadata": {"comparison_data": {"replaced_objects": objects},
                         "variant": {"keep_indices": [0, 0]}},
        }
        self.assertEqual(gold_changes_from_task(duplicate_keep), [])

    def test_malformed_variants_and_attribute_values_fail_closed(self):
        for variant in (True, [], {}, {"num_attr_changes": 4},
                        {"keep_indices": [0], "num_attr_changes": 2.9},
                        {"keep_indices": [0], "num_attr_changes": "2"}):
            task = {"metadata": {"comparison_data": {"replaced_objects": self.objects},
                                 "variant": variant}}
            with self.subTest(variant=variant):
                self.assertEqual(gold_changes_from_task(task), [])
        objects = copy.deepcopy(self.objects)
        objects[0]["replacement"]["material"] = "plastic"
        self.assertEqual(gold_changes_from_task({
            "metadata": {"comparison_data": {"replaced_objects": objects}},
        }), [])

    def test_oversized_count_is_unavailable_instead_of_raising(self):
        task = {"solution": "changed_attributes=" + "9" * 5000,
                "metadata": {"comparison_data": {"replaced_objects": self.objects}}}
        self.assertEqual(gold_changes_from_task(task), [])
        bad = "<answer>spy=" + "9" * 5000 + "; changed_attributes=4</answer>"
        self.assertIsNone(parse_structured_answer(bad)["spy"])
        self.assertEqual(structured_exact_match_reward(
            bad, "<answer>spy=2; changed_attributes=4</answer>",
        )[0], 0.0)

    def test_existing_invalid_scene_cannot_fall_back_to_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            task = {"scene_path": str(path),
                    "metadata": {"comparison_data": {"replaced_objects": self.objects}}}
            for payload in ("null", "{", '{"modification": {"replaced_objects": null}}'):
                with self.subTest(payload=payload):
                    path.write_text(payload)
                    self.assertEqual(gold_changes_from_task(task), [])

    def test_misplaced_or_unbalanced_change_tags_cannot_make_full_certificate(self):
        gold = [{"attribute": "color", "before": "red", "after": "blue"}]
        tag = "<change>color:red->blue</change>"
        for completion in (
            "<think>" + tag + "</think><answer>x</answer>",
            "<think>x</think><answer>x</answer>" + tag,
            "<think>x</think>" + tag + "<answer>x</answer></change>",
        ):
            with self.subTest(completion=completion):
                score, details = score_visual_changes(completion, gold)
                self.assertLess(score, 1.0)
                self.assertGreater(details["num_invalid"], 0)


if __name__ == "__main__":
    unittest.main()
