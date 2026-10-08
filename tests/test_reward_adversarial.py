"""Adversarial CPU checks for the verified-reward path."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.exporters import build_grpo_task_examples
from open_r1.self_evolve.buffer import route_with_config
from open_r1.self_evolve.grounding_iou import parse_evidence_boxes, score_model_bbox_grounding
from open_r1.self_evolve.iteration import score_and_route_trajectories
from open_r1.self_evolve.live_reward import get_last_breakdowns, self_evolve_refined_reward
from open_r1.self_evolve.reward_config import RewardConfigError, load_reward_config
from open_r1.self_evolve.rewards import compute_group_reward_vectors, is_format_valid
import open_r1.self_evolve.live_reward as live_reward


ROOT = Path(__file__).resolve().parents[1]
VISUAL_CONFIG = ROOT / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json"


class BBoxAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_reward_config(VISUAL_CONFIG)
        self.gold = [[40.0, 30.0, 120.0, 90.0]]
        self.base = '<bbox player="2">[40,30,120,90]</bbox>'

    def score(self, bbox):
        return score_model_bbox_grounding(
            bbox, self.gold, 320, 240, [2], self.cfg.grounding_cfg
        )

    def test_unambiguous_attribute_spellings_match_body(self):
        expected = self.score(self.base)["grounding_reward"]
        for bbox in (
            '<bbox player="2" x1="40" y1="30" x2="120" y2="90"></bbox>',
            '<bbox player="2" x1="40,30,120,90"></bbox>',
            '<bbox player="2" bbox2d="[40,30,120,90]"></bbox>',
            '<bbox player="2" coords="[40,30,120,90]"></bbox>',
        ):
            with self.subTest(bbox=bbox):
                self.assertAlmostEqual(self.score(bbox)["grounding_reward"], expected)

    def test_wrong_player_out_of_bounds_and_copied_alt_cannot_earn_iou(self):
        for bbox in (
            '<bbox player="3" x1="40,30,120,90"></bbox>',
            '<bbox player="2" x1="440,330,520,390"></bbox>',
            '<bbox player="2" alt="[40,30,120,90]"></bbox>',
            '<bbox player="2" alt="x1=40,30,120,90"></bbox>',
            '<bbox player="2" x1="40,30,120,90" coords="[50,30,120,90]"></bbox>',
            '<bbox player="2" x1="40,30,120,90"/>',
        ):
            with self.subTest(bbox=bbox):
                self.assertEqual(self.score(bbox)["grounding_reward"], 0.0)

    def test_invalid_extra_box_and_dummy_player_reduce_reward(self):
        good = self.score(self.base)["grounding_reward"]
        extra = self.score(self.base + '<bbox player="2" x1="999,999,1000,1000"></bbox>')
        self.assertLess(extra["grounding_reward"], good)
        self.assertEqual(extra["num_bbox_found"], 2)
        self.assertEqual(extra["num_bbox_valid"], 1)
        dummy = self.score(
            '<bbox player="2" alt="[40,30,120,90]"></bbox>'
            '<bbox player="3">[40,30,120,90]</bbox>'
        )
        self.assertEqual(dummy["grounding_reward"], 0.0)
        self.assertFalse(dummy["bbox_player_id_valid"])

    def test_duplicate_json_keys_cannot_select_a_convenient_player_or_box(self):
        for bbox in (
            '{"player":2,"player":3,"bbox_2d":[40,30,120,90]}',
            '{"player":2,"bbox_2d":[40,30,120,90],"bbox_2d":[45,30,120,90]}',
        ):
            with self.subTest(bbox=bbox):
                scored = self.score(bbox)
                self.assertEqual(scored["grounding_reward"], 0.0)
                self.assertFalse(scored["bbox_valid"])

    def test_unclosed_tag_is_recorded_as_invalid_attempt(self):
        parsed = parse_evidence_boxes('<bbox player="2" x1="40,30,120,90">', 320, 240, [2], "auto")
        self.assertEqual(parsed["num_found"], 1)
        self.assertEqual(parsed["num_valid"], 0)
        self.assertEqual(parsed["bbox_parse_error"], "unclosed_bbox")

    def test_malformed_player_values_cannot_crash_or_coerce_to_gold(self):
        for player in ("9" * 5000, "²", "2.0", "-2", "0", "True", "nan", "inf"):
            with self.subTest(player=player[:20]):
                score = self.score(f'<bbox player="{player}">[40,30,120,90]</bbox>')
                self.assertEqual(score["grounding_reward"], 0.0)
                self.assertFalse(score["bbox_player_id_valid"])
        for player in ("2.0", "2.5", "true", '"2.0"', "9" * 5000):
            with self.subTest(json_player=player[:20]):
                score = self.score('{"player":' + player + ',"bbox_2d":[40,30,120,90]}')
                self.assertEqual(score["grounding_reward"], 0.0)

    def test_invalid_grounding_metadata_is_unavailable(self):
        for ids in ([float("inf")], [float("nan")], [True], [-2], [2.0], ["2"], "2", []):
            with self.subTest(ids=ids):
                score = score_model_bbox_grounding(
                    self.base, self.gold, 320, 240, ids, self.cfg.grounding_cfg,
                )
                self.assertEqual(score["grounding_reward"], 0.0)
        for size in (float("inf"), float("nan"), True, -320, "bad"):
            with self.subTest(size=size):
                score, details = live_reward._live_grounding(self.base, {"grounding": {
                    "gold_evidence_boxes": self.gold, "image_width": size,
                    "image_height": 240, "valid_player_ids": [2],
                }}, self.cfg)
                self.assertEqual(score, 0.0)
                self.assertEqual(details["bbox_invalid_reason"], "invalid_image_size")

    def test_finite_pixel_coordinates_preserve_fractional_precision(self):
        gold = [[40.5, 30.25, 120.75, 90.5]]
        score = score_model_bbox_grounding(
            '<bbox player="2">[40.5,30.25,120.75,90.5]</bbox>',
            gold, 320, 240, [2], self.cfg.grounding_cfg,
        )
        self.assertEqual(score["grounding_reward"], 1.0)

    def test_live_offline_identical_for_attribute_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "comparison.json"
            scene.write_text(json.dumps({"modification": {"replaced_objects": [{
                "original": {"color": "red", "shape": "cube", "size": "small", "material": "metal"},
                "replacement": {"color": "blue", "shape": "cube", "size": "small", "material": "metal"},
            }]}}), encoding="utf-8")
            task = {
                "task_id": "bbox-parity", "scene_path": str(scene),
                "problem": "Find the spy", "solution": "<answer>spy=2; changed_attributes=1</answer>",
                "image_path": ["a.png", "b.png", "c.png"], "gold_bbox": self.gold,
                "metadata": {"spy_player": 2, "num_players": 3, "gold_evidence_boxes": self.gold},
            }
            se = build_grpo_task_examples([task])[0]["self_evolve"]
            boxes = (
                '<bbox player="2" x1="40,30,120,90"></bbox>',
                '<bbox player="3" x1="40,30,120,90"></bbox>',
                '<bbox player="2" x1="440,330,520,390"></bbox>',
                '<bbox player="2" x1="40,30,120,90"></bbox><bbox player="2" x1="999,999,1000,1000"></bbox>',
            )
            completions = [
                '<think>Player 2 has a red cube changed to blue.</think>' + box
                + '<change>color:red->blue</change><answer>spy=2; changed_attributes=1</answer>'
                for box in boxes
            ]
            previous = live_reward._REWARD_CONFIG
            live_reward._REWARD_CONFIG = self.cfg
            try:
                live_scores = self_evolve_refined_reward(
                    completions, solution=[task["solution"]] * len(boxes),
                    self_evolve=[se] * len(boxes), problem=[task["problem"]] * len(boxes),
                )
                live_details = get_last_breakdowns()
            finally:
                live_reward._REWARD_CONFIG = previous

            offline = compute_group_reward_vectors(
                completions, task["solution"], task_metadata={
                    "gold_visual_changes": se["visual_facts"]["gold_visual_changes"],
                    "image_width": 320, "image_height": 240, "valid_player_ids": [2],
                }, gold_evidence_boxes=self.gold, max_reasoning_words=400,
                reward_config=self.cfg, include_details=True,
            )
            for score, details, vector in zip(live_scores, live_details, offline):
                self.assertAlmostEqual(details["grounding"], vector["grounding"])
                self.assertAlmostEqual(score, self.cfg.scalarize(
                    vector, spy_correct=True, answer_correct=True
                ))
            self.assertGreater(live_scores[0], live_scores[1])
            self.assertGreater(live_scores[0], live_scores[2])
            self.assertGreater(live_scores[0], live_scores[3])

            # Correct fields and boxes without the required output structure
            # used to earn almost the whole GRPO scalar despite being
            # ineligible for positive replay.
            malformed = (self.base + '<change>color:red->blue</change>'
                         'spy=2; changed_attributes=1')
            previous = live_reward._REWARD_CONFIG
            live_reward._REWARD_CONFIG = self.cfg
            try:
                malformed_score = self_evolve_refined_reward(
                    [malformed], solution=[task["solution"]],
                    self_evolve=[se], problem=[task["problem"]],
                )[0]
                malformed_details = get_last_breakdowns()[0]
            finally:
                live_reward._REWARD_CONFIG = previous
            self.assertFalse(malformed_details["format_valid"])
            self.assertEqual(malformed_details["format_gate_factor"], 0.0)
            self.assertGreater(malformed_details["grounding"], 0.0)
            self.assertLess(malformed_score, live_scores[0])
            self.assertLessEqual(malformed_score, self.cfg.weights["answer"] + self.cfg.weights["budget"])
            malformed_trajectory = {
                "task_id": task["task_id"], "problem": task["problem"],
                "solution": task["solution"], "completion": malformed,
                "metadata": {**task["metadata"], "self_evolve_task": task},
            }
            with patch.dict("os.environ", {"SELF_EVOLVE_ANSWER_JUDGE": "0"}):
                scored = score_and_route_trajectories(
                    [malformed_trajectory], reward_config=self.cfg
                )[0]
            self.assertAlmostEqual(scored["reward_scalar"], malformed_score)
            self.assertEqual(scored["buffer"], "failure")

            # A saved trajectory may contain keyword-derived boxes or an
            # external grounding score. Neither exists in live GRPO, so neither
            # can make a box-free completion positive in offline routing.
            no_box = ('<think>Player 2 changed a red cube to blue.</think>'
                      '<change>color:red->blue</change>'
                      '<answer>spy=2; changed_attributes=1</answer>')
            trajectory = {
                "task_id": task["task_id"], "problem": task["problem"],
                "solution": task["solution"], "completion": no_box,
                "predicted_evidence_boxes": self.gold, "grounding_score": 1.0,
                "metadata": {**task["metadata"], "self_evolve_task": task},
            }
            with patch.dict("os.environ", {"SELF_EVOLVE_ANSWER_JUDGE": "0"}):
                routed = score_and_route_trajectories([trajectory], reward_config=self.cfg)[0]
            self.assertEqual(routed["reward_vector"]["grounding"], 0.0)
            self.assertEqual(routed["buffer"], "failure")


class RewardConfigAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(VISUAL_CONFIG.read_text(encoding="utf-8"))

    def assert_rejected(self, transform):
        raw = json.loads(json.dumps(self.raw))
        transform(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(RewardConfigError):
                load_reward_config(path)

    def test_nonfinite_and_out_of_range_parameters_are_rejected(self):
        self.assert_rejected(lambda raw: raw["weights"].update(answer=float("nan")))
        self.assert_rejected(lambda raw: raw["routing_thresholds"].update(positive_min_process=float("nan")))
        self.assert_rejected(lambda raw: raw["components"]["answer"].update(fields={"spy": 1.2, "changed_attributes": -0.2}))
        self.assert_rejected(lambda raw: raw["components"]["gating"].update(floor=float("nan")))
        self.assert_rejected(lambda raw: raw["components"]["budget"].update(fallback_tokens=0))
        self.assert_rejected(lambda raw: raw["components"]["grounding"].update(require_correct_player_for_iou="true"))
        self.assert_rejected(lambda raw: raw["components"]["process"].update(require_valid_format_for_aux="true"))
        self.assert_rejected(lambda raw: raw["routing_thresholds"].update(positive_min_vusual_facts=1.0))
        self.assert_rejected(lambda raw: raw["components"]["process"].update(mode=["visual_facts"]))

    def test_nonfinite_reward_cannot_enter_positive_buffer(self):
        cfg = load_reward_config(VISUAL_CONFIG)
        details = {"format_valid": True, "process": {"visual_facts_score": 1.0}}
        base = {"answer": 1.0, "grounding": 1.0, "process": 1.0}
        for name in base:
            vector = {**base, name: float("nan")}
            self.assertEqual(route_with_config(vector, [], cfg, details),
                             ("failure", "invalid_reward_vector"))
        details["process"]["visual_facts_score"] = float("nan")
        self.assertEqual(route_with_config(base, [], cfg, details),
                         ("failure", "correct_answer_but_visual_facts_incomplete"))

    def test_duplicate_or_trailing_answers_are_not_positive_format(self):
        good = '<think>x</think><answer>spy=2; changed_attributes=1</answer>'
        self.assertTrue(is_format_valid(good))
        self.assertFalse(is_format_valid(good + '<answer>spy=3; changed_attributes=1</answer>'))
        self.assertFalse(is_format_valid('leading text ' + good))
        self.assertFalse(is_format_valid(good + '<change>color:red->blue</change>'))
        self.assertTrue(is_format_valid(good + '<bbox player="2">[40,30,120,90]</bbox>'))

    def test_answer_gate_offline_uses_live_exact_match(self):
        raw = json.loads(json.dumps(self.raw))
        raw["components"]["gating"]["mode"] = "answer"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            cfg = load_reward_config(path)
            solution = "<answer>spy=2; changed_attributes=1</answer>"
            completion = "<think>Player 2 changed.</think><answer>spy = 2 ; changed attributes : 1</answer>"
            trajectory = {"task_id": "answer-gate", "problem": "Find spy",
                          "solution": solution, "completion": completion, "metadata": {}}
            with patch.dict("os.environ", {"SELF_EVOLVE_ANSWER_JUDGE": "0"}):
                scored = score_and_route_trajectories([trajectory], reward_config=cfg)[0]
            self.assertEqual(scored["reward_vector"]["answer"], 1.0)
            self.assertEqual(scored["reward_vector"]["reward_details"]["gate_factor"], 0.0)
            self.assertAlmostEqual(scored["reward_scalar"],
                                   cfg.weights["answer"] + cfg.weights["budget"] * scored["reward_vector"]["budget"])


if __name__ == "__main__":
    unittest.main()
