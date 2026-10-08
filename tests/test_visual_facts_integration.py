"""CPU checks for the optional visual certificate training path."""

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.exporters import (
    _cap_auxiliary_sft,
    build_grpo_task_examples,
    build_verified_oracle_replay_examples,
    write_training_exports,
)
from open_r1.self_evolve.iteration import score_and_route_trajectories
from open_r1.self_evolve.buffer import route_with_config
from open_r1.self_evolve.live_reward import (
    get_last_breakdowns,
    self_evolve_refined_reward,
)
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.self_evolve.rewards import (
    compute_group_reward_vectors,
    is_format_valid,
)
from open_r1.self_evolve.visual_facts import score_visual_changes
import open_r1.self_evolve.live_reward as live_reward


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json"


class VisualFactsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        scene_path = Path(self.tmp.name) / "comparison.json"
        original_image = Path(self.tmp.name) / "original.png"
        modified_image = Path(self.tmp.name) / "modified.png"
        original_image.touch()
        modified_image.touch()
        scene_path.write_text(json.dumps({
            "original_image": str(original_image),
            "original_scene": {"objects": [
                {"size": "small", "color": "purple", "material": "metal", "shape": "cube", "pixel_coords": [30, 100, 4]},
                {"size": "small", "color": "red", "material": "metal", "shape": "cube", "pixel_coords": [220, 80, 8]},
                {"size": "large", "color": "yellow", "material": "rubber", "shape": "sphere", "pixel_coords": [150, 120, 7]},
            ]},
            "modification": {"replaced_objects": [{
                "original": {"color": "purple", "shape": "cube", "size": "small", "material": "metal"},
                "replacement": {"color": "brown", "shape": "sphere", "size": "small", "material": "metal"},
            }]},
        }), encoding="utf-8")
        self.task = {
            "task_id": "iter_000::scene",
            "scene_path": str(scene_path),
            "problem": "Find the spy.",
            "solution": "<answer>spy=2; changed_attributes=2</answer>",
            "image_path": [str(original_image), str(modified_image), str(original_image)],
            "gold_bbox": [[10, 20, 90, 100]],
            "metadata": {
                "spy_player": 2,
                "num_players": 3,
                "gold_evidence_boxes": [[10, 20, 90, 100]],
            },
        }

    def test_export_and_oracle_certificate_are_verifiable(self):
        exported = build_grpo_task_examples([self.task])[0]
        gold = exported["self_evolve"]["visual_facts"]["gold_visual_changes"]
        self.assertEqual(len(gold), 2)
        oracle = build_verified_oracle_replay_examples([self.task], 1)[0]
        self.assertTrue(is_format_valid(oracle["completion"]))
        self.assertEqual(score_visual_changes(oracle["completion"], gold)[0], 1.0)
        self.assertIn('<bbox player="2">[10,20,90,100]</bbox>', oracle["completion"])

    def test_invalid_oracle_targets_are_not_exported(self):
        for box in ([10, 20, float("nan"), 100], [90, 20, 10, 100],
                    [-1, 20, 90, 100], [10, 20, 321, 100]):
            task = copy.deepcopy(self.task)
            task["metadata"]["gold_evidence_boxes"] = [box]
            with self.subTest(box=box):
                self.assertEqual(build_verified_oracle_replay_examples([task], 1), [])
        for spy in (True, 2.9, 4):
            task = copy.deepcopy(self.task)
            task["metadata"]["spy_player"] = spy
            with self.subTest(spy=spy):
                self.assertEqual(build_verified_oracle_replay_examples([task], 1), [])
        task = copy.deepcopy(self.task)
        task["solution"] = "<answer>spy=1; changed_attributes=2</answer>"
        self.assertEqual(build_verified_oracle_replay_examples([task], 1), [])

    def test_oracle_selection_is_independent_of_candidate_order(self):
        tasks = [{**copy.deepcopy(self.task), "task_id": f"task-{index}"}
                 for index in range(8)]
        self.assertEqual(build_verified_oracle_replay_examples(tasks, 3),
                         build_verified_oracle_replay_examples(list(reversed(tasks)), 3))

    def test_generator_image_paths_win_over_stale_image_alias(self):
        task = copy.deepcopy(self.task)
        task["image"] = [task["image_path"][0]] * 3
        self.assertEqual(build_grpo_task_examples([task])[0]["image"], task["image_path"])
        oracle = build_verified_oracle_replay_examples([task], 1)[0]
        self.assertEqual(oracle["image"], task["image_path"])
        self.assertEqual(oracle["image_path"], task["image_path"])

    def test_invalid_metadata_cannot_supply_grounding_gold(self):
        for metadata in ([], "bad", {"spy_player": True},
                         {"spy_player": 2, "num_players": 4},
                         {"spy_player": 2, "variant": {"image_width": float("inf")}}):
            task = copy.deepcopy(self.task)
            task["metadata"] = metadata
            context = build_grpo_task_examples([task])[0]["self_evolve"]["grounding"]
            with self.subTest(metadata=metadata):
                if isinstance(metadata, dict):
                    self.assertEqual(context["gold_evidence_boxes"], [])
                self.assertIsInstance(context["image_width"], int)
        task["reference_judge"] = {"reasoning_budget_tokens": float("inf")}
        reference = build_grpo_task_examples([task])[0]["self_evolve"]["reference"]
        self.assertIsNone(reference["reasoning_budget_tokens"])

    def test_live_and_offline_scoring_match_and_false_facts_lose(self):
        cfg = load_reward_config(CONFIG)
        exported = build_grpo_task_examples([self.task])[0]
        se = exported["self_evolve"]
        gold = se["visual_facts"]["gold_visual_changes"]
        good = build_verified_oracle_replay_examples([self.task], 1)[0]["completion"]
        bad = good.replace("color:purple->brown", "color:purple->red")
        previous = live_reward._REWARD_CONFIG
        live_reward._REWARD_CONFIG = cfg
        try:
            live_scores = self_evolve_refined_reward(
                [good, bad],
                solution=[self.task["solution"]] * 2,
                self_evolve=[se, se],
                problem=[self.task["problem"]] * 2,
            )
            live_details = get_last_breakdowns()
        finally:
            live_reward._REWARD_CONFIG = previous

        metadata = {
            "gold_visual_changes": gold,
            "gold_evidence_boxes": self.task["gold_bbox"],
            "valid_player_ids": [2],
            "image_width": 320,
            "image_height": 240,
        }
        offline = compute_group_reward_vectors(
            [good, bad], self.task["solution"],
            task_metadata=metadata,
            gold_evidence_boxes=self.task["gold_bbox"],
            max_reasoning_words=400,
            reward_config=cfg,
            include_details=True,
        )
        for actual, row in zip(live_scores, offline):
            expected = cfg.scalarize(row, spy_correct=True, answer_correct=True)
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(live_details[0]["visual_facts_score"], 1.0)
        self.assertEqual(live_details[1]["visual_facts_score"], 0.5)
        self.assertGreater(live_scores[0], live_scores[1])
        self.assertEqual(route_with_config(offline[0], [], cfg)[0], "positive")
        self.assertEqual(
            route_with_config(offline[1], [], cfg),
            ("failure", "correct_answer_but_visual_facts_incomplete"),
        )

        # Exercise the production offline entry point without an explicit
        # reference budget. It must inherit the same config fallback as GRPO.
        trajectory = {
            "task_id": self.task["task_id"],
            "problem": self.task["problem"],
            "solution": self.task["solution"],
            "completion": good,
            "metadata": {**self.task["metadata"], "self_evolve_task": self.task},
        }
        with patch.dict("os.environ", {"SELF_EVOLVE_ANSWER_JUDGE": "0"}):
            routed = score_and_route_trajectories([trajectory], reward_config=cfg)
        self.assertAlmostEqual(routed[0]["reward_scalar"], live_scores[0], places=6)

    def test_export_starts_sft_with_verified_targets_when_positive_buffer_empty(self):
        with patch.dict("os.environ", {
            "SELF_EVOLVE_VISUAL_FACTS": "1",
            "SELF_EVOLVE_ORACLE_SFT_MAX": "1",
            "SELF_EVOLVE_SCENE_QA_MAX": "3",
            "SELF_EVOLVE_AUX_SFT_MAX_RATIO": "1.0",
        }):
            summary = write_training_exports(
                Path(self.tmp.name) / "exports", [self.task], []
            )
        self.assertEqual(summary["num_sft_solver"], 0)
        self.assertEqual(summary["num_sft_verified_oracle"], 1)
        self.assertEqual(summary["num_sft_scene_qa"], 3)
        self.assertEqual(summary["num_sft_non_proposer"], 4)
        rows = [json.loads(line) for line in Path(summary["sft_replay_jsonl"]).read_text().splitlines()]
        self.assertEqual({row["source_buffer"] for row in rows}, {"verified_oracle", "scene_qa"})
        self.assertTrue(all(row.get("completion") and row.get("image_path") for row in rows))

    def test_auxiliary_sft_anneals_as_real_positives_accumulate(self):
        oracle = [{"task_id": f"oracle-{i}"} for i in range(4)]
        scene_qa = [{"task_id": f"scene-{i}"} for i in range(4)]
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle, scene_qa, 0, 1.0))), (4, 4))
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle, scene_qa, 4, 1.0))), (2, 2))
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle, scene_qa, 20, 1.0))), (4, 4))
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle, scene_qa, 1, 1.0))), (1, 0))
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle, scene_qa, 4, 0.0))), (0, 0))
        self.assertEqual(tuple(map(len, _cap_auxiliary_sft(oracle[:1], scene_qa, 4, 1.0))), (1, 3))
        with self.assertRaises(ValueError):
            _cap_auxiliary_sft(oracle, scene_qa, 0, float("nan"))

    def test_auxiliary_cap_changes_export_counts_but_retains_both_sources(self):
        positives = [
            {
                "task_id": f"positive-{i}",
                "problem": self.task["problem"],
                "completion": self.task["solution"],
                "image_path": self.task["image_path"],
                "buffer": "positive",
            }
            for i in range(3)
        ]
        with patch.dict("os.environ", {
            "SELF_EVOLVE_VISUAL_FACTS": "1",
            "SELF_EVOLVE_ORACLE_SFT_MAX": "1",
            "SELF_EVOLVE_SCENE_QA_MAX": "3",
            "SELF_EVOLVE_AUX_SFT_MAX_RATIO": "1.0",
        }):
            summary = write_training_exports(
                Path(self.tmp.name) / "annealed", [self.task], positives,
            )
        self.assertEqual(summary["num_sft_solver"], 3)
        self.assertEqual(summary["num_sft_verified_oracle"], 1)
        self.assertEqual(summary["num_sft_scene_qa"], 2)
        self.assertEqual(summary["num_sft_non_proposer"], 6)


if __name__ == "__main__":
    unittest.main()
