"""Proposer choices must produce distinct solver groups when player counts differ."""

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.iteration import group_by_task
from open_r1.self_evolve.policy_clevr_generator import (
    PolicyCLEVRGeneratorConfig,
    PolicyControlledCLEVRGenerator,
    SPY_PLAYER_PROMPT,
    _spy_player_prompt,
)
from open_r1.self_evolve.proposer import score_proposals, tag_counterfactual_pairs
from open_r1.self_evolve.regret import counterfactual_sensitivity, task_statistics


class ProposerTaskIdTests(unittest.TestCase):
    def test_counterfactual_tags_require_matching_player_counts(self):
        proposals = [
            {"proposal_id": "small3", "scene_id": "scene", "keep": [0], "num_players": 3},
            {"proposal_id": "small5", "scene_id": "scene", "keep": [0], "num_players": 5},
            {"proposal_id": "large3", "scene_id": "scene", "keep": [0, 1], "num_players": 3},
            {"proposal_id": "large5", "scene_id": "scene", "keep": [0, 1], "num_players": 5},
        ]
        tagged = {p["proposal_id"]: p for p in tag_counterfactual_pairs(proposals)}
        self.assertEqual(tagged["small3"]["pair_id"], tagged["large3"]["pair_id"])
        self.assertEqual(tagged["small5"]["pair_id"], tagged["large5"]["pair_id"])
        self.assertNotEqual(tagged["small3"]["pair_id"], tagged["small5"]["pair_id"])
        mismatched = tag_counterfactual_pairs([proposals[0], proposals[3]])
        self.assertTrue(all("pair_id" not in p for p in mismatched))
        stale = tag_counterfactual_pairs([
            dict(proposals[0], pair_id="old", pair_role="small"),
            dict(proposals[3], pair_id="old", pair_role="large"),
        ])
        self.assertTrue(all("pair_id" not in p for p in stale))

    def test_interleaved_player_counts_keep_pair_spy_slot(self):
        class LastSlotRandom:
            def randrange(self, count):
                return count - 1

        generator = PolicyControlledCLEVRGenerator.__new__(PolicyControlledCLEVRGenerator)
        generator.config = PolicyCLEVRGeneratorConfig(
            dataset_root="unused", num_tasks=3, num_players=3,
        )
        proposals = tag_counterfactual_pairs([
            {"scene_id": "scene", "keep": [0], "num_players": 5},
            {"scene_id": "scene", "keep": [0], "num_players": 3},
            {"scene_id": "scene", "keep": [0, 1], "num_players": 5},
        ])
        plans = generator._plan_from_proposals(proposals, LastSlotRandom(), budget=3)
        self.assertEqual([p["spy_player"] for p in plans], [5, 3, 5])
        self.assertEqual(plans[0]["pair_id"], plans[2]["pair_id"])
        self.assertIsNone(plans[1]["pair_id"])

    def test_sensitivity_counts_only_verified_one_object_pairs(self):
        def task(name, pair, role, scene, players, spy, keep):
            return {
                "task_id": name, "pair_id": pair, "pair_role": role,
                "scene_id": scene,
                "metadata": {
                    "num_players": players, "spy_player": spy,
                    "variant": {"keep_indices": keep},
                },
            }

        tasks = [
            task("valid_s", "valid", "small", "scene", 5, 4, [0]),
            task("valid_l", "valid", "large", "scene", 5, 4, [0, 1]),
            task("players_s", "players", "small", "scene", 3, 2, [0]),
            task("players_l", "players", "large", "scene", 5, 2, [0, 1]),
            task("spy_s", "spy", "small", "scene", 5, 2, [0]),
            task("spy_l", "spy", "large", "scene", 5, 3, [0, 1]),
            task("scene_s", "scene", "small", "first", 5, 2, [0]),
            task("scene_l", "scene", "large", "second", 5, 2, [0, 1]),
            task("keep_s", "keep", "small", "scene", 5, 2, [0]),
            task("keep_l", "keep", "large", "scene", 5, 2, [1, 2]),
        ]
        scored = [
            {"task_id": t["task_id"],
             "completion": ("<answer>spy=1; changed_attributes=1</answer>"
                            if t["pair_role"] == "small" else
                            "<answer>spy=1; changed_attributes=2</answer>")}
            for t in tasks
        ]
        result = counterfactual_sensitivity(scored, tasks)
        self.assertEqual(result["num_pairs"], 1)
        self.assertEqual(result["counterfactual_sensitivity"], 1.0)
        self.assertEqual(result["counterfactual_direction_correct"], 1.0)
        self.assertEqual(counterfactual_sensitivity(scored)["num_pairs"], 0)

    def test_visual_facts_prompt_is_opt_in(self):
        original = SPY_PLAYER_PROMPT.format(
            num_players=5, image_width=320, image_height=240,
        )
        with patch.dict(os.environ):
            os.environ.pop("SELF_EVOLVE_VISUAL_FACTS", None)
            self.assertEqual(_spy_player_prompt(5, 320, 240), original)
        with patch.dict(os.environ, {"SELF_EVOLVE_VISUAL_FACTS": "1"}):
            prompt = _spy_player_prompt(5, 320, 240)
        self.assertLess(prompt.index("<bbox player="), prompt.index("<change>"))
        self.assertLess(prompt.index("<change>"), prompt.index("<answer>"))
        self.assertIn("<change>attribute:before->after</change>", prompt)
        self.assertIn("color, shape, size, or material", prompt)
        self.assertIn("ordinary players' images", prompt)
        self.assertIn("spy's image", prompt)

    def test_player_count_is_part_of_choice_and_solver_group(self):
        with tempfile.TemporaryDirectory() as root:
            config = PolicyCLEVRGeneratorConfig(
                dataset_root=root, num_tasks=8, edit_fraction=1.0, num_players=3,
            )
            generator = PolicyControlledCLEVRGenerator.__new__(PolicyControlledCLEVRGenerator)
            generator.config = config
            generator.root = Path(root)
            generator.images_dir = Path(root) / "images"
            generator.scenes_dir = Path(root) / "scenes"
            scene = {
                "modification": {"replaced_objects": [
                    {"original": {"color": "red"}, "replacement": {"color": "blue"}},
                    {"original": {"color": "green"}, "replacement": {"color": "yellow"}},
                ]},
            }
            generator._load_scene = lambda _scene_id: scene
            proposals = [
                {"proposal_id": "p0", "scene_id": "scene", "keep": [0], "num_players": 3},
                {"proposal_id": "p1", "scene_id": "scene", "keep": [0], "num_players": 5},
                {"proposal_id": "p2", "scene_id": "scene", "keep": [1], "num_players": 3},
                {"proposal_id": "duplicate", "scene_id": "scene", "keep": [0], "num_players": 3},
            ]
            plans = generator._plan_from_proposals(proposals, random.Random(0), budget=8)
            self.assertEqual([p["proposal_id"] for p in plans], ["p0", "p1", "p2"])

            def fake_variant(_scene_id, _scene, keep, _images, _cache):
                suffix = "-".join(map(str, sorted(keep)))
                return {
                    "variant_id": f"scene__keep{suffix}",
                    "spy_image_path": "modified.png",
                    "civilian_image_path": "original.png",
                    "num_attr_changes": len(keep),
                    "gold_boxes": [[0, 0, 10, 10]],
                }

            prompt = SPY_PLAYER_PROMPT.format(
                num_players=3, image_width=320, image_height=240,
            )
            with patch("open_r1.self_evolve.policy_clevr_generator.build_variant", side_effect=fake_variant):
                tasks = generator._build_variant_tasks(plans, prompt, policy_version=0)

            self.assertEqual(len(tasks), 3)
            self.assertEqual(len({t["task_id"] for t in tasks}), 3)
            by_proposal = {t["proposal_id"]: t for t in tasks}
            self.assertEqual(by_proposal["p0"]["base_task_id"], by_proposal["p1"]["base_task_id"])
            self.assertNotEqual(by_proposal["p0"]["task_id"], by_proposal["p1"]["task_id"])
            self.assertEqual(len(by_proposal["p0"]["image_path"]), 3)
            self.assertEqual(len(by_proposal["p1"]["image_path"]), 5)
            self.assertIn("5 players", by_proposal["p1"]["prompt"])

            outcomes = {"p0": [False, True], "p1": [True, True], "p2": [False, False]}
            trajectories = [
                {"task_id": task["task_id"], "reward_scalar": float(correct),
                 "reward_vector": {"answer": float(correct)}}
                for task in tasks for correct in outcomes[task["proposal_id"]]
            ]
            grouped = group_by_task(trajectories)
            self.assertEqual(len(grouped), 3)
            self.assertEqual([len(group) for group in grouped.values()], [2, 2, 2])
            stats = task_statistics(grouped)
            linked = [dict(p, task_id=by_proposal[p["proposal_id"]]["task_id"])
                      for p in proposals[:3]]
            scored = score_proposals(linked, stats)
            self.assertEqual([p["pass_rate"] for p in scored], [0.5, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
