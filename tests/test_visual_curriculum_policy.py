"""Evidence feedback must agree with global and per-scene task difficulty."""

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.iteration_state import default_generator_policy, update_generator_policy
from open_r1.self_evolve.policy_clevr_generator import PolicyCLEVRGeneratorConfig, PolicyControlledCLEVRGenerator


SCENE = {"modification": {"replaced_objects": [
    {"original": {"color": "red"}, "replacement": {"color": "blue"}},
    {"original": {"color": "green"}, "replacement": {"color": "yellow"}},
]}}


class VisualCurriculumPolicyTests(unittest.TestCase):
    def test_missing_answer_feedback_does_not_increase_difficulty(self):
        previous = default_generator_policy()
        for profile in ({}, {"reward_means": {"process": 0.5}}):
            updated = update_generator_policy(previous, profile)
            self.assertEqual(updated["difficulty_policy"], previous["difficulty_policy"])

    def test_solvability_only_feedback_preserves_its_reason(self):
        previous = default_generator_policy()
        updated = update_generator_policy(previous, {"solvability": {
            "mean_solve_rate": 1.0, "mean_verified_pass_rate": 0.25,
        }})
        self.assertEqual(updated["difficulty_policy"], previous["difficulty_policy"])
        self.assertIn("hold_difficulty_until_visual_evidence_mastered",
                      updated["policy_update_reason"])

    def test_missing_certificates_hold_difficulty_without_implying_failure(self):
        previous = default_generator_policy()
        updated = update_generator_policy(previous, {"solvability": {
            "mean_solve_rate": 1.0, "mean_verified_pass_rate": None,
            "num_certified_tasks": 0,
        }})
        self.assertEqual(updated["difficulty_policy"], previous["difficulty_policy"])
        self.assertIn("hold_difficulty_until_visual_certificate_available",
                      updated["policy_update_reason"])

    def test_evidence_not_mastered_holds_global_distribution(self):
        previous = default_generator_policy()
        profile = {"reward_means": {"answer": 1.0}, "solvability": {
            "mean_solve_rate": 1.0, "mean_verified_pass_rate": 0.25,
            "frontier_counts": {"evidence": 8},
        }}
        updated = update_generator_policy(previous, profile)
        self.assertEqual(updated["difficulty_policy"], previous["difficulty_policy"])
        self.assertIn("hold_difficulty_until_visual_evidence_mastered", updated["policy_update_reason"])
        self.assertEqual(updated["solvability_feedback"]["mean_verified_pass_rate"], 0.25)
        self.assertEqual(updated["solvability_feedback"]["frontier_counts"], {"evidence": 8})
        profile["solvability"]["mean_verified_pass_rate"] = 1.0
        mastered = update_generator_policy(previous, profile)
        self.assertGreater(mastered["difficulty_policy"]["distribution"]["hard"],
                           previous["difficulty_policy"]["distribution"]["hard"])

    def test_spy_failure_and_explicit_reference_still_reduce_difficulty(self):
        previous = default_generator_policy()
        profile = {"reward_means": {"answer": 0.0}, "solvability": {
            "mean_solve_rate": 0.0, "mean_verified_pass_rate": 0.0,
        }}
        updated = update_generator_policy(previous, profile)
        self.assertLess(updated["difficulty_policy"]["distribution"]["hard"],
                        previous["difficulty_policy"]["distribution"]["hard"])
        profile["solvability"]["mean_solve_rate"] = 1.0
        reduced = update_generator_policy(previous, profile, {"too_difficult_count": 1})
        self.assertLess(reduced["difficulty_policy"]["distribution"]["hard"],
                        previous["difficulty_policy"]["distribution"]["hard"])

    def test_hold_keeps_subset_and_player_count_and_complete_contrast(self):
        for kept in ([0], [0, 1]):
            with self.subTest(kept=kept), tempfile.TemporaryDirectory() as directory:
                generator = PolicyControlledCLEVRGenerator(PolicyCLEVRGeneratorConfig(
                    dataset_root=directory, num_tasks=4, edit_fraction=1.0,
                ))
                seed = {"scene_id": "scene", "direction": "hold", "keep_indices": kept,
                        "num_players": 7, "regret": 0.1}
                with patch.object(generator, "_load_scene", return_value=SCENE):
                    plans = generator._plan_edits({"seeds": [seed]}, random.Random(3))
                self.assertEqual(len(plans), 2)
                self.assertEqual(list(plans[0]["keep"]), kept)
                self.assertTrue(all(plan["num_players"] == 7 for plan in plans))
                self.assertEqual(plans[0]["spy_player"], plans[1]["spy_player"])
                self.assertEqual(plans[0]["pair_id"], plans[1]["pair_id"])

    def test_bad_hold_does_not_consume_scene_before_a_valid_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = PolicyControlledCLEVRGenerator(PolicyCLEVRGeneratorConfig(
                dataset_root=directory, num_tasks=4, edit_fraction=1.0,
            ))
            seeds = [{"scene_id": "scene", "direction": "hold", "keep_indices": [4]},
                     {"scene_id": "scene", "direction": "hold", "keep_indices": [0]}]
            with patch.object(generator, "_load_scene", return_value=SCENE):
                plans = generator._plan_edits({"seeds": seeds}, random.Random(3))
            self.assertEqual(len(plans), 2)
            self.assertEqual(list(plans[0]["keep"]), [0])


if __name__ == "__main__":
    unittest.main()
