"""Evidence-aware curriculum checks using the production offline reward schema."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.regret import summarize, task_statistics
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.self_evolve.rewards import compute_group_reward_vectors
from open_r1.self_evolve.visual_facts import gold_changes_from_task


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_reward_config(ROOT / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json")
BOX = '<bbox player="2">[40,30,120,90]</bbox>'
CHANGE = '<change>color:red->blue</change>'


def make_task(task_id="task-a", *, variant=False):
    replaced = [
        {"original": {"color": "red", "shape": "cube", "size": "small", "material": "metal"},
         "replacement": {"color": "blue", "shape": "cube", "size": "small", "material": "metal"}},
        {"original": {"color": "green", "shape": "sphere", "size": "large", "material": "rubber"},
         "replacement": {"color": "yellow", "shape": "sphere", "size": "large", "material": "rubber"}},
    ]
    metadata = {
        "spy_player": 2, "num_players": 3,
        "comparison_data": {"replaced_objects": replaced, "gold_evidence_boxes": [[40, 30, 120, 90]]},
        "gold_evidence_boxes": [[40, 30, 120, 90]],
    }
    if variant:
        metadata["variant"] = {"keep_indices": [0], "num_attr_changes": 1,
                               "image_width": 320, "image_height": 240}
    return {
        "task_id": task_id, "scene_id": "scene-a", "scene_path": "/missing/scene.json",
        "solution": "<answer>spy=2; changed_attributes=1</answer>", "metadata": metadata,
    }


def make_example(task, *, spy=2, count=1, change=CHANGE, box=BOX, scalar=0.7):
    completion = ("<think>Player 2 is the spy; its red cube became blue.</think>"
                  + box + change
                  + f"<answer>spy={spy}; changed_attributes={count}</answer>")
    metadata = {
        "gold_visual_changes": gold_changes_from_task(task),
        "valid_player_ids": [2], "image_width": 320, "image_height": 240,
    }
    vector = compute_group_reward_vectors(
        [completion], task["solution"], task_metadata=metadata,
        gold_evidence_boxes=[[40, 30, 120, 90]], max_reasoning_words=400,
        reward_config=CONFIG, include_details=True,
    )[0]
    return {"task_id": task["task_id"], "scene_id": task["scene_id"],
            "solution": task["solution"], "completion": completion,
            "reward_vector": vector, "reward_scalar": scalar}


class EvidenceCurriculumTests(unittest.TestCase):
    def test_unverifiable_tasks_do_not_count_as_visual_failures(self):
        task = make_task(variant=True)
        missing = make_task("missing", variant=True)
        groups = {task["task_id"]: [make_example(task)],
                  missing["task_id"]: [make_example(missing)]}
        stats = task_statistics(groups, mastery_mode="verified_visual", accepted_tasks=[task])
        summary = summarize(stats)
        self.assertEqual(summary["num_certified_tasks"], 1)
        self.assertEqual(summary["mean_verified_pass_rate"], 1.0)
        unavailable = summarize(task_statistics(
            groups, mastery_mode="verified_visual", accepted_tasks=[]))
        self.assertEqual(unavailable["num_certified_tasks"], 0)
        self.assertIsNone(unavailable["mean_verified_pass_rate"])

    def test_default_spy_mode_remains_compatible(self):
        task = make_task()
        examples = [make_example(task, count=1), make_example(task, count=2)]
        stats = task_statistics({task["task_id"]: examples})
        self.assertEqual(stats[task["task_id"]]["class"], "mastered")
        self.assertEqual(summarize(stats)["retire"][0]["task_id"], task["task_id"])

    def test_spy_mastered_with_count_error_stays_on_evidence_frontier(self):
        task = make_task(variant=True)
        examples = [make_example(task, count=1, scalar=0.8),
                    make_example(task, count=2, scalar=0.6)]
        stats = task_statistics({task["task_id"]: examples}, mastery_mode="verified_visual",
                                accepted_tasks=[task])
        item = stats[task["task_id"]]
        self.assertEqual(item["pass_rate"], 1.0)
        self.assertEqual(item["verified_pass_rate"], 0.5)
        self.assertEqual(item["class"], "trainable")
        self.assertEqual(item["frontier"], "evidence")
        self.assertEqual(item["keep_indices"], [0])
        summary = summarize(stats)
        self.assertEqual(summary["retire"], [])
        self.assertEqual(summary["seeds"][0]["direction"], "hold")
        self.assertEqual(summary["seeds"][0]["keep_indices"], [0])
        self.assertEqual(summary["seeds"][0]["num_kept"], 1)
        self.assertEqual(summary["seeds"][0]["num_players"], 3)

    def test_incorrect_fact_or_grounding_prevents_retirement(self):
        task = make_task(variant=True)
        for changed in ({"change": '<change>color:red->green</change>'},
                        {"box": '<bbox player="3">[40,30,120,90]</bbox>'},
                        {"box": ''}):
            with self.subTest(changed=changed):
                examples = [make_example(task), make_example(task, **changed)]
                stats = task_statistics({task["task_id"]: examples}, mastery_mode="verified_visual",
                                        accepted_tasks=[task])
                self.assertEqual(stats[task["task_id"]]["class"], "trainable")
                self.assertEqual(summarize(stats)["seeds"][0]["direction"], "hold")

    def test_full_evidence_mastery_retires_and_spy_zero_remains_too_hard(self):
        task = make_task(variant=True)
        perfect = [make_example(task, scalar=0.9), make_example(task, scalar=0.9)]
        stats = task_statistics({task["task_id"]: perfect}, mastery_mode="verified_visual",
                                accepted_tasks=[task])
        self.assertEqual(stats[task["task_id"]]["class"], "mastered")
        self.assertEqual(summarize(stats)["retire"][0]["scene_id"], "scene-a")

        zero = [make_example(task, spy=1), make_example(task, spy=3)]
        stats = task_statistics({task["task_id"]: zero}, mastery_mode="verified_visual",
                                accepted_tasks=[task])
        self.assertEqual(stats[task["task_id"]]["class"], "too_hard")

    def test_scene_retires_only_when_every_evaluated_variant_is_mastered(self):
        mastered = make_task("mastered", variant=True)
        sibling = make_task("sibling", variant=True)
        first = [make_example(mastered), make_example(mastered)]
        cases = (
            ([make_example(sibling), make_example(sibling, count=2)], sibling,
             "trainable"),
            ([make_example(sibling, spy=1), make_example(sibling, spy=3)], sibling,
             "too_hard"),
            ([make_example(sibling), make_example(sibling)],
             {**sibling, "solution": "<answer>spy=2; changed_attributes=9</answer>"},
             "trainable"),
        )
        for second, accepted_sibling, expected_class in cases:
            with self.subTest(expected_class=expected_class,
                              accepted_solution=accepted_sibling["solution"]):
                stats = task_statistics({"mastered": first, "sibling": second},
                                        mastery_mode="verified_visual",
                                        accepted_tasks=[mastered, accepted_sibling])
                summary = summarize(stats)
                self.assertEqual(stats["mastered"]["class"], "mastered")
                self.assertEqual(stats["sibling"]["class"], expected_class)
                self.assertEqual(summary["retire"], [])
                self.assertEqual(summary["retirement_deferred"], 1)
                self.assertEqual(summary["class_counts"]["mastered"], 1)

        both = task_statistics({"mastered": first,
                                "sibling": [make_example(sibling), make_example(sibling)]},
                               mastery_mode="verified_visual",
                               accepted_tasks=[mastered, sibling])
        summary = summarize(both)
        self.assertEqual({item["task_id"] for item in summary["retire"]},
                         {"mastered", "sibling"})
        self.assertEqual(summary["retirement_deferred"], 0)

    def test_unverifiable_or_duplicate_accepted_task_cannot_retire_or_seed(self):
        task = make_task(variant=True)
        perfect = [make_example(task), make_example(task)]
        for accepted in ([], [task, task],
                         [{**task, "solution": "<answer>spy=2; changed_attributes=9</answer>"}],
                         [{**task, "solution": "<answer>spy=3; changed_attributes=1</answer>"}],
                         [{**task, "metadata": {**task["metadata"], "num_players": True}}]):
            with self.subTest(accepted_count=len(accepted)):
                stats = task_statistics({task["task_id"]: perfect}, mastery_mode="verified_visual",
                                        accepted_tasks=accepted)
                item = stats[task["task_id"]]
                self.assertEqual(item["class"], "trainable")
                self.assertEqual(item["frontier"], "unverifiable")
                self.assertFalse(item["gold_certificate_available"])
                self.assertEqual(summarize(stats)["retire"], [])
                self.assertEqual(summarize(stats)["seeds"], [])

    def test_stale_rollout_solution_cannot_certify_accepted_task(self):
        task = make_task(variant=True)
        stale = make_example(task)
        stale["solution"] = "<answer>spy=3; changed_attributes=1</answer>"
        stats = task_statistics({task["task_id"]: [stale]},
                                mastery_mode="verified_visual", accepted_tasks=[task])
        item = stats[task["task_id"]]
        self.assertTrue(item["gold_certificate_available"])
        self.assertEqual(item["verified_pass_rate"], 0.0)
        self.assertEqual(item["frontier"], "evidence")
        self.assertEqual(summarize(stats)["retire"], [])

    def test_missing_shortcut_audit_cannot_certify_mastery(self):
        task = make_task(variant=True)
        example = make_example(task)
        example["reward_vector"]["reward_details"].pop("shortcut_detected")
        stats = task_statistics({task["task_id"]: [example]},
                                mastery_mode="verified_visual", accepted_tasks=[task])
        self.assertEqual(stats[task["task_id"]]["frontier"], "evidence")
        self.assertEqual(summarize(stats)["retire"], [])

    def test_regret_precedes_evidence_gap_in_seed_order(self):
        a, b, c = (make_task(name, variant=True) for name in ("a", "b", "c"))
        by_task = {
            "a": [make_example(a, count=1, scalar=0.9), make_example(a, count=2, scalar=0.5)],
            "b": [make_example(b, count=2, scalar=0.8), make_example(b, count=2, scalar=0.6)],
            "c": [make_example(c, count=1, scalar=0.8), make_example(c, count=2, scalar=0.6)],
        }
        stats = task_statistics(by_task, mastery_mode="verified_visual", accepted_tasks=[a, b, c])
        seeds = summarize(stats)["seeds"]
        self.assertEqual([s["task_id"] for s in seeds], ["a", "b", "c"])
        self.assertEqual([s["direction"] for s in seeds], ["hold"] * 3)

    def test_verified_mode_requires_gold_authority(self):
        with self.assertRaises(ValueError):
            task_statistics({}, mastery_mode="verified_visual")
        with self.assertRaises(ValueError):
            task_statistics({}, mastery_mode="unknown")


if __name__ == "__main__":
    unittest.main()
