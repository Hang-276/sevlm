"""Exercise mixed-task export, the real reward dispatcher and prompt grammar."""

import json
import math
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve import live_reward
from open_r1.self_evolve.exporters import build_grpo_task_examples, build_mixed_grpo_task_examples
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.self_evolve.verified_qa_reward import render_training_question, score_verified_count_qa
from test_counterfactual_qa import _fixture


CONFIG_PATH = (Path(__file__).resolve().parents[1]
               / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json")


class VerifiedQaIntegrationTests(unittest.TestCase):
    def test_strict_grammar_and_trusted_gold(self):
        context = {"task_kind": "verified_count_qa", "verified_qa": {"gold_count": 2}}
        for bad in ("2", "<answer>02</answer>", "<answer>2.0</answer>",
                    "<answer>2</answer><answer>3</answer>",
                    "<think>2</think><answer>2</answer>",
                    "<answer>2</answer> extra", "<answer>-2</answer>",
                    "<answer>" + "9" * 5000 + "</answer>"):
            with self.subTest(bad=bad[:100]):
                self.assertEqual(score_verified_count_qa(bad, "<answer>2</answer>", context)[0], 0)
        reward, details = score_verified_count_qa(" <answer> 2 </answer>\n", "<answer>2</answer>", context)
        self.assertEqual(reward, 1)
        self.assertTrue(details["qa_available"])
        self.assertEqual(score_verified_count_qa("<answer>2</answer>", "<answer>3</answer>", context)[0], 0)
        for gold in (True, None, 2.0, -1, 1001, math.nan):
            with self.subTest(gold=gold):
                self.assertEqual(score_verified_count_qa(
                    "<answer>2</answer>", "<answer>2</answer>",
                    {"verified_qa": {"gold_count": gold}},
                )[0], 0)
        self.assertEqual(score_verified_count_qa(
            "<answer>0</answer>", "0", {"verified_qa": {"gold_count": 0}},
        )[0], 1)

    def test_prompt_loader_does_not_append_conflicting_think_instruction(self):
        template = "{Question} First output <think>reasoning</think> then <answer>answer</answer>."
        problem = "How many red objects? Respond with <answer>number</answer> only."
        self.assertEqual(render_training_question(problem, {"task_kind": "verified_count_qa"}, template), problem)
        self.assertEqual(render_training_question(problem, {"task_kind": "clevr_spy"}, template), template.format(Question=problem))

    def test_mixture_keeps_budget_game_pairs_and_verified_twins(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = [_fixture(directory, name=f"scene-{i}")[0] for i in range(16)]
            for task in tasks[:2]:
                task["pair_id"] = "protected-game-pair"
            for task in tasks:
                task["problem"] = "Find the spy"
            mixed = build_mixed_grpo_task_examples(tasks, fraction=0.25, seed=7)
            self.assertEqual(mixed, build_mixed_grpo_task_examples(tasks, fraction=0.25, seed=7))
            self.assertEqual(len(mixed), len(tasks))
            qa = [row for row in mixed if row["self_evolve"]["task_kind"] == "verified_count_qa"]
            self.assertEqual(len(qa), 4)
            spy_ids = {row["task_id"] for row in mixed if row["self_evolve"]["task_kind"] == "clevr_spy"}
            self.assertTrue({task["task_id"] for task in tasks[:2]} <= spy_ids)
            pairs = defaultdict(list)
            for row in qa:
                pairs[row["self_evolve"]["verified_qa"]["pair_id"]].append(row)
                completion = row["conversations"][1]["value"]
                reward, details = score_verified_count_qa(completion, completion, row["self_evolve"])
                self.assertEqual(reward, 1)
                self.assertIsNone(details["grounding"])
            for rows in pairs.values():
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["conversations"][0], rows[1]["conversations"][0])
                self.assertNotEqual(rows[0]["image"], rows[1]["image"])
                self.assertNotEqual(rows[0]["conversations"][1], rows[1]["conversations"][1])
            # Exercise real Arrow inference of mixed nested contexts and counts.
            from datasets import Dataset
            dataset = Dataset.from_list(mixed)
            self.assertEqual(len(dataset), len(tasks))
            self.assertEqual(sum(row["self_evolve"]["task_kind"] == "verified_count_qa" for row in dataset), 4)

    def test_disabled_missing_images_and_all_paired_rows_preserve_game_data(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = [_fixture(directory, name=f"scene-{i}")[0] for i in range(4)]
            self.assertEqual(build_mixed_grpo_task_examples(tasks, 0), build_grpo_task_examples(tasks))
            for task in tasks:
                task["pair_id"] = "protected"
            self.assertEqual(build_mixed_grpo_task_examples(tasks, 0.5), build_grpo_task_examples(tasks))
        missing_tasks = [{"task_id": str(i)} for i in range(8)]
        self.assertEqual(build_mixed_grpo_task_examples(missing_tasks, 0.5), build_grpo_task_examples(missing_tasks))
        for fraction in (-0.1, 0.51, math.nan, math.inf):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_mixed_grpo_task_examples([], fraction)

    def test_live_dispatch_is_independent_of_twin_batch_order_and_spy_weights(self):
        config = load_reward_config(CONFIG_PATH)
        context = {"task_kind": "verified_count_qa", "verified_qa": {"gold_count": 2}}
        contexts = [context, json.dumps(context), {"task_kind": "verified_count_qa", "verified_qa": {"gold_count": 0}}]
        with patch.object(live_reward, "_REWARD_CONFIG", config):
            rewards = live_reward.self_evolve_refined_reward(
                ["<answer>2</answer>", "<answer>3</answer>", "<answer>0</answer>"],
                solution=["<answer>2</answer>", "<answer>2</answer>", "<answer>0</answer>"],
                self_evolve=contexts,
            )
        self.assertEqual(rewards, [1, 0, 1])
        breakdowns = live_reward.get_last_breakdowns()
        self.assertEqual(len(breakdowns), 3)
        self.assertTrue(all(row["task_kind"] == "verified_count_qa" for row in breakdowns))
        self.assertTrue(all(row["process"] is None for row in breakdowns))


if __name__ == "__main__":
    unittest.main()
