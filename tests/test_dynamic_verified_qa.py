"""Dynamic datasets retain verified QA rewards and their answer-only prompts."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve import live_reward
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.trainer.dynamic_dataset import (
    CyclicDynamicDataset,
    DynamicIterableDataset,
    EpochAwareIterableDataset,
)


_TEMPLATE = "{Question} First output <think>reasoning</think>, then <answer>answer</answer>."
_QA = {"task_kind": "verified_count_qa", "verified_qa": {
    "gold_count": 2, "pair_id": "pair", "pair_role": "original"}}
_PROBLEM = "How many red objects? Respond with <answer>number</answer> only."


def _sample(context=_QA):
    return {"problem": _PROBLEM, "solution": "2", "image_path": ["/image.png"],
            "self_evolve": copy.deepcopy(context)}


def _dataset(sample, cls=DynamicIterableDataset):
    return cls(lambda *args, **kwargs: copy.deepcopy(sample), epoch_size=2,
               question_prompt=_TEMPLATE)


class DynamicVerifiedQaTests(unittest.TestCase):
    def test_legacy_image_text_and_sudoku_metadata_are_unchanged(self):
        sample = {"problem": "What is shown?", "solution": "cat",
                  "image_path": ["/first.png", "/second.png"],
                  "puzzle_board": [[0]], "solution_board": [[1]], "sudoku_metadata": {"size": 1}}
        dataset = _dataset(sample)
        expected = {"problem": sample["problem"], "solution": "<answer> cat </answer>",
                    "accu_reward_method": "default", "image_path": sample["image_path"],
                    "prompt": [{"role": "user", "content": [
                        {"type": "image", "text": None}, {"type": "image", "text": None},
                        {"type": "text", "text": _TEMPLATE.format(Question=sample["problem"])}]}],
                    "puzzle_board": [[0]], "solution_board": [[1]], "sudoku_metadata": [{"size": 1}]}
        self.assertEqual(dataset._process_sample(sample), expected)
        text = {"problem": "Question", "solution": "<answer>1</answer>"}
        processed = dataset._process_sample(text)
        self.assertEqual(processed["prompt"][0]["content"], [
            {"type": "text", "text": _TEMPLATE.format(Question="Question")}])
        self.assertNotIn("self_evolve", processed)
        special = {"accu_reward_method": "clevr_spotdiff", "game_data": {"players": 3}}
        self.assertIs(dataset._process_sample(special), special)

    def test_qa_context_and_answer_grammar_survive_both_image_and_text_paths(self):
        for image in (True, False):
            with self.subTest(image=image):
                sample = _sample()
                if not image:
                    del sample["image_path"]
                processed = _dataset(sample)._process_sample(sample)
                self.assertEqual(json.loads(processed["self_evolve"]), _QA)
                self.assertEqual(processed["solution"], "<answer> 2 </answer>")
                self.assertEqual(processed["prompt"][0]["content"][-1]["text"], _PROBLEM)
                self.assertNotIn("pair_role", processed["prompt"][0]["content"][-1]["text"])

    def test_serialized_context_is_parsed_once_and_game_prompt_stays_wrapped(self):
        sample = _sample(json.dumps(_QA))
        processed = _dataset(sample)._process_sample(sample)
        self.assertIsInstance(json.loads(processed["self_evolve"]), dict)
        self.assertEqual(json.loads(processed["self_evolve"]), _QA)
        self.assertEqual(processed["prompt"][0]["content"][-1]["text"], _PROBLEM)
        game = {"task_kind": "clevr_spy", "reference": {"reasoning_budget_steps": 3}}
        sample = _sample(game)
        processed = _dataset(sample)._process_sample(sample)
        self.assertEqual(json.loads(processed["self_evolve"]), game)
        self.assertEqual(processed["prompt"][0]["content"][-1]["text"],
                         _TEMPLATE.format(Question=_PROBLEM))

    def test_actual_live_reward_receives_dynamic_verified_context(self):
        rows = list(_dataset(_sample()))
        config_path = (Path(__file__).resolve().parents[1]
                       / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json")
        config = load_reward_config(config_path)
        with patch.object(live_reward, "_REWARD_CONFIG", config):
            reward = live_reward.self_evolve_refined_reward(
                ["<answer>2</answer>", "<answer>9</answer>"],
                solution=[row["solution"] for row in rows],
                self_evolve=[row["self_evolve"] for row in rows],
            )
        self.assertEqual(reward, [1, 0])
        self.assertTrue(all(row["task_kind"] == "verified_count_qa"
                            for row in live_reward.get_last_breakdowns()))

    def test_epoch_and_cyclic_wrappers_keep_context_without_prompt_conflicts(self):
        base = _dataset(_sample(), EpochAwareIterableDataset)
        cyclic = CyclicDynamicDataset(base, num_generations=2, num_iterations=1)
        for dataset in (base, cyclic):
            rows = list(dataset)
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(json.loads(row["self_evolve"]), _QA)
                self.assertEqual(row["prompt"][0]["content"][-1]["text"], _PROBLEM)

    def test_invalid_context_is_rejected_before_reward_routing(self):
        for context in ("{bad json", "[]", [], True, 7):
            with self.subTest(context=context), self.assertRaises(ValueError):
                sample = _sample(context)
                _dataset(sample)._process_sample(sample)


if __name__ == "__main__":
    unittest.main()
