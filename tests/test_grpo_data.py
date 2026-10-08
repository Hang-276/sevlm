import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.grpo_data import normalize_reward_context, prepare_grpo_sample
from open_r1.self_evolve import live_reward
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.trainer.dynamic_dataset import CyclicDynamicDataset, DynamicIterableDataset, EpochAwareIterableDataset


TEMPLATE = "{Question} First output <think>reasoning</think>, then <answer>answer</answer>."
CONTEXT = {"task_kind": "verified_count_qa", "verified_qa": {"gold_count": 2}}


def test_dict_and_serialized_context_survive_arrow_and_reward_dispatch(tmp_path):
    from datasets import Dataset

    image = tmp_path / "image.png"
    image.touch()
    raw = [{"problem": "Count red objects.", "solution": "2", "image_path": [str(image)],
            "self_evolve": context} for context in (CONTEXT, json.dumps(CONTEXT))]
    rows = [prepare_grpo_sample(row, TEMPLATE, check_images=True) for row in raw]
    dataset = Dataset.from_list(rows)
    assert rows[0] == rows[1]
    assert json.loads(dataset[0]["self_evolve"]) == CONTEXT
    assert "<think>" not in dataset[0]["prompt"][0]["content"][-1]["text"]
    assert "gold_count" not in dataset[0]["prompt"][0]["content"][-1]["text"]
    config = load_reward_config(Path(__file__).resolve().parents[1]
                               / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json")
    with patch.object(live_reward, "_REWARD_CONFIG", config):
        rewards = live_reward.self_evolve_refined_reward(
            ["<answer>2</answer>", "<answer>9</answer>"],
            solution=dataset["solution"], self_evolve=dataset["self_evolve"],
        )
    assert rewards == [1.0, 0.0]


@pytest.mark.parametrize("context", ["{bad", "[]", [], True, 7, json.dumps(json.dumps(CONTEXT))])
def test_malformed_and_double_encoded_contexts_are_rejected(context):
    with pytest.raises(ValueError):
        normalize_reward_context(context)


def test_static_and_dynamic_standard_samples_match():
    raw = {"problem": "Count.", "solution": 2, "image_path": ["/a.png"],
           "self_evolve": json.dumps(CONTEXT)}
    dataset = DynamicIterableDataset(lambda: raw, epoch_size=2, question_prompt=TEMPLATE)
    assert prepare_grpo_sample(raw, TEMPLATE) == dataset._process_sample(raw)


def test_bad_image_path_types_and_missing_files_fail_before_training(tmp_path):
    for images in ("/a.png", [None], [""]):
        with pytest.raises(ValueError):
            prepare_grpo_sample({"image_path": images}, TEMPLATE)
    with pytest.raises(ValueError, match="do not exist"):
        prepare_grpo_sample({"image_path": [str(tmp_path / "missing.png")]}, TEMPLATE, check_images=True)


def _base(epoch_size, generator=None):
    counter = iter(range(100))
    if generator is None:
        generator = lambda: {"problem": str(next(counter)), "solution": "0"}
    return DynamicIterableDataset(generator, epoch_size=epoch_size)


def test_dynamic_cycles_repeat_entire_update_batch_and_drop_only_incomplete_tail():
    cyclic = CyclicDynamicDataset(_base(10), num_generations=2, num_iterations=2, batch_size=2)
    values = [row["problem"] for row in cyclic]
    assert values == ["0", "0", "1", "1", "0", "0", "1", "1",
                      "2", "2", "3", "3", "2", "2", "3", "3"]
    assert len(cyclic) == len(values) == 16
    assert [row["problem"] for row in cyclic] == values
    cyclic.set_epoch(1)
    assert next(iter(cyclic))["problem"] == "4"


def test_dynamic_cycles_do_not_share_mutable_prompt_between_replays():
    cyclic = CyclicDynamicDataset(_base(2), num_generations=2, num_iterations=2)
    iterator = iter(cyclic)
    first = next(iterator)
    first["prompt"][0]["content"][0]["text"] = "corrupted"
    assert next(iterator)["prompt"][0]["content"][0]["text"] != "corrupted"


def test_exhausted_dynamic_generation_cannot_create_fake_or_partial_epoch():
    cyclic = CyclicDynamicDataset(_base(4, lambda: None), num_generations=2, num_iterations=2)
    with pytest.raises(RuntimeError, match="complete epoch"):
        list(cyclic)
    assert not cyclic._cached_samples


@pytest.mark.parametrize("name,value", [("num_generations", 0), ("num_generations", True),
                                       ("num_iterations", -1), ("batch_size", 0), ("batch_size", 1.5)])
def test_invalid_dynamic_cycle_sizes_fail_early(name, value):
    kwargs = dict(num_generations=2, num_iterations=1, batch_size=1)
    kwargs[name] = value
    with pytest.raises(ValueError, match="positive integer"):
        CyclicDynamicDataset(_base(4), **kwargs)
    with pytest.raises(ValueError, match="complete GRPO batch"):
        CyclicDynamicDataset(_base(2), num_generations=2, num_iterations=1, batch_size=2)


def test_dynamic_generation_is_rank_independent_without_changing_model_rng():
    def generate(**kwargs):
        numbers = [random.random(), float(np.random.rand()), float(torch.rand(1))]
        return {"problem": json.dumps(numbers), "solution": "0"}

    def rank_rows(rank):
        random.seed(1000 + rank)
        np.random.seed(1000 + rank)
        torch.manual_seed(1000 + rank)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        base = EpochAwareIterableDataset(generate, epoch_size=8, seed=42)
        cyclic = CyclicDynamicDataset(base, num_generations=2, num_iterations=2, batch_size=2)
        cyclic.set_epoch(3)
        rows = list(cyclic)
        assert random.getstate() == python_state
        restored = np.random.get_state()
        assert restored[0] == numpy_state[0] and np.array_equal(restored[1], numpy_state[1])
        assert restored[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
        return rows

    assert rank_rows(0) == rank_rows(1)


def test_generator_internal_typeerror_is_not_retried_with_another_signature():
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        raise TypeError("bad data implementation")

    base = EpochAwareIterableDataset(generate, epoch_size=2, seed=42)
    assert list(base) == []
    assert calls == [{"epoch": 0, "sample_idx": 0}, {"epoch": 0, "sample_idx": 1}]
