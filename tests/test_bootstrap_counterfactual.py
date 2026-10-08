"""CPU regression checks for the optional first-round visual curriculum."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.policy_clevr_generator import (
    PolicyCLEVRGeneratorConfig,
    PolicyControlledCLEVRGenerator,
)
from open_r1.self_evolve.regret import counterfactual_sensitivity


def _make_scene(root: Path, index: int) -> None:
    images = root / "output/replacement_images"
    scenes = root / "output/replacement_scenes"
    images.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(parents=True, exist_ok=True)
    base = f"CLEVR_REPLACEMENT_replacement_{index:06d}"
    original = Image.new("RGB", (320, 240), "white")
    modified = Image.new("RGB", (320, 240), "white")
    for image, colors in ((original, ("purple", "red")),
                          (modified, ("brown", "green"))):
        draw = ImageDraw.Draw(image)
        for center, color in zip(((60, 110), (250, 110)), colors):
            x, y = center
            draw.rectangle((x - 20, y - 20, x + 20, y + 20), fill=color)
    original.save(images / f"{base}_original.png")
    modified.save(images / f"{base}_modified.png")
    replaced = []
    for object_index, (x, before, after) in enumerate(((60, "purple", "brown"),
                                                       (250, "red", "green"))):
        shared = {"shape": "cube", "material": "metal", "size": "small",
                  "pixel_coords": [x, 110, 6]}
        replaced.append({
            "index": object_index,
            "original": {**shared, "color": before},
            "replacement": {**shared, "color": after},
        })
    data = {
        "original_scene": {"objects": [item["original"] for item in replaced]},
        "modification": {"replaced_objects": replaced},
    }
    (scenes / f"{base}_comparison.json").write_text(json.dumps(data), encoding="utf-8")


class BootstrapCounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index in range(6):
            _make_scene(self.root, index)
        self.generator = PolicyControlledCLEVRGenerator(PolicyCLEVRGeneratorConfig(
            dataset_root=str(self.root), num_tasks=4, seed=31,
            edit_fraction=0.0, variant_cache_dir=str(self.root / "variants"),
        ))

    def test_no_seed_bootstraps_one_atomic_pair(self):
        with patch.dict("os.environ", {"SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION": "0.5"}):
            tasks = self.generator.generate({})
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len({task["task_id"] for task in tasks}), 4)
        self.assertEqual(self.generator.last_generation_report["num_bootstrap"], 2)
        pair = [task for task in tasks if task["generation_source"] == "bootstrap_counterfactual_variant"]
        self.assertEqual({task["pair_role"] for task in pair}, {"small", "large"})
        self.assertEqual(len({task["pair_id"] for task in pair}), 1)
        self.assertEqual(len({task["scene_id"] for task in pair}), 1)
        self.assertEqual(len({task["problem"] for task in pair}), 1)
        self.assertEqual(len({task["metadata"]["spy_player"] for task in pair}), 1)
        self.assertEqual(sorted(task["metadata"]["variant"]["num_attr_changes"] for task in pair), [1, 2])
        scored = [{"task_id": task["task_id"], "completion": task["solution"]} for task in pair]
        sensitivity = counterfactual_sensitivity(scored, pair)
        self.assertEqual(sensitivity["num_pairs"], 1)
        self.assertEqual(sensitivity["counterfactual_sensitivity"], 1.0)
        self.assertEqual(sensitivity["counterfactual_direction_correct"], 1.0)
        other = [task for task in tasks if task not in pair]
        self.assertTrue(all(task["scene_id"] != pair[0]["scene_id"] for task in other))

    def test_bootstrap_is_opt_in_and_fraction_is_checked(self):
        with patch.dict("os.environ", {"SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION": "0"}):
            tasks = self.generator.generate({})
        self.assertEqual(len(tasks), 4)
        self.assertEqual(self.generator.last_generation_report["num_bootstrap"], 0)
        with patch.dict("os.environ", {"SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION": "nan"}):
            with self.assertRaisesRegex(ValueError, "BOOTSTRAP_PAIR_FRACTION"):
                self.generator.generate({})

    def test_bootstrap_does_not_reuse_a_proposed_scene(self):
        generator = PolicyControlledCLEVRGenerator(PolicyCLEVRGeneratorConfig(
            dataset_root=str(self.root), num_tasks=4, seed=31,
            edit_fraction=0.5, variant_cache_dir=str(self.root / "variants"),
        ))
        proposed_scene = "CLEVR_REPLACEMENT_replacement_000000"
        proposals = [{"scene_id": proposed_scene, "keep": [0], "num_players": 5}]
        with patch.dict("os.environ", {"SELF_EVOLVE_BOOTSTRAP_PAIR_FRACTION": "0.5"}):
            tasks = generator.generate({}, proposals=proposals)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len({task["task_id"] for task in tasks}), 4)
        self.assertEqual(generator.last_generation_report["num_bootstrap"], 2)
        bootstrap = [task for task in tasks
                     if task["generation_source"] == "bootstrap_counterfactual_variant"]
        self.assertTrue(all(task["scene_id"] != proposed_scene for task in bootstrap))

    def test_malformed_proposals_cannot_make_one_player_or_coerced_tasks(self):
        scene_id = "CLEVR_REPLACEMENT_replacement_000000"
        for keep, players in (([0], True), ([0], 1), ([0], "3"), ([0], float("inf")),
                              ([True], 3), ([0.5], 3), ([0, 0], 3), ("0", 3)):
            with self.subTest(keep=keep, players=players):
                proposal = {"scene_id": scene_id, "keep": keep, "num_players": players}
                self.assertEqual(self.generator._plan_from_proposals([proposal], random.Random(1), 2), [])


if __name__ == "__main__":
    unittest.main()
