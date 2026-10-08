"""Scene QA replay uses only verifiable original-scene facts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.scene_qa import build_scene_qa_sft_examples


def _object(color, shape, x, depth, *, size="small", material="metal"):
    return {
        "size": size,
        "color": color,
        "material": material,
        "shape": shape,
        "pixel_coords": [x, 100, depth],
    }


def _scene(directory, objects, *, original_image=True):
    root = Path(directory)
    scenes = root / "replacement_scenes"
    images = root / "replacement_images"
    scenes.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    scene_path = scenes / "scene_comparison.json"
    image_path = images / "scene_original.png"
    image_path.touch()
    payload = {"original_scene": {"objects": objects}}
    if original_image:
        payload["original_image"] = image_path.name
    scene_path.write_text(json.dumps(payload), encoding="utf-8")
    return scene_path, image_path


class SceneQaTests(unittest.TestCase):
    def test_three_types_have_checkable_gold_and_single_original_image(self):
        objects = [
            _object("red", "cube", 20, 7.5),
            _object("blue", "sphere", 160, 10, size="large", material="rubber"),
            _object("green", "cube", 290, 13, size="large"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            scene_path, image_path = _scene(directory, objects)
            rows = build_scene_qa_sft_examples([{"scene_path": str(scene_path)}], 10)

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["task_id"].rsplit("::", 1)[-1] for row in rows},
                         {"count", "left_right", "depth"})
        for row in rows:
            self.assertEqual(row["image_path"], [str(image_path)])
            self.assertEqual(row["image"], [str(image_path)])
            self.assertEqual(row["source_buffer"], "scene_qa")
            self.assertEqual(row["solution"], row["completion"])
            self.assertEqual(row["prompt"], row["problem"])
        by_kind = {row["task_id"].rsplit("::", 1)[-1]: row for row in rows}
        self.assertEqual(by_kind["count"]["completion"], "<answer>2</answer>")
        for kind, first_description, second_description, lower, higher in (
            ("left_right", "small red metal cube", "large green metal cube", "left", "right"),
            ("depth", "large blue rubber sphere", "large green metal cube", "closer", "farther"),
        ):
            row = by_kind[kind]
            # The widest clear pair is used. Depth additionally requires
            # equal physical size, so apparent-size evidence is meaningful.
            self.assertIn(first_description, row["problem"])
            self.assertIn(second_description, row["problem"])
            first_is_lower = row["problem"].index(first_description) < row["problem"].index(second_description)
            self.assertEqual(row["completion"],
                             f"<answer>{lower if first_is_lower else higher}</answer>")

    def test_duplicate_variants_and_input_order_are_deterministic(self):
        objects = [
            _object("red", "cube", 20, 7),
            _object("blue", "sphere", 160, 10),
            _object("green", "cube", 290, 13),
        ]
        with tempfile.TemporaryDirectory() as directory:
            scene_path, _image_path = _scene(directory, objects)
            tasks = [
                {"scene_path": str(scene_path), "task_id": "z-variant"},
                {"scene_path": str(scene_path), "task_id": "a-fresh"},
            ]
            first = build_scene_qa_sft_examples(tasks, 3, seed=23)
            reversed_order = build_scene_qa_sft_examples(reversed(tasks), 3, seed=23)
            capped = build_scene_qa_sft_examples(tasks, 2, seed=23)
        self.assertEqual(first, reversed_order)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(capped), 2)
        self.assertEqual(len({row["task_id"] for row in first}), 3)

    def test_duplicate_descriptions_are_not_used_to_reference_objects(self):
        objects = [
            _object("red", "cube", 10, 7),
            _object("red", "cube", 120, 10),
            _object("blue", "sphere", 300, 14),
        ]
        with tempfile.TemporaryDirectory() as directory:
            scene_path, _image_path = _scene(directory, objects)
            rows = build_scene_qa_sft_examples([{"scene_path": str(scene_path)}], 10)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["task_id"].endswith("::count"))
        self.assertEqual(rows[0]["completion"], "<answer>2</answer>")

    def test_spatial_margins_reject_near_ties_and_accept_boundary(self):
        def kinds(x_gap, depth_gap):
            objects = [
                _object("red", "cube", 80, 10),
                _object("blue", "sphere", 80 + x_gap, 10 + depth_gap),
            ]
            with tempfile.TemporaryDirectory() as directory:
                scene_path, _image_path = _scene(directory, objects)
                rows = build_scene_qa_sft_examples([{"scene_path": str(scene_path)}], 10)
            return {row["task_id"].rsplit("::", 1)[-1] for row in rows}

        self.assertEqual(kinds(31.9, 1.99), set())
        self.assertEqual(kinds(32, 2.0), {"left_right", "depth"})

    def test_missing_scene_and_image_are_skipped_and_civilian_fallback_works(self):
        self.assertEqual(build_scene_qa_sft_examples([{}, {"scene_path": "/missing/scene.json"}], 10), [])
        objects = [
            _object("red", "cube", 20, 7),
            _object("blue", "sphere", 290, 13),
        ]
        with tempfile.TemporaryDirectory() as directory:
            scene_path, original_image = _scene(directory, objects, original_image=False)
            original_image.unlink()
            self.assertEqual(build_scene_qa_sft_examples([{"scene_path": str(scene_path)}], 10), [])
            civilian = Path(directory) / "civilian.png"
            spy = Path(directory) / "spy.png"
            civilian.touch()
            spy.touch()
            task = {
                "scene_path": str(scene_path),
                "task_id": "b-has-civilian",
                "image_path": [str(civilian), str(spy), str(civilian)],
                "metadata": {"spy_player": 2},
            }
            rows = build_scene_qa_sft_examples([
                {"scene_path": str(scene_path), "task_id": "a-missing-image"},
                task,
            ], 10)
            self.assertEqual({row["task_id"].rsplit("::", 1)[-1] for row in rows},
                             {"left_right", "depth"})
            self.assertTrue(all(row["image_path"] == [str(civilian)] for row in rows))

    def test_untrusted_image_fallback_and_unknown_attributes_are_skipped(self):
        objects = [_object("red", "cube", 20, 7),
                   _object("blue", "sphere", 290, 13)]
        with tempfile.TemporaryDirectory() as directory:
            scene_path, original = _scene(directory, objects)
            other = Path(directory) / "other.png"
            other.touch()
            task = {"scene_path": str(scene_path),
                    "image_path": [str(other), str(other), str(other)],
                    "metadata": {"spy_player": 2}}
            original.unlink()
            self.assertEqual(build_scene_qa_sft_examples([task], 10), [])
            payload = json.loads(scene_path.read_text())
            payload.pop("original_image")
            scene_path.write_text(json.dumps(payload))
            third = Path(directory) / "third.png"
            third.touch()
            task["image_path"] = [str(other), str(other), str(third)]
            self.assertEqual(build_scene_qa_sft_examples([task], 10), [])
            task["image_path"] = [str(other)] * 3
            payload["original_scene"]["objects"][0]["material"] = "plastic"
            scene_path.write_text(json.dumps(payload))
            self.assertEqual(build_scene_qa_sft_examples([task], 10), [])


if __name__ == "__main__":
    unittest.main()
