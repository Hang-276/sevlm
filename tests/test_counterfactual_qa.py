"""Counterfactual count supervision requires matching scene/image certificates."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.counterfactual_qa import build_counterfactual_qa_tasks


def _object(color, shape="cube", *, size="small", material="rubber"):
    return {"color": color, "shape": shape, "size": size, "material": material}


def _fixture(directory, *, name="scene", keep=None):
    root = Path(directory)
    scenes = root / "replacement_scenes"
    images = root / "replacement_images"
    scenes.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    original, modified, variant = (images / f"{name}_{suffix}.png"
                                  for suffix in ("original", "modified", "variant"))
    for image in (original, modified, variant):
        image.touch()
    objects = [_object("red"), _object("blue", "sphere"), _object("green")]
    replacements = [
        {"index": 0, "original": dict(objects[0]),
         "replacement": _object("yellow", "cylinder", size="large", material="metal")},
        {"index": 1, "original": dict(objects[1]),
         "replacement": _object("blue", "cube", material="metal")},
    ]
    after = [dict(obj) for obj in objects]
    for edit in replacements:
        after[edit["index"]] = dict(edit["replacement"])
    scene = {"original_scene": {"objects": objects}, "modified_scene": {"objects": after},
             "modification": {"replaced_objects": replacements},
             "original_image": original.name, "modified_image": modified.name}
    scene_path = scenes / f"{name}_comparison.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    selected = list(range(len(replacements))) if keep is None else keep
    num_changes = sum(sum(edit["original"][attr] != edit["replacement"][attr]
                          for attr in ("color", "shape", "size", "material"))
                      for index, edit in enumerate(replacements) if index in selected)
    edited = modified if keep is None else variant
    task = {"task_id": name, "scene_path": str(scene_path),
            "solution": f"<answer>spy=2; changed_attributes={num_changes}</answer>",
            "image_path": [str(original), str(edited), str(original)],
            "metadata": {"spy_player": 2, "num_players": 3}}
    if keep is not None:
        task["metadata"]["variant"] = {"keep_indices": keep,
            "num_attr_changes": num_changes, "spy_image_path": str(edited),
            "civilian_image_path": str(original)}
    return task, scene, scene_path


def _write(scene_path, scene):
    scene_path.write_text(json.dumps(scene), encoding="utf-8")


class CounterfactualQaTests(unittest.TestCase):
    def test_identical_question_different_gold_complete_single_image_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _scene, _scene_path = _fixture(directory)
            rows = build_counterfactual_qa_tasks([task], 16)
            self.assertEqual(len(rows), 8)
            for left, right in zip(rows[::2], rows[1::2]):
                self.assertEqual(left["problem"], right["problem"])
                self.assertEqual(left["prompt"], left["problem"])
                self.assertEqual(left["pair_id"], right["pair_id"])
                self.assertNotEqual(left["solution"], right["solution"])
                self.assertEqual((left["pair_role"], right["pair_role"]), ("original", "edited"))
                self.assertEqual(left["image_path"], [task["image_path"][0]])
                self.assertEqual(right["image_path"], [task["image_path"][1]])
                for row in (left, right):
                    self.assertEqual(len(row["image"]), 1)
                    self.assertEqual(row["image"], row["image_path"])
                    self.assertEqual(row["source_buffer"], "counterfactual_qa")
                    self.assertEqual(row["metadata"]["kind"], "verified_count_qa")
                    self.assertEqual(row["solution"], f"<answer>{row['metadata']['gold_count']}</answer>")
                    self.assertTrue(Path(row["image"][0]).is_file())
                    self.assertNotIn("spy", row["problem"].lower())
                    self.assertNotIn("changed", row["problem"].lower())
                    self.assertNotIn("scene", row["problem"].lower())

    def test_variant_keep_uses_replacement_position_not_object_index(self):
        with tempfile.TemporaryDirectory() as directory:
            task, scene, scene_path = _fixture(directory, keep=[0])
            # Keep position 0 now denotes the edit at object index 1.
            scene["modification"]["replaced_objects"].reverse()
            task["solution"] = "<answer>spy=2; changed_attributes=2</answer>"
            task["metadata"]["variant"]["num_attr_changes"] = 2
            _write(scene_path, scene)
            rows = build_counterfactual_qa_tasks([task], 12)
        self.assertEqual({row["metadata"]["attribute"] for row in rows}, {"shape", "material"})
        for row in rows:
            attr, value = row["metadata"]["attribute"], row["metadata"]["value"]
            objects = [dict(obj) for obj in scene["original_scene"]["objects"]]
            if row["pair_role"] == "edited":
                objects[1] = scene["modification"]["replaced_objects"][0]["replacement"]
            self.assertEqual(row["metadata"]["gold_count"], sum(obj[attr] == value for obj in objects))

    def test_original_questions_are_deduplicated_across_variants_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            full, _scene, _path = _fixture(directory)
            partial = copy.deepcopy(full)
            partial["task_id"] = "partial"
            partial["image_path"][1] = str(Path(directory) / "replacement_images" / "scene_variant.png")
            partial["metadata"]["variant"] = {"keep_indices": [0], "num_attr_changes": 4,
                "spy_image_path": partial["image_path"][1], "civilian_image_path": partial["image_path"][0]}
            partial["solution"] = "<answer>spy=2; changed_attributes=4</answer>"
            rows = build_counterfactual_qa_tasks([full, partial, full], 32, seed=99)
            reverse = build_counterfactual_qa_tasks([partial, full], 32, seed=99)
            capped = build_counterfactual_qa_tasks([partial, full], 5, seed=99)
        self.assertEqual(rows, reverse)
        self.assertEqual(len(rows), 8)
        self.assertEqual(capped, rows[:4])
        original = rows[::2]
        self.assertEqual(len({row["problem"] for row in original}), len(original))
        self.assertEqual(len({row["metadata"]["attribute"] for row in original}), len(original))
        self.assertEqual(len({row["task_id"] for row in rows}), len(rows))

    def test_zero_counts_are_valid_but_net_count_preserving_swaps_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            task, scene, scene_path = _fixture(directory)
            scene["modification"]["replaced_objects"] = [scene["modification"]["replaced_objects"][0]]
            scene["modification"]["replaced_objects"][0]["replacement"] = _object("yellow")
            scene["modified_scene"]["objects"] = [
                _object("yellow"), _object("blue", "sphere"), _object("green")]
            task["solution"] = "<answer>spy=2; changed_attributes=1</answer>"
            _write(scene_path, scene)
            rows = build_counterfactual_qa_tasks([task], 2)
            self.assertEqual({row["metadata"]["gold_count"] for row in rows}, {0, 1})
            scene["modification"]["replaced_objects"] = [
                {"index": 0, "original": _object("red"), "replacement": _object("green")},
                {"index": 2, "original": _object("green"), "replacement": _object("red")},
            ]
            scene["modified_scene"]["objects"] = [
                _object("green"), _object("blue", "sphere"), _object("red")]
            task["solution"] = "<answer>spy=2; changed_attributes=2</answer>"
            _write(scene_path, scene)
            self.assertEqual(build_counterfactual_qa_tasks([task], 10), [])

    def test_invalid_scene_or_edit_cannot_produce_supervision(self):
        cases = {
            "duplicate index": lambda scene: scene["modification"]["replaced_objects"][1].update(index=0),
            "boolean index": lambda scene: scene["modification"]["replaced_objects"][0].update(index=True),
            "out of range": lambda scene: scene["modification"]["replaced_objects"][0].update(index=9),
            "wrong original": lambda scene: scene["modification"]["replaced_objects"][0]["original"].update(color="cyan"),
            "unknown replacement": lambda scene: scene["modification"]["replaced_objects"][0]["replacement"].update(shape="triangle"),
            "missing replacement": lambda scene: scene["modification"]["replaced_objects"][0]["replacement"].pop("color"),
            "unchanged replacement": lambda scene: scene["modification"]["replaced_objects"][0].update(replacement=_object("red")),
            "unknown untouched": lambda scene: scene["original_scene"]["objects"][2].update(material="plastic"),
            "wrong original index": lambda scene: scene["original_scene"]["objects"][0].update(index=1),
            "stale rendered attrs": lambda scene: scene["modified_scene"]["objects"][0].update(color="red"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                task, scene, scene_path = _fixture(directory)
                mutate(scene)
                _write(scene_path, scene)
                self.assertEqual(build_counterfactual_qa_tasks([task], 20), [])

    def test_invalid_task_image_keep_and_gold_certificates_are_rejected(self):
        cases = {
            "duplicate keep": lambda task: task["metadata"]["variant"].update(keep_indices=[0, 0]),
            "empty keep": lambda task: task["metadata"]["variant"].update(keep_indices=[]),
            "boolean keep": lambda task: task["metadata"]["variant"].update(keep_indices=[True]),
            "out of range keep": lambda task: task["metadata"]["variant"].update(keep_indices=[2]),
            "missing keep": lambda task: task["metadata"]["variant"].pop("keep_indices"),
            "stale count": lambda task: task.update(solution="<answer>spy=2; changed_attributes=1</answer>"),
            "stale variant count": lambda task: task["metadata"]["variant"].update(num_attr_changes=6),
            "string variant count": lambda task: task["metadata"]["variant"].update(num_attr_changes="4"),
            "wrong spy index": lambda task: task["metadata"].update(spy_player=1),
            "boolean spy index": lambda task: task["metadata"].update(spy_player=True),
            "wrong player count": lambda task: task["metadata"].update(num_players=4),
            "wrong civilian path": lambda task: task["metadata"]["variant"].update(civilian_image_path=task["image_path"][1]),
            "missing edited image": lambda task: task["metadata"]["variant"].update(spy_image_path="missing.png"),
            "same paired image": lambda task: task["metadata"]["variant"].update(spy_image_path=task["image_path"][0]),
            "missing source scene": lambda task: task.update(scene_path="missing.json"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                task, _scene, _path = _fixture(directory, keep=[0])
                mutate(task)
                self.assertEqual(build_counterfactual_qa_tasks([task], 20), [])

    def test_missing_original_image_and_invalid_limits_do_not_emit_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _scene, _path = _fixture(directory)
            self.assertEqual(build_counterfactual_qa_tasks([None, {}, task], 1), [])
            for limit in (0, -1, 1, True, 2.5):
                self.assertEqual(build_counterfactual_qa_tasks([task], limit), [])
            Path(task["image_path"][0]).unlink()
            self.assertEqual(build_counterfactual_qa_tasks([task], 20), [])

    def test_partial_variant_cannot_claim_the_full_edited_image(self):
        with tempfile.TemporaryDirectory() as directory:
            task, scene, _path = _fixture(directory, keep=[0])
            modified = str(Path(directory) / "replacement_images" / scene["modified_image"])
            task["image_path"][1] = modified
            task["metadata"]["variant"]["spy_image_path"] = modified
            self.assertEqual(build_counterfactual_qa_tasks([task], 20), [])

    def test_inconsistent_image_alias_cannot_source_verified_twins(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _scene, _path = _fixture(directory)
            task["image"] = list(reversed(task["image_path"]))
            self.assertTrue(build_counterfactual_qa_tasks([task], 20))
            task["image"][1] = task["image_path"][0]
            self.assertEqual(build_counterfactual_qa_tasks([task], 20), [])


if __name__ == "__main__":
    unittest.main()
