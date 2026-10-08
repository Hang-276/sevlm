"""Count QA pairs with fixed questions and verified, image-dependent answers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from open_r1.self_evolve.visual_facts import ATTRIBUTES, ATTRIBUTE_VALUES, gold_changes_from_task


def _attributes(obj: Any) -> Optional[Dict[str, str]]:
    if not isinstance(obj, dict):
        return None
    attrs: Dict[str, str] = {}
    for attribute in ATTRIBUTES:
        value = obj.get(attribute)
        if not isinstance(value, str) or value.strip().lower() not in ATTRIBUTE_VALUES[attribute]:
            return None
        attrs[attribute] = value.strip().lower()
    return attrs


def _image(path: Any, scene_path: Path) -> Optional[Path]:
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    image = Path(path)
    candidates = [image] if image.is_absolute() else [
        scene_path.parent / image,
        scene_path.parent.parent / "replacement_images" / image,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _fact_counts(facts: Iterable[Dict[str, str]]) -> Counter:
    return Counter((fact.get("attribute"), fact.get("before"), fact.get("after"))
                   for fact in facts)


def _verified_scene(task: Dict[str, Any]) -> Optional[Tuple[Path, Path, Path, List[dict], List[dict]]]:
    path = task.get("scene_path")
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    scene_path = Path(path)
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(scene, dict):
        return None
    original_scene = scene.get("original_scene")
    objects = original_scene.get("objects") if isinstance(original_scene, dict) else None
    modification = scene.get("modification")
    replaced = modification.get("replaced_objects") if isinstance(modification, dict) else None
    if not isinstance(objects, list) or not objects or not isinstance(replaced, list) or not replaced:
        return None
    before = [_attributes(obj) for obj in objects]
    if any(attrs is None for attrs in before):
        return None

    edits: List[Tuple[int, dict, dict]] = []
    seen: set[int] = set()
    for item in replaced:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        if type(index) is not int or index < 0 or index >= len(objects) or index in seen:
            return None
        seen.add(index)
        old, new = _attributes(item.get("original")), _attributes(item.get("replacement"))
        if old is None or new is None or old != before[index] or old == new:
            return None
        for obj in (objects[index], item["original"], item["replacement"]):
            if "index" in obj and (type(obj["index"]) is not int or obj["index"] != index):
                return None
        edits.append((index, old, new))

    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        return None
    variant = metadata.get("variant")
    if variant is not None and not isinstance(variant, dict):
        return None
    if variant is not None:
        keep = variant.get("keep_indices")
        if (not isinstance(keep, list) or not keep
                or any(type(i) is not int or i < 0 or i >= len(edits) for i in keep)
                or len(keep) != len(set(keep))):
            return None
    else:
        keep = list(range(len(edits)))

    # keep selects replacement-list positions; each edit carries an object index.
    after = [dict(attrs) for attrs in before]
    expected_facts: List[Dict[str, str]] = []
    for edit_position in sorted(keep):
        index, old, new = edits[edit_position]
        after[index] = dict(new)
        expected_facts.extend(
            {"attribute": attribute, "before": old[attribute], "after": new[attribute]}
            for attribute in ATTRIBUTES if old[attribute] != new[attribute]
        )
    if _fact_counts(gold_changes_from_task(task)) != _fact_counts(expected_facts):
        return None
    if variant is not None and "num_attr_changes" in variant:
        count = variant["num_attr_changes"]
        if type(count) is not int or count != len(expected_facts):
            return None

    # Validate the full render; partial variants reconstruct from the original.
    modified_scene = scene.get("modified_scene")
    if modified_scene is not None:
        modified_objects = modified_scene.get("objects") if isinstance(modified_scene, dict) else None
        full_after = [dict(attrs) for attrs in before]
        for index, _old, new in edits:
            full_after[index] = dict(new)
        if (not isinstance(modified_objects, list) or len(modified_objects) != len(full_after)
                or [_attributes(obj) for obj in modified_objects] != full_after):
            return None

    # Paths must agree with the trusted render/edit certificate and player list.
    original = _image(scene.get("original_image"), scene_path)
    edited = _image(variant.get("spy_image_path") if variant is not None
                    else scene.get("modified_image"), scene_path)
    if original is None or edited is None or original == edited:
        return None
    if variant is not None and len(keep) < len(edits):
        if edited == _image(scene.get("modified_image"), scene_path):
            return None
    if variant is not None and "spliced" in variant:
        spliced = variant["spliced"]
        if type(spliced) is not bool or spliced != (len(keep) < len(edits)):
            return None
    if variant is not None and "num_changed_objects" in variant:
        count = variant["num_changed_objects"]
        if type(count) is not int or count != len(keep):
            return None
    if variant is not None and _image(variant.get("civilian_image_path"), scene_path) != original:
        return None
    player_images, spy = task.get("image_path"), metadata.get("spy_player")
    if (not isinstance(player_images, (list, tuple)) or len(player_images) < 2
            or type(spy) is not int or not 1 <= spy <= len(player_images)):
        return None
    if "num_players" in metadata and (
        type(metadata["num_players"]) is not int or metadata["num_players"] != len(player_images)
    ):
        return None
    for player, image_path in enumerate(player_images, 1):
        expected = edited if player == spy else original
        if _image(image_path, scene_path) != expected:
            return None
    image_alias = task.get("image")
    if image_alias is not None and (
        not isinstance(image_alias, (list, tuple)) or len(image_alias) != len(player_images)
        or any(_image(alias, scene_path) != _image(path, scene_path)
               for alias, path in zip(image_alias, player_images))
    ):
        return None
    return scene_path.resolve(), original, edited, before, after


def _question(attribute: str, value: str) -> str:
    if attribute == "shape":
        target = {"cube": "cubes", "sphere": "spheres", "cylinder": "cylinders"}[value]
    else:
        target = f"{value} objects"
    return f"How many {target} are in the image? Respond with <answer>number</answer> only."


def build_counterfactual_qa_tasks(
    accepted_tasks: Iterable[dict], max_examples: int, seed: int = 1701
) -> List[Dict[str, Any]]:
    """Return complete pairs, at most one per scene/attribute; round the cap down."""
    if type(max_examples) is not int or max_examples < 2:
        return []
    candidates: Dict[Tuple[str, str, str, str], Tuple[str, Path, Path, str, str, int, int, str]] = {}
    for task in accepted_tasks:
        if not isinstance(task, dict):
            continue
        verified = _verified_scene(task)
        if verified is None:
            continue
        scene_path, original, edited, before, after = verified
        for attribute in ATTRIBUTES:
            old_counts = Counter(obj[attribute] for obj in before)
            new_counts = Counter(obj[attribute] for obj in after)
            for value in sorted(set(old_counts) | set(new_counts)):
                old, new = old_counts[value], new_counts[value]
                if old == new:
                    continue
                key = (str(scene_path), attribute, value, str(edited))
                rank = hashlib.sha256(f"{seed}\0".encode() + "\0".join(key).encode()).hexdigest()
                candidate = (rank, original, edited, attribute, value, old, new,
                             str(task.get("task_id") or ""))
                previous = candidates.get(key)
                if previous is None or candidate[-1] < previous[-1]:
                    candidates[key] = candidate

    rows: List[Dict[str, Any]] = []
    used: set[Tuple[str, str]] = set()
    for key, candidate in sorted(candidates.items(), key=lambda item: (item[1][0], item[0])):
        scene_path = key[0]
        _rank, original, edited, attribute, value, old, new, source_task = candidate
        scene_attribute = (scene_path, attribute)
        if scene_attribute in used:
            continue
        used.add(scene_attribute)
        pair_id = "counterfactual_qa::" + hashlib.sha256("\0".join(key).encode()).hexdigest()[:20]
        problem = _question(attribute, value)
        for role, image, count in (("original", original, old), ("edited", edited, new)):
            solution = f"<answer>{count}</answer>"
            rows.append({
                "task_id": f"{pair_id}::{role}",
                "pair_id": pair_id,
                "pair_role": role,
                "problem": problem,
                "prompt": problem,
                "solution": solution,
                "answer": solution,
                "ground_truth": str(count),
                "image": [str(image)],
                "image_path": [str(image)],
                "source_buffer": "counterfactual_qa",
                "rule_type": "verified_count_qa",
                "metadata": {
                    "kind": "verified_count_qa",
                    "pair_id": pair_id,
                    "pair_role": role,
                    "gold_count": count,
                    "attribute": attribute,
                    "value": value,
                    "source_scene_path": scene_path,
                    "source_task_id": source_task,
                },
            })
        if len(rows) + 2 > max_examples:
            break
    return rows
