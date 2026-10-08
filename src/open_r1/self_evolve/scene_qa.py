"""Generate small, verifiable single-image QA replay from CLEVR scenes.

Only the *original* scene and its civilian image are used.  In particular, a
replacement image or a benchmark example can never become the answer source.
Each scene contributes at most one count, left/right, and depth question.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from open_r1.self_evolve.visual_facts import ATTRIBUTE_VALUES


_KINDS = ("count", "left_right", "depth")
_ATTRIBUTES = ("size", "color", "material", "shape")
_ATTRIBUTE_WORD = re.compile(r"[a-z][a-z-]*\Z")
_MIN_HORIZONTAL_PIXELS = 32.0  # CLEVR source images are 320 pixels wide.
_MIN_DEPTH_UNITS = 2.0
_MIN_DEPTH_FRACTION = 0.20


def _word(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if _ATTRIBUTE_WORD.fullmatch(value) else None


def _attributes(obj: Any) -> Optional[Tuple[str, str, str, str]]:
    if not isinstance(obj, dict):
        return None
    words = tuple(_word(obj.get(key)) for key in _ATTRIBUTES)
    return words if all(word in ATTRIBUTE_VALUES[key]
                        for key, word in zip(_ATTRIBUTES, words)) else None


def _coordinate(obj: Dict[str, Any], index: int) -> Optional[float]:
    coords = obj.get("pixel_coords")
    if not isinstance(coords, (list, tuple)) or len(coords) <= index:
        return None
    value = coords[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or (index == 2 and value <= 0):
        return None
    return value


def _image_candidates(path: Any, scene_path: Path) -> List[Path]:
    if not isinstance(path, (str, Path)) or not str(path):
        return []
    image = Path(path)
    if image.is_absolute():
        return [image]
    return [
        scene_path.parent / image,
        scene_path.parent.parent / "replacement_images" / image,
    ]


def _first_existing(paths: Iterable[Path]) -> Optional[str]:
    for path in paths:
        if path.is_file():
            return str(path.resolve())
    return None


def _original_image(task: Dict[str, Any], data: Dict[str, Any], scene_path: Path) -> Optional[str]:
    if data.get("original_image") is not None:
        return _first_existing(_image_candidates(data["original_image"], scene_path))

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    variant = metadata.get("variant") if isinstance(metadata.get("variant"), dict) else {}
    if variant.get("civilian_image_path") is not None:
        return _first_existing(_image_candidates(variant["civilian_image_path"], scene_path))

    player_images = task.get("image_path")
    if not isinstance(player_images, (list, tuple)) or not player_images:
        return None
    spy = metadata.get("spy_player")
    if type(spy) is int and 1 <= spy <= len(player_images):
        civilian = [path for index, path in enumerate(player_images, 1) if index != spy]
    else:
        # With no spy index, only a strict majority can identify the civilian.
        counts = Counter(str(path) for path in player_images if isinstance(path, (str, Path)))
        civilian = [path for path, count in counts.most_common(1) if count > len(player_images) / 2]
    images = [_first_existing(_image_candidates(value, scene_path)) for value in civilian]
    return images[0] if images and None not in images and len(set(images)) == 1 else None


def _count_question(objects: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    total = len(objects)
    if total < 3:
        return None
    for attr in ("shape", "color", "material", "size"):
        words = [_word(obj.get(attr)) for obj in objects]
        if any(word is None for word in words):
            continue
        counts = Counter(words)
        for value, count in sorted(counts.items()):
            if 2 <= count < total:
                if attr == "shape":
                    noun = {"cube": "cubes", "sphere": "spheres", "cylinder": "cylinders"}.get(
                        value, f"objects shaped like a {value}"
                    )
                    target = noun
                else:
                    target = f"{value} objects"
                return (
                    f"How many {target} are in the image? "
                    "Respond with <answer>number</answer> only.",
                    str(count),
                )
    return None


def _relation_question(
    objects: List[Dict[str, Any]], kind: str, rng: random.Random
) -> Optional[Tuple[str, str]]:
    descriptions = [" ".join(attrs) if (attrs := _attributes(obj)) else None for obj in objects]
    frequencies = Counter(description for description in descriptions if description)
    valid = [
        (index, obj, descriptions[index])
        for index, obj in enumerate(objects)
        if descriptions[index] and frequencies[descriptions[index]] == 1
    ]
    coordinate_index = 0 if kind == "left_right" else 2
    pairs: List[Tuple[float, str, str, float, float]] = []
    for i, first, first_description in valid:
        first_coord = _coordinate(first, coordinate_index)
        if first_coord is None:
            continue
        for j, second, second_description in valid:
            if j <= i:
                continue
            # Apparent size is a useful depth cue only when the objects have
            # the same physical CLEVR size. Otherwise near/far can be hard to
            # infer from one image even when camera-space depths differ.
            if kind == "depth" and first.get("size") != second.get("size"):
                continue
            second_coord = _coordinate(second, coordinate_index)
            if second_coord is None:
                continue
            gap = abs(first_coord - second_coord)
            if kind == "left_right":
                clear = gap >= _MIN_HORIZONTAL_PIXELS
            else:
                clear = gap >= max(_MIN_DEPTH_UNITS, _MIN_DEPTH_FRACTION * min(first_coord, second_coord))
            if clear:
                pairs.append((gap, first_description, second_description, first_coord, second_coord))
    if not pairs:
        return None
    # Widest separation is easiest to verify visually; lexical tie-breaks are stable.
    _gap, first_description, second_description, first_coord, second_coord = sorted(
        pairs, key=lambda pair: (-pair[0], pair[1], pair[2])
    )[0]
    if rng.randrange(2):
        first_description, second_description = second_description, first_description
        first_coord, second_coord = second_coord, first_coord
    if kind == "left_right":
        answer = "left" if first_coord < second_coord else "right"
        question = (
            f"In the image, is the {first_description} to the left or right of "
            f"the {second_description}? Respond with <answer>left</answer> or "
            "<answer>right</answer> only."
        )
    else:
        # CLEVR pixel_coords[2] is camera-space depth; smaller means closer.
        answer = "closer" if first_coord < second_coord else "farther"
        question = (
            f"Is the {first_description} closer to or farther from the camera "
            f"than the {second_description}? Respond with <answer>closer</answer> "
            "or <answer>farther</answer> only."
        )
    return question, answer


def _scene_records(task: Dict[str, Any], scene_path: Path, seed: int) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    original = data.get("original_scene")
    objects = original.get("objects") if isinstance(original, dict) else None
    if not isinstance(objects, list) or not objects or any(_attributes(obj) is None for obj in objects):
        return {}
    image = _original_image(task, data, scene_path)
    if not image:
        return {}

    scene_key = str(scene_path.resolve())
    digest = hashlib.sha256(f"{seed}\0{scene_key}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    stem = scene_path.stem
    scene_hash = hashlib.sha256(scene_key.encode("utf-8")).hexdigest()[:10]
    candidates = {
        "count": _count_question(objects),
        "left_right": _relation_question(objects, "left_right", rng),
        "depth": _relation_question(objects, "depth", rng),
    }
    records: Dict[str, Dict[str, Any]] = {}
    for kind, candidate in candidates.items():
        if candidate is None:
            continue
        problem, answer = candidate
        completion = f"<answer>{answer}</answer>"
        records[kind] = {
            "task_id": f"scene_qa::{stem}::{scene_hash}::{kind}",
            "problem": problem,
            "prompt": problem,
            "completion": completion,
            "solution": completion,
            "image": [image],
            "image_path": [image],
            "source_buffer": "scene_qa",
            "reward_vector": None,
            "reward_scalar": None,
            "routing_reason": "original_scene_gold",
            "failure_tags": [],
        }
    return records


def build_scene_qa_sft_examples(
    accepted_tasks: Iterable[dict], max_examples: int, seed: int = 1701
) -> List[Dict[str, Any]]:
    """Build deterministic, capped SFT replay for count, 2D, and depth QA.

    Accepted task variants sharing a scene file are deduplicated.  Rows are
    interleaved by question type so a small cap still covers each available
    skill. Missing or malformed scene/image metadata is silently skipped.
    """
    if max_examples <= 0:
        return []
    tasks = sorted(
        (task for task in accepted_tasks if isinstance(task, dict)),
        key=lambda task: (str(task.get("scene_path") or ""), str(task.get("task_id") or "")),
    )
    seen: set[str] = set()
    by_kind: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in _KINDS}
    for task in tasks:
        path_value = task.get("scene_path")
        if not isinstance(path_value, (str, Path)) or not str(path_value):
            continue
        scene_path = Path(path_value)
        scene_key = str(scene_path.resolve())
        if scene_key in seen:
            continue
        records = _scene_records(task, scene_path, seed)
        if not records:
            # A later variant may carry the civilian-image fallback that this
            # task lacks, so only claim the scene after producing usable QA.
            continue
        seen.add(scene_key)
        for kind, record in records.items():
            by_kind[kind].append(record)

    rng = random.Random(seed)
    for records in by_kind.values():
        rng.shuffle(records)
    replay: List[Dict[str, Any]] = []
    for index in range(max(map(len, by_kind.values()), default=0)):
        for kind in _KINDS:
            if index < len(by_kind[kind]):
                replay.append(by_kind[kind][index])
                if len(replay) >= max_examples:
                    return replay
    return replay
