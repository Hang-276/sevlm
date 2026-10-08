"""Deterministic visual-change facts and reward for CLEVR replacement tasks.

The scene metadata is reward-side context: it must not be inserted into the
model's prompt. A fact is one changed attribute of one replaced object, stored
as ``attribute``, ``before``, and ``after``. Repeated identical facts retain
their multiplicity when scored.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ATTRIBUTES = ("color", "shape", "size", "material")
ATTRIBUTE_VALUES = {
    "color": frozenset(("gray", "red", "blue", "green", "yellow", "purple", "brown", "cyan")),
    "shape": frozenset(("cube", "sphere", "cylinder")),
    "size": frozenset(("small", "large")),
    "material": frozenset(("rubber", "metal")),
}
_CHANGE_OPEN = re.compile(r"<change\b[^>]*>", re.IGNORECASE)
_CHANGE_CLOSE = re.compile(r"</change\s*>", re.IGNORECASE)
_CHANGE_TAG = re.compile(r"<change>(.*?)</change>", re.IGNORECASE | re.DOTALL)
_CHANGE_BODY = re.compile(
    r"\s*(color|shape|size|material)\s*:\s*"
    r"([a-z][a-z0-9_-]*)\s*->\s*([a-z][a-z0-9_-]*)\s*",
    re.IGNORECASE,
)


def _normalise_fact(fact: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(fact, dict):
        return None
    values = (fact.get("attribute"), fact.get("before"), fact.get("after"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    attribute, before, after = (value.strip().lower() for value in values)
    if (attribute not in ATTRIBUTES or before == after
            or before not in ATTRIBUTE_VALUES[attribute] or after not in ATTRIBUTE_VALUES[attribute]):
        return None
    return attribute, before, after


def _task_and_metadata(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Accept both generator tasks and raw trajectories wrapping one."""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    wrapped = metadata.get("self_evolve_task")
    if isinstance(wrapped, dict):
        task = wrapped
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return task, metadata


def _replaced_objects(task: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[List[Any]]:
    path = task.get("scene_path")
    if isinstance(path, (str, Path)) and str(path):
        try:
            scene = json.loads(Path(path).read_text(encoding="utf-8"))
            objects = (scene.get("modification") or {}).get("replaced_objects")
        except OSError:
            pass
        except (ValueError, TypeError, AttributeError):
            return None
        else:
            return objects if isinstance(objects, list) else None

    comparison = metadata.get("comparison_data")
    if isinstance(comparison, dict):
        objects = comparison.get("replaced_objects")
        if isinstance(objects, list):
            return objects
    return None


def gold_changes_from_task(task: dict) -> List[Dict[str, str]]:
    """Return gold ``attribute/before/after`` triples for the visible changes.

    Edited variants select positions in ``replaced_objects`` via
    ``metadata.variant.keep_indices``; these are list positions, not the
    objects' own ``index`` fields. Fresh tasks use every replaced object.
    Missing or inconsistent metadata returns ``[]`` (unavailable).
    """
    if not isinstance(task, dict):
        return []
    task, metadata = _task_and_metadata(task)
    objects = _replaced_objects(task, metadata)
    if not objects:
        return []

    variant = metadata.get("variant")
    if variant is not None:
        if not isinstance(variant, dict):
            return []
        keep = variant.get("keep_indices")
        if (
            not isinstance(keep, list)
            or not keep
            or any(type(index) is not int or index < 0 or index >= len(objects) for index in keep)
            or len(set(keep)) != len(keep)
        ):
            return []
        selected = [objects[index] for index in sorted(set(keep))]
    else:
        selected = objects

    facts: List[Dict[str, str]] = []
    for item in selected:
        if not isinstance(item, dict):
            return []
        before = item.get("original")
        after = item.get("replacement")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return []
        for attribute in ATTRIBUTES:
            if not all(isinstance(values.get(attribute), str)
                       and values[attribute].strip().lower() in ATTRIBUTE_VALUES[attribute]
                       for values in (before, after)):
                return []
            fact = _normalise_fact({
                "attribute": attribute,
                "before": before.get(attribute),
                "after": after.get(attribute),
            })
            if fact is not None:
                facts.append(dict(zip(("attribute", "before", "after"), fact)))
    # A stale scene path or edited variant must never certify a task whose
    # stated count disagrees with the number of actual attribute transitions.
    solution = task.get("solution") or task.get("answer")
    if isinstance(solution, str):
        match = re.search(r"changed[\s_-]*attributes?\s*[:=]\s*(\d+)", solution, re.I)
        if match and (len(match.group(1)) > 10 or int(match.group(1)) != len(facts)):
            return []
    if isinstance(variant, dict) and "num_attr_changes" in variant:
        count = variant["num_attr_changes"]
        if type(count) is not int or count != len(facts):
            return []
    return facts


def score_visual_changes(
    completion: str, gold_changes: List[Dict[str, str]]
) -> Tuple[float, Dict[str, Any]]:
    """Score ``<change>attribute:before->after</change>`` with multiset F1.

    Every extra or repeated claim is a false positive. An opening change tag
    with an invalid body or no closing tag is also a false positive. If no
    usable gold facts exist, return zero with ``available=False`` so callers
    cannot award a fallback format bonus.
    """
    gold_items = gold_changes if isinstance(gold_changes, (list, tuple)) else []
    gold = Counter(fact for item in gold_items
                   if (fact := _normalise_fact(item)) is not None)
    if not gold:
        return 0.0, {"available": False, "reason": "no_gold_changes"}

    text = completion if isinstance(completion, str) else ""
    tags = list(_CHANGE_TAG.finditer(text))
    num_open = len(_CHANGE_OPEN.findall(text))
    num_close = len(_CHANGE_CLOSE.findall(text))
    predicted: Counter[Tuple[str, str, str]] = Counter()
    num_invalid = max(0, num_open - len(tags)) + max(0, num_close - len(tags))
    think_end = text.find("</think>")
    answer_start = text.find("<answer>")
    for tag in tags:
        if (think_end >= 0 and tag.start() < think_end + len("</think>")) or (
            answer_start >= 0 and tag.end() > answer_start
        ):
            num_invalid += 1
            continue
        match = _CHANGE_BODY.fullmatch(tag.group(1))
        fact = _normalise_fact({
            "attribute": match.group(1),
            "before": match.group(2),
            "after": match.group(3),
        }) if match else None
        if fact is None:
            num_invalid += 1
        else:
            predicted[fact] += 1

    true_positive = sum((predicted & gold).values())
    num_predicted = sum(predicted.values()) + num_invalid
    num_gold = sum(gold.values())
    false_positive = num_predicted - true_positive
    false_negative = num_gold - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    score = 2 * true_positive / denominator if denominator else 0.0
    return score, {
        "available": True,
        "num_gold": num_gold,
        "num_predicted": num_predicted,
        "num_invalid": num_invalid,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / num_predicted if num_predicted else 0.0,
        "recall": true_positive / num_gold,
    }
