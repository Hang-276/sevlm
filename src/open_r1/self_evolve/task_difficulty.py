"""Label-independent difficulty for CLEVR spot-diff tasks.

The original estimator bucketed a task by ``num_attr_changes``, which is also
the gold answer — so every curriculum move was really a move of the answer prior
(``medium`` == exactly ``changed_attributes==4``) and the solver could ride that
prior instead of counting.

Here difficulty comes from how hard the change is to SEE: attribute salience,
apparent size of the changed object, scene clutter — never the count.
:func:`label_difficulty_correlation` is the guard that keeps it that way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

ATTRS = ("color", "shape", "size", "material")

# How visible a change of each attribute is at CLEVR resolution. Colour swaps
# jump out; a rubber→metal swap is a shading cue that survives few pixels.
DEFAULT_ATTR_SALIENCE: Dict[str, float] = {
    "color": 1.0,
    "shape": 0.8,
    "size": 0.55,
    "material": 0.35,
}

# Difficulty-score cut points (score is "how visible", so low = hard).
DEFAULT_BUCKET_EDGES = {"hard": 0.35, "medium": 0.6}


def _object_scale(obj: Dict[str, Any]) -> float:
    """Apparent size proxy in [0,1]: large + close objects are easier to see."""
    size_factor = 1.0 if str(obj.get("size", "")).lower() == "large" else 0.6
    coords = obj.get("pixel_coords") or []
    # pixel_coords[2] is CLEVR's camera depth; nearer objects render bigger.
    depth = float(coords[2]) if len(coords) >= 3 else 10.0
    depth_factor = max(0.3, min(1.0, 12.0 / max(depth, 1e-3)))
    return max(0.0, min(1.0, size_factor * depth_factor))


def changed_object_records(modification: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per changed object: which attributes changed, salience, apparent size."""
    records: List[Dict[str, Any]] = []
    for item in modification.get("replaced_objects") or []:
        if not isinstance(item, dict):
            continue
        orig = item.get("original", {}) or {}
        repl = item.get("replacement", {}) or {}
        changed = [a for a in ATTRS if orig.get(a) != repl.get(a)]
        records.append({
            "index": item.get("index"),
            "changed": changed,
            "max_salience": max((DEFAULT_ATTR_SALIENCE.get(a, 0.5) for a in changed), default=0.0),
            "scale": _object_scale(repl or orig),
        })
    return records


def estimate_difficulty(
    modification: Dict[str, Any],
    num_scene_objects: Optional[int] = None,
    bucket_edges: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Difficulty of spotting the change, independent of the gold count.

    ``num_attr_changes`` is still reported (it is the gold answer) but is not
    an input to the score.
    """
    edges = {**DEFAULT_BUCKET_EDGES, **(bucket_edges or {})}
    records = changed_object_records(modification)
    num_attr_changes = sum(len(r["changed"]) for r in records)

    if not records:
        return {
            "difficulty_estimate": "hard",
            "difficulty_score": 0.0,
            "num_changed_objects": 0,
            "num_attr_changes": 0,
            "salience": 0.0,
            "scale": 0.0,
            "clutter": num_scene_objects or 0,
            "difficulty_source": "label_independent_salience",
        }

    # The solver only has to find ONE changed object to name the spy, so the
    # task is as easy as its most visible change.
    salience = max(r["max_salience"] for r in records)
    scale = max(r["scale"] for r in records)
    clutter = float(num_scene_objects or 0)
    clutter_penalty = max(0.5, min(1.0, 8.0 / clutter)) if clutter else 1.0

    score = salience * (0.5 + 0.5 * scale) * clutter_penalty
    level = "hard" if score < edges["hard"] else ("medium" if score < edges["medium"] else "easy")
    return {
        "difficulty_estimate": level,
        "difficulty_score": round(score, 4),
        "num_changed_objects": len(records),
        "num_attr_changes": num_attr_changes,
        "salience": round(salience, 4),
        "scale": round(scale, 4),
        "clutter": clutter,
        "difficulty_source": "label_independent_salience",
    }


def label_difficulty_correlation(
    difficulty_scores: Sequence[float], gold_labels: Sequence[float]
) -> Optional[float]:
    """Pearson r between difficulty and the gold label; ~0 is the design goal.

    A large |r| means difficulty is leaking the answer again.
    """
    n = len(difficulty_scores)
    if n < 2 or n != len(gold_labels):
        return None
    mx = sum(difficulty_scores) / n
    my = sum(gold_labels) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(difficulty_scores, gold_labels))
    vx = sum((x - mx) ** 2 for x in difficulty_scores)
    vy = sum((y - my) ** 2 for y in gold_labels)
    if vx <= 0 or vy <= 0:
        return None
    return round(cov / (vx ** 0.5 * vy ** 0.5), 4)


def balance_by_label(
    candidates: Sequence[Any],
    label_of: Any,
    num_wanted: int,
    balance: str = "uniform",
) -> List[Any]:
    """Pick ``num_wanted`` candidates with the flattest possible label histogram.

    A constant answer earns zero advantage inside its own GRPO group, so a
    collapsed field only pays off through a skewed label prior across prompts.
    """
    items = list(candidates)
    if balance != "uniform" or num_wanted >= len(items):
        return items[:num_wanted]

    by_label: Dict[Any, List[Any]] = {}
    for item in items:
        by_label.setdefault(label_of(item), []).append(item)

    chosen: List[Any] = []
    counts = {label: 0 for label in by_label}
    while len(chosen) < num_wanted:
        # Least-used label that still has candidates left.
        available = [lbl for lbl, pool in by_label.items() if pool]
        if not available:
            break
        label = min(available, key=lambda l: (counts[l], l))
        chosen.append(by_label[label].pop(0))
        counts[label] += 1
    return chosen
