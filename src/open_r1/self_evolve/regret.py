"""Per-task statistics read off the rollout group GRPO already samples.

``max_i r_i - mean_i r_i`` over a task's G trajectories is its regret, and it
costs nothing extra. Pass rate 0 means no usable signal yet, 1 means no headroom
left, in between the higher the regret the more room there is. It is also what
the trainer sees as advantage collapse: no spread, no gradient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Pass-rate classes.
TOO_HARD = "too_hard"
TRAINABLE = "trainable"
MASTERED = "mastered"


def _scalar(example: Dict[str, Any]) -> float:
    value = example.get("reward_scalar")
    if value is None:
        value = (example.get("reward_vector") or {}).get("answer", 0.0)
    try:
        scalar = float(value)
        return scalar if math.isfinite(scalar) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _answer_correct(example: Dict[str, Any]) -> bool:
    """Solved means the rollout found the odd image — the field the gate uses.

    Requiring every answer field instead puts the pass rate at 0 whenever any
    one of them is unreliable, which empties the seed list and stops the
    generator editing. The other fields still move the reward, just not this
    flag.
    """
    fields = ((example.get("reward_vector") or {})
              .get("reward_details", {})
              .get("answer", {})
              .get("field_correct")) or {}
    if "spy" in fields:
        return bool(fields["spy"])
    return float((example.get("reward_vector") or {}).get("answer", 0.0)) >= 1.0


def _accepted_by_id(accepted_tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Use only uniquely identified accepted tasks as certificate authority."""
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: set[str] = set()
    for task in accepted_tasks:
        if not isinstance(task, dict) or task.get("task_id") is None:
            continue
        task_id = str(task["task_id"])
        if task_id in by_id:
            duplicates.add(task_id)
        else:
            by_id[task_id] = task
    for task_id in duplicates:
        del by_id[task_id]
    return by_id


def _visible_keep_indices(task: Dict[str, Any]) -> Optional[List[int]]:
    """Preserve an edited task's visible subset for a hold-difficulty seed."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
    if "keep_indices" in variant:
        keep = variant["keep_indices"]
        if not isinstance(keep, list) or not keep or any(type(i) is not int or i < 0 for i in keep):
            return None
        return sorted(keep) if len(set(keep)) == len(keep) else None
    comparison = meta.get("comparison_data") if isinstance(meta.get("comparison_data"), dict) else {}
    replaced = comparison.get("replaced_objects")
    if not isinstance(replaced, list):
        path = task.get("scene_path")
        if not isinstance(path, (str, Path)) or not str(path):
            return None
        try:
            scene = json.loads(Path(path).read_text(encoding="utf-8"))
            replaced = (scene.get("modification") or {}).get("replaced_objects")
        except (OSError, ValueError, TypeError, AttributeError):
            return None
    return list(range(len(replaced))) if isinstance(replaced, list) and replaced else None


def _verified_visual_success(
    example: Dict[str, Any], *, accepted_solution: str,
    gold_changes: List[Dict[str, str]], min_grounding: float,
    min_visual_facts: float,
) -> bool:
    """Only a complete, scored visual certificate can establish mastery."""
    from open_r1.self_evolve.rewards import parse_structured_answer
    from open_r1.self_evolve.visual_facts import score_visual_changes

    vector = example.get("reward_vector") or {}
    details = vector.get("reward_details") or {}
    answer = details.get("answer") or {}
    fields = answer.get("field_correct") or {}
    trusted_gold = parse_structured_answer(accepted_solution)
    rollout_gold = parse_structured_answer(example.get("solution") or "")
    predicted = parse_structured_answer(example.get("completion") or "")
    if any(trusted_gold.get(field) is None or rollout_gold.get(field) != trusted_gold.get(field)
           or predicted.get(field) != trusted_gold.get(field)
           for field in ("spy", "changed_attributes")):
        return False
    if "spy" in fields and "changed_attributes" in fields:
        full_answer = fields["spy"] is True and fields["changed_attributes"] is True
    else:
        full_answer = True
    if (not full_answer or details.get("format_valid") is not True
            or details.get("shortcut_detected") is not False):
        return False

    ground_details = details.get("grounding") or {}
    try:
        grounding = float(vector.get("grounding", 0.0))
    except (TypeError, ValueError):
        return False
    if (not math.isfinite(grounding) or grounding < min_grounding
            or ground_details.get("grounding_source") != "model_bbox_iou"
            or ground_details.get("bbox_player_id_valid") is not True):
        return False

    process = details.get("process") or {}
    certificate = process.get("visual_facts") or {}
    try:
        score = float(process.get("visual_facts_score"))
    except (TypeError, ValueError):
        return False
    verified_score, verified_details = score_visual_changes(
        example.get("completion") or "", gold_changes
    )
    return bool(
        math.isfinite(score) and score >= min_visual_facts
        and verified_score >= min_visual_facts
        and verified_details.get("available") is True
        and verified_details.get("false_positive") == 0
        and verified_details.get("false_negative") == 0
        and certificate.get("available") is True
        and certificate.get("num_gold") == len(gold_changes)
        and certificate.get("false_positive") == 0
        and certificate.get("false_negative") == 0
    )


def task_statistics(
    by_task: Dict[str, List[Dict[str, Any]]], *,
    mastery_mode: str = "spy",
    accepted_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    min_grounding: float = 0.2,
    min_visual_facts: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute regret; optionally require accepted-task visual evidence for mastery."""
    if mastery_mode not in ("spy", "verified_visual"):
        raise ValueError(f"Unknown mastery_mode: {mastery_mode!r}")
    if mastery_mode == "verified_visual":
        if accepted_tasks is None:
            raise ValueError("verified_visual mastery requires accepted_tasks")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in (min_grounding, min_visual_facts)):
            raise ValueError("evidence thresholds must be finite values in [0, 1]")
        from open_r1.self_evolve.visual_facts import gold_changes_from_task
        accepted_by_id = _accepted_by_id(accepted_tasks)
    else:
        accepted_by_id = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for task_id, examples in by_task.items():
        if not examples:
            continue
        rewards = [_scalar(e) for e in examples]
        n = len(rewards)
        mean = sum(rewards) / n
        pass_rate = sum(1.0 for e in examples if _answer_correct(e)) / n
        accepted = accepted_by_id.get(str(task_id)) if mastery_mode == "verified_visual" else None
        gold = gold_changes_from_task(accepted) if accepted is not None else []
        accepted_meta = accepted.get("metadata") if accepted and isinstance(accepted.get("metadata"), dict) else {}
        num_players = accepted_meta.get("num_players")
        spy_player = accepted_meta.get("spy_player")
        if accepted is not None:
            from open_r1.self_evolve.rewards import parse_structured_answer
            accepted_fields = parse_structured_answer(accepted.get("solution") or "")
        else:
            accepted_fields = {}
        gold_available = bool(
            gold and type(num_players) is int and num_players >= 2
            and type(spy_player) is int and 1 <= spy_player <= num_players
            and accepted_fields.get("spy") == spy_player
            and accepted_fields.get("changed_attributes") == len(gold)
        )
        verified = [
            _verified_visual_success(
                example, accepted_solution=accepted["solution"], gold_changes=gold,
                min_grounding=min_grounding,
                min_visual_facts=min_visual_facts,
            ) if gold_available else False
            for example in examples
        ] if mastery_mode == "verified_visual" else []
        verified_rate = sum(verified) / n if verified else 0.0
        if pass_rate <= 0.0:
            klass = TOO_HARD
        elif pass_rate >= 1.0 and (mastery_mode == "spy" or verified_rate >= 1.0):
            klass = MASTERED
        else:
            klass = TRAINABLE
        var = sum((r - mean) ** 2 for r in rewards) / n
        task_stat = {
            "n": n,
            "pass_rate": round(pass_rate, 4),
            "mean_reward": round(mean, 4),
            "max_reward": round(max(rewards), 4),
            "regret": round(max(rewards) - mean, 4),
            "reward_std": round(var ** 0.5, 4),
            "class": klass,
            "scene_id": examples[0].get("scene_id") or examples[0].get("base_task_id"),
        }
        if mastery_mode == "verified_visual":
            keep = _visible_keep_indices(accepted) if accepted is not None and gold_available else None
            task_stat.update({
                "mastery_mode": mastery_mode,
                "scene_id": (accepted.get("scene_id") or task_stat["scene_id"]) if accepted else task_stat["scene_id"],
                "gold_certificate_available": gold_available,
                "gold_visual_fact_count": len(gold),
                "num_players": num_players if type(num_players) is int else None,
                "verified_pass_rate": round(verified_rate, 4),
                "evidence_gap": round(1.0 - verified_rate, 4) if gold_available else None,
                "evidence_std": round((verified_rate * (1.0 - verified_rate)) ** 0.5, 4),
                "frontier": (
                    "unverifiable" if not gold_available else
                    "evidence" if pass_rate >= 1.0 and klass == TRAINABLE else
                    "spy" if klass == TRAINABLE else None
                ),
                "keep_indices": keep,
                "num_kept": len(keep) if keep is not None else None,
            })
        stats[str(task_id)] = task_stat
    return stats


def summarize(
    stats: Dict[str, Dict[str, Any]],
    num_seeds: int = 16,
    std_floor: float = 0.02,
) -> Dict[str, Any]:
    """Iteration summary plus the seed / retire / too_hard lists the generator needs.

    Seeds are the trainable tasks with the most headroom, to be edited into
    neighbours; retire are the mastered ones; too_hard come back easier rather
    than being discarded.
    """
    if not stats:
        return {"num_tasks": 0, "mean_solve_rate": None}

    values = list(stats.values())
    n = len(values)
    by_class: Dict[str, List[str]] = {TOO_HARD: [], TRAINABLE: [], MASTERED: []}
    for task_id, s in stats.items():
        by_class[s["class"]].append(task_id)

    verified_mode = any(s.get("mastery_mode") == "verified_visual" for s in values)
    trainable = sorted(
        (t for t in by_class[TRAINABLE]
         if not verified_mode or stats[t].get("gold_certificate_available")),
        key=(
            (lambda t: (-stats[t]["regret"], -(stats[t].get("evidence_gap") or 0.0),
                        -stats[t].get("evidence_std", 0.0), t))
            if verified_mode else (lambda t: -stats[t]["regret"])
        ),
    )
    collapsed = sum(1 for s in values if s["reward_std"] <= std_floor)
    retire_ids = by_class[MASTERED]
    if verified_mode:
        certified = [s for s in values if s.get("gold_certificate_available") is True]
        # The generator retires scenes, so all evaluated variants must be mastered.
        scenes: Dict[str, List[str]] = {}
        for task_id, stat in stats.items():
            scene_id = stat.get("scene_id")
            if scene_id:
                scenes.setdefault(str(scene_id), []).append(task_id)
        retire_ids = [
            task_id for task_id in retire_ids
            if stats[task_id].get("scene_id")
            and all(
                stats[sibling]["class"] == MASTERED
                and stats[sibling].get("gold_certificate_available") is True
                for sibling in scenes[str(stats[task_id]["scene_id"])]
            )
        ]

    result = {
        "num_tasks": n,
        "mean_solve_rate": round(sum(s["pass_rate"] for s in values) / n, 4),
        "mean_regret": round(sum(s["regret"] for s in values) / n, 4),
        # Share of tasks whose group carried no usable gradient this round.
        "advantage_collapse_rate": round(collapsed / n, 4),
        "class_counts": {k: len(v) for k, v in by_class.items()},
        "seeds": [],
        "retire": [
            {"task_id": t, "scene_id": stats[t]["scene_id"]}
            for t in retire_ids
        ],
        "too_hard": [
            {"task_id": t, "scene_id": stats[t]["scene_id"], "direction": "easier"}
            for t in by_class[TOO_HARD]
            if not verified_mode or stats[t].get("gold_certificate_available")
        ],
        "solve_rate_by_task": {t: s["pass_rate"] for t, s in sorted(stats.items())},
    }
    for task_id in trainable[:num_seeds]:
        stat = stats[task_id]
        seed = {
            "task_id": task_id, "scene_id": stat["scene_id"],
            "regret": stat["regret"], "pass_rate": stat["pass_rate"],
            "direction": (
                "hold" if verified_mode and stat.get("frontier") == "evidence" else
                "harder" if stat["pass_rate"] >= 0.5 else "easier"
            ),
        }
        if verified_mode:
            seed.update({
                "frontier": stat["frontier"],
                "verified_pass_rate": stat["verified_pass_rate"],
                "evidence_gap": stat["evidence_gap"],
                "keep_indices": stat["keep_indices"],
                "num_kept": stat["num_kept"],
                "num_players": stat["num_players"],
            })
        result["seeds"].append(seed)
    if verified_mode:
        result["retirement_deferred"] = len(by_class[MASTERED]) - len(retire_ids)
        result["frontier_counts"] = {
            key: sum(s.get("frontier") == key for s in values)
            for key in ("spy", "evidence", "unverifiable")
        }
        result["num_certified_tasks"] = len(certified)
        result["mean_verified_pass_rate"] = (
            round(sum(s["verified_pass_rate"] for s in certified) / len(certified), 4)
            if certified else None
        )
    return result


def counterfactual_sensitivity(
    scored_examples: Sequence[Dict[str, Any]],
    tasks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Share of verified one-object pairs whose predicted change count moved."""
    from open_r1.self_evolve.rewards import parse_structured_answer

    by_task_id: Dict[str, Dict[str, Any]] = {}
    ambiguous_ids: set[str] = set()
    for task in tasks or []:
        task_id = task.get("task_id")
        if task_id is None:
            continue
        task_id = str(task_id)
        if task_id in by_task_id:
            ambiguous_ids.add(task_id)
        else:
            by_task_id[task_id] = task

    # The accepted task carries the generated inputs. A rollout's copied pair
    # tags cannot prove that both sides saw the same scene, player count and
    # actual spy slot, so only task metadata may establish a valid pair.
    pairs: Dict[tuple[str, str, int, int], Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for ex in scored_examples:
        task_id = ex.get("task_id")
        if task_id is None or str(task_id) in ambiguous_ids:
            continue
        task_id = str(task_id)
        task = by_task_id.get(task_id)
        if task is None:
            continue
        task_meta = task.get("metadata") or {}
        pair_id = task.get("pair_id") or task_meta.get("pair_id")
        role = task.get("pair_role") or task_meta.get("pair_role")
        scene_id = task.get("scene_id") or task_meta.get("base_name")
        keep_values = (task_meta.get("variant") or {}).get("keep_indices")
        if (not pair_id or role not in ("small", "large") or not scene_id
                or not isinstance(keep_values, (list, tuple))):
            continue
        try:
            num_players = int(task_meta["num_players"])
            spy_player = int(task_meta["spy_player"])
            keep = tuple(int(i) for i in keep_values)
        except (KeyError, TypeError, ValueError):
            continue
        if (num_players < 1 or not 1 <= spy_player <= num_players
                or not keep or min(keep) < 0 or len(set(keep)) != len(keep)):
            continue
        pred = parse_structured_answer(ex.get("completion") or "")
        key = (str(pair_id), str(scene_id), num_players, spy_player)
        entry = pairs.setdefault(key, {}).setdefault(role, {}).setdefault(
            task_id, {"keep": set(keep), "predictions": []}
        )
        entry["predictions"].append(pred.get("changed_attributes"))

    # Compare the modal prediction per side, not the mean: with G stochastic
    # rollouts, means are almost never exactly equal, so a mean-inequality test
    # saturates near 1 even for a prior-following policy.
    def mode(values: List[Optional[int]]) -> Optional[int]:
        counts: Dict[int, int] = {}
        for x in values:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1
        return max(sorted(counts), key=lambda k: counts[k]) if counts else None

    usable = moved = moved_correctly = 0
    for sides in pairs.values():
        small_tasks = sides.get("small", {})
        large_tasks = sides.get("large", {})
        if len(small_tasks) != 1 or len(large_tasks) != 1:
            continue
        small = next(iter(small_tasks.values()))
        large = next(iter(large_tasks.values()))
        if (len(large["keep"]) != len(small["keep"]) + 1
                or not small["keep"] < large["keep"]):
            continue
        a, b = mode(small["predictions"]), mode(large["predictions"])
        if a is None or b is None:
            continue
        usable += 1
        if b != a:
            moved += 1
            if b > a:
                moved_correctly += 1
    if not usable:
        return {"num_pairs": 0, "counterfactual_sensitivity": None}
    return {
        "num_pairs": usable,
        "counterfactual_sensitivity": round(moved / usable, 4),
        "counterfactual_direction_correct": round(moved_correctly / max(moved, 1), 4),
    }
