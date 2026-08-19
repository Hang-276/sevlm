"""Per-task statistics read off the rollout group GRPO already samples.

``max_i r_i - mean_i r_i`` over a task's G trajectories is its regret, and it
costs nothing extra. Pass rate 0 means no usable signal yet, 1 means no headroom
left, in between the higher the regret the more room there is. It is also what
the trainer sees as advantage collapse: no spread, no gradient.
"""

from __future__ import annotations

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
        return float(value)
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


def task_statistics(by_task: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Pass rate, reward spread and regret for each task in an iteration."""
    stats: Dict[str, Dict[str, Any]] = {}
    for task_id, examples in by_task.items():
        if not examples:
            continue
        rewards = [_scalar(e) for e in examples]
        n = len(rewards)
        mean = sum(rewards) / n
        pass_rate = sum(1.0 for e in examples if _answer_correct(e)) / n
        if pass_rate <= 0.0:
            klass = TOO_HARD
        elif pass_rate >= 1.0:
            klass = MASTERED
        else:
            klass = TRAINABLE
        var = sum((r - mean) ** 2 for r in rewards) / n
        stats[str(task_id)] = {
            "n": n,
            "pass_rate": round(pass_rate, 4),
            "mean_reward": round(mean, 4),
            "max_reward": round(max(rewards), 4),
            "regret": round(max(rewards) - mean, 4),
            "reward_std": round(var ** 0.5, 4),
            "class": klass,
            "scene_id": examples[0].get("scene_id") or examples[0].get("base_task_id"),
        }
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

    trainable = sorted(
        (t for t in by_class[TRAINABLE]),
        key=lambda t: -stats[t]["regret"],
    )
    collapsed = sum(1 for s in values if s["reward_std"] <= std_floor)

    return {
        "num_tasks": n,
        "mean_solve_rate": round(sum(s["pass_rate"] for s in values) / n, 4),
        "mean_regret": round(sum(s["regret"] for s in values) / n, 4),
        # Share of tasks whose group carried no usable gradient this round.
        "advantage_collapse_rate": round(collapsed / n, 4),
        "class_counts": {k: len(v) for k, v in by_class.items()},
        "seeds": [
            {"task_id": t, "scene_id": stats[t]["scene_id"],
             "regret": stats[t]["regret"], "pass_rate": stats[t]["pass_rate"],
             "direction": "harder" if stats[t]["pass_rate"] >= 0.5 else "easier"}
            for t in trainable[:num_seeds]
        ],
        "retire": [
            {"task_id": t, "scene_id": stats[t]["scene_id"]}
            for t in by_class[MASTERED]
        ],
        "too_hard": [
            {"task_id": t, "scene_id": stats[t]["scene_id"], "direction": "easier"}
            for t in by_class[TOO_HARD]
        ],
        "solve_rate_by_task": {t: s["pass_rate"] for t, s in sorted(stats.items())},
    }


def counterfactual_sensitivity(
    scored_examples: Sequence[Dict[str, Any]],
    tasks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Share of pairs where the answer moved when the input did; 0 for a prior-following policy."""
    from open_r1.self_evolve.rewards import parse_structured_answer

    by_task_id = {str(t.get("task_id")): t for t in (tasks or [])}

    pairs: Dict[str, Dict[str, List[Optional[int]]]] = {}
    for ex in scored_examples:
        meta = ex.get("self_evolve") or ex.get("metadata") or {}
        task = by_task_id.get(str(ex.get("task_id")), {})
        task_meta = task.get("metadata") or {}
        pair_id = (ex.get("pair_id") or meta.get("pair_id")
                   or task.get("pair_id") or task_meta.get("pair_id"))
        role = (ex.get("pair_role") or meta.get("pair_role")
                or task.get("pair_role") or task_meta.get("pair_role"))
        if not pair_id or role not in ("small", "large"):
            continue
        pred = parse_structured_answer(ex.get("completion") or "")
        pairs.setdefault(str(pair_id), {}).setdefault(role, []).append(
            pred.get("changed_attributes")
        )

    complete = {k: v for k, v in pairs.items() if "small" in v and "large" in v}
    if not complete:
        return {"num_pairs": 0, "counterfactual_sensitivity": None}

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
    for v in complete.values():
        a, b = mode(v["small"]), mode(v["large"])
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
