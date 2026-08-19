"""
Failure tag assignment for self-evolving visual reasoning agent.

Input:
    a five-dimensional reward vector

Output:
    a failure tag used by later buffer routing and task generation
"""

from typing import Dict, List

# Shared reward threshold.
THRESHOLD = 0.5


def assign_failure_tag(reward: Dict[str, float]) -> str:
    """
    Assign a single failure tag based on the five-dimensional reward vector.

    Refined definition (core reasoning dims = answer, process, grounding):
    - positive:    answer == 1 AND process >= t AND grounding >= t
    - shortcut:    answer == 1 AND grounding < t  (ungrounded success)
    - answer fail: answer < 1
    - then budget / process / consistency shortfalls

    NOTE: budget and consistency are NOT part of the positive gate — a
    budget-only or consistency-only shortfall is recorded as a tag but does not
    by itself demote a correct, grounded, well-structured trajectory.
    """
    answer = reward.get("answer", 0.0)
    budget = reward.get("budget", 0.0)
    process = reward.get("process", 0.0)
    grounding = reward.get("grounding", 0.0)
    consistency = reward.get("consistency", 0.0)

    if answer >= 1.0 and process >= THRESHOLD and grounding >= THRESHOLD:
        return "positive"

    if answer >= 1.0 and grounding < THRESHOLD:
        return "shortcut_or_ungrounded_success"

    if answer < 1.0:
        return "reasoning_or_final_decision_failure"

    if process < THRESHOLD:
        return "process_failure"

    if budget < THRESHOLD:
        return "budget_failure"

    if consistency < THRESHOLD:
        return "consistency_failure"

    return "unknown_failure"


def assign_failure_tags(reward: Dict[str, float]) -> List[str]:
    """
    Assign multiple failure tags based on the five-dimensional reward vector.

    One trajectory can carry several tags. The "positive" tag follows the
    core-dimension gate (answer + process + grounding); budget / consistency
    shortfalls are added as informative tags but do NOT block "positive".
    """
    answer = reward.get("answer", 0.0)
    budget = reward.get("budget", 0.0)
    process = reward.get("process", 0.0)
    grounding = reward.get("grounding", 0.0)
    consistency = reward.get("consistency", 0.0)

    tags = []

    if answer >= 1.0 and process >= THRESHOLD and grounding >= THRESHOLD:
        tags.append("positive")

    if answer >= 1.0 and grounding < THRESHOLD:
        tags.append("shortcut_or_ungrounded_success")

    if answer < 1.0:
        tags.append("reasoning_or_final_decision_failure")

    if budget < THRESHOLD:
        tags.append("budget_failure")

    if process < THRESHOLD:
        tags.append("process_failure")

    if grounding < THRESHOLD:
        tags.append("grounding_failure")

    if consistency < THRESHOLD:
        tags.append("consistency_failure")

    if len(tags) == 0:
        tags.append("unknown_failure")

    return tags