"""
Buffer routing for self-evolving visual reasoning agent.

The active self-evolve loop uses two training buffers:

1. positive buffer
2. failure buffer

Routing depends on the reward VECTOR (not the scalar reward) plus a small set
of thresholds that come from the single reward config (``RewardConfig``). There
is intentionally NO scalar gate / cap here — shortcut samples are kept out of
SFT by these vector conditions, not by clamping the scalar reward.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from open_r1.self_evolve.reward_config import RewardConfig



def _flags(
    reward_vector: Dict[str, Any],
    reward_details: Optional[Dict[str, Any]],
    failure_tags: List[str],
) -> Tuple[bool, bool]:
    """Resolve (format_valid, shortcut_detected) from details / tags."""
    details = reward_details or reward_vector.get("reward_details") or {}
    format_valid = bool(details.get("format_valid", True))
    shortcut_detected = bool(
        details.get("shortcut_detected", False)
        or "shortcut_or_ungrounded_success" in failure_tags
    )
    return format_valid, shortcut_detected


def route_with_config(
    reward_vector: Dict[str, Any],
    failure_tags: List[str],
    config: RewardConfig,
    reward_details: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Route one trajectory using reward-vector conditions + config thresholds.

    Returns ``(buffer_name, routing_reason)``.

    - positive:      answer == 1 AND grounding >= positive_min_grounding AND
                     process >= positive_min_process AND format_valid AND
                     NOT shortcut_detected.
    - failure:       every trajectory that does not satisfy the positive gate,
                     including correct answers with weak reasoning/grounding.
    """
    try:
        answer = float(reward_vector.get("answer", 0.0))
        grounding = float(reward_vector.get("grounding", 0.0))
        process = float(reward_vector.get("process", 0.0))
    except (TypeError, ValueError):
        return "failure", "invalid_reward_vector"
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
           for value in (answer, grounding, process)):
        return "failure", "invalid_reward_vector"
    format_valid, shortcut_detected = _flags(reward_vector, reward_details, failure_tags)
    evidence_mismatch = "evidence_mismatch" in failure_tags

    # Failure: wrong final answer (answer reward is binary exact-match).
    if answer < 1.0:
        return "failure", "answer_incorrect"

    # answer == 1 below.
    # Correct answers with weak format/reasoning/grounding count as failures;
    # the detailed reason is preserved for failure-profile feedback.
    if not format_valid:
        return "failure", "correct_answer_but_format_invalid"

    if grounding < config.positive_min_grounding:
        return "failure", "correct_answer_but_grounding_below_threshold"
    if process < config.positive_min_process:
        return "failure", "correct_answer_but_process_below_threshold"
    if config.positive_min_visual_facts > 0.0:
        details = reward_details or reward_vector.get("reward_details") or {}
        process_details = details.get("process") or {}
        fact_score = process_details.get("visual_facts_score")
        try:
            fact_score = float(fact_score)
        except (TypeError, ValueError):
            fact_score = float("nan")
        if not math.isfinite(fact_score) or not 0.0 <= fact_score <= 1.0 or fact_score < config.positive_min_visual_facts:
            return "failure", "correct_answer_but_visual_facts_incomplete"
    if shortcut_detected:
        return "failure", "correct_answer_but_shortcut_detected"
    if evidence_mismatch:
        return "failure", "correct_answer_but_evidence_mismatch"

    # Positive: correct, grounded, structured, well-formatted, no shortcut.
    return "positive", "correct_grounded_structured"
