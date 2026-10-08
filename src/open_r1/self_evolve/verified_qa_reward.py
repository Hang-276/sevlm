"""Exact count rewards; gold and pair metadata stay outside the prompt."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


VERIFIED_COUNT_QA = "verified_count_qa"
_ANSWER = re.compile(r"\s*<answer>\s*(0|[1-9][0-9]*)\s*</answer>\s*\Z")
_COUNT = re.compile(r"\s*(0|[1-9][0-9]*)\s*\Z")


def render_training_question(problem: str, context: Any, template: str) -> str:
    """Respect the count task's answer-only grammar in the mixed GRPO loader."""
    if isinstance(context, dict) and context.get("task_kind") == VERIFIED_COUNT_QA:
        return problem
    return template.format(Question=problem)


def score_verified_count_qa(
    completion: str, solution: Any, context: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Reward one correct numeric answer when exported gold matches solution."""
    qa = context.get("verified_qa")
    qa = qa if isinstance(qa, dict) else {}
    gold = qa.get("gold_count")
    valid_gold = type(gold) is int and 0 <= gold <= 1000
    solution_text = solution if isinstance(solution, str) else ""
    solution_match = _ANSWER.fullmatch(solution_text) or _COUNT.fullmatch(solution_text)
    # Bound integer parsing so malformed counts cannot crash a training step.
    valid_solution = bool(solution_match and len(solution_match.group(1)) <= 4)
    available = bool(valid_gold and valid_solution
                     and int(solution_match.group(1)) == gold)
    match = _ANSWER.fullmatch(completion if isinstance(completion, str) else "")
    format_valid = bool(match and len(match.group(1)) <= 4)
    prediction = int(match.group(1)) if format_valid else None
    correct = bool(available and format_valid and prediction == gold)
    return float(correct), {
        "task_kind": VERIFIED_COUNT_QA,
        "qa_available": available,
        "qa_gold_count": gold if valid_gold else None,
        "qa_pred_count": prediction,
        "answer": float(correct),
        "answer_exact_match": correct,
        "format_valid": format_valid,
        "grounding": None,
        "process": None,
        "consistency": None,
        "budget": None,
        "fallback_flags": {},
    }
