"""Reference judge abstractions for candidate task filtering.

Provides:
- Unified ``JudgeResult`` output schema shared by all judge backends.
- ``HeuristicReferenceJudge`` — rule-based default (no API keys required).
- ``OpenAIReferenceVLMJudge`` — optional GPT-4o adapter (cached, fallback-safe).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Unified output schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# "Too hard for the difficulty target" — every judge path calls this one.
# ---------------------------------------------------------------------------
#
# A task is rejected as too_hard ONLY when its estimated difficulty is at least
# ``SELF_EVOLVE_TOO_HARD_GAP`` bands above the task's difficulty_target. The gap
# defaults to 2 (conservative: only hard-task-under-easy-target fires), but is
# env-tunable so the policy's down-shift sensitivity can be adjusted without a
# code change. All three judge paths (heuristic / dry-run / OpenAI) read this
# one helper so the threshold can never drift between them.

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
TOO_HARD_REJECT_REASON = "too_hard_for_difficulty_target"


def too_hard_gap_threshold() -> int:
    """Resolve the difficulty-band gap that triggers a too_hard rejection.

    ``SELF_EVOLVE_TOO_HARD_GAP`` (default 2). Clamped to >= 1 so a value of 0
    cannot make every task "too hard".
    """
    try:
        gap = int(os.environ.get("SELF_EVOLVE_TOO_HARD_GAP", "2").strip() or "2")
    except ValueError:
        gap = 2
    return max(1, gap)


def is_too_hard_for_target(difficulty_level: str, difficulty_target: str) -> bool:
    """True iff ``difficulty_level`` is >= the gap threshold above the target."""
    gap = too_hard_gap_threshold()
    lvl = _DIFFICULTY_ORDER.get(str(difficulty_level), 1)
    tgt = _DIFFICULTY_ORDER.get(str(difficulty_target), 1)
    return (lvl - tgt) >= gap


def suggested_difficulty_on_too_hard(difficulty_target: str) -> str:
    """The easier difficulty to suggest when a task is rejected as too hard.

    Strictly one band below the target, floored at easy: hard → medium,
    medium → easy, easy → easy. Keeping it a genuine *reduction* relative to the
    target lets the policy update tell it apart from a target echo downstream.
    """
    tgt = _DIFFICULTY_ORDER.get(str(difficulty_target), 1)
    return {2: "medium", 1: "easy", 0: "easy"}[tgt]


# ---------------------------------------------------------------------------
# The accept/reject gate every judge path goes through.
# ---------------------------------------------------------------------------
#
# Every judge backend — heuristic, dry-run, and the live OpenRouter/OpenAI
# vision judge routed through the runner — funnels its structured signals
# through ``decide_accept`` so the accept/reject threshold can never drift
# between paths. The gate is intentionally LENIENT: it only rejects tasks that
# are clearly over-hard for their difficulty target, clearly unsolvable
# (solvability below τ), or — when explicitly enabled — visually ambiguous.
# CLEVR spot-diff tasks carry inherent ambiguity, so ambiguity rejection is OFF
# by default.

DEFAULT_SOLVABILITY_THRESHOLD = 0.4
AMBIGUITY_REJECT_THRESHOLD = 0.8

LOW_SOLVABILITY_REJECT_REASON = "low_solvability"
HIGH_AMBIGUITY_REJECT_REASON = "high_ambiguity"


def decide_accept(
    signals: Dict[str, Any],
    difficulty_target: str,
    *,
    solvability_threshold: float = DEFAULT_SOLVABILITY_THRESHOLD,
    reject_on_ambiguity: bool = False,
) -> Tuple[bool, str]:
    """Decide accept/reject for one candidate from its structured signals.

    ``signals`` may carry ``difficulty_level`` (str), ``solvability_score``
    (float in [0,1]), and ``ambiguity_score`` (float in [0,1]). Missing signals
    default to permissive values so a partial judgment never over-rejects.

    Returns ``(accepted, reject_reason)`` where ``reject_reason`` is empty on
    accept. Rejection order (most-specific first):
      1. too hard for the difficulty target (shared gap threshold)
      2. solvability below ``solvability_threshold`` (τ)
      3. ambiguity above the fixed threshold, only when ``reject_on_ambiguity``
    """
    difficulty_level = str(signals.get("difficulty_level", "medium"))
    if is_too_hard_for_target(difficulty_level, str(difficulty_target)):
        return False, TOO_HARD_REJECT_REASON

    solvability = float(signals.get("solvability_score", 1.0))
    if solvability < solvability_threshold:
        return False, LOW_SOLVABILITY_REJECT_REASON

    if reject_on_ambiguity:
        ambiguity = float(signals.get("ambiguity_score", 0.0))
        if ambiguity > AMBIGUITY_REJECT_THRESHOLD:
            return False, HIGH_AMBIGUITY_REJECT_REASON

    return True, ""


# ---------------------------------------------------------------------------
# HeuristicReferenceJudge  (default, no API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenAIReferenceVLMJudge  (optional, cached, fallback-safe)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VisionReferenceVLMJudge  (optional vision reference VLM — NO text-only fallback)
# ---------------------------------------------------------------------------
