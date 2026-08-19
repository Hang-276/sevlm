"""
Reward functions for the self-evolving visual reasoning agent.

Five reward dimensions per the proposal:

1. Answer Reward       — exact match after normalization
2. Budget Reward       — per-task reasoning budget (step-based or token-based)
3. Process Reward      — reference-step distance + causal structure + shortcut detection
4. Grounding Reward    — IoU-priority grounding (box IoU > external score > 0)
5. Consistency Reward  — component-based (answer + evidence + reasoning), optional LLM judge

All rewards output values in [0.0, 1.0].
Backward compatible: existing callers that only expect ``reward_vector`` continue to work.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from open_r1.self_evolve.grounding_iou import resolve_grounding_score


def _active_reward_config():
    """The reward config offline scoring shares with the live GRPO reward.

    Both must read the same one, or buffer routing disagrees with what
    training actually optimized.
    """
    from open_r1.self_evolve.reward_config import load_reward_config

    return load_reward_config(os.environ.get("SELF_EVOLVE_REWARD_CONFIG") or None)


def _active_grounding_cfg() -> Dict[str, Any]:
    return _active_reward_config().grounding_cfg


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def extract_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def extract_reasoning(text: Optional[str]) -> str:
    if text is None:
        return ""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    if match is not None:
        return match.group(1).strip()
    return ""


# Evidence <bbox> tags may sit between </think> and <answer> (evidence-before-
# conclusion order) or after </answer>; both are valid.
_FORMAT_VALID_RE = re.compile(
    r"<think>.*?</think>\s*(?:<bbox[^>]*>[^<]*</bbox>\s*)*<answer>.*?</answer>",
    re.DOTALL,
)


def is_format_valid(completion: Optional[str]) -> bool:
    """True iff the completion has <think>…</think> then <answer>…</answer>."""
    return bool(_FORMAT_VALID_RE.search(completion or ""))


# ---------------------------------------------------------------------------
# Answer reward
# ---------------------------------------------------------------------------

def answer_reward(completion: str, solution: str) -> float:
    """Binary exact-match answer reward."""
    pred_answer = _normalize_text(extract_answer(completion))
    gt_answer = _normalize_text(extract_answer(solution))
    if pred_answer == "" or gt_answer == "":
        return 0.0
    return 1.0 if pred_answer == gt_answer else 0.0


# ---------------------------------------------------------------------------
# Answer reward, scored field by field
# ---------------------------------------------------------------------------
#
# Whole-string match makes the two sub-answers multiplicative: a collapsed
# `changed_attributes` zeroes the reward even when the spy is right, so GRPO
# cannot assign credit to either field.

_ANSWER_FIELD_RES: Dict[str, "re.Pattern[str]"] = {
    "spy": re.compile(r"spy\s*(?:player)?\s*[:=]?\s*(?:player\s*)?(\d+)", re.I),
    "changed_attributes": re.compile(
        r"changed[\s_-]*attributes?\s*[:=]?\s*(\d+)", re.I
    ),
}


def parse_structured_answer(text: Optional[str]) -> Dict[str, Optional[int]]:
    """Tolerantly parse ``spy`` / ``changed_attributes`` ints from an answer."""
    answer = extract_answer(text)
    out: Dict[str, Optional[int]] = {}
    for field_name, pattern in _ANSWER_FIELD_RES.items():
        match = pattern.search(answer)
        # Widen to the whole completion only when there is no <answer> block, so
        # numbers inside <think> cannot be mined for credit.
        if match is None and "<answer>" not in (text or ""):
            match = pattern.search(text or "")
        out[field_name] = int(match.group(1)) if match else None
    return out


def structured_answer_reward(
    completion: str,
    solution: str,
    fields: Optional[Dict[str, float]] = None,
    off_by_one_credit: float = 0.0,
    graded_fields: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Field-wise answer reward: weighted per-field correctness in [0, 1].

    ``off_by_one_credit`` applies only to ``graded_fields`` — fields whose value
    is a magnitude (a count). A player id is categorical: player 2 is not
    "nearly" player 3, so it must never earn near-miss credit.
    """
    fields = fields or {"spy": 0.5, "changed_attributes": 0.5}
    graded = set(graded_fields or [])
    pred = parse_structured_answer(completion)
    gold = parse_structured_answer(solution)

    score = 0.0
    per_field: Dict[str, Any] = {}
    for field_name, weight in fields.items():
        p, g = pred.get(field_name), gold.get(field_name)
        if g is None:
            credit = 0.0  # no gold for this field -> nothing to earn
        elif p is None:
            credit = 0.0
        elif p == g:
            credit = 1.0
        elif off_by_one_credit and field_name in graded and abs(p - g) == 1:
            credit = float(off_by_one_credit)
        else:
            credit = 0.0
        score += float(weight) * credit
        per_field[field_name] = {"pred": p, "gold": g, "credit": credit}

    details = {
        "answer_mode": "structured_fields",
        "answer_fields": per_field,
        "answer_parse_failed": [k for k, v in pred.items() if v is None],
        "field_correct": {k: v["credit"] >= 1.0 for k, v in per_field.items()},
    }
    return max(0.0, min(1.0, score)), details


# ---------------------------------------------------------------------------
# Answer LLM judge — for answers exact match cannot verify
# ---------------------------------------------------------------------------
#
# Layer 1 (reliable): exact normalized string match (``answer_reward``).
# Layer 2 (this):      a real LLM judge, invoked ONLY when exact match fails and
#                      the answer cannot be verified by the rule (parse failure,
#                      synonym / natural-language phrasing, answer buried in the
#                      reasoning). It reuses the Reference VLM provider/base_url/
#                      model configuration.

def _answer_judge_should_invoke(completion: str, solution: str) -> Tuple[bool, str]:
    """Decide whether the answer judge should be invoked (conditional invocation).

    This is conditional invocation for cost control — NOT a reward gate or cap.
    It only chooses whether to *call* the judge; it never gates, caps, or
    rescales the reward scalar. The judge is invoked only when there is a gold
    answer and the rule-based exact match did NOT already verify the answer —
    i.e. the rule path is uncertain. Correct answers (the common case after
    training) skip the judge entirely, so the API cost is bounded.
    """
    gt_answer = _normalize_text(extract_answer(solution))
    if gt_answer == "":
        return False, "no_gold_answer"
    if answer_reward(completion, solution) >= 1.0:
        return False, "exact_match_succeeded"

    pred_answer = _normalize_text(extract_answer(completion))
    # Parse failure (no <answer> token resolved) — answer may be in reasoning.
    if pred_answer == "" or "<answer>" not in (completion or ""):
        return True, "final_answer_parse_failed"
    # Gold token appears somewhere in the completion text (format/synonym noise).
    if gt_answer and gt_answer in _normalize_text(completion):
        return True, "gold_token_present_but_not_exact"
    # Exact match failed but a non-empty answer exists — could be synonym /
    # natural-language phrasing differing from the gold string.
    return True, "exact_mismatch_possible_synonym"


def _resolve_answer_judge_endpoint(
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str, Optional[str]]:
    """Resolve (provider, model, base_url, api_key) reusing the Reference VLM env.

    Mirrors ``OpenAIReferenceVLMJudge`` so the answer judge speaks to the SAME
    backend as the Reference VLM (no new config surface). The key is read from
    the environment only at call time and never logged.
    """
    provider = (provider or os.environ.get("SELF_EVOLVE_REFERENCE_PROVIDER")
                or os.environ.get("REFERENCE_PROVIDER"))
    # Auto-detect provider from whichever API key is actually present when none
    # was explicitly configured: prefer OpenAI when only OPENAI_API_KEY is set
    # (the common .env case), else OpenRouter. This keeps the answer judge on the
    # SAME backend as the Reference VLM without a new config surface.
    if not provider:
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
            provider = "openai"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            provider = "openai"
    provider = provider.lower()
    if provider == "openrouter":
        default_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        default_model = os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o"
        key_env = "OPENROUTER_API_KEY"
    else:
        provider = "openai"
        default_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        default_model = "gpt-4o"
        key_env = "OPENAI_API_KEY"
    model = (model or os.environ.get("REFERENCE_VLM_MODEL")
             or os.environ.get("OPENAI_MODEL") or default_model)
    base_url = base_url or default_base
    api_key = api_key or os.environ.get(key_env)
    return provider, model, base_url, api_key


def answer_judge(
    completion: str,
    solution: str,
    problem: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Real answer judge (LLM) used as a fallback to exact match.

    Returns a structured dict with the mentor-specified schema. When no API key
    is available or ``dry_run`` is set, returns ``answer_judge_used=False`` with
    ``blocked_reason="no_api_or_dry_run"`` — it never fabricates a judgment.
    """
    gold = extract_answer(solution)
    pred = extract_answer(completion)
    reasoning = extract_reasoning(completion)
    exact = answer_reward(completion, solution) >= 1.0

    result: Dict[str, Any] = {
        "answer_judge_used": False,
        "answer_judge_provider": None,
        "answer_judge_model": None,
        "exact_match_result": exact,
        "judge_correct": None,
        "judge_confidence": None,
        "judge_reason": "",
        "answer_source_type": "rule_based",
        "blocked_reason": None,
    }

    provider, model, base_url, key = _resolve_answer_judge_endpoint(
        provider, model, base_url, api_key
    )
    if dry_run or not key:
        result["blocked_reason"] = "no_api_or_dry_run"
        result["judge_reason"] = (
            "answer judge not invoked: dry-run or no API key"
        )
        return result

    sys_prompt = (
        "You are a strict answer-grading judge for a visual reasoning task. "
        "You are given the gold answer and a model's full response (reasoning + "
        "final answer). Decide whether the model's answer is CORRECT, allowing "
        "for synonyms, natural-language phrasing, or an answer stated only in the "
        "reasoning. Output ONLY a JSON object with keys: "
        '"judge_correct" (bool), "judge_confidence" (number 0-1), '
        '"judge_reason" (short string).'
    )
    user_prompt = (
        (f"Task:\n{problem}\n\n" if problem else "")
        + f"Gold answer: {gold}\n\n"
        f"Model final answer (parsed): {pred or '(none parsed)'}\n\n"
        f"Model reasoning: {reasoning[:800]}\n\n"
        "Is the model's answer correct? Return the JSON."
    )

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            client = OpenAI(api_key=key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            if not raw:
                last_error = "empty_response"
                continue
            data = json.loads(raw)
            result.update({
                "answer_judge_used": True,
                "answer_judge_provider": provider,
                "answer_judge_model": model,
                "judge_correct": bool(data.get("judge_correct", False)),
                "judge_confidence": float(data.get("judge_confidence", 0.0)),
                "judge_reason": str(data.get("judge_reason", ""))[:300],
                "answer_source_type": "llm_judge",
                "blocked_reason": None,
            })
            return result
        except Exception as exc:  # noqa: BLE001 — never crash the reward path
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt >= max_retries:
                break
    result["blocked_reason"] = f"api_error: {last_error}" if last_error else "api_error"
    result["judge_reason"] = "answer judge API call failed (no fabricated verdict)"
    return result


# ---------------------------------------------------------------------------
# Reasoning step parser  (shared by budget / process / consistency)
# ---------------------------------------------------------------------------

def parse_reasoning_steps(text: Optional[str]) -> List[str]:
    """Parse reasoning text into discrete steps.

    Priority:
    1. Numbered steps: ``1.``, ``Step 1:``, ``-`` bullet, ``*`` bullet.
    2. Sentence splitting if the text looks like prose.
    """
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []

    # Numbered / bullet patterns
    patterns = [
        r"(?:^|\n)\s*(?:\d+[\.\)]\s*)(.+?)(?=(?:\n\s*(?:\d+[\.\)]|Step\s*\d|$))|\Z)",
        r"(?:^|\n)\s*(?:Step\s*\d+[:\-\.]\s*)(.+?)(?=(?:\n\s*(?:\d+[\.\)]|Step\s*\d|$))|\Z)",
        r"(?:^|\n)\s*[-*]\s+(.+?)(?=(?:\n\s*[-*]\s|\n\s*(?:\d+[\.\)]|Step\s*\d|$))|\Z)",
    ]

    for pattern in patterns:
        steps = re.findall(pattern, text, flags=re.S)
        cleaned = [s.strip() for s in steps if s.strip()]
        if len(cleaned) >= 2:
            # Remove the delimiter from the last step's trailing context
            return [re.sub(r"\n\s*(?:\d+[\.\)]|Step\s*\d|[-*]\s).*$", "", s).strip() for s in cleaned]

    # Fallback: sentence split
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 3]
    if len(sents) >= 2:
        return sents

    # Single chunk
    return [text] if text else []


# ---------------------------------------------------------------------------
# Budget reward — per-task, step-based
# ---------------------------------------------------------------------------

def _resolve_budget_from_metadata(
    task_metadata: Optional[Dict[str, Any]] = None,
    default_steps: int = 4,
    default_tokens: int = 120,
) -> Dict[str, Any]:
    """Extract budget parameters from task-level metadata.

    Priority chain:
    1. ``reasoning_budget_steps`` (per-task)
    2. ``reasoning_budget_tokens`` (per-task)
    3. ``reference_judge.reasoning_budget_steps``
    4. ``reference_judge.reasoning_budget_tokens``
    5. Fallback defaults
    """
    meta = task_metadata or {}
    ref_judge = meta.get("reference_judge", {}) or {}

    budget_steps = meta.get("reasoning_budget_steps") or ref_judge.get("reasoning_budget_steps")
    budget_tokens = meta.get("reasoning_budget_tokens") or ref_judge.get("reasoning_budget_tokens")
    budget_source = meta.get("budget_source") or ref_judge.get("budget_source") or "fallback_default"
    budget_mode = "fallback"

    if budget_steps is not None:
        budget_mode = "steps"
        result_steps = int(budget_steps)
    else:
        result_steps = default_steps

    if budget_tokens is not None:
        result_tokens = int(budget_tokens)
        if budget_mode == "fallback":
            budget_mode = "tokens"
    else:
        result_tokens = default_tokens

    return {
        "budget_steps": result_steps,
        "budget_tokens": result_tokens,
        "budget_source": budget_source,
        "budget_mode": budget_mode,
        "difficulty_level": meta.get("difficulty_level") or ref_judge.get("difficulty_level"),
        "expected_reasoning_steps": (
            meta.get("expected_reasoning_steps") or ref_judge.get("expected_reasoning_steps")
        ),
    }


def _compute_budget_score_step_based(
    num_steps: int,
    budget_steps: int,
) -> Tuple[float, Dict[str, Any]]:
    """Step-based budget raw score.

    - steps <= B        : +1.0
    - B < steps <= B+2  : +0.5
    - B+2 < steps <= B+4: +0.2
    - steps > B+4       : -1.0
    """
    if num_steps <= budget_steps:
        raw = 1.0
    elif num_steps <= budget_steps + 2:
        raw = 0.5
    elif num_steps <= budget_steps + 4:
        raw = 0.2
    else:
        raw = -1.0

    normalized = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return normalized, {
        "budget_mode": "steps",
        "reasoning_steps": num_steps,
        "budget_steps": budget_steps,
        "budget_raw": raw,
        "budget_normalized": normalized,
    }


def _compute_budget_score_token_based(
    num_tokens: int,
    budget_tokens: int,
) -> Tuple[float, Dict[str, Any]]:
    """Token/word-based budget with linear decay (backward compat).

    Uses the same ``(raw + 1) / 2`` normalization as step-based,
    where raw ∈ [-1, +1]:
    - tokens ≤ B              → raw = +1.0  → normalized = 1.0
    - B < tokens ≤ 2B         → linear decay from +1.0 to -1.0
    - tokens > 2B             → raw = -1.0  → normalized = 0.0

    This is equivalent to the original formula ``max(0, 1 - overflow/B)``
    but expressed in the unified (raw+1)/2 convention.
    """
    if num_tokens == 0:
        return 0.0, {
            "budget_mode": "tokens",
            "reasoning_words": 0,
            "budget_tokens": budget_tokens,
            "budget_raw": -1.0,
            "budget_normalized": 0.0,
        }

    if num_tokens <= budget_tokens:
        raw = 1.0
    else:
        overflow = num_tokens - budget_tokens
        # raw ∈ [-1, +1]: +1 at B, 0 at 1.5B, -1 at 2B
        raw = 1.0 - 2.0 * overflow / budget_tokens
        raw = max(-1.0, raw)

    normalized = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return normalized, {
        "budget_mode": "tokens",
        "reasoning_words": num_tokens,
        "budget_tokens": budget_tokens,
        "budget_raw": round(raw, 4),
        "budget_normalized": round(normalized, 4),
    }


def budget_reward(
    completion: str,
    max_reasoning_words: int = 120,
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Budget reward with per-task budget.

    Priority:
    1. If per-task ``reasoning_budget_steps`` available → step-based scoring.
    2. If per-task ``reasoning_budget_tokens`` available → token-based scoring.
    3. Fallback → word-count with ``max_reasoning_words`` (original behavior).

    Returns:
        (score, details_dict)
    """
    reasoning = extract_reasoning(completion)
    budget_info = _resolve_budget_from_metadata(task_metadata)

    # Count steps by parsing
    steps = parse_reasoning_steps(reasoning)
    num_steps = len(steps)
    num_tokens = len(reasoning.split()) if reasoning else 0

    details = {
        "reasoning_steps": num_steps,
        "reasoning_words": num_tokens,
        "budget_source": budget_info["budget_source"],
        "budget_steps": budget_info["budget_steps"],
        "budget_tokens": budget_info["budget_tokens"],
        "difficulty_level": budget_info.get("difficulty_level"),
    }

    if budget_info["budget_mode"] == "steps":
        score, extra = _compute_budget_score_step_based(num_steps, budget_info["budget_steps"])
        details.update(extra)
        return score, details

    elif budget_info["budget_mode"] == "tokens":
        score, extra = _compute_budget_score_token_based(num_tokens, budget_info["budget_tokens"])
        details.update(extra)
        return score, details

    else:
        # Fallback: original word-count behavior for backward compat
        score, extra = _compute_budget_score_token_based(num_tokens, max_reasoning_words)
        details.update(extra)
        details["budget_mode"] = "fallback"
        details["budget_source"] = "fallback_default"
        return score, details


# ---------------------------------------------------------------------------
# Process reward — reference-step coverage, causal structure, shortcut check
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set:
    text = _normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    return set(tokens)


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = _token_set(a)
    set_b = _token_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _step_similarity(model_step: str, ref_step: str) -> float:
    """Similarity between a model step and a reference step.

    Uses a combination: 0.5 * Jaccard + 0.5 * difflib SequenceMatcher.
    No heavy dependencies.
    """
    jac = _jaccard_similarity(model_step, ref_step)
    try:
        sm = difflib.SequenceMatcher(None, _normalize_text(model_step), _normalize_text(ref_step))
        seq = sm.ratio()
    except Exception:
        seq = 0.0
    return 0.5 * jac + 0.5 * seq


def _compute_reference_step_coverage(
    model_steps: List[str],
    reference_steps: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """For each reference step, compute max similarity against any model step.

    Returns:
        (coverage_score, details)
    """
    if not reference_steps:
        return 0.0, {"model_reasoning_steps": model_steps, "reference_reasoning_steps": []}

    if not model_steps:
        return 0.0, {"model_reasoning_steps": [], "reference_reasoning_steps": reference_steps}

    per_ref_scores: List[float] = []
    best_model_indices: List[int] = []

    for ref_step in reference_steps:
        scores = [_step_similarity(ms, ref_step) for ms in model_steps]
        best = max(scores)
        best_idx = scores.index(best)
        per_ref_scores.append(best)
        best_model_indices.append(best_idx)

    coverage = sum(per_ref_scores) / len(per_ref_scores)

    return coverage, {
        "model_reasoning_steps": model_steps,
        "reference_reasoning_steps": reference_steps,
        "per_reference_step_scores": per_ref_scores,
        "reference_step_coverage": coverage,
    }


def _compute_causal_structure_score(reasoning: str) -> Tuple[float, Dict[str, Any]]:
    """Check whether reasoning has key causal structure.

    Checks:
    1. Mentions visual evidence
    2. Compares candidate players/objects
    3. Identifies difference / outlier / spy
    4. Derives final answer from evidence (not just output)
    """
    r = _normalize_text(reasoning)
    if not r:
        return 0.0, {"causal_checks": {}}

    checks = {}

    # 1. Visual evidence
    visual_terms = [
        "color", "colour", "shape", "size", "material", "object", "objects",
        "red", "blue", "green", "yellow", "purple", "cyan", "brown", "gray", "grey",
        "metal", "metallic", "rubber", "matte", "shiny",
        "sphere", "cube", "cylinder", "large", "small",
        "left", "right", "front", "behind",
    ]
    visual_hits = sum(1 for t in visual_terms if t in r)
    checks["visual_evidence"] = visual_hits >= 2

    # 2. Comparison
    comparison_terms = [
        "compare", "compared", "comparison", "same", "similar", "match", "matches",
        "different", "differs", "unlike", "whereas", "while", "but",
        "majority", "others", "other two", "remaining",
    ]
    checks["comparison"] = any(t in r for t in comparison_terms)

    # 3. Outlier / spy identification
    outlier_patterns = [
        r"\bdifferent from\b", r"\bdiffers from\b", r"\bdoes not match\b",
        r"\bdoesn't match\b", r"\bnot match\b", r"\bunique\b", r"\boutlier\b",
        r"\bminority\b", r"\bonly\b", r"\bstands out\b",
    ]
    checks["outlier_identification"] = any(re.search(p, r) for p in outlier_patterns)

    # 4. Evidence-to-answer derivation
    derivation_terms = [
        "therefore", "thus", "so", "hence", "conclude", "conclusion",
        "must be", "should be", "based on", "according to", "indicates",
        "suggests", "confirm", "confirmed",
    ]
    checks["evidence_to_answer"] = any(t in r for t in derivation_terms)

    # Score: each check = 0.25
    score = sum(1.0 for v in checks.values() if v) / 4.0
    return score, {"causal_checks": checks, "causal_structure_score": score}


def _detect_shortcut(reasoning: str) -> Tuple[bool, Dict[str, Any]]:
    """Detect shortcut reasoning patterns.

    Returns (is_shortcut, evidence_dict).
    """
    r = _normalize_text(reasoning)
    if not r:
        return True, {"shortcut_reasons": ["empty_reasoning"]}

    reasons: List[str] = []
    num_words = len(r.split())

    # Very short reasoning (< 8 words) with answer
    if num_words < 8:
        reasons.append("very_short_reasoning")

    # No visual evidence
    visual_basic = {"color", "shape", "object", "image", "player"}
    if not any(t in r for t in visual_basic):
        reasons.append("missing_visual_evidence")

    # No comparison
    compare_terms = {"compare", "same", "match", "different", "whereas", "majority"}
    if not any(t in r for t in compare_terms):
        reasons.append("missing_comparison")

    # Guess language without evidence
    guess_terms = {"i guess", "probably", "maybe", "perhaps", "might be"}
    has_guess = any(t in r for t in guess_terms)
    has_evidence = bool(re.search(r"\b(?:because|since|as the|due to)\b", r))
    if has_guess and not has_evidence:
        reasons.append("guess_without_evidence")

    is_shortcut = len(reasons) >= 2
    return is_shortcut, {
        "shortcut_flag": is_shortcut,
        "shortcut_reasons": reasons,
        "reasoning_word_count": num_words,
    }


# Legacy CLEVR spy-player structure score — kept as fallback heuristic
def _clevr_spy_reasoning_structure_score(
    reasoning: str,
    reference_reasoning: Optional[str] = None,
) -> float:
    """Task-specific CLEVR spy-player heuristic.  Kept for backward compat."""
    reasoning_norm = _normalize_text(reasoning)
    if reasoning_norm == "":
        return 0.0

    score = 0.0

    # 1. Multiple players/images
    numbered = set(re.findall(r"\b(?:player|image|agent)\s*[-#:]*\s*(\d+)\b", reasoning_norm))
    ordinal = set(re.findall(r"\b(?:first|second|third|1st|2nd|3rd|one|two|three)\b", reasoning_norm))
    mentions_player = any(t in reasoning_norm for t in ["player", "players", "image", "images"])
    if len(numbered) >= 2:
        score += 0.20
    elif mentions_player and len(ordinal) >= 2:
        score += 0.20
    elif mentions_player:
        score += 0.10

    # 2. Comparison language
    comp_terms = [
        "compare", "compared", "comparison", "same", "similar", "match", "matches",
        "matching", "different", "differs", "unlike", "whereas", "while", "but",
        "majority", "others", "other two", "remaining",
    ]
    if any(t in reasoning_norm for t in comp_terms):
        score += 0.20

    # 3. Minority/outlier logic
    minority = [
        r"\bdifferent from\b", r"\bdiffers from\b", r"\bdoes not match\b",
        r"\bdoesn't match\b", r"\bnot match\b", r"\bunique\b", r"\boutlier\b",
        r"\bminority\b", r"\bonly\b", r"\bstands out\b",
    ]
    if any(re.search(p, reasoning_norm) for p in minority):
        score += 0.20

    # 4. Visual evidence
    visual = [
        "color", "colour", "shape", "size", "material", "object", "objects",
        "red", "blue", "green", "yellow", "purple", "cyan", "brown", "gray", "grey",
        "metal", "metallic", "rubber", "matte", "shiny",
        "sphere", "spheres", "cube", "cubes", "cylinder", "cylinders",
        "large", "small", "left", "right", "front", "behind",
    ]
    visual_hits = sum(1 for t in visual if t in reasoning_norm)
    if visual_hits >= 2:
        score += 0.20
    elif visual_hits == 1:
        score += 0.10

    # 5. Spy selection connection
    has_spy = "spy" in reasoning_norm
    sel_lang = any(
        t in reasoning_norm
        for t in [
            "therefore", "thus", "so", "hence", "conclude", "select",
            "choose", "chosen", "answer", "must be", "should be", "is the spy",
        ]
    )
    spy_sel = [
        r"\b(?:player|image|agent)\s*[-#:]*\s*\d+\s+(?:is|must be|should be|appears to be|seems to be)\s+(?:the\s+)?spy\b",
        r"\b(?:spy\s+is|answer\s+is|choose|select)\s+(?:the\s+)?(?:player|image|agent)?\s*[-#:]*\s*\d+\b",
    ]
    has_explicit = any(re.search(p, reasoning_norm) for p in spy_sel)
    if has_spy and (sel_lang or has_explicit):
        score += 0.20
    elif has_spy:
        score += 0.10

    return min(1.0, score)


def process_reward(
    completion: str,
    reference_reasoning: Optional[str] = None,
    reference_reasoning_steps: Optional[List[str]] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Process reward: reference-step distance + causal structure + shortcut detection.

    If ``reference_reasoning_steps`` is available:
        process = 0.5 * reference_step_coverage + 0.3 * causal_structure + 0.2 * heuristic_structure
    Else:
        process = max(jaccard(reasoning, reference_reasoning), heuristic_structure)

    Also applies a shortcut penalty: if shortcut is detected, process is capped at 0.4.

    Returns:
        (score, details_dict)
    """
    reasoning = extract_reasoning(completion)
    if reasoning == "":
        return 0.0, {"process_source": "empty_reasoning", "model_reasoning_steps": []}

    model_steps = parse_reasoning_steps(reasoning)

    # Resolve reference steps
    meta = task_metadata or {}
    ref_steps = reference_reasoning_steps or meta.get("reference_reasoning_steps")
    # Also try to parse from reference_reasoning string
    if not ref_steps and reference_reasoning:
        ref_steps = parse_reasoning_steps(reference_reasoning)

    # Shortcut detection
    is_shortcut, shortcut_details = _detect_shortcut(reasoning)

    # Causal structure
    causal_score, causal_details = _compute_causal_structure_score(reasoning)

    # Heuristic structure, used when there are no reference steps to cover
    heuristic_score = _clevr_spy_reasoning_structure_score(reasoning, reference_reasoning)

    if ref_steps and len(ref_steps) >= 1:
        # Reference-step based mode
        coverage, coverage_details = _compute_reference_step_coverage(model_steps, ref_steps)

        process_raw = 0.5 * coverage + 0.3 * causal_score + 0.2 * heuristic_score
        process_source = "reference_step_distance"

        details = {
            "process_source": process_source,
            "reference_step_coverage": coverage,
            "causal_structure_score": causal_score,
            "heuristic_structure_score": heuristic_score,
            **coverage_details,
            **causal_details,
            **shortcut_details,
        }
    elif reference_reasoning and reference_reasoning.strip():
        # Legacy mode: Jaccard + structure
        jaccard = _jaccard_similarity(reasoning, reference_reasoning)
        process_raw = max(jaccard, heuristic_score)
        process_source = "lexical_jaccard_with_heuristic"

        details = {
            "process_source": process_source,
            "reference_step_coverage": None,
            "causal_structure_score": causal_score,
            "heuristic_structure_score": heuristic_score,
            "jaccard_with_reference": jaccard,
            "model_reasoning_steps": model_steps,
            "reference_reasoning_steps": ref_steps or [],
            **causal_details,
            **shortcut_details,
        }
    else:
        # No reference at all — pure heuristic
        process_raw = heuristic_score
        process_source = "heuristic_only"

        details = {
            "process_source": process_source,
            "reference_step_coverage": None,
            "causal_structure_score": causal_score,
            "heuristic_structure_score": heuristic_score,
            "model_reasoning_steps": model_steps,
            "reference_reasoning_steps": [],
            **causal_details,
            **shortcut_details,
        }

    # Shortcut penalty
    if is_shortcut:
        process_raw = min(process_raw, 0.4)
        details["shortcut_penalty_applied"] = True
    else:
        details["shortcut_penalty_applied"] = False

    process_score = max(0.0, min(1.0, process_raw))
    details["shortcut_flag"] = is_shortcut
    return process_score, details


# ---------------------------------------------------------------------------
# Grounding reward — IoU first
# ---------------------------------------------------------------------------

def grounding_reward(
    grounding_score: Optional[float] = None,
    gold_evidence_boxes: Optional[List[List[float]]] = None,
    predicted_evidence_boxes: Optional[List[List[float]]] = None,
    box_format: str = "xyxy",
    task_metadata: Optional[Dict[str, Any]] = None,
    completion: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Grounding reward with model-bbox priority.

    Priority:
    1. Model-emitted normalized ``<bbox>`` parsed from the completion → convert
       to pixels → IoU vs gold pixel boxes. source_type=metadata_bbox_iou.
    2. Else pre-supplied predicted_evidence_boxes (keyword→metadata proxy) → IoU.
    3. Else external grounding_score → clamp (fallback).
    4. Else 0.0.

    Returns (score, details_dict) with the full grounding_details schema.
    """
    # Pull boxes / sizes from metadata if not directly provided
    meta = task_metadata or {}
    if gold_evidence_boxes is None:
        gold_evidence_boxes = meta.get("gold_evidence_boxes")
    if predicted_evidence_boxes is None:
        predicted_evidence_boxes = meta.get("predicted_evidence_boxes")
    if grounding_score is None:
        grounding_score = meta.get("grounding_score")
    if image_width is None:
        image_width = meta.get("image_width")
    if image_height is None:
        image_height = meta.get("image_height")
    # CLEVR replacement images are 320x240; default when the task metadata does
    # not carry an explicit size so model-emitted normalized bbox can still be
    # converted to pixels and scored against the gold pixel boxes.
    if image_width is None:
        image_width = 320
    if image_height is None:
        image_height = 240

    # --- Layer 1: parse model-emitted normalized bbox from the completion ---
    # Delegate to score_model_bbox_grounding so the
    # offline scorer and the live GRPO reward compute the SAME model-bbox IoU
    # and emit the SAME audit fields (no divergent second implementation).
    from open_r1.self_evolve.grounding_iou import (
        parse_evidence_boxes,
        score_model_bbox_grounding,
    )

    valid_player_ids = meta.get("valid_player_ids")
    if not isinstance(valid_player_ids, list) or not valid_player_ids:
        valid_player_ids = None

    if gold_evidence_boxes:
        shared = score_model_bbox_grounding(
            completion,
            gold_boxes_pixel=gold_evidence_boxes,
            image_width=int(image_width),
            image_height=int(image_height),
            valid_player_ids=valid_player_ids,
            config=_active_grounding_cfg(),
        )
        # Only short-circuit to the model-bbox signal when a valid box exists;
        # otherwise fall through to the proxy / external-score layers below so we
        # do not regress tasks that rely on those.
        if shared["bbox_present"]:
            base_details = {
                "pred_bbox_normalized": shared.get("predicted_bbox_norm"),
                "pred_bbox_pixel": shared.get("predicted_bbox_pixel"),
                "gold_bbox_pixel": gold_evidence_boxes,
                "bbox_present": shared["bbox_present"],
                "bbox_valid": shared["bbox_valid"],
                "bbox_invalid_reason": shared.get("bbox_invalid_reason"),
                "bbox_player_id": shared.get("bbox_player_id"),
                "bbox_player_id_valid": shared.get("bbox_player_id_valid"),
                "predicted_bbox_norm": shared.get("predicted_bbox_norm"),
                "reference_bbox_norm": shared.get("reference_bbox_norm"),
                "metadata_bbox_iou": shared.get("metadata_bbox_iou"),
                "bbox_iou": shared.get("metadata_bbox_iou"),
                "grounding_iou": shared.get("metadata_bbox_iou"),
                "grounding_reward_components": shared.get("grounding_reward_components"),
                "grounding_source": shared["grounding_reward_source"],
                "grounding_source_type": shared["grounding_reward_source"],
                "per_box_ious": shared.get("per_box_ious", []),
                "gold_evidence_boxes": gold_evidence_boxes,
                "predicted_evidence_boxes": shared.get("predicted_bbox_pixel"),
                "box_format": "xyxy",
                "grounding_error": None,
                "fallback_used": shared["grounding_reward_source"] != "model_bbox_iou",
            }
            return float(shared["grounding_reward"]), base_details

    bbox_parsed = parse_evidence_boxes(completion, image_width, image_height)
    pred_bbox_norm = bbox_parsed["boxes_normalized"] or None
    pred_bbox_pixel = bbox_parsed["boxes_pixel"] or None

    base_details: Dict[str, Any] = {
        "pred_bbox_normalized": pred_bbox_norm,
        "pred_bbox_pixel": pred_bbox_pixel,
        "gold_bbox_pixel": gold_evidence_boxes,
        "bbox_iou": None,
        "bbox_valid": bbox_parsed["bbox_valid"],
        "bbox_parse_error": bbox_parsed["bbox_parse_error"],
        "fallback_used": False,
    }
    result = resolve_grounding_score(
        gold_evidence_boxes=gold_evidence_boxes,
        predicted_evidence_boxes=predicted_evidence_boxes,
        box_format=box_format,
        grounding_score=grounding_score,
        grounding_source=meta.get("grounding_source"),
    )

    score = float(result["grounding_score"])
    fallback_reason = (
        "no_valid_model_bbox" if not pred_bbox_pixel else "no_gold_pixel_boxes"
    )
    base_details.update({
        "grounding_source": result["grounding_source"],
        "grounding_source_type": result["grounding_source"],
        "grounding_iou": result.get("grounding_iou"),
        "gold_evidence_boxes": result.get("gold_evidence_boxes"),
        "predicted_evidence_boxes": result.get("predicted_evidence_boxes"),
        "box_format": result.get("box_format"),
        "grounding_error": result.get("grounding_error"),
        "per_box_ious": result.get("per_box_ious", []),
        "fallback_used": True,
        "fallback_reason": fallback_reason,
    })
    return score, base_details


# ---------------------------------------------------------------------------
# Consistency reward — per-component
# ---------------------------------------------------------------------------

# Optional LLM-as-judge backend for consistency
def _llm_consistency_judge(
    completions: List[str],
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Call LLM to judge consistency across trajectories.

    Only called when OPENAI_API_KEY is set and the caller explicitly opts in.
    Returns None on any failure — caller must fall back to heuristic.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("LLM_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("REFERENCE_VLM_MODEL") or "gpt-4o"
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if len(completions) < 2:
        return {
            "reasoning_consistency_score": 1.0,
            "evidence_consistency_score": 1.0,
            "contradiction_flag": False,
            "shortcut_flag": False,
            "judge_rationale": "Single trajectory — consistency is 1.0 by definition.",
        }

    # Extract answers and reasoning for the judge
    answers = [extract_answer(c) for c in completions]
    reasonings = [extract_reasoning(c) for c in completions]

    prompt = (
        "You are evaluating consistency across multiple trajectories for the SAME visual reasoning task.\n\n"
        "Trajectories:\n"
    )
    for i, (ans, reason) in enumerate(zip(answers, reasonings)):
        prompt += f"Trajectory {i}:\n  Answer: {ans}\n  Reasoning: {reason[:500]}\n\n"

    prompt += (
        "Output a JSON object with:\n"
        '  "reasoning_consistency_score": float in [0,1] — how consistent the reasoning paths are\n'
        '  "evidence_consistency_score": float in [0,1] — how consistent the evidence cited is\n'
        '  "contradiction_flag": bool — whether any two trajectories directly contradict\n'
        '  "shortcut_flag": bool — whether any trajectory appears to be a shortcut\n'
        '  "judge_rationale": string — brief explanation\n'
        "Only output the JSON object."
    )

    try:
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a rigorous consistency evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _compute_evidence_consistency(
    completions: List[str],
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Compute evidence consistency across trajectories.

    Priority:
    1. If predicted_evidence_boxes exist → compare box overlap
    2. Else → fallback to visual keyword overlap
    """
    meta = task_metadata or {}
    n = len(completions)
    if n <= 1:
        return [1.0] * n, {"evidence_method": "single_trajectory"}

    # Try box-based evidence comparison
    all_boxes = meta.get("predicted_evidence_boxes_per_trajectory")
    if all_boxes and isinstance(all_boxes, list) and len(all_boxes) == n:
        # Compare box sets via IoU
        from open_r1.self_evolve.grounding_iou import compute_grounding_iou
        scores = []
        for i in range(n):
            others = [all_boxes[j] for j in range(n) if j != i]
            if not others or all_boxes[i] is None:
                scores.append(0.0)
                continue
            ious = []
            for other_boxes in others:
                if other_boxes is None:
                    continue
                result = compute_grounding_iou(
                    gold_boxes=other_boxes,
                    predicted_boxes=all_boxes[i],
                )
                ious.append(result["grounding_iou"])
            scores.append(sum(ious) / len(ious) if ious else 0.0)
        return scores, {"evidence_method": "box_iou_comparison"}

    # Fallback: visual keyword overlap
    reasonings = [extract_reasoning(c) for c in completions]
    visual_keywords = {
        "color", "colour", "shape", "size", "material", "object",
        "red", "blue", "green", "yellow", "purple", "cyan", "brown", "gray", "grey",
        "sphere", "cube", "cylinder", "metal", "rubber", "large", "small",
    }
    keyword_sets = []
    for r in reasonings:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", _normalize_text(r)))
        keyword_sets.append(tokens & visual_keywords)

    scores = []
    for i in range(n):
        if not keyword_sets[i]:
            scores.append(0.0)
            continue
        overlaps = []
        for j in range(n):
            if i == j:
                continue
            union = keyword_sets[i] | keyword_sets[j]
            if not union:
                continue
            overlaps.append(len(keyword_sets[i] & keyword_sets[j]) / len(union))
        scores.append(sum(overlaps) / len(overlaps) if overlaps else 0.0)
    return scores, {"evidence_method": "visual_keyword_overlap"}


def field_consistency_reward(
    completion: str,
    boxes: Optional[List[List[float]]] = None,
    players: Optional[List[Optional[int]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Self-consistency of a trajectory: do its own fields agree?

    Unlike group-modal agreement (constant inside a rollout group, hence zero
    GRPO gradient), every check here varies per rollout and ties the answer,
    the evidence boxes and the reasoning to each other.
    """
    if boxes is None or players is None:
        from open_r1.self_evolve.grounding_iou import parse_evidence_boxes

        parsed = parse_evidence_boxes(completion)
        boxes = parsed["boxes_normalized"]
        players = parsed["players"]

    pred = parse_structured_answer(completion)
    spy, count = pred.get("spy"), pred.get("changed_attributes")
    reasoning = _normalize_text(extract_reasoning(completion))

    # Hard checks are outright self-contradictions and gate the score;
    # soft checks are averaged.
    hard: Dict[str, bool] = {}
    soft: Dict[str, bool] = {"has_evidence_box": bool(boxes)}
    if boxes and spy is not None:
        hard["box_player_matches_spy"] = all(
            p is not None and int(p) == int(spy) for p in players
        )
    if boxes and count is not None:
        # A changed object contributes >= 1 changed attribute, so #boxes must
        # lie in [1, changed_attributes]. Makes box-spam self-contradictory.
        hard["box_count_within_attribute_count"] = 1 <= len(boxes) <= int(count)
    if boxes:
        soft["boxes_distinct"] = _boxes_are_distinct(boxes)
    if spy is not None and reasoning:
        soft["reasoning_names_answered_spy"] = bool(
            re.search(rf"\b(?:player|image|agent)\s*[-#:]*\s*{int(spy)}\b", reasoning)
        )

    checks = {**hard, **soft}
    score = (sum(1.0 for v in soft.values() if v) / len(soft)) if soft else 0.0
    if not all(hard.values()):
        score = 0.0
    return score, {
        "consistency_mode": "field_consistency",
        "consistency_checks": checks,
        "predicted_spy": spy,
        "predicted_changed_attributes": count,
        "num_evidence_boxes": len(boxes or []),
    }


def _boxes_are_distinct(boxes: List[List[float]], iou_threshold: float = 0.9) -> bool:
    """False if any two predicted boxes are near-duplicates."""
    from open_r1.self_evolve.grounding_iou import box_iou

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                return False
    return True


def reasoning_consistency_reward(
    completions: List[str],
    answer_weight: float = 0.4,
    evidence_weight: float = 0.3,
    reasoning_weight: float = 0.3,
    task_metadata: Optional[Dict[str, Any]] = None,
    use_llm_judge: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    """Component-based reasoning consistency reward.

    consistency = 0.4 * answer_consistency + 0.3 * evidence_consistency + 0.3 * reasoning_consistency

    Can optionally use LLM-as-judge for the reasoning/evidence components.

    Returns:
        (list_of_scores, details_dict)
    """
    n = len(completions)
    if n == 0:
        return [], {"consistency_source": "empty"}

    if n == 1:
        return [1.0], {"consistency_source": "single_trajectory"}

    answers = [_normalize_text(extract_answer(c)) for c in completions]
    reasonings = [extract_reasoning(c) for c in completions]

    # --- Answer consistency (same as before) ---
    answer_scores = []
    for ans in answers:
        if ans == "":
            answer_scores.append(0.0)
        else:
            same_count = sum(1 for other in answers if other == ans)
            answer_scores.append(same_count / n)

    # --- Evidence consistency ---
    evidence_scores, evidence_details = _compute_evidence_consistency(completions, task_metadata)

    # --- Reasoning consistency ---
    is_llm = False
    llm_result = None

    if use_llm_judge and os.environ.get("OPENAI_API_KEY"):
        llm_result = _llm_consistency_judge(completions, task_metadata)
        if llm_result is not None:
            is_llm = True

    if is_llm and llm_result is not None:
        reasoning_cons = llm_result.get("reasoning_consistency_score", 0.5)
        evidence_llm = llm_result.get("evidence_consistency_score")
        reasoning_scores = [reasoning_cons] * n
        if evidence_llm is not None:
            evidence_scores = [float(evidence_llm)] * n
    else:
        # Heuristic: Jaccard similarity of reasoning
        reasoning_scores = []
        for i, ri in enumerate(reasonings):
            if ri == "":
                reasoning_scores.append(0.0)
                continue
            sims = [_jaccard_similarity(ri, reasonings[j]) for j in range(n) if j != i]
            reasoning_scores.append(sum(sims) / len(sims) if sims else 0.0)

    # --- Combine ---
    final_scores = []
    for a, e, r in zip(answer_scores, evidence_scores, reasoning_scores):
        s = answer_weight * a + evidence_weight * e + reasoning_weight * r
        final_scores.append(max(0.0, min(1.0, s)))

    details = {
        "consistency_source": "llm_judge" if is_llm else "heuristic",
        "llm_judge_used": is_llm,
        "answer_consistency": answer_scores,
        "evidence_consistency": evidence_scores,
        "reasoning_consistency": reasoning_scores,
        "evidence_method": evidence_details.get("evidence_method", "unknown"),
        "consistency_judge_rationale": llm_result.get("judge_rationale") if llm_result else None,
        "contradiction_flag": llm_result.get("contradiction_flag") if llm_result else None,
    }

    return final_scores, details


# ---------------------------------------------------------------------------
# compute_reward_vector  (backward compatible + reward_details)
# ---------------------------------------------------------------------------

REWARD_KEYS = ["answer", "budget", "process", "grounding", "consistency"]


def compute_reward_vector(
    completion: str,
    solution: str,
    reference_reasoning: Optional[str] = None,
    grounding_score: Optional[float] = None,
    consistency_score: Optional[float] = None,
    max_reasoning_words: int = 120,
    task_metadata: Optional[Dict[str, Any]] = None,
    gold_evidence_boxes: Optional[List[List[float]]] = None,
    predicted_evidence_boxes: Optional[List[List[float]]] = None,
    box_format: str = "xyxy",
    reference_reasoning_steps: Optional[List[str]] = None,
    include_details: bool = False,
    enable_answer_judge: bool = False,
    answer_judge_dry_run: bool = True,
    force_answer_judge: bool = False,
    problem: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the five-dimensional reward vector for one trajectory.

    Returns:
        dict with keys ``answer``, ``budget``, ``process``, ``grounding``, ``consistency``.
        If ``include_details=True``, also includes ``reward_details``.
    """
    # Answer — same definition as the live reward: per-field when configured.
    _cfg = _active_reward_config()
    exact_match = answer_reward(completion, solution)
    if _cfg.answer_cfg["mode"] == "structured_fields":
        answer, _answer_det = structured_answer_reward(
            completion, solution,
            fields=_cfg.answer_cfg["fields"],
            off_by_one_credit=float(_cfg.answer_cfg["off_by_one_credit"]),
            graded_fields=_cfg.answer_cfg["graded_fields"],
        )
    else:
        answer, _answer_det = exact_match, {"answer_mode": "exact_match"}

    # Answer judge (layer 2): real LLM judge as a fallback / audit path, invoked
    # by CONDITIONAL INVOCATION (not a reward gate/cap) — only when exact match
    # is uncertain (parse failure / synonym / NL phrasing) and
    # ``enable_answer_judge`` is on. Exact-match successes skip the judge.
    answer_judge_result: Optional[Dict[str, Any]] = None
    answer_judge_invoke_reason = "judge_disabled"
    if enable_answer_judge:
        should, answer_judge_invoke_reason = _answer_judge_should_invoke(
            completion, solution
        )
        # Experiment override: force invocation even on exact-match success so a
        # live-judge probe can demonstrate real judge involvement. Default off;
        # only set by the caller from SELF_EVOLVE_ANSWER_JUDGE_FORCE / _SAMPLE_N.
        if force_answer_judge and not should:
            should = True
            answer_judge_invoke_reason = "forced_experiment_probe"
        if should:
            answer_judge_result = answer_judge(
                completion, solution, problem=problem,
                dry_run=answer_judge_dry_run,
            )
            # Promote a confident judge verdict to the scalar answer score when
            # the rule could not verify it. A real judge correcting a false
            # negative is recorded with answer_source_type=llm_judge — never
            # disguised as an exact match. A forced probe on an already-correct
            # answer does NOT change the score (it stays 1.0 from exact match).
            if (
                answer_judge_result.get("answer_judge_used")
                and answer_judge_result.get("judge_correct") is True
                and answer < 1.0
            ):
                answer = 1.0

    # Budget (returns tuple now)
    budget_val, budget_details = budget_reward(
        completion,
        max_reasoning_words=max_reasoning_words,
        task_metadata=task_metadata,
    )

    # Process (returns tuple now)
    process_val, process_details = process_reward(
        completion,
        reference_reasoning=reference_reasoning,
        reference_reasoning_steps=reference_reasoning_steps,
        task_metadata=task_metadata,
    )

    # Grounding (returns tuple now)
    grounding_val, grounding_details = grounding_reward(
        grounding_score=grounding_score,
        gold_evidence_boxes=gold_evidence_boxes,
        predicted_evidence_boxes=predicted_evidence_boxes,
        box_format=box_format,
        task_metadata=task_metadata,
        completion=completion,
    )

    # Consistency (single-trajectory: use passed-in value)
    if consistency_score is None:
        consistency_score = 0.0
    consistency_val = max(0.0, min(1.0, float(consistency_score)))

    reward_vector = {
        "answer": answer,
        "budget": budget_val,
        "process": process_val,
        "grounding": grounding_val,
        "consistency": consistency_val,
    }

    if include_details:
        # source_type per dimension: rule_based terms are computed
        # by code and must NOT call an LLM; metadata_based reads task metadata;
        # reference_vlm_based comes from the Reference VLM budget/steps;
        # llm_based is only used where rules cannot judge reliably.
        budget_src_type = (
            "reference_vlm_based"
            if str(budget_details.get("budget_source", "")).startswith("openai_reference")
            or str(budget_details.get("budget_source", "")).startswith("reference")
            else ("metadata_based"
                  if budget_details.get("budget_mode") in ("steps", "tokens")
                  and budget_details.get("budget_source") != "fallback_default"
                  else "rule_based")
        )
        grounding_src = str(grounding_details.get("grounding_source", ""))
        grounding_src_type = (
            "metadata_bbox_iou" if "bbox_iou" in grounding_src
            else "metadata_based" if "metadata" in grounding_src or "iou" in grounding_src
            else "rule_based"
        )
        reward_vector["reward_details"] = {
            # Top-level auditable flags (rule_based; routing reads these).
            "format_valid": is_format_valid(completion),
            "shortcut_detected": bool(process_details.get("shortcut_flag", False)),
            "answer": {
                "score": answer,
                # Per-field breakdown when the config scores fields separately.
                # Without this the offline records carry only the blended score,
                # and every per-field diagnostic the docs ask you to watch —
                # spy vs count accuracy, the count's collapse toward one value —
                # has nothing to read.
                **_answer_det,
                "source_type": (
                    answer_judge_result.get("answer_source_type", "rule_based")
                    if answer_judge_result and answer_judge_result.get("answer_judge_used")
                    else "rule_based"
                ),
                "reason": (
                    "per-field match against gold answer"
                    if _answer_det.get("answer_mode") == "structured_fields"
                    else "exact normalized string match against gold answer"
                ) + ("" if not answer_judge_result
                     else f"; judge_path={answer_judge_invoke_reason}"),
                "answer_match": bool(answer_reward(completion, solution) >= 1.0),
                "exact_match_result": bool(answer_reward(completion, solution) >= 1.0),
                "answer_judge_used": bool(
                    answer_judge_result.get("answer_judge_used")
                    if answer_judge_result else False
                ),
                "answer_judge_live": bool(
                    answer_judge_result
                    and answer_judge_result.get("answer_judge_used")
                    and answer_judge_result.get("blocked_reason") is None
                ),
                "answer_judge_model": (
                    answer_judge_result.get("answer_judge_model")
                    if answer_judge_result else None
                ),
                "answer_judge_score": (
                    answer_judge_result.get("judge_confidence")
                    if answer_judge_result else None
                ),
                "answer_judge_correct": (
                    answer_judge_result.get("judge_correct")
                    if answer_judge_result else None
                ),
                "answer_judge_error": (
                    answer_judge_result.get("blocked_reason")
                    if answer_judge_result and str(
                        answer_judge_result.get("blocked_reason") or "").startswith("api_error")
                    else None
                ),
                "judge_result": answer_judge_result,
                "judge_reason": (
                    answer_judge_result.get("judge_reason") if answer_judge_result else None
                ),
            },
            "budget": {
                "score": budget_val,
                "source_type": budget_src_type,
                "reason": f"budget_mode={budget_details.get('budget_mode')}, "
                          f"source={budget_details.get('budget_source')} "
                          "(rule-computed step/token count vs budget; no LLM judge)",
                **budget_details,
            },
            "process": {
                "score": process_val,
                "source_type": "rule_based",
                "reason": f"process_source={process_details.get('process_source')} "
                          "(reference-step coverage + causal structure + shortcut; "
                          "lexical/rule-based, no LLM judge)",
                **process_details,
            },
            "grounding": {
                "score": grounding_val,
                "source_type": grounding_src_type,
                "reason": f"grounding_source={grounding_src} "
                          "(metadata box IoU proxy; rule-computed)",
                **grounding_details,
            },
            "consistency": {
                "score": consistency_val,
                "source_type": "rule_based",
                "reason": "single-trajectory wrapper assigned value; group "
                          "consistency (rule_based or llm_based) attached at group level",
                "consistency_source": "single_trajectory_wrapper",
                "assigned_consistency_score": consistency_val,
            },
        }

    return reward_vector


def compute_group_reward_vectors(
    completions: List[str],
    solution: str,
    reference_reasoning: Optional[str] = None,
    grounding_scores: Optional[List[Optional[float]]] = None,
    max_reasoning_words: int = 120,
    task_metadata: Optional[Dict[str, Any]] = None,
    gold_evidence_boxes: Optional[List[List[float]]] = None,
    predicted_evidence_boxes_per_traj: Optional[List[Optional[List[List[float]]]]] = None,
    box_format: str = "xyxy",
    reference_reasoning_steps: Optional[List[str]] = None,
    use_llm_consistency_judge: bool = False,
    include_details: bool = False,
    enable_answer_judge: bool = False,
    answer_judge_dry_run: bool = True,
    answer_judge_force: bool = False,
    answer_judge_sample_n: int = 0,
    problem: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute reward vectors for multiple trajectories of the same task.

    This is the main entry point for the self-evolution pipeline.
    Consistency reward is computed across all trajectories of the task.
    Per-task budget from metadata is passed through to each trajectory.

    ``answer_judge_force`` invokes the answer judge on every trajectory;
    ``answer_judge_sample_n`` invokes it on the first N trajectories of this
    group even when exact match succeeds (experiment probe). Both default off.
    """
    n = len(completions)

    # Consistency (multi-trajectory)
    if _active_reward_config().consistency_cfg["mode"] == "field_consistency":
        _fc = [field_consistency_reward(c) for c in completions]
        consistency_scores = [x[0] for x in _fc]
        consistency_details = [x[1] for x in _fc]
    else:
        consistency_scores, consistency_details = reasoning_consistency_reward(
            completions,
            task_metadata=task_metadata,
            use_llm_judge=use_llm_consistency_judge,
        )

    if grounding_scores is None:
        grounding_scores = [None] * n
    if len(grounding_scores) != n:
        raise ValueError("grounding_scores must have the same length as completions.")

    # Resolve per-trajectory predicted boxes
    pred_boxes_list: List[Optional[List[List[float]]]] = [None] * n
    if predicted_evidence_boxes_per_traj is not None:
        if len(predicted_evidence_boxes_per_traj) != n:
            raise ValueError("predicted_evidence_boxes_per_traj must have same length as completions.")
        pred_boxes_list = predicted_evidence_boxes_per_traj

    # Resolve reference steps from metadata if not directly provided
    ref_steps = reference_reasoning_steps
    if not ref_steps and task_metadata:
        ref_steps = task_metadata.get("reference_reasoning_steps")

    reward_vectors = []
    for i, (completion, cons_score, gs) in enumerate(zip(completions, consistency_scores, grounding_scores)):
        # Experiment probe: force the judge on every trajectory, or on the first
        # ``answer_judge_sample_n`` of this group.
        force_judge = bool(answer_judge_force) or (i < int(answer_judge_sample_n))
        rv = compute_reward_vector(
            completion=completion,
            solution=solution,
            reference_reasoning=reference_reasoning,
            grounding_score=gs,
            consistency_score=cons_score,
            max_reasoning_words=max_reasoning_words,
            task_metadata=task_metadata,
            gold_evidence_boxes=gold_evidence_boxes,
            predicted_evidence_boxes=pred_boxes_list[i],
            box_format=box_format,
            reference_reasoning_steps=ref_steps,
            include_details=include_details,
            enable_answer_judge=enable_answer_judge,
            answer_judge_dry_run=answer_judge_dry_run,
            force_answer_judge=force_judge,
            problem=problem,
        )
        reward_vectors.append(rv)

    if include_details and len(reward_vectors) > 0:
        # field_consistency scores each trajectory against itself, so its
        # details are per-trajectory and belong on that trajectory. The legacy
        # group-modal path returns one dict for the whole group instead.
        if isinstance(consistency_details, list):
            for rv, det in zip(reward_vectors, consistency_details):
                if "reward_details" not in rv:
                    continue
                rv["reward_details"]["consistency_group"] = {
                    **(det or {}),
                    "source_type": "rule_based",
                    "reason": "field consistency: the trajectory agrees with itself",
                }
        elif "reward_details" in reward_vectors[0]:
            group_details = dict(consistency_details)
            group_details["source_type"] = (
                "llm_based" if consistency_details.get("llm_judge_used") else "rule_based"
            )
            group_details["reason"] = (
                "LLM-as-judge consistency (only where rules cannot judge)"
                if consistency_details.get("llm_judge_used")
                else "rule_based: answer agreement + evidence/keyword + reasoning Jaccard"
            )
            reward_vectors[0]["reward_details"]["consistency_group"] = group_details

    return reward_vectors
