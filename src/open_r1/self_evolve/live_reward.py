"""Five-dimensional reward computed online during GRPO.

``self_evolve_refined_reward`` matches the trainer's reward-func contract and
returns the config-driven weighted sum over answer / grounding / process /
consistency / budget, with the auxiliary dims multiplied by the config's
outcome gate. Per-dimension logic lives in ``rewards``; the config comes from
``$SELF_EVOLVE_REWARD_CONFIG``. Grounding is a CLEVR metadata-box IoU proxy,
not open-world grounding. When context is missing a dimension falls back and
the fallback is recorded in the breakdown.
"""

from __future__ import annotations

import json
import os
import random
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from open_r1.self_evolve.rewards import (
    answer_reward,
    budget_reward,
    process_reward,
    structured_answer_reward,
    structured_exact_match_reward,
    field_consistency_reward,
    parse_structured_answer,
    extract_answer,
    is_format_valid,
)
from open_r1.self_evolve.reward_config import load_reward_config, RewardConfig

_REWARD_CONFIG_ENV = "SELF_EVOLVE_REWARD_CONFIG"

_REWARD_CONFIG: Optional[RewardConfig] = None
_config_lock = threading.Lock()

# Thread-local so concurrent reward evaluations don't clobber each other.
_last_breakdowns = threading.local()


def get_last_breakdowns() -> Optional[List[Dict[str, Any]]]:
    """Per-sample breakdowns from the last reward call, aligned 1:1 with its
    completions. ``None`` before the reward func has run on this thread."""
    return getattr(_last_breakdowns, "value", None)


def _get_reward_config() -> RewardConfig:
    """Load + cache the reward config."""
    global _REWARD_CONFIG
    if _REWARD_CONFIG is None:
        with _config_lock:
            if _REWARD_CONFIG is None:
                path = os.environ.get(_REWARD_CONFIG_ENV) or None
                _REWARD_CONFIG = load_reward_config(path)
    return _REWARD_CONFIG

_log_lock = threading.Lock()

# Set by the trainer each step so the grounding warmup can be applied.
_training_step = 0


def set_training_step(step: int) -> None:
    global _training_step
    _training_step = int(step)


def _completion_text(c: Any) -> str:
    """Trainer passes completions as [{'role','content'}] or plain strings."""
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c or "")


def _gold_answer(sol: Any) -> str:
    return extract_answer(sol if isinstance(sol, str) else str(sol or ""))


def _live_grounding(
    completion: str, se: Dict[str, Any], cfg: Optional[RewardConfig] = None
) -> tuple[float, Dict[str, Any]]:
    """Grounding from the emitted boxes + the gold boxes in ``se.grounding``.

    Returns ``(score, audit_fields)``. Credits and match mode come from the
    config; a missing or invalid box scores 0.
    """
    from open_r1.self_evolve.grounding_iou import score_model_bbox_grounding

    g = (se or {}).get("grounding") if isinstance(se, dict) else None
    g = g if isinstance(g, dict) else {}

    gold_boxes = g.get("gold_evidence_boxes")
    image_width = int(g.get("image_width") or 320)
    image_height = int(g.get("image_height") or 240)
    valid_player_ids = g.get("valid_player_ids")
    if not isinstance(valid_player_ids, list) or not valid_player_ids:
        valid_player_ids = None

    fields = score_model_bbox_grounding(
        completion,
        gold_boxes_pixel=gold_boxes if gold_boxes else None,
        image_width=image_width,
        image_height=image_height,
        valid_player_ids=valid_player_ids,
        config=cfg.grounding_cfg if cfg is not None else None,
    )
    score = float(fields["grounding_reward"])
    # grounding_source is an alias kept for existing log consumers.
    fields["grounding_source"] = fields["grounding_reward_source"]
    fields["fallback"] = fields["grounding_reward_source"] != "model_bbox_iou"
    return score, fields

def _group_consistency(answers: List[str]) -> float:
    """Fraction of the group sharing the modal answer (1.0 if singleton/empty)."""
    vals = [a for a in answers if a]
    if len(vals) <= 1:
        return 1.0
    counts = defaultdict(int)
    for a in vals:
        counts[a] += 1
    return max(counts.values()) / len(vals)


def self_evolve_refined_reward(completions, **kwargs) -> List[float]:
    """Live five-dimensional refined reward.

    Expected kwargs (provided by the trainer from dataset columns):
      - solution: list[str] gold answers (aligned with completions)
      - self_evolve: list[dict] per-task context (optional, for budget/grounding)
      - problem: list[str] (optional)
    """
    texts = [_completion_text(c) for c in completions]
    n = len(texts)
    cfg = _get_reward_config()
    solutions = kwargs.get("solution", [None] * n)
    se_blocks = kwargs.get("self_evolve", [None] * n)
    problems = kwargs.get("problem", [None] * n)

    # Group structure is only needed by the legacy group-modal consistency.
    groups: Dict[Any, List[int]] = defaultdict(list)
    group_answer: Dict[Any, float] = {}
    if cfg.consistency_cfg["mode"] == "group_modal":
        for i in range(n):
            key = (str(problems[i]) if i < len(problems) else "",
                   str(solutions[i]) if i < len(solutions) else "")
            groups[key].append(i)
        for key, idxs in groups.items():
            group_answer[key] = _group_consistency([extract_answer(texts[i]) for i in idxs])

    rewards: List[float] = []
    breakdowns: List[Dict[str, Any]] = []
    for i in range(n):
        comp = texts[i]
        sol = solutions[i] if i < len(solutions) else None
        se = se_blocks[i] if i < len(se_blocks) else None
        if isinstance(se, str):
            try:
                se = json.loads(se)
            except Exception:
                se = {}
        se = se if isinstance(se, dict) else {}
        prob = problems[i] if i < len(problems) else None

        gold = _gold_answer(sol)
        sol_text = sol if isinstance(sol, str) else f"<answer>{gold}</answer>"

        exact = answer_reward(comp, sol_text)
        answer_mode = cfg.answer_cfg["mode"]
        if answer_mode == "structured_fields":
            a, a_det = structured_answer_reward(
                comp, sol_text,
                fields=cfg.answer_cfg["fields"],
                off_by_one_credit=float(cfg.answer_cfg["off_by_one_credit"]),
                graded_fields=cfg.answer_cfg["graded_fields"],
            )
        elif answer_mode == "structured_exact_match":
            a, a_det = structured_exact_match_reward(comp, sol_text)
        else:
            a, a_det = exact, {"answer_mode": "exact_match", "answer_fields": {}}
        pred_fields = parse_structured_answer(comp)
        gold_fields = parse_structured_answer(sol_text)
        spy_correct = (
            None if gold_fields.get("spy") is None
            else pred_fields.get("spy") == gold_fields.get("spy")
        )

        # Reference block supplies budget / reference steps when present.
        ref_block = se.get("reference") if isinstance(se.get("reference"), dict) else {}
        task_meta = dict(se.get("task_metadata") or {})
        if ref_block.get("reasoning_budget_steps") is not None:
            task_meta.setdefault("reasoning_budget_steps", ref_block["reasoning_budget_steps"])
            task_meta["budget_source"] = ref_block.get("reference_source", "reference_block")
        b, b_det = budget_reward(
            comp,
            max_reasoning_words=int(cfg.budget_cfg["fallback_tokens"]),
            task_metadata=task_meta,
        )
        ref_steps = ref_block.get("reference_reasoning_steps") or se.get("reference_reasoning_steps")
        ref_reasoning = se.get("reference_reasoning")
        p, p_det = process_reward(
            comp,
            reference_reasoning=ref_reasoning,
            reference_reasoning_steps=ref_steps,
            task_metadata=task_meta,
        )
        g, g_det = _live_grounding(comp, se, cfg)
        warmup = int(cfg.grounding_cfg.get("warmup_steps", 0) or 0)
        grounding_active = _training_step >= warmup
        if not grounding_active:
            g = 0.0
        key = (str(problems[i]) if i < len(problems) else "",
               str(solutions[i]) if i < len(solutions) else "")
        if cfg.consistency_cfg["mode"] == "field_consistency":
            c, c_det = field_consistency_reward(
                comp,
                boxes=g_det.get("predicted_bbox_norm") or [],
                players=g_det.get("predicted_players") or [],
            )
            consistency_fallback = False
        else:
            c = group_answer.get(key, 0.5)
            c_det = {"consistency_mode": "group_modal"}
            consistency_fallback = len(groups.get(key, [])) <= 1
        f = is_format_valid(comp)

        reward_vector = {
            "answer": a,
            "grounding": g,
            "process": p,
            "consistency": c,
            "budget": b,
        }
        # The only training scalar: weighted sum after the outcome gate.
        total = cfg.scalarize(
            reward_vector, spy_correct=spy_correct, answer_correct=exact >= 1.0
        )
        # Control exps: replace the scalar with a signal that says nothing about
        # correctness. Only for the attribution baselines, never for a real run.
        if cfg.control_mode == "random":
            total = random.random()
        elif cfg.control_mode == "format_only":
            total = float(is_format_valid(comp))
        gate = cfg.gate_factor(spy_correct=spy_correct, answer_correct=exact >= 1.0)
        rewards.append(float(total))
        breakdowns.append({
            "answer": round(a, 4), "budget": round(b, 4), "process": round(p, 4),
            "grounding": round(g, 4), "consistency": round(c, 4),
            "format_valid": bool(f),
            "reward_scalar_used": round(float(total), 4),
            "reward_config_path": cfg.path,
            "reward_scalarization": cfg.scalarization,
            "reward_config_version": cfg.version,
            "reward_weights": dict(cfg.weights),
            # Collapse diagnostics: these move long before the aggregate reward does.
            "answer_exact_match": bool(exact >= 1.0),
            "answer_mode": a_det.get("answer_mode"),
            "answer_fields": a_det.get("answer_fields", {}),
            "pred_spy": pred_fields.get("spy"),
            "pred_changed_attributes": pred_fields.get("changed_attributes"),
            "gold_spy": gold_fields.get("spy"),
            "gold_changed_attributes": gold_fields.get("changed_attributes"),
            "spy_correct": spy_correct,
            "num_pred_boxes": len(g_det.get("predicted_bbox_norm") or []),
            "gate_factor": gate,
            "control_mode": cfg.control_mode,
            "grounding_active": grounding_active,
            "training_step": _training_step,
            "consistency_mode": c_det.get("consistency_mode"),
            "consistency_checks": c_det.get("consistency_checks", {}),
            # The model-<bbox> grounding audit fields, as scored.
            "grounding_source": g_det.get("grounding_source"),
            "grounding_reward_source": g_det.get("grounding_reward_source"),
            "bbox_present": bool(g_det.get("bbox_present", False)),
            "bbox_valid": bool(g_det.get("bbox_valid", False)),
            "bbox_invalid_reason": g_det.get("bbox_invalid_reason"),
            "bbox_player_id": g_det.get("bbox_player_id"),
            "bbox_player_id_valid": bool(g_det.get("bbox_player_id_valid", False)),
            "metadata_bbox_iou": g_det.get("metadata_bbox_iou"),
            "grounding_reward_components": g_det.get("grounding_reward_components"),
            "grounding_precision": g_det.get("grounding_precision"),
            "grounding_recall": g_det.get("grounding_recall"),
            "predicted_bbox_norm": g_det.get("predicted_bbox_norm"),
            "reference_bbox_norm": g_det.get("reference_bbox_norm"),
            "fallback_flags": {
                "grounding": bool(g_det.get("fallback")),
                "consistency": bool(consistency_fallback),
                "budget": b_det.get("budget_source") == "fallback_default",
            },
        })

    # Expose the full per-sample breakdowns so the GRPO trainer can pair each
    # rollout's five-dim score with its trajectory text for the per-step dump.
    _last_breakdowns.value = breakdowns
    _maybe_log(breakdowns)
    return rewards


def _maybe_log(breakdowns: List[Dict[str, Any]]) -> None:
    """Append a small reward-breakdown sample to LIVE_REWARD_LOG if set."""
    path = os.getenv("LIVE_REWARD_LOG")
    if not path or not breakdowns:
        return
    try:
        with _log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": breakdowns[: min(4, len(breakdowns))]}, ensure_ascii=False) + "\n")
    except Exception:
        pass
