#!/usr/bin/env python
"""
Closed-loop self-evolving VLM driver (real-input, multi-iteration).

This is the loop driver the mentor asked for: it runs N closed-loop iterations
where iteration N writes state and iteration N+1 reads it and changes behaviour.

    Image Pool
      -> Generator (policy-controlled, verifiable CLEVR tasks)
      -> Reference VLM (accept / reject; rejected -> regenerate)
      -> Solver rollout (real | dry-run | replay-fallback; source tagged)
      -> Five-dim reward + failure tags + buffers
      -> Training exports (SFT replay / refined GRPO)
      -> Trainer invocation (SFT→GRPO or GRPO-only)
      -> solver_update_state + generator_policy update
      -> save iteration_state

Iteration N+1 reads iteration N's:
    generator_policy.json, failure_profile.json, reference_feedback.jsonl,
    solver_update_state.json (-> solver_model_path).

HARD GUARD: --num-iterations > 2 is rejected unless
--allow-more-than-two-iterations is given (do NOT use it this round).

MANDATORY pathways: API-based Reference VLM and GRPO; SFT is recipe-controlled.
When an enabled pathway cannot execute, the run records the blocking reason and
fails rather than silently degrading to another recipe.

This driver does NOT fabricate solver performance, does NOT pretend dry-run is
real training, and does NOT print the API key. Real API calls happen only when
--enable-openai-reference-vlm is set without --dry-run-reference-vlm and a key
is present in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from open_r1.self_evolve.io import read_jsonl, write_jsonl  # noqa: E402
from open_r1.self_evolve.iteration_state import (  # noqa: E402
    IterationState,
    build_failure_profile_from_counts,
    default_generator_policy,
    default_training_pathways,
    diff_generator_policy,
    load_failure_profile,
    load_generator_policy,
    load_solver_update_state,
    reference_feedback_path,
    save_failure_profile,
    save_generator_policy,
    save_solver_update_state,
    update_generator_policy,
)
from open_r1.self_evolve.policy_clevr_generator import (  # noqa: E402
    PolicyCLEVRGeneratorConfig,
    PolicyControlledCLEVRGenerator,
)
from open_r1.self_evolve.iteration import score_and_route_trajectories, split_buffers  # noqa: E402
from open_r1.self_evolve.reward_config import load_reward_config  # noqa: E402
from open_r1.self_evolve.regret import counterfactual_sensitivity, summarize, task_statistics  # noqa: E402
from open_r1.self_evolve.proposer import ProposerConfig  # noqa: E402
from open_r1.self_evolve.reference_judge import (  # noqa: E402
    TOO_HARD_REJECT_REASON,
    decide_accept,
    is_too_hard_for_target,
    suggested_difficulty_on_too_hard,
)
import math  # noqa: E402
import random  # noqa: E402
from open_r1.self_evolve.experiment_config import (  # noqa: E402
    HARD_DEFAULTS,
    resolve_settings,
)
from open_r1.self_evolve.exporters import write_training_exports  # noqa: E402
from open_r1.self_evolve.failure_tags import assign_failure_tags  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

REWARD_KEYS = ["answer", "budget", "process", "grounding", "consistency"]


# ===========================================================================
# Reference VLM stage
# ===========================================================================

def _dry_run_reference_judgment(task: Dict[str, Any]) -> Dict[str, Any]:
    """Structured Reference-VLM judgment WITHOUT any API call.

    Rejects tasks that are over-hard for the difficulty target or visually
    ambiguous, so the regenerate path is genuinely exercised in dry-run. This
    is a stand-in for the API judge; the real OpenRouter GPT-4o judge produces
    the same schema (see reference_vlm/run_openai_reference_vlm_judge.py).
    """
    est = task.get("difficulty_estimate", "medium")
    target = task.get("difficulty_target", "medium")
    diff_meta = (task.get("metadata") or {}).get("difficulty_metadata", {})
    num_attr = int(diff_meta.get("num_attr_changes", 3))

    accepted = True
    reject_reason = ""
    # Reject if a task is at least the configured gap above the target band
    # (shared by every judge path; env-tunable, default >= 2).
    if is_too_hard_for_target(est, target):
        accepted = False
        reject_reason = TOO_HARD_REJECT_REASON
    # Reject visually over-saturated changes as ambiguous.
    elif num_attr >= 6:
        accepted = False
        reject_reason = "high_visual_ambiguity"

    solvability = 0.9 if accepted else 0.4
    grounding = 0.85 if num_attr >= 3 else 0.6
    # When rejected as too hard, suggest a strictly EASIER difficulty so the
    # next-round policy update can actually reduce hard.
    # Otherwise echo the task's own target (NOT a reduce signal).
    too_hard = reject_reason == TOO_HARD_REJECT_REASON
    suggested_difficulty = (
        suggested_difficulty_on_too_hard(target) if too_hard else target
    )
    return {
        "task_id": task.get("task_id"),
        "accepted": accepted,
        "reject_reason": reject_reason,
        "difficulty_level": est,
        "difficulty_target": target,
        "difficulty_score": float(diff_meta.get("difficulty_score", 3.0)),
        "solvability_score": solvability,
        "visual_grounding_score": grounding,
        "ambiguity_score": 0.7 if num_attr >= 6 else 0.2,
        "reasoning_budget_steps": 3 + num_attr // 2,
        "reasoning_budget_tokens": 120 + 20 * (num_attr // 2),
        "required_visual_evidence": ["changed_object_attributes", "player_image_comparison"],
        "reference_reasoning_steps": [],
        "reference_feedback": (
            "dry-run reference judge: "
            + ("accepted" if accepted else f"rejected ({reject_reason})")
        ),
        "generator_feedback": {
            "avoid_scene_id": None if accepted else task.get("scene_id"),
            "suggested_difficulty": suggested_difficulty,
            "reason": "reduce_difficulty" if too_hard else "",
        },
        "reference_model": "dry_run_heuristic",
        "reference_provider": "dry_run",
        "reference_source": "dry_run_reference_judge",
        "image_evidence": task.get("image_path", []),
    }


def _passthrough_reference_judgment(task: Dict[str, Any]) -> Dict[str, Any]:
    """Judgment for a candidate that exceeded the Reference VLM budget cap.

    The cap limits how many candidates the (paid) Reference VLM judges — it must
    NOT silently drop the rest. Such tasks pass through into the accepted pool so
    the solver still trains on the full ``num_train_tasks``, but they are clearly
    marked as NOT judged: ``reference_judged=False``,
    ``reference_source="unjudged_budget_passthrough"``, and NO fabricated
    ``generator_feedback`` (None). Difficulty is the generator's metadata proxy;
    budget/reasoning fields use stable fallbacks so the trainer/exporter type
    contract (``_sanitize_reference_for_training``) never sees mixed types.
    """
    est = str(task.get("difficulty_estimate", "medium"))
    target = str(task.get("difficulty_target", "medium"))
    diff_meta = (task.get("metadata") or {}).get("difficulty_metadata", {})
    num_attr = int(diff_meta.get("num_attr_changes", 3))
    return {
        "task_id": task.get("task_id"),
        "accepted": True,
        "reference_judged": False,
        "reject_reason": "",
        "difficulty_level": est,
        "difficulty_target": target,
        "difficulty_score": float(diff_meta.get("difficulty_score", 3.0)),
        "solvability_score": 1.0,
        "visual_grounding_score": 0.7,
        "ambiguity_score": 0.2,
        "ambiguity_flag": False,
        # Stable fallbacks (same shape the dry-run/live judges emit) so exporters
        # and the live GRPO reward read type-consistent values.
        "reasoning_budget_steps": 3 + num_attr // 2,
        "reasoning_budget_tokens": 120 + 20 * (num_attr // 2),
        "required_visual_evidence": ["changed_object_attributes", "player_image_comparison"],
        "reference_reasoning_steps": [],
        "reference_feedback": "unjudged: exceeded Reference VLM budget cap (passthrough)",
        # Explicitly no fabricated generator feedback — this task was not judged.
        "generator_feedback": None,
        "reference_model": "none",
        "reference_provider": "none",
        "reference_source": "unjudged_budget_passthrough",
        "image_evidence": task.get("image_path", []),
    }


def _live_reference_judgment(
    task: Dict[str, Any],
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    dataset_root: Optional[str],
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call the real OpenRouter/OpenAI vision Reference VLM for one task.

    Reuses the runner module's request/parse helpers so there is a single
    request implementation. Never logs the API key or Authorization header.
    """
    import importlib.util

    runner_path = (
        REPO_ROOT / "local_scripts" / "self_evolve" / "reference_vlm"
        / "run_openai_reference_vlm_judge.py"
    )
    spec = importlib.util.spec_from_file_location("scv_ref_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)  # type: ignore

    ds_root = Path(dataset_root) if dataset_root else None
    image_paths = runner._resolve_image_paths(task, ds_root)
    judgment = runner._call_gpt4o(
        task, image_paths, model, api_key, base_url, max_retries, provider
    )
    judgment["reference_provider"] = provider
    judgment["reference_model"] = model
    judgment.setdefault("reference_source", f"{provider}_reference_vlm")
    return judgment


def _resolve_reference_api_key(dry_run: bool, provider: str) -> Tuple[Optional[str], str]:
    """Resolve the API key + mode label for a reference judging run."""
    if dry_run:
        return None, "dry_run"
    # REFERENCE_VLM_API_KEY is the dedicated credential so the Reference VLM can
    # use a different key from the answer judge; the shared provider key stays
    # as the fallback for callers that only configure that.
    if provider == "openrouter":
        api_key = (os.environ.get("REFERENCE_VLM_API_KEY")
                   or os.environ.get("OPENROUTER_API_KEY"))
        mode = "live_openrouter"
        key_name = "REFERENCE_VLM_API_KEY or OPENROUTER_API_KEY"
    else:
        api_key = (os.environ.get("REFERENCE_VLM_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
        mode = "live_openai"
        key_name = "REFERENCE_VLM_API_KEY or OPENAI_API_KEY"
    if not api_key:
        raise SystemExit(
            f"Reference VLM live mode requires an API key for provider={provider}. "
            f"Set {key_name}."
        )
    return api_key, mode


def _gate_reference_judgment(
    judgment: Dict[str, Any],
    task: Dict[str, Any],
    *,
    solvability_threshold: float,
    reject_on_ambiguity: bool,
) -> Dict[str, Any]:
    """Apply the shared decide_accept gate to a raw judgment (live or dry-run).

    The raw judgment supplies the difficulty/solvability/ambiguity SIGNALS; the
    final accept/reject is decided here so every path funnels through the single
    source of truth. ``generator_feedback`` is rebuilt to stay consistent with
    the gated decision (a genuine reduce signal only on a too_hard reject).
    """
    target = str(
        judgment.get("difficulty_target")
        or task.get("difficulty_target")
        or "medium"
    )
    accepted, reject_reason = decide_accept(
        {
            "difficulty_level": judgment.get("difficulty_level", "medium"),
            "solvability_score": judgment.get("solvability_score", 1.0),
            "ambiguity_score": judgment.get("ambiguity_score", 0.0),
        },
        target,
        solvability_threshold=solvability_threshold,
        reject_on_ambiguity=reject_on_ambiguity,
    )
    judgment["accepted"] = accepted
    judgment["reject_reason"] = "" if accepted else reject_reason
    judgment["reference_judged"] = True

    too_hard = reject_reason == TOO_HARD_REJECT_REASON
    judgment["generator_feedback"] = {
        "avoid_scene_id": None if accepted else task.get("scene_id"),
        "suggested_difficulty": (
            suggested_difficulty_on_too_hard(target) if too_hard else target
        ),
        "reject_reason": "" if accepted else reject_reason,
        "reason": "reduce_difficulty" if too_hard else "",
    }
    return judgment


def _make_feedback_record(
    task: Dict[str, Any], judgment: Dict[str, Any], attempt: int
) -> Dict[str, Any]:
    """Build the per-task reference_feedback row (shared by both stage paths)."""
    return {
        "task_id": task.get("task_id"),
        "scene_id": task.get("scene_id"),
        "attempt": attempt,
        "accepted": judgment.get("accepted"),
        "reference_judged": True,
        "reject_reason": judgment.get("reject_reason"),
        "reference_provider": judgment.get("reference_provider"),
        "reference_model": judgment.get("reference_model"),
        "generator_feedback": judgment.get("generator_feedback"),
        "difficulty_level": judgment.get("difficulty_level"),
        "difficulty_target": (
            judgment.get("difficulty_target") or task.get("difficulty_target")
        ),
    }


def run_reference_stage(
    first_batch: List[Dict[str, Any]],
    *,
    dry_run: bool,
    provider: str,
    model: str,
    base_url: str,
    dataset_root: Optional[str],
    quota_target: int,
    judge_budget_factor: float,
    solvability_threshold: float,
    reject_on_ambiguity: bool,
    generator: Optional[PolicyControlledCLEVRGenerator],
    generator_policy: Dict[str, Any],
    failure_profile: Optional[Dict[str, Any]],
    max_regenerate_attempts: int,
    stream_dir: Optional[str] = None,
    max_workers: int = 16,
    progress_every: int = 16,
    log: Any,
) -> Dict[str, Any]:
    """Judge-until-quota Reference VLM stage.

    Judges the oversampled ``first_batch``; regenerates (excluding all seen
    scenes) until ``quota_target`` candidates are accepted, the judge-call
    budget ``ceil(judge_budget_factor * quota_target)`` is exhausted, or the
    generator runs dry. There is NO budget-cap passthrough: every candidate that
    enters the accepted pool was actually judged. When the quota cannot be met
    the shortfall is reported honestly (WARNING) — never padded.
    """
    api_key, mode = _resolve_reference_api_key(dry_run, provider)
    judge_budget = math.ceil(judge_budget_factor * quota_target)

    def judge(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            raw = (
                _dry_run_reference_judgment(task) if dry_run
                else _live_reference_judgment(
                    task, provider, model, base_url, api_key, dataset_root
                )
            )
        except Exception as exc:  # noqa: BLE001 — never crash the whole run on one task
            log(f"  [WARN] Reference VLM call failed after retries for "
                f"task={task.get('task_id')}: {type(exc).__name__}: {str(exc)[:200]} "
                f"— skipping this task (marked reference_api_error).")
            return None
        return _gate_reference_judgment(
            raw, task,
            solvability_threshold=solvability_threshold,
            reject_on_ambiguity=reject_on_ambiguity,
        )

    accepted: List[Dict[str, Any]] = []
    feedback: List[Dict[str, Any]] = []
    seen_scene_ids: set = set()
    rejected_scene_ids: set = set()
    reject_reason_counter: Dict[str, int] = {}
    num_judged = 0
    num_api_calls = 0
    num_regenerated = 0
    budget_exhausted = False
    generator_exhausted = False

    # Streaming sidecar writers: append each judgment the moment it lands so the
    # stage is tailable and crash-resumable-by-inspection. These are ".partial"
    # files, NOT the canonical accepted_tasks.jsonl / reference_feedback.jsonl —
    # the call site writes those atomically at end-of-stage, so a crash here
    # never leaves a half-written canonical file that the resume gate would
    # mistake for a completed stage.
    _acc_fh = None
    _fb_fh = None
    if stream_dir:
        os.makedirs(stream_dir, exist_ok=True)
        _acc_fh = open(os.path.join(stream_dir, "accepted_tasks.partial.jsonl"), "w")
        _fb_fh = open(os.path.join(stream_dir, "reference_feedback.partial.jsonl"), "w")

    def _stream(fh, record: Dict[str, Any]) -> None:
        if fh is None:
            return
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()

    # I/O-bound judge (network round-trip per task) → thread pool. The GIL is
    # released across the request, so threads overlap the waits. Bookkeeping
    # (accept/reject, counters, streaming) stays on this thread, applied to each
    # wave's results IN ORDER, so the sequence is deterministic and lock-free.
    effective_workers = max(1, int(max_workers))

    pending = list(first_batch)
    for t in pending:
        seen_scene_ids.add(t.get("scene_id"))
    attempt = 0

    from concurrent.futures import ThreadPoolExecutor

    def _apply_result(task: Dict[str, Any], judgment: Optional[Dict[str, Any]]) -> None:
        """Bookkeeping + streaming for one judged task. Called on THIS thread,
        in submission order, so counters/lists stay deterministic and lock-free."""
        if judgment is None:
            # Reference VLM API failed after retries — skip this task, keep
            # training running. Bookkeeping mirrors a rejected task so the
            # scene isn't re-proposed in this iteration.
            rejected_scene_ids.add(task.get("scene_id"))
            reject_reason_counter["reference_api_error"] = (
                reject_reason_counter.get("reference_api_error", 0) + 1
            )
            stub_judgment = {
                "accepted": False,
                "reject_reason": "reference_api_error",
                "reference_provider": provider,
                "reference_model": model,
                "reference_source": f"{provider}_reference_vlm",
            }
            fb = _make_feedback_record(task, stub_judgment, attempt)
            fb["status"] = "reference_api_error"
            feedback.append(fb)
            _stream(_fb_fh, fb)
            return
        record = dict(task)
        record["reference_judge"] = judgment
        fb = _make_feedback_record(task, judgment, attempt)
        feedback.append(fb)
        if judgment.get("accepted"):
            accepted.append(record)
            _stream(_acc_fh, record)
        else:
            rejected_scene_ids.add(task.get("scene_id"))
            rr = judgment.get("reject_reason") or "unspecified"
            reject_reason_counter[rr] = reject_reason_counter.get(rr, 0) + 1
            fb["status"] = "reference_rejected"
        _stream(_fb_fh, fb)

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            while True:
                # Judge `pending` in waves of `effective_workers`. Between waves
                # re-check quota/budget; a wave is clamped to the remaining
                # budget so it stays a hard cap (we may judge up to wave_size-1
                # extra past the exact quota hit — harmless under oversampling).
                idx = 0
                while idx < len(pending):
                    if len(accepted) >= quota_target:
                        break
                    room = judge_budget - num_judged
                    if room <= 0:
                        budget_exhausted = True
                        break
                    wave = pending[idx: idx + min(effective_workers, room)]
                    idx += len(wave)
                    # Concurrent network round-trips; results collected in order.
                    judgments = list(pool.map(judge, wave))
                    for task, judgment in zip(wave, judgments):
                        num_judged += 1
                        num_api_calls += 0 if dry_run else 1
                        _apply_result(task, judgment)
                    log(f"  Reference VLM: judged {num_judged}/{judge_budget}, "
                        f"accepted {len(accepted)}/{quota_target} "
                        f"(x{effective_workers} concurrent).")

                if len(accepted) >= quota_target or budget_exhausted:
                    break
                if generator is None or attempt >= max_regenerate_attempts:
                    break

                attempt += 1
                remaining_budget = judge_budget - num_judged
                if remaining_budget <= 0:
                    budget_exhausted = True
                    break
                regenerated = generator.generate(
                    generator_policy, failure_profile, exclude_scene_ids=seen_scene_ids
                )
                regenerated = [
                    t for t in regenerated if t.get("scene_id") not in seen_scene_ids
                ]
                if not regenerated:
                    generator_exhausted = True
                    log(f"  Regenerate attempt {attempt}: generator exhausted "
                        f"(no unseen scenes left); stopping short of quota.")
                    break
                for t in regenerated:
                    seen_scene_ids.add(t.get("scene_id"))
                num_regenerated += len(regenerated)
                log(f"  Regenerate attempt {attempt}: {len(regenerated)} new candidate(s) "
                    f"(accepted={len(accepted)}/{quota_target}, judged={num_judged}/{judge_budget}).")
                pending = regenerated
    finally:
        for _fh in (_acc_fh, _fb_fh):
            if _fh is not None:
                try:
                    _fh.close()
                except Exception:
                    pass

    quota_shortfall = max(0, quota_target - len(accepted))
    if quota_shortfall > 0:
        log(f"  [WARN] Reference VLM quota shortfall: accepted={len(accepted)} < "
            f"target={quota_target} (budget_exhausted={budget_exhausted}, "
            f"generator_exhausted={generator_exhausted}). Training on fewer tasks; "
            f"NOT padded.")

    reference_accounting = {
        "quota_target": quota_target,
        "accepted": len(accepted),
        "judged": num_judged,
        "judge_budget": judge_budget,
        "reject_count": sum(reject_reason_counter.values()),
        "budget_exhausted": budget_exhausted,
        "generator_exhausted": generator_exhausted,
        "quota_shortfall": quota_shortfall,
        "num_regenerated": num_regenerated,
        "passthrough_count": 0,
        "reject_reason_histogram": dict(reject_reason_counter),
    }

    return {
        "accepted_tasks": accepted,
        "reference_feedback": feedback,
        "mode": mode,
        "num_judged": len(feedback),
        "num_accepted": len(accepted),
        "num_rejected": sum(1 for f in feedback if f.get("accepted") is False),
        "num_regenerated": num_regenerated,
        "num_rejected_final": reference_accounting["reject_count"],
        "num_api_calls": num_api_calls,
        # Audit keys read by the call-site log line and downstream readers,
        # alongside the quota accounting.
        "candidate_tasks_count": len(feedback),
        "reference_judged_tasks_count": num_judged,
        "unjudged_passthrough_tasks_count": 0,
        "accepted_tasks_count": len(accepted),
        "reference_judge_budget_cap": judge_budget,
        "dropped_candidate_count": reference_accounting["reject_count"],
        "drop_reason_histogram": dict(reject_reason_counter),
        "reference_accounting": reference_accounting,
    }


def _run_reference_stage_legacy(
    candidate_tasks: List[Dict[str, Any]],
    *,
    dry_run: bool,
    provider: str,
    model: str,
    base_url: str,
    dataset_root: Optional[str],
    max_tasks: Optional[int],
    generator: Optional[PolicyControlledCLEVRGenerator],
    generator_policy: Dict[str, Any],
    failure_profile: Optional[Dict[str, Any]],
    max_regenerate_attempts: int,
    log: Any,
) -> Dict[str, Any]:
    """DEPRECATED budget-cap passthrough path (--disable-reference-quota only).

    Judges the first ``max_tasks`` candidates and passes the rest through
    unjudged into the accepted pool. Retained for ablation; the default path is
    the judge-until-quota ``run_reference_stage`` above.
    """
    api_key = None
    mode = "dry_run"
    if not dry_run:
        if provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            mode = "live_openrouter"
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            mode = "live_openai"
        if not api_key:
            raise SystemExit(
                f"Reference VLM live mode requires an API key for provider={provider}. "
                f"Set {'OPENROUTER_API_KEY' if provider == 'openrouter' else 'OPENAI_API_KEY'}."
            )

    def judge(task: Dict[str, Any]) -> Dict[str, Any]:
        judgment = (
            _dry_run_reference_judgment(task) if dry_run
            else _live_reference_judgment(
                task, provider, model, base_url, api_key, dataset_root
            )
        )
        # Mark every actually-judged record so the judgment dict itself is
        # symmetric with the unjudged passthrough (which sets False). Both the
        # feedback record and the embedded reference_judge dict now agree.
        judgment["reference_judged"] = True
        return judgment

    # The budget cap bounds how many candidates the (paid) Reference VLM judges
    # — it is a JUDGE BUDGET, not a training-set cap. Candidates beyond the cap
    # are NOT dropped: they pass through into the accepted pool (clearly marked
    # unjudged) so the solver still trains on the full num_train_tasks.
    judge_budget_cap = max_tasks
    judged_tasks = candidate_tasks
    passthrough_tasks: List[Dict[str, Any]] = []
    if judge_budget_cap is not None and len(candidate_tasks) > judge_budget_cap:
        judged_tasks = candidate_tasks[:judge_budget_cap]
        passthrough_tasks = candidate_tasks[judge_budget_cap:]
        log(f"  Reference VLM budget cap={judge_budget_cap}: judging first "
            f"{judge_budget_cap} of {len(candidate_tasks)} candidates; "
            f"{len(passthrough_tasks)} pass through unjudged into the accepted pool.")

    accepted: List[Dict[str, Any]] = []
    feedback: List[Dict[str, Any]] = []
    rejected_scene_ids: set = set()
    num_regenerated = 0
    num_rejected_final = 0
    num_api_calls = 0
    num_reference_judged = 0
    drop_reason_counter: Dict[str, int] = {}

    pending = list(judged_tasks)
    attempt = 0
    while pending:
        next_pending: List[Dict[str, Any]] = []
        for task in pending:
            judgment = judge(task)
            num_api_calls += 0 if dry_run else 1
            num_reference_judged += 1
            record = dict(task)
            record["reference_judge"] = judgment
            feedback.append({
                "task_id": task.get("task_id"),
                "scene_id": task.get("scene_id"),
                "attempt": attempt,
                "accepted": judgment.get("accepted"),
                "reference_judged": True,
                "reject_reason": judgment.get("reject_reason"),
                "reference_provider": judgment.get("reference_provider"),
                "reference_model": judgment.get("reference_model"),
                "generator_feedback": judgment.get("generator_feedback"),
                "difficulty_level": judgment.get("difficulty_level"),
                # The task's own difficulty target — needed by the next-round
                # summary to tell a genuine "reduce" suggestion (suggested <
                # target) apart from a target echo (suggested == target).
                "difficulty_target": (
                    judgment.get("difficulty_target")
                    or task.get("difficulty_target")
                ),
            })
            if judgment.get("accepted"):
                accepted.append(record)
            else:
                rejected_scene_ids.add(task.get("scene_id"))
                if attempt < max_regenerate_attempts and generator is not None:
                    next_pending.append(task)  # placeholder; regenerated below
                else:
                    num_rejected_final += 1
                    feedback[-1]["status"] = "reference_rejected_final"
                    rr = judgment.get("reject_reason") or "unspecified"
                    drop_reason_counter[rr] = drop_reason_counter.get(rr, 0) + 1

        if not next_pending:
            break
        attempt += 1
        # Regenerate replacement tasks, excluding all rejected scenes so far.
        n_replace = len(next_pending)
        regenerated = generator.generate(
            generator_policy, failure_profile, exclude_scene_ids=rejected_scene_ids
        )[:n_replace]
        num_regenerated += len(regenerated)
        log(f"  Regenerate attempt {attempt}: {n_replace} rejected -> "
            f"{len(regenerated)} new candidate(s).")
        pending = regenerated
        if not regenerated:
            # Generator exhausted; mark the remaining as final rejects.
            num_rejected_final += n_replace
            drop_reason_counter["regenerator_exhausted"] = (
                drop_reason_counter.get("regenerator_exhausted", 0) + n_replace
            )
            break

    # Unjudged passthrough: over-budget candidates enter the accepted pool with a
    # non-judged judgment, so they are never silently dropped.
    num_passthrough = 0
    for task in passthrough_tasks:
        judgment = _passthrough_reference_judgment(task)
        record = dict(task)
        record["reference_judge"] = judgment
        accepted.append(record)
        num_passthrough += 1
        feedback.append({
            "task_id": task.get("task_id"),
            "scene_id": task.get("scene_id"),
            "attempt": 0,
            "accepted": True,
            "reference_judged": False,
            "reject_reason": "",
            "reference_provider": "none",
            "reference_model": "none",
            "generator_feedback": None,
            "difficulty_level": judgment.get("difficulty_level"),
            "difficulty_target": judgment.get("difficulty_target"),
            "status": "unjudged_budget_passthrough",
        })

    candidate_count = len(candidate_tasks)
    accepted_count = len(accepted)
    dropped_count = num_rejected_final
    return {
        "accepted_tasks": accepted,
        "reference_feedback": feedback,
        "mode": mode,
        "num_judged": len(feedback),
        "num_accepted": accepted_count,
        "num_rejected": sum(1 for f in feedback if f.get("accepted") is False),
        "num_regenerated": num_regenerated,
        "num_rejected_final": num_rejected_final,
        "num_api_calls": num_api_calls,
        # Audit counts: keep judge budget and training-set
        # size distinct and fully accountable.
        "candidate_tasks_count": candidate_count,
        "reference_judged_tasks_count": num_reference_judged,
        "unjudged_passthrough_tasks_count": num_passthrough,
        "accepted_tasks_count": accepted_count,
        "reference_judge_budget_cap": judge_budget_cap,
        "dropped_candidate_count": dropped_count,
        "drop_reason_histogram": dict(drop_reason_counter),
        # Quota-mode accounting shape, filled with legacy-equivalent values so the
        # call site can read ref["reference_accounting"] regardless of path.
        "reference_accounting": {
            "quota_target": None,
            "accepted": accepted_count,
            "judged": num_reference_judged,
            "judge_budget": judge_budget_cap,
            "reject_count": dropped_count,
            "budget_exhausted": False,
            "generator_exhausted": False,
            "quota_shortfall": 0,
            "num_regenerated": num_regenerated,
            "passthrough_count": num_passthrough,
            "reject_reason_histogram": dict(drop_reason_counter),
        },
    }


# ===========================================================================
# Solver rollout stage
# ===========================================================================

def run_solver_stage(
    accepted_tasks: List[Dict[str, Any]],
    *,
    dry_run: bool,
    num_generations: int,
    solver_model_path: Optional[str],
    base_model_path: Optional[str] = None,
    num_gpus: int = 1,
    scratch_dir: Optional[str] = None,
    solver_max_new_tokens: int = 1024,
    solver_temperature: float = 1.0,
    solver_top_p: float = 0.95,
    solver_min_pixels: Optional[int] = None,
    solver_max_pixels: Optional[int] = None,
    log: Any,
) -> Dict[str, Any]:
    """Produce solver trajectories.

    Path selection (each trajectory is tagged with ``rollout_source``):
      - real_solver:    online VLM rollout (requires GPU; not run here).
      - dry_run_solver: prompt is built, NO model is loaded; a clearly-flagged
                        diagnostic completion is emitted to exercise the reward /
                        buffer / export machinery. NOT a real solve.
    Replay fallback (reading pre-existing trajectories) is available via the
    OfflineTrajectorySampler but is not the dry-run default; it would be tagged
    ``replay_fallback`` if used.
    """
    if not dry_run:
        # Real rollout path: the online vLLM sampler.
        # This loads a local Qwen2.5-VL model and generates real trajectories on
        # GPU. Gated by NOT passing --dry-run-solver; the sampler itself raises a
        # clear error if vllm/transformers/qwen_vl_utils or the model are unavailable.
        from open_r1.self_evolve.online_solver import (
            OnlineVLLMSolverSampler,
            OnlineSolveConfig,
            generate_rollouts_multi_gpu,
        )
        from tqdm.auto import tqdm
        # Base solver defaults to the model path passed on the CLI (--model-path),
        # NOT a hardcoded location. iteration 0 has no carried-forward checkpoint,
        # so it MUST fall back to this base model; a missing base_model_path is a
        # hard error rather than a silent wrong-path load.
        base_solver = base_model_path
        if not base_solver:
            raise SystemExit(
                "run_solver_stage: real rollout requires base_model_path "
                "(pass --model-path). iteration 0 has no solver checkpoint to "
                "carry forward and there is no hardcoded fallback."
            )
        # Carry the latest trained checkpoint forward when available; otherwise
        # fall back to the base model. Record the effective path on every
        # trajectory so the rollout source is auditable.
        effective_solver_path = solver_model_path or base_solver

        # --- Multi-GPU data-parallel path (default when num_gpus > 1) ---
        # One full model copy per card, tasks sharded round-robin. ~num_gpus
        # speedup vs the old single-process device_map="auto" model-parallel path.
        if num_gpus and num_gpus > 1 and len(accepted_tasks) > 1:
            log(f"  Solver: REAL online rollout (data-parallel, {num_gpus} GPUs) "
                f"from {effective_solver_path} [rollout_source=real_solver].")
            _scratch = scratch_dir or "solver_rollout_shards"
            trajectories = generate_rollouts_multi_gpu(
                accepted_tasks,
                num_generations=num_generations,
                effective_solver_path=effective_solver_path,
                num_gpus=num_gpus,
                scratch_dir=_scratch,
                max_new_tokens=solver_max_new_tokens,
                temperature=solver_temperature,
                top_p=solver_top_p,
                min_pixels=solver_min_pixels,
                max_pixels=solver_max_pixels,
                log=log,
            )
            log(f"  Solver: {len(trajectories)} REAL trajectories generated "
                f"(solver_model_path={effective_solver_path}, "
                f"data_parallel_gpus={num_gpus}).")
            return {
                "trajectories": trajectories,
                "rollout_source": "real_solver",
                "solver_model_path": effective_solver_path,
            }

        # --- Single-GPU / single-process fallback ---
        log(f"  Solver: REAL online rollout (single process) from "
            f"{effective_solver_path} [rollout_source=real_solver].")
        sampler = OnlineVLLMSolverSampler(
            OnlineSolveConfig(
                model_path=effective_solver_path,
                max_images=max((len(t.get("image_path") or []) for t in accepted_tasks), default=1),
                dry_run=False,
                max_new_tokens=solver_max_new_tokens,
                temperature=solver_temperature,
                top_p=solver_top_p,
                min_pixels=solver_min_pixels,
                max_pixels=solver_max_pixels,
            )
        )
        trajectories: List[Dict[str, Any]] = []
        total_rollouts = len(accepted_tasks) * num_generations
        pbar = tqdm(
            total=total_rollouts,
            desc="solver rollout",
            unit="traj",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for task in accepted_tasks:
            img_paths = task.get("image_path") or []
            for g in range(num_generations):
                traj = sampler.generate_one(
                    task_id=str(task["task_id"]),
                    image_paths=list(img_paths),
                    prompt_text=str(task.get("prompt") or task.get("problem") or ""),
                    metadata={**task.get("metadata", {}), "self_evolve_task": task},
                    gen_index=g,
                    diagnostic_only=False,
                    rollout_source="real_solver",
                )
                traj["problem"] = task.get("problem")
                traj["prompt"] = task.get("prompt") or task.get("problem")
                traj["solution"] = task.get("solution")
                traj["ground_truth"] = task.get("ground_truth")
                traj["scene_id"] = task.get("scene_id")
                traj["base_task_id"] = task.get("base_task_id")
                traj["reference_reasoning"] = task.get("reference_reasoning")
                traj["rollout_source"] = "real_solver"
                traj["solver_model_path"] = effective_solver_path
                trajectories.append(traj)
                pbar.update(1)
        pbar.close()
        log(f"  Solver: {len(trajectories)} REAL trajectories generated "
            f"(solver_model_path={effective_solver_path}).")
        # Release the rollout model BEFORE any trainer subprocess runs — the loop
        # shares a single GPU between real_solver and the GRPO/SFT smoke.
        sampler.unload()
        return {
            "trajectories": trajectories,
            "rollout_source": "real_solver",
            "solver_model_path": effective_solver_path,
        }

    trajectories: List[Dict[str, Any]] = []
    for task in accepted_tasks:
        gold = task.get("ground_truth", "")
        # A normalized gold bbox derived from the task's first gold pixel box, so
        # the dry-run generation-0 trajectory exercises the model-bbox IoU path
        # (NOT a real solve — synthetic, clearly flagged).
        gold_bbox_norm = None
        gb = (task.get("gold_bbox") or
              (task.get("metadata", {}) or {}).get("gold_evidence_boxes"))
        if gb:
            x1, y1, x2, y2 = gb[0]
            gold_bbox_norm = [round(x1 / 320, 4), round(y1 / 240, 4),
                              round(x2 / 320, 4), round(y2 / 240, 4)]
        for g in range(num_generations):
            # Diagnostic dry-run completion: generation 0 mirrors the gold
            # answer (routes positive), others diverge (route failure). This is
            # explicitly NOT a real solve — only mechanism exercise.
            if g == 0:
                _spy_num = (task.get("metadata") or {}).get("spy_player", 1)
                bbox_tag = (
                    f"<bbox player=\"{_spy_num}\">[{gold_bbox_norm[0]},{gold_bbox_norm[1]},"
                    f"{gold_bbox_norm[2]},{gold_bbox_norm[3]}]</bbox>"
                    if gold_bbox_norm else ""
                )
                completion = (
                    "<think>Compare the three player images, locate the object whose "
                    "color/shape/material differs from the others, and select that "
                    "player.</think>"
                    f"<answer>{gold}</answer>{bbox_tag}"
                )
                # Diagnostic grounding signal so this trajectory routes to the
                # positive buffer (exercises SFT replay). The
                # dry-run solver has no real boxes; this is explicitly synthetic.
                grounding_score = 0.9
            elif g == 1:
                # Correct answer but ungrounded shortcut → failure buffer.
                completion = (
                    "<think>I guess based on a quick glance.</think>"
                    f"<answer>{gold}</answer>"
                )
                grounding_score = 0.1
            else:
                # Wrong answer → failure buffer (adaptive sampling signal).
                wrong = "0"  # deliberately unparseable answer for the negative sample
                completion = (
                    "<think>The images look similar.</think>"
                    f"<answer>{wrong}</answer>"
                )
                grounding_score = 0.1
            trajectories.append({
                "task_id": task["task_id"],
                "base_task_id": task.get("base_task_id"),
                "trajectory_id": f"{task['task_id']}::dry_run_solver_{g}",
                "problem": task.get("problem"),
                "prompt": task.get("prompt"),
                "solution": task.get("solution"),
                "ground_truth": gold,
                "completion": completion,
                "grounding_score": grounding_score,
                "image": task.get("image"),
                "image_path": task.get("image_path"),
                "scene_id": task.get("scene_id"),
                "reference_reasoning": task.get("reference_reasoning"),
                "metadata": {**task.get("metadata", {}), "self_evolve_task": task},
                "rollout_source": "dry_run_solver",
                "diagnostic_only": True,
                "solver_model_path": solver_model_path,
            })
    log(f"  Solver: {len(trajectories)} trajectories from {len(accepted_tasks)} "
        f"accepted task(s) [rollout_source=dry_run_solver].")
    return {
        "trajectories": trajectories,
        "rollout_source": "dry_run_solver",
        "solver_model_path": solver_model_path,
    }


# ===========================================================================
# Trainer invocation stage
# ===========================================================================

class TrainerScale:
    """Multi-GPU / batch / generation knobs for the real trainer commands.

    GRPO computes the advantage WITHIN a group of ``grpo_num_generations``
    completions of the same prompt. A group of 2 makes the advantage estimate
    high-variance (the mentor's note), so the recommended real-training group
    size is >= 4, ideally 6. ``num_generations=2`` is permitted ONLY for an API
    wiring probe / dry-run and is flagged as not a valid GRPO learning config.

    The GRPO global batch ``per_device_train_batch_size * num_gpus`` must be
    divisible by ``grpo_num_generations`` (a TRL / VLMGRPOTrainer requirement —
    gradient_accumulation does NOT participate). ``validate()`` enforces this
    and fails fast.
    """

    # Mentor guidance: real GRPO group size must be >= this; 6 is ideal.
    RECOMMENDED_MIN_GROUP_SIZE = 4
    RECOMMENDED_GROUP_SIZE = 6

    def __init__(
        self,
        num_gpus: int = 1,
        per_device_train_batch_size: int = 6,
        gradient_accumulation_steps: int = 1,
        grpo_num_generations: int = 6,
        deepspeed_config: Optional[str] = None,
        trainer_backend: str = "deepspeed",
        fsdp_config: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        is_probe: bool = False,
    ) -> None:
        self.num_gpus = int(num_gpus)
        self.per_device_train_batch_size = int(per_device_train_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.grpo_num_generations = int(grpo_num_generations)
        self.deepspeed_config = deepspeed_config or None
        # Distributed backend for the GRPO route only ("deepspeed" | "fsdp2").
        # SFT always uses deepspeed regardless of this flag.
        self.trainer_backend = (trainer_backend or "deepspeed").lower()
        self.fsdp_config = fsdp_config or None
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        # Probe = API wiring / dry-run only; allows a sub-recommended group size
        # but the report must flag it as not a valid GRPO learning config.
        self.is_probe = bool(is_probe)

    @property
    def group_advantage_warning(self) -> Optional[str]:
        """Warning string when the GRPO group size is below the recommendation."""
        if self.grpo_num_generations < self.RECOMMENDED_MIN_GROUP_SIZE:
            return (
                "GRPO group size is below recommended threshold; advantage "
                f"estimates may be high variance (num_generations="
                f"{self.grpo_num_generations} < {self.RECOMMENDED_MIN_GROUP_SIZE}; "
                f"recommended={self.RECOMMENDED_GROUP_SIZE})."
            )
        return None

    def scale_audit(self) -> Dict[str, Any]:
        """Auditable GRPO group-size fields for logs / reports."""
        return {
            "num_generations_requested": self.grpo_num_generations,
            "num_generations_effective": self.grpo_num_generations,
            "grpo_group_size": self.grpo_num_generations,
            "grpo_global_batch": self.per_device_train_batch_size * self.num_gpus,
            "group_advantage_warning": self.group_advantage_warning,
            "is_probe": self.is_probe,
        }

    def validate(self) -> None:
        if self.num_gpus < 1:
            raise SystemExit(f"trainer.num_gpus must be >= 1, got {self.num_gpus}.")
        if self.grpo_num_generations < 2:
            raise SystemExit(
                f"trainer.grpo_num_generations must be >= 2, got "
                f"{self.grpo_num_generations}."
            )
        # Real GRPO (non-probe) must use a group size >= the recommended minimum.
        # A degenerate group of 2 is allowed ONLY as an explicit probe.
        if not self.is_probe and self.grpo_num_generations < self.RECOMMENDED_MIN_GROUP_SIZE:
            raise SystemExit(
                f"Invalid real-GRPO num_generations={self.grpo_num_generations}: "
                f"must be >= {self.RECOMMENDED_MIN_GROUP_SIZE} "
                f"(recommended {self.RECOMMENDED_GROUP_SIZE}). Pass "
                "--grpo-probe to allow a sub-threshold group for API wiring only "
                "(not a valid GRPO learning config)."
            )
        # TRL GRPO: global batch = per_device_batch * num_gpus (grad_accum
        # excluded). num_generations must divide it.
        global_batch = self.per_device_train_batch_size * self.num_gpus
        if global_batch % self.grpo_num_generations != 0:
            valid = [n for n in range(2, global_batch + 1) if global_batch % n == 0]
            raise SystemExit(
                "Invalid GRPO scale: global batch "
                f"(per_device {self.per_device_train_batch_size} x num_gpus "
                f"{self.num_gpus} = {global_batch}) must be divisible by "
                f"grpo_num_generations ({self.grpo_num_generations}). "
                f"Valid num_generations for this batch: {valid or '[increase batch]'}."
            )
        if self.trainer_backend not in ("deepspeed", "fsdp2"):
            raise SystemExit(
                f"trainer.trainer_backend must be 'deepspeed' or 'fsdp2', got "
                f"{self.trainer_backend!r}."
            )
        if self.deepspeed_config and not Path(self.deepspeed_config).is_file():
            raise SystemExit(
                f"trainer.deepspeed_config not found: {self.deepspeed_config}"
            )
        if self.trainer_backend == "fsdp2":
            if not self.fsdp_config:
                raise SystemExit(
                    "trainer_backend=fsdp2 requires --trainer-fsdp-config "
                    "(path to an HF --fsdp_config JSON, e.g. "
                    "local_scripts/fsdp2_qwen2_5vl.json)."
                )
            if not Path(self.fsdp_config).is_file():
                raise SystemExit(
                    f"trainer.fsdp_config not found: {self.fsdp_config}"
                )


def run_proposer_stage(
    generator: Any,
    *,
    dry_run: bool,
    scene_ids: List[str],
    solvability: Optional[Dict[str, Any]],
    solver_model_path: Optional[str],
    base_model_path: Optional[str],
    config: Any,
    edit_budget: int,
    seed: int,
    solver_min_pixels: Optional[int] = None,
    solver_max_pixels: Optional[int] = None,
    num_gpus: int = 1,
    log: Any,
) -> Dict[str, Any]:
    """Let the model choose which changes to keep — the proposing half of self-play.

    Each scene gets ``group_size`` proposals, and each proposal becomes one of
    that scene's task slots, so the solver rollout budget does not change; we
    only moved who picks the tasks. The gold label still comes from splicing, so
    a proposal can make a task easier or harder but never wrong.
    """
    from open_r1.self_evolve.proposer import (
        build_proposer_prompt,
        parse_proposal,
        tag_counterfactual_pairs,
    )

    proposals: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    # Every proposal takes one edit slot, so proposing more than the budget can
    # hold would leave picks unevaluated — and an unevaluated pick is not a bad
    # pick. Size the scene count so the whole group gets answered.
    affordable = int(edit_budget) // max(1, config.group_size)
    max_scenes = min(config.max_scenes, affordable)
    if max_scenes <= 0:
        log(f"  Proposer: the edit budget ({edit_budget}) cannot hold one group "
            f"of {config.group_size} proposals, so no pick could be answered; "
            f"skipping the proposing side. Lower PROPOSER_GROUP_SIZE or raise "
            f"EDIT_FRACTION / the task count.")
        return {"proposals": [], "report": {"num_proposals": 0},
                "source": "budget_too_small"}
    if max_scenes < config.max_scenes:
        log(f"  Proposer: the edit budget fits {max_scenes} scene(s) x "
            f"{config.group_size} proposals, not {config.max_scenes}.")
    # `scene_ids` comes from the seed list, which is a list of TASKS — under
    # self-play one scene yields several of them, so the same scene arrives more
    # than once. Proposing over it twice would emit two proposals with the same
    # id, and the scoring join would then pay one proposal for the other's task.
    seen_scenes: set = set()
    unique_scene_ids = []
    for scene_id in scene_ids:
        if str(scene_id) in seen_scenes:
            continue
        seen_scenes.add(str(scene_id))
        unique_scene_ids.append(scene_id)

    # Only scenes that can actually be proposed over consume a slot; filtering
    # after the slice would let one unusable seed silently no-op the whole
    # proposing side.
    for scene_id in unique_scene_ids:
        if len(prompts) >= max_scenes:
            break
        scene = generator._load_scene(str(scene_id))
        if not scene:
            continue
        prompt = build_proposer_prompt(
            scene, solvability if config.show_competence else None,
            config.min_players, config.max_players)
        if not prompt:
            continue
        replaced = (scene.get("modification", {}) or {}).get("replaced_objects") or []
        original = generator.images_dir / f"{scene_id}_original.png"
        modified = generator.images_dir / f"{scene_id}_modified.png"
        if not (original.is_file() and modified.is_file()):
            continue
        prompts.append({
            "scene_id": str(scene_id),
            "prompt": prompt,
            "num_objects": len(replaced),
            "image_path": [str(original), str(modified)],
        })

    if not prompts:
        log("  Proposer: no scene had more than one changed object; "
            "nothing to propose over this round.")
        return {"proposals": [], "report": {"num_proposals": 0}, "source": "none"}

    if dry_run:
        # No model, so nothing proposes. Walk the action space instead — the
        # subsets a proposer could have picked — purely to exercise the
        # splice -> solve -> learnability plumbing without a GPU. These records
        # carry no completion, so they can never become proposer training data:
        # the model is never taught from a choice it did not make.
        from open_r1.self_evolve.task_editor import enumerate_variants

        for item in prompts:
            scene = generator._load_scene(item["scene_id"])
            subsets = enumerate_variants(scene, max_variants=config.group_size)
            for g, keep in enumerate(subsets):
                proposals.append({
                    "proposal_id": f"{item['scene_id']}::enum{g}",
                    "scene_id": item["scene_id"],
                    "prompt": item["prompt"],
                    "image_path": item["image_path"],
                    "completion": None,
                    "keep": list(keep),
                    "num_players": (config.min_players + config.max_players) // 2,
                    "proposal_source": "enumerated_no_model",
                })
        tagged = tag_counterfactual_pairs(proposals)
        log(f"  Proposer: NO model loaded (dry run) — walked {len(tagged)} "
            f"subset(s) over {len(prompts)} scene(s) to exercise the plumbing. "
            f"These are not proposals and are never trained on.")
        return {"proposals": tagged, "report": {"num_proposals": len(tagged),
                                                "source": "enumerated_no_model"},
                "source": "enumerated_no_model"}

    effective = solver_model_path or base_model_path
    if not effective:
        raise SystemExit(
            "run_proposer_stage: self-play needs a model (pass --model-path)."
        )
    log(f"  Proposer: sampling {config.group_size} proposal(s) for "
        f"{len(prompts)} scene(s) from {effective}.")
    # Data-parallel over all cards: one task per scene, group_size generations
    # each, sharded round-robin across GPUs — same engine as the solver rollout.
    # Each worker's generate_one uses config.seed + gen_index for its per-sample
    # seed, so passing seed=seed reproduces the old serial seed+g exactly.
    from open_r1.self_evolve.online_solver import generate_rollouts_multi_gpu

    prompt_by_scene = {str(item["scene_id"]): item for item in prompts}
    proposer_tasks = [
        {
            "task_id": f"propose::{item['scene_id']}",
            "image_path": list(item["image_path"]),
            "prompt": item["prompt"],
            "metadata": {"role": "proposer", "scene_id": str(item["scene_id"])},
            "scene_id": str(item["scene_id"]),
        }
        for item in prompts
    ]

    scratch_dir = tempfile.mkdtemp(prefix="proposer_rollout_")
    try:
        trajectories = generate_rollouts_multi_gpu(
            proposer_tasks,
            num_generations=config.group_size,
            effective_solver_path=effective,
            num_gpus=max(1, int(num_gpus)),
            scratch_dir=scratch_dir,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=0.95,
            seed=seed,
            min_pixels=solver_min_pixels,
            max_pixels=solver_max_pixels,
            rollout_source="proposer",
            log=log,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    # Reassemble proposals from (scene_id, gen_index). Keyed lookup rather than
    # positional zip: the multi-GPU concat is in shard order, not prompt order.
    completions: Dict[str, str] = {}
    for traj in trajectories:
        scene_id = str((traj.get("metadata") or {}).get("scene_id")
                       or traj.get("scene_id") or "")
        g = int((traj.get("generation_info") or {}).get("gen_index", 0))
        completions[f"{scene_id}::p{g}"] = traj.get("completion") or ""

    # Rebuild in the original prompt/gen order so downstream ids stay stable.
    for item in prompts:
        scene_id = str(item["scene_id"])
        for g in range(config.group_size):
            completion = completions.get(f"{scene_id}::p{g}", "")
            pick = parse_proposal(completion, item["num_objects"],
                                  config.min_players, config.max_players)
            proposals.append({
                "proposal_id": f"{scene_id}::p{g}",
                "scene_id": item["scene_id"],
                "prompt": item["prompt"],
                "image_path": item["image_path"],
                "completion": completion,
                "keep": pick["keep"] if pick else None,
                "num_players": pick["num_players"] if pick else None,
                "proposal_source": "proposer",
            })

    tagged = tag_counterfactual_pairs(proposals)
    parsed = sum(1 for p in tagged if p.get("keep"))
    log(f"  Proposer: {parsed}/{len(tagged)} proposals parsed "
        f"({len(prompts)} scene(s)).")
    return {"proposals": tagged, "report": {"num_proposals": len(tagged),
                                            "num_parsed": parsed},
            "source": "proposer"}


def _extra_trainer_args(route: str) -> List[str]:
    """Per-route trainer hyperparameter passthrough.

    Reads ``SELF_EVOLVE_{GRPO,SFT}_EXTRA_ARGS`` (shell-quoted string) and
    returns it as a token list to be APPENDED to the trainer command. The
    underlying trainers use TRL's argparse-based ``TrlParser`` over
    ``GRPOConfig``/``SFTConfig``, where a later value for the same
    flag overrides an earlier one — so these tokens override the hardcoded
    defaults above (e.g. ``--learning_rate 5e-6 --warmup_ratio 0.03 --beta 0.04``).
    Empty/unset -> no change.
    """
    import shlex
    raw = os.environ.get(f"SELF_EVOLVE_{route.upper()}_EXTRA_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _build_grpo_command(
    grpo_jsonl: str, model_path: str, output_dir: str, max_steps: int,
    use_lora: bool = False, scale: Optional["TrainerScale"] = None,
) -> List[str]:
    """Construct the real GRPO trainer command (grpo_jsonl.py).

    grpo_jsonl.py reads ``--data_file_paths`` (colon-separated JSONL) paired with
    ``--image_folders`` (same count); absolute image paths in the export stay
    absolute under os.path.join, so an empty-root "/" image folder works.
    Reuses the registered ``self_evolve_refined`` reward.

    ``use_lora`` switches to PEFT/LoRA (verified to run a real GRPO step on a
    single 4090: ~43M trainable params, checkpoint produced). The full-parameter
    path (use_lora=False) needs ZeRO-3/offload or more GPU memory.

    ``scale`` carries the multi-GPU / batch / generation knobs. TRL GRPO requires
    ``per_device_train_batch_size * num_gpus * grad_accum`` to be divisible by
    ``num_generations`` — the scale config is validated for this upstream.
    """
    scale = scale or TrainerScale()
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={scale.num_gpus}",
        str(SRC_DIR / "open_r1" / "grpo_jsonl.py"),
        "--model_name_or_path", model_path,
        "--dataset_name", "self_evolve_refined_grpo",
        "--data_file_paths", grpo_jsonl,
        "--image_folders", "/",
        "--reward_funcs", "self_evolve_refined",
        "--output_dir", output_dir,
        "--max_steps", str(max_steps),
        "--per_device_train_batch_size", str(scale.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(scale.gradient_accumulation_steps),
        "--num_generations", str(scale.grpo_num_generations),
        "--gradient_checkpointing", "true",
        "--report_to", os.environ.get("SELF_EVOLVE_REPORT_TO", "none"),
        "--save_steps", str(max_steps),
        "--save_only_model", "true",
        "--logging_steps", "1",
        "--bf16", "true",
    ]
    # Distributed backend selection (GRPO route only). fsdp2 uses HF's native
    # --fsdp / --fsdp_config path; the required FSDP_VERSION=2 and
    # FSDP_STATE_DICT_TYPE=FULL_STATE_DICT env vars are set in _launch (HF does
    # NOT propagate the version or state-dict type from the JSON — see the config
    # file's _comment2). deepspeed keeps the existing --deepspeed flag.
    if scale.trainer_backend == "fsdp2":
        cmd += ["--fsdp", "full_shard auto_wrap", "--fsdp_config", scale.fsdp_config]
    elif scale.deepspeed_config:
        cmd += ["--deepspeed", scale.deepspeed_config]
    if use_lora:
        cmd += [
            "--use_peft", "true",
            "--lora_r", str(scale.lora_r), "--lora_alpha", str(scale.lora_alpha),
            "--lora_dropout", str(scale.lora_dropout),
            "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        ]
    cmd += _extra_trainer_args("grpo")
    return cmd


def _build_sft_command(sft_yaml: str, model_path: str, output_dir: str, max_steps: int,
                       use_lora: bool = False, scale: Optional["TrainerScale"] = None) -> List[str]:
    """Construct the real SFT trainer command (sft_jsonl.py, TRL SFTTrainer).

    Uses ``sft_jsonl.py`` (NOT the grounding-only ``sft.py``): the self-evolve
    replay schema is multi-image + free-text ``<think>/<answer>`` completions,
    which ``sft.py``'s bbox collator cannot consume. ``sft_jsonl.py`` reads the
    same YAML data-config convention (``datasets: - json_path: ...``).
    """
    scale = scale or TrainerScale()
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={scale.num_gpus}",
        str(SRC_DIR / "open_r1" / "sft_jsonl.py"),
        "--model_name_or_path", model_path,
        "--dataset_name", sft_yaml,
        "--output_dir", output_dir,
        "--max_steps", str(max_steps),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(scale.gradient_accumulation_steps),
        "--gradient_checkpointing", "true",
        "--report_to", os.environ.get("SELF_EVOLVE_REPORT_TO", "none"),
        "--save_steps", str(max_steps),
        "--save_only_model", "true",
        "--logging_steps", "1",
        "--bf16", "true",
    ]
    if scale.deepspeed_config:
        cmd += ["--deepspeed", scale.deepspeed_config]
    if use_lora:
        cmd += [
            "--use_peft", "true",
            "--lora_r", str(scale.lora_r), "--lora_alpha", str(scale.lora_alpha),
            "--lora_dropout", str(scale.lora_dropout),
            "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        ]
    cmd += _extra_trainer_args("sft")
    return cmd


def _write_sft_yaml(sft_jsonl: str, out_dir: Path) -> Optional[str]:
    """Write a sft.py-compatible YAML pointing at the SFT replay JSONL."""
    if not sft_jsonl or not Path(sft_jsonl).is_file():
        return None
    yaml_path = out_dir / "sft_replay.yaml"
    yaml_path.write_text(
        "datasets:\n"
        f"  - json_path: {sft_jsonl}\n"
        "    sampling_strategy: all\n",
        encoding="utf-8",
    )
    return str(yaml_path)


def _ensure_grpo_train_init_model(
    *,
    prev_solver_model_path: Optional[str],
    base_model_path: str,
    iter_dir: Path,
    iteration_index: int,
    carry_forward_enabled: bool,
    do_merge: bool,
    full_param: bool = False,
    log: Any,
) -> Dict[str, Any]:
    """Resolve (and, when ``do_merge``, actually build) the model GRPO trains FROM.

    Full-parameter (``full_param=True``): the previous iteration's GRPO
    checkpoint is ALREADY a complete model dir (no adapter), so this iteration
    just trains FROM it directly — no merge, no merged_* dir. The LoRA branch
    below (merge-then-fresh-adapter) is skipped entirely.

    iteration 0 -- or carry-forward disabled, or no previous checkpoint -- keeps
    the historical behaviour: GRPO starts from the original ``base_model_path``.

    For iteration N>0 with carry-forward on, the previous iteration's GRPO LoRA
    adapter is merged into ITS base (merge-then-fresh-adapter) to produce a full
    ``merged_accumulated`` model directory, which becomes this iteration's GRPO
    init model so solver weights truly compound across rounds. The merged dir is
    named with ``qwen`` in it so grpo_jsonl.py's ``get_vlm_module`` substring
    check still resolves the model family.

    ``do_merge`` gates the heavy GPU work: when False (dry-run trainer, or LIVE
    with no --execute-grpo-smoke) the *intended* command/path is still recorded
    for audit, but no subprocess runs. When True the merge is executed and every
    inconsistency raises immediately -- there is NO silent fallback to
    base, because a silent fallback hides weights that never accumulated.
    """
    info: Dict[str, Any] = {
        "grpo_train_init_model": base_model_path,
        "grpo_train_init_kind": "base",
        "carry_forward_enabled": bool(carry_forward_enabled),
        "prev_solver_model_path": prev_solver_model_path,
        "merged_base_path": None,
        "merged_base_from_adapter": None,
        "merge_rc": None,
        "merge_command": None,
    }

    log(f"[carry_forward] enabled={str(bool(carry_forward_enabled)).lower()}")
    log(f"[carry_forward] iteration={iteration_index}")
    log(f"[carry_forward] prev_solver_model_path={prev_solver_model_path}")

    if (not carry_forward_enabled) or iteration_index == 0 or not prev_solver_model_path:
        log(f"[carry_forward] grpo_train_init_kind={info['grpo_train_init_kind']}")
        log(f"[carry_forward] grpo_train_init_model={info['grpo_train_init_model']}")
        return info

    # Full-parameter carry-forward: prev GRPO checkpoint is already a full model
    # dir — train straight from it, no merge (no merged_* dir, no merge_lora.py).
    if full_param:
        prev = Path(prev_solver_model_path)
        if do_merge and not (prev / "config.json").is_file():
            raise RuntimeError(
                f"[carry_forward] full-parameter carry-forward for iter "
                f"{iteration_index} but prev_solver_model_path is not a full model "
                f"dir: {prev} (missing config.json). Refusing to fall back to base."
            )
        info["grpo_train_init_model"] = str(prev)
        info["grpo_train_init_kind"] = "full_param_accumulated"
        log(f"[carry_forward] grpo_train_init_kind={info['grpo_train_init_kind']}")
        log(f"[carry_forward] grpo_train_init_model={info['grpo_train_init_model']}")
        return info

    # Carry-forward requested for iter>0: the prev checkpoint MUST be a PEFT dir.
    prev = Path(prev_solver_model_path)
    adapter_cfg = prev / "adapter_config.json"
    merged_dir = iter_dir / "merged_solver_qwen2_5_vl"  # name MUST contain 'qwen'
    merge_cmd = [
        sys.executable,
        str(REPO_ROOT / "local_scripts" / "self_evolve" / "workflow" / "merge_lora.py"),
        "--adapter-path", str(prev),
        "--output-dir", str(merged_dir),
        "--torch-dtype", "bf16",
        "--trust-remote-code",
        "--overwrite",
    ]
    # Intended plan is recorded regardless of do_merge (auditable next_action).
    info["merge_command"] = " ".join(merge_cmd)
    info["merged_base_from_adapter"] = str(prev)
    info["merged_base_path"] = str(merged_dir)
    info["grpo_train_init_model"] = str(merged_dir)
    info["grpo_train_init_kind"] = "merged_accumulated"

    if do_merge and (not prev.is_dir() or not adapter_cfg.is_file()):
        raise RuntimeError(
            f"[carry_forward] carry-forward enabled for iter {iteration_index} but "
            f"prev_solver_model_path is not a LoRA adapter dir: {prev} "
            f"(missing {adapter_cfg.name}). Refusing to silently fall back to base."
        )

    if do_merge:
        log(f"[carry_forward] merging {prev} -> {merged_dir}")
        log("    " + info["merge_command"])
        import subprocess
        run_env = dict(os.environ)
        run_env["PYTHONPATH"] = f"{SRC_DIR}:{run_env.get('PYTHONPATH', '')}"
        rc = subprocess.run(merge_cmd, cwd=str(REPO_ROOT), env=run_env).returncode
        info["merge_rc"] = rc
        log(f"[carry_forward] merge_rc={rc}")
        if rc != 0:
            raise RuntimeError(
                f"[carry_forward] merge_lora.py exited rc={rc} for adapter {prev}; "
                f"NOT launching GRPO. See merge output above."
            )
        if not (merged_dir / "config.json").is_file():
            raise RuntimeError(
                f"[carry_forward] merged dir missing config.json: {merged_dir}"
            )
        weights = (list(merged_dir.glob("*.safetensors"))
                   + list(merged_dir.glob("*.bin")))
        if not weights:
            raise RuntimeError(
                f"[carry_forward] merged dir has no *.safetensors/*.bin weights: "
                f"{merged_dir}"
            )
        # retain-latest-1 GC: once THIS iteration's merge has succeeded, the
        # previous iteration's merged base is dead weight. Its input was the
        # prev iteration's *adapter* (prev_solver_model_path), never the prev
        # merged dir, so deleting it cannot break this or any later merge. This
        # bounds carry-forward disk to ~one full merged model (~16GB) instead of
        # one per round — without it a long run fills the disk. Best
        # effort: a GC failure warns but never fails the run.
        # SELF_EVOLVE_DISABLE_MERGE_GC=1 keeps every round's merged model on disk
        # so the per-round checkpoints survive for a per-iteration eval curve
        # (base vs iter1..N). Only safe when disk can hold ~one merged model per
        # round (~16GB each). Default (unset/0) preserves retain-latest-1 GC.
        if os.environ.get("SELF_EVOLVE_DISABLE_MERGE_GC", "0") == "1":
            info["gc_disabled"] = True
            log("[carry_forward] GC disabled (SELF_EVOLVE_DISABLE_MERGE_GC=1): "
                "retaining prev merged base for per-round eval.")
        else:
            try:
                prev_iter_dir = prev.parent.parent  # iter_(N-1)/checkpoints/grpo -> iter_(N-1)
                stale_merged = prev_iter_dir / "merged_solver_qwen2_5_vl"
                if (stale_merged.is_dir()
                        and stale_merged.resolve() != merged_dir.resolve()):
                    import shutil
                    shutil.rmtree(stale_merged)
                    info["gc_removed_prev_merged"] = str(stale_merged)
                    log(f"[carry_forward] GC: removed stale merged base {stale_merged}")
            except Exception as e:  # noqa: BLE001
                log(f"[carry_forward] GC WARNING: could not remove prev merged "
                    f"base (non-fatal): {e}")
    else:
        log("[carry_forward] do_merge=false (dry-run / not executing GRPO): "
            "recording intended merged init model only, NOT merging.")

    log(f"[carry_forward] grpo_train_init_kind={info['grpo_train_init_kind']}")
    log(f"[carry_forward] grpo_train_init_model={info['grpo_train_init_model']}")
    return info


def run_trainer_stage(
    export_summary: Dict[str, Any],
    *,
    dry_run: bool,
    prev_solver_model_path: Optional[str],
    base_model_path: str,
    iter_dir: Path,
    iteration_index: int,
    max_steps: int,
    execute_grpo: bool,
    execute_sft: bool,
    use_lora: bool,
    carry_forward_enabled: bool = True,
    scale: Optional["TrainerScale"] = None,
    log: Any,
) -> Dict[str, Any]:
    """Build and optionally execute the SFT→GRPO recipe (either or both).

    The recipe is a chain in which each trained stage feeds the next via a
    merged full model, so the following stage starts from the LATEST checkpoint:

      round_model  (base on iter 0, else merged prev-GRPO adapter)
        → [SFT]  → merge → post_sft_model
        → [GRPO] adapter                        (starts from post_sft else
                                                 post_sft else round)

    Each of ``execute_sft`` / ``execute_grpo`` independently
    gates its stage; a disabled stage is skipped and the chain closes over it.
    """
    pathways = default_training_pathways()
    scale = scale or TrainerScale()
    # Full-parameter path: no PEFT adapter is produced, so each trainer's output
    # dir IS already a complete model. The LoRA-only merge glue (merge_lora.py,
    # merged_* dirs) is skipped and the next stage starts directly from the prev
    # stage's checkpoint dir. LoRA path (use_lora=True) keeps the merge chain.
    full_param = not use_lora

    sft_path = export_summary.get("sft_replay_jsonl")
    grpo_path = export_summary.get("grpo_tasks_jsonl")
    grpo_yaml = export_summary.get("grpo_data_config")

    # Directory names carry a "qwen2_5_vl" suffix: under full-parameter training
    # these checkpoints are passed to the next stage's trainer as --model_path,
    # and qwen_module.get_model_class picks the model class purely by whether the
    # path string contains qwen25vl/qwen2vl. A bare "sft"/"grpo" directory name
    # has no "qwen" -> ValueError: Unsupported model. After normalization the
    # suffix contains qwen25vl and hits the 2.5 branch.
    grpo_ckpt_dir = str(iter_dir / "checkpoints" / "grpo_qwen2_5_vl")
    sft_ckpt_dir = str(iter_dir / "checkpoints" / "sft_qwen2_5_vl")

    num_sft = int(export_summary.get("num_sft_replay", 0))
    # SFT runs only when the SOLVER produced positives. Proposer records ride
    # the same file, so counting the total here would let a round with zero
    # solver positives launch SFT on proposer-only data — which teaches the
    # model to answer every prompt with <keep>[...]</keep>, and GRPO then
    # starts from that checkpoint.
    num_sft_solver = int(export_summary.get("num_sft_solver", num_sft))

    pathways["grpo"].export_ready = bool(export_summary.get("num_grpo_tasks", 0))
    pathways["grpo"].export_path = grpo_path
    pathways["grpo"].trainer_input_ready = bool(grpo_yaml)
    pathways["sft"].export_ready = bool(num_sft_solver)
    pathways["sft"].export_path = sft_path
    pathways["sft"].trainer_input_ready = bool(num_sft_solver)

    # Resolve the accumulated model for this round. Iteration 0 is a cold start
    # from base; later rounds merge the previous GRPO adapter first.
    do_merge = (not dry_run) and bool(execute_grpo) and bool(grpo_path)
    solver_lineage = _ensure_grpo_train_init_model(
        prev_solver_model_path=prev_solver_model_path,
        base_model_path=base_model_path,
        iter_dir=iter_dir,
        iteration_index=iteration_index,
        carry_forward_enabled=carry_forward_enabled,
        do_merge=do_merge,
        full_param=full_param,
        log=log,
    )
    round_model = solver_lineage["grpo_train_init_model"]

    # --- SFT stage (positive buffer) ---
    sft_yaml = _write_sft_yaml(str(sft_path), iter_dir / "exports") if (sft_path and num_sft_solver) else None
    sft_cmd = _build_sft_command(sft_yaml, round_model, sft_ckpt_dir, max_steps, use_lora=use_lora, scale=scale) if sft_yaml else None
    # Full-parameter: sft_ckpt_dir is itself a full model, no merge → the next
    # stage starts straight from it. LoRA: merge the adapter into merged_sft_*.
    post_sft_model = sft_ckpt_dir if full_param else str(iter_dir / "merged_sft_qwen2_5_vl")

    # --- GRPO stage — starts from post-SFT if SFT ran, else the round model.
    grpo_input_model = post_sft_model if execute_sft else round_model
    grpo_cmd = _build_grpo_command(str(grpo_path), grpo_input_model, grpo_ckpt_dir, max_steps, use_lora=use_lora, scale=scale) if grpo_path else None

    _recipe = "_".join(
        (["sft"] if execute_sft else []) + (["grpo"] if grpo_cmd else [])
    ) or "none"
    solver_lineage["training_recipe"] = _recipe
    solver_lineage["round_model_path"] = round_model
    solver_lineage["sft_adapter_path"] = sft_ckpt_dir if execute_sft else None
    solver_lineage["post_sft_merged_model_path"] = post_sft_model if execute_sft else None
    solver_lineage["grpo_input_model_path"] = grpo_input_model
    # Store commands in next_action so they land in state JSON (readable proof).
    if grpo_cmd:
        pathways["grpo"].next_action = "EXEC: " + " ".join(grpo_cmd)
    if sft_cmd:
        pathways["sft"].next_action = "EXEC: " + " ".join(sft_cmd)

    if execute_sft and not sft_cmd:
        pathways["sft"].status = "mandatory_but_blocked"
        pathways["sft"].blocked_reason = (
            "No solver SFT replay records this iteration (the positive buffer "
            "is empty under this solver). SFT is mandatory; it needs >=1 "
            "positive solver trajectory. Proposer records alone do not qualify: "
            "training on them without any solving example teaches the model to "
            "propose instead of answer."
        )
        pathways["sft"].next_action = (
            "Increase accepted tasks / generations until a positive trajectory "
            "is produced, then run src/open_r1/sft_jsonl.py."
        )
    elif not execute_sft:
        pathways["sft"].mandatory = False
        pathways["sft"].status = "disabled"
        pathways["sft"].next_action = "Disabled by GRPO-only recipe."

    trainer_mode = "dry_run"
    checkpoint_created = False

    def _launch(name: str, cmd: List[str], ckpt_dir: str) -> bool:
        """Launch one trainer smoke; record state; return checkpoint_created."""
        nonlocal trainer_mode
        log(f"  Trainer: LAUNCHING real {name.upper()} smoke (GPU)…")
        log("    " + " ".join(cmd))
        import subprocess
        run_env = dict(os.environ)
        # Default the visible devices to the first `num_gpus` if the operator
        # has not pinned CUDA_VISIBLE_DEVICES themselves (their choice wins).
        run_env.setdefault(
            "CUDA_VISIBLE_DEVICES",
            ",".join(str(i) for i in range(scale.num_gpus)),
        )
        # wandb off by default; opt in via SELF_EVOLVE_REPORT_TO=wandb (the trainer
        # subprocess then reports to whatever wandb project/entity the env sets).
        if os.environ.get("SELF_EVOLVE_REPORT_TO", "none") == "none":
            run_env["WANDB_MODE"] = "disabled"
        # Colocated vLLM sleep mode uses CuMemAllocator. PyTorch expandable
        # segments are incompatible with that memory pool and abort vLLM init.
        run_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        run_env.pop("PYTORCH_ALLOC_CONF", None)
        run_env["PYTHONPATH"] = f"{SRC_DIR}:{run_env.get('PYTHONPATH', '')}"
        # FSDP2 (GRPO route only). transformers 4.49 exports ACCELERATE_USE_FSDP
        # from --fsdp but NOT the FSDP version or state-dict type, and accelerate
        # defaults to FSDP1 / SHARDED_STATE_DICT. Both must be forced via env or
        # the run silently degrades to FSDP1 and writes unloadable sharded
        # checkpoints. Only the GRPO command carries --fsdp; SFT stays on
        # deepspeed, so scope these to name == "grpo".
        is_fsdp2 = getattr(scale, "trainer_backend", "deepspeed") == "fsdp2" and name == "grpo"
        if is_fsdp2:
            run_env["FSDP_VERSION"] = "1"
            run_env["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
        # Ensure the trainer interpreter's bin dir is on PATH so DeepSpeed's JIT
        # build of the CPU-Adam C++ extension can find `ninja` (it shells out to
        # the executable, not the python package) even when the env was not
        # `conda activate`-d before launching. Not needed under FSDP2 (no C++
        # extension build), but harmless, so keep it unconditional.
        py_bindir = str(Path(cmd[0]).resolve().parent)
        run_env["PATH"] = py_bindir + os.pathsep + run_env.get("PATH", "")
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=run_env).returncode
        trainer_mode = "live"
        pathways[name].trainer_executed = (rc == 0)
        pathways[name].status = "trainer_executed" if rc == 0 else "trainer_failed"
        made = False
        ckpt = Path(ckpt_dir)
        if rc == 0 and ckpt.is_dir() and any(ckpt.iterdir()):
            pathways[name].checkpoint_created = True
            pathways[name].checkpoint_path = ckpt_dir
            made = True
        if rc != 0:
            pathways[name].blocked_reason = (
                f"{name} trainer exited rc={rc}; see stdout above."
            )
        elif not made:
            pathways[name].status = "trainer_failed"
            pathways[name].blocked_reason = (
                f"{name} trainer exited successfully but created no checkpoint."
            )
        log(f"  Trainer: {name.upper()} smoke rc={rc}, checkpoint_created={made}")
        return made

    def _merge_adapter(name: str, adapter_dir: str, merged_dir: str) -> bool:
        """Merge a LoRA adapter into its base -> full model dir. Returns ok."""
        merge_cmd = [
            sys.executable,
            str(REPO_ROOT / "local_scripts" / "self_evolve" / "workflow" / "merge_lora.py"),
            "--adapter-path", adapter_dir,
            "--output-dir", merged_dir,
            "--torch-dtype", "bf16",
            "--trust-remote-code",
            "--overwrite",
        ]
        log(f"  Trainer: merging {name.upper()} adapter -> {merged_dir}")
        import subprocess
        run_env = dict(os.environ)
        run_env["PYTHONPATH"] = f"{SRC_DIR}:{run_env.get('PYTHONPATH', '')}"
        merge_rc = subprocess.run(merge_cmd, cwd=str(REPO_ROOT), env=run_env).returncode
        solver_lineage[f"{name}_merge_rc"] = merge_rc
        ok = merge_rc == 0 and (Path(merged_dir) / "config.json").is_file()
        if not ok:
            pathways[name].status = "trainer_failed"
            pathways[name].blocked_reason = f"{name.upper()} adapter merge failed rc={merge_rc}."
        return ok

    if not dry_run:
        # SFT → GRPO. SFT merges into a full model that GRPO starts from, so
        # GRPO always begins from the latest weights.
        stage_ready = True
        if execute_sft:
            if not sft_cmd:
                log("  Trainer: WARNING — no positive SFT replay examples this "
                    "iteration (expected early on: positives now require a "
                    "grounded box). Skipping SFT; GRPO starts from the round model.")
                execute_sft = False
        if execute_sft:
            stage_ready = _launch("sft", sft_cmd, sft_ckpt_dir)
            # Full-parameter: sft_ckpt_dir is already the full model (post_sft_model
            # points at it), so no merge. LoRA: merge the adapter into post_sft_model.
            if stage_ready and not full_param:
                stage_ready = _merge_adapter("sft", sft_ckpt_dir, post_sft_model)

        # GRPO is launched only after the optional SFT stage has merged.
        if execute_grpo and grpo_cmd and stage_ready:
            made = _launch("grpo", grpo_cmd, grpo_ckpt_dir)
            if made:
                checkpoint_created = True
        if not (execute_grpo or execute_sft):
            log("  Trainer: LIVE mode but no --execute-*-smoke flag set; "
                "commands constructed, NOT launched.")
            for tag, cmd in (("GRPO", grpo_cmd), ("SFT", sft_cmd)):
                if cmd:
                    log(f"    [{tag}] " + " ".join(cmd))
    else:
        # Dry-run: print the real commands; nothing launched.
        log("  Trainer: DRY-RUN — real commands constructed, NOT launched:")
        if execute_sft and sft_cmd:
            log("    [SFT ] " + " ".join(sft_cmd))
            log(f"    [MERGE] {sft_ckpt_dir} -> {post_sft_model}")
        if grpo_cmd:
            log("    [GRPO] " + " ".join(grpo_cmd))

    # Resolve the solver_model_path CARRIED FORWARD (this becomes the next
    # iteration's prev_solver_model_path and thus the merge adapter source).
    # It must point at THIS iteration's GRPO ADAPTER dir, never at base —
    # carrying base forward is exactly what made iter_001 try to merge base into
    # base. Distinguish created (real ckpt on disk) / planned (command built but
    # intentionally not launched: dry-run or LIVE w/o --execute-grpo-smoke) /
    # no_checkpoint (real GRPO ran but produced nothing — do NOT fall back to
    # base) / no_grpo_export (no GRPO tasks this round).
    if checkpoint_created:
        solver_model_path = grpo_ckpt_dir
        solver_model_path_status = "created"
    elif grpo_cmd is not None and (dry_run or not execute_grpo):
        solver_model_path = grpo_ckpt_dir
        solver_model_path_status = "planned"
    elif grpo_cmd is not None:
        # Real GRPO attempted but no checkpoint — explicit, never silent base.
        solver_model_path = None
        solver_model_path_status = "no_checkpoint"
        log("  Trainer: WARNING — GRPO executed but no checkpoint was created; "
            "carrying solver_model_path=None (status=no_checkpoint) instead of "
            "silently falling back to base. Next iteration cannot carry forward.")
    else:
        solver_model_path = None
        solver_model_path_status = "no_grpo_export"
    log(f"  Trainer: solver_model_path={solver_model_path} "
        f"(status={solver_model_path_status})")

    return {
        "trainer_mode": trainer_mode,
        "solver_model_path": solver_model_path,
        "solver_model_path_status": solver_model_path_status,
        "checkpoint_created": checkpoint_created,
        "grpo_command": grpo_cmd,
        "sft_command": sft_cmd,
        "training_pathways": {k: v.to_dict() for k, v in pathways.items()},
        "solver_lineage": solver_lineage,
    }


# ===========================================================================
# One iteration
# ===========================================================================

def _summarize_reference_feedback(
    prev_iteration_dir: Optional[Path],
    run_level_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate the previous iteration's Reference VLM feedback.

    Reads ``reference_feedback.jsonl`` and separates the difficulty signals so
    the policy update can be bidirectional rather than
    monotonically lowering difficulty:

      - ``suggested_difficulty_histogram_all`` — raw histogram over every record
        (kept for logging/audit; includes target echoes).
      - ``reduce_difficulty_count`` — records that explicitly asked to reduce
        difficulty (``generator_feedback.reason == "reduce_difficulty"`` OR a
        too-hard reject). These are genuine down-shift signals.
      - ``strict_easier_suggestion_count`` — records whose ``suggested_difficulty``
        is strictly below the record's own ``difficulty_target`` (target echoes
        excluded).
      - ``target_echo_count`` — accepted / not-too-hard records whose suggestion
        merely echoes the target. These MUST NOT trigger a down-shift.
      - ``too_difficult_count`` — too-hard rejects (canonical bucket).
      - ``difficulty_target_histogram`` / ``difficulty_gap_histogram`` — audit.

    Older feedback records may predate the
    ``difficulty_target`` field. Rather than silently treating them as a real
    ``medium`` target (which would mislabel echoes/strict-easier), the target is
    resolved in order: ``record.difficulty_target`` → ``run_level_target`` →
    ``"medium"``, and the missing/fallback usage is counted for audit.

    Only the first three (reduce / strict-easier / too_difficult) are allowed to
    drive an easier shift downstream. Returns an empty summary when no prior
    feedback is available.
    """
    from collections import Counter

    _order = {"easy": 0, "medium": 1, "hard": 2}
    summary: Dict[str, Any] = {
        "reject_reason_counts": {},
        "suggested_difficulty_histogram": {},
        "suggested_difficulty_histogram_all": {},
        "difficulty_target_histogram": {},
        "difficulty_gap_histogram": {},
        "reduce_difficulty_count": 0,
        "strict_easier_suggestion_count": 0,
        "target_echo_count": 0,
        "too_difficult_count": 0,
        "difficulty_target_missing_count": 0,
        "difficulty_target_fallback_used_count": 0,
        "difficulty_target_fallback_value_histogram": {},
        "num_judged": 0,
        "num_rejected": 0,
    }
    if prev_iteration_dir is None:
        return summary
    fb_path = reference_feedback_path(prev_iteration_dir)
    if not Path(fb_path).is_file():
        return summary

    from open_r1.self_evolve.io import read_jsonl

    records = read_jsonl(fb_path)
    reject_counter: Counter = Counter()
    suggested_counter: Counter = Counter()
    target_counter: Counter = Counter()
    gap_counter: Counter = Counter()
    fallback_value_counter: Counter = Counter()
    reduce_count = 0
    strict_easier_count = 0
    target_echo_count = 0
    too_difficult_count = 0
    target_missing_count = 0
    target_fallback_used_count = 0
    num_rejected = 0

    for rec in records:
        # Unjudged budget-passthrough records carry no Reference VLM judgment;
        # they must not contribute any difficulty signal.
        if rec.get("reference_judged") is False:
            continue

        accepted = rec.get("accepted")
        rr = rec.get("reject_reason") or ""
        too_hard = ("too_hard" in rr) or ("too_difficult" in rr)
        if accepted is False:
            num_rejected += 1
            canon = "too_difficult" if too_hard else (rr or "unspecified")
            reject_counter[canon] += 1
            if too_hard:
                too_difficult_count += 1

        gf = rec.get("generator_feedback") or {}
        sd = gf.get("suggested_difficulty")
        reason = gf.get("reason") or ""

        # Target resolution: record → run-level →
        # "medium", counting missing/fallback usage instead of silently assuming
        # a real medium target.
        rec_target = rec.get("difficulty_target")
        if rec_target:
            target = str(rec_target)
        else:
            target_missing_count += 1
            target = str(run_level_target) if run_level_target else "medium"
            target_fallback_used_count += 1
            fallback_value_counter[target] += 1
        target_counter[str(target)] += 1

        if sd:
            suggested_counter[str(sd)] += 1
            # gap = how much easier the suggestion is than the target
            # (positive => suggestion is easier than target).
            gap = _order.get(str(target), 1) - _order.get(str(sd), 1)
            gap_counter[str(gap)] += 1
            if gap > 0:
                strict_easier_count += 1
            elif gap == 0 and accepted is not False and not too_hard:
                target_echo_count += 1

        # Explicit reduce signal: the judge tagged the record to reduce
        # difficulty, or it was rejected as too hard.
        if reason == "reduce_difficulty" or too_hard:
            reduce_count += 1

    summary.update({
        "reject_reason_counts": dict(reject_counter),
        # Raw histogram, kept for audit/logging only. The policy update does not
        # decides "easier" from this histogram (a medium-target echo would skew
        # it); it keys on the explicit counts below instead.
        "suggested_difficulty_histogram": dict(suggested_counter),
        "suggested_difficulty_histogram_all": dict(suggested_counter),
        "difficulty_target_histogram": dict(target_counter),
        "difficulty_gap_histogram": dict(gap_counter),
        "reduce_difficulty_count": reduce_count,
        "strict_easier_suggestion_count": strict_easier_count,
        "target_echo_count": target_echo_count,
        "too_difficult_count": too_difficult_count,
        "difficulty_target_missing_count": target_missing_count,
        "difficulty_target_fallback_used_count": target_fallback_used_count,
        "difficulty_target_fallback_value_histogram": dict(fallback_value_counter),
        "num_judged": len(records),
        "num_rejected": num_rejected,
    })
    return summary


def run_one_iteration(
    *,
    iteration_index: int,
    output_root: Path,
    prev_iteration_dir: Optional[Path],
    args: argparse.Namespace,
    log: Any,
) -> Path:
    iteration_id = f"iter_{iteration_index:03d}"
    iter_dir = output_root / iteration_id
    iter_dir.mkdir(parents=True, exist_ok=True)
    # Visual tokens are 28x28 pixels each. If the images alone cannot fit in the
    # prompt budget the processor truncates them, and the model then answers
    # about images it never saw — a failure that looks like a reward problem.
    _px = int(getattr(args, "solver_min_pixels", None) or 200704)
    _max_players = int(getattr(args, "proposer_max_players", args.num_players))
    _img_tokens = (_px // (28 * 28)) * max(args.num_players, _max_players)
    # max_prompt_length is a trainer flag, so it arrives through the extra-args
    # string rather than this parser.
    _extra = os.environ.get("SELF_EVOLVE_GRPO_EXTRA_ARGS", "").split()
    _budget = 0
    if "--max_prompt_length" in _extra:
        try:
            _budget = int(_extra[_extra.index("--max_prompt_length") + 1])
        except (IndexError, ValueError):
            _budget = 0
    if _budget and _img_tokens >= _budget:
        log(f"  [WARN] {max(args.num_players, _max_players)} images at "
            f"{_px} pixels need ~{_img_tokens} visual tokens but the prompt "
            f"budget is {_budget}. The images will be truncated. Raise "
            f"GRPO_MAX_PROMPT_LEN or lower the resolution / player count.")
    proposer_cfg = ProposerConfig(
        group_size=int(getattr(args, "proposer_group_size", 4)),
        temperature=float(getattr(args, "proposer_temperature", 1.0)),
        max_new_tokens=int(getattr(args, "proposer_max_new_tokens", 512)),
        learnability_threshold=float(
            getattr(args, "proposer_learnability_threshold", 0.5)),
        max_scenes=int(getattr(args, "proposer_max_scenes", 8)),
        show_competence=not bool(getattr(args, "proposer_no_competence", False)),
        min_players=int(getattr(args, "proposer_min_players", 3)),
        max_players=int(getattr(args, "proposer_max_players", 8)),
    )

    log(f"\n{'='*64}\n  ITERATION {iteration_index}  ({iteration_id})\n{'='*64}")

    # --- 1. Load prior state (proves N+1 reads N) ---
    prev_policy: Optional[Dict[str, Any]] = None
    prev_failure_profile: Optional[Dict[str, Any]] = None
    prev_solver_state: Optional[Dict[str, Any]] = None
    read_prev = False
    if prev_iteration_dir is not None:
        prev_policy = load_generator_policy(prev_iteration_dir)
        prev_failure_profile = load_failure_profile(prev_iteration_dir)
        prev_solver_state = load_solver_update_state(prev_iteration_dir)
        read_prev = any(x is not None for x in
                        (prev_policy, prev_failure_profile, prev_solver_state))
        log(f"  Read prev state from {prev_iteration_dir.name}: "
            f"policy={prev_policy is not None}, "
            f"failure_profile={prev_failure_profile is not None}, "
            f"solver_state={prev_solver_state is not None}")

    # --- 2. Generator policy for THIS iteration ---
    if iteration_index == 0 or prev_policy is None:
        policy = default_generator_policy()
        policy_changes: Dict[str, Any] = {}
    else:
        # Build a Reference VLM feedback summary from the PREVIOUS iteration so
        # the policy update is driven by real reject reasons + suggested
        # difficulty, not just reward means.
        # Run-level difficulty target fallback for older feedback records that
        # predate the per-record difficulty_target field.
        prev_run_target = (
            (prev_policy.get("difficulty_policy", {}) or {}).get("difficulty_target")
            if prev_policy else None
        )
        reference_summary = _summarize_reference_feedback(
            prev_iteration_dir, run_level_target=prev_run_target
        )
        log(f"  Reference feedback summary (prev iter): "
            f"reject_reasons={reference_summary.get('reject_reason_counts')}, "
            f"reduce={reference_summary.get('reduce_difficulty_count')}, "
            f"strict_easier={reference_summary.get('strict_easier_suggestion_count')}, "
            f"target_echo={reference_summary.get('target_echo_count')}, "
            f"too_difficult={reference_summary.get('too_difficult_count')}, "
            f"target_missing={reference_summary.get('difficulty_target_missing_count')}, "
            f"suggested_all={reference_summary.get('suggested_difficulty_histogram_all')}")
        policy = update_generator_policy(
            prev_policy, prev_failure_profile or {},
            reference_summary=reference_summary,
        )
        policy_changes = diff_generator_policy(prev_policy, policy)
        fb = policy.get("feedback_inputs", {})
        log(f"  Generator policy updated: v{prev_policy.get('generator_policy_version')} "
            f"-> v{policy.get('generator_policy_version')}; "
            f"reason='{policy.get('policy_update_reason')}'; "
            f"suggests_easier={fb.get('suggests_easier')}; "
            f"changed: {list(policy_changes.keys())}")
    save_generator_policy(iter_dir, policy)

    # --- 3+4. Generator + Reference VLM ---
    # RESUME shortcut: if this iter_dir already has candidate_tasks.jsonl +
    # accepted_tasks.jsonl + reference_feedback.jsonl on disk (a previous run
    # got past the Reference VLM stage before crashing / being killed), skip
    # re-generating candidates AND skip the Reference VLM (paid GPT-4o calls).
    # Every prior stage lands atomically via write_jsonl at end-of-stage, so
    # "file exists" is a valid stage-complete marker. Set
    # SELF_EVOLVE_DISABLE_RESUME=1 to force a fresh run.
    # Defined here so the resume path (which skips generation entirely) and the
    # non-self-play path both reach the scoring stage with these set.
    proposals: List[Dict[str, Any]] = []
    proposer_report: Dict[str, Any] = {}
    proposer_stage_source = "none"
    quota_target = int(args.num_train_tasks)
    use_quota = not bool(getattr(args, "disable_reference_quota", False))
    if use_quota:
        gen_count = math.ceil(float(args.reference_oversample_factor) * quota_target)
    else:
        gen_count = quota_target

    _cand_path = iter_dir / "candidate_tasks.jsonl"
    _accepted_path = iter_dir / "accepted_tasks.jsonl"
    _ref_fb_path = reference_feedback_path(iter_dir)
    _resume_disabled = os.environ.get("SELF_EVOLVE_DISABLE_RESUME", "0") == "1"
    _can_resume = (
        (not _resume_disabled)
        and _cand_path.is_file()
        and _accepted_path.is_file()
        and _ref_fb_path.is_file()
    )

    if _can_resume:
        candidate_tasks = read_jsonl(_cand_path)
        accepted_tasks = read_jsonl(_accepted_path)
        reference_feedback = read_jsonl(_ref_fb_path)
        gen_source = (
            candidate_tasks[0].get("generation_source", "clevr_spot_diff_generator")
            if candidate_tasks else "clevr_spot_diff_generator"
        )
        # Reconstruct only what downstream (solver stage + IterationState)
        # actually reads. Counts derive from the loaded lists; other fields
        # fill to 0/empty. mode="resumed_from_disk" + resumed=True make this
        # auditable in the run's iteration_state.json.
        ref = {
            "accepted_tasks": accepted_tasks,
            "reference_feedback": reference_feedback,
            "mode": "resumed_from_disk",
            "num_judged": len(reference_feedback),
            "num_accepted": len(accepted_tasks),
            "num_rejected": 0,
            "num_regenerated": 0,
            "num_rejected_final": 0,
            "reference_judged_tasks_count": len(reference_feedback),
            "unjudged_passthrough_tasks_count": 0,
            "accepted_tasks_count": len(accepted_tasks),
            "candidate_tasks_count": len(candidate_tasks),
            "reference_judge_budget_cap": 0,
            "dropped_candidate_count": 0,
            "drop_reason_histogram": {},
            "reference_accounting": {
                "quota_target": quota_target,
                "accepted": len(accepted_tasks),
                "judged": len(reference_feedback),
                "judge_budget": 0,
                "reject_count": 0,
                "num_regenerated": 0,
                "passthrough_count": 0,
                "quota_shortfall": 0,
                "budget_exhausted": False,
                "generator_exhausted": False,
                "reject_reason_histogram": {},
            },
            "resumed": True,
        }
        _prop_path = iter_dir / "proposals.jsonl"
        if args.self_play and _prop_path.is_file():
            proposals = read_jsonl(_prop_path)
        log(f"  [resume] iter_{iteration_index:03d}: reusing "
            f"candidate_tasks.jsonl ({len(candidate_tasks)}) + "
            f"accepted_tasks.jsonl ({len(accepted_tasks)}) + "
            f"reference_feedback.jsonl ({len(reference_feedback)}) from disk; "
            f"SKIPPING generator + Reference VLM (GPT-4o) stages")
    else:
        gen_cfg = PolicyCLEVRGeneratorConfig(
            dataset_root=args.dataset_root,
            iteration_id=iteration_id,
            num_tasks=gen_count,
            seed=args.seed,
            num_players=args.num_players,
            edit_fraction=args.edit_fraction,
            counterfactual_pairs=not args.no_counterfactual_pairs,
            label_balance=args.label_balance,
        )
        generator = PolicyControlledCLEVRGenerator(gen_cfg)

        # --- 3a. Proposer stage (self-play): the model picks the edits ---
        _edit_budget = int(gen_count * float(args.edit_fraction))
        if args.self_play and _edit_budget <= 0:
            log("  Proposer: the edit budget is 0, so no proposal could become a "
                "task; skipping the proposing side this round.")
        elif args.self_play:
            prev_solvability = (prev_failure_profile or {}).get("solvability") or {}
            _editing = policy.get("editing_policy", {}) or {}
            _retired = {s for s in _editing.get("retire_scene_ids", []) if s}
            seed_scenes = [s.get("scene_id") for s in _editing.get("seeds", [])
                           if s.get("scene_id") and s.get("scene_id") not in _retired]
            if not seed_scenes:
                # Round 0 has no seeds yet; propose over the pool instead so
                # self-play starts on iteration 0 rather than iteration 1.
                seed_scenes = generator._discover_base_names()
                random.Random(args.seed).shuffle(seed_scenes)
            proposer_stage = run_proposer_stage(
                generator,
                dry_run=args.dry_run_solver,
                scene_ids=[str(x) for x in seed_scenes],
                solvability=prev_solvability,
                solver_model_path=(prev_solver_state or {}).get("solver_model_path"),
                base_model_path=args.model_path,
                config=proposer_cfg,
                edit_budget=_edit_budget,
                seed=args.seed,
                solver_min_pixels=args.solver_min_pixels,
                solver_max_pixels=args.solver_max_pixels,
                num_gpus=(
                    int(os.environ.get("SELF_EVOLVE_SOLVER_NUM_GPUS", 0))
                    or int(getattr(args, "trainer_num_gpus", 1) or 1)
                ),
                log=log,
            )
            proposals = proposer_stage["proposals"]
            proposer_report = proposer_stage["report"]
            proposer_stage_source = proposer_stage["source"]
            write_jsonl(iter_dir / "proposals.jsonl", proposals)

        candidate_tasks = generator.generate(
            policy, prev_failure_profile, proposals=proposals or None
        )
        log(f"  Generator: {generator.last_generation_report}")
        write_jsonl(iter_dir / "candidate_tasks.jsonl", candidate_tasks)
        gen_source = (
            candidate_tasks[0].get("generation_source", "clevr_spot_diff_generator")
            if candidate_tasks else "clevr_spot_diff_generator"
        )
        log(f"  Generator: {len(candidate_tasks)} candidate task(s) "
            f"[source={gen_source}, "
            f"policy_v={policy.get('generator_policy_version')}, "
            f"quota_target={quota_target}, oversample_count={gen_count}, "
            f"quota_mode={use_quota}]")

        # --- 4. Reference VLM stage (MANDATORY module) ---
        if use_quota:
            ref = run_reference_stage(
                candidate_tasks,
                dry_run=args.dry_run_reference_vlm,
                provider=args.reference_provider,
                model=args.reference_model,
                base_url=args.reference_base_url,
                dataset_root=args.dataset_root,
                quota_target=quota_target,
                judge_budget_factor=float(args.reference_judge_budget_factor),
                solvability_threshold=float(args.reference_solvability_threshold),
                reject_on_ambiguity=bool(args.reference_reject_on_ambiguity),
                generator=generator,
                generator_policy=policy,
                failure_profile=prev_failure_profile,
                max_regenerate_attempts=args.max_regenerate_attempts,
                stream_dir=str(iter_dir),
                max_workers=int(os.environ.get("SELF_EVOLVE_REFERENCE_CONCURRENCY", 16)),
                log=log,
            )
        else:
            ref = _run_reference_stage_legacy(
                candidate_tasks,
                dry_run=args.dry_run_reference_vlm,
                provider=args.reference_provider,
                model=args.reference_model,
                base_url=args.reference_base_url,
                dataset_root=args.dataset_root,
                max_tasks=args.openai_reference_max_tasks,
                generator=generator,
                generator_policy=policy,
                failure_profile=prev_failure_profile,
                max_regenerate_attempts=args.max_regenerate_attempts,
                log=log,
            )
        write_jsonl(reference_feedback_path(iter_dir), ref["reference_feedback"])
        write_jsonl(iter_dir / "accepted_tasks.jsonl", ref["accepted_tasks"])
        log(f"  Reference VLM [{ref['mode']}]: judged={ref['num_judged']} "
            f"accepted={ref['num_accepted']} rejected={ref['num_rejected']} "
            f"regenerated={ref['num_regenerated']} rejected_final={ref['num_rejected_final']}")
    _acct = ref["reference_accounting"]
    log(f"  Reference accounting: quota_target={_acct['quota_target']}, "
        f"accepted={_acct['accepted']}, judged={_acct['judged']}/"
        f"{_acct['judge_budget']}, rejected={_acct['reject_count']}, "
        f"regenerated={_acct['num_regenerated']}, "
        f"passthrough={_acct['passthrough_count']}, "
        f"shortfall={_acct['quota_shortfall']}, "
        f"budget_exhausted={_acct['budget_exhausted']}, "
        f"generator_exhausted={_acct['generator_exhausted']}, "
        f"reject_reasons={_acct['reject_reason_histogram']}")

    # --- 5. Solver rollout (source-tagged) ---
    # Solver rollout GPUs: default to the trainer's GPU count so real rollout
    # runs data-parallel across all cards (one full model per card). Override
    # with SELF_EVOLVE_SOLVER_NUM_GPUS if you want a different split.
    _solver_num_gpus = int(
        os.environ.get("SELF_EVOLVE_SOLVER_NUM_GPUS", 0)
    ) or int(getattr(args, "trainer_num_gpus", 1) or 1)
    # Stage-level resume: raw_solver_trajectories.jsonl is written atomically
    # right after the (expensive, multi-hour GPU) rollout finishes. If it's on
    # disk, read it back instead of re-rolling. Downstream only needs the traj
    # list + rollout_source (recovered from the first record).
    _solver_traj_path = iter_dir / "raw_solver_trajectories.jsonl"
    _resume_solver = (
        os.environ.get("SELF_EVOLVE_DISABLE_RESUME", "0") != "1"
        and _solver_traj_path.is_file()
    )
    if _resume_solver:
        _trajs = read_jsonl(_solver_traj_path)
        _src = (_trajs[0].get("rollout_source") if _trajs else None) or (
            "dry_run_solver" if args.dry_run_solver else "real_solver"
        )
        _spath = (_trajs[0].get("solver_model_path") if _trajs else None)
        solver = {
            "trajectories": _trajs,
            "rollout_source": _src,
            "solver_model_path": _spath,
        }
        log(f"  [resume] iter_{iteration_index:03d}: reusing "
            f"raw_solver_trajectories.jsonl ({len(_trajs)} trajs) from disk; "
            f"SKIPPING solver rollout")
    else:
        solver = run_solver_stage(
            ref["accepted_tasks"],
            dry_run=args.dry_run_solver,
            num_generations=args.num_generations,
            solver_model_path=(prev_solver_state or {}).get("solver_model_path"),
            base_model_path=args.model_path,
            num_gpus=_solver_num_gpus,
            scratch_dir=str(iter_dir / "solver_rollout_shards"),
            solver_max_new_tokens=args.solver_max_new_tokens,
            solver_temperature=args.solver_temperature,
            solver_top_p=args.solver_top_p,
            solver_min_pixels=args.solver_min_pixels,
            solver_max_pixels=args.solver_max_pixels,
            log=log,
        )
        write_jsonl(iter_dir / "raw_solver_trajectories.jsonl", solver["trajectories"])

    # --- 6. Reward + failure tags + buffers ---
    # stream_path => each task's rows append to scored_trajectories.jsonl the
    # moment its (paid) GPT-4o answer-judge finishes, and a crashed rerun skips
    # already-judged task_ids instead of re-paying for them.
    _stream = iter_dir / "scored_trajectories.jsonl"
    if _resume_disabled and _stream.is_file():
        _stream.unlink()   # fresh run: never reuse judgments from old rollouts
    scored = score_and_route_trajectories(
        solver["trajectories"],
        include_details=True,
        reward_config=getattr(args, "_reward_config", None),
        dataset_root=args.dataset_root,
        stream_path=_stream,
    )
    cf_stats = counterfactual_sensitivity(scored, ref["accepted_tasks"])
    if cf_stats.get("num_pairs"):
        log(f"  Counterfactual: {cf_stats['num_pairs']} pairs, "
            f"sensitivity={cf_stats['counterfactual_sensitivity']}, "
            f"direction_correct={cf_stats['counterfactual_direction_correct']}")

    buffers = split_buffers(scored)
    buffer_dir = iter_dir / "buffers"
    for name, examples in buffers.items():
        write_jsonl(buffer_dir / f"{name}.jsonl", examples)
    buffer_counts = {k: len(v) for k, v in buffers.items()}

    # Aggregate failure profile for this iteration.
    from collections import Counter
    tag_counter: Counter = Counter()
    reward_sums: Dict[str, float] = {k: 0.0 for k in REWARD_KEYS}
    for ex in scored:
        rv = ex.get("reward_vector", {})
        for k in REWARD_KEYS:
            reward_sums[k] += float(rv.get(k, 0.0))
        if ex.get("buffer") == "failure":
            for t in assign_failure_tags(rv):
                if t != "positive":
                    tag_counter[t] += 1
    n = max(1, len(scored))
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _ex in scored:
        by_task[str(_ex.get("task_id"))].append(_ex)
    task_stats = task_statistics(by_task)
    solvability = summarize(task_stats)
    log(f"  Solvability: mean_solve_rate={solvability.get('mean_solve_rate')} "
        f"mean_regret={solvability.get('mean_regret')} "
        f"collapse={solvability.get('advantage_collapse_rate')} "
        f"seeds={len(solvability.get('seeds', []))}")

    # --- 6b. Close the self-play loop: pay the proposer what the solver did ---
    # A proposal is good exactly when the solver landed in the middle on the
    # task it produced, so the proposing side is scored by the solving side.
    scored_proposals: List[Dict[str, Any]] = []
    if args.self_play and proposals:
        from open_r1.self_evolve.proposer import (
            proposal_report,
            score_proposals as _score_proposals,
        )
        task_by_proposal = {
            str(t.get("proposal_id")): str(t.get("task_id"))
            for t in ref["accepted_tasks"] if t.get("proposal_id")
        }
        linked = []
        for proposal in proposals:
            record = dict(proposal)
            record["task_id"] = task_by_proposal.get(str(proposal.get("proposal_id")))
            linked.append(record)
        scored_proposals = _score_proposals(linked, task_stats)
        proposer_report = {**proposal_report(scored_proposals),
                           "source": proposer_stage_source}
        write_jsonl(iter_dir / "proposals_scored.jsonl", scored_proposals)
        log(f"  Proposer: {proposer_report}")

    failure_profile = build_failure_profile_from_counts(
        failure_tag_counts=dict(tag_counter),
        reward_means={k: reward_sums[k] / n for k in REWARD_KEYS},
        buffer_counts=buffer_counts,
        num_trajectories=len(scored),
        source="dry_run_solver" if args.dry_run_solver else "real_solver",
        solvability=solvability,
    )
    if cf_stats.get("num_pairs"):
        failure_profile["counterfactual"] = cf_stats
    if proposer_report:
        failure_profile["proposer"] = proposer_report
    save_failure_profile(iter_dir, failure_profile)
    log(f"  Reward/buffers: {buffer_counts}; "
        f"failure_tags={dict(tag_counter)}")

    # --- 7. Training exports (SFT/GRPO) ---
    export_summary = write_training_exports(
        iter_dir / "exports", ref["accepted_tasks"], scored,
        scored_proposals=scored_proposals,
        proposer_threshold=proposer_cfg.learnability_threshold,
    )
    log(f"  Exports: sft={export_summary['num_sft_replay']} "
        f"(solver={export_summary['num_sft_solver']}, "
        f"proposer={export_summary['num_sft_proposer']}) "
        f"grpo={export_summary['num_grpo_tasks']}")

    # --- 8. Trainer invocation state (MANDATORY pathways) ---
    trainer = run_trainer_stage(
        export_summary,
        dry_run=args.dry_run_trainer,
        prev_solver_model_path=(prev_solver_state or {}).get("solver_model_path"),
        base_model_path=args.model_path,
        iter_dir=iter_dir,
        iteration_index=iteration_index,
        max_steps=args.max_trainer_steps,
        execute_grpo=args.execute_grpo_smoke,
        execute_sft=args.execute_sft_smoke,
        use_lora=args.use_lora,
        carry_forward_enabled=not bool(args.disable_grpo_carry_forward),
        scale=getattr(args, "_trainer_scale", None),
        log=log,
    )

    # --- 8b. Stop the run when a mandatory trainer stage failed ---
    # A trainer that exited rc!=0 (crash: OOM, bad deps, DDP error, …) is
    # recorded as `trainer_failed` but does NOT raise on its own. Without this
    # gate the loop would still write iteration_state.json below, mark the iter
    # "complete", and roll into the next iteration — re-running generator /
    # Reference-VLM / solver / answer-judge (all paid GPT-4o + hours of GPU) on
    # top of a model that never actually trained. That is the exact silent-waste
    # defect. So: if any pathway crashed, abort the whole run right here,
    # BEFORE any state is persisted, so a plain re-run resumes at this same iter.
    #
    # Distinction that matters: `trainer_failed` (rc!=0, a real crash) aborts;
    # `mandatory_but_blocked` (empty export — a data condition, e.g. no positive
    # trajectory this round) does NOT, that path is handled downstream as before.
    # Escape hatch: SELF_EVOLVE_ALLOW_TRAINER_FAILURE=1 downgrades to a warning.
    # Only MANDATORY pathways abort the run.
    _failed = [
        name for name, pw in trainer["training_pathways"].items()
        if pw.get("status") == "trainer_failed" and pw.get("mandatory", True)
    ]
    if _failed:
        _reasons = "; ".join(
            f"{name}: {trainer['training_pathways'][name].get('blocked_reason', 'rc!=0')}"
            for name in _failed
        )
        if os.environ.get("SELF_EVOLVE_ALLOW_TRAINER_FAILURE", "0") == "1":
            log(f"  Trainer: WARNING — mandatory trainer(s) failed [{', '.join(_failed)}] "
                f"but SELF_EVOLVE_ALLOW_TRAINER_FAILURE=1 set; continuing anyway. "
                f"Reasons: {_reasons}")
        else:
            raise RuntimeError(
                f"Mandatory trainer(s) failed this iteration: {', '.join(_failed)}. "
                f"Reasons: {_reasons}. Aborting the loop BEFORE writing "
                f"iteration_state.json so no partial iteration is marked complete "
                f"(a plain re-run will resume at iter_{iteration_index:03d} and reuse "
                f"the generator/Reference-VLM/solver artefacts already on disk). "
                f"Fix the trainer error above, then re-run. To override and continue "
                f"despite the failure, set SELF_EVOLVE_ALLOW_TRAINER_FAILURE=1."
            )

    # --- 9. Solver update state (carries solver_model_path forward) ---
    solver_update_state = {
        "iteration_id": iteration_id,
        "trainer_mode": trainer["trainer_mode"],
        "solver_model_path": trainer["solver_model_path"],
        "solver_model_path_status": trainer["solver_model_path_status"],
        "prev_solver_model_path": (prev_solver_state or {}).get("solver_model_path"),
        "checkpoint_created": trainer["checkpoint_created"],
        "training_pathways": trainer["training_pathways"],
        "solver_lineage": trainer["solver_lineage"],
        "rollout_source": solver["rollout_source"],
    }
    save_solver_update_state(iter_dir, solver_update_state)

    # --- 10. Top-level iteration state ---
    state = IterationState(
        iteration_index=iteration_index,
        iteration_id=iteration_id,
        output_dir=str(iter_dir),
        prev_iteration_dir=str(prev_iteration_dir) if prev_iteration_dir else None,
        read_prev_state=read_prev,
        generator_policy_version=int(policy.get("generator_policy_version", 0)),
        generator_policy_changes_from_prev=policy_changes,
        num_candidate_tasks=len(candidate_tasks),
        generator_source=gen_source,
        reference_provider=args.reference_provider,
        reference_model=args.reference_model,
        reference_base_url=args.reference_base_url,
        reference_mode=ref["mode"],
        num_reference_judged=ref["reference_judged_tasks_count"],
        num_accepted=ref["num_accepted"],
        num_rejected=ref["num_rejected"],
        num_regenerated=ref["num_regenerated"],
        num_rejected_final=ref["num_rejected_final"],
        reference_accounting={
            "candidate_tasks_count": ref["candidate_tasks_count"],
            "reference_judged_tasks_count": ref["reference_judged_tasks_count"],
            "unjudged_passthrough_tasks_count": ref["unjudged_passthrough_tasks_count"],
            "accepted_tasks_count": ref["accepted_tasks_count"],
            "num_train_tasks_requested": int(args.num_train_tasks),
            "num_train_tasks_effective": ref["accepted_tasks_count"],
            "reference_judge_budget_cap": ref["reference_judge_budget_cap"],
            "dropped_candidate_count": ref["dropped_candidate_count"],
            "drop_reason_histogram": ref["drop_reason_histogram"],
        },
        solver_rollout_source=solver["rollout_source"],
        num_solver_trajectories=len(solver["trajectories"]),
        solver_model_path=trainer["solver_model_path"],
        solver_lineage=trainer["solver_lineage"],
        buffer_counts=buffer_counts,
        trainer_mode=trainer["trainer_mode"],
        training_pathways=trainer["training_pathways"],
        notes=[
            "The active training recipe is SFT→GRPO when SFT is enabled, "
            "otherwise GRPO-only.",
            "rollout_source and reference_mode reflect the ACTUAL path used.",
        ],
    )
    state.save()
    log(f"  State saved: {iter_dir}/iteration_state.json")
    return iter_dir


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop self-evolving VLM driver (multi-iteration; "
                    "SFT→GRPO or GRPO-only)."
    )
    # NOTE on defaults: every setting below defaults to None (or const=True /
    # default=None for boolean flags) so "given on the CLI" is distinguishable
    # from "not given". Final values are resolved as explicit CLI > --config
    # YAML > hard default (see experiment_config.resolve_settings). The hard
    # defaults live in experiment_config.HARD_DEFAULTS, not here.

    # Experiment config (YAML). Groups all settings below; CLI flags override it.

    # Data / model
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model-path", default=None,
                        help="Base solver model path (iteration-0 solver_model_path).")
    parser.add_argument("--output-dir", default=None,
                        help="Root output dir; each iteration writes iter_XXX/ under "
                             "it. Point this at a large disk; checkpoints fill a "
                             "small system disk quickly.")

    # Iteration control (HARD GUARD)
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Closed-loop iterations to run (HARD-CAPPED at 2).")
    parser.add_argument("--allow-more-than-two-iterations", action="store_const",
                        const=True, default=None,
                        help="Escape hatch for >2 iterations. DO NOT use this round.")

    # Task config
    parser.add_argument("--num-train-tasks", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None,
                        help="Solver trajectories sampled per accepted task.")
    parser.add_argument("--seed", type=int, default=None)

    # Reference VLM (MANDATORY module) — unified param names
    parser.add_argument("--reference-provider", default=None,
                        choices=["openrouter", "openai"],
                        help="Reference VLM provider (default: openrouter).")
    parser.add_argument("--reference-model", default=None,
                        help="Reference VLM model slug (default: openai/gpt-4o).")
    parser.add_argument("--reference-base-url", default=None,
                        help="Reference VLM base URL (default: provider default).")
    parser.add_argument("--openai-model", dest="reference_model_alias", default=None,
                        help="DEPRECATED alias for --reference-model (kept for "
                             "backward compatibility).")
    parser.add_argument("--enable-openai-reference-vlm", action="store_const",
                        const=True, default=None,
                        help="Enable LIVE Reference VLM API calls (else dry-run).")
    parser.add_argument("--dry-run-reference-vlm", action="store_const",
                        const=True, default=None,
                        help="Reference VLM uses the no-API structured judge.")
    parser.add_argument("--openai-reference-max-tasks", type=int, default=None,
                        help="DEPRECATED (judge-until-quota uses "
                             "--reference-judge-budget-factor). Budget cap only "
                             "honoured under --disable-reference-quota.")
    parser.add_argument("--max-regenerate-attempts", type=int, default=None,
                        help="Max regenerate rounds for rejected tasks before "
                             "marking reference_rejected_final.")
    # Judge-until-quota knobs (coefficients of num_train_tasks N).
    parser.add_argument("--reference-oversample-factor", type=float, default=None,
                        help="First-batch candidate count = ceil(factor * N) "
                             "(default 1.25).")
    parser.add_argument("--reference-judge-budget-factor", type=float, default=None,
                        help="Judge-call ceiling = ceil(factor * N) (default 1.5).")
    parser.add_argument("--reference-solvability-threshold", type=float, default=None,
                        help="Lenient solvability floor tau for accept (default 0.4).")
    parser.add_argument("--reference-reject-on-ambiguity", action="store_const",
                        const=True, default=None,
                        help="Reject visually ambiguous candidates (default OFF; "
                             "CLEVR spot-diff is inherently ambiguous).")
    parser.add_argument("--disable-reference-quota", action="store_const",
                        const=True, default=None,
                        help="Ablation: revert to the old budget-cap passthrough "
                             "path instead of judge-until-quota.")

    # Solver
    parser.add_argument("--dry-run-solver", action="store_const",
                        const=True, default=None,
                        help="Solver builds prompts but loads NO model "
                             "(diagnostic completions; rollout_source=dry_run_solver).")

    # Trainer
    parser.add_argument("--dry-run-trainer", action="store_const",
                        const=True, default=None,
                        help="Do not launch any trainer; construct + record the real "
                             "commands only.")
    parser.add_argument("--max-trainer-steps", type=int, default=None,
                        help="Max trainer steps for the GRPO/SFT smoke command (default 1).")
    parser.add_argument("--execute-grpo-smoke", action="store_const",
                        const=True, default=None,
                        help="Actually LAUNCH the GRPO smoke (GPU). Requires NOT "
                             "--dry-run-trainer. Gated; use only after confirmation.")
    parser.add_argument("--execute-sft-smoke", action="store_const",
                        const=True, default=None,
                        help="Actually LAUNCH the SFT smoke (GPU, sft_jsonl.py). "
                             "Requires NOT --dry-run-trainer. Gated.")
    parser.add_argument("--use-lora", action="store_const",
                        const=True, default=None,
                        help="Use PEFT/LoRA for the GRPO smoke (verified to run a real "
                             "step on a single 24G GPU). Default is full-parameter "
                             "(interface preserved; needs ZeRO-3/offload or more VRAM).")

    # Trainer scale (multi-GPU / batch / generations / deepspeed / LoRA hparams).
    # All default to None so YAML/hard-defaults fill them; CLI overrides both.
    parser.add_argument("--trainer-num-gpus", type=int, default=None,
                        help="GPUs for the trainer (torchrun --nproc_per_node). "
                             "Sets CUDA_VISIBLE_DEVICES to 0..N-1 unless pinned.")
    parser.add_argument("--trainer-per-device-train-batch-size", type=int, default=None,
                        help="GRPO per-device train batch size.")
    parser.add_argument("--trainer-gradient-accumulation-steps", type=int, default=None,
                        help="Gradient accumulation steps (GRPO/SFT).")
    parser.add_argument("--trainer-grpo-num-generations", type=int, default=None,
                        help="GRPO generations per prompt. Global batch "
                             "(per_device*gpus*grad_accum) must be divisible by this. "
                             "Real GRPO must be >= 4 (recommended 6); below that "
                             "requires --grpo-probe.")
    parser.add_argument("--grpo-probe", action="store_true", default=False,
                        help="Allow a sub-threshold GRPO group size (e.g. "
                             "num_generations=2) for API wiring / dry-run ONLY. "
                             "Flagged in logs as NOT a valid GRPO learning config.")
    parser.add_argument("--disable-grpo-carry-forward", action="store_true",
                        default=False,
                        help="Restore the OLD behaviour: GRPO trains from the "
                             "original base each iteration (no LoRA merge / no "
                             "cross-round weight accumulation). Default OFF, i.e. "
                             "carry-forward is enabled and iter>0 GRPO trains from "
                             "the merged accumulated solver.")
    parser.add_argument("--trainer-deepspeed-config", default=None,
                        help="Path to a DeepSpeed JSON (e.g. "
                             "local_scripts/zero3.json).")
    parser.add_argument("--trainer-backend", default=None,
                        choices=["deepspeed", "fsdp2"],
                        help="Distributed backend for the GRPO route only "
                             "(default: deepspeed). 'fsdp2' uses PyTorch native "
                             "FSDP2 via --trainer-fsdp-config; SFT always uses "
                             "deepspeed.")
    parser.add_argument("--trainer-fsdp-config", default=None,
                        help="Path to an HF --fsdp_config JSON (e.g. "
                             "local_scripts/fsdp2_qwen2_5vl.json). Required when "
                             "--trainer-backend=fsdp2.")
    parser.add_argument("--trainer-lora-r", type=int, default=None)
    parser.add_argument("--trainer-lora-alpha", type=int, default=None)
    parser.add_argument("--trainer-lora-dropout", type=float, default=None)

    # Export toggles (exports always run; flags kept for explicitness/compat)

    # Reward scalarization config (the one place weights +
    # routing thresholds). Weights are adjusted ONLY by editing this file;
    # there are intentionally no per-dimension CLI weight flags and no
    # --reward-scheme. Omitting this uses the default config; a missing /
    # invalid config fails fast (no equal-weight fallback).
    # Solver rollout decoding. Explicit rather than implicit defaults
    # (256 new tokens, temperature 0.2): 256 tokens truncates a 5-image
    # comparison before the evidence boxes are emitted, and temperature 0.2
    # makes the rollout group nearly identical, which flattens the GRPO
    # group-relative advantage.
    parser.add_argument("--solver-max-new-tokens", type=int, default=1024)
    parser.add_argument("--solver-temperature", type=float, default=1.0)
    parser.add_argument("--solver-top-p", type=float, default=0.95)
    # Must match the trainer's --min_pixels/--max_pixels, or the buffers are
    # scored at a different resolution than the policy is trained at.
    # Share of each round built by editing the previous round's highest-regret
    # scenes; the rest is sampled fresh. 0 falls back to fresh sampling only.
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--edit-fraction", type=float, default=0.5)
    parser.add_argument("--no-counterfactual-pairs", action="store_true", default=False)
    parser.add_argument("--label-balance", choices=["uniform", "none"], default="uniform")

    # --- Self-play: the model proposes the edits instead of a heuristic ---
    parser.add_argument("--self-play", action="store_true", default=False,
                        help="The model picks which changes to keep, and is trained "
                             "on the picks the solver found learnable.")
    parser.add_argument("--proposer-group-size", type=int, default=4,
                        help="Proposals sampled per scene. Each becomes one task "
                             "slot, so the solver budget is unchanged.")
    parser.add_argument("--proposer-max-scenes", type=int, default=8,
                        help="Scenes to propose over per round.")
    parser.add_argument("--proposer-temperature", type=float, default=1.0)
    parser.add_argument("--proposer-max-new-tokens", type=int, default=512)
    parser.add_argument("--proposer-min-players", type=int, default=3)
    parser.add_argument("--proposer-max-players", type=int, default=8,
                        help="The proposer's second axis: more players means "
                             "more images to compare. The spy slot itself stays "
                             "random — it is the answer.")
    parser.add_argument("--proposer-no-competence", action="store_true", default=False,
                        help="Do not tell the proposer the solver's current solve "
                             "rate (ablation).")
    parser.add_argument("--proposer-learnability-threshold", type=float, default=0.5,
                        help="Lowest realized learnability a proposal needs to be "
                             "trained on. 0.5 means a pass rate near 0.15-0.85.")

    parser.add_argument("--solver-min-pixels", type=int, default=None)
    parser.add_argument("--solver-max-pixels", type=int, default=None)

    parser.add_argument(
        "--reward-config",
        default=None,
        help="Path to reward weights JSON "
             "(default: configs/reward/reward_weights.json).",
    )

    args = parser.parse_args()

    # Resolve settings: explicit CLI > --config YAML > hard default. Every
    # schema-covered argparse default is None (sentinel for "not given"), so
    # this is where the real values are filled in.
    cli_values = {dest: getattr(args, dest, None) for dest in HARD_DEFAULTS}
    resolved = resolve_settings(cli_values)
    for dest, value in resolved.items():
        setattr(args, dest, value)

    for dest, flag in (("dataset_root", "--dataset-root"),
                       ("model_path", "--model-path"),
                       ("output_dir", "--output-dir")):
        if not getattr(args, dest, None):
            raise SystemExit(f"[ERROR] {flag} is required (no default; set it in paths.sh)")

    # Load .env (gitignored) if present — shell env always wins (override=False).
    try:
        from open_r1.self_evolve.dotenv_loader import load_dotenv
        loaded = load_dotenv()
        if loaded:
            print(f"[dotenv] loaded keys from .env: {sorted(loaded)}")
    except Exception:
        pass

    # Backward-compat alias resolution (only when --reference-model not given).
    if args.reference_model_alias and args.reference_model == "openai/gpt-4o":
        args.reference_model = args.reference_model_alias
        print(f"[compat] --openai-model is deprecated; using --reference-model="
              f"{args.reference_model}")

    # Provider base URL defaulting.
    if args.reference_base_url is None:
        args.reference_base_url = (
            OPENROUTER_BASE_URL if args.reference_provider == "openrouter" else OPENAI_BASE_URL
        )

    # Reference dry-run resolution: dry-run unless explicitly enabled live.
    if not args.dry_run_reference_vlm and not args.enable_openai_reference_vlm:
        args.dry_run_reference_vlm = True

    # HARD GUARD: never run more than 2 closed-loop iterations this round.
    if args.num_iterations > 2 and not args.allow_more_than_two_iterations:
        raise SystemExit(
            f"--num-iterations={args.num_iterations} exceeds the hard cap of 2. "
            "Pass --allow-more-than-two-iterations to override (NOT this round)."
        )
    if args.num_iterations < 1:
        raise SystemExit("--num-iterations must be >= 1.")

    # Load and validate the reward config up front. Export the path so
    # the GRPO live reward (subprocess) reads the SAME config. No equal-weight
    # fallback: an invalid/missing config raises here.
    reward_config = load_reward_config(args.reward_config)
    os.environ["SELF_EVOLVE_REWARD_CONFIG"] = reward_config.path
    if getattr(args, "reference_provider", None):
        os.environ.setdefault("SELF_EVOLVE_REFERENCE_PROVIDER", args.reference_provider)
    args._reward_config = reward_config

    # Export the base solver model so the online_solver module
    # default to THIS run's --model-path instead of a hardcoded machine path.
    if args.model_path:
        os.environ["SELF_EVOLVE_BASE_MODEL"] = args.model_path

    # Build + validate the trainer scale (multi-GPU / batch / generations).
    # validate() fails fast on an invalid GRPO global-batch / num_generations
    # ratio or a missing deepspeed config.
    # A sub-threshold GRPO group (e.g. num_generations=2) is permitted ONLY as
    # an explicit probe or in dry-run (no real training launched). Real training
    # must use >= 4 (recommended 6).
    grpo_is_probe = bool(args.grpo_probe) or bool(args.dry_run_trainer)
    trainer_scale = TrainerScale(
        num_gpus=args.trainer_num_gpus,
        per_device_train_batch_size=args.trainer_per_device_train_batch_size,
        gradient_accumulation_steps=args.trainer_gradient_accumulation_steps,
        grpo_num_generations=args.trainer_grpo_num_generations,
        deepspeed_config=args.trainer_deepspeed_config,
        trainer_backend=args.trainer_backend,
        fsdp_config=args.trainer_fsdp_config,
        lora_r=args.trainer_lora_r,
        lora_alpha=args.trainer_lora_alpha,
        lora_dropout=args.trainer_lora_dropout,
        is_probe=grpo_is_probe,
    )
    trainer_scale.validate()
    args._trainer_scale = trainer_scale

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"{'='*64}\n  CLOSED-LOOP SELF-EVOLVING VLM DRIVER\n{'='*64}")
    log(f"  Iterations:        {args.num_iterations}")
    log(f"  Dataset root:      {args.dataset_root}")
    log(f"  Reference VLM:     provider={args.reference_provider} "
        f"model={args.reference_model} base_url={args.reference_base_url} "
        f"mode={'DRY-RUN' if args.dry_run_reference_vlm else 'LIVE'}")
    log(f"  Solver:            {'DRY-RUN' if args.dry_run_solver else 'REAL (GPU)'}")
    log(f"  Trainer:           {'DRY-RUN' if args.dry_run_trainer else 'LIVE'} "
        f"(gpus={trainer_scale.num_gpus}, per_device_bs={trainer_scale.per_device_train_batch_size}, "
        f"grad_accum={trainer_scale.gradient_accumulation_steps}, "
        f"grpo_gen={trainer_scale.grpo_num_generations}, "
        f"deepspeed={trainer_scale.deepspeed_config or 'none'}, lora={bool(args.use_lora)})")
    _scale_audit = trainer_scale.scale_audit()
    log(f"  GRPO group:        num_generations_requested={_scale_audit['num_generations_requested']}, "
        f"num_generations_effective={_scale_audit['num_generations_effective']}, "
        f"grpo_group_size={_scale_audit['grpo_group_size']}, "
        f"global_batch={_scale_audit['grpo_global_batch']}, "
        f"probe={_scale_audit['is_probe']}")
    if _scale_audit["group_advantage_warning"]:
        log(f"  [WARN] {_scale_audit['group_advantage_warning']}")
        if _scale_audit["is_probe"]:
            log("  [WARN] num_generations is for API wiring / dry-run probe ONLY "
                "— NOT a valid GRPO learning config.")
    log(f"  Reward config:     {reward_config.path}")
    log(f"  Reward weights:    {reward_config.weights} "
        f"(scalarization={reward_config.scalarization}, v={reward_config.version})")
    log(f"  Output root:       {output_root}")
    _recipe_stages = (
        (["SFT"] if args.execute_sft_smoke else [])
        + (["GRPO"] if args.execute_grpo_smoke else [])
    )
    log(f"  Training recipe:   "
        f"{' -> '.join(_recipe_stages) if _recipe_stages else 'none (dry-run / no --execute-* flag)'}")
    log(f"{'='*64}")

    # ---- Iteration-level resume ----------------------------------------
    # A finished iteration writes iter_NNN/iteration_state.json as its LAST step
    # (stage 10). So an iter with that file is complete; scan from iter_000
    # upward, skip every complete iter (feeding it forward as prev_dir), and
    # re-enter at the first incomplete/missing one. run_one_iteration then does
    # stage-level resume within that iter (skips generator+GPT-4o / solver whose
    # artefacts already landed). Set SELF_EVOLVE_DISABLE_RESUME=1 to force fresh.
    resume_disabled = os.environ.get("SELF_EVOLVE_DISABLE_RESUME", "0") == "1"
    prev_dir: Optional[Path] = None
    start_index = 0
    if not resume_disabled:
        for i in range(args.num_iterations):
            cand = output_root / f"iter_{i:03d}"
            if (cand / "iteration_state.json").is_file():
                prev_dir = cand
                start_index = i + 1
                log(f"  [resume] iter_{i:03d} already complete "
                    f"(iteration_state.json present) — skipping")
            else:
                break
        if start_index >= args.num_iterations:
            log(f"  [resume] all {args.num_iterations} iteration(s) already "
                f"complete under {output_root}; nothing to do")
        elif start_index > 0 or (output_root / f"iter_{start_index:03d}").exists():
            log(f"  [resume] re-entering at iter_{start_index:03d} "
                f"(prev_dir={prev_dir.name if prev_dir else None})")

    for i in range(start_index, args.num_iterations):
        prev_dir = run_one_iteration(
            iteration_index=i,
            output_root=output_root,
            prev_iteration_dir=prev_dir,
            args=args,
            log=log,
        )

    log(f"\n{'='*64}\n  CLOSED LOOP COMPLETE — {args.num_iterations} iteration(s)\n{'='*64}")
    log("  Verify: iter_001/iteration_state.json should show read_prev_state=true,")
    log("  a bumped generator_policy_version, and policy changes from iter_000.")
    sys.exit(0)


if __name__ == "__main__":
    main()
