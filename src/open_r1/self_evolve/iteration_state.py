"""Iteration state machine for the closed-loop self-evolving VLM agent.

This module is what makes the pipeline a closed loop rather than a single pass.
Each iteration N writes a set of state files; iteration N+1 reads them and
changes its behaviour accordingly:

    iteration_<id>/
        iteration_state.json        — top-level state for the iteration
        generator_policy.json       — sampling / difficulty / image / rule policy
        failure_profile.json        — aggregated failure tags + reward means
        solver_update_state.json    — solver model path + trainer pathway state
        reference_feedback.jsonl    — per-task Reference VLM feedback (accept/reject)

The loop driver (``run_real_input_self_evolve_loop.py``) consumes these. The
contract that proves the loop is real:

    generator_policy(N+1) = update(generator_policy(N), failure_profile(N),
                                    reference_summary(N))
    solver_model_path(N+1) = solver_update_state(N).solver_model_path

Nothing here calls an API, loads a model, or trains. It is pure schema + IO +
policy-update logic so it can be unit-tested and dry-run cheaply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_r1.self_evolve.io import read_json, write_json


# ===========================================================================
# Generator policy: sampling / difficulty / image / rule
# ===========================================================================

# Failure-tag → generator focus mapping.
TAG_TO_FOCUS = {
    "shortcut_or_ungrounded_success": "visual_evidence_grounding",
    "grounding_failure": "visual_evidence_grounding",
    "process_failure": "multi_step_reasoning_structure",
    "budget_failure": "reasoning_budget_control",
    "consistency_failure": "answer_consistency_control",
    "reasoning_or_final_decision_failure": "final_decision_reasoning",
}

_DIFFICULTY_BALANCED = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

# Outside this pass-rate band the generator moves difficulty; inside it,
# per-task regret decides which scenes get edited.
LEARNABILITY_BAND = (0.35, 0.65)


def default_generator_policy() -> Dict[str, Any]:
    """Iteration-0 generator policy (no prior feedback)."""
    return {
        "generator_policy_version": 0,
        "difficulty_policy": {
            "distribution": dict(_DIFFICULTY_BALANCED),
            "difficulty_target": "medium",
        },
        "sampling_policy": {
            # Per-focus sampling weights — uniform at the start.
            "focus_weights": {},
        },
        "image_selection_policy": {
            # The seed offset shifts the sampling window each iteration so N+1
            # does not reuse the exact N images.
            "seed_offset": 0,
        },
        "policy_update_reason": "default_initial_policy",
        "derived_from": "default_initial_policy",
    }


def _shift_distribution_easier(dist: Dict[str, float], delta: float = 0.15) -> Dict[str, float]:
    """Move probability mass from ``hard`` toward ``medium``/``easy``.

    Used when the Reference VLM / feedback says tasks were too difficult. The
    shift is bounded so a single round cannot wipe out hard entirely.
    """
    d = {k: float(dist.get(k, 0.0)) for k in ("easy", "medium", "hard")}
    move = min(d["hard"], delta)
    d["hard"] -= move
    d["medium"] += move * 0.5
    d["easy"] += move * 0.5
    total = sum(d.values()) or 1.0
    return {k: round(v / total, 4) for k, v in d.items()}


def _shift_distribution_harder(dist: Dict[str, float], delta: float = 0.1) -> Dict[str, float]:
    """Move probability mass toward ``hard`` (tasks too easy)."""
    d = {k: float(dist.get(k, 0.0)) for k in ("easy", "medium", "hard")}
    move = min(d["easy"], delta)
    d["easy"] -= move
    d["hard"] += move
    total = sum(d.values()) or 1.0
    return {k: round(v / total, 4) for k, v in d.items()}


def update_generator_policy(
    prev_policy: Optional[Dict[str, Any]],
    failure_profile: Dict[str, Any],
    reference_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive the next-iteration generator policy from REAL feedback.

    This is the mechanism that makes iteration N+1 *behave differently* from N.
    The difficulty knob is bidirectional. Priority order:

      1. Explicit reduce signal from the Reference VLM — a too-hard reject, an
         explicit ``reduce_difficulty`` tag, or a suggestion strictly easier than
         the task's own target → shift the distribution easier. A target *echo*
         (suggested == target on an accepted task) is NOT a reduce signal and
         never triggers this branch.
      2. Solvability feedback — move difficulty so the solver's pass rate lands
         inside ``LEARNABILITY_BAND`` (the band where a GRPO group carries the
         most gradient). This is the primary knob once rollouts exist.
      3. No solvability signal yet → fall back to the mean answer reward.
      4. No actionable signal → keep the previous distribution.

    Grounding/process weakness feeds the FOCUS weights, never the difficulty:
    a bbox-format failure is not a reason to make the visual task harder.

    Every update records a human-readable ``policy_update_reason`` and a full
    ``feedback_inputs`` block so the change is auditable. Pure function — returns
    a NEW dict; does not mutate inputs.
    """
    prev = prev_policy or default_generator_policy()
    prev_version = int(prev.get("generator_policy_version", 0))
    prev_dist = (prev.get("difficulty_policy", {}) or {}).get(
        "distribution", dict(_DIFFICULTY_BALANCED)
    )

    tag_counts: Dict[str, int] = dict(failure_profile.get("failure_tag_counts", {}))
    reward_means: Dict[str, float] = dict(failure_profile.get("reward_means", {}))
    buffer_stats: Dict[str, int] = dict(failure_profile.get("buffer_counts", {}))

    ref = reference_summary or {}
    reject_counts: Dict[str, int] = dict(ref.get("reject_reason_counts", {}) or {})
    suggested_hist_all: Dict[str, int] = dict(
        ref.get("suggested_difficulty_histogram_all",
                ref.get("suggested_difficulty_histogram", {})) or {}
    )
    target_hist: Dict[str, int] = dict(ref.get("difficulty_target_histogram", {}) or {})
    gap_hist: Dict[str, int] = dict(ref.get("difficulty_gap_histogram", {}) or {})

    # Only an explicit, echo-free signal may trigger an easier shift; a
    # medium-target echo lands in ``target_echo_count`` and is excluded.
    reduce_difficulty_count = int(ref.get("reduce_difficulty_count", 0))
    strict_easier_count = int(ref.get("strict_easier_suggestion_count", 0))
    target_echo_count = int(ref.get("target_echo_count", 0))
    too_difficult_count = int(
        ref.get("too_difficult_count",
                reject_counts.get("too_difficult", 0)
                + reject_counts.get("too_hard_for_difficulty_target", 0))
    )

    reasons: List[str] = []

    # --- difficulty regime ---
    grounding = float(reward_means.get("grounding", 1.0))
    process = float(reward_means.get("process", 1.0))
    answer = float(reward_means.get("answer", 1.0))
    solvability = dict(failure_profile.get("solvability") or {})
    solve_rate = solvability.get("mean_solve_rate")
    solve_rate = float(solve_rate) if solve_rate is not None else None

    # Only an EXPLICIT reduce signal counts as "suggests easier". Target echoes
    # are ignored here on purpose.
    suggests_easier = (
        reduce_difficulty_count > 0
        or strict_easier_count > 0
        or too_difficult_count > 0
    )

    # Priority 1: Reference VLM explicitly asked to reduce → REDUCE hard.
    if suggests_easier:
        distribution = _shift_distribution_easier(prev_dist)
        difficulty_target = "medium" if distribution["hard"] > 0.05 else "easy"
        if too_difficult_count > 0:
            reasons.append(
                f"reduce_hard_due_to_reference_too_difficult(count={too_difficult_count})"
            )
        if reduce_difficulty_count > 0 and too_difficult_count == 0:
            reasons.append(
                f"reduce_hard_due_to_explicit_reduce_signal(count={reduce_difficulty_count})"
            )
        if strict_easier_count > 0:
            reasons.append(
                f"reduce_hard_due_to_strict_easier_suggestion(count={strict_easier_count})"
            )
    # Priority 2: steer toward the learnable frontier, not toward "as hard as
    # possible" — group spread, and so the gradient, peaks near a 0.5 solve rate.
    elif solve_rate is not None:
        if solve_rate > LEARNABILITY_BAND[1]:
            distribution = _shift_distribution_harder(prev_dist)
            difficulty_target = "hard"
            reasons.append(f"increase_difficulty_solve_rate_above_band({solve_rate:.2f})")
        elif solve_rate < LEARNABILITY_BAND[0]:
            distribution = _shift_distribution_easier(prev_dist)
            difficulty_target = "medium"
            reasons.append(f"reduce_difficulty_solve_rate_below_band({solve_rate:.2f})")
        else:
            distribution = {k: round(float(prev_dist.get(k, _DIFFICULTY_BALANCED[k])), 4)
                            for k in ("easy", "medium", "hard")}
            difficulty_target = (prev.get("difficulty_policy", {}) or {}).get(
                "difficulty_target", "medium"
            )
            reasons.append(f"hold_difficulty_solve_rate_in_band({solve_rate:.2f})")
    # Priority 3: no solvability signal yet, fall back to the reward means.
    # Grounding/process weakness only raises the focus weights, never difficulty.
    elif answer >= 0.8:
        distribution = _shift_distribution_harder(prev_dist)
        difficulty_target = "hard"
        reasons.append("increase_hard_due_to_strong_answer_accuracy")
    elif answer <= 0.2:
        distribution = _shift_distribution_easier(prev_dist)
        difficulty_target = "medium"
        reasons.append("reduce_difficulty_due_to_low_answer_accuracy")
    else:
        distribution = {k: round(float(prev_dist.get(k, _DIFFICULTY_BALANCED[k])), 4)
                        for k in ("easy", "medium", "hard")}
        reasons.append("no_difficulty_signal_keep_previous")
        difficulty_target = (prev.get("difficulty_policy", {}) or {}).get(
            "difficulty_target", "medium"
        )

    # Note in the audit trail when echoes were seen but correctly ignored.
    if target_echo_count > 0 and not suggests_easier:
        reasons.append(
            f"ignored_target_echo_suggestions(count={target_echo_count})"
        )

    # --- focus weights from failure tags ---
    focus_weights: Dict[str, float] = {}
    total = sum(tag_counts.values())
    if total > 0:
        for tag, count in tag_counts.items():
            focus = TAG_TO_FOCUS.get(tag)
            if focus:
                focus_weights[focus] = round(
                    focus_weights.get(focus, 0.0) + count / total, 4
                )

    # --- grounding / bbox requirement ---
    grounding_failures = int(tag_counts.get("grounding_failure", 0)) + int(
        tag_counts.get("shortcut_or_ungrounded_success", 0)
    )
    if grounding_failures > 0 or grounding < 0.5:
        reasons.append("increase_grounding_focus_due_to_grounding_failure")
    if "process_failure" in tag_counts and tag_counts["process_failure"] > 0:
        reasons.append("increase_process_focus_due_to_process_failure")

    # --- minimal-update bookkeeping when feedback is empty ---
    has_feedback = bool(
        tag_counts or reward_means or reject_counts or suggested_hist_all
        or reduce_difficulty_count or strict_easier_count or too_difficult_count
    )
    if not has_feedback:
        reasons = ["no_feedback_keep_previous_or_minimal_decay"]

    # --- image window advance (so N+1 does not reuse the same images) ---
    prev_img = prev.get("image_selection_policy", {}) or {}
    new_seed_offset = int(prev_img.get("seed_offset", 0)) + 1

    policy = {
        "generator_policy_version": prev_version + 1,
        "difficulty_policy": {
            "distribution": distribution,
            "difficulty_target": difficulty_target,
        },
        "sampling_policy": {"focus_weights": focus_weights},
        "solvability_feedback": {
            "mean_solve_rate": solve_rate,
            "mean_regret": solvability.get("mean_regret"),
            "advantage_collapse_rate": solvability.get("advantage_collapse_rate"),
            "learnability_band": list(LEARNABILITY_BAND),
            "class_counts": solvability.get("class_counts", {}),
        },
        # What the next round edits: highest-regret seeds, too_hard ones made
        # easier, mastered ones dropped.
        "editing_policy": {
            "seeds": list(solvability.get("seeds", []))
            + list(solvability.get("too_hard", [])),
            "retire_scene_ids": [r.get("scene_id") for r in solvability.get("retire", [])
                                 if r.get("scene_id")],
        },
        "image_selection_policy": {"seed_offset": new_seed_offset},
        "policy_update_reason": "; ".join(reasons),
        "feedback_inputs": {
            "reject_reason_counts": reject_counts,
            "reduce_difficulty_count": reduce_difficulty_count,
            "strict_easier_suggestion_count": strict_easier_count,
            "target_echo_count": target_echo_count,
            "too_difficult_count": too_difficult_count,
            "suggested_difficulty_histogram_all": suggested_hist_all,
            "difficulty_target_histogram": target_hist,
            "difficulty_gap_histogram": gap_hist,
            "reward_means": {k: round(float(v), 4) for k, v in reward_means.items()},
            "failure_tag_counts": tag_counts,
            "buffer_stats": buffer_stats,
            "suggests_easier": bool(suggests_easier),
        },
        "derived_from": "reward_means+failure_tags"
        + ("+reference_feedback" if reference_summary else ""),
        "prev_generator_policy_version": prev_version,
    }
    return policy


def diff_generator_policy(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable diff proving the policy changed across iterations."""
    def _g(d: Dict[str, Any], *keys: str) -> Any:
        cur: Any = d
        for k in keys:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        return cur

    changes: Dict[str, Any] = {}
    if _g(old, "generator_policy_version") != _g(new, "generator_policy_version"):
        changes["generator_policy_version"] = [
            _g(old, "generator_policy_version"),
            _g(new, "generator_policy_version"),
        ]
    if _g(old, "difficulty_policy", "distribution") != _g(
        new, "difficulty_policy", "distribution"
    ):
        changes["difficulty_distribution"] = [
            _g(old, "difficulty_policy", "distribution"),
            _g(new, "difficulty_policy", "distribution"),
        ]
    if _g(old, "sampling_policy", "focus_weights") != _g(
        new, "sampling_policy", "focus_weights"
    ):
        changes["focus_weights"] = [
            _g(old, "sampling_policy", "focus_weights"),
            _g(new, "sampling_policy", "focus_weights"),
        ]
    if _g(old, "image_selection_policy", "seed_offset") != _g(
        new, "image_selection_policy", "seed_offset"
    ):
        changes["image_seed_offset"] = [
            _g(old, "image_selection_policy", "seed_offset"),
            _g(new, "image_selection_policy", "seed_offset"),
        ]
    if _g(old, "difficulty_policy", "difficulty_target") != _g(
        new, "difficulty_policy", "difficulty_target"
    ):
        changes["difficulty_target"] = [
            _g(old, "difficulty_policy", "difficulty_target"),
            _g(new, "difficulty_policy", "difficulty_target"),
        ]
    if _g(new, "policy_update_reason"):
        changes["policy_update_reason"] = _g(new, "policy_update_reason")
    return changes


# ===========================================================================
# Training pathway state
# ===========================================================================

@dataclass
class TrainingPathway:
    """State for one training pathway."""

    name: str
    mandatory: bool = True
    export_ready: bool = False
    trainer_input_ready: bool = False
    trainer_invocation_ready: bool = False
    trainer_executed: bool = False
    checkpoint_created: bool = False
    checkpoint_path: Optional[str] = None
    status: str = "pending"  # pending|export_ready|trainer_invocation_ready|trainer_executed|mandatory_but_blocked
    blocked_reason: Optional[str] = None
    next_action: Optional[str] = None
    trainer_entrypoint: Optional[str] = None
    export_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_training_pathways() -> Dict[str, TrainingPathway]:
    """Initial state for the active GRPO and SFT pathways."""
    return {
        "grpo": TrainingPathway(
            name="grpo",
            trainer_entrypoint="src/open_r1/grpo_jsonl.py",
            status="trainer_invocation_ready",
            trainer_invocation_ready=True,
            next_action="Run grpo_jsonl.py with reward_funcs=self_evolve_refined on the "
            "refined GRPO export (max 1 step smoke).",
        ),
        "sft": TrainingPathway(
            name="sft",
            trainer_entrypoint="src/open_r1/sft_jsonl.py",
            status="trainer_invocation_ready",
            trainer_invocation_ready=True,
            next_action="Run sft.py (TRL SFTTrainer) on the positive-buffer SFT replay "
            "YAML/JSONL export.",
        ),
    }


# ===========================================================================
# IterationState
# ===========================================================================

# Canonical state filenames.
ITERATION_STATE_FILE = "iteration_state.json"
GENERATOR_POLICY_FILE = "generator_policy.json"
FAILURE_PROFILE_FILE = "failure_profile.json"
SOLVER_UPDATE_STATE_FILE = "solver_update_state.json"
REFERENCE_FEEDBACK_FILE = "reference_feedback.jsonl"


@dataclass
class IterationState:
    """Top-level state for one closed-loop iteration."""

    iteration_index: int
    iteration_id: str
    output_dir: str

    # Where this iteration's inputs came from (proves N+1 read N).
    prev_iteration_dir: Optional[str] = None
    read_prev_state: bool = False

    # Generator
    generator_policy_version: int = 0
    generator_policy_changes_from_prev: Dict[str, Any] = field(default_factory=dict)
    num_candidate_tasks: int = 0
    generator_source: str = "unknown"

    # Reference VLM
    reference_provider: Optional[str] = None
    reference_model: Optional[str] = None
    reference_base_url: Optional[str] = None
    reference_mode: str = "dry_run"  # dry_run|live_openrouter|live_openai|heuristic
    num_reference_judged: int = 0
    num_accepted: int = 0
    num_rejected: int = 0
    num_regenerated: int = 0
    num_rejected_final: int = 0

    # Reference budget accounting (judge cap vs training-set size are distinct).
    reference_accounting: Dict[str, Any] = field(default_factory=dict)

    # Solver
    solver_rollout_source: str = "unknown"  # real_solver|dry_run_solver|replay_fallback
    num_solver_trajectories: int = 0
    solver_model_path: Optional[str] = None

    # Solver lineage / carry-forward (proves what GRPO trained FROM this round).
    # Absence of an explicit "train init model" field is exactly what let the
    # non-accumulation bug hide; keep it first-class and auditable.
    solver_lineage: Dict[str, Any] = field(default_factory=dict)

    # Reward / buffers
    buffer_counts: Dict[str, int] = field(default_factory=dict)

    # Training
    trainer_mode: str = "dry_run"  # dry_run|live
    training_pathways: Dict[str, Any] = field(default_factory=dict)

    # Bookkeeping
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- IO -----------------------------------------------------------------

    def save(self) -> None:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / ITERATION_STATE_FILE, self.to_dict())


def save_generator_policy(iteration_dir: str | Path, policy: Dict[str, Any]) -> Path:
    path = Path(iteration_dir) / GENERATOR_POLICY_FILE
    write_json(path, policy)
    return path


def load_generator_policy(iteration_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(iteration_dir) / GENERATOR_POLICY_FILE
    return read_json(path) if path.is_file() else None


def save_failure_profile(iteration_dir: str | Path, profile: Dict[str, Any]) -> Path:
    path = Path(iteration_dir) / FAILURE_PROFILE_FILE
    write_json(path, profile)
    return path


def load_failure_profile(iteration_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(iteration_dir) / FAILURE_PROFILE_FILE
    return read_json(path) if path.is_file() else None


def save_solver_update_state(iteration_dir: str | Path, state: Dict[str, Any]) -> Path:
    path = Path(iteration_dir) / SOLVER_UPDATE_STATE_FILE
    write_json(path, state)
    return path


def load_solver_update_state(iteration_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(iteration_dir) / SOLVER_UPDATE_STATE_FILE
    return read_json(path) if path.is_file() else None


def reference_feedback_path(iteration_dir: str | Path) -> Path:
    return Path(iteration_dir) / REFERENCE_FEEDBACK_FILE


# ===========================================================================
# Failure profile builder
# ===========================================================================

def build_failure_profile_from_counts(
    failure_tag_counts: Dict[str, int],
    reward_means: Dict[str, float],
    buffer_counts: Optional[Dict[str, int]] = None,
    num_trajectories: int = 0,
    source: str = "scored_trajectories",
    solvability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a failure_profile.json payload from aggregated counts.

    Kept separate from the reward-computing code so the loop driver can build a
    profile from either real scored trajectories or a dry-run synthesis, using
    the same schema in both cases.
    """
    profile = {
        "source": source,
        "num_trajectories": num_trajectories,
        "failure_tag_counts": dict(failure_tag_counts),
        "reward_means": {k: round(float(v), 4) for k, v in reward_means.items()},
        "buffer_counts": dict(buffer_counts or {}),
    }
    if solvability:
        profile["solvability"] = solvability
    return profile
