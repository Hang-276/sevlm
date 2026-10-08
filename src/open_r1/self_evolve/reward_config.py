"""Reward scalarization + routing thresholds, read from a JSON config.

Defines how the five-dimensional reward vector becomes the scalar GRPO
optimizes, and the thresholds that route a trajectory into positive / failure.

Weights are changed by editing a config file (``--reward-config <path>``), not
by CLI flags. A wrong config is rejected on load, so it cannot silently train
against the wrong objective.

The optional ``components`` block configures what each dimension means; omitting
it inherits the shipped defaults.
``components.gating`` is the only non-additive piece: it multiplies the
auxiliary dims by an outcome-conditioned factor before the sum, so
format/keyword credit cannot be farmed on tasks the solver failed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# The five reward dimensions. Order is irrelevant for the weighted sum but the
# set is fixed — every dimension must be present in the config.
REWARD_DIMENSIONS: List[str] = ["answer", "grounding", "process", "consistency", "budget"]

# Required routing-threshold keys.
ROUTING_THRESHOLD_KEYS: List[str] = [
    "positive_min_grounding",
    "positive_min_process",
]

# Permitted floating-point error on the weight sum.
WEIGHT_SUM_TOL = 1e-6

# Fallback only; every entry point passes a config explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REWARD_CONFIG_PATH = (
    _REPO_ROOT
    / "local_scripts"
    / "self_evolve"
    / "configs"
    / "reward"
    / "reward_weights.json"
)


class RewardConfigError(ValueError):
    """Raised on an invalid or missing reward config."""


# Merge base for configs that omit keys; values mirror reward_weights.json.

DEFAULT_COMPONENTS: Dict[str, Dict[str, Any]] = {
    # structured_fields scores `spy=` and `changed_attributes=` separately, so
    # one field cannot mask the other.
    "answer": {
        "mode": "structured_fields",
        "fields": {"spy": 0.4, "changed_attributes": 0.6},
        "off_by_one_credit": 0.25,
        # Fields where a near miss is meaningful (counts). Categorical fields
        # such as the player id must stay out of this list.
        "graded_fields": ["changed_attributes"],
    },
    # format/player_id are unconditional floors. "recall" has no precision term,
    # so box spam is strictly dominant; "f1" charges for extra boxes.
    "grounding": {
        "format_credit": 0.0,
        "player_id_credit": 0.1,
        "iou_credit": 0.9,
        "match_mode": "f1",
        "iou_threshold": 0.1,
        "coordinate_mode": "auto",
        # Experimental strict mode: a box claiming another player's image is
        # not evidence for the spy, even if its coordinates happen to overlap.
        "require_correct_player_for_iou": False,
        # Steps before grounding starts paying; 0 keeps the pressure on from
        # the start. >0 is the timing ablation. Live-only by design: offline
        # routing keeps judging the raw box quality.
        "warmup_steps": 0,
    },
    # The optional visual-facts mode verifies explicit attribute transitions
    # against CLEVR scene metadata. Legacy mode preserves existing runs.
    "process": {
        "mode": "legacy", "verified_weight": 0.75,
        # In visual-facts runs, invalid output structure cannot collect
        # grounding/process/consistency credit while the answer still gives
        # an exploration signal. Legacy runs retain their original scalar.
        "require_valid_format_for_aux": False,
    },
    # group_modal is constant within a rollout group, so it has no GRPO
    # gradient. field_consistency scores the trajectory against itself.
    "consistency": {"mode": "field_consistency"},
    "budget": {"fallback_tokens": 400},
    # Outcome gate for the auxiliary dims; floor is what a closed gate still
    # pays. "spy" keeps grounding dense while blocking format farming.
    "gating": {"mode": "spy", "floor": 0.0,
               "dims": ["grounding", "process", "consistency"]},
    # Attribution controls: replace the scalar with a signal that says nothing
    # about correctness.
    "control": {"mode": "none"},
}

_COMPONENT_ENUMS = {
    ("answer", "mode"): {
        "exact_match",
        "structured_exact_match",
        "structured_fields",
    },
    ("grounding", "match_mode"): {"recall", "f1"},
    ("grounding", "coordinate_mode"): {"normalized", "pixel", "auto"},
    ("process", "mode"): {"legacy", "visual_facts"},
    ("consistency", "mode"): {"group_modal", "field_consistency"},
    ("gating", "mode"): {"none", "spy", "answer"},
    ("control", "mode"): {"none", "random", "format_only"},
}


def _unit_float(value: Any, label: str, path: str) -> float:
    """Reject NaN/Inf and out-of-range values before they reach GRPO."""
    if isinstance(value, bool):
        raise RewardConfigError(f"{path}: {label} must be a number in [0, 1].")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardConfigError(f"{path}: {label} must be a number in [0, 1].") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RewardConfigError(f"{path}: {label} must be a finite number in [0, 1].")
    return number


def _nonnegative_int(value: Any, label: str, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise RewardConfigError(f"{path}: {label} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RewardConfigError(f"{path}: {label} must be an integer.") from exc
    if str(number) != str(value) and not (isinstance(value, int) and number == value):
        raise RewardConfigError(f"{path}: {label} must be an integer.")
    if number < (1 if positive else 0):
        raise RewardConfigError(f"{path}: {label} must be {'positive' if positive else 'non-negative'}.")
    return number


def _merge_components(raw: Optional[Dict[str, Any]], path: str) -> Dict[str, Dict[str, Any]]:
    """Merge config ``components`` over the defaults; an unknown key is rejected."""
    merged = {k: dict(v) for k, v in DEFAULT_COMPONENTS.items()}
    if raw is None:
        return merged
    if not isinstance(raw, dict):
        raise RewardConfigError(f"{path}: 'components' must be an object.")
    for section, block in raw.items():
        if section not in merged:
            raise RewardConfigError(
                f"{path}: unknown components section {section!r}. "
                f"Allowed: {sorted(merged)}."
            )
        if not isinstance(block, dict):
            raise RewardConfigError(f"{path}: components.{section} must be an object.")
        unknown = [k for k in block if k not in merged[section]]
        if unknown:
            raise RewardConfigError(
                f"{path}: unknown key(s) {unknown} in components.{section}. "
                f"Allowed: {sorted(merged[section])}."
            )
        merged[section].update(block)

    for (section, key), allowed in _COMPONENT_ENUMS.items():
        value = merged[section][key]
        if not isinstance(value, str) or value not in allowed:
            raise RewardConfigError(
                f"{path}: components.{section}.{key}={value!r} is invalid. "
                f"Allowed: {sorted(allowed)}."
            )
    gate_dims = merged["gating"]["dims"]
    if not isinstance(gate_dims, list) or any(d not in REWARD_DIMENSIONS for d in gate_dims):
        raise RewardConfigError(
            f"{path}: components.gating.dims must be a list of reward dimensions "
            f"({REWARD_DIMENSIONS}), got {gate_dims!r}."
        )
    if "answer" in gate_dims:
        raise RewardConfigError(
            f"{path}: components.gating.dims must not contain 'answer' — gating "
            "the answer on itself is meaningless."
        )
    fields = merged["answer"]["fields"]
    if not isinstance(fields, dict) or not fields:
        raise RewardConfigError(f"{path}: components.answer.fields must be a non-empty object.")
    unknown_fields = set(fields) - {"spy", "changed_attributes"}
    if unknown_fields:
        raise RewardConfigError(f"{path}: unknown answer field(s): {sorted(unknown_fields)}.")
    merged["answer"]["fields"] = {
        key: _unit_float(value, f"components.answer.fields.{key}", path)
        for key, value in fields.items()
    }
    field_total = sum(merged["answer"]["fields"].values())
    if abs(field_total - 1.0) > WEIGHT_SUM_TOL:
        raise RewardConfigError(
            f"{path}: components.answer.fields sum to {field_total} — must be 1.0 so "
            "the answer dimension stays in [0, 1]."
        )
    graded = merged["answer"]["graded_fields"]
    if not isinstance(graded, list) or any(field not in fields for field in graded):
        raise RewardConfigError(f"{path}: components.answer.graded_fields must use configured fields.")
    merged["answer"]["off_by_one_credit"] = _unit_float(
        merged["answer"]["off_by_one_credit"], "components.answer.off_by_one_credit", path
    )
    for key in ("format_credit", "player_id_credit", "iou_credit", "iou_threshold"):
        merged["grounding"][key] = _unit_float(
            merged["grounding"][key], f"components.grounding.{key}", path
        )
    if sum(merged["grounding"][key] for key in ("format_credit", "player_id_credit", "iou_credit")) > 1.0 + WEIGHT_SUM_TOL:
        raise RewardConfigError(f"{path}: grounding credits must sum to at most 1.0.")
    if type(merged["grounding"]["require_correct_player_for_iou"]) is not bool:
        raise RewardConfigError(
            f"{path}: components.grounding.require_correct_player_for_iou must be boolean."
        )
    merged["grounding"]["warmup_steps"] = _nonnegative_int(
        merged["grounding"]["warmup_steps"], "components.grounding.warmup_steps", path
    )
    merged["process"]["verified_weight"] = _unit_float(
        merged["process"]["verified_weight"], "components.process.verified_weight", path
    )
    if type(merged["process"]["require_valid_format_for_aux"]) is not bool:
        raise RewardConfigError(
            f"{path}: components.process.require_valid_format_for_aux must be boolean."
        )
    if merged["process"]["require_valid_format_for_aux"] and merged["process"]["mode"] != "visual_facts":
        raise RewardConfigError(
            f"{path}: require_valid_format_for_aux requires process.mode='visual_facts'."
        )
    merged["gating"]["floor"] = _unit_float(
        merged["gating"]["floor"], "components.gating.floor", path
    )
    merged["budget"]["fallback_tokens"] = _nonnegative_int(
        merged["budget"]["fallback_tokens"], "components.budget.fallback_tokens", path,
        positive=True,
    )
    return merged


@dataclass(frozen=True)
class RewardConfig:
    """Validated reward scalarization + routing configuration."""

    path: str
    scalarization: str
    version: str
    weights: Dict[str, float]
    routing_thresholds: Dict[str, float]
    notes: str = ""
    components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        k: dict(v) for k, v in DEFAULT_COMPONENTS.items()
    })

    # ----- component accessors -----
    @property
    def answer_cfg(self) -> Dict[str, Any]:
        return self.components["answer"]

    @property
    def grounding_cfg(self) -> Dict[str, Any]:
        return self.components["grounding"]

    @property
    def process_cfg(self) -> Dict[str, Any]:
        return self.components["process"]

    @property
    def consistency_cfg(self) -> Dict[str, Any]:
        return self.components["consistency"]

    @property
    def budget_cfg(self) -> Dict[str, Any]:
        return self.components["budget"]

    @property
    def gating_cfg(self) -> Dict[str, Any]:
        return self.components["gating"]

    @property
    def control_mode(self) -> str:
        return self.components["control"]["mode"]

    # ----- outcome-conditioned gating -----
    def gate_factor(self, *, spy_correct: Optional[bool], answer_correct: Optional[bool]) -> float:
        """Multiplier for the gated auxiliary dims (1.0 open, ``floor`` closed).

        An unknown outcome (None) leaves the gate open.
        """
        mode = self.gating_cfg["mode"]
        if mode == "none":
            return 1.0
        opened = spy_correct if mode == "spy" else answer_correct
        if opened is None or opened:
            return 1.0
        return float(self.gating_cfg["floor"])

    def gated_dims(self) -> List[str]:
        return list(self.gating_cfg["dims"]) if self.gating_cfg["mode"] != "none" else []

    # ----- scalarization -----
    def scalarize(
        self,
        reward_vector: Dict[str, Any],
        *,
        spy_correct: Optional[bool] = None,
        answer_correct: Optional[bool] = None,
        format_valid: Optional[bool] = None,
    ) -> float:
        """Weighted sum over the five dims, after applying the outcome gate."""
        gate = self.gate_factor(spy_correct=spy_correct, answer_correct=answer_correct)
        gated = set(self.gated_dims()) if gate != 1.0 else set()
        suppress_aux = self.process_cfg["require_valid_format_for_aux"] and format_valid is False
        return sum(
            self.weights[dim] * float(reward_vector.get(dim, 0.0))
            * (gate if dim in gated else 1.0)
            * (0.0 if suppress_aux and dim in {"grounding", "process", "consistency"} else 1.0)
            for dim in REWARD_DIMENSIONS
        )

    # ----- audit block -----
    def audit_fields(
        self,
        reward_vector: Dict[str, Any],
        *,
        spy_correct: Optional[bool] = None,
        answer_correct: Optional[bool] = None,
        format_valid: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Auditable scalarization metadata for ``reward_details``."""
        gate = self.gate_factor(spy_correct=spy_correct, answer_correct=answer_correct)
        return {
            "reward_config_path": self.path,
            "reward_scalarization": self.scalarization,
            "reward_config_version": self.version,
            "reward_weights": dict(self.weights),
            "reward_scalar_used": self.scalarize(
                reward_vector, spy_correct=spy_correct, answer_correct=answer_correct,
                format_valid=format_valid,
            ),
            "reward_components": {k: dict(v) for k, v in self.components.items()},
            "gate_factor": gate,
            "gated_dims": self.gated_dims() if gate != 1.0 else [],
            "format_gate_factor": (
                0.0 if self.process_cfg["require_valid_format_for_aux"] and format_valid is False
                else 1.0
            ),
        }

    # ----- routing thresholds -----
    @property
    def positive_min_grounding(self) -> float:
        return self.routing_thresholds["positive_min_grounding"]

    @property
    def positive_min_process(self) -> float:
        return self.routing_thresholds["positive_min_process"]

    @property
    def positive_min_visual_facts(self) -> float:
        return self.routing_thresholds.get("positive_min_visual_facts", 0.0)

def _validate(raw: Dict[str, Any], path: str) -> RewardConfig:
    scalarization = raw.get("reward_scalarization")
    if scalarization != "configured_weighted_sum":
        raise RewardConfigError(
            f"{path}: reward_scalarization must be 'configured_weighted_sum', "
            f"got {scalarization!r}. No equal-weight / other scheme is supported."
        )

    version = str(raw.get("version") or "")
    if not version:
        raise RewardConfigError(f"{path}: 'version' is required.")

    weights = raw.get("weights")
    if not isinstance(weights, dict):
        raise RewardConfigError(f"{path}: 'weights' must be an object.")

    # Every dimension must be present — no defaulting to equal weight.
    missing = [d for d in REWARD_DIMENSIONS if d not in weights]
    if missing:
        raise RewardConfigError(
            f"{path}: missing reward weight(s) for dimension(s): {missing}. "
            "All five dimensions must be specified explicitly."
        )
    extra = [k for k in weights if k not in REWARD_DIMENSIONS]
    if extra:
        raise RewardConfigError(
            f"{path}: unknown reward weight key(s): {extra}. "
            f"Allowed dimensions: {REWARD_DIMENSIONS}."
        )

    typed_weights: Dict[str, float] = {}
    for dim in REWARD_DIMENSIONS:
        typed_weights[dim] = _unit_float(weights[dim], f"weight for '{dim}'", path)

    total = sum(typed_weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise RewardConfigError(
            f"{path}: reward weights sum to {total!r}, expected 1.0 "
            f"(tolerance {WEIGHT_SUM_TOL}). Weights are NOT auto-normalized — "
            "fix the config so the sum is exactly 1.0."
        )

    routing = raw.get("routing_thresholds")
    if not isinstance(routing, dict):
        raise RewardConfigError(f"{path}: 'routing_thresholds' must be an object.")
    unknown_thr = set(routing) - set(ROUTING_THRESHOLD_KEYS) - {"positive_min_visual_facts"}
    if unknown_thr:
        raise RewardConfigError(f"{path}: unknown routing threshold(s): {sorted(unknown_thr)}.")
    missing_thr = [k for k in ROUTING_THRESHOLD_KEYS if k not in routing]
    if missing_thr:
        raise RewardConfigError(
            f"{path}: missing routing threshold(s): {missing_thr}. "
            f"Required: {ROUTING_THRESHOLD_KEYS}."
        )
    typed_routing: Dict[str, float] = {}
    for k in ROUTING_THRESHOLD_KEYS:
        typed_routing[k] = _unit_float(routing[k], f"routing threshold '{k}'", path)
    if "positive_min_visual_facts" in routing:
        typed_routing["positive_min_visual_facts"] = _unit_float(
            routing["positive_min_visual_facts"], "routing threshold 'positive_min_visual_facts'", path
        )

    components = _merge_components(raw.get("components"), path)
    if components["process"]["mode"] == "visual_facts" and "positive_min_visual_facts" not in routing:
        raise RewardConfigError(f"{path}: visual_facts process requires an explicit positive_min_visual_facts threshold.")
    if typed_routing.get("positive_min_visual_facts", 0.0) > 0.0 and components["process"]["mode"] != "visual_facts":
        raise RewardConfigError(f"{path}: positive_min_visual_facts requires process.mode='visual_facts'.")

    return RewardConfig(
        path=path,
        scalarization=scalarization,
        version=version,
        weights=typed_weights,
        routing_thresholds=typed_routing,
        notes=str(raw.get("notes") or ""),
        components=components,
    )


def load_reward_config(config_path: Optional[str | Path] = None) -> RewardConfig:
    """Load + validate a reward config.

    If ``config_path`` is None, the default config file is used. A missing file
    is a hard error — we never fall back to equal weight.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_REWARD_CONFIG_PATH
    if not path.is_file():
        raise RewardConfigError(
            f"Reward config not found: {path}. A valid config is mandatory; "
            "there is no equal-weight fallback."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RewardConfigError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RewardConfigError(f"{path}: top-level JSON must be an object.")
    return _validate(raw, str(path))
