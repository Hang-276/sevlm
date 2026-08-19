"""Offline checks for the config-driven weighted reward (no API, no training).

Covers:
  C. Reward config validation (presence, dims, non-negative, sum==1, routing
     thresholds) and rejection of a missing field / bad sum / missing file. Confirms
     there is no equal-weight fallback.
  D. Five unit reward cases (correct_grounded / correct_shortcut /
     wrong_but_structured / verbose_correct / format_invalid) showing the
     configured weighted scalar + routing decision.

Run:
    python local_scripts/self_evolve/reward_alignment/check_reward_config.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make src importable when run directly.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from open_r1.self_evolve.reward_config import (  # noqa: E402
    DEFAULT_REWARD_CONFIG_PATH,
    RewardConfig,
    RewardConfigError,
    load_reward_config,
)
from open_r1.self_evolve.buffer import route_with_config  # noqa: E402
from open_r1.self_evolve.failure_tags import assign_failure_tags  # noqa: E402


def _expect_fail(label: str, fn) -> bool:
    try:
        fn()
    except RewardConfigError as exc:
        print(f"  [PASS] {label}: rejected as expected -> {type(exc).__name__}")
        return True
    print(f"  [FAIL] {label}: expected RewardConfigError, none raised")
    return False


def _write_tmp(d: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, f)
    f.close()
    return f.name


def check_config_validation() -> bool:
    print("\n=== Reward config validation ===")
    ok = True

    # The shipped config loads and has the shape it must have.
    cfg = load_reward_config()
    print(f"  [PASS] default config exists: {cfg.path}")
    dims = ["answer", "grounding", "process", "consistency", "budget"]
    assert all(d in cfg.weights for d in dims), "missing dim"
    print(f"  [PASS] all 5 dimensions present: {sorted(cfg.weights)}")
    assert all(w >= 0 for w in cfg.weights.values()), "negative weight"
    print(f"  [PASS] all weights non-negative")
    total = sum(cfg.weights.values())
    assert abs(total - 1.0) <= 1e-6, total
    print(f"  [PASS] weights sum to 1.0 (={total})")
    for k in ["positive_min_grounding", "positive_min_process"]:
        assert k in cfg.routing_thresholds, k
    print(f"  [PASS] routing thresholds present: {cfg.routing_thresholds}")
    assert cfg.scalarization == "configured_weighted_sum"
    print(f"  [PASS] scalarization = {cfg.scalarization}")

    base = {
        "reward_scalarization": "configured_weighted_sum",
        "version": "test",
        "weights": {"answer": 0.45, "grounding": 0.25, "process": 0.20,
                    "consistency": 0.07, "budget": 0.03},
        "routing_thresholds": {"positive_min_grounding": 0.5, "positive_min_process": 0.5},
    }

    # A dimension missing from the config.
    missing = json.loads(json.dumps(base)); del missing["weights"]["grounding"]
    ok &= _expect_fail("missing dimension", lambda: load_reward_config(_write_tmp(missing)))

    # Weights that do not sum to 1.
    bad_sum = json.loads(json.dumps(base)); bad_sum["weights"]["answer"] = 0.90
    ok &= _expect_fail("weights sum != 1", lambda: load_reward_config(_write_tmp(bad_sum)))

    # A negative weight.
    neg = json.loads(json.dumps(base))
    neg["weights"]["budget"] = -0.03; neg["weights"]["answer"] = 0.51
    ok &= _expect_fail("negative weight", lambda: load_reward_config(_write_tmp(neg)))

    # A config path that does not exist — there is no equal-weight fallback.
    ok &= _expect_fail(
        "missing config file (no equal-weight fallback)",
        lambda: load_reward_config(str(_REPO / "does_not_exist_reward.json")),
    )

    # equal-weight scheme not accepted.
    eq = json.loads(json.dumps(base)); eq["reward_scalarization"] = "equal_weight"
    ok &= _expect_fail("equal_weight scheme rejected", lambda: load_reward_config(_write_tmp(eq)))

    return ok


# --- D. Five unit reward cases (synthetic reward vectors) ------------------

CASES = [
    ("correct_grounded_reasoning",
     {"answer": 1.0, "grounding": 0.9, "process": 0.85, "consistency": 0.8, "budget": 0.9},
     {"format_valid": True, "shortcut_detected": False}),
    ("correct_shortcut",
     {"answer": 1.0, "grounding": 0.1, "process": 0.2, "consistency": 0.5, "budget": 0.95},
     {"format_valid": True, "shortcut_detected": True}),
    ("wrong_but_structured",
     {"answer": 0.0, "grounding": 0.9, "process": 0.9, "consistency": 0.7, "budget": 0.6},
     {"format_valid": True, "shortcut_detected": False}),
    ("verbose_correct",
     {"answer": 1.0, "grounding": 0.85, "process": 0.8, "consistency": 0.75, "budget": 0.05},
     {"format_valid": True, "shortcut_detected": False}),
    ("format_invalid",
     {"answer": 1.0, "grounding": 0.9, "process": 0.8, "consistency": 0.7, "budget": 0.5},
     {"format_valid": False, "shortcut_detected": False}),
]


def check_unit_cases(cfg: RewardConfig) -> bool:
    print("\n=== D. Five unit reward cases ===")
    rows = []
    for name, rv, flags in CASES:
        details = dict(flags)
        rv_with = dict(rv); rv_with["reward_details"] = details
        tags = assign_failure_tags(rv)
        buffer_name, reason = route_with_config(rv_with, tags, cfg, reward_details=details)
        scalar = cfg.scalarize(rv)
        rows.append((name, rv, scalar, buffer_name, reason))
        print(f"\n  case: {name}")
        print(f"    reward_vector            = {rv}")
        print(f"    configured_weighted_scalar = {scalar:.4f}")
        print(f"    reward_scalar_used       = {scalar:.4f}")
        print(f"    reward_config_path       = {cfg.path}")
        print(f"    reward_weights           = {cfg.weights}")
        print(f"    routing_decision         = {buffer_name}")
        print(f"    routing_reason           = {reason}")

    # Assertions matching the spec's expectations.
    by_name = {r[0]: r for r in rows}
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(f"    [{'PASS' if cond else 'FAIL'}] {msg}")
        ok &= cond

    print("\n  --- expectations ---")
    grounded = by_name["correct_grounded_reasoning"]
    shortcut = by_name["correct_shortcut"]
    chk(grounded[3] == "positive", "correct_grounded -> positive")
    chk(shortcut[3] == "failure", "correct_shortcut -> failure (not SFT)")
    chk(shortcut[2] < grounded[2], "correct_shortcut scalar < grounded scalar")
    chk(by_name["wrong_but_structured"][3] == "failure", "wrong_but_structured -> failure")
    chk(by_name["wrong_but_structured"][2] < grounded[2],
        "wrong_but_structured scalar < grounded (answer weight dominates)")
    verbose = by_name["verbose_correct"]
    chk(verbose[3] == "positive", "verbose_correct -> positive (budget weight only 0.03)")
    chk(verbose[2] > shortcut[2], "verbose_correct scalar > shortcut scalar")
    chk(by_name["format_invalid"][3] != "positive",
        "format_invalid -> not positive (routing, not gate)")
    return ok


def main() -> int:
    print(f"Reward config under test: {DEFAULT_REWARD_CONFIG_PATH}")
    cfg = load_reward_config()
    ok = True
    ok &= check_config_validation()
    ok &= check_unit_cases(cfg)
    print(f"\n=== RESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
