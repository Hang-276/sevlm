"""E. Offline reward/routing recompute on existing real_solver trajectories.

NO API calls, NO solver rollout, NO training. Reads existing
``raw_solver_trajectories.jsonl`` files and re-scores them with the
config-driven weighted reward + vector routing, then reports buffer / SFT
counts and shortcut stats under the NEW weights.

Run:
    python local_scripts/self_evolve/reward_alignment/offline_recompute_reward.py \
        $RUNS_ROOT/<run>/iter_000/raw_solver_trajectories.jsonl \
        $RUNS_ROOT/<run>/iter_001/raw_solver_trajectories.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from open_r1.self_evolve.iteration import score_and_route_trajectories  # noqa: E402
from open_r1.self_evolve.exporters import build_sft_replay_examples  # noqa: E402
from open_r1.self_evolve.reward_config import load_reward_config  # noqa: E402


def recompute(paths):
    cfg = load_reward_config()
    trajectories = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    scored = score_and_route_trajectories(
        trajectories, include_details=True, reward_config=cfg
    )

    buffers = {"positive": 0, "failure": 0, "unused": 0}
    scalars = []
    shortcut_correct = []  # (scalar) for answer==1 & shortcut/hard-neg
    for ex in scored:
        buffers[ex["buffer"]] = buffers.get(ex["buffer"], 0) + 1
        scalars.append(float(ex["reward_scalar"]))
        rv = ex["reward_vector"]
        det = rv.get("reward_details", {})
        is_short = bool(det.get("shortcut_detected")) or \
            "shortcut_or_ungrounded_success" in ex.get("failure_tags", [])
        if float(rv.get("answer", 0)) >= 1.0 and is_short:
            shortcut_correct.append(float(ex["reward_scalar"]))

    sft = build_sft_replay_examples(scored)

    mean_scalar = sum(scalars) / len(scalars) if scalars else 0.0
    short_avg = sum(shortcut_correct) / len(shortcut_correct) if shortcut_correct else 0.0

    report = {
        "reward_config_path": cfg.path,
        "reward_weights": cfg.weights,
        "num_trajectories": len(scored),
        "mean_configured_weighted_reward": round(mean_scalar, 4),
        "positive_count_new": buffers["positive"],
        "failure_count_new": buffers["failure"],
        "unused_count_new": buffers["unused"],
        "sft_sample_count_new": len(sft),
        "shortcut_correct_count": len(shortcut_correct),
        "shortcut_correct_avg_configured_weighted_reward": round(short_avg, 4),
        "note": "old results from previous run are not recomputed here; "
                "no equal-weight comparison table is produced.",
    }
    return report


def main(argv) -> int:
    if not argv:
        print(
            "usage: offline_recompute_reward.py <raw_solver_trajectories.jsonl> [...]\n"
            "  e.g. $RUNS_ROOT/<run>/iter_000/raw_solver_trajectories.jsonl",
            file=sys.stderr,
        )
        return 2
    report = recompute(argv)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
