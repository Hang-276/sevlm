#!/usr/bin/env python
"""Scan a self-evolve run dir and report grounding + num_generations stats.

Reports grounding-signal statistics by:
  1. reading ``iter_*/scored_trajectories.jsonl`` (offline scored trajectories,
     carry the model completion + reward_details grounding fields);
  2. reading ``iter_*/exports/grpo_tasks.jsonl`` (the GRPO training records) and
     confirming the ``self_evolve.grounding`` context block is embedded;
  3. RE-RUNNING the LIVE GRPO reward (``self_evolve_refined_reward``) over the
     (completion, grounding-context) pairs reconstructed by joining the two, so
     the reported bbox stats come from the ACTUAL training reward path — not the
     offline scorer.

Usage:
    PYTHONPATH=src python local_scripts/self_evolve/reward_alignment/scan_grounding_numgen_stats.py <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_r1.self_evolve.live_reward import self_evolve_refined_reward  # noqa: E402
from open_r1.self_evolve.grounding_iou import score_model_bbox_grounding  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _completion_text(traj: Dict[str, Any]) -> str:
    c = traj.get("completion", "")
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c or "")


def scan_run(run_dir: Path) -> Dict[str, Any]:
    iter_dirs = sorted([d for d in run_dir.glob("iter_*") if d.is_dir()])
    if not iter_dirs:
        iter_dirs = [run_dir]

    stats: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "iterations_scanned": [d.name for d in iter_dirs],
        "trajectory_count": 0,
        "bbox_present_count": 0,
        "bbox_valid_count": 0,
        "bbox_missing_count": 0,
        "bbox_invalid_count": 0,
        "invalid_player_id_count": 0,
        "grounding_reward_nonzero_count": 0,
        "grpo_tasks_count": 0,
        "grpo_examples_with_grounding_fields": 0,
        "fallback_used_count": 0,
        "_grounding_sum": 0.0,
        "_total_before_sum": 0.0,
        "_total_after_sum": 0.0,
        "_scored_n": 0,
    }

    for it in iter_dirs:
        scored = _read_jsonl(it / "scored_trajectories.jsonl")
        grpo = _read_jsonl(it / "exports" / "grpo_tasks.jsonl")

        # Build task_id -> grounding context from the GRPO export.
        ground_ctx: Dict[str, Dict[str, Any]] = {}
        for ex in grpo:
            stats["grpo_tasks_count"] += 1
            se = ex.get("self_evolve", {}) or {}
            g = se.get("grounding") or {}
            if g.get("gold_evidence_boxes") is not None and "image_width" in g:
                stats["grpo_examples_with_grounding_fields"] += 1
            tid = str(ex.get("task_id"))
            ground_ctx[tid] = g

        # Re-run the LIVE reward over each scored trajectory using the embedded
        # grounding context (this is the real training reward path).
        for traj in scored:
            stats["trajectory_count"] += 1
            comp = _completion_text(traj)
            tid = str(traj.get("task_id"))
            g_ctx = ground_ctx.get(tid)
            if g_ctx is None:
                # Fall back to the trajectory's own metadata gold boxes.
                meta = traj.get("metadata", {}) or {}
                g_ctx = {
                    "image_width": 320, "image_height": 240,
                    "gold_evidence_boxes": meta.get("gold_evidence_boxes")
                    or (meta.get("comparison_data", {}) or {}).get("gold_evidence_boxes"),
                    "valid_player_ids": [1, 2, 3],
                }
            se_block = {"grounding": g_ctx}

            rewards = self_evolve_refined_reward(
                [[{"role": "assistant", "content": comp}]],
                solution=[traj.get("solution") or ""],
                self_evolve=[se_block],
                problem=[traj.get("problem") or ""],
            )
            # Recompute the audit fields directly for per-sample stats.
            f = score_model_bbox_grounding(
                comp,
                gold_boxes_pixel=g_ctx.get("gold_evidence_boxes") or None,
                image_width=int(g_ctx.get("image_width") or 320),
                image_height=int(g_ctx.get("image_height") or 240),
                valid_player_ids=g_ctx.get("valid_player_ids") or [1, 2, 3],
            )
            if f["bbox_present"]:
                stats["bbox_present_count"] += 1
            else:
                stats["bbox_missing_count"] += 1
            if f["bbox_valid"]:
                stats["bbox_valid_count"] += 1
            elif f["bbox_present"]:
                stats["bbox_invalid_count"] += 1
            if f.get("bbox_invalid_reason") == "invalid_player_id":
                stats["invalid_player_id_count"] += 1
            gr = float(f["grounding_reward"])
            if gr > 0.0:
                stats["grounding_reward_nonzero_count"] += 1
            if f["grounding_reward_source"] != "model_bbox_iou":
                stats["fallback_used_count"] += 1
            stats["_grounding_sum"] += gr
            stats["_scored_n"] += 1
            # before/after grounding totals from the weighted scalar.
            from open_r1.self_evolve.reward_config import load_reward_config
            cfg = load_reward_config()
            gw = cfg.weights.get("grounding", 0.0)
            total = rewards[0]
            stats["_total_after_sum"] += total
            stats["_total_before_sum"] += (total - gw * gr)

    n = max(1, stats["_scored_n"])
    stats["mean_grounding_reward"] = round(stats["_grounding_sum"] / n, 4)
    stats["mean_total_reward_before_grounding"] = round(stats["_total_before_sum"] / n, 4)
    stats["mean_total_reward_after_grounding"] = round(stats["_total_after_sum"] / n, 4)
    for k in ("_grounding_sum", "_total_before_sum", "_total_after_sum", "_scored_n"):
        stats.pop(k, None)
    return stats


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: scan_grounding_numgen_stats.py <run_dir>")
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    stats = scan_run(run_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
