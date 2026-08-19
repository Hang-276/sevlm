#!/usr/bin/env python3
"""Read collapse signals off a finished run, without retraining.

The aggregate reward curve hides both a field collapsing onto the label prior
and grounding credit earned from format rather than overlap; these per-iteration
numbers do not.

    collapse_diagnostics.py --run-dir <run> [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from open_r1.self_evolve.proposer import proposal_report
from open_r1.self_evolve.regret import (  # noqa: E402
    counterfactual_sensitivity,
    summarize,
    task_statistics,
)
from open_r1.self_evolve.rewards import parse_structured_answer  # noqa: E402


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def answer_diagnostics(scored):
    """Prior share and per-field accuracy — the first two curves to move."""
    pred_ca, pred_spy, gold_ca, gold_spy = [], [], [], []
    for ex in scored:
        pred = parse_structured_answer(ex.get("completion") or "")
        gold = parse_structured_answer(ex.get("solution") or ex.get("answer") or "")
        pred_ca.append(pred.get("changed_attributes"))
        pred_spy.append(pred.get("spy"))
        gold_ca.append(gold.get("changed_attributes"))
        gold_spy.append(gold.get("spy"))

    n = len(scored) or 1
    hist = Counter(x for x in pred_ca if x is not None)
    mode, mode_n = hist.most_common(1)[0] if hist else (None, 0)
    correct = lambda p, g: sum(1 for a, b in zip(p, g) if a is not None and a == b)
    return {
        "n": len(scored),
        # 1.0 means the field has collapsed onto a single value.
        "pred_ca_mode": mode,
        "pred_ca_mode_share": round(mode_n / n, 4),
        "spy_acc": round(correct(pred_spy, gold_spy) / n, 4),
        "changed_attributes_acc": round(correct(pred_ca, gold_ca) / n, 4),
        "gold_ca_histogram": dict(sorted(Counter(x for x in gold_ca if x is not None).items())),
        "pred_ca_histogram": dict(sorted(hist.items())),
        "answer_parse_failed": sum(1 for x in pred_ca if x is None),
    }


def grounding_diagnostics(scored):
    """Whether grounding credit came from overlap or just from emitting a box."""
    valid, ious, box_counts = 0, [], []
    for ex in scored:
        details = (ex.get("reward_vector") or {}).get("reward_details") or {}
        if details.get("bbox_valid"):
            valid += 1
            iou = details.get("metadata_bbox_iou")
            if iou is not None:
                ious.append(float(iou))
        boxes = details.get("predicted_bbox_norm") or []
        box_counts.append(len(boxes))
    n = len(scored) or 1
    return {
        "bbox_valid_rate": round(valid / n, 4),
        # A rising valid rate with a flat IoU is format farming, not grounding.
        "mean_iou_given_valid": round(sum(ious) / len(ious), 4) if ious else None,
        "iou_above_0.3_rate": round(sum(1 for i in ious if i > 0.3) / n, 4),
        "mean_pred_boxes": round(sum(box_counts) / n, 4),
        "max_pred_boxes": max(box_counts) if box_counts else 0,
    }


def group_by_task(scored):
    groups = {}
    for ex in scored:
        groups.setdefault(str(ex.get("task_id")), []).append(ex)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    iters = sorted(p for p in run_dir.glob("iter_*") if p.is_dir())
    if not iters:
        print(f"[ERROR] no iter_* under {run_dir}", file=sys.stderr)
        return 2

    report = {"run_dir": str(run_dir), "iterations": []}
    for it in iters:
        scored = read_jsonl(it / "scored_trajectories.jsonl")
        if not scored:
            continue
        tasks = read_jsonl(it / "accepted_tasks.jsonl")
        stats = task_statistics(group_by_task(scored))
        summary = summarize(stats)
        report["iterations"].append({
            "iteration": it.name,
            "answer": answer_diagnostics(scored),
            "grounding": grounding_diagnostics(scored),
            "solvability": {k: summary[k] for k in
                            ("num_tasks", "mean_solve_rate", "mean_regret",
                             "advantage_collapse_rate", "class_counts")
                            if k in summary},
            "counterfactual": counterfactual_sensitivity(scored, tasks),
            "proposer": proposal_report(
                read_jsonl(it / "proposals_scored.jsonl")),
        })

    header = (f"{'iter':<10}{'ca_mode%':>10}{'spy_acc':>9}{'ca_acc':>8}"
              f"{'bbox_ok':>9}{'IoU|ok':>8}{'boxes':>7}{'regret':>8}{'ACR':>7}{'cf_sens':>9}")
    print(header)
    print("-" * len(header))
    for row in report["iterations"]:
        a, g, s = row["answer"], row["grounding"], row["solvability"]
        cf = row["counterfactual"].get("counterfactual_sensitivity")
        iou = g["mean_iou_given_valid"]
        print(f"{row['iteration']:<10}{a['pred_ca_mode_share']:>10.2f}{a['spy_acc']:>9.2f}"
              f"{a['changed_attributes_acc']:>8.2f}{g['bbox_valid_rate']:>9.2f}"
              f"{(iou if iou is not None else float('nan')):>8.2f}"
              f"{g['mean_pred_boxes']:>7.1f}"
              f"{(s.get('mean_regret') or 0.0):>8.2f}"
              f"{(s.get('advantage_collapse_rate') or 0.0):>7.2f}"
              f"{(cf if cf is not None else float('nan')):>9.2f}")

    if any(r["proposer"].get("num_proposals") for r in report["iterations"]):
        head2 = (f"\n{'iter':<10}{'parse':>8}{'learn':>8}{'trainable':>11}"
                 f"{'distinct':>10}{'diversity':>11}{'cf_pairs':>10}")
        print(head2)
        print("-" * (len(head2) - 1))
        for row in report["iterations"]:
            p = row["proposer"]
            if not p.get("num_proposals"):
                continue
            ml = p.get("mean_learnability")
            print(f"{row['iteration']:<10}{p.get('parse_rate', 0.0):>8.2f}"
                  f"{(ml if ml is not None else float('nan')):>8.2f}"
                  f"{p.get('num_trainable', 0):>11}"
                  f"{p.get('distinct_subsets', 0):>10}"
                  f"{p.get('subset_diversity', 0.0):>11.2f}"
                  f"{p.get('num_counterfactual_pairs', 0):>10}")
        print("\nlearn is what the solver realized on the model's own puzzles; diversity"
              "\nfalling toward 0 means the proposer collapsed onto one pick and stopped"
              "\nbeing an opponent.")

    print("\nca_mode% -> 1 means the count field collapsed; bbox_ok rising with a flat"
          "\nIoU|ok means format farming; growing boxes means box spam; ACR should fall"
          "\nacross iterations; cf_sens is the number a prior-following policy cannot get.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
