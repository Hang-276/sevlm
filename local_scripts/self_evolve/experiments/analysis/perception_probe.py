#!/usr/bin/env python3
"""Base-model ceiling on the CLEVR spy task at several input resolutions.

Answers the question every reward change depends on: can the model see these
attribute changes at all. If changed_attributes accuracy still loses to the
label prior at 4x, then answering with the mode is correct and no reward design
will help — the task has to change.

Reports spy and changed_attributes accuracy separately against the
always-answer-the-mode baseline. No API needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from open_r1.self_evolve.policy_clevr_generator import (  # noqa: E402
    PolicyCLEVRGeneratorConfig,
    PolicyControlledCLEVRGenerator,
)
from open_r1.self_evolve.rewards import parse_structured_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-tasks", type=int, default=200)
    ap.add_argument("--num-players", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-pixels", type=int, nargs="+", default=[200704, 802816],
        help="one run per value; 200704=256*28*28 (1x), 802816 (4x)",
    )
    ap.add_argument("--max-pixels", type=int, default=4014080)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def evaluate(tasks, model_path, min_pixels, max_pixels, max_new_tokens, seed):
    from open_r1.self_evolve.online_solver import (
        OnlineSolveConfig,
        OnlineVLLMSolverSampler,
    )

    sampler = OnlineVLLMSolverSampler(OnlineSolveConfig(
        model_path=model_path,
        max_images=max((len(t.get("image_path") or []) for t in tasks), default=1),
        dry_run=False,
        max_new_tokens=max_new_tokens,
        temperature=0.0,          # greedy: this measures a ceiling, not diversity
        seed=seed,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    ))

    rows = []
    for i, task in enumerate(tasks):
        traj = sampler.generate_one(
            task_id=str(task["task_id"]),
            image_paths=list(task.get("image_path") or []),
            prompt_text=str(task.get("prompt") or ""),
            gen_index=0,
            diagnostic_only=False,
        )
        completion = traj.get("completion") or traj.get("response") or ""
        pred = parse_structured_answer(completion)
        gold = parse_structured_answer(task["solution"])
        rows.append({
            "task_id": task["task_id"],
            "pred_spy": pred["spy"], "gold_spy": gold["spy"],
            "pred_ca": pred["changed_attributes"], "gold_ca": gold["changed_attributes"],
        })
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(tasks)}", flush=True)
    sampler.unload()
    return rows


def summarize(rows):
    n = len(rows) or 1
    spy_acc = sum(r["pred_spy"] == r["gold_spy"] for r in rows) / n
    ca_acc = sum(r["pred_ca"] == r["gold_ca"] for r in rows) / n
    gold_hist = Counter(r["gold_ca"] for r in rows)
    pred_hist = Counter(r["pred_ca"] for r in rows)
    mode_label, mode_count = gold_hist.most_common(1)[0]
    pred_mode, pred_mode_count = pred_hist.most_common(1)[0]
    return {
        "n": len(rows),
        "spy_acc": round(spy_acc, 4),
        "ca_acc": round(ca_acc, 4),
        # Accuracy of always answering the most common gold count. ca_acc at or
        # below this means the model is not counting.
        "ca_prior_acc": round(mode_count / n, 4),
        "ca_mode_label": mode_label,
        "pred_mode_share": round(pred_mode_count / n, 4),
        "pred_mode_label": pred_mode,
        "parse_fail": sum(r["pred_ca"] is None or r["pred_spy"] is None for r in rows),
        "gold_ca_hist": dict(sorted(gold_hist.items(), key=lambda kv: str(kv[0]))),
        "pred_ca_hist": dict(sorted(pred_hist.items(), key=lambda kv: str(kv[0]))),
    }


def main() -> int:
    args = parse_args()
    generator = PolicyControlledCLEVRGenerator(PolicyCLEVRGeneratorConfig(
        dataset_root=args.dataset_root,
        num_tasks=args.num_tasks,
        seed=args.seed,
        num_players=args.num_players,
    ))
    tasks = generator.generate({})
    print(f"[probe] {len(tasks)} tasks | {generator.last_generation_report}")

    results = {}
    for min_px in args.min_pixels:
        print(f"[probe] min_pixels={min_px} ({min_px / 200704:.1f}x)")
        rows = evaluate(tasks, args.model_path, min_px, args.max_pixels,
                        args.max_new_tokens, args.seed)
        results[str(min_px)] = summarize(rows)

    print(f"\n{'min_pixels':>12}{'scale':>7}{'spy_acc':>9}{'ca_acc':>8}"
          f"{'ca_prior':>10}{'pred_mode':>11}{'parse_fail':>12}")
    for min_px, s in results.items():
        print(f"{min_px:>12}{int(min_px) / 200704:>6.1f}x{s['spy_acc']:>9.3f}"
              f"{s['ca_acc']:>8.3f}{s['ca_prior_acc']:>10.3f}"
              f"{s['pred_mode_share']:>11.3f}{s['parse_fail']:>12}")

    best = max(results.values(), key=lambda s: s["ca_acc"])
    print()
    if best["ca_acc"] <= best["ca_prior_acc"] + 0.05:
        print("[verdict] changed_attributes accuracy never beats the label prior. "
              "Counting is not learnable at these resolutions — change the task "
              "(e.g. predict the changed (object, attribute) set and score F1) "
              "rather than the reward.")
    else:
        print(f"[verdict] changed_attributes is learnable "
              f"(best ca_acc={best['ca_acc']:.3f} vs prior {best['ca_prior_acc']:.3f}). "
              "Train at the resolution that achieved it.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
