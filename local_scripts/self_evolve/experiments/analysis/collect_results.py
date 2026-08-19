#!/usr/bin/env python3
"""Collect VLMEvalKit results into one model x benchmark table.

    collect_results.py --work-dir <eval/results> [--labels a b c] [--format md]

Benchmarks use different column names in their *_acc.csv, so the score is looked
up by priority (Overall / Acc / last numeric column). A cell that cannot be read
is reported with the reason rather than guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Column names to try, in order, lowercased.
SCORE_COLUMNS = ["overall", "acc", "accuracy", "score", "avg", "mean"]
# Row names that hold the overall score in per-split tables.
OVERALL_ROWS = ["overall", "all", "avg", "average", "total"]


def _to_float(value: str) -> Optional[float]:
    try:
        f = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return f * 100 if 0.0 <= f <= 1.0 and "." in str(value) else f


def read_score(csv_path: Path) -> Dict[str, object]:
    """Overall score from one *_acc.csv, plus how it was found."""
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        return {"score": None, "why": f"unreadable: {exc}"}
    if not rows:
        return {"score": None, "why": "empty"}

    cols = {c.lower().strip(): c for c in rows[0].keys() if c}

    # An explicit overall row wins.
    first_col = list(rows[0].keys())[0]
    for row in rows:
        name = str(row.get(first_col, "")).lower().strip()
        if name in OVERALL_ROWS:
            for key in SCORE_COLUMNS:
                if key in cols:
                    v = _to_float(row[cols[key]])
                    if v is not None:
                        return {"score": v, "why": f"row {name} / column {cols[key]}"}

    # Otherwise an overall column.
    for key in SCORE_COLUMNS:
        if key in cols:
            v = _to_float(rows[-1][cols[key]])
            if v is not None:
                return {"score": v, "why": f"column {cols[key]} (last row)"}

    # Last resort: the last numeric cell of the last row.
    for col in reversed(list(rows[-1].keys())):
        v = _to_float(rows[-1][col])
        if v is not None:
            return {"score": v, "why": f"column {col} (last row, fallback)"}
    return {"score": None, "why": "no numeric column"}


def scan(work_dir: Path, labels: Optional[List[str]]) -> Dict[str, Dict[str, Dict]]:
    table: Dict[str, Dict[str, Dict]] = {}
    for model_dir in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        label = model_dir.name
        if labels and label not in labels:
            continue
        for csv_path in sorted(model_dir.glob("*acc.csv")):
            bench = re.sub(r"^%s[_-]?" % re.escape(label), "", csv_path.stem)
            bench = re.sub(r"[_-]?acc$", "", bench) or csv_path.stem
            table.setdefault(label, {})[bench] = {
                **read_score(csv_path), "file": str(csv_path)}
    return table


def render(table: Dict[str, Dict[str, Dict]], fmt: str) -> str:
    """Model x benchmark table.

    AVG covers only the benchmarks every model has a score for; averaging over
    different subsets per row would not be comparable.
    """
    if not table:
        return "(no *acc.csv found)"
    benches = sorted({b for row in table.values() for b in row})
    common = [b for b in benches
              if all(table[m].get(b, {}).get("score") is not None for m in table)]
    sep = " | " if fmt == "md" else "  "
    lines = []
    head = ["model"] + benches + [f"AVG({len(common)})"]
    lines.append(sep.join(head))
    if fmt == "md":
        lines.append(sep.join(["---"] * len(head)))

    for label in sorted(table):
        cells = [label]
        for b in benches:
            score = table[label].get(b, {}).get("score")
            cells.append(f"{score:.2f}" if score is not None else "-")
        got = [table[label][b]["score"] for b in common]
        cells.append(f"{sum(got)/len(got):.2f}" if got else "-")
        lines.append(sep.join(cells))

    if len(common) < len(benches):
        skipped = [b for b in benches if b not in common]
        lines += ["",
                  f"AVG over the {len(common)} benchmarks every model has: {', '.join(common)}",
                  f"excluded (some model has no score): {', '.join(skipped)}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--labels", nargs="*", default=None, help="only these model labels")
    ap.add_argument("--format", choices=["md", "plain"], default="md")
    ap.add_argument("--out", default=None, help="also write the table as json")
    ap.add_argument("--verbose", action="store_true", help="show how each score was located")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        print(f"[ERROR] work-dir not found: {work_dir}")
        return 2

    table = scan(work_dir, args.labels)
    print(render(table, args.format))

    if args.verbose:
        print("\nscore sources:")
        for label, row in sorted(table.items()):
            for bench, cell in sorted(row.items()):
                print(f"  {label:<28}{bench:<16}{cell['why']}")

    missing = [(l, b) for l, row in table.items() for b, c in row.items() if c["score"] is None]
    if missing:
        print(f"\n[WARN] {len(missing)} cell(s) had no readable score; --verbose for why")

    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
