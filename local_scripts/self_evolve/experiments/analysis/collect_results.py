#!/usr/bin/env python3
"""Collect the target VLMEvalKit scores without silently mixing protocols.

The official MMVP ``Overall`` is paired accuracy, while ``Average`` is the
single-question score. MCQ ``Overall`` cells are fractions; ChartQA reports a
percentage. Current VLMEvalKit may place files under per-invocation eval IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

TARGET_DATASETS = (
    "MMVP", "MMStar", "BLINK", "RealWorldQA", "AI2D_TEST",
    "ChartQA_TEST", "MMMU_Pro_10c", "CV-Bench-2D", "CV-Bench-3D",
)
EXTRA_DATASETS = ("VStarBench",)
SUPPORTED_DATASETS = TARGET_DATASETS + EXTRA_DATASETS


def _to_percentage(value: object, benchmark: str) -> Optional[float]:
    raw = str(value).strip()
    has_percent = raw.endswith("%")
    try:
        score = float(raw.rstrip("%"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0:
        return None
    # VLMEvalKit's MCQ report_acc and report_acc_MMVP return hit fractions;
    # ImageVQADataset's ChartQA relaxed accuracy already multiplies by 100.
    if benchmark != "ChartQA_TEST" and not has_percent:
        if score > 1:
            return None
        score *= 100
    return score if score <= 100 else None


def read_score(csv_path: Path, benchmark: str) -> Dict[str, Any]:
    """Read only an explicit overall metric, never the last category row."""
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"score": None, "why": f"unreadable: {exc}"}
    if not fields or not rows:
        return {"score": None, "why": "empty"}
    normalized = [field.strip().lower() for field in fields if field and field.strip()]
    if len(normalized) != len(set(normalized)):
        return {"score": None, "why": "duplicate score columns"}
    cols = {field.strip().lower(): field for field in fields if field}
    name_col = fields[0]
    overall_rows = [row for row in rows
                    if str(row.get(name_col, "")).strip().lower() in {"overall", "all"}]
    if "overall" in cols:
        if len(overall_rows) == 1:
            row = overall_rows[0]
            location = "overall row / Overall column"
        elif len(rows) == 1:
            row = rows[0]
            location = "single row / Overall column"
        else:
            return {"score": None, "why": "multiple split rows without overall row"}
        score = _to_percentage(row.get(cols["overall"]), benchmark)
        return {"score": score, "why": location if score is not None else "invalid Overall score"}
    if len(overall_rows) != 1:
        return {"score": None, "why": "no unique Overall metric"}
    for key in ("acc", "accuracy", "score", "mean"):
        if key in cols:
            score = _to_percentage(overall_rows[0].get(cols[key]), benchmark)
            return {"score": score, "why": f"overall row / {cols[key]} column" if score is not None else "invalid overall score"}
    return {"score": None, "why": "overall row has no accuracy column"}


def _score_files(model_dir: Path, label: str, benchmark: str) -> List[Path]:
    prefix = f"{label}_{benchmark}"
    return sorted(path for path in model_dir.rglob("*_acc.csv")
                  if path.name == f"{prefix}_acc.csv"
                  or (path.name.startswith(prefix + "_") and path.name.endswith("_acc.csv")))


def _protocol_signature(model_dir: Path, datasets: List[str]) -> Optional[str]:
    path = model_dir / "eval_protocol.json"
    if not path.is_file():
        return None
    try:
        protocol = json.loads(path.read_text())
        models = protocol["config"]["model"]
        if len(models) != 1:
            return None
        model_cfg = dict(next(iter(models.values())))
        model_cfg.pop("model_path", None)
        data = protocol["config"]["data"]
        if any(d not in data for d in datasets):
            return None
        signature = {
            "model_settings": model_cfg,
            "datasets": {d: data.get(d) for d in datasets},
            "judge": protocol["judge"],
            "judge_base_url": protocol.get("judge_base_url"),
            "use_vllm": protocol["use_vllm"],
            "vlmeval_commit": protocol.get("vlmeval_commit"),
            "vlmeval_code_sha256": protocol.get("vlmeval_code_sha256"),
            "dataset_fingerprints": {d: protocol.get("dataset_fingerprints", {}).get(d)
                                     for d in datasets},
            "lmu_data_path": protocol.get("lmu_data_path"),
        }
        if signature["vlmeval_commit"] is None:
            signature["vlmeval_path"] = protocol.get("vlmeval_path")
        return json.dumps(signature, sort_keys=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def scan(work_dir: Path, labels: Optional[List[str]], datasets: List[str]) -> Dict[str, Dict[str, Dict]]:
    table: Dict[str, Dict[str, Dict]] = {}
    names = labels if labels is not None else sorted(p.name for p in work_dir.iterdir() if p.is_dir())
    for label in names:
        model_dir = work_dir / label
        if not model_dir.is_dir():
            table[label] = {d: {"score": None, "why": "model directory missing"} for d in datasets}
            continue
        row: Dict[str, Dict] = {}
        for benchmark in datasets:
            paths = _score_files(model_dir, label, benchmark)
            if not paths:
                row[benchmark] = {"score": None, "why": "score file missing"}
                continue
            parsed = [(path, read_score(path, benchmark)) for path in paths]
            scores = {entry["score"] for _, entry in parsed}
            if len(scores) > 1:
                row[benchmark] = {"score": None, "why": "conflicting score files", "files": [str(p) for p in paths]}
            else:
                chosen_path, chosen = parsed[0]
                row[benchmark] = {**chosen, "file": str(chosen_path), "num_files": len(paths)}
        # Ignore VLMEvalKit's logs/status directories when labels were not requested.
        if labels is None and all(cell["why"] == "score file missing" for cell in row.values()) and not (model_dir / "eval_protocol.json").is_file():
            continue
        row["_protocol"] = {"signature": _protocol_signature(model_dir, datasets)}
        table[label] = row
    return table


def render(table: Dict[str, Dict[str, Dict]], fmt: str, datasets: List[str]) -> str:
    if not table:
        return "(no model results found)"
    protocols = [row.get("_protocol", {}).get("signature") for row in table.values()]
    # Even a single-model average needs its protocol record. Otherwise the
    # reported nine-way aggregate could silently mix old cached score files.
    comparable = bool(protocols) and all(protocols) and len(set(protocols)) == 1
    complete = all(row.get(d, {}).get("score") is not None for row in table.values() for d in datasets)
    can_average = comparable and complete
    sep = " | " if fmt == "md" else "  "
    heading = ["model", *datasets, f"AVG({len(datasets)})"]
    lines = [sep.join(heading)]
    if fmt == "md":
        lines.append(sep.join(["---"] * len(heading)))
    for label, row in sorted(table.items()):
        scores = [row.get(d, {}).get("score") for d in datasets]
        cells = [label, *(f"{v:.2f}" if v is not None else "-" for v in scores)]
        cells.append(f"{sum(scores) / len(scores):.2f}" if can_average else "-")
        lines.append(sep.join(cells))
    if not complete:
        lines.append("AVG withheld: at least one requested benchmark has no unambiguous score.")
    if not comparable:
        lines.append("AVG withheld: eval_protocol.json is missing or evaluation settings differ across models.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--labels", nargs="*", default=None, help="only these model labels")
    ap.add_argument("--datasets", nargs="+", default=list(TARGET_DATASETS))
    ap.add_argument("--format", choices=["md", "plain"], default="md")
    ap.add_argument("--out", default=None, help="also write the scores as JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if len(args.datasets) != len(set(args.datasets)) or any(d not in SUPPORTED_DATASETS for d in args.datasets):
        ap.error(f"datasets must be unique and among: {', '.join(SUPPORTED_DATASETS)}")
    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        print(f"[ERROR] work-dir not found: {work_dir}")
        return 2
    table = scan(work_dir, args.labels, args.datasets)
    print(render(table, args.format, args.datasets))
    if args.verbose:
        print("\nscore sources:")
        for label, row in sorted(table.items()):
            for benchmark in args.datasets:
                cell = row[benchmark]
                print(f"  {label:<28}{benchmark:<16}{cell['why']}  {cell.get('file', '')}")
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
