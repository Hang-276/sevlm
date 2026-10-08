#!/usr/bin/env python3
"""Reject skipped evaluations and incomplete VLMEvalKit predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from collect_results import read_score

PRED_SUFFIXES = (".xlsx", ".tsv", ".csv", ".json")
FAIL_MARKER = "Failed to obtain answer"
MISSING_IMAGE = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                 "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
                 "n/a", "nan", "null"}


def status_snapshot(result_dir: Path) -> dict[str, list[int]]:
    return {str(path): [path.stat().st_mtime_ns, path.stat().st_size]
            for path in result_dir.rglob("status.json") if not path.is_symlink()}


def dataset_fingerprints(lmu_data: Path, datasets) -> dict[str, str | None]:
    fingerprints = {}
    for dataset in datasets:
        source = lmu_data / f"{dataset}.tsv"
        if not source.is_file():
            fingerprints[dataset] = None
            continue
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[dataset] = digest.hexdigest()
    return fingerprints


def read_table(path: Path):
    if path.suffix in (".csv", ".tsv"):
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
    elif path.suffix == ".xlsx":
        import pandas as pd
        yield from pd.read_excel(path, dtype=str, keep_default_na=False).to_dict("records")
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "columns" in data and "data" in data:
            data = [dict(zip(data["columns"], row)) for row in data["data"]]
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError(f"unsupported prediction table: {path}")
        yield from data
    else:
        raise ValueError(f"unsupported prediction format: {path}")


def check_predictions(prediction: Path, dataset: str, lmu_data: Path) -> int:
    if not prediction.is_file():
        raise ValueError(f"prediction file missing: {prediction}")
    rows = list(read_table(prediction))
    if not rows or any("index" not in row or "prediction" not in row for row in rows):
        raise ValueError("prediction table is empty or lacks index/prediction")
    indices = [str(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate prediction indices")
    failed = sum(FAIL_MARKER in str(row["prediction"]) for row in rows)
    empty = sum(row["prediction"] is None or not str(row["prediction"]).strip() for row in rows)
    if failed or empty:
        raise ValueError(f"inference failed={failed}, empty={empty}, total={len(rows)}")
    source = lmu_data / f"{dataset}.tsv"
    if not source.is_file():
        raise ValueError(f"dataset TSV missing; cannot verify completeness: {source}")
    try:
        from pandas._libs.parsers import STR_NA_VALUES as missing_image
    except ImportError:
        missing_image = MISSING_IMAGE
    source_rows = [row for row in read_table(source)
                   if "image" not in row or row["image"] not in missing_image]
    source_indices = [str(row["index"]) for row in source_rows]
    if (len(source_indices) != len(set(source_indices))
            or any(not value.strip() for value in source_indices)):
        raise ValueError("dataset TSV has duplicate or empty indices")
    expected = set(source_indices)
    missing, extra = expected - set(indices), set(indices) - expected
    if not expected or missing or extra:
        raise ValueError(f"incomplete predictions: expected={len(expected)}, actual={len(rows)}, "
                         f"missing={len(missing)}, extra={len(extra)}")
    return len(rows)


def check_scores(prediction: Path, label: str, dataset: str) -> list[Path]:
    prefix = f"{label}_{dataset}"
    scores = [path for path in prediction.parent.glob("*_acc.csv")
              if path.name == f"{prefix}_acc.csv" or path.name.startswith(prefix + "_")]
    if not scores:
        raise ValueError("score artifact missing (*_acc.csv)")
    parsed = [read_score(path, dataset)["score"] for path in scores]
    if any(score is None for score in parsed) or len(set(parsed)) != 1:
        raise ValueError("score artifacts contain invalid or conflicting overall metrics")
    return scores


def check_results(result_dir: Path, label: str, mode: str, datasets: list[str],
                  lmu_data: Path, previous: dict | None = None, require_status: bool = False,
                  since: float | None = None) -> list[str]:
    stamps = status_snapshot(result_dir)
    eligible = [Path(path) for path, stamp in stamps.items()
                if (previous is None or stamp != previous.get(path))
                and (since is None or stamp[0] >= since * 1e9)]
    if (stamps or require_status) and not eligible:
        return ["no status.json from this invocation"]
    status = None
    status_file = None
    if eligible:
        status_file = max(eligible, key=lambda path: path.stat().st_mtime_ns)
        try:
            status = json.loads(status_file.read_text())
            if status.get("model_name") != label or not isinstance(status.get("datasets"), dict):
                raise ValueError("model_name/datasets do not match this evaluation")
            if status.get("mode", mode) != mode:
                raise ValueError("run mode does not match this evaluation")
        except (OSError, ValueError, AttributeError) as exc:
            return [f"invalid status.json: {exc}"]
    errors = []
    for dataset in datasets:
        try:
            if status is not None:
                entry = status["datasets"].get(dataset)
                if not isinstance(entry, dict) or entry.get("status") != "done":
                    raise ValueError("dataset did not finish")
                if entry.get("error_message"):
                    raise ValueError(f"evaluation error: {entry['error_message']}")
                skip = entry.get("skip_reason")
                if skip and not (mode == "infer" and skip == "mode_infer"):
                    raise ValueError(f"evaluation skipped: {skip}")
                filename = entry.get("prediction_file")
                if not filename:
                    raise ValueError("status has no prediction_file")
                prediction = Path(filename)
                if not prediction.is_absolute():
                    prediction = status_file.parent / prediction
                if (not prediction.resolve().is_relative_to(result_dir.resolve())
                        or prediction.stem != f"{label}_{dataset}"):
                    raise ValueError("status prediction_file belongs to another evaluation")
            else:
                candidates = [path for suffix in PRED_SUFFIXES
                              for path in result_dir.rglob(f"{label}_{dataset}{suffix}")]
                if not candidates:
                    raise ValueError("prediction file missing (legacy VLMEvalKit)")
                prediction = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            total = check_predictions(prediction, dataset, lmu_data)
            artifacts = [prediction]
            if mode != "infer":
                artifacts += check_scores(prediction, label, dataset)
            if (status is None and since is not None
                    and all(path.stat().st_mtime_ns < since * 1e9 for path in artifacts)):
                raise ValueError("no fresh predictions or scores from this invocation (legacy VLMEvalKit)")
            print(f"[verified] {dataset}: {total} predictions, inference failed=0"
                  + (", score artifact valid" if mode != "infer" else ""))
        except (OSError, ValueError, KeyError, ImportError) as exc:
            errors.append(f"{dataset}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--lmu-data", type=Path)
    parser.add_argument("--previous-status", help="JSON emitted by --snapshot")
    parser.add_argument("--require-status", action="store_true")
    parser.add_argument("--since", type=float, help="invocation start time in Unix seconds")
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(status_snapshot(args.result_dir)))
        return 0
    if not args.label or not args.datasets or args.lmu_data is None:
        parser.error("--label, --datasets and --lmu-data are required for verification")
    previous = json.loads(args.previous_status) if args.previous_status is not None else None
    errors = check_results(args.result_dir, args.label, args.mode, args.datasets,
                           args.lmu_data, previous, args.require_status, args.since)
    if not errors and (args.result_dir / "eval_protocol.json").is_file():
        protocol_file = args.result_dir / "eval_protocol.json"
        protocol = json.loads(protocol_file.read_text())
        fingerprints = dataset_fingerprints(
            args.lmu_data, protocol["config"]["data"])
        for name, previous in protocol.get("dataset_fingerprints", {}).items():
            if previous is not None and previous != fingerprints.get(name):
                errors.append(f"{name}: dataset TSV changed during evaluation")
        if not errors:
            protocol["dataset_fingerprints"] = fingerprints
            temporary = protocol_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(protocol, indent=2))
            temporary.replace(protocol_file)
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
