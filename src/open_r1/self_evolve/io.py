"""Small JSON/JSONL helpers for self-evolution artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def read_jsonl(path: str | Path) -> List[dict[str, Any]]:
    input_path = Path(path)
    records: List[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomic write: stage-complete resume markers rely on file presence."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output_path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
