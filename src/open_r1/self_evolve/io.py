"""Small JSON/JSONL helpers for self-evolution artifacts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
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


@contextmanager
def _atomic_output(path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            yield f
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(output_path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Publish only complete artifacts; failed writes preserve the old file."""
    with _atomic_output(path) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    with _atomic_output(path) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
