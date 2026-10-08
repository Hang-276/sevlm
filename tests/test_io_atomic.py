from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.self_evolve.io import read_json, read_jsonl, write_json, write_jsonl


def test_failed_json_serialization_preserves_previous_marker_and_cleans_temp(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"complete": False})
    with pytest.raises(TypeError):
        write_json(path, {"complete": True, "bad": object()})
    assert read_json(path) == {"complete": False}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_jsonl_generator_cannot_publish_partial_data(tmp_path):
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [{"old": True}])

    def broken():
        yield {"new": True}
        raise RuntimeError("interrupted generator")

    with pytest.raises(RuntimeError, match="interrupted"):
        write_jsonl(path, broken())
    assert read_jsonl(path) == [{"old": True}]
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_writers_publish_one_complete_artifact(tmp_path):
    path = tmp_path / "tasks.jsonl"
    barrier = threading.Barrier(2)

    def write(value):
        def records():
            yield {"value": value, "index": 0}
            barrier.wait(timeout=5)
            yield {"value": value, "index": 1}
        write_jsonl(path, records())

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, [1, 2]))
    rows = read_jsonl(path)
    assert len(rows) == 2 and rows[0]["value"] == rows[1]["value"]
    assert [row["index"] for row in rows] == [0, 1]
    assert list(tmp_path.iterdir()) == [path]
