import ast
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from open_r1.self_evolve.dotenv_loader import _candidate_paths, load_dotenv


@pytest.mark.parametrize("subdirectory", ["", "local_scripts/workflow"])
def test_env_search_stops_at_project_root_without_git(tmp_path, monkeypatch, subdirectory):
    project = tmp_path / "project"
    (project / "src/open_r1").mkdir(parents=True)
    (project / "setup.py").touch()
    current = project / subdirectory
    current.mkdir(parents=True, exist_ok=True)
    key = "SEVLM_ARCHIVE_ENV_TEST"
    monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(current)
    (tmp_path / ".env").write_text(f"{key}=outside\n")

    assert not (project / ".git").exists()
    assert _candidate_paths(None)[-1] == project / ".env"
    assert load_dotenv() == []
    assert key not in os.environ

    (project / ".env").write_text(f"{key}=inside\n")
    assert load_dotenv() == [key]
    assert os.environ[key] == "inside"


@pytest.mark.parametrize("missing_git", [False, True])
def test_eval_version_falls_back_silently_without_git(tmp_path, monkeypatch, capfd, missing_git):
    source = Path(__file__).resolve().parents[1] / "eval/VLMEvalKit/vlmeval/smp/misc.py"
    if not source.is_file():
        pytest.skip("VLMEvalKit is not installed locally")
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_minimal_ext_cmd", "githash"}]
    assert len(functions) == 2
    namespace = {"os": os, "subprocess": subprocess}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    package = ModuleType("vlmeval")
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "vlmeval", package)
    if missing_git:
        monkeypatch.setenv("PATH", str(tmp_path))

    assert namespace["githash"]() == "unknown"
    assert capfd.readouterr().err == ""
