import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("launcher", ["ours.sh", "ours_visual_curriculum.sh"])
def test_training_launcher_without_dotenv_preserves_recipe(tmp_path, launcher):
    repo = tmp_path / "repo"
    configs = repo / "local_scripts"
    configs.mkdir(parents=True)
    (configs / "fsdp2_qwen2_5vl.json").write_text("{}")
    data = tmp_path / "data"
    (data / "output/replacement_images").mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    capture = tmp_path / "capture.json"
    python = tmp_path / "capture-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = ['SELF_EVOLVE_VISUAL_FACTS', 'SELF_EVOLVE_MASTERY_MODE', "
        "'SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION', 'SELF_EVOLVE_ANSWER_JUDGE', "
        "'SELF_EVOLVE_GRPO_EXTRA_ARGS', 'SELF_EVOLVE_SFT_EXTRA_ARGS']\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps({'args': sys.argv[1:], "
        "'env': {key: os.environ.get(key) for key in keys}}))\n"
    )
    python.chmod(0o700)
    tag = Path(launcher).stem + "_seed42"
    env = {
        "PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1",
        "WORKSPACE": str(tmp_path), "RUNS_ROOT": str(tmp_path / "runs"),
        "REPO": str(repo), "CONDA_BASE": str(tmp_path / "no-conda"),
        "CONDA_ENV": "easy-r1", "PY": str(python),
        "DATASET_ROOT": str(data), "BASE_MODEL": str(model),
        "SEED": "42", "RUN_TAG": tag, "NUM_GPUS": "8",
        "TRAINER_BACKEND": "fsdp2", "CAPTURE_PATH": str(capture),
    }
    assert not (repo / ".env").exists()
    result = subprocess.run(
        ["bash", str(ROOT / "local_scripts/self_evolve/experiments/main" / launcher)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    launch = json.loads(capture.read_text())
    args = launch["args"]
    value = lambda flag: args[args.index(flag) + 1]
    assert value("--seed") == "42"
    assert value("--trainer-num-gpus") == "8"
    assert value("--trainer-backend") == "fsdp2"
    assert value("--num-iterations") == "2"
    assert value("--num-generations") == "8"
    assert "--execute-sft-smoke" in args and "--execute-grpo-smoke" in args
    assert "--enable-openai-reference-vlm" in args
    enhanced = launcher == "ours_visual_curriculum.sh"
    marker = "ours_visual_curriculum" if enhanced else "ours"
    recorded = tmp_path / "runs/.markers" / marker
    assert Path(recorded.read_text().strip()) == tmp_path / "runs" / tag
    if enhanced:
        settings = launch["env"]
        assert settings["SELF_EVOLVE_VISUAL_FACTS"] == "1"
        assert settings["SELF_EVOLVE_MASTERY_MODE"] == "verified_visual"
        assert settings["SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION"] == "0.125"
        assert settings["SELF_EVOLVE_ANSWER_JUDGE"] == "0"
        assert value("--reward-config").endswith("reward_visual_facts.json")
        extra = shlex.split(settings["SELF_EVOLVE_GRPO_EXTRA_ARGS"])
        assert extra[extra.index("--loss_type") + 1] == "dr_grpo"
        assert extra[extra.index("--loss_normalization_length") + 1] == "512"
    else:
        assert value("--reward-config").endswith("reward_weights.json")
