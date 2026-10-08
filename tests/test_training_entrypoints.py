import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["open_r1.grpo_jsonl", "open_r1.sft_jsonl"])
def test_training_module_cli_imports_in_existing_environment(module):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--model_name_or_path" in result.stdout


def test_refined_reward_entrypoint_does_not_require_optional_detection_dependency():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("OPENAI_API_KEY", None)
    script = """
import sys
sys.modules['pycocotools'] = None
from open_r1.grpo_jsonl import client, llm_reward, reward_funcs_registry
assert client is None
assert llm_reward('<answer>2</answer>', '<answer>2</answer>') == 1.0
assert llm_reward('<answer>7</answer>', '<answer>2</answer>') == 0.0
context = {'task_kind': 'verified_count_qa', 'verified_qa': {'gold_count': 2}}
reward = reward_funcs_registry['self_evolve_refined'](
    ['<answer>2</answer>', '<answer>7</answer>'],
    solution=['<answer>2</answer>'] * 2, self_evolve=[context] * 2,
)
assert reward == [1.0, 0.0], reward
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
