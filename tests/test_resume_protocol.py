"""Scored replay must not reuse a different reward or trajectory batch."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.iteration import score_and_route_trajectories
from open_r1.self_evolve.reward_config import load_reward_config
from open_r1.self_evolve.protocol_fingerprint import self_evolve_protocol_fingerprint


class ResumeProtocolTests(unittest.TestCase):
    def test_rollout_and_prompt_dependencies_change_the_run_fingerprint(self):
        repo = Path(__file__).resolve().parents[1]
        config = repo / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json"
        initial = self_evolve_protocol_fingerprint(config)
        original_read = Path.read_bytes
        for source in ("src/open_r1/grpo_data.py", "src/open_r1/trainer/vllm_grpo_trainer.py",
                       "src/open_r1/trainer/vllm_rollout.py", "src/open_r1/trainer/vllm_rollout_worker.py"):
            target = repo / source

            def altered_read(path):
                content = original_read(path)
                return content + b"\nchanged dependency\n" if path == target else content

            with patch.object(Path, "read_bytes", altered_read):
                self.assertNotEqual(self_evolve_protocol_fingerprint(config), initial, source)

    def test_scored_stream_rejects_changed_input_or_reward_code(self):
        visual_config = (
            Path(__file__).resolve().parents[1]
            / "local_scripts/self_evolve/configs/reward/reward_visual_facts.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "reward.json"
            config_path.write_bytes(visual_config.read_bytes())
            config = load_reward_config(config_path)
            stream_path = Path(directory) / "scored_trajectories.jsonl"
            trajectory = {
                "task_id": "one", "problem": "Find the spy",
                "solution": "<answer>spy=2; changed_attributes=1</answer>",
                "completion": "<think>Player 2 differs.</think>"
                              "<answer>spy=2; changed_attributes=1</answer>",
                "metadata": {"spy_player": 2},
            }
            with patch.dict("os.environ", {"SELF_EVOLVE_ANSWER_JUDGE": "0"}):
                first = score_and_route_trajectories(
                    [trajectory], reward_config=config, stream_path=stream_path
                )
                resumed = score_and_route_trajectories(
                    [trajectory], reward_config=config, stream_path=stream_path
                )
                self.assertEqual(first, resumed)
                self.assertEqual(len(stream_path.read_text().splitlines()), 1)
                changed = {**trajectory, "completion": trajectory["completion"] + " changed"}
                with self.assertRaisesRegex(RuntimeError, "another task/reward/code protocol"):
                    score_and_route_trajectories(
                        [changed], reward_config=config, stream_path=stream_path
                    )
                config_path.write_text(json.dumps({
                    **json.loads(config_path.read_text()), "notes": "changed protocol"
                }))
                with self.assertRaisesRegex(RuntimeError, "another task/reward/code protocol"):
                    score_and_route_trajectories(
                        [trajectory], reward_config=config, stream_path=stream_path
                    )


if __name__ == "__main__":
    unittest.main()
