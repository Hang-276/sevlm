"""CPU-only checks for the training-stage command and launcher handoff."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
LOOP_FILE = ROOT / "local_scripts/self_evolve/workflow/run_real_input_self_evolve_loop.py"
DEFAULTS_FILE = ROOT / "local_scripts/self_evolve/configs/train_defaults.sh"
spec = importlib.util.spec_from_file_location("self_evolve_loop_for_tests", LOOP_FILE)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class TrainingStageTests(unittest.TestCase):
    def test_force_fresh_cannot_leave_previous_completion_markers_under_new_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_protocol = {"sha256": "old", "method_env": {}}
            loop.write_json(root / "experiment_protocol.json", old_protocol)
            iteration = root / "iter_000"
            iteration.mkdir()
            loop.write_json(iteration / "iteration_state.json", {"complete": True})
            with self.assertRaisesRegex(SystemExit, "fresh RUN_TAG"):
                loop._prepare_run_protocol(root, "new", {}, resume_disabled=True)
            self.assertEqual(loop.read_json(root / "experiment_protocol.json"), old_protocol)
            self.assertTrue((iteration / "iteration_state.json").is_file())

    def test_fresh_protocol_and_unchanged_resume_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop._prepare_run_protocol(root, "same", {}, resume_disabled=True)
            (root / "iter_000").mkdir()
            loop._prepare_run_protocol(root, "same", {}, resume_disabled=False)
            self.assertEqual(loop.read_json(root / "experiment_protocol.json")["sha256"], "same")
            with self.assertRaisesRegex(SystemExit, "different code"):
                loop._prepare_run_protocol(root, "changed", {}, resume_disabled=False)
            self.assertEqual(loop.read_json(root / "experiment_protocol.json")["sha256"], "same")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.iter_dir = Path(self.temp.name) / "iter_000"
        (self.iter_dir / "exports").mkdir(parents=True)
        self.base_model = str(Path(self.temp.name) / "base_qwen2_5_vl")

    def run_stage(self, summary, **overrides):
        kwargs = dict(
            dry_run=True,
            prev_solver_model_path=None,
            base_model_path=self.base_model,
            iter_dir=self.iter_dir,
            iteration_index=0,
            max_steps=1,
            execute_grpo=True,
            execute_sft=True,
            use_lora=False,
            log=lambda _: None,
        )
        kwargs.update(overrides)
        return loop.run_trainer_stage(summary, **kwargs)

    def test_empty_solver_positive_buffer_runs_grpo_from_round_model(self):
        summary = {
            "sft_replay_jsonl": str(self.iter_dir / "exports/sft_replay.jsonl"),
            "num_sft_replay": 3,  # proposer examples do not make SFT viable
            "num_sft_solver": 0,
            "grpo_tasks_jsonl": str(self.iter_dir / "exports/grpo_tasks.jsonl"),
            "grpo_data_config": str(self.iter_dir / "exports/grpo_tasks.yaml"),
            "num_grpo_tasks": 2,
        }
        launched = []

        def fake_run(cmd, *, cwd, env):
            launched.append((cmd, env))
            ckpt = self.iter_dir / "checkpoints/grpo_qwen2_5_vl"
            ckpt.mkdir(parents=True)
            (ckpt / "config.json").write_text("{}")
            (ckpt / "model.safetensors").write_bytes(b"fake weights")
            return SimpleNamespace(returncode=0)

        scale = loop.TrainerScale(trainer_backend="fsdp2", fsdp_config="fsdp.json")
        with patch("subprocess.run", side_effect=fake_run):
            result = self.run_stage(summary, dry_run=False, scale=scale)

        self.assertEqual(len(launched), 1)
        cmd, env = launched[0]
        self.assertTrue(any(part.endswith("grpo_jsonl.py") for part in cmd))
        self.assertEqual(cmd[cmd.index("--model_name_or_path") + 1], self.base_model)
        self.assertEqual(env["FSDP_VERSION"], "2")
        self.assertEqual(env["FSDP_STATE_DICT_TYPE"], "FULL_STATE_DICT")
        self.assertEqual(result["solver_lineage"]["training_recipe"], "grpo")
        self.assertIsNone(result["solver_lineage"]["post_sft_merged_model_path"])
        self.assertEqual(result["solver_model_path_status"], "created")

    def test_success_without_model_weights_is_a_failed_stage(self):
        summary = {"grpo_tasks_jsonl": str(self.iter_dir / "exports/grpo_tasks.jsonl"),
                   "num_grpo_tasks": 2}

        def fake_run(cmd, *, cwd, env):
            checkpoint = self.iter_dir / "checkpoints/grpo_qwen2_5_vl"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            return SimpleNamespace(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            result = self.run_stage(summary, dry_run=False, execute_sft=False)
        self.assertEqual(result["training_pathways"]["grpo"]["status"], "trainer_failed")
        self.assertFalse(result["checkpoint_created"])

    def test_success_without_updating_existing_weights_is_a_failed_stage(self):
        checkpoint = self.iter_dir / "checkpoints/grpo_qwen2_5_vl"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"old weights")
        summary = {"grpo_tasks_jsonl": str(self.iter_dir / "exports/grpo_tasks.jsonl"),
                   "num_grpo_tasks": 2}
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            result = self.run_stage(summary, dry_run=False, execute_sft=False)
        self.assertEqual(result["training_pathways"]["grpo"]["status"], "trainer_failed")
        self.assertFalse(result["checkpoint_created"])

    def test_solver_resume_requires_exact_task_groups(self):
        tasks = [{"task_id": "a"}, {"task_id": "b"}]
        valid = [{"task_id": key} for key in ("a", "a", "b", "b")]
        loop._validate_solver_replay(valid, tasks, 2)
        for corrupt in (valid[:3], valid + [{"task_id": "unknown"}],
                        [{"task_id": "a"}] * 4):
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                loop._validate_solver_replay(corrupt, tasks, 2)

    def test_checkpoint_validation_covers_shards_and_adapters(self):
        checkpoint = self.iter_dir / "model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"first": "model-00001-of-00002.safetensors",
                           "second": "model-00002-of-00002.safetensors"}}))
        (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"weights")
        self.assertFalse(loop._has_model_checkpoint(checkpoint))
        (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"weights")
        self.assertTrue(loop._has_model_checkpoint(checkpoint))
        adapter = self.iter_dir / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.bin").write_bytes(b"weights")
        self.assertTrue(loop._has_model_checkpoint(adapter))

    def test_corrupt_or_incomplete_completion_marker_cannot_skip_iteration(self):
        marker = self.iter_dir / "iteration_state.json"
        self.assertFalse(loop._completed_iteration(self.iter_dir, 0))
        marker.write_text("{")
        with self.assertRaisesRegex(RuntimeError, "Cannot resume completed iteration"):
            loop._completed_iteration(self.iter_dir, 0)
        marker.write_text(json.dumps({"iteration_index": 0, "iteration_id": "iter_000",
                                     "solver_model_path": None}))
        with self.assertRaisesRegex(RuntimeError, "Cannot resume completed iteration"):
            loop._completed_iteration(self.iter_dir, 0)
        for name in ("generator_policy.json", "failure_profile.json"):
            (self.iter_dir / name).write_text("{}")
        (self.iter_dir / "solver_update_state.json").write_text(json.dumps({
            "iteration_id": "iter_000", "solver_model_path": None,
            "checkpoint_created": False}))
        self.assertTrue(loop._completed_iteration(self.iter_dir, 0))

    def test_parallel_reference_judging_does_not_overshoot_quota(self):
        tasks = [{"task_id": str(i), "scene_id": str(i)} for i in range(10)]
        with patch.object(loop, "_dry_run_reference_judgment", return_value={"accepted": True}), \
             patch.object(loop, "_gate_reference_judgment", side_effect=lambda raw, *a, **kw: raw):
            result = loop.run_reference_stage(
                tasks, dry_run=True, provider="openai", model="fake", base_url="unused",
                dataset_root=str(self.iter_dir), quota_target=3, max_workers=8,
                judge_budget_factor=3, solvability_threshold=0.5, reject_on_ambiguity=True,
                generator=None, generator_policy={}, failure_profile=None,
                max_regenerate_attempts=0, log=lambda _: None)
        self.assertEqual(len(result["accepted_tasks"]), 3)
        self.assertEqual(result["num_judged"], 3)

    def test_reference_resume_preserves_actual_shortfall_and_rejections(self):
        candidates = [{"task_id": "accepted"}, {"task_id": "rejected"}]
        accepted = [{"task_id": "accepted"}]
        feedback = [{"task_id": "accepted", "accepted": True},
                    {"task_id": "rejected", "accepted": False}]
        state = {"mode": "live_openai", "num_accepted": 1, "num_judged": 2,
                 "num_rejected": 1, "reference_accounting": {
                     "accepted": 1, "judged": 2, "quota_target": 3,
                     "quota_shortfall": 2, "reject_count": 1, "judge_budget": 6}}
        state.update({"candidate_sha256": loop._records_digest(candidates),
                      "accepted_sha256": loop._records_digest(accepted),
                      "feedback_sha256": loop._records_digest(feedback)})
        loop.write_jsonl(self.iter_dir / "candidate_tasks.jsonl", candidates)
        loop.write_jsonl(self.iter_dir / "accepted_tasks.jsonl", accepted)
        loop.write_jsonl(self.iter_dir / "reference_feedback.jsonl", feedback)
        loop.write_json(self.iter_dir / "reference_stage_state.json", state)
        resumed = loop._restore_reference_stage(self.iter_dir)
        self.assertEqual(resumed["reference_accounting"], state["reference_accounting"])
        self.assertEqual(resumed["num_rejected"], 1)
        self.assertEqual(resumed["mode"], "live_openai")
        for filename, records in (("candidate_tasks.jsonl", candidates),
                                  ("accepted_tasks.jsonl", accepted),
                                  ("reference_feedback.jsonl", feedback)):
            with self.subTest(filename=filename):
                loop.write_jsonl(self.iter_dir / filename, [{**records[0], "task_id": "stale"},
                                                           *records[1:]])
                with self.assertRaisesRegex(RuntimeError, "inconsistent Reference VLM stage"):
                    loop._restore_reference_stage(self.iter_dir)
                loop.write_jsonl(self.iter_dir / filename, records)

    def test_positive_solver_buffer_hands_sft_model_to_grpo(self):
        sft_path = self.iter_dir / "exports/sft_replay.jsonl"
        sft_path.write_text("{}\n")
        summary = {
            "sft_replay_jsonl": str(sft_path),
            "num_sft_replay": 1,
            "num_sft_solver": 1,
            "grpo_tasks_jsonl": str(self.iter_dir / "exports/grpo_tasks.jsonl"),
            "num_grpo_tasks": 1,
        }
        result = self.run_stage(summary)

        expected = str(self.iter_dir / "checkpoints/sft_qwen2_5_vl")
        cmd = result["grpo_command"]
        self.assertEqual(cmd[cmd.index("--model_name_or_path") + 1], expected)
        self.assertEqual(result["solver_lineage"]["training_recipe"], "sft_grpo")
        self.assertIsNotNone(result["sft_command"])

    def test_verified_targets_can_start_sft_without_solver_positive(self):
        sft_path = self.iter_dir / "exports/sft_replay.jsonl"
        sft_path.write_text("{}\n")
        summary = {
            "sft_replay_jsonl": str(sft_path),
            "num_sft_replay": 4,
            "num_sft_solver": 0,
            "num_sft_non_proposer": 4,
            "num_sft_verified_oracle": 1,
            "num_sft_scene_qa": 3,
            "grpo_tasks_jsonl": str(self.iter_dir / "exports/grpo_tasks.jsonl"),
            "num_grpo_tasks": 1,
        }
        result = self.run_stage(summary)
        self.assertEqual(result["solver_lineage"]["training_recipe"], "sft_grpo")
        self.assertIsNotNone(result["sft_command"])

    def test_offline_solver_default_tracks_grpo_completion_length(self):
        env = dict(os.environ)
        env.pop("GRPO_MAX_COMPLETION_LEN", None)
        env.pop("SOLVER_MAX_NEW_TOKENS", None)
        cmd = ["bash", "-c", f'source "{DEFAULTS_FILE}"; printf "%s %s" "$GRPO_MAX_COMPLETION_LEN" "$SOLVER_MAX_NEW_TOKENS"']
        for grpo_length, expected in [(None, "2048 2048"), ("1536", "1536 1536")]:
            case_env = dict(env)
            if grpo_length:
                case_env["GRPO_MAX_COMPLETION_LEN"] = grpo_length
            result = subprocess.run(cmd, env=case_env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_sft_pixel_limits_are_opt_in_and_reach_the_trainer(self):
        base_env = dict(os.environ)
        for name in ("SFT_MIN_PIXELS", "SFT_MAX_PIXELS"):
            base_env.pop(name, None)
        shell = ["bash", "-c", f'source "{DEFAULTS_FILE}"; build_sft_extra']
        default = subprocess.run(shell, env=base_env, capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertNotIn("--min_pixels", default.stdout)
        self.assertNotIn("--max_pixels", default.stdout)

        case_env = {**base_env, "SFT_MIN_PIXELS": "401408", "SFT_MAX_PIXELS": "602112"}
        configured = subprocess.run(shell, env=case_env, capture_output=True, text=True)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        with patch.dict(os.environ, {"SELF_EVOLVE_SFT_EXTRA_ARGS": configured.stdout}):
            cmd = loop._build_sft_command("data.yaml", self.base_model, "out", 2)
        self.assertEqual(cmd[cmd.index("--min_pixels") + 1], "401408")
        self.assertEqual(cmd[cmd.index("--max_pixels") + 1], "602112")


if __name__ == "__main__":
    unittest.main()
