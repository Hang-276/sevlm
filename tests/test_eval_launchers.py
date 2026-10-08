"""CPU regressions for evaluation launchers and artifact validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'local_scripts/self_evolve/experiments'


class EvalLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'base model'
        self.ckpt = self.root / 'trained "model"'
        for model in (self.base, self.ckpt):
            model.mkdir()
            (model / 'config.json').write_text('{}')
        self.vlm = self.root / 'fake vlmeval'
        self.vlm.mkdir()
        (self.vlm / 'run.py').write_text(
            "# parser.add_argument('--reuse')\n"
            '''import json, os, sys
from pathlib import Path
Path(os.environ["FAKE_CALLED"]).write_text(json.dumps(sys.argv[1:]))
print("fake evaluator; no inference")
if os.environ.get("FAKE_NO_OUTPUT") or os.environ.get("FAKE_STALE_STATUS"):
    sys.exit(0)
args = sys.argv[1:]
config = json.loads(Path(args[args.index("--config") + 1]).read_text())
label = next(iter(config["model"]))
mode = args[args.index("--mode") + 1]
root = Path(args[args.index("--work-dir") + 1]) / label / "T_fake"
root.mkdir(exist_ok=True)
status = {"model_name": label, "mode": mode, "datasets": {}}
for dataset in config["data"]:
    if os.environ.get("FAKE_DROPPED_DATASET") == dataset:
        continue
    filename = f"{label}_{dataset}.tsv"
    prediction = "Failed to obtain answer" if os.environ.get("FAKE_FAILED_PRED") else "A"
    if os.environ.get("FAKE_EMPTY_PRED"):
        prediction = ""
    rows = f"index\\tprediction\\n0\\t{prediction}\\n"
    if not os.environ.get("FAKE_MISSING_ROW"):
        rows += "1\\tB\\n"
    if not os.environ.get("FAKE_KEEP_PRED"):
        (root / filename).write_text(rows)
    entry = {"status": "done", "prediction_file": filename}
    if mode == "infer":
        entry["skip_reason"] = "mode_infer"
    if os.environ.get("FAKE_ERROR"):
        entry["error_message"] = "simulated dataset failure"
    if os.environ.get("FAKE_SKIP"):
        entry["skip_reason"] = os.environ["FAKE_SKIP"]
    if os.environ.get("FAKE_PENDING"):
        entry["status"] = "pending"
    status["datasets"][dataset] = entry
    if mode != "infer" and not os.environ.get("FAKE_MISSING_SCORE"):
        score = "NaN" if os.environ.get("FAKE_BAD_SCORE") else "0.5"
        (root / f"{label}_{dataset}_acc.csv").write_text(f"split,Overall\\nnone,{score}\\n")
if os.environ.get("FAKE_MUTATE_DATASET"):
    source = Path(os.environ["LMUData"]) / (os.environ["FAKE_MUTATE_DATASET"] + ".tsv")
    source.write_text(source.read_text().replace("zero.png", "changed.png"))
if not os.environ.get("FAKE_LEGACY"):
    (root / "status.json").write_text(json.dumps(status))
'''
        )
        data_dir = self.root / 'data'
        data_dir.mkdir()
        for dataset in ('MMVP', 'MMStar', 'BLINK', 'RealWorldQA', 'AI2D_TEST',
                        'ChartQA_TEST', 'MMMU_Pro_10c', 'CV-Bench-2D', 'CV-Bench-3D'):
            (data_dir / f'{dataset}.tsv').write_text('index\timage_path\n0\tzero.png\n1\tone.png\n')
        self.marker = self.root / 'called.json'
        self.env = {
            'PATH': os.environ['PATH'], 'HOME': str(self.root),
            'PYTHONDONTWRITEBYTECODE': '1',
            'EVAL_PY': sys.executable, 'BASE_MODEL': str(self.base),
            'VLMK': str(self.vlm), 'LMUData': str(self.root / 'data'),
            'WORK_DIR': str(self.root / 'results'),
            'ENV_FILE': str(self.root / 'absent.env'),
            'JUDGE': 'exact_matching', 'DATASETS': 'MMVP MMStar',
            'FAKE_CALLED': str(self.marker),
        }

    def run_script(self, name='analysis/eval_checkpoint.sh', args=(), **overrides):
        env = {**self.env, **overrides}
        return subprocess.run(['bash', str(EXP / name), *map(str, args)],
                              env=env, text=True, capture_output=True)

    def config(self, label):
        return json.loads((self.root / 'results' / label / 'eval_config.json').read_text())

    def test_env_checkpoint_label_and_quoted_path_without_dotenv(self):
        result = self.run_script(CKPT=str(self.ckpt), LABEL='vision_zero')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.config('vision_zero')['model']['vision_zero']['model_path'], str(self.ckpt))
        self.assertTrue(self.marker.exists())
        self.assertIn('fake evaluator', (self.root / 'results/vision_zero/eval.log').read_text())

    def test_cli_checkpoint_overrides_environment(self):
        result = self.run_script(args=[self.ckpt], CKPT=str(self.base), LABEL='cli')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.config('cli')['model']['cli']['model_path'], str(self.ckpt))

    def test_base_never_uses_inherited_checkpoint(self):
        result = self.run_script('main/base_zeroshot.sh', CKPT=str(self.ckpt))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.config('base')['model']['base']['model_path'], str(self.base))

    def test_preview_never_launches_evaluator_or_requires_api(self):
        result = self.run_script(DRY_RUN='1', JUDGE='gpt-4o-mini', LABEL='preview')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.marker.exists())
        self.assertEqual(list(self.config('preview')['data']), ['MMVP', 'MMStar'])

    def test_default_is_exactly_the_nine_target_benchmarks(self):
        result = self.run_script(DATASETS='', DRY_RUN='1', LABEL='nine')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(self.config('nine')['data']), [
            'MMVP', 'MMStar', 'BLINK', 'RealWorldQA', 'AI2D_TEST',
            'ChartQA_TEST', 'MMMU_Pro_10c', 'CV-Bench-2D', 'CV-Bench-3D',
        ])

    def test_dataset_subset_can_expand_without_changing_cache_protocol(self):
        self.assertEqual(self.run_script(DRY_RUN='1', LABEL='incremental', DATASETS='MMVP').returncode, 0)
        result = self.run_script(DRY_RUN='1', LABEL='incremental', DATASETS='MMVP BLINK')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        protocol = json.loads((self.root / 'results/incremental/eval_protocol.json').read_text())
        self.assertEqual(list(protocol['config']['data']), ['MMVP', 'BLINK'])
        self.assertEqual(self.run_script(DRY_RUN='1', LABEL='incremental', DATASETS='BLINK').returncode, 0)
        protocol = json.loads((self.root / 'results/incremental/eval_protocol.json').read_text())
        self.assertEqual(list(protocol['config']['data']), ['MMVP', 'BLINK'])

    def test_checkpoint_file_change_cannot_reuse_stale_predictions(self):
        self.assertEqual(self.run_script(DRY_RUN='1', LABEL='frozen').returncode, 0)
        (self.base / 'config.json').write_text('{"changed": true}')
        result = self.run_script(DRY_RUN='1', LABEL='frozen')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('protocol changed', result.stderr)

    def test_legacy_evaluator_cannot_reuse_results_without_running(self):
        first = self.run_script(FAKE_LEGACY='1', LABEL='legacy_noop')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        stale = self.run_script(FAKE_LEGACY='1', LABEL='legacy_noop', FAKE_NO_OUTPUT='1')
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn('no fresh predictions', stale.stderr)

    def test_legacy_eval_mode_accepts_fresh_scores_for_cached_predictions(self):
        first = self.run_script(FAKE_LEGACY='1', LABEL='legacy_eval')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        scored = self.run_script(FAKE_LEGACY='1', LABEL='legacy_eval', MODE='eval',
                                 FAKE_KEEP_PRED='1')
        self.assertEqual(scored.returncode, 0, scored.stdout + scored.stderr)

    def test_duplicate_dataset_indices_fail_verification(self):
        (self.root / 'data/MMVP.tsv').write_text(
            'index\timage_path\n0\tzero.png\n1\tone.png\n1\ttwo.png\n')
        result = self.run_script(LABEL='duplicate_source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate or empty indices', result.stderr)

    def test_changed_dataset_content_cannot_reuse_cached_predictions(self):
        first = self.run_script(LABEL='dataset_change')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        (self.root / 'data/MMVP.tsv').write_text(
            'index\timage_path\n0\tchanged.png\n1\tone.png\n')
        changed = self.run_script(LABEL='dataset_change')
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn('Dataset source changed', changed.stderr)

    def test_dataset_changed_during_evaluation_does_not_seal_new_fingerprint(self):
        result = self.run_script(LABEL='midrun_change', FAKE_MUTATE_DATASET='MMVP')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('dataset TSV changed during evaluation', result.stderr)

    def test_changed_judge_endpoint_cannot_reuse_cached_scores(self):
        common = {'LABEL': 'endpoint_change', 'JUDGE': 'gpt-4o-mini', 'OPENAI_API_KEY': 'fake'}
        first = self.run_script(**common, OPENAI_BASE_URL='https://one.example/v1')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        changed = self.run_script(**common, OPENAI_BASE_URL='https://two.example/v1')
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn('protocol changed', changed.stderr)

    def test_changed_vlmeval_code_cannot_reuse_cached_scores(self):
        self.assertEqual(self.run_script(DRY_RUN='1', LABEL='kit_change').returncode, 0)
        run_file = self.vlm / 'run.py'
        run_file.write_text(run_file.read_text() + '\n# evaluator changed\n')
        result = self.run_script(DRY_RUN='1', LABEL='kit_change')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('protocol changed', result.stderr)

    def test_orphaned_artifacts_without_protocol_are_not_relabelled(self):
        result_dir = self.root / 'results/orphaned'
        result_dir.mkdir(parents=True)
        (result_dir / 'orphaned_MMVP_acc.csv').write_text('split,Overall\nnone,0.5\n')
        result = self.run_script(LABEL='orphaned')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('artifacts have no protocol', result.stderr)
        self.assertFalse(self.marker.exists())

    def test_duplicate_datasets_fail_before_evaluator(self):
        result = self.run_script(DATASETS='MMVP MMVP')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate names', result.stderr)
        self.assertFalse(self.marker.exists())

    def test_main_eval_uses_the_same_nine_dataset_protocol(self):
        result = self.run_script('main/eval.sh', args=[self.ckpt], WITH_BASE='1', DATASETS='', LABEL='trained')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(self.config('trained')['data']), list(self.config('base')['data']))
        self.assertEqual(len(self.config('base')['data']), 9)
        self.assertEqual(self.config('trained')['model']['trained']['model_path'], str(self.ckpt))

    def test_main_eval_chooses_latest_numeric_trainer_checkpoint(self):
        trainer = self.root / 'run-with-hyphens'
        for step in (9, 10):
            checkpoint = trainer / f'checkpoint-{step}'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'config.json').write_text('{}')
        result = self.run_script('main/eval.sh', args=[trainer], WITH_BASE='0', LABEL='trainer')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.config('trainer')['model']['trainer']['model_path'],
                         str(trainer / 'checkpoint-10'))

    def test_passes_reuse_when_vlmevalkit_supports_it(self):
        result = self.run_script(LABEL='reuse')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--reuse', json.loads(self.marker.read_text()))

    def test_live_judge_missing_key_fails_before_evaluator(self):
        result = self.run_script(JUDGE='gpt-4o-mini')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())

    def test_infer_mode_does_not_require_judge(self):
        result = self.run_script(MODE='infer', JUDGE='gpt-4o-mini')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(self.marker.read_text())
        self.assertNotIn('--judge', argv)

    def test_unknown_dataset_and_changed_cache_protocol_are_rejected(self):
        result = self.run_script(DATASETS='not_a_benchmark')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.run_script(DRY_RUN='1').returncode, 0)
        result = self.run_script(DRY_RUN='1', MAX_NEW_TOKENS='512')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('protocol changed', result.stderr)

    def test_shell_judge_endpoint_wins_over_dotenv(self):
        dot = self.root / 'local.env'
        dot.write_text('OPENAI_API_KEY=dummy_file\nOPENAI_BASE_URL=https://file.invalid\n')
        result = self.run_script(JUDGE='gpt-4o-mini', ENV_FILE=str(dot),
                                 OPENAI_API_KEY='dummy_shell', OPENAI_BASE_URL='https://shell.invalid')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads(self.marker.read_text())
        self.assertEqual(argv[argv.index('--judge-base-url') + 1], 'https://shell.invalid')
        self.assertNotIn('dummy_shell', result.stdout + result.stderr)

    def test_caught_evaluator_errors_cannot_report_success(self):
        result = self.run_script(FAKE_ERROR='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('simulated dataset failure', result.stderr)
        self.assertNotIn('[done]', result.stdout)

    def test_skipped_or_pending_datasets_fail(self):
        for overrides in ({'FAKE_SKIP': 'invalid_dataset'}, {'FAKE_PENDING': '1'},
                          {'FAKE_DROPPED_DATASET': 'MMStar'}):
            with self.subTest(overrides=overrides):
                result = self.run_script(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('[done]', result.stdout)

    def test_failed_empty_or_partial_predictions_fail(self):
        for key, error in (('FAKE_FAILED_PRED', 'inference failed=1'),
                           ('FAKE_EMPTY_PRED', 'empty=1'),
                           ('FAKE_MISSING_ROW', 'missing=1')):
            with self.subTest(key=key):
                result = self.run_script(**{key: '1'})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_all_mode_requires_score_artifacts(self):
        result = self.run_script(FAKE_MISSING_SCORE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('score artifact missing', result.stderr)

    def test_invalid_overall_score_fails(self):
        result = self.run_script(FAKE_BAD_SCORE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid or conflicting', result.stderr)

    def test_old_success_status_cannot_mask_this_invocation(self):
        self.assertEqual(self.run_script().returncode, 0)
        result = self.run_script(FAKE_STALE_STATUS='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no status.json from this invocation', result.stderr)

    def test_legacy_kit_uses_artifact_checks(self):
        result = self.run_script(FAKE_LEGACY='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('[verified]', result.stdout)
        result = self.run_script(FAKE_LEGACY='1', FAKE_MISSING_ROW='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('incomplete predictions', result.stderr)

    def test_no_status_and_no_artifacts_cannot_report_success(self):
        result = self.run_script(FAKE_NO_OUTPUT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('prediction file missing', result.stderr)

    def test_current_kit_requires_a_status_file(self):
        report = self.vlm / 'vlmeval/smp/status_report.py'
        report.parent.mkdir(parents=True)
        report.write_text('')
        result = self.run_script(FAKE_LEGACY='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no status.json from this invocation', result.stderr)

    def test_missing_image_rows_match_official_dataset_filter(self):
        for dataset in ('MMVP', 'MMStar'):
            (self.root / 'data' / f'{dataset}.tsv').write_text(
                'index\timage\n0\tbase64-zero\n1\tbase64-one\n'
                '2\tNaN\n3\t\n4\tNA\n5\tnull\n6\tNone\n')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
