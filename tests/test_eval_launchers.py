"""CPU-only regression tests. Fake VLMEvalKit records argv; no torch/model imports.

Run: python -m unittest discover -s tests -p test_eval_launchers.py
"""
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
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["FAKE_CALLED"]).write_text(json.dumps(sys.argv[1:]))\n'
            'print("fake evaluator; no inference")\n'
        )
        self.marker = self.root / 'called.json'
        self.env = {
            'PATH': os.environ['PATH'], 'HOME': str(self.root),
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


if __name__ == '__main__':
    unittest.main()
