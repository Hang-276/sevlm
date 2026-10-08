"""CPU-only checks for nine-benchmark score collection and comparability."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "local_scripts/self_evolve/experiments/analysis/collect_results.py"
SPEC = importlib.util.spec_from_file_location("collect_results", COLLECTOR)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def score_file(self, label, benchmark, text, eval_id="run_1"):
        directory = self.root / label / eval_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{label}_{benchmark}_acc.csv"
        path.write_text(text)
        return path

    def protocol(self, label, *, judge="exact_matching", tokens=2048):
        path = self.root / label / "eval_protocol.json"
        data = {"MMVP": {"class": "ImageMCQDataset", "dataset": "MMVP"}}
        path.write_text(json.dumps({
            "config": {
                "model": {label: {"class": "Qwen2VLChat", "model_path": f"/models/{label}",
                                  "max_new_tokens": tokens}},
                "data": data,
            },
            "judge": judge, "use_vllm": "1", "vlmeval_commit": "abc123",
            "vlmeval_path": "/eval/VLMEvalKit",
        }))

    def test_mmvp_uses_pair_accuracy_not_single_question_average(self):
        path = self.score_file("base", "MMVP", "index,Average,Overall\n0,0.75,0.5\n")
        self.assertEqual(collector.read_score(path, "MMVP")["score"], 50)

    def test_chartqa_is_already_percent_and_one_is_valid_mcq_fraction(self):
        chart = self.score_file("base", "ChartQA_TEST", "index,Overall\n0,0.5\n")
        mcq = self.score_file("base", "MMStar", "split,Overall\nnone,1\n")
        self.assertEqual(collector.read_score(chart, "ChartQA_TEST")["score"], 0.5)
        self.assertEqual(collector.read_score(mcq, "MMStar")["score"], 100)

    def test_mcq_fraction_above_one_or_duplicate_overall_columns_are_invalid(self):
        invalid = self.score_file("base", "MMStar", "split,Overall\nnone,1.5\n")
        duplicate = self.score_file("base", "MMVP", "split,Overall,overall\nnone,0.5,0.8\n")
        self.assertIsNone(collector.read_score(invalid, "MMStar")["score"])
        self.assertIsNone(collector.read_score(duplicate, "MMVP")["score"])

    def test_changed_endpoint_code_or_dataset_fingerprint_withholds_comparison(self):
        for field, value in (("judge_base_url", "https://other.example/v1"),
                             ("vlmeval_code_sha256", "different-code"),
                             ("dataset_fingerprints", {"MMVP": "different-data"})):
            with self.subTest(field=field):
                for label in ("base", "ours"):
                    self.score_file(label, "MMVP", "split,Overall\nnone,0.5\n")
                    self.protocol(label)
                path = self.root / "ours/eval_protocol.json"
                protocol = json.loads(path.read_text())
                protocol[field] = value
                path.write_text(json.dumps(protocol))
                table = collector.scan(self.root, ["base", "ours"], ["MMVP"])
                self.assertIn("AVG withheld: eval_protocol.json",
                              collector.render(table, "md", ["MMVP"]))

    def test_malformed_protocol_cannot_crash_collection_or_enable_average(self):
        self.score_file("base", "MMVP", "split,Overall\nnone,0.5\n")
        (self.root / "base/eval_protocol.json").write_text(json.dumps({
            "config": {"model": ["invalid"]}}))
        table = collector.scan(self.root, ["base"], ["MMVP"])
        self.assertIn("AVG withheld", collector.render(table, "md", ["MMVP"]))

    def test_multiple_splits_or_unlabelled_numeric_categories_are_not_guessed(self):
        split = self.score_file("base", "MMStar", "split,Overall\na,0.4\nb,0.5\n")
        category = self.score_file("base", "MMVP", "category,Count,Depth\ncount,40,60\n")
        self.assertIsNone(collector.read_score(split, "MMStar")["score"])
        self.assertIsNone(collector.read_score(category, "MMVP")["score"])

    def test_nested_results_and_matching_protocol_produce_comparable_average(self):
        for label, score in (("base", 0.4), ("ours", 0.6)):
            self.score_file(label, "MMVP", f"index,Average,Overall\n0,0.9,{score}\n")
            self.protocol(label)
        table = collector.scan(self.root, ["base", "ours"], ["MMVP"])
        rendered = collector.render(table, "md", ["MMVP"])
        self.assertIn("base | 40.00 | 40.00", rendered)
        self.assertIn("ours | 60.00 | 60.00", rendered)

    def test_conflicting_repeated_runs_withhold_score(self):
        self.score_file("base", "MMVP", "index,Overall\n0,0.4\n", "run_1")
        self.score_file("base", "MMVP", "index,Overall\n0,0.7\n", "run_2")
        row = collector.scan(self.root, ["base"], ["MMVP"])["base"]
        self.assertIsNone(row["MMVP"]["score"])
        self.assertIn("conflicting", row["MMVP"]["why"])

    def test_mismatched_judges_withhold_average_even_with_complete_scores(self):
        for label, judge in (("base", "exact_matching"), ("ours", "gpt-4o-mini")):
            self.score_file(label, "MMVP", "index,Overall\n0,0.5\n")
            self.protocol(label, judge=judge)
        table = collector.scan(self.root, ["base", "ours"], ["MMVP"])
        rendered = collector.render(table, "md", ["MMVP"])
        self.assertIn("AVG withheld: eval_protocol.json", rendered)
        self.assertIn("base | 50.00 | -", rendered)

    def test_single_model_without_protocol_withholds_average(self):
        self.score_file("base", "MMVP", "index,Overall\n0,0.5\n")
        table = collector.scan(self.root, ["base"], ["MMVP"])
        rendered = collector.render(table, "md", ["MMVP"])
        self.assertIn("base | 50.00 | -", rendered)
        self.assertIn("AVG withheld: eval_protocol.json", rendered)


if __name__ == "__main__":
    unittest.main()
