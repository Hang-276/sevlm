"""Proposer feedback distinguishes spy detection from visual certification."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.self_evolve.proposer import (
    build_proposer_sft_examples,
    competence_line,
    learnability,
    score_proposals,
)


def proposal(task_id):
    return {
        "proposal_id": f"proposal-{task_id}",
        "task_id": task_id,
        "scene_id": "scene-a",
        "prompt": "Choose a subset.",
        "completion": "<keep>[0]</keep><players>3</players>",
        "image_path": ["a.png", "b.png"],
        "keep": [0],
    }


class EvidenceProposerTests(unittest.TestCase):
    def test_evidence_frontier_uses_verified_rate_but_keeps_spy_rate(self):
        scored = score_proposals([proposal("evidence")], {
            "evidence": {
                "mastery_mode": "verified_visual", "frontier": "evidence",
                "gold_certificate_available": True,
                "pass_rate": 1.0, "verified_pass_rate": 0.5,
                "class": "trainable", "regret": 0.2,
            },
        })[0]
        self.assertEqual(scored["pass_rate"], 1.0)
        self.assertEqual(scored["verified_pass_rate"], 0.5)
        self.assertEqual(scored["learnability"], 1.0)
        self.assertEqual(scored["learnability_basis"], "verified_visual")
        replay = build_proposer_sft_examples([scored])
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay[0]["learnability_basis"], "verified_visual")
        self.assertEqual(replay[0]["verified_pass_rate"], 0.5)

    def test_spy_frontier_and_legacy_use_spy_rate(self):
        scored = score_proposals([proposal("spy"), proposal("legacy")], {
            "spy": {
                "mastery_mode": "verified_visual", "frontier": "spy",
                "gold_certificate_available": True,
                "pass_rate": 0.5, "verified_pass_rate": 0.0,
                "class": "trainable", "regret": 0.2,
            },
            "legacy": {"pass_rate": 0.5, "class": "trainable", "regret": 0.2},
        })
        self.assertEqual([row["learnability"] for row in scored], [1.0, 1.0])
        self.assertEqual(scored[0]["learnability_basis"], "spy")
        self.assertNotIn("learnability_basis", scored[1])
        self.assertNotIn("verified_pass_rate", scored[1])
        replay = build_proposer_sft_examples(scored)
        self.assertEqual(replay[0]["learnability_basis"], "spy")
        self.assertNotIn("learnability_basis", replay[1])

    def test_uncertified_proposals_never_enter_replay(self):
        for certificate in (None, False):
            stats = {"mastery_mode": "verified_visual", "frontier": "unverifiable",
                     "gold_certificate_available": certificate, "pass_rate": 0.5,
                     "verified_pass_rate": 0.0, "class": "trainable", "regret": 0.2}
            scored = score_proposals([proposal("bad")], {"bad": stats})
            self.assertEqual(scored[0]["learnability"], 0.0)
            self.assertEqual(scored[0]["learnability_basis"], "unverifiable")
            self.assertEqual(build_proposer_sft_examples(scored, threshold=0.0), [])
        missing = score_proposals([proposal("missing")], {})
        self.assertEqual(build_proposer_sft_examples(missing, threshold=0.0), [])

    def test_nonfinite_pass_rates_have_no_learnability(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(learnability(value), 0.0)

    def test_competence_line_shows_both_rates_only_with_certificate_feedback(self):
        legacy = competence_line({"mean_solve_rate": 0.75})
        self.assertEqual(legacy, "Right now the player solves 75% of the puzzles it is given.\n")
        evidence = competence_line({"mean_solve_rate": 0.75,
                                    "mean_verified_pass_rate": 0.25})
        self.assertIn("identifies the spy in 75%", evidence)
        self.assertIn("visual certificate", evidence)
        self.assertIn("in 25%", evidence)
        self.assertIn("keeping the puzzle's difficulty steady", evidence)
        self.assertEqual(competence_line({"mean_verified_pass_rate": 0.25}), "")


if __name__ == "__main__":
    unittest.main()
