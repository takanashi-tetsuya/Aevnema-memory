from __future__ import annotations

import unittest

from benchmarks.support.stage8 import (
    normalize_answer_pair_judgment,
    summarize_answer_pair_trials,
)


class Stage8AnswerUtilityTests(unittest.TestCase):
    def test_normalize_recomputes_score_and_unblinds(self):
        payload = {
            "scores": {
                "A": {
                    "evidence_completeness": 4,
                    "reasoning_coherence": 4,
                    "fact_boundary": 3,
                    "uncertainty_calibration": 3,
                },
                "B": {
                    "evidence_completeness": 2,
                    "reasoning_coherence": 2,
                    "fact_boundary": 4,
                    "uncertainty_calibration": 4,
                },
            },
            "winner": "B",
        }
        result = normalize_answer_pair_judgment(
            payload, label_to_arm={"A": "T2", "B": "M2"}
        )
        self.assertEqual(result["winning_arm"], "T2")
        self.assertEqual(result["score_delta_t2_minus_m2"], 2)

    def test_normalize_clamps_invalid_dimension_scores(self):
        result = normalize_answer_pair_judgment(
            {"scores": {"A": {"evidence_completeness": 99}, "B": {}}},
            label_to_arm={"A": "M2", "B": "T2"},
        )
        self.assertEqual(result["scores"]["A"]["evidence_completeness"], 4)
        self.assertEqual(result["winning_arm"], "M2")

    def test_summary_aggregates_judge_and_trial_wins(self):
        trials = [
            {
                "judgments": [
                    {"winning_arm": "T2", "score_delta_t2_minus_m2": 2},
                    {"winning_arm": "T2", "score_delta_t2_minus_m2": 1},
                ]
            },
            {
                "judgments": [
                    {"winning_arm": "M2", "score_delta_t2_minus_m2": -1},
                    {"winning_arm": "tie", "score_delta_t2_minus_m2": 0},
                ]
            },
        ]
        result = summarize_answer_pair_trials(trials)
        self.assertEqual(result["judge_wins"], {"T2": 2, "M2": 1, "tie": 1})
        self.assertEqual(result["trial_wins"], {"T2": 1, "M2": 1, "tie": 0})
        self.assertEqual(result["mean_score_delta_t2_minus_m2"], 0.5)


if __name__ == "__main__":
    unittest.main()
