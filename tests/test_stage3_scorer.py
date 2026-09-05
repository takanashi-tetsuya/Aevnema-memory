import unittest

from benchmarks.support.evidence_scorer import score_result


class Stage3ScorerTests(unittest.TestCase):
    def _criteria(self):
        return {
            "q": {
                "required_episode_groups": [[1]],
                "required_sources": ["main/test.json"],
                "required_terms": ["答案"],
            }
        }

    def _result(self):
        return {
            "episode_ids": [1],
            "evidence_episodes": [
                {"id": 1, "source_key": "main/test.json"}
            ],
            "answer": "答案",
            "answer_audits": [{"valid": True}],
            "association_ids": [9],
            "new_association_ids": [],
            "reinforced_association_ids": [],
        }

    def test_growing_mode_accepts_counterfactually_proven_safe_zero_write(self):
        result = {
            **self._result(),
            "growth_counterfactual_utility": {
                "enabled": True,
                "causal_utility_observed": False,
                "persistable_changed_ids": [],
            },
            "growth_staging": {
                "enabled": True,
                "committed": False,
            },
        }

        score = score_result(
            "q", result, "graph_growing", "completed", self._criteria()
        )

        self.assertTrue(score["safe_counterfactual_zero_write"])
        self.assertTrue(score["mode_behavior_passed"])
        self.assertTrue(score["passed"])

    def test_growing_mode_does_not_treat_unexplained_zero_write_as_success(self):
        score = score_result(
            "q",
            self._result(),
            "graph_growing",
            "completed",
            self._criteria(),
        )

        self.assertFalse(score["safe_counterfactual_zero_write"])
        self.assertFalse(score["mode_behavior_passed"])
        self.assertFalse(score["passed"])


if __name__ == "__main__":
    unittest.main()
