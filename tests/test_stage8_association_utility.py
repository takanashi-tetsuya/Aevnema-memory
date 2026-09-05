from __future__ import annotations

import unittest

from memory_demo.association_overlay import AssociationDelta
from benchmarks.support.stage8 import (
    association_utility_diagnostics,
    hard_group_score,
    minimal_probation_promotion_decision,
    probation_promotion_decision,
    replay_bundle_variant,
)


class Stage8AssociationUtilityTests(unittest.TestCase):
    def test_replay_variant_does_not_mutate_source(self):
        source = {
            "configuration": {
                "graph_max_hops": 3,
                "answer_episode_limit": 20,
            },
            "final_seed_hits": [],
        }
        variant = replay_bundle_variant(
            source, graph_max_hops=0, answer_episode_limit=12
        )
        self.assertEqual(source["configuration"]["graph_max_hops"], 3)
        self.assertEqual(source["configuration"]["answer_episode_limit"], 20)
        self.assertEqual(variant["configuration"]["graph_max_hops"], 0)
        self.assertEqual(variant["configuration"]["answer_episode_limit"], 12)

    def test_hard_group_accepts_alternatives(self):
        score = hard_group_score(
            {"episode_ids": [9, 2], "candidate_episode_ids": [3, 8]},
            [2, 3],
        )
        self.assertTrue(score["selected"])
        self.assertEqual(score["selected_matches"], [2])
        self.assertEqual(score["candidate_matches"], [3])

    def test_causal_utility_requires_used_delta_and_gain(self):
        row = {"id": 90}
        delta = AssociationDelta(
            created=[{"id": 90, "before": None, "after": row}],
            reinforced=[],
        )
        criterion = {
            "required_episode_groups": [[1], [2]],
            "required_sources": [],
        }
        treatment = {
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
            "association_paths": [
                {
                    "association_id": 90,
                    "from": ["episode", 1],
                    "to": ["episode", 2],
                }
            ],
        }
        masked = {
            "episode_ids": [1],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
            "association_paths": [],
        }
        result = association_utility_diagnostics(
            treatment, masked, delta, criterion, [2]
        )
        self.assertTrue(result["learned_path_used"])
        self.assertTrue(result["hard_group_gain"])
        self.assertTrue(result["causal_utility_observed"])

    def test_recall_gain_without_delta_path_is_not_association_utility(self):
        delta = AssociationDelta(
            created=[{"id": 90, "before": None, "after": {"id": 90}}],
            reinforced=[],
        )
        criterion = {
            "required_episode_groups": [[1], [2]],
            "required_sources": [],
        }
        treatment = {
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
            "association_paths": [],
        }
        masked = {
            "episode_ids": [1],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
            "association_paths": [],
        }
        result = association_utility_diagnostics(
            treatment, masked, delta, criterion, [2]
        )
        self.assertGreater(result["selected_recall_delta_t_minus_m"], 0)
        self.assertFalse(result["causal_utility_observed"])

    def test_probation_promotes_only_q2_used_rows_after_hard_group_gain(self):
        delta = AssociationDelta(
            created=[
                {"id": 90, "before": None, "after": {"id": 90}},
                {"id": 91, "before": None, "after": {"id": 91}},
            ],
            reinforced=[],
        )
        decision = probation_promotion_decision(
            {
                "learned_path_ids": [90],
                "causal_utility_observed": True,
                "hard_group_gain": True,
            },
            delta,
        )
        self.assertEqual(decision["promote_association_ids"], [90])
        self.assertEqual(decision["reject_association_ids"], [91])
        self.assertTrue(decision["promotion_earned"])

    def test_probation_rejects_used_row_without_hard_group_gain(self):
        delta = AssociationDelta(
            created=[{"id": 90, "before": None, "after": {"id": 90}}],
            reinforced=[],
        )
        decision = probation_promotion_decision(
            {
                "learned_path_ids": [90],
                "causal_utility_observed": True,
                "hard_group_gain": False,
            },
            delta,
        )
        self.assertFalse(decision["promotion_earned"])
        self.assertEqual(decision["reject_association_ids"], [90])

    def test_probation_accepts_cue_usage_as_cross_query_use(self):
        delta = AssociationDelta(
            created=[{"id": 90, "before": None, "after": {"id": 90}}],
            reinforced=[],
        )
        decision = probation_promotion_decision(
            {
                "learned_cue_ids": [90],
                "causal_utility_observed": True,
                "hard_group_gain": True,
            },
            delta,
        )
        self.assertEqual(decision["promote_association_ids"], [90])

    def test_minimal_promotion_keeps_best_redundant_sufficient_row(self):
        delta = AssociationDelta(
            created=[
                {
                    "id": 90,
                    "before": None,
                    "after": {
                        "id": 90,
                        "generation": 1,
                        "confidence": 0.7,
                        "weight": 0.9,
                    },
                },
                {
                    "id": 91,
                    "before": None,
                    "after": {
                        "id": 91,
                        "generation": 1,
                        "confidence": 0.9,
                        "weight": 0.8,
                    },
                },
            ],
            reinforced=[],
        )
        decision = minimal_probation_promotion_decision(
            delta,
            individually_sufficient_ids=[90, 91],
            individually_necessary_ids=[],
        )
        self.assertEqual(decision["promote_association_ids"], [91])
        self.assertEqual(decision["reject_association_ids"], [90])

    def test_minimal_promotion_keeps_all_necessary_rows(self):
        delta = AssociationDelta(
            created=[
                {"id": 90, "before": None, "after": {"id": 90}},
                {"id": 91, "before": None, "after": {"id": 91}},
            ],
            reinforced=[],
        )
        decision = minimal_probation_promotion_decision(
            delta,
            individually_sufficient_ids=[],
            individually_necessary_ids=[90, 91],
        )
        self.assertEqual(decision["promote_association_ids"], [90, 91])


if __name__ == "__main__":
    unittest.main()
