from __future__ import annotations

import unittest

from memory_demo.config import AppConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.association_overlay import AssociationDelta
from benchmarks.support.stage10 import (
    hard_negative_diagnostics,
    positive_actionable_opportunity,
)


class _Rows:
    def __init__(self, rows):
        self.rows = {int(row["id"]): row for row in rows}

    def get(self, row_id):
        return self.rows.get(int(row_id))


class _GateModel:
    logger = None

    def chat_json(self, _system, _user):
        return {
            "decisions": [
                {"association_id": 9, "accept": True, "reason": "两端均相关"},
                {"association_id": 10, "accept": False, "reason": "第二端题外"},
            ]
        }


class Stage10HardNegativeTests(unittest.TestCase):
    def test_actionable_opportunity_requires_a_delta_endpoint_in_missing_group(self):
        delta = AssociationDelta(
            created=[
                {
                    "id": 9,
                    "before": None,
                    "after": {
                        "id": 9,
                        "from_type": "episode",
                        "from_id": 3,
                        "to_type": "episode",
                        "to_id": 4,
                    },
                }
            ],
            reinforced=[
                {
                    "id": 8,
                    "before": {"id": 8},
                    "after": {
                        "id": 8,
                        "from_type": "concept",
                        "from_id": 99,
                        "to_type": "episode",
                        "to_id": 6,
                    },
                }
            ],
        )
        result = positive_actionable_opportunity(
            {"episode_ids": [1]},
            delta,
            {
                "required_episode_groups": [[1, 2], [3, 5], [6], [7]],
            },
        )
        self.assertTrue(result["actionable"])
        self.assertEqual(result["missing_group_indexes"], [1, 2, 3])
        self.assertEqual(result["actionable_group_indexes"], [1, 2])
        self.assertEqual(result["actionable_episode_ids"], [3, 6])
        self.assertEqual(result["delta_episode_endpoint_ids"], [3, 4, 6])

    def test_missing_group_without_delta_endpoint_is_not_actionable(self):
        delta = AssociationDelta(
            created=[
                {
                    "id": 9,
                    "before": None,
                    "after": {
                        "id": 9,
                        "from_type": "episode",
                        "from_id": 3,
                        "to_type": "episode",
                        "to_id": 4,
                    },
                }
            ],
            reinforced=[],
        )
        result = positive_actionable_opportunity(
            {"episode_ids": [1]},
            delta,
            {"required_episode_groups": [[1], [7, 8]]},
        )
        self.assertFalse(result["actionable"])
        self.assertEqual(result["missing_group_indexes"], [1])
        self.assertEqual(result["actionable_group_indexes"], [])

    def test_semantic_gate_keeps_only_explicitly_accepted_relation(self):
        config = AppConfig()
        config.model.embedding_dimension = 3
        config.retrieval.association_cue_enabled = True
        config.retrieval.association_cue_semantic_gate_enabled = True
        associations = _Rows(
            [
                {
                    "id": association_id,
                    "from_type": "episode",
                    "from_id": 1,
                    "to_type": "episode",
                    "to_id": endpoint,
                    "relation_key": "theme_bridge",
                    "relation_text": f"关系 {association_id}",
                    "audit_status": "dual_accepted",
                    "created_reason": "查询中自主增长：测试",
                    "generation": 1,
                    "claim_level": "supported_inference",
                }
                for association_id, endpoint in ((9, 2), (10, 3))
            ]
        )
        episodes = _Rows(
            [
                {"id": node_id, "source_key": "source", "text": f"节点 {node_id}"}
                for node_id in (1, 2, 3)
            ]
        )
        engine = QueryEngine(
            config,
            _GateModel(),
            EmbeddingIndex(3),
            EmbeddingIndex(3),
            episodes,
            _Rows([]),
            object(),
            associations,
            association_index=EmbeddingIndex(3),
        )
        accepted, decisions = engine._semantic_gate_association_cues(
            "只需要关系九",
            [
                {"association_id": 9, "cosine": 0.8},
                {"association_id": 10, "cosine": 0.7},
            ],
        )
        self.assertEqual([row["association_id"] for row in accepted], [9])
        self.assertEqual(len(decisions), 2)

    def test_candidate_cue_is_allowed_when_it_does_not_intrude(self):
        delta = AssociationDelta(
            created=[{"id": 9, "before": None, "after": {"id": 9}}],
            reinforced=[],
        )
        graph = {
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
        }
        treatment = {
            **graph,
            "association_cue_ids": [9],
            "association_paths": [],
        }
        result = hard_negative_diagnostics(
            treatment,
            graph,
            delta,
            {
                "required_episode_groups": [[1], [2]],
                "required_sources": [],
                "forbidden_episode_ids": [7, 8],
            },
        )
        self.assertTrue(result["target_cue_candidate_seen"])
        self.assertTrue(result["answer_contained"])

    def test_new_forbidden_endpoint_is_an_intrusion(self):
        delta = AssociationDelta(
            created=[{"id": 9, "before": None, "after": {"id": 9}}],
            reinforced=[],
        )
        graph = {
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2, 7],
            "evidence_episodes": [],
        }
        treatment = {
            "episode_ids": [1, 7],
            "candidate_episode_ids": [1, 2, 7],
            "evidence_episodes": [],
            "association_cue_ids": [9],
            "association_paths": [
                {"association_id": 9, "learned_bridge_slot_used": True}
            ],
        }
        result = hard_negative_diagnostics(
            treatment,
            graph,
            delta,
            {
                "required_episode_groups": [[1], [2]],
                "required_sources": [],
                "forbidden_episode_ids": [7],
            },
        )
        self.assertEqual(result["forbidden_intrusion_episode_ids"], [7])
        self.assertFalse(result["required_recall_non_regression"])
        self.assertFalse(result["answer_contained"])


if __name__ == "__main__":
    unittest.main()
