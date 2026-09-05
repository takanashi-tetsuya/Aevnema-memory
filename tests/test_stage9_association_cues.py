from __future__ import annotations

import unittest

import numpy as np

from memory_demo.association_overlay import AssociationDelta
from memory_demo.associations.traversal import TraversedNode
from memory_demo.config import AppConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.types import QueryIntent
from benchmarks.support.stage9 import (
    association_cue_replay_variant,
    association_cue_utility_diagnostics,
)


class FakeAssociations:
    def __init__(self, rows):
        self.rows = {int(row["id"]): dict(row) for row in rows}

    def get(self, association_id):
        return self.rows.get(int(association_id))

    def neighbors(self, *_args, **_kwargs):
        return []


class Stage9AssociationCueTests(unittest.TestCase):
    def _engine(self):
        config = AppConfig()
        config.model.embedding_dimension = 3
        config.retrieval.association_cue_enabled = True
        config.retrieval.association_cue_top_k = 4
        config.retrieval.association_cue_min_similarity = 0.0
        index = EmbeddingIndex(3)
        index.upsert(9, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        rows = [
            {
                "id": 9,
                "from_type": "episode",
                "from_id": 11,
                "to_type": "episode",
                "to_id": 22,
                "relation_key": "evidence_bridge",
                "relation_text": "经审计的查询综合推论",
                "audit_status": "dual_accepted",
                "confidence": 0.91,
                "evidence_count": 1,
                "generation": 1,
                "created_reason": "查询中自主增长：前置问题",
            }
        ]
        engine = QueryEngine(
            config,
            object(),
            EmbeddingIndex(3),
            EmbeddingIndex(3),
            object(),
            object(),
            object(),
            FakeAssociations(rows),
            association_index=index,
        )
        return engine

    @staticmethod
    def _direct_episode(node_id):
        return {
            "id": node_id,
            "text": f"原始证据 {node_id}",
            "participants": [],
            "source_key": f"source-{node_id}",
            "score": 0.1,
            "generation": 0,
            "evidence_origin": "source",
        }

    def test_relation_embedding_recalls_both_endpoints(self):
        engine = self._engine()
        entries = engine._association_cue_entries_from_matrix(
            ["抽象改写问题"],
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
        hits, cue_ids, active = engine._active_association_cue_hits(entries)
        self.assertEqual(cue_ids, [9])
        self.assertEqual(
            {(hit.node_type, hit.node_id) for hit in hits},
            {("episode", 11), ("episode", 22)},
        )
        self.assertEqual(active[0]["association_id"], 9)

    def test_exact_query_vector_override_skips_embedding_request(self):
        class NoEmbedModel:
            def embed(self, _texts):
                raise AssertionError("embedding endpoint must not be called")

        engine = self._engine()
        engine.model = NoEmbedModel()
        hits, cue_ids, _entries, _rankings = engine._vector_seed_hits_with_cues(
            ["缓存问题"],
            query_embeddings_override={
                "缓存问题": np.array([1.0, 0.0, 0.0], dtype=np.float32)
            },
        )

        self.assertEqual([9], cue_ids)
        self.assertEqual(
            {("episode", 11), ("episode", 22)},
            {(item.node_type, item.node_id) for item in hits},
        )
        self.assertEqual(["缓存问题"], engine.last_query_embedding_cache_trace["hits"])
        self.assertEqual([], engine.last_query_embedding_cache_trace["misses"])

    def test_high_confidence_cue_reuses_direct_endpoints_without_reranker(self):
        engine = self._engine()
        engine.config.retrieval.rerank_enabled = True
        engine.config.retrieval.association_cue_fast_path_enabled = True
        engine.config.retrieval.association_cue_fast_path_min_similarity = 0.8

        selected, trace = engine._rerank_answer_episodes(
            "抽象改写问题",
            QueryIntent.from_dict({"search_queries": ["抽象改写问题"]}),
            ["抽象改写问题"],
            [self._direct_episode(11), self._direct_episode(22)],
            2,
            association_cue_entries=[
                {"association_id": 9, "cosine": 0.91}
            ],
        )

        self.assertEqual([11, 22], selected)
        self.assertEqual("association_capsule", trace["backend"])
        self.assertTrue(trace["cache_hit"])
        self.assertEqual(1, trace["cloud_requests_avoided"])

    def test_stronger_fact_edge_still_requires_independent_reinforcement(self):
        engine = self._engine()
        engine.config.retrieval.association_cue_fast_path_enabled = True
        engine.associations.rows[9]["relation_key"] = "causal_context"

        protected = engine._association_cue_fast_endpoint_keys(
            [
                {
                    "association_id": 9,
                    "cosine": 0.91,
                    "endpoints": [["episode", 11], ["episode", 22]],
                }
            ]
        )

        self.assertEqual(set(), protected)

    def test_cue_fast_path_rejects_inferred_episode_endpoint(self):
        engine = self._engine()
        engine.config.retrieval.association_cue_fast_path_enabled = True
        episodes = [self._direct_episode(11), self._direct_episode(22)]
        episodes[1]["generation"] = 1

        result = engine._association_cue_fast_rerank(
            episodes=episodes,
            limit=2,
            required_ids=[],
            preferred_ids=[],
            association_cue_entries=[
                {"association_id": 9, "cosine": 0.91}
            ],
        )

        self.assertIsNone(result)

    def test_fast_cue_endpoints_survive_candidate_pool_truncation(self):
        nodes = [
            TraversedNode("episode", node_id, 1.0 / node_id)
            for node_id in range(1, 8)
        ]

        selected = QueryEngine._truncate_traversed_nodes(
            nodes,
            4,
            {("episode", 6), ("episode", 7)},
        )

        self.assertEqual([1, 2, 6, 7], [item.node_id for item in selected])

    def test_fast_cue_abstains_when_top_relations_are_ambiguous(self):
        engine = self._engine()
        engine.config.retrieval.association_cue_fast_path_enabled = True
        engine.config.retrieval.association_cue_fast_path_min_similarity = 0.6
        engine.config.retrieval.association_cue_fast_path_min_margin = 0.08

        protected = engine._association_cue_fast_endpoint_keys(
            [
                {
                    "association_id": 9,
                    "cosine": 0.90,
                    "endpoints": [["episode", 11], ["episode", 22]],
                },
                {
                    "association_id": 10,
                    "cosine": 0.85,
                    "endpoints": [["episode", 33], ["episode", 44]],
                },
            ]
        )

        self.assertEqual(set(), protected)

    def test_replay_variant_removes_every_cue_configuration_key(self):
        bundle = {
            "version": 4,
            "configuration": {
                "graph_max_hops": 3,
                "answer_episode_limit": 20,
                "association_cue_enabled": True,
                "association_cue_top_k": 8,
                "association_cue_min_similarity": 0.45,
                "association_cue_rrf_weight": 0.8,
            },
            "base_final_seed_hits": [
                {"node_type": "episode", "node_id": 1, "score": 1.0}
            ],
            "final_seed_hits": [
                {"node_type": "episode", "node_id": 1, "score": 1.0},
                {"node_type": "episode", "node_id": 2, "score": 0.1},
            ],
        }
        variant = association_cue_replay_variant(
            bundle,
            enabled=False,
            graph_max_hops=3,
            answer_episode_limit=20,
        )
        self.assertEqual(variant["final_seed_hits"], bundle["base_final_seed_hits"])
        self.assertNotIn("association_cue_enabled", variant["configuration"])

    def test_utility_requires_a_changed_cue_and_recall_gain(self):
        delta = AssociationDelta(
            created=[{"id": 9, "before": None, "after": {"id": 9}}],
            reinforced=[],
        )
        criterion = {
            "required_episode_groups": [[1], [2]],
            "required_sources": [],
        }
        treatment = {
            "association_cue_ids": [9],
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2],
            "evidence_episodes": [],
        }
        masked = {
            "association_cue_ids": [],
            "episode_ids": [1],
            "candidate_episode_ids": [1],
            "evidence_episodes": [],
        }
        result = association_cue_utility_diagnostics(
            treatment, masked, delta, criterion
        )
        self.assertTrue(result["causal_utility_observed"])
        self.assertEqual(result["matched_group_delta_s_minus_sm"], 1)

    def test_direct_cue_gets_a_bounded_bridge_slot_despite_lexical_overlap(self):
        episodes = [
            {
                "id": node_id,
                "text": "同一主题的高度相似证据",
                "participants": [],
                "source_key": f"source-{node_id}",
                "score": 1.0 / node_id,
            }
            for node_id in (1, 2, 3)
        ]
        path = {
            "association_id": 9,
            "from": ["episode", 1],
            "to": ["episode", 2],
            "relation_key": "theme_bridge",
            "relation_text": "抽象改写问题对应的主题桥",
            "polarity": 1,
            "path_score": 0.5,
            "audit_status": "dual_accepted",
            "created_reason": "查询中自主增长：前置问题",
        }
        selected, paths = QueryEngine._select_answer_evidence(
            episodes,
            [path],
            episode_limit=2,
            path_limit=4,
            preferred_association_ids={9},
            question="抽象改写问题",
            preferred_episode_ids=[1, 3],
            learned_bridge_slots=1,
            learned_bridge_duplicate_threshold=0.0,
            preferred_association_scores={9: 0.8},
        )
        self.assertEqual({int(item["id"]) for item in selected}, {1, 2})
        self.assertTrue(paths[0]["learned_bridge_slot_used"])
        self.assertEqual(paths[0]["association_cue_similarity"], 0.8)

    def test_semantic_cue_does_not_require_lexical_path_overlap(self):
        episodes = [
            {
                "id": node_id,
                "text": f"证据 {node_id}",
                "participants": [],
                "source_key": f"source-{node_id}",
                "score": 1.0 / node_id,
            }
            for node_id in (1, 2, 3)
        ]
        path = {
            "association_id": 9,
            "from": ["episode", 1],
            "to": ["episode", 2],
            "relation_key": "theme_bridge",
            "relation_text": "词面完全不同但向量确认相关",
            "polarity": 1,
            "path_score": 0.5,
            "audit_status": "dual_accepted",
            "created_reason": "查询中自主增长：前置问题",
        }

        selected, _paths = QueryEngine._select_answer_evidence(
            episodes,
            [path],
            episode_limit=2,
            path_limit=4,
            preferred_association_ids={9},
            question="unrelated lexical tokens",
            preferred_episode_ids=[1, 3],
            learned_bridge_slots=1,
            preferred_association_scores={9: 0.8},
        )

        self.assertEqual({1, 2}, {int(item["id"]) for item in selected})

    def test_preferred_paths_rank_semantic_cue_before_lexical_overlap(self):
        paths = [
            {
                "association_id": 9,
                "from": ["episode", 1],
                "to": ["episode", 2],
                "relation_key": "bridge",
                "relation_text": "语义匹配但词面不同",
                "path_score": 0.5,
            },
            {
                "association_id": 10,
                "from": ["episode", 2],
                "to": ["episode", 3],
                "relation_key": "bridge",
                "relation_text": "exact lexical query",
                "path_score": 0.5,
            },
        ]

        ranked = QueryEngine._rank_answer_paths(
            paths,
            "exact lexical query",
            4,
            {9, 10},
            {9: 0.80, 10: 0.48},
        )

        self.assertEqual(9, ranked[0]["association_id"])

    def test_secondary_bridge_cannot_evict_capsule_endpoints(self):
        episodes = [
            {
                "id": node_id,
                "text": f"证据 {node_id}",
                "participants": [],
                "source_key": f"source-{node_id}",
                "score": 1.0 / node_id,
            }
            for node_id in (1, 2, 3)
        ]
        secondary = {
            "association_id": 10,
            "from": ["episode", 2],
            "to": ["episode", 3],
            "relation_key": "bridge",
            "relation_text": "次要但仍相关的边",
            "polarity": 1,
            "path_score": 0.8,
            "audit_status": "dual_accepted",
            "created_reason": "查询中自主增长：次要关系",
        }

        selected, _paths = QueryEngine._select_answer_evidence(
            episodes,
            [secondary],
            episode_limit=2,
            path_limit=4,
            preferred_association_ids={10},
            question="次要但仍相关的边",
            preferred_episode_ids=[1, 2],
            learned_bridge_slots=1,
            preferred_association_scores={10: 0.48},
            protected_episode_ids={1, 2},
        )

        self.assertEqual({1, 2}, {int(item["id"]) for item in selected})


if __name__ == "__main__":
    unittest.main()
