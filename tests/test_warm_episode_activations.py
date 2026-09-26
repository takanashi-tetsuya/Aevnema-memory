from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
)
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.types import EpisodeDraft

from tests.helpers import FakeModel, test_config


class WarmEpisodeActivationTests(unittest.TestCase):
    def test_rejects_unbounded_or_invalid_warm_seeds(self):
        normalize = QueryEngine._normalize_warm_episode_activations
        for invalid in (
            {0: 0.2},
            {True: 0.2},
            {1: 0.0},
            {1: -0.1},
            {1: float("nan")},
            {1: float("inf")},
            {1: 1.1},
            {index: 0.2 for index in range(1, 34)},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize(invalid)
        self.assertEqual({1: 0.2}, normalize({1: 0.2}))
        self.assertIsNone(normalize(None))

    def test_warm_seed_reaches_initial_graph_without_becoming_independent_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory))
            config.retrieval.growth_max_rounds = 0
            config.retrieval.sparse_enabled = False
            config.retrieval.source_key_cohort_enabled = False
            config.retrieval.graph_max_hops = 0
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            source_id = sources.insert("A source-bound memory from an earlier round.")
            episode_id = episodes.insert(
                source_id,
                "memory-1",
                0,
                EpisodeDraft(text="A source-bound memory from an earlier round."),
                encode_embedding([1.0] + [0.0] * 7, 8),
            )
            fresh_source_id = sources.insert("A separate current-query memory.")
            fresh_episode_id = episodes.insert(
                fresh_source_id,
                "memory-2",
                0,
                EpisodeDraft(text="A separate current-query memory."),
                encode_embedding([1.0] + [0.0] * 7, 8),
            )
            episode_index = EmbeddingIndex(8)
            episode_index.add(fresh_episode_id, [1.0] + [0.0] * 7)
            engine = QueryEngine(
                config,
                FakeModel(8),
                episode_index,
                EmbeddingIndex(8),
                episodes,
                ConceptRepository(db),
                sources,
                AssociationRepository(db, config.weights),
            )
            question = "Which unrelated record was retained?"
            cold = engine.query(question, stop_after="evidence")
            self.assertEqual([], cold["episode_activation_trace"]["warm_input"])
            self.assertNotIn(
                episode_id,
                [
                    entry["episode_id"]
                    for entry in cold["episode_activation_trace"]["initial_graph_top_episodes"]
                ],
            )
            self.assertIn(
                fresh_episode_id,
                [
                    entry["episode_id"]
                    for entry in cold["episode_activation_trace"]["initial_graph_top_episodes"]
                ],
            )
            with patch.object(engine, "_v17_automatic_exact_revisit", side_effect=AssertionError(
                "warm query must not enter automatic exact revisit"
            )), patch.object(engine, "_t16_automatic_restricted_rewrite_revisit", side_effect=AssertionError(
                "warm query must not enter automatic restricted rewrite"
            )):
                result = engine.query(
                    question,
                    stop_after="evidence",
                    warm_episode_activations={episode_id: 0.4},
                )

            trace = result["episode_activation_trace"]
            self.assertEqual("retrieval_attention_only", trace["role"])
            self.assertEqual(
                [{"episode_id": episode_id, "score": 0.4}],
                trace["warm_input"],
            )
            self.assertIn(
                episode_id,
                [entry["episode_id"] for entry in trace["initial_graph_top_episodes"]],
            )
            self.assertIn(
                fresh_episode_id,
                [entry["episode_id"] for entry in trace["initial_graph_top_episodes"]],
            )
            self.assertIn(
                fresh_episode_id,
                result["contextual_association"]["base_episode_ids"],
            )
            self.assertNotIn(
                episode_id,
                result["contextual_association"]["base_episode_ids"],
            )


if __name__ == "__main__":
    unittest.main()
