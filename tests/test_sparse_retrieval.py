from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.helpers import test_config
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.associations.traversal import TraversedNode
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine, SQLiteSparseIndex
from memory_demo.types import EpisodeDraft


class SparseRetrievalTests(unittest.TestCase):
    def test_source_expansions_round_robin_across_ranked_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), dimension=4)
            config.retrieval.sparse_source_episode_expansion_limit = 2
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            first_source = sources.insert("第一来源")
            second_source = sources.insert("第二来源")
            first_ids = [
                episodes.insert(
                    first_source,
                    "main/first.json",
                    0,
                    EpisodeDraft(f"第一来源 Episode {index}"),
                    encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
                )
                for index in range(2)
            ]
            second_ids = [
                episodes.insert(
                    second_source,
                    "main/second.json",
                    0,
                    EpisodeDraft(f"第二来源 Episode {index}"),
                    encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
                )
                for index in range(2)
            ]
            for episode_id in [*first_ids, *second_ids]:
                episode_index.upsert(episode_id, [1.0, 0.0, 0.0, 0.0])
            engine = QueryEngine(
                config,
                model=None,
                episode_index=episode_index,
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
            )

            expansions = engine._source_episode_expansions(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                [(first_source, 1.0), (second_source, 0.9)],
            )

            self.assertEqual(
                [(item["source_rank"], item["episode_local_rank"]) for item in expansions],
                [(1, 1), (2, 1), (1, 2), (2, 2)],
            )

    def test_repeated_source_key_anchors_add_bounded_file_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), dimension=4)
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            source_a = sources.insert("文件前段")
            source_b = sources.insert("文件后段")
            ids = [
                episodes.insert(
                    source_id,
                    "main/shared.json",
                    index,
                    EpisodeDraft(text),
                    encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
                )
                for index, (source_id, text) in enumerate(
                    [
                        (source_a, "前段锚点一"),
                        (source_a, "前段锚点二"),
                        (source_b, "后段关键事实"),
                    ]
                )
            ]
            engine = QueryEngine(
                config,
                model=None,
                episode_index=episode_index,
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
            )

            hits, trace = engine._source_key_cohort_hits(
                ids[:2],
                [
                    TraversedNode("episode", ids[0], 0.9),
                    TraversedNode("episode", ids[1], 0.8),
                ],
            )

            self.assertEqual([hit.node_id for hit in hits], ids)
            self.assertEqual(trace["supported_source_keys"], ["main/shared.json"])
            self.assertEqual(trace["added_episode_ids"], [ids[2]])
            self.assertEqual(trace["boosted_episode_ids"], [ids[1]])
    def test_multilingual_fts_tracks_episode_and_source_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), dimension=4)
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            source_id = sources.insert(
                "古圣堂一开始就被埋了炸药。Arius forces entered underground."
            )
            episode_id = episodes.insert(
                source_id,
                "main/33070.json",
                0,
                EpisodeDraft("巡航导弹不可能造成这么大的爆炸。"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            episode_sparse = SQLiteSparseIndex(db, "episode")
            source_sparse = SQLiteSparseIndex(db, "source")

            self.assertEqual(episode_sparse.search("巡航导弹", 5)[0][0], episode_id)
            self.assertEqual(source_sparse.search("古圣堂被埋了炸药", 5)[0][0], source_id)
            self.assertEqual(source_sparse.search("Arius underground", 5)[0][0], source_id)

            paradox_source = sources.insert("乐园存在的证明悖论无法由外部观测。")
            paradox_episode = episodes.insert(
                paradox_source,
                "main/paradox.json",
                0,
                EpisodeDraft("乐园存在的证明悖论无法由外部观测。"),
                encode_embedding([0.0, 1.0, 0.0, 0.0], 4),
            )
            self.assertEqual(
                episode_sparse.search("乐园悖论", 5)[0][0], paradox_episode
            )

            episodes.update_draft(
                episode_id,
                EpisodeDraft("袭击者后来被确认属于阿里乌斯。"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            self.assertEqual(episode_sparse.search("阿里乌斯", 5)[0][0], episode_id)
            self.assertEqual(episode_sparse.search("巡航导弹", 5), [])

    def test_source_hit_is_mapped_to_episode_and_used_as_atomic_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), dimension=4)
            config.retrieval.episode_top_k = 1
            config.retrieval.concept_top_k = 0
            config.retrieval.sparse_episode_top_k = 4
            config.retrieval.sparse_source_top_k = 4
            config.retrieval.sparse_source_episode_expansion_limit = 2
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)

            source_one = sources.insert("关键原话：古圣堂地下墓穴仍然是废墟。")
            source_two = sources.insert("无关原文")
            episode_one = episodes.insert(
                source_one,
                "main/a.json",
                0,
                EpisodeDraft("众人讨论会场尚未完全修复。"),
                encode_embedding([0.7, 0.7, 0.0, 0.0], 4),
            )
            episode_two = episodes.insert(
                source_two,
                "main/b.json",
                0,
                EpisodeDraft("另一个直接向量命中。"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            episode_index.upsert(episode_one, [0.7, 0.7, 0.0, 0.0])
            episode_index.upsert(episode_two, [1.0, 0.0, 0.0, 0.0])
            engine = QueryEngine(
                config,
                model=None,
                episode_index=episode_index,
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
                episode_sparse_index=SQLiteSparseIndex(db, "episode"),
                source_sparse_index=SQLiteSparseIndex(db, "source"),
            )

            anchors: list[int] = []
            hits, rankings = engine._vector_seed_hits_from_matrix(
                ["古圣堂地下墓穴"],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                anchors,
                2,
            )
            self.assertEqual(rankings["episode"][0][0]["id"], episode_two)
            self.assertEqual(rankings["sparse_source"][0][0]["id"], source_one)
            self.assertIn(episode_one, {hit.node_id for hit in hits})
            self.assertIn(episode_one, anchors)

            config.retrieval.sparse_enabled = False
            disabled_hits, disabled_rankings = engine._vector_seed_hits_from_matrix(
                ["古圣堂地下墓穴"],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self.assertEqual(disabled_rankings["sparse_source"], [[]])
            self.assertEqual({hit.node_id for hit in disabled_hits}, {episode_two})


if __name__ == "__main__":
    unittest.main()
