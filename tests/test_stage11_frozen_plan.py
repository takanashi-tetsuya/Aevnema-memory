from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.helpers import FakeModel, test_config
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine
from benchmarks.support.stage11 import run_frozen_query_plan
from memory_demo.types import EpisodeDraft, ParagraphDraft


class CountingRerankModel(FakeModel):
    def __init__(self, dimension: int = 4):
        super().__init__(dimension)
        self.rerank_calls = 0
        self.rerank_prompts: list[str] = []

    def chat_json(self, system: str, user: str, **kwargs):
        if "证据覆盖侦察器" in system:
            self.rerank_calls += 1
            self.rerank_prompts.append(user)
            return {
                "selected_episode_ids": [1],
                "coverage": [],
                "missing_aspects": [],
            }
        if "覆盖缺口审计器" in system:
            self.rerank_calls += 1
            return {"coverage": [], "missing_aspects": []}
        if "槽位压缩器" in system:
            self.rerank_calls += 1
            return {
                "valid": True,
                "final_episode_ids": [1],
                "replacements": [],
                "missing_aspects": [],
            }
        return super().chat_json(system, user, **kwargs)


class DedicatedRerankModel(FakeModel):
    def __init__(self, dimension: int = 4):
        super().__init__(dimension)
        self.calls = 0

    def rerank(self, _query, documents, *, top_n=None):
        self.calls += 1
        order = sorted(
            range(len(documents)),
            key=lambda index: "第二证据" not in documents[index],
        )
        return [
            {"index": index, "relevance_score": 1.0 - rank * 0.1}
            for rank, index in enumerate(order[:top_n])
        ]


class Stage11FrozenPlanTests(unittest.TestCase):
    def test_cross_encoder_backend_ranks_candidates_without_llm_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.retrieval.rerank_backend = "cross_encoder"
            config.retrieval.sparse_enabled = False
            config.retrieval.graph_max_hops = 0
            config.retrieval.episode_top_k = 2
            database = Database(config.database_path)
            database.initialize()
            sources = SourceRepository(database)
            episodes = EpisodeRepository(database)
            concepts = ConceptRepository(database)
            associations = AssociationRepository(database, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            source_id = sources.insert("两条候选")
            first_id = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("第一证据"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            second_id = episodes.insert(
                source_id,
                "main/b.json",
                0,
                EpisodeDraft("第二证据"),
                encode_embedding([0.9, 0.1, 0.0, 0.0], 4),
            )
            episode_index.upsert(first_id, [1.0, 0.0, 0.0, 0.0])
            episode_index.upsert(second_id, [0.9, 0.1, 0.0, 0.0])
            model = DedicatedRerankModel()
            engine = QueryEngine(
                config,
                model,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
            )
            plan = {
                "version": 1,
                "question": "哪条是第二证据？",
                "intent": {},
                "initial_queries": ["证据"],
                "followup_queries": [],
                "initial_query_embeddings_float32": [
                    [1.0, 0.0, 0.0, 0.0]
                ],
                "followup_query_embeddings_float32": [],
            }

            result = run_frozen_query_plan(
                engine, plan, episode_limit=1, rerank=True
            )

            self.assertEqual(1, model.calls)
            self.assertEqual("cross_encoder", result["rerank_trace"]["backend"])
            self.assertEqual([second_id], result["reranked_episode_ids"])
            self.assertFalse(
                result["rerank_trace"]["coverage_audit_performed"]
            )

    def test_recomputes_paragraph_channel_from_shared_query_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.paragraph.enabled = True
            config.retrieval.sparse_enabled = False
            config.retrieval.episode_top_k = 1
            config.retrieval.concept_top_k = 0
            config.retrieval.paragraph_top_k = 1
            config.retrieval.paragraph_episode_expansion_limit = 1
            config.retrieval.graph_max_hops = 0
            database = Database(config.database_path)
            database.initialize()
            sources = SourceRepository(database)
            episodes = EpisodeRepository(database)
            concepts = ConceptRepository(database)
            paragraphs = ParagraphRepository(database)
            associations = AssociationRepository(database, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            paragraph_index = EmbeddingIndex(4)

            source_one = sources.insert("原文含有精确线索")
            source_two = sources.insert("摘要直接相似")
            episode_one_vector = np.array([0.2, 0.98, 0.0, 0.0], dtype=np.float32)
            episode_two_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            episode_one = episodes.insert(
                source_one,
                "main/a.json",
                0,
                EpisodeDraft("摘要省略线索"),
                encode_embedding(episode_one_vector, 4),
            )
            episode_two = episodes.insert(
                source_two,
                "main/b.json",
                0,
                EpisodeDraft("摘要直接命中"),
                encode_embedding(episode_two_vector, 4),
            )
            episode_index.upsert(episode_one, episode_one_vector)
            episode_index.upsert(episode_two, episode_two_vector)
            paragraph_id = paragraphs.insert_many(
                source_one,
                "main/a.json",
                0,
                [ParagraphDraft(0, "原文精确线索")],
                [encode_embedding([1.0, 0.0, 0.0, 0.0], 4)],
            )[0]
            paragraph_index.upsert(paragraph_id, [1.0, 0.0, 0.0, 0.0])
            engine = QueryEngine(
                config,
                None,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
                paragraph_index=paragraph_index,
                paragraphs=paragraphs,
            )
            plan = {
                "version": 1,
                "question": "原文精确线索",
                "intent": {},
                "initial_queries": ["原文精确线索"],
                "followup_queries": [],
                "initial_query_embeddings_float32": [[1.0, 0.0, 0.0, 0.0]],
                "followup_query_embeddings_float32": [],
            }

            enabled = run_frozen_query_plan(
                engine, plan, episode_limit=2, rerank=False
            )
            self.assertEqual(enabled["paragraph_ranking_ids"], [[paragraph_id]])
            self.assertIn(episode_one, enabled["graph_candidate_episode_ids"])
            self.assertIn(episode_one, enabled["episode_ids"])

            config.paragraph.enabled = False
            disabled = run_frozen_query_plan(
                engine, plan, episode_limit=2, rerank=False
            )
            self.assertEqual(disabled["paragraph_ranking_ids"], [[]])
            self.assertNotIn(episode_one, disabled["graph_candidate_episode_ids"])
            self.assertEqual(disabled["episode_ids"], [episode_two])

    def test_identical_rerank_input_reuses_one_complete_model_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.retrieval.sparse_enabled = False
            config.retrieval.graph_max_hops = 0
            database = Database(config.database_path)
            database.initialize()
            sources = SourceRepository(database)
            episodes = EpisodeRepository(database)
            concepts = ConceptRepository(database)
            associations = AssociationRepository(database, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            source_id = sources.insert("直接证据")
            episode_id = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("直接证据"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            episode_index.upsert(episode_id, [1.0, 0.0, 0.0, 0.0])
            model = CountingRerankModel()
            engine = QueryEngine(
                config,
                model,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
            )
            plan = {
                "version": 1,
                "question": "直接证据",
                "intent": {},
                "initial_queries": ["直接证据"],
                "followup_queries": [],
                "initial_query_embeddings_float32": [[1.0, 0.0, 0.0, 0.0]],
                "followup_query_embeddings_float32": [],
            }
            cache: dict = {}

            first = run_frozen_query_plan(
                engine, plan, episode_limit=1, rerank=True, rerank_cache=cache
            )
            calls_after_first = model.rerank_calls
            second = run_frozen_query_plan(
                engine, plan, episode_limit=1, rerank=True, rerank_cache=cache
            )

            self.assertGreater(calls_after_first, 0)
            self.assertEqual(model.rerank_calls, calls_after_first)
            self.assertFalse(first["rerank_trace"]["cache_hit"])
            self.assertTrue(second["rerank_trace"]["cache_hit"])
            self.assertEqual(
                first["rerank_trace"]["input_hash"],
                second["rerank_trace"]["input_hash"],
            )

    def test_paragraph_can_supply_rerank_context_without_adding_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.paragraph.enabled = True
            config.retrieval.paragraph_seed_enabled = False
            config.retrieval.paragraph_rerank_context_enabled = True
            config.retrieval.sparse_enabled = False
            config.retrieval.graph_max_hops = 0
            config.retrieval.episode_top_k = 2
            config.retrieval.concept_top_k = 0
            config.retrieval.paragraph_top_k = 1
            database = Database(config.database_path)
            database.initialize()
            sources = SourceRepository(database)
            episodes = EpisodeRepository(database)
            concepts = ConceptRepository(database)
            paragraphs = ParagraphRepository(database)
            associations = AssociationRepository(database, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            paragraph_index = EmbeddingIndex(4)
            source_id = sources.insert("原始记录里的关键对白")
            episode_id = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("摘要没有保留关键措辞"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            episode_index.upsert(episode_id, [1.0, 0.0, 0.0, 0.0])
            paragraph_id = paragraphs.insert_many(
                source_id,
                "main/a.json",
                0,
                [ParagraphDraft(0, "原始记录里的关键对白")],
                [encode_embedding([1.0, 0.0, 0.0, 0.0], 4)],
            )[0]
            paragraph_index.upsert(paragraph_id, [1.0, 0.0, 0.0, 0.0])
            model = CountingRerankModel()
            engine = QueryEngine(
                config,
                model,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
                paragraph_index=paragraph_index,
                paragraphs=paragraphs,
            )
            plan = {
                "version": 1,
                "question": "关键对白是什么",
                "intent": {},
                "initial_queries": ["关键对白是什么"],
                "followup_queries": [],
                "initial_query_embeddings_float32": [[1.0, 0.0, 0.0, 0.0]],
                "followup_query_embeddings_float32": [],
            }

            result = run_frozen_query_plan(
                engine, plan, episode_limit=1, rerank=True
            )

            self.assertEqual(result["graph_candidate_episode_ids"], [episode_id])
            self.assertEqual(result["paragraph_context_source_ids"], [source_id])
            self.assertEqual(
                result["rerank_trace"]["paragraph_context_ids"], [paragraph_id]
            )
            self.assertIn("原始记录里的关键对白", model.rerank_prompts[0])
            self.assertIn("source_level_raw_paragraph", model.rerank_prompts[0])


if __name__ == "__main__":
    unittest.main()
