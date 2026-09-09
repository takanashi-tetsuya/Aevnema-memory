from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingCoordinator, EmbeddingIndex
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
)
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.types import EvidenceSlot, QueryIntent, QueryVectorRequest


class _CountingEmbeddingModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for index, _text in enumerate(texts, start=1):
            row = np.zeros(self.dimension, dtype=np.float32)
            row[(index - 1) % self.dimension] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


class _QueryModel(_CountingEmbeddingModel):
    def chat_json(self, *_args, **_kwargs):
        raise AssertionError("query test supplies an explicit intent and follow-up")


class QueryVectorBundleV3Tests(unittest.TestCase):
    def _engine(self, model: _QueryModel) -> QueryEngine:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        config = AppConfig(
            database_path=root / "memory.db",
            log_dir=root / "logs",
            model=ModelConfig(embedding_dimension=model.dimension),
        )
        config.retrieval.rerank_enabled = False
        config.retrieval.growth_max_rounds = 0
        config.retrieval.followup_planning_mode = "off"
        database = Database(config.database_path)
        database.initialize()
        return QueryEngine(
            config,
            model,
            EmbeddingIndex(model.dimension),
            EmbeddingIndex(model.dimension),
            EpisodeRepository(database),
            ConceptRepository(database),
            SourceRepository(database),
            AssociationRepository(database, config.weights),
        )

    def test_one_physical_vector_keeps_every_duplicate_logical_slot(self) -> None:
        model = _CountingEmbeddingModel()
        bundle = EmbeddingCoordinator(model, model_id="bge-test", dimension=3).embed_request_bundle_sync(
            [
                QueryVectorRequest("whole", "same query", "whole-q"),
                QueryVectorRequest("atomic", "same query", "slot-a-q", "slot-a"),
                QueryVectorRequest("followup", "same query", "slot-b-q", "slot-b"),
            ]
        )

        self.assertEqual([["same query"]], model.calls)
        self.assertEqual(1, bundle.physical_count)
        self.assertEqual(3, bundle.logical_count)
        self.assertEqual(
            {
                bundle.queries[0].physical_id,
                bundle.queries[1].physical_id,
                bundle.queries[2].physical_id,
            },
            {bundle.whole_ref.physical_id},
        )
        self.assertEqual(
            (bundle.queries[1].physical_id,),
            tuple(ref.physical_id for ref in bundle.query_refs_for_slot("slot-a")),
        )
        self.assertEqual(
            (bundle.queries[2].physical_id,),
            tuple(ref.physical_id for ref in bundle.query_refs_for_slot("slot-b")),
        )
        self.assertEqual(bundle.queries[1].embedding_space_id, bundle.embedding_space_id)

    def test_same_text_can_bind_two_authoritative_slots_without_fuzzy_cross_binding(self) -> None:
        resolution = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(
                EvidenceSlot(
                    "slot-a",
                    "same requirement",
                    query_id="query-a",
                    query_refs=("query-a",),
                ),
                EvidenceSlot(
                    "slot-b",
                    "same requirement",
                    query_id="query-b",
                    query_refs=("query-b",),
                ),
            ),
            planner_origin="explicit",
        )
        specs = QueryEngine._request_vector_specs(
            ["same requirement"],
            role="atomic",
            authoritative_requirements=resolution,
        )
        bundle = EmbeddingCoordinator(
            _CountingEmbeddingModel(), model_id="bge-test", dimension=3
        ).embed_request_bundle_sync(specs)

        self.assertEqual(["slot-a", "slot-b"], [item.slot_id for item in specs])
        self.assertEqual(1, bundle.physical_count)
        self.assertEqual(2, bundle.logical_count)
        self.assertEqual("slot-a", QueryEngine._slot_query_vector(bundle, resolution.requirements[0]).slot_id)
        self.assertEqual("slot-b", QueryEngine._slot_query_vector(bundle, resolution.requirements[1]).slot_id)

    def test_public_query_uses_one_initial_bundle_and_reuses_shared_followup(self) -> None:
        model = _QueryModel()
        engine = self._engine(model)
        result = engine.query(
            "whole query",
            generate_answer=False,
            intent_override=QueryIntent(search_queries=["shared requirement"]),
            followup_queries_override=["shared requirement"],
        )

        self.assertEqual([["whole query", "shared requirement"]], model.calls)
        self.assertEqual(2, result["query_vector_bundle"]["physical_vector_count"])
        self.assertEqual(3, result["query_vector_bundle"]["logical_binding_count"])
        vector_records = result["contextual_association"]["query_vectors"]
        self.assertTrue(vector_records)
        self.assertTrue(all("text" not in record for record in vector_records))
        shared = [record for record in vector_records if record["text_hash"] != vector_records[0]["text_hash"]]
        self.assertEqual(2, len(shared))
        self.assertEqual(shared[0]["physical_id"], shared[1]["physical_id"])

    def test_unique_followup_is_an_explicit_second_coordinator_batch(self) -> None:
        model = _QueryModel()
        engine = self._engine(model)
        result = engine.query(
            "whole query",
            generate_answer=False,
            intent_override=QueryIntent(search_queries=["initial requirement"]),
            followup_queries_override=["new followup discovery"],
        )

        self.assertEqual(
            [
                ["whole query", "initial requirement"],
                ["new followup discovery"],
            ],
            model.calls,
        )
        # `_vector_seed_hits_with_cues` receives aligned bundle overrides for
        # both stages rather than issuing a hidden direct embedding request.
        self.assertEqual(0, result["query_embedding_cache"]["miss_count"])
        self.assertEqual(3, result["query_vector_bundle"]["physical_vector_count"])
        self.assertEqual(3, result["query_vector_bundle"]["logical_binding_count"])

    def test_strict_bundle_rejects_missing_dynamic_followup_without_embedding(self) -> None:
        model = _QueryModel()
        engine = self._engine(model)
        coordinator = engine._new_query_embedding_coordinator()
        initial = coordinator.bundle_from_precomputed_vectors(
            [
                QueryVectorRequest("whole", "whole query", "whole"),
                QueryVectorRequest("atomic", "known requirement", "known", "slot-known"),
            ],
            [np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)],
        )

        with self.assertRaisesRegex(ValueError, "strict vector bundle is missing"):
            engine.query(
                "whole query",
                generate_answer=False,
                intent_override=QueryIntent(search_queries=["known requirement"]),
                followup_queries_override=["unplanned dynamic followup"],
                query_vector_bundle=initial,
                strict_vector_bundle=True,
            )
        self.assertEqual([], model.calls)

    def test_strict_bundle_rejects_legacy_override_before_provider_work(self) -> None:
        model = _QueryModel()
        engine = self._engine(model)

        with self.assertRaisesRegex(ValueError, "does not allow legacy"):
            engine.query(
                "whole query",
                generate_answer=False,
                intent_override=QueryIntent(search_queries=["known requirement"]),
                query_embeddings_override={
                    "whole query": np.array([1, 0, 0], dtype=np.float32)
                },
                strict_vector_bundle=True,
            )
        self.assertEqual([], model.calls)

    def test_query_rejects_same_dimension_but_different_embedding_space(self) -> None:
        model = _QueryModel()
        engine = self._engine(model)
        foreign = EmbeddingCoordinator(
            _CountingEmbeddingModel(),
            model_id="foreign-embedding-model",
            dimension=3,
        ).bundle_from_precomputed_vectors(
            [
                QueryVectorRequest("whole", "whole query", "whole"),
                QueryVectorRequest("atomic", "known requirement", "known", "slot-known"),
            ],
            [np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)],
        )

        with self.assertRaisesRegex(ValueError, "incompatible embedding space"):
            engine.query(
                "whole query",
                generate_answer=False,
                intent_override=QueryIntent(search_queries=["known requirement"]),
                query_vector_bundle=foreign,
                strict_vector_bundle=True,
            )
        self.assertEqual([], model.calls)
