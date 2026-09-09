from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.association_overlay import AssociationDelta, AssociationOverlay
from memory_demo.associations.plasticity import (
    classify_treatment_masked,
    derive_contextual_candidates,
)
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingCoordinator, EmbeddingIndex
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.coverage import (
    EvidenceCandidate,
    EvidenceSlot,
    select_evidence,
    strict_contextual_attribution,
)
from memory_demo.types import QueryVector, QueryVectorBundle, SlotCandidate


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.config = ModelConfig(embedding_dimension=3)

    def embed(self, texts):
        self.calls += 1
        rows = []
        for text in texts:
            if "need" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


class ContextualAssociationTests(unittest.TestCase):
    def test_search_many_matches_independent_searches(self):
        index = EmbeddingIndex(3, initial_capacity=1)
        vectors = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])
        for node_id, vector in enumerate(vectors, 1):
            index.add(node_id, vector)
        queries = np.asarray([[1, 0.2, 0], [0, 0.2, 1]], dtype=np.float32)
        batch = index.search_many(queries, 3, block_rows=2)
        expected = [index.search(row, 3) for row in queries]
        self.assertEqual([[item[0] for item in row] for row in expected], [[item[0] for item in row] for row in batch])

    def test_coordinator_deduplicates_and_batches_once(self):
        model = CountingModel()
        coordinator = EmbeddingCoordinator(model, model_id="test", dimension=3)
        bundle = coordinator.embed_query_bundle_sync(
            [("whole", "context"), ("atomic", "need"), ("followup", "need")]
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(coordinator.last_batch_size, 2)
        # The physical provider batch is deduplicated, but both logical need
        # roles remain available to downstream slot-aware selection.
        self.assertEqual(len(bundle.queries), 3)
        self.assertEqual(2, bundle.physical_count)
        self.assertEqual(
            bundle.queries[1].physical_id,
            bundle.queries[2].physical_id,
        )
        self.assertEqual(bundle.whole.dtype, np.float32)

    def test_double_key_requires_both_prototypes(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryApplication(
                AppConfig(
                    database_path=Path(directory) / "memory.db",
                    log_dir=Path(directory) / "logs",
                    model=ModelConfig(embedding_dimension=3),
                )
            )
            # Create minimal valid endpoints directly through repositories.
            with app.db.transaction() as connection:
                source = connection.execute(
                    "INSERT INTO source(raw_text) VALUES('x')"
                ).lastrowid
                now = "2026-01-01T00:00:00+00:00"
                for text in ("anchor", "target"):
                    connection.execute(
                        "INSERT INTO episode(source_id, source_key, segment_index, text, embedding, created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                        (source, "x", 0, text, np.asarray([1, 0, 0], dtype=np.float32).tobytes(), now, now),
                    )
            anchor_id, target_id = 1, 2
            context_id = app.associations.get_or_create_cue_prototype(
                domain="knowledge", cue_kind="context", model_id="test", dimension=3,
                vector=[1, 0, 0], text_hash="c", display_text="context",
                source_request_hash="r1",
            )
            need_id = app.associations.get_or_create_cue_prototype(
                domain="knowledge", cue_kind="need", model_id="test", dimension=3,
                vector=[0, 1, 0], text_hash="n", display_text="need",
                source_request_hash="r1",
            )
            candidate = derive_contextual_candidates(
                base_episode_ids=[anchor_id], selected_episode_ids=[anchor_id, target_id],
                target_episode_ids=[target_id], context_query_id="qc", need_query_id="qn",
                anchor_id=anchor_id, source_request_hash="r1",
            )[0]
            edge_id = app.associations.create_contextual(
                candidate, context_cue_id=context_id, need_cue_id=need_id,
            )
            app.context_cue_index.add(context_id, [1, 0, 0])
            app.need_cue_index.add(need_id, [0, 1, 0])
            matcher = ContextualAssociationMatcher(
                app.context_cue_index, app.need_cue_index, app.associations,
                context_threshold=0.5, need_threshold=0.5,
            )
            bundle = QueryVectorBundle(
                model_id="test", dimension=3, whole=np.asarray([1, 0, 0], dtype=np.float32),
                queries=(QueryVector("q", "h", "atomic", np.asarray([0, 1, 0], dtype=np.float32)),),
            )
            result = matcher.match_bundle(
                bundle,
                domain="knowledge",
                active_anchor_ids={anchor_id: 1.0},
                unresolved_slot_ids=["q"],
                evaluation_as_of="2026-01-01T00:00:00+00:00",
            )
            self.assertEqual([edge_id], [hit.association_id for hit in result["hits"]])
            self.assertEqual([target_id], result["attached_episode_ids"])
            self.assertEqual("q", result["hits"][0].matched_slot_id)
            self.assertEqual(0, result["external_calls"])

    def test_prepared_early_key_modes_keep_the_same_anchor_contract(self):
        class Repo:
            @staticmethod
            def get_contextual_for_prototypes(*_args, **_kwargs):
                return [
                    {
                        "id": 17,
                        "from_type": "episode",
                        "from_id": 1,
                        "to_id": 2,
                        "context_cue_id": 10,
                        "need_cue_id": 20,
                        "lifecycle_state": "active",
                        "utility_weight": 1.0,
                    }
                ]

        context = EmbeddingIndex(3)
        need = EmbeddingIndex(3)
        context.add(10, [1, 0, 0])
        need.add(20, [0, 1, 0])
        matcher = ContextualAssociationMatcher(
            context,
            need,
            Repo(),
            context_threshold=0.5,
            need_threshold=0.5,
        )
        # The current request still supplies a legal slot vector, but it does
        # not pass the N cue gate. C and W are intentionally diagnostic key
        # ablations; CN retains the ordinary double-key rejection.
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1, 0, 0], dtype=np.float32),
            queries=(
                QueryVector(
                    "slot",
                    "slot-hash",
                    "atomic",
                    np.asarray([1, 0, 0], dtype=np.float32),
                    slot_id="slot",
                ),
            ),
        )
        common = {
            "domain": "knowledge",
            "active_anchor_ids": {1: 1.0},
            "unresolved_slot_ids": ["slot"],
            "evaluation_as_of": "2026-01-01T00:00:00+00:00",
        }
        cn = matcher.match_bundle(bundle, scoring_mode="cn", **common)
        c = matcher.match_bundle(bundle, scoring_mode="c", **common)
        n = matcher.match_bundle(bundle, scoring_mode="n", **common)
        w = matcher.match_bundle(bundle, scoring_mode="w", **common)

        self.assertEqual([], cn["pre_target_proposals"])
        self.assertEqual([], n["pre_target_proposals"])
        self.assertEqual([17], [item.association_id for item in c["pre_target_proposals"]])
        self.assertEqual([17], [item.association_id for item in w["pre_target_proposals"]])
        self.assertEqual("c", c["gate_trace"]["scoring_mode"])
        self.assertEqual("w", w["gate_trace"]["scoring_mode"])

    def test_double_key_never_runs_without_a_base_anchor(self):
        class Repo:
            def get_contextual_for_prototypes(self, *_args, **_kwargs):
                raise AssertionError("repository must not be consulted")

        index = EmbeddingIndex(3)
        index.add(1, [1, 0, 0])
        matcher = ContextualAssociationMatcher(index, index, Repo())
        result = matcher.match(
            [1, 0, 0],
            [[1, 0, 0]],
            query_ids=["q"],
            slot_ids=["slot-a"],
            unresolved_slot_ids=["slot-a"],
            evaluation_as_of="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual([], result["hits"])
        self.assertEqual("no_active_anchor", result["reason"])

    def test_double_key_requires_local_target_support_for_the_missing_slot(self):
        class Repo:
            @staticmethod
            def get_contextual_for_prototypes(*_args, **_kwargs):
                return [
                    {
                        "id": 17,
                        "from_type": "episode",
                        "from_id": 1,
                        "to_id": 2,
                        "context_cue_id": 10,
                        "need_cue_id": 20,
                        "lifecycle_state": "probation",
                        "utility_weight": 1.0,
                    }
                ]

        context = EmbeddingIndex(3)
        need = EmbeddingIndex(3)
        context.add(10, [1, 0, 0])
        need.add(20, [0, 1, 0])
        matcher = ContextualAssociationMatcher(
            context,
            need,
            Repo(),
            context_threshold=0.5,
            need_threshold=0.5,
        )
        no_support = matcher.match(
            [1, 0, 0],
            [[0, 1, 0]],
            query_ids=["q"],
            slot_ids=["missing"],
            active_anchor_ids={1: 1.0},
            unresolved_slot_ids=["missing"],
            target_support_scores={},
            target_support_floor=0.05,
            evaluation_as_of="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual([], no_support["hits"])
        supported = matcher.match(
            [1, 0, 0],
            [[0, 1, 0]],
            query_ids=["q"],
            slot_ids=["missing"],
            active_anchor_ids={1: 1.0},
            unresolved_slot_ids=["missing"],
            target_support_scores={("missing", 2): 0.6},
            target_support_floor=0.05,
            evaluation_as_of="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual([17], [item.association_id for item in supported["hits"]])

    def test_contextual_bonus_cannot_replace_an_already_covered_slot(self):
        slots = [EvidenceSlot("required", "required fact")]
        base = EvidenceCandidate(
            episode_id=1,
            supported_slots=frozenset({"required"}),
            direct_relevance=0.8,
            source_quality=0.1,
        )
        contextual_duplicate = SlotCandidate(
            episode_id=2,
            slot_ids=frozenset({"required"}),
            direct_score=0.1,
            contextual_score=1.0,
            contextual_edge_id=9,
        )
        selection = select_evidence(
            [base, contextual_duplicate], slots, budget=1
        )
        self.assertEqual([1], [item.episode_id for item in selection.selected])

    def test_coverage_slot_reuses_its_batched_query_vector_identity(self):
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1, 0, 0], dtype=np.float32),
            queries=(
                QueryVector(
                    "q-need",
                    "h",
                    "atomic",
                    np.asarray([0, 1, 0], dtype=np.float32),
                    text="missing fact",
                ),
            ),
        )
        slots, support = QueryEngine._request_evidence_slots(
            {
                "merged_coverage": {
                    "coverage": [
                        {"query": "missing fact", "episode_ids": [4]}
                    ]
                }
            },
            bundle,
        )
        self.assertEqual("q-need", slots[0].slot_id)
        self.assertEqual("q-need", slots[0].query_id)
        self.assertEqual({"q-need"}, support[4])

    def test_strict_attribution_requires_a_selected_new_slot(self):
        slots = [
            EvidenceSlot("anchor", "anchor fact"),
            EvidenceSlot("missing", "recovered fact"),
        ]
        base = SlotCandidate(
            episode_id=1,
            slot_ids=frozenset({"anchor"}),
            direct_score=0.9,
        )
        recovered = SlotCandidate(
            episode_id=2,
            slot_ids=frozenset({"missing"}),
            direct_score=0.3,
            contextual_edge_id=7,
            contextual_score=0.8,
        )
        trace = strict_contextual_attribution([base, recovered], slots, budget=2)
        self.assertEqual(["missing"], trace["delta"]["new_slots"])
        self.assertEqual(7, trace["edges"][0]["association_id"])
        self.assertTrue(trace["edges"][0]["sufficient"])
        self.assertTrue(trace["edges"][0]["necessary"])

    def test_utility_is_future_only_and_duplicate_hash_is_ignored(self):
        observation = classify_treatment_masked(7, "q1", ["slot-a"], [])
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryApplication(
                AppConfig(database_path=Path(directory) / "memory.db", log_dir=Path(directory) / "logs", model=ModelConfig(embedding_dimension=3))
            )
            with app.db.transaction() as connection:
                source = connection.execute("INSERT INTO source(raw_text) VALUES('x')").lastrowid
                now = "2026-01-01T00:00:00+00:00"
                for text in ("a", "b"):
                    connection.execute(
                        "INSERT INTO episode(source_id,source_key,segment_index,text,embedding,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (source, "x", 0, text, np.asarray([1, 0, 0], dtype=np.float32).tobytes(), now, now),
                    )
            c = app.associations.get_or_create_cue_prototype(domain="knowledge", cue_kind="context", model_id="t", dimension=3, vector=[1,0,0], text_hash="c", source_request_hash="r")
            n = app.associations.get_or_create_cue_prototype(domain="knowledge", cue_kind="need", model_id="t", dimension=3, vector=[0,1,0], text_hash="n", source_request_hash="r")
            candidate = derive_contextual_candidates(base_episode_ids=[1], selected_episode_ids=[1,2], target_episode_ids=[2], context_query_id="c", need_query_id="n", anchor_id=1, source_request_hash="r")[0]
            edge_id = app.associations.create_contextual(candidate, context_cue_id=c, need_cue_id=n)
            # The creation itself did not count as success.
            self.assertEqual(0, app.associations.get(edge_id)["utility_successes"])
            observation = classify_treatment_masked(edge_id, "q1", ["slot-a"], [])
            app.associations.record_utility([observation, observation])
            self.assertEqual(1, app.associations.get(edge_id)["utility_successes"])
            app.associations.record_utility([classify_treatment_masked(edge_id, "q2", ["slot-a"], [])])
            self.assertEqual("probation", app.associations.get(edge_id)["lifecycle_state"])

    def test_contextual_overlay_hides_edge_and_cues_without_writes(self):
        rows = {
            9: {
                "id": 9, "from_type": "episode", "from_id": 1,
                "to_type": "episode", "to_id": 2, "association_mode": "contextual_recall",
                "context_cue_id": 3, "need_cue_id": 4, "lifecycle_state": "probation",
            }
        }

        class Repo:
            def get_contextual_for_prototypes(self, *_args, **_kwargs):
                return [rows[9]]
            def snapshot(self, *_args, **_kwargs): return rows
            def neighbors(self, *_args, **_kwargs): return []
            def get(self, association_id): return rows.get(association_id)

        overlay = AssociationOverlay(Repo(), hidden_ids=[9], hidden_context_cue_ids=[3], hidden_need_cue_ids=[4])
        self.assertEqual([], overlay.get_contextual_for_prototypes([3], [4]))
        self.assertIsNone(overlay.get(9))
        self.assertEqual(0, overlay.record_utility([])["updated"])
