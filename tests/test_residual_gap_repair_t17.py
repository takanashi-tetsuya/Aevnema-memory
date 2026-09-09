from __future__ import annotations

import json
import unittest

import numpy as np

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import (
    RequirementResolution,
    ResidualSourceMappingReceipt,
    plan_local_residual_repair,
)
from memory_demo.types import EvidenceSlot, QueryVector, QueryVectorBundle


class _RowsRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = {int(row["id"]): dict(row) for row in rows}

    def get_many(self, ids):
        return [
            dict(self.rows[int(value)])
            for value in ids
            if int(value) in self.rows
        ]


def _source(source_id: int, raw_text: str) -> dict:
    return {"id": source_id, "raw_text": raw_text}


def _episode(
    episode_id: int,
    source_id: int,
    *,
    source_key: str,
    raw_text: str,
) -> dict:
    return {
        "id": episode_id,
        "source_id": source_id,
        "source_key": source_key,
        "text": raw_text,
        "participants_json": "[]",
        "segment_index": episode_id,
        "story_time_text": "",
        "story_order": episode_id,
        "timeline_scope": "",
        "epistemic_note": "",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "evidence_origin": "source",
        "generation": 0,
        "epistemic_status": "asserted",
        "evidence_basis": "reasoning_view_nonempty_lines_v1",
        "evidence_quotes_json": json.dumps([raw_text]),
        "evidence_spans_json": json.dumps([[1, 2]]),
    }


def _episode_view(episode_id: int, score: float) -> dict:
    return {
        "id": episode_id,
        "score": score,
        "source_key": f"source-{episode_id}",
    }


class ResidualRepairPlanTests(unittest.TestCase):
    def test_plan_contains_only_the_current_missing_required_slot(self) -> None:
        resolution = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(
                EvidenceSlot("covered", "covered question", query_id="q-covered"),
                EvidenceSlot("missing", "missing question", query_id="q-missing"),
                EvidenceSlot(
                    "optional",
                    "optional question",
                    query_id="q-optional",
                    required=False,
                ),
            ),
            planner_origin="explicit",
        )

        plan = plan_local_residual_repair(resolution, ["missing"])

        self.assertTrue(plan.enabled)
        self.assertEqual(("missing",), plan.slot_ids)
        payload = plan.as_trace_payload()
        self.assertEqual(["missing"], payload["slot_ids"])
        self.assertNotIn("missing question", repr(payload))
        self.assertFalse(payload["whole_query_fallback"])

    def test_plan_fails_closed_for_stale_unknown_missing_slot(self) -> None:
        resolution = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(EvidenceSlot("known", "known question"),),
            planner_origin="explicit",
        )

        plan = plan_local_residual_repair(resolution, ["unknown"])

        self.assertFalse(plan.enabled)
        self.assertEqual("invalid_dynamic_missing_slots", plan.reason)
        self.assertEqual(("unknown",), plan.invalid_missing_slot_ids)

    def test_plan_caps_local_queries_and_records_dropped_slots(self) -> None:
        resolution = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(
                EvidenceSlot(
                    f"slot-{index}",
                    f"question {index}",
                    query_id=f"query-{index}",
                    clause_ids=(f"clause-{index}",),
                )
                for index in range(1, 6)
            ),
            planner_origin="explicit",
        )

        plan = plan_local_residual_repair(
            resolution,
            [f"slot-{index}" for index in range(1, 6)],
        )

        self.assertEqual(("slot-1", "slot-2", "slot-3", "slot-4"), plan.slot_ids)
        self.assertEqual(("slot-5",), plan.dropped_slot_ids)
        self.assertEqual(4, plan.as_trace_payload()["query_limit"])


class ResidualGapRepairEngineTests(unittest.TestCase):
    def _engine(self, rows: list[dict], sources: list[dict]) -> QueryEngine:
        config = AppConfig(model=ModelConfig(embedding_dimension=3))
        config.retrieval.contextual_association_enabled = True
        config.retrieval.answer_episode_limit = 2
        config.retrieval.candidate_limit = 8
        config.retrieval.episode_top_k = 4
        config.retrieval.concept_top_k = 0
        config.retrieval.sparse_enabled = False
        return QueryEngine(
            config,
            model=object(),
            episode_index=EmbeddingIndex(3),
            concept_index=EmbeddingIndex(3),
            episodes=_RowsRepository(rows),
            concepts=_RowsRepository([]),
            sources=_RowsRepository(sources),
            associations=object(),
        )

    @staticmethod
    def _requirements() -> RequirementResolution:
        return RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(
                EvidenceSlot(
                    "slot-a",
                    "fact a",
                    query_id="query-a",
                    query_refs=("query-a",),
                    clause_ids=("clause-a",),
                ),
                EvidenceSlot(
                    "slot-b",
                    "fact b",
                    query_id="query-b",
                    query_refs=("query-b",),
                    clause_ids=("clause-b",),
                ),
            ),
            planner_origin="explicit",
        )

    @staticmethod
    def _bundle(
        *,
        slot_b_query_id: str = "query-b",
        slot_b_vector: object | None = None,
    ) -> QueryVectorBundle:
        resolved_slot_b_vector = np.asarray(
            [0.0, 0.0, 1.0]
            if slot_b_vector is None
            else slot_b_vector,
            dtype=np.float32,
        )
        return QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=(
                QueryVector(
                    query_id="whole",
                    text_hash="whole",
                    role="whole",
                    vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    text="whole question",
                ),
                QueryVector(
                    query_id="query-a",
                    text_hash="a",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="slot-a",
                    text="fact a",
                ),
                QueryVector(
                    query_id=slot_b_query_id,
                    text_hash="b",
                    role="atomic",
                    vector=resolved_slot_b_vector,
                    slot_id="slot-b",
                    text="fact b",
                ),
            ),
        )

    @staticmethod
    def _bundle_with_wrong_slot_b_query_id() -> QueryVectorBundle:
        # Same slot text/vector but a different logical query ID must not be
        # reused for slot-b residual repair.
        return ResidualGapRepairEngineTests._bundle(
            slot_b_query_id="other-query-b",
        )

    def _repair_inputs(self):
        raw_a = "[record: 1]\nproof a"
        raw_b = "[record: 1]\nproof b"
        rows = [
            _episode(1, 11, source_key="story/a.json", raw_text=raw_a),
            _episode(2, 22, source_key="story/b.json", raw_text=raw_b),
        ]
        return rows, [_source(11, raw_a), _source(22, raw_b)]

    def _receipts(
        self,
        engine: QueryEngine,
        *,
        episode_ids: tuple[int, ...] = (1, 2),
    ) -> tuple[ResidualSourceMappingReceipt, ...]:
        facts, _reasons = engine._v3_source_fact_closure(episode_ids)
        requirements = {item.slot_id: item for item in self._requirements().requirements}
        result: list[ResidualSourceMappingReceipt] = []
        for episode_id, slot_id in ((1, "slot-a"), (2, "slot-b")):
            if episode_id not in episode_ids:
                continue
            slot = requirements[slot_id]
            result.append(
                ResidualSourceMappingReceipt(
                    episode_id=episode_id,
                    slot_id=slot.slot_id,
                    query_id=slot.query_id,
                    clause_ids=slot.clause_ids,
                    source_fact_id=facts[episode_id].fact_id,
                )
            )
        return tuple(result)

    def test_candidate_missing_uses_only_missing_slot_vector_but_remains_undelivered(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)
        bundle = self._bundle()
        observed: list[tuple[list[str], np.ndarray]] = []

        def local_only(queries, matrix, *args, **kwargs):
            observed.append((list(queries), np.asarray(matrix).copy()))
            return [], {
                "atomic_episode": [[]],
                "sparse_source_episode_expansion": [[]],
                "fused_episode": [[]],
            }

        engine._vector_seed_hits_from_matrix = local_only  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={
                "deterministic_evidence_floor": {
                    "atomic_slots": [
                        {
                            "candidate_missing_episode_ids": [2],
                        }
                    ],
                    "constraint_slots": [],
                },
            },
            authoritative_requirements=self._requirements(),
            query_vector_bundle=bundle,
            mapping_receipts=self._receipts(engine),
            dynamic_missing_slot_ids=("slot-b",),
        )

        self.assertEqual([["fact b"]], [queries for queries, _matrix in observed])
        self.assertEqual(
            [0.0, 0.0, 1.0],
            observed[0][1][0].tolist(),
        )
        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([1], trace["preserved_selected_episode_ids"])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual([2], trace["candidate_missing_candidate_episode_ids"])
        self.assertEqual([2], trace["ranking_only_candidate_episode_ids"])
        self.assertEqual("semantic_verifier_unavailable", trace["reason"])
        self.assertFalse(trace["semantic_verifier_available"])
        self.assertTrue(trace["candidate_missing_delivery_disabled"])
        self.assertIn(
            "current_source_fact_identity_diagnostic",
            [item.get("status") for item in trace["mapping_receipts"]],
        )
        self.assertTrue(trace["full_rerank_skipped"])
        self.assertFalse(trace["whole_query_fallback"])
        self.assertEqual(0, trace["provider_calls"])

    def test_vector_only_candidate_is_traced_and_early_stops_without_mapping(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)
        engine.episode_index.add(2, [0.0, 0.0, 1.0])
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={
                "deterministic_evidence_floor": {
                    "atomic_slots": [],
                    "constraint_slots": [],
                },
            },
            authoritative_requirements=self._requirements(),
            query_vector_bundle=self._bundle(),
            mapping_receipts=self._receipts(engine, episode_ids=(1,)),
            dynamic_missing_slot_ids=("slot-b",),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual("semantic_verifier_unavailable", trace["reason"])
        self.assertTrue(trace["early_stopped"])
        self.assertIn(2, trace["ranking_only_candidate_episode_ids"])
        self.assertEqual(["slot-b"], trace["bound_slot_ids"])

    def test_no_dynamic_missing_slots_stops_before_local_index_read(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("no dynamic gap must not read the local index")

        engine._vector_seed_hits_from_matrix = should_not_run  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={"merged_coverage": {"coverage": []}},
            authoritative_requirements=self._requirements(),
            query_vector_bundle=self._bundle(),
            mapping_receipts=self._receipts(engine, episode_ids=(1,)),
            dynamic_missing_slot_ids=(),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual("no_dynamic_missing_slots", trace["reason"])
        self.assertTrue(trace["early_stopped"])
        self.assertEqual(0, trace["local_index_rounds"])

    def test_mismatched_query_or_fact_receipt_fails_closed_and_is_trace_only(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)
        bundle = self._bundle()
        valid_a, valid_b = self._receipts(engine)
        invalid_b = ResidualSourceMappingReceipt(
            episode_id=valid_b.episode_id,
            slot_id=valid_b.slot_id,
            query_id="query-a",
            clause_ids=valid_b.clause_ids,
            source_fact_id=valid_b.source_fact_id,
        )
        wrong_fact_b = ResidualSourceMappingReceipt(
            episode_id=valid_b.episode_id,
            slot_id=valid_b.slot_id,
            query_id=valid_b.query_id,
            clause_ids=valid_b.clause_ids,
            source_fact_id="source-fact:sha256:" + "0" * 64,
        )

        def local_only(_queries, _matrix, *args, **kwargs):
            return [], {
                "atomic_episode": [[]],
                "sparse_source_episode_expansion": [[]],
                "fused_episode": [[]],
            }

        engine._vector_seed_hits_from_matrix = local_only  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={
                "deterministic_evidence_floor": {
                    "atomic_slots": [{"candidate_missing_episode_ids": [2]}],
                    "constraint_slots": [],
                }
            },
            authoritative_requirements=self._requirements(),
            query_vector_bundle=bundle,
            mapping_receipts=(valid_a, invalid_b, wrong_fact_b),
            dynamic_missing_slot_ids=("slot-b",),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual("semantic_verifier_unavailable", trace["reason"])
        self.assertTrue(trace["early_stopped"])
        self.assertIn(
            "query_id_mismatch",
            [item.get("status") for item in trace["mapping_receipts"]],
        )
        self.assertIn(
            "source_fact_identity_mismatch",
            [item.get("status") for item in trace["mapping_receipts"]],
        )

    def test_malformed_slot_vector_binding_fails_closed_before_local_read(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("malformed slot vector must stop before local read")

        engine._vector_seed_hits_from_matrix = should_not_run  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={"deterministic_evidence_floor": {"atomic_slots": [], "constraint_slots": []}},
            authoritative_requirements=self._requirements(),
            query_vector_bundle=self._bundle(slot_b_vector=[0.0, 0.0, 2.0]),
            mapping_receipts=self._receipts(engine),
            dynamic_missing_slot_ids=("slot-b",),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual("missing_exact_slot_vector_binding", trace["reason"])
        self.assertEqual(["slot-b"], trace["skipped_slot_ids"])
        self.assertEqual(0, trace["local_index_rounds"])

    def test_same_text_different_logical_query_binding_fails_closed(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("mismatched logical query must stop before local read")

        engine._vector_seed_hits_from_matrix = should_not_run  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[_episode_view(1, 0.8), _episode_view(2, 0.6)],
            rerank_trace={"deterministic_evidence_floor": {"atomic_slots": [], "constraint_slots": []}},
            authoritative_requirements=self._requirements(),
            query_vector_bundle=self._bundle_with_wrong_slot_b_query_id(),
            mapping_receipts=self._receipts(engine),
            dynamic_missing_slot_ids=("slot-b",),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual("missing_exact_slot_vector_binding", trace["reason"])
        self.assertEqual(["slot-b"], trace["skipped_slot_ids"])
        self.assertEqual(0, trace["local_index_rounds"])

    def test_local_query_cap_limits_one_round_to_four_missing_slots(self) -> None:
        rows, sources = self._repair_inputs()
        engine = self._engine(rows, sources)
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(
                EvidenceSlot(
                    f"slot-{index}",
                    f"fact {index}",
                    query_id=f"query-{index}",
                    clause_ids=(f"clause-{index}",),
                )
                for index in range(1, 6)
            ),
            planner_origin="explicit",
        )
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=(
                QueryVector(
                    query_id="whole",
                    text_hash="whole",
                    role="whole",
                    vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    text="whole question",
                ),
                *tuple(
                    QueryVector(
                        query_id=f"query-{index}",
                        text_hash=f"hash-{index}",
                        role="atomic",
                        vector=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                        slot_id=f"slot-{index}",
                        text=f"fact {index}",
                    )
                    for index in range(1, 6)
                ),
            ),
        )
        observed: list[tuple[list[str], np.ndarray]] = []

        def local_only(queries, matrix, *args, **kwargs):
            observed.append((list(queries), np.asarray(matrix).copy()))
            return [], {}

        engine._vector_seed_hits_from_matrix = local_only  # type: ignore[method-assign]
        selected, trace = engine._repair_residual_evidence_slots(
            selected_episodes=[_episode_view(1, 0.8)],
            candidate_episodes=[],
            rerank_trace={},
            authoritative_requirements=requirements,
            query_vector_bundle=bundle,
            mapping_receipts=(),
            dynamic_missing_slot_ids=(
                "slot-5",
                "slot-4",
                "slot-3",
                "slot-2",
                "slot-1",
            ),
        )

        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([], trace["added_episode_ids"])
        self.assertEqual(
            [["fact 1", "fact 2", "fact 3", "fact 4"]],
            [queries for queries, _matrix in observed],
        )
        self.assertEqual((4, 3), observed[0][1].shape)
        self.assertEqual(
            ["slot-1", "slot-2", "slot-3", "slot-4"],
            trace["bound_slot_ids"],
        )
        self.assertEqual(["slot-5"], trace["dropped_slot_ids"])
        self.assertEqual(1, trace["round_limit"])
        self.assertEqual(1, trace["local_index_rounds"])
        self.assertEqual("semantic_verifier_unavailable", trace["reason"])


if __name__ == "__main__":
    unittest.main()
