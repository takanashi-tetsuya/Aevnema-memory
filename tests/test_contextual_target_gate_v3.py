from __future__ import annotations

from unittest.mock import patch
import unittest

import numpy as np

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.retrieval.contextual_association import ContextualPreTargetProposal
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.types import ContextualSlotHit, EvidenceSlot, QueryVector, QueryVectorBundle


AS_OF = "2026-01-01T00:00:00+00:00"
LATER_AS_OF = "2026-01-03T00:00:00+00:00"


class _RowsRepository:
    def __init__(self, rows: list[dict]):
        self.rows = {int(row["id"]): dict(row) for row in rows}

    def get_many(self, ids):
        return [dict(self.rows[int(item)]) for item in ids if int(item) in self.rows]


class _CapturingMatcher:
    endpoint_limit = 1

    def __init__(self, result: dict | None = None):
        self.calls: list[dict] = []
        self.result = result

    def match_bundle(self, _bundle, **kwargs):
        self.calls.append(dict(kwargs))
        fallback = {
            "hits": [],
            "pre_target_proposals": [],
            "context_hits": [],
            "need_hits": [],
            "external_calls": 0,
            "attached_episode_ids": [],
            "attached_edges": [],
        }
        return dict(self.result) if self.result is not None else fallback


def _episode(
    episode_id: int,
    source_id: int,
    *,
    source_key: str = "story/example.json",
    source_version: str | None = "v1",
    updated_at: str = "2025-12-31T00:00:00+00:00",
) -> dict:
    row = {
        "id": episode_id,
        "source_id": source_id,
        "source_key": source_key,
        "updated_at": updated_at,
        "evidence_origin": "source",
        "generation": 0,
        "epistemic_status": "asserted",
        "evidence_basis": "reasoning_view_nonempty_lines_v1",
        "evidence_quotes_json": '["source evidence"]',
        "evidence_spans_json": "[[0, 15]]",
    }
    if source_version is not None:
        row["source_version"] = source_version
    return row


def _source(source_id: int, *, version: str | None = "v1") -> dict:
    row = {"id": source_id, "raw_text": f"source {source_id} text"}
    if version is not None:
        row["source_version"] = version
    return row


def _proposal(
    association_id: int,
    target_episode_id: int,
    slot_id: str,
    *,
    score: float,
    rank: int,
) -> ContextualPreTargetProposal:
    return ContextualPreTargetProposal(
        association_id=association_id,
        anchor_episode_id=100,
        target_episode_id=target_episode_id,
        context_cue_id=10,
        need_cue_id=20,
        matched_slot_id=slot_id,
        matched_query_id=f"query-{slot_id}",
        matched_query_role="atomic",
        matched_physical_id=f"physical-{slot_id}",
        embedding_space_id="test-space",
        context_similarity=0.9,
        need_similarity=0.9,
        context_gate_score=0.8,
        need_gate_score=0.8,
        anchor_activation=1.0,
        utility_weight=1.0,
        lifecycle_state="active",
        pre_target_score=score,
        proposal_key=f"proposal-{association_id}-{slot_id}",
        rank_before_endpoint_cap=rank,
    )


class ContextualTargetGateV3Tests(unittest.TestCase):
    def _engine(
        self,
        *,
        episodes: list[dict],
        sources: list[dict],
        target_vectors: dict[int, list[float]],
        matcher=None,
    ) -> QueryEngine:
        config = AppConfig(model=ModelConfig(embedding_dimension=3))
        config.retrieval.contextual_association_enabled = True
        config.retrieval.contextual_association_shadow = False
        episode_index = EmbeddingIndex(3)
        for episode_id, vector in target_vectors.items():
            episode_index.add(episode_id, vector)
        return QueryEngine(
            config,
            model=object(),
            episode_index=episode_index,
            concept_index=EmbeddingIndex(3),
            episodes=_RowsRepository(episodes),
            concepts=object(),
            sources=_RowsRepository(sources),
            associations=object(),
            contextual_matcher=matcher,
        )

    @staticmethod
    def _bundle_and_slots(*slot_ids: str):
        queries = tuple(
            QueryVector(
                query_id=f"query-{slot_id}",
                text_hash=f"hash-{slot_id}",
                role="atomic",
                vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id=slot_id,
                text=f"need {slot_id}",
            )
            for slot_id in slot_ids
        )
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=queries,
        )
        slots = [
            EvidenceSlot(
                slot_id=slot_id,
                question=f"need {slot_id}",
                query_id=f"query-{slot_id}",
            )
            for slot_id in slot_ids
        ]
        return bundle, slots

    def test_invalid_top_target_does_not_consume_cap_before_later_valid_target(self):
        bundle, slots = self._bundle_and_slots("slot")
        engine = self._engine(
            episodes=[_episode(1, 99), _episode(2, 2)],
            sources=[_source(2)],
            target_vectors={1: [0, 1, 0], 2: [0, 1, 0]},
        )

        result = engine._gate_contextual_targets(
            proposals=[
                _proposal(10, 1, "slot", score=0.95, rank=1),
                _proposal(11, 2, "slot", score=0.80, rank=2),
            ],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=AS_OF,
        )

        self.assertEqual([2], list(result.selected_endpoint_ids))
        self.assertEqual([2], [item.target_episode_id for item in result.hits])
        self.assertEqual(
            ["target_source_missing", "target_current_and_relevant"],
            [item.reason for item in result.outcomes],
        )

    def test_one_target_can_retain_multiple_slots_while_using_one_endpoint(self):
        bundle, slots = self._bundle_and_slots("slot-a", "slot-b", "slot-c")
        engine = self._engine(
            episodes=[_episode(2, 2), _episode(3, 3)],
            sources=[_source(2), _source(3)],
            target_vectors={2: [0, 1, 0], 3: [0, 1, 0]},
        )

        result = engine._gate_contextual_targets(
            proposals=[
                _proposal(20, 2, "slot-a", score=0.95, rank=1),
                _proposal(21, 2, "slot-b", score=0.90, rank=2),
                _proposal(22, 3, "slot-c", score=0.80, rank=3),
            ],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=AS_OF,
        )

        self.assertEqual((2,), result.selected_endpoint_ids)
        self.assertEqual({"slot-a", "slot-b"}, {item.matched_slot_id for item in result.hits})
        self.assertEqual(2, len(result.hits))
        self.assertEqual(
            "endpoint_budget_truncated",
            result.outcomes[2].reason,
        )
        self.assertEqual("rejected_endpoint_cap", result.outcomes[2].status)

    def test_missing_or_conflicting_source_version_fails_closed(self):
        bundle, slots = self._bundle_and_slots("slot")
        engine = self._engine(
            episodes=[
                _episode(1, 99),
                _episode(2, 2, source_version=""),
                _episode(3, 3, source_version="episode-v1"),
            ],
            sources=[_source(2), _source(3, version="source-v2")],
            target_vectors={1: [0, 1, 0], 2: [0, 1, 0], 3: [0, 1, 0]},
        )

        result = engine._gate_contextual_targets(
            proposals=[
                _proposal(30, 1, "slot", score=0.9, rank=1),
                _proposal(31, 2, "slot", score=0.8, rank=2),
                _proposal(32, 3, "slot", score=0.7, rank=3),
            ],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=3,
            evaluation_as_of=AS_OF,
        )

        self.assertEqual((), result.selected_endpoint_ids)
        self.assertEqual(
            {
                "target_source_missing",
                "target_source_version_missing",
                "target_source_version_conflict",
            },
            {item.reason for item in result.outcomes},
        )
        trace = result.as_trace_payload()
        self.assertNotIn("source_key", trace["outcomes"][0])
        self.assertIn("source_key_sha256", trace["outcomes"][0])

    def test_relevance_estimate_cannot_override_invalid_current_source_evidence(self):
        bundle, slots = self._bundle_and_slots("slot")
        invalid_evidence = _episode(2, 2)
        invalid_evidence["evidence_quotes_json"] = "[]"
        engine = self._engine(
            episodes=[invalid_evidence],
            sources=[_source(2)],
            # Perfect current vector relevance remains only an estimate.
            target_vectors={2: [0, 1, 0]},
        )

        result = engine._gate_contextual_targets(
            proposals=[_proposal(35, 2, "slot", score=1.0, rank=1)],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=AS_OF,
        )

        self.assertEqual((), result.selected_endpoint_ids)
        self.assertEqual("target_evidence_quotes_missing", result.outcomes[0].reason)
        self.assertEqual(1.0, result.outcomes[0].relevance_estimate)
        self.assertEqual("estimated", result.outcomes[0].relevance_status)

    def test_pinned_timestamp_is_deterministic_and_normal_wrapper_captures_once(self):
        bundle, slots = self._bundle_and_slots("slot")
        proposal = _proposal(40, 2, "slot", score=0.9, rank=1)
        engine = self._engine(
            episodes=[
                _episode(
                    2,
                    2,
                    updated_at="2026-01-02T00:00:00+00:00",
                )
            ],
            sources=[_source(2)],
            target_vectors={2: [0, 1, 0]},
        )

        early = engine._gate_contextual_targets(
            proposals=[proposal],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=AS_OF,
        )
        later_first = engine._gate_contextual_targets(
            proposals=[proposal],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=LATER_AS_OF,
        )
        later_second = engine._gate_contextual_targets(
            proposals=[proposal],
            bundle=bundle,
            unresolved_slots=slots,
            endpoint_limit=1,
            evaluation_as_of=LATER_AS_OF,
        )
        self.assertEqual("target_version_after_evaluation_as_of", early.outcomes[0].reason)
        self.assertEqual((2,), later_first.selected_endpoint_ids)
        self.assertEqual(
            later_first.as_trace_payload(), later_second.as_trace_payload()
        )

        capturing = _CapturingMatcher()
        normal_engine = self._engine(
            episodes=[], sources=[], target_vectors={}, matcher=capturing
        )
        with patch(
            "memory_demo.retrieval.engine.utc_now",
            side_effect=["2026-02-01T00:00:00+00:00", AssertionError("second clock read")],
        ):
            normal_engine.contextual_recall(
                bundle,
                active_anchor_ids={100: 1.0},
                unresolved_slot_ids=["slot"],
            )
        self.assertEqual(
            "2026-02-01T00:00:00+00:00",
            capturing.calls[0]["evaluation_as_of"],
        )

    def test_slot_selector_uses_one_uncapped_proposal_call_not_compatibility_hits(self):
        bundle, slots = self._bundle_and_slots("slot")
        invalid_proposal = _proposal(49, 1, "slot", score=0.99, rank=1)
        proposal = _proposal(50, 2, "slot", score=0.9, rank=2)
        matcher = _CapturingMatcher(
            {
                # A historical compatibility score for a missing target must
                # never override the current source/evidence gate. Only the
                # uncapped proposals below enter T09.
                "hits": [
                    ContextualSlotHit(
                        association_id=49,
                        anchor_episode_id=100,
                        target_episode_id=1,
                        matched_slot_id="slot",
                        target_support_score=999.0,
                        total_score=999.0,
                    )
                ],
                "pre_target_proposals": [invalid_proposal, proposal],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
            }
        )
        engine = self._engine(
            episodes=[_episode(2, 2)],
            sources=[_source(2)],
            target_vectors={2: [0, 1, 0]},
            matcher=matcher,
        )
        target_record = {
            **_episode(2, 2),
            "text": "target evidence",
            "participants_json": "[]",
            "segment_index": 0,
            "story_time_text": "",
            "story_order": None,
            "timeline_scope": "",
            "epistemic_note": "",
        }
        base_episode = {
            "id": 99,
            "score": 0.1,
            "source_key": "base",
            "generation": 0,
            "evidence_origin": "source",
        }
        engine._materialize_nodes = lambda _nodes, include_sources=True: ([dict(target_record)], [])  # type: ignore[method-assign]
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )

        selected, trace = engine._select_contextual_slots(
            episodes=[base_episode],
            baseline_selected=[base_episode],
            rerank_trace={},
            reranked_episode_ids=[99],
            bundle=bundle,
            domain=None,
            endpoint_limit=1,
            anchor_activations={100: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of=AS_OF,
        )

        self.assertEqual(1, len(matcher.calls))
        self.assertEqual(AS_OF, matcher.calls[0]["evaluation_as_of"])
        self.assertEqual([2], [item.target_episode_id for item in trace["hits"]])
        # V3 keeps the configured request budget instead of shrinking it to
        # the one-episode pre-treatment pool.  The contextual hit is only a
        # relevance-only route here (there is no existing slot mapping), so
        # both deliverable candidates remain visible and the requirement stays
        # uncovered rather than being fabricated as a fact.
        self.assertEqual({2, 99}, {item["id"] for item in selected})
        self.assertEqual(
            "v3_contribution_selector",
            trace["compatibility_projection"],
        )
        self.assertIn(
            "slot",
            trace["treatment_selector"]["missing_required_clauses"],
        )

    def test_prepared_early_shadow_uses_current_requirements_and_never_stops(self):
        bundle, slots = self._bundle_and_slots("slot")
        proposal = _proposal(71, 2, "slot", score=0.9, rank=1)
        matcher = _CapturingMatcher(
            {
                "hits": [],
                "pre_target_proposals": [proposal],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
            }
        )
        engine = self._engine(
            episodes=[_episode(2, 2)],
            sources=[_source(2)],
            target_vectors={2: [0, 1, 0]},
            matcher=matcher,
        )
        engine.config.retrieval.contextual_prepared_early_enabled = True
        engine.config.retrieval.contextual_prepared_early_candidate_pool_enabled = True
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )

        trace = engine._prepared_early_contextual_proposal(
            bundle=bundle,
            domain="test",
            endpoint_limit=1,
            anchor_activations={100: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of=AS_OF,
        )

        self.assertTrue(trace["proposal_generated"])
        self.assertTrue(trace["source_validated"])
        self.assertFalse(trace["current_requirement_supported"])
        self.assertFalse(trace["early_stop_allowed"])
        self.assertFalse(trace["candidate_pool_mutated"])
        self.assertTrue(trace["candidate_pool_requested"])
        self.assertEqual([2], trace["candidate_pool_validated_episode_ids"])
        self.assertEqual([], trace["candidate_pool_injected_episode_ids"])
        self.assertTrue(trace["ordinary_retrieval_continues"])
        self.assertEqual("candidate_only", trace["retention_reason"])
        self.assertEqual(
            "prepared_early_before_initial_graph_expansion_v1", trace["stage"]
        )
        self.assertEqual(["slot"], matcher.calls[0]["unresolved_slot_ids"])
        self.assertEqual(AS_OF, matcher.calls[0]["evaluation_as_of"])
        self.assertEqual([2], trace["target_gate"]["selected_endpoint_ids"])
        self.assertEqual(71, trace["pre_target_candidate_order"][0]["association_id"])

    def test_prepared_early_shadow_records_source_rejection_without_fallback(self):
        bundle, slots = self._bundle_and_slots("slot")
        proposal = _proposal(72, 2, "slot", score=0.9, rank=1)
        matcher = _CapturingMatcher(
            {
                "hits": [],
                "pre_target_proposals": [proposal],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
            }
        )
        rejected = _episode(2, 2)
        rejected["evidence_quotes_json"] = "[]"
        engine = self._engine(
            episodes=[rejected],
            sources=[_source(2)],
            target_vectors={2: [0, 1, 0]},
            matcher=matcher,
        )
        engine.config.retrieval.contextual_prepared_early_enabled = True
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )

        trace = engine._prepared_early_contextual_proposal(
            bundle=bundle,
            domain="test",
            endpoint_limit=1,
            anchor_activations={100: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of=AS_OF,
        )

        self.assertTrue(trace["proposal_generated"])
        self.assertFalse(trace["source_validated"])
        self.assertEqual("source_invalid", trace["retention_reason"])
        self.assertEqual(
            "target_evidence_quotes_missing",
            trace["target_gate"]["outcomes"][0]["reason"],
        )
        self.assertFalse(trace["candidate_pool_mutated"])

    def test_prepared_candidate_is_not_reclassified_as_base_route(self):
        bundle, slots = self._bundle_and_slots("slot")
        engine = self._engine(
            episodes=[_episode(2, 2)],
            sources=[_source(2)],
            target_vectors={2: [0, 1, 0]},
            matcher=_CapturingMatcher(),
        )
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )

        _selected, trace = engine._select_contextual_slots(
            episodes=[_episode(2, 2)],
            baseline_selected=[],
            rerank_trace={},
            reranked_episode_ids=[],
            bundle=bundle,
            domain="test",
            endpoint_limit=1,
            anchor_activations={100: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of=AS_OF,
            prepared_early_contextual_episode_ids=(2,),
        )

        self.assertEqual([], trace["base_endpoint_manifest"])
        self.assertEqual(
            [2], trace["prepared_early_contextual_endpoint_manifest"]
        )

    def test_prepared_candidate_pool_seeds_only_verified_nonbase_targets(self):
        hits, endpoint_ids = QueryEngine._prepared_early_candidate_pool_hits(
            [
                {"target_episode_id": 2, "combined_score": 0.25},
                {"target_episode_id": 2, "combined_score": 0.50},
                {"target_episode_id": 9, "combined_score": 0.40},
                {"target_episode_id": 3, "combined_score": -0.10},
                {"target_episode_id": 4, "combined_score": "not-a-score"},
            ],
            {9},
        )

        self.assertEqual((2,), endpoint_ids)
        self.assertEqual([("episode", 2, 0.25)], [
            (item.node_type, item.node_id, item.score) for item in hits
        ])


if __name__ == "__main__":
    unittest.main()
