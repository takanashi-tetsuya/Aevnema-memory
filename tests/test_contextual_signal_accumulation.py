from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import unittest

from memory_demo.retrieval.coverage import (
    mask_contribution_aggregates,
    run_contribution_counterfactual,
    select_contribution_evidence,
)
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.types import (
    CandidateAggregate,
    CandidateContribution,
    ClauseRequirement,
    ClauseSupport,
    ContextualSlotHit,
    EvidenceSelectionBudget,
    EvidenceSlot,
    SourceFactRef,
)


_FACT = SourceFactRef(
    source_revision_id="target-revision",
    record_span=("target-record",),
    span_hash="target-span",
    raw_span_hash="target-raw-span",
)


def _contextual(
    edge_id: int,
    priority: float,
    source_id: float | None = None,
    *,
    query_ref: str = "query",
) -> CandidateContribution:
    return CandidateContribution(
        contribution_id=f"edge-{edge_id}-{query_ref}",
        episode_id=9,
        slot_id="answer",
        lane="contextual",
        edge_id=edge_id,
        query_ref=query_ref,
        rank_features={
            "contextual_priority_score": priority,
            "contextual_anchor_source_id": source_id,
            "contextual_anchor_episode_id": edge_id,
            "contextual_context_cue_id": edge_id,
            "contextual_need_cue_id": edge_id,
        },
        source_facts=(_FACT,),
        clause_supports=(ClauseSupport(
            slot_id="answer",
            clause_id="answer",
            source_fact=_FACT,
            verification_status="relevance_only",
        ),),
    )


def _base(priority: float) -> CandidateContribution:
    return CandidateContribution(
        contribution_id="base",
        episode_id=9,
        slot_id="__base_rank__",
        lane="dense",
        rank_features={"fusion_rank_score": priority},
        source_facts=(_FACT,),
    )


class ContextualSignalAccumulationTests(unittest.TestCase):
    def test_two_weak_independent_sources_raise_inspection_priority(self):
        first = _contextual(1, 0.2, 10)
        second = _contextual(2, 0.3, 20)
        aggregate = CandidateAggregate(9, (first, second))

        self.assertAlmostEqual(0.44, aggregate.relevance_score)
        self.assertGreater(aggregate.relevance_score, 0.4)
        self.assertLess(CandidateAggregate(9, (first,)).relevance_score, 0.4)
        self.assertLess(CandidateAggregate(9, (second,)).relevance_score, 0.4)
        # Both routes remain independently maskable, while the target fact is
        # still counted once for source delivery and semantic coverage.
        self.assertEqual(2, len(aggregate.contributions))
        self.assertEqual((_FACT,), aggregate.source_facts)

    def test_same_source_episodes_edges_cues_and_rephrasing_cannot_multiply(self):
        original = _contextual(1, 0.2, 10)
        rewritten = _contextual(2, 0.3, 10, query_ref="rephrased")
        repeated = replace(rewritten, contribution_id="repeated-contribution")

        aggregate = CandidateAggregate(9, (original, rewritten, repeated, original))

        self.assertAlmostEqual(0.3, aggregate.relevance_score)
        self.assertEqual(3, len(aggregate.contributions))
        self.assertEqual((_FACT,), aggregate.source_facts)

    def test_identical_cues_do_not_prevent_distinct_sources_from_converging(self):
        first = _contextual(1, 0.2, 10)
        second_features = _contextual(2, 0.2, 20).feature_map
        second_features.update({
            "contextual_context_cue_id": 1,
            "contextual_need_cue_id": 1,
        })
        second = replace(_contextual(2, 0.2, 20), rank_features=second_features)

        self.assertAlmostEqual(0.36, CandidateAggregate(9, (first, second)).relevance_score)

    def test_legacy_unknown_and_invalid_sources_keep_conservative_max(self):
        known = _contextual(1, 0.2, 10)
        for source_id in (None, 0, -1, 1.5, 2**53):
            with self.subTest(source_id=source_id):
                unknown = _contextual(2, 0.3, source_id)
                self.assertAlmostEqual(
                    0.3, CandidateAggregate(9, (known, unknown)).relevance_score
                )
        self.assertAlmostEqual(0.3, CandidateAggregate(9, (
            _contextual(1, 0.2), _contextual(2, 0.3),
        )).relevance_score)
        self.assertAlmostEqual(0.44, CandidateAggregate(9, (
            known, _contextual(2, 0.3, 20), _contextual(3, 0.4),
        )).relevance_score)

    def test_mask_recomputes_remaining_sources_and_restores_exact_base(self):
        contributions = (
            _base(0.1),
            _contextual(1, 0.2, 10),
            _contextual(2, 0.4, 10),
            _contextual(3, 0.3, 20),
        )
        original = CandidateAggregate(9, contributions)
        weaker_same_source = mask_contribution_aggregates([original], edge_ids=(2,))[0]
        one_source = mask_contribution_aggregates([original], edge_ids=(1, 2))[0]
        base_only = mask_contribution_aggregates([original], edge_ids=(1, 2, 3))[0]

        self.assertAlmostEqual(0.622, original.relevance_score)
        self.assertAlmostEqual(0.496, weaker_same_source.relevance_score)
        self.assertAlmostEqual(0.37, one_source.relevance_score)
        self.assertEqual(0.1, base_only.relevance_score)
        self.assertEqual(("base",), base_only.contribution_ids)
        self.assertEqual(4, len(original.contributions))
        self.assertEqual(original.relevance_score, CandidateAggregate(9, contributions).relevance_score)

    def test_union_is_order_independent_bounded_and_saturates(self):
        contributions = (_base(0.1), _contextual(1, 0.2, 10), _contextual(2, 0.3, 20))
        scores = {CandidateAggregate(9, ordering).relevance_score for ordering in permutations(contributions)}
        self.assertEqual(1, len(scores))
        self.assertAlmostEqual(0.496, scores.pop())
        self.assertEqual(1.0, CandidateAggregate(9, (
            _contextual(1, 4.0, 10), _contextual(2, -3.0, 20),
        )).relevance_score)
        self.assertEqual(0.0, CandidateAggregate(9, (_contextual(1, -3.0, 10),)).relevance_score)

    def test_priority_cannot_create_verified_coverage_or_counterfactual_gain(self):
        contributions = (_base(0.1), _contextual(1, 0.6, 10), _contextual(2, 0.6, 20))
        requirement = ClauseRequirement("answer")
        budget = EvidenceSelectionBudget(episode_limit=1)
        selected = select_contribution_evidence(contributions, [requirement], budget)
        delta = run_contribution_counterfactual(
            contributions, [requirement], budget, edge_ids=(1, 2)
        )

        self.assertEqual(frozenset(), selected.covered_required_clauses)
        self.assertEqual(frozenset({"answer"}), selected.missing_required_clauses)
        self.assertEqual(frozenset(), delta.gained_required_clauses)
        self.assertFalse(delta.pure_success)

    def test_single_contextual_and_base_only_behavior_is_compatible(self):
        self.assertAlmostEqual(0.76, CandidateAggregate(9, (
            _base(0.4), _contextual(1, 0.6),
        )).relevance_score)
        self.assertEqual(1.4, CandidateAggregate(9, (_base(1.4),)).relevance_score)


class _Rows:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get_many(self, ids):
        self.calls.append(tuple(ids))
        return [row for row in self.rows if row["id"] in ids]


class ContextualSignalProducerTests(unittest.TestCase):
    def _build(self, source_ids):
        engine = QueryEngine.__new__(QueryEngine)
        engine.episodes = _Rows([
            {"id": 1, "source_id": 10}, {"id": 2, "source_id": 20},
            {"id": 3, "source_id": 10},
        ])
        engine.sources = _Rows([{"id": source_id} for source_id in source_ids])
        engine._v3_source_fact_closure = lambda ids, cache=None: ({}, {})
        contributions, _, _ = engine._v3_build_candidate_contributions(
            episodes=[{"id": 9}],
            slots=[EvidenceSlot("answer", "answer")],
            slot_support={},
            base_episode_ids=[],
            contextual_hits=[ContextualSlotHit(
                association_id=anchor,
                anchor_episode_id=anchor,
                target_episode_id=9,
                matched_slot_id="answer",
                matched_query_id=f"rewrite-{anchor}",
                target_support_score=0.2,
            ) for anchor in (1, 2, 3)],
        )
        return engine, CandidateAggregate(9, tuple(contributions))

    def test_engine_resolves_source_provenance_once_without_duplicate_episode_boost(self):
        engine, aggregate = self._build((10, 20))

        self.assertEqual([(1, 2, 3)], engine.episodes.calls)
        self.assertEqual([(10, 20)], engine.sources.calls)
        self.assertEqual([10, 10, 20], sorted(
            item.feature_map["contextual_anchor_source_id"] for item in aggregate.contributions
        ))
        self.assertAlmostEqual(0.36, aggregate.relevance_score)
        self.assertTrue(all(
            support.verification_status == "relevance_only"
            for support in aggregate.clause_supports
        ))

    def test_missing_anchor_source_cannot_claim_independence(self):
        _, aggregate = self._build((10,))

        self.assertAlmostEqual(0.2, aggregate.relevance_score)
        self.assertEqual(1, sum(
            item.feature_map["contextual_anchor_source_id"] is None
            for item in aggregate.contributions
        ))


if __name__ == "__main__":
    unittest.main()
