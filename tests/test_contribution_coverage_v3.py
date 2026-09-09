from __future__ import annotations

import unittest

from memory_demo.retrieval.coverage import (
    aggregate_contributions,
    compare_greedy_to_oracle,
    contribution_coverage_state,
    exhaustive_contribution_oracle,
    select_contribution_evidence,
)
from memory_demo.types import (
    CandidateContribution,
    ClauseRequirement,
    ClauseSupport,
    EvidenceSelectionBudget,
    SourceFactRef,
)


def _fact(
    index: str,
    *,
    source_key: str = "main/long-story.json",
    revision: str = "source-revision-a",
) -> SourceFactRef:
    return SourceFactRef(
        source_revision_id=revision,
        record_span=(f"record-{index}",),
        span_hash=f"span-{index}",
        raw_span_hash=f"raw-{index}",
        source_key=source_key,
    )


def _contribution(
    contribution_id: str,
    episode_id: int,
    *,
    slot_id: str,
    clause_id: str,
    fact: SourceFactRef,
    support_mode: str = "alternative",
    verification_status: str = "verified",
    lane: str = "dense",
    edge_id: int | None = None,
    relevance: float = 0.0,
) -> CandidateContribution:
    return CandidateContribution(
        contribution_id=contribution_id,
        episode_id=episode_id,
        slot_id=slot_id,
        lane=lane,
        edge_id=edge_id,
        rank_features={"current_relevance_estimate": relevance},
        source_facts=(fact,),
        clause_supports=(
            ClauseSupport(
                slot_id=slot_id,
                clause_id=clause_id,
                source_fact=fact,
                support_mode=support_mode,  # type: ignore[arg-type]
                verification_status=verification_status,  # type: ignore[arg-type]
            ),
        ),
    )


class ContributionCoverageV3Tests(unittest.TestCase):
    def test_source_fact_identity_uses_revision_span_and_raw_hash_not_source_key(self):
        fact = _fact("1", source_key="main/a.json")
        renamed_path = _fact("1", source_key="archive/renamed.json")
        different_span = _fact("2", source_key="main/a.json")
        revised = _fact("1", source_key="main/a.json", revision="source-revision-b")

        self.assertEqual(fact, renamed_path)
        self.assertEqual(fact.identity_key, renamed_path.identity_key)
        self.assertNotEqual(fact, different_span)
        self.assertNotEqual(fact, revised)
        self.assertNotIn("source_key", fact.fact_id)

    def test_aggregate_preserves_base_and_contextual_routes_to_one_episode(self):
        fact = _fact("1")
        base = _contribution(
            "dense-1", 7, slot_id="who", clause_id="who", fact=fact, relevance=0.4
        )
        contextual = _contribution(
            "edge-9-slot-who",
            7,
            slot_id="who",
            clause_id="who",
            fact=fact,
            lane="contextual",
            edge_id=9,
            relevance=0.8,
        )

        aggregate = aggregate_contributions([contextual, base])[0]
        self.assertEqual(("dense-1", "edge-9-slot-who"), aggregate.contribution_ids)
        self.assertEqual(("dense-1",), aggregate.base_contribution_ids)
        self.assertEqual(("edge-9-slot-who",), aggregate.contextual_contribution_ids)
        self.assertEqual(1, len(aggregate.source_facts))

    def test_relevance_only_mapping_never_satisfies_required_clause(self):
        weak_fact = _fact("weak")
        verified_fact = _fact("verified")
        relevance_only = _contribution(
            "relevance-only",
            1,
            slot_id="fact",
            clause_id="fact",
            fact=weak_fact,
            verification_status="relevance_only",
            relevance=0.99,
        )
        verified = _contribution(
            "verified",
            2,
            slot_id="fact",
            clause_id="fact",
            fact=verified_fact,
            relevance=0.01,
        )
        requirements = [ClauseRequirement("fact")]

        raw_state = contribution_coverage_state([relevance_only], requirements)
        selection = select_contribution_evidence(
            [relevance_only, verified],
            requirements,
            EvidenceSelectionBudget(episode_limit=1),
        )

        self.assertEqual(frozenset(), raw_state.covered_required_clauses)
        self.assertEqual((2,), selection.selected_episode_ids)
        self.assertEqual(frozenset({"fact"}), selection.covered_required_clauses)

    def test_complete_required_coverage_stops_before_relevance_tail(self):
        required = _contribution(
            "required",
            1,
            slot_id="fact",
            clause_id="fact",
            fact=_fact("required"),
            relevance=0.01,
        )
        relevance_tail = _contribution(
            "tail",
            2,
            slot_id="background",
            clause_id="background",
            fact=_fact("tail"),
            relevance=100.0,
        )

        selection = select_contribution_evidence(
            [required, relevance_tail],
            [ClauseRequirement("fact")],
            EvidenceSelectionBudget(episode_limit=2),
        )

        self.assertEqual((1,), selection.selected_episode_ids)
        self.assertEqual("all_required_clauses_covered", selection.stop_reason)

    def test_joint_prerequisites_have_progress_before_completed_clause_gain(self):
        first_fact = _fact("joint-a")
        second_fact = _fact("joint-b")
        distractor_fact = _fact("background")
        first = _contribution(
            "joint-a",
            1,
            slot_id="joint",
            clause_id="a",
            fact=first_fact,
            support_mode="joint",
            relevance=0.01,
        )
        second = _contribution(
            "joint-b",
            2,
            slot_id="joint",
            clause_id="b",
            fact=second_fact,
            support_mode="joint",
            relevance=0.01,
        )
        distractor = _contribution(
            "background",
            3,
            slot_id="background",
            clause_id="background",
            fact=distractor_fact,
            relevance=100.0,
        )
        requirements = [ClauseRequirement("joint", ("a", "b"), "joint")]

        selection = select_contribution_evidence(
            [distractor, first, second],
            requirements,
            EvidenceSelectionBudget(episode_limit=2),
        )

        self.assertEqual((1, 2), selection.selected_episode_ids)
        self.assertEqual(frozenset({"joint"}), selection.covered_required_clauses)
        self.assertFalse(selection.delivery_loss)
        self.assertEqual((), selection.decisions[0].newly_covered_required_clauses)
        self.assertEqual(("joint",), selection.decisions[0].joint_incomplete_clauses)

    def test_explicit_request_budget_can_deliver_more_than_any_initial_lane_pool(self):
        requirements = [
            ClauseRequirement("a"),
            ClauseRequirement("b"),
            ClauseRequirement("c"),
        ]
        candidates = [
            _contribution("base-a", 1, slot_id="a", clause_id="a", fact=_fact("a")),
            _contribution("context-b", 2, slot_id="b", clause_id="b", fact=_fact("b"), lane="contextual", edge_id=2),
            _contribution("source-c", 3, slot_id="c", clause_id="c", fact=_fact("c"), lane="source", edge_id=None),
        ]

        selection = select_contribution_evidence(
            candidates,
            requirements,
            EvidenceSelectionBudget(episode_limit=3),
        )

        self.assertEqual((1, 2, 3), selection.selected_episode_ids)
        self.assertEqual(frozenset({"a", "b", "c"}), selection.covered_required_clauses)
        self.assertEqual(3, selection.actual_delivery_episode_count)
        self.assertFalse(selection.budget_exhausted)

    def test_same_source_key_independent_spans_are_not_redundancy_collapsed(self):
        first_fact = _fact("first", source_key="main/one-long-file.json")
        second_fact = _fact("second", source_key="main/one-long-file.json")
        selection = select_contribution_evidence(
            [
                _contribution("first", 1, slot_id="first", clause_id="first", fact=first_fact),
                _contribution("second", 2, slot_id="second", clause_id="second", fact=second_fact),
            ],
            [ClauseRequirement("first"), ClauseRequirement("second")],
            EvidenceSelectionBudget(episode_limit=2, source_fact_limit=2),
        )

        self.assertEqual((1, 2), selection.selected_episode_ids)
        self.assertEqual(2, selection.actual_delivery_source_fact_count)
        self.assertFalse(selection.source_fact_budget_exhausted)

    def test_alternative_clause_is_complete_when_any_verified_variant_is_selected(self):
        alternate = _contribution(
            "alternate-b",
            2,
            slot_id="alternative",
            clause_id="b",
            fact=_fact("alternate-b"),
            relevance=0.1,
        )
        selection = select_contribution_evidence(
            [alternate],
            [ClauseRequirement("alternative", ("a", "b"), "alternative")],
            EvidenceSelectionBudget(episode_limit=1),
        )

        self.assertEqual(frozenset({"alternative"}), selection.covered_required_clauses)
        self.assertEqual(frozenset(), selection.missing_required_clauses)

    def test_source_fact_budget_reports_a_real_delivery_block(self):
        fact_a = _fact("fact-a")
        fact_b = _fact("fact-b")
        supplemental_b = _fact("fact-b-supplement")
        candidate_a = _contribution(
            "a", 1, slot_id="a", clause_id="a", fact=fact_a
        )
        candidate_b = CandidateContribution(
            contribution_id="b",
            episode_id=2,
            slot_id="b",
            lane="source",
            source_facts=(fact_b, supplemental_b),
            clause_supports=(ClauseSupport("b", "b", fact_b, "alternative", "verified"),),
        )
        selection = select_contribution_evidence(
            [candidate_a, candidate_b],
            [ClauseRequirement("a"), ClauseRequirement("b")],
            EvidenceSelectionBudget(episode_limit=2, source_fact_limit=2),
        )

        self.assertEqual((1,), selection.selected_episode_ids)
        self.assertTrue(selection.source_fact_budget_exhausted)
        self.assertTrue(selection.delivery_loss)
        self.assertEqual(frozenset({"b"}), selection.delivery_loss_clauses)

    def test_budget_loss_is_explicit_when_verified_candidate_is_available_but_not_delivered(self):
        selection = select_contribution_evidence(
            [
                _contribution("a", 1, slot_id="a", clause_id="a", fact=_fact("a")),
                _contribution("b", 2, slot_id="b", clause_id="b", fact=_fact("b")),
            ],
            [ClauseRequirement("a"), ClauseRequirement("b")],
            EvidenceSelectionBudget(episode_limit=1),
        )

        self.assertTrue(selection.budget_exhausted)
        self.assertTrue(selection.episode_budget_exhausted)
        self.assertTrue(selection.delivery_loss)
        self.assertEqual(frozenset({"b"}), selection.delivery_loss_clauses)
        self.assertEqual(1, selection.actual_delivery_episode_count)

    def test_small_exhaustive_oracle_detects_greedy_joint_loss(self):
        fact_a = _fact("a")
        fact_b = _fact("b")
        fact_c = _fact("c")
        a = CandidateContribution(
            contribution_id="a-components",
            episode_id=1,
            slot_id="joint-one",
            lane="dense",
            source_facts=(fact_a,),
            clause_supports=(
                ClauseSupport("joint-one", "a", fact_a, "joint", "verified"),
                ClauseSupport("joint-two", "a", fact_a, "joint", "verified"),
            ),
        )
        b = CandidateContribution(
            contribution_id="b-components",
            episode_id=2,
            slot_id="joint-one",
            lane="dense",
            source_facts=(fact_b,),
            clause_supports=(
                ClauseSupport("joint-one", "b", fact_b, "joint", "verified"),
                ClauseSupport("joint-two", "b", fact_b, "joint", "verified"),
            ),
        )
        c = _contribution(
            "standalone",
            3,
            slot_id="standalone",
            clause_id="standalone",
            fact=fact_c,
        )
        requirements = [
            ClauseRequirement("joint-one", ("a", "b"), "joint"),
            ClauseRequirement("joint-two", ("a", "b"), "joint"),
            ClauseRequirement("standalone"),
        ]
        budget = EvidenceSelectionBudget(episode_limit=2)

        comparison = compare_greedy_to_oracle([a, b, c], requirements, budget)
        oracle = exhaustive_contribution_oracle([a, b, c], requirements, budget)

        self.assertTrue(comparison["greedy_loss"])
        self.assertEqual((1, 2), oracle.selected_episode_ids)
        self.assertEqual(
            frozenset({"joint-one", "joint-two"}),
            oracle.covered_required_clauses,
        )


if __name__ == "__main__":
    unittest.main()
