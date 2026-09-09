from __future__ import annotations

import unittest

from memory_demo.retrieval.coverage import (
    contribution_contextual_attribution,
    mask_contribution_aggregates,
    run_contribution_counterfactual,
    select_contribution_evidence,
)
from memory_demo.types import (
    CandidateContribution,
    ClauseRequirement,
    ClauseSupport,
    EvidenceSelectionBudget,
    SourceFactRef,
)


def _fact(name: str) -> SourceFactRef:
    return SourceFactRef(
        source_revision_id="revision-1",
        record_span=(f"record-{name}",),
        span_hash=f"span-{name}",
        raw_span_hash=f"raw-{name}",
        source_key="story/example.json",
    )


def _contribution(
    contribution_id: str,
    episode_id: int,
    *,
    slot_id: str,
    clause_id: str | None = None,
    edge_id: int | None = None,
    relevance: float = 0.0,
) -> CandidateContribution:
    fact = _fact(contribution_id)
    return CandidateContribution(
        contribution_id=contribution_id,
        episode_id=episode_id,
        slot_id=slot_id,
        lane="contextual" if edge_id is not None else "dense",
        edge_id=edge_id,
        rank_features={"current_relevance_estimate": relevance},
        source_facts=(fact,),
        clause_supports=(
            ClauseSupport(
                slot_id=slot_id,
                clause_id=clause_id or slot_id,
                source_fact=fact,
                support_mode="alternative",
                verification_status="verified",
            ),
        ),
    )


class ContributionCounterfactualV3Tests(unittest.TestCase):
    def test_edge_mask_keeps_independent_base_contribution_for_same_episode(self):
        base = _contribution("base-a", 7, slot_id="a")
        edge = _contribution("edge-9-b", 7, slot_id="b", edge_id=9)
        requirements = [ClauseRequirement("a"), ClauseRequirement("b")]
        budget = EvidenceSelectionBudget(episode_limit=1)

        survivors = mask_contribution_aggregates([base, edge], edge_ids=(9,))
        masked = select_contribution_evidence(survivors, requirements, budget)
        delta = run_contribution_counterfactual(
            [base, edge], requirements, budget, edge_ids=(9,)
        )

        self.assertEqual((7,), tuple(item.episode_id for item in survivors))
        self.assertEqual(("base-a",), survivors[0].contribution_ids)
        self.assertEqual(("base-a", "edge-9-b"), tuple(
            contribution.contribution_id for contribution in (base, edge)
        ))
        self.assertEqual((7,), masked.selected_episode_ids)
        self.assertEqual(frozenset({"a"}), masked.covered_required_clauses)
        self.assertEqual(frozenset({"a", "b"}), delta.treatment.covered_required_clauses)
        self.assertEqual(frozenset({"a"}), delta.masked.covered_required_clauses)
        self.assertEqual(frozenset({"b"}), delta.gained_required_clauses)

    def test_exact_masks_keep_other_edges_to_the_same_episode(self):
        base = _contribution("base-a", 17, slot_id="a")
        first_edge = _contribution("edge-41-b", 17, slot_id="b", edge_id=41)
        second_edge = _contribution("edge-42-c", 17, slot_id="c", edge_id=42)
        requirements = [
            ClauseRequirement("a"),
            ClauseRequirement("b"),
            ClauseRequirement("c"),
        ]

        survivors = mask_contribution_aggregates(
            [base, first_edge, second_edge],
            contribution_ids=("edge-41-b",),
        )
        edge_masked = mask_contribution_aggregates(
            [base, first_edge, second_edge], edge_ids=(41,)
        )
        selection = select_contribution_evidence(
            edge_masked, requirements, EvidenceSelectionBudget(episode_limit=1)
        )

        self.assertEqual(("base-a", "edge-42-c"), survivors[0].contribution_ids)
        self.assertEqual(("base-a", "edge-42-c"), edge_masked[0].contribution_ids)
        self.assertEqual(frozenset({"a", "c"}), selection.covered_required_clauses)

    def test_single_edge_and_leave_one_out_are_deterministic_for_two_slots(self):
        edge_a = _contribution("edge-11-a", 1, slot_id="a", edge_id=11)
        edge_b = _contribution("edge-22-b", 2, slot_id="b", edge_id=22)
        attribution = contribution_contextual_attribution(
            [edge_b, edge_a],
            [ClauseRequirement("a"), ClauseRequirement("b")],
            EvidenceSelectionBudget(episode_limit=2),
        )

        self.assertEqual((1, 2), attribution.treatment.selected_episode_ids)
        self.assertEqual((), attribution.masked.selected_episode_ids)
        self.assertEqual((11, 22), attribution.edge_ids)
        first, second = attribution.edges
        self.assertEqual(("a",), tuple(sorted(first.single_edge.gained_required_clauses)))
        self.assertEqual(("a",), tuple(sorted(first.leave_one_out.gained_required_clauses)))
        self.assertTrue(first.sufficient)
        self.assertTrue(first.necessary)
        self.assertEqual(("b",), tuple(sorted(second.single_edge.gained_required_clauses)))
        self.assertEqual(("b",), tuple(sorted(second.leave_one_out.gained_required_clauses)))
        self.assertTrue(second.sufficient)
        self.assertTrue(second.necessary)

    def test_alternative_edge_is_not_necessary_when_loo_retains_coverage(self):
        selected_edge = _contribution(
            "edge-31-variant-a",
            1,
            slot_id="answer",
            clause_id="variant-a",
            edge_id=31,
            relevance=1.0,
        )
        alternative_edge = _contribution(
            "edge-32-variant-b",
            2,
            slot_id="answer",
            clause_id="variant-b",
            edge_id=32,
            relevance=0.1,
        )
        attribution = contribution_contextual_attribution(
            [selected_edge, alternative_edge],
            [ClauseRequirement("answer", ("variant-a", "variant-b"), "alternative")],
            EvidenceSelectionBudget(episode_limit=1),
        )
        edge_31 = next(item for item in attribution.edges if item.edge_id == 31)

        self.assertTrue(edge_31.selected_in_treatment)
        self.assertTrue(edge_31.sufficient)
        self.assertFalse(edge_31.necessary)
        self.assertEqual(frozenset(), edge_31.leave_one_out.gained_required_clauses)
        self.assertFalse(edge_31.leave_one_out.harm)

    def test_gained_and_lost_required_coverage_is_harmful_not_pure_success(self):
        treatment_edge = _contribution(
            "edge-90-a", 1, slot_id="a", edge_id=90, relevance=1.0
        )
        baseline = _contribution("base-b", 2, slot_id="b", relevance=0.0)
        delta = run_contribution_counterfactual(
            [baseline, treatment_edge],
            [ClauseRequirement("a"), ClauseRequirement("b")],
            EvidenceSelectionBudget(episode_limit=1),
            edge_ids=(90,),
        )

        self.assertEqual(frozenset({"a"}), delta.gained_required_clauses)
        self.assertEqual(frozenset({"b"}), delta.lost_required_clauses)
        self.assertTrue(delta.mixed)
        self.assertTrue(delta.harm)
        self.assertEqual("harmful", delta.classification)
        self.assertFalse(delta.pure_success)
        self.assertFalse(delta.as_dict()["pure_success"])

    def test_attribution_fingerprints_are_order_independent_and_opaque(self):
        first = _contribution("edge-71-a", 1, slot_id="a", edge_id=71)
        second = _contribution("edge-72-b", 2, slot_id="b", edge_id=72)
        requirements = [ClauseRequirement("a"), ClauseRequirement("b")]
        budget = EvidenceSelectionBudget(episode_limit=2, source_fact_limit=2)

        forward = contribution_contextual_attribution(
            [first, second], requirements, budget
        )
        reversed_input = contribution_contextual_attribution(
            [second, first], requirements, budget
        )

        self.assertEqual(
            forward.candidate_universe_fingerprint,
            reversed_input.candidate_universe_fingerprint,
        )
        self.assertEqual(
            forward.requirements_fingerprint,
            reversed_input.requirements_fingerprint,
        )
        self.assertEqual(forward.budget_fingerprint, reversed_input.budget_fingerprint)
        self.assertEqual(forward.input_fingerprint, reversed_input.input_fingerprint)
        self.assertTrue(forward.input_fingerprint.startswith("sha256:"))
        self.assertNotIn("story/example.json", forward.input_fingerprint)


if __name__ == "__main__":
    unittest.main()
