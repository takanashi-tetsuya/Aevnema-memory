from __future__ import annotations

import unittest

from memory_demo.associations.plasticity import (
    ContextualPlasticity,
    derive_contextual_candidates,
    plan_learning_candidates,
    select_independent_learning_anchors,
)
from memory_demo.types import (
    ContextualRecallCandidate,
    ContextualUtilityObservation,
    LearningAnchor,
    PlasticityEvent,
    RecallLearningEvent,
    SourceFactRef,
)


def _fact(label: str) -> SourceFactRef:
    return SourceFactRef(
        source_revision_id=f"source-revision:{label}",
        record_span=(f"record:{label}",),
        span_hash=f"span:{label}",
        raw_span_hash=f"raw-span:{label}",
        source_key=f"private/{label}",
    )


def _anchor(
    episode_id: int,
    *,
    activation: float = 1.0,
    lane: str = "base",
    independent: bool = True,
) -> LearningAnchor:
    return LearningAnchor(
        anchor_type="episode",
        anchor_id=episode_id,
        contribution_id=f"base-contribution:{episode_id}",
        source_facts=(_fact(f"anchor-{episode_id}"),),
        vector_ref=f"vector:anchor:{episode_id}",
        provenance_refs=(f"provenance:anchor:{episode_id}",),
        activation=activation,
        lane=lane,
        independent=independent,
    )


def _event(
    *,
    initial_candidates: tuple[int, ...],
    initial_delivered: tuple[int, ...],
    final_selected: tuple[int, ...] = (1, 2),
    final_delivered: tuple[int, ...] = (1, 2),
    contextual: tuple[int, ...] = (),
    costly_stages: tuple[str, ...] = (),
) -> RecallLearningEvent:
    return RecallLearningEvent(
        request_id="request-01",
        request_hash="request-hash-01",
        domain="knowledge",
        target_episode_ids=(2,),
        initial_candidate_episode_ids=initial_candidates,
        initial_delivered_episode_ids=initial_delivered,
        final_selected_episode_ids=final_selected,
        final_delivered_episode_ids=final_delivered,
        contextual_expansion_episode_ids=contextual,
        slot_id="slot:need",
        context_query_id="query:whole",
        need_query_id="query:need",
        context_vector_ref="vector:whole",
        need_vector_ref="vector:need",
        source_facts=(_fact("target-2"),),
        verification_refs=("verification:target-2",),
        verification_status="verified",
        target_provenance_refs=("provenance:target-2",),
        costly_stage_refs=costly_stages,
    )


class LearningCandidateV3Tests(unittest.TestCase):
    def test_base_present_repaired_target_is_a_valid_present_dropped_candidate(self) -> None:
        # Episode 2 was already a base candidate, but its first selection did
        # not deliver it.  Its presence must not erase the repair provenance.
        event = _event(
            initial_candidates=(1, 2),
            initial_delivered=(1,),
        )
        plan = plan_learning_candidates(event, [_anchor(1)])
        self.assertEqual(1, len(plan.candidates))
        candidate = plan.candidates[0]
        self.assertEqual("candidate_present_dropped", candidate.reason)
        self.assertEqual(2, candidate.target_episode_id)
        self.assertEqual(1, candidate.anchor_id)
        self.assertFalse(candidate.creation_is_quality_observation)
        self.assertEqual(
            {"source-revision:anchor-1", "source-revision:target-2"},
            {fact.source_revision_id for fact in candidate.source_facts},
        )

        # The original public helper remains usable when it receives the
        # missing initial-delivery fact needed to prove this reason.
        legacy = derive_contextual_candidates(
            base_episode_ids=[1, 2],
            selected_episode_ids=[1, 2],
            target_episode_ids=[2],
            context_query_id="query:whole",
            need_query_id="query:need",
            anchor_id=1,
            initial_delivered_episode_ids=[1],
        )
        self.assertEqual(["candidate_present_dropped"], [item.reason for item in legacy])

    def test_first_correct_but_costly_round_creates_reuse_candidate(self) -> None:
        event = _event(
            initial_candidates=(1, 2),
            initial_delivered=(1, 2),
            costly_stages=("followup:query-3", "rerank:cross-encoder"),
        )
        plan = plan_learning_candidates(event, [_anchor(1)])
        self.assertEqual(1, len(plan.candidates))
        candidate = plan.candidates[0]
        self.assertEqual("costly_success_reuse", candidate.reason)
        self.assertEqual(
            ("followup:query-3", "rerank:cross-encoder"),
            candidate.costly_stage_refs,
        )

    def test_same_round_contextual_target_cannot_self_confirm_a_candidate(self) -> None:
        event = _event(
            initial_candidates=(1,),
            initial_delivered=(1,),
            contextual=(2,),
        )
        plan = plan_learning_candidates(event, [_anchor(1)])
        self.assertEqual((), plan.candidates)
        self.assertIn(
            "target_returned_by_contextual_expansion",
            [item.code for item in plan.rejections],
        )

    def test_anchor_selection_and_pairing_are_bounded_and_deterministic(self) -> None:
        anchors = [
            _anchor(8, activation=8.0, lane="contextual"),
            _anchor(7, activation=7.0, independent=False),
            _anchor(5, activation=5.0),
            _anchor(4, activation=4.0),
            _anchor(3, activation=3.0),
            _anchor(2, activation=2.0),
            _anchor(1, activation=1.0),
        ]
        selected = select_independent_learning_anchors(
            list(reversed(anchors)),
            max_anchors=99,
        )
        self.assertEqual([5, 4, 3, 2], [item.anchor_id for item in selected])

        event = _event(initial_candidates=(1,), initial_delivered=(1,))
        forward = plan_learning_candidates(event, anchors, max_anchors=99, max_candidates=99)
        backward = plan_learning_candidates(
            event, list(reversed(anchors)), max_anchors=99, max_candidates=99
        )
        self.assertEqual(4, len(forward.selected_anchors))
        self.assertEqual(4, len(forward.candidates))
        self.assertEqual(
            [item.candidate_id for item in forward.candidates],
            [item.candidate_id for item in backward.candidates],
        )

    def test_creation_round_observation_does_not_count_as_future_success(self) -> None:
        class Repository:
            def __init__(self) -> None:
                self.calls: list[tuple[ContextualUtilityObservation, ...]] = []

            def record_utility(self, observations):
                self.calls.append(tuple(observations))
                return {"updated": len(observations)}

        repository = Repository()
        plasticity = ContextualPlasticity(repository)
        legacy_candidate = ContextualRecallCandidate(
            anchor_type="episode",
            anchor_id=1,
            target_episode_id=2,
            context_query_id="query:whole",
            need_query_id="query:need",
        )
        creation = PlasticityEvent(
            request_hash="creation-request",
            domain="knowledge",
            candidates=(legacy_candidate,),
            observations=(
                ContextualUtilityObservation(
                    association_id=9,
                    query_hash="creation-request",
                    outcome="sufficient",
                ),
            ),
        )
        receipt = plasticity.apply_event(creation)
        self.assertEqual({"updated": 0}, receipt["utility"])
        self.assertTrue(receipt["pending_creation"])
        self.assertEqual(1, receipt["creation_observations_ignored"])
        self.assertEqual([], repository.calls)

        # A distinct later request remains a future observation and is the
        # only one sent to the repository's utility ledger.
        later = PlasticityEvent(
            request_hash="later-request",
            domain="knowledge",
            observations=(
                ContextualUtilityObservation(
                    association_id=9,
                    query_hash="later-request",
                    outcome="sufficient",
                ),
            ),
        )
        later_receipt = plasticity.apply_event(later)
        self.assertEqual({"updated": 1}, later_receipt["utility"])
        self.assertEqual(1, len(repository.calls))


if __name__ == "__main__":
    unittest.main()
