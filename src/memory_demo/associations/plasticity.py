from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from memory_demo.types import (
    ContextualRecallCandidate,
    ContextualUtilityObservation,
)


def request_hash(question: str) -> str:
    return hashlib.sha256(str(question).strip().encode("utf-8")).hexdigest()


def derive_contextual_candidates(
    *,
    base_episode_ids: Iterable[int],
    selected_episode_ids: Iterable[int],
    target_episode_ids: Iterable[int],
    context_query_id: str,
    need_query_id: str,
    slot_id: str = "",
    source_request_hash: str = "",
    anchor_type: str = "episode",
    anchor_id: int = 0,
    max_candidates: int = 1,
    contextual_episode_ids: Iterable[int] = (),
) -> list[ContextualRecallCandidate]:
    """Derive evidence-repair candidates from a local retrieval trace only."""
    base = {int(value) for value in base_episode_ids}
    selected = {int(value) for value in selected_episode_ids}
    contextual = {int(value) for value in contextual_episode_ids}
    result: list[ContextualRecallCandidate] = []
    for target in dict.fromkeys(int(value) for value in target_episode_ids):
        if target in base or target in contextual or target not in selected or target <= 0:
            continue
        result.append(
            ContextualRecallCandidate(
                anchor_type=anchor_type,
                anchor_id=int(anchor_id),
                target_episode_id=target,
                context_query_id=str(context_query_id),
                need_query_id=str(need_query_id),
                slot_id=str(slot_id),
                source_request_hash=str(source_request_hash),
            )
        )
        if len(result) >= max(0, int(max_candidates)):
            break
    return result


def classify_treatment_masked(
    association_id: int,
    query_hash: str,
    treatment_slots: Iterable[str],
    masked_slots: Iterable[str],
    treatment_episode_ids: Iterable[int] = (),
    masked_episode_ids: Iterable[int] = (),
) -> ContextualUtilityObservation:
    treatment = {str(value) for value in treatment_slots}
    masked = {str(value) for value in masked_slots}
    delta = treatment - masked
    lost = masked - treatment
    if lost and not delta:
        outcome = "harmful"
    elif delta:
        outcome = "sufficient"
    elif treatment == masked:
        outcome = "no_op"
    else:
        outcome = "redundant"
    return ContextualUtilityObservation(
        association_id=int(association_id),
        query_hash=str(query_hash),
        outcome=outcome,
        delta_slots=len(delta),
        treatment_episode_ids=tuple(int(value) for value in treatment_episode_ids),
        masked_episode_ids=tuple(int(value) for value in masked_episode_ids),
    )


@dataclass(slots=True)
class ContextualPlasticity:
    repository: object
    probation_ttl: int = 2_592_000
    initial_utility: float = 0.20

    def create(self, candidate: ContextualRecallCandidate, context_cue_id: int, need_cue_id: int) -> int:
        return int(self.repository.create_contextual(
            candidate,
            context_cue_id=context_cue_id,
            need_cue_id=need_cue_id,
            utility_weight=self.initial_utility,
            probation_ttl=self.probation_ttl,
        ))

    def update(self, observations: Iterable[ContextualUtilityObservation]) -> dict[str, int]:
        return dict(self.repository.record_utility(list(observations)))

    def apply_event(self, event) -> dict[str, object]:
        """Apply only post-answer observations and return a compact receipt."""
        if not event.answer_guard_passed:
            return {"created": [], "utility": {"updated": 0}, "skipped": "guard"}
        utility = self.update(event.observations)
        return {
            "created": [],
            "utility": utility,
            "candidate_count": len(event.candidates),
        }
