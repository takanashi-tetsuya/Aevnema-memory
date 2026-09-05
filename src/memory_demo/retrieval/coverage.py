from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from memory_demo.types import EvidenceSlot, SlotCandidate


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Compatibility input for the original v1 selector.

    New callers should use :class:`SlotCandidate`; accepting both avoids
    forcing every offline replay fixture to change at once.
    """

    episode_id: int
    supported_slots: frozenset[str] = frozenset()
    direct_relevance: float = 0.0
    contextual_cue_score: float = 0.0
    redundancy: float = 0.0
    source_key: str = ""
    source_quality: float = 0.0
    contextual: bool = False


@dataclass(frozen=True, slots=True)
class CoverageSelection:
    selected: tuple[EvidenceCandidate | SlotCandidate, ...]
    covered_slots: frozenset[str]
    missing_required: frozenset[str]


def _slot_ids(candidate: EvidenceCandidate | SlotCandidate) -> frozenset[str]:
    if isinstance(candidate, SlotCandidate):
        return candidate.slot_ids
    return candidate.supported_slots


def _direct_score(candidate: EvidenceCandidate | SlotCandidate) -> float:
    if isinstance(candidate, SlotCandidate):
        return float(candidate.direct_score) + float(candidate.specificity_score)
    return float(candidate.direct_relevance)


def _contextual_score(candidate: EvidenceCandidate | SlotCandidate) -> float:
    if isinstance(candidate, SlotCandidate):
        return float(candidate.contextual_score)
    return float(candidate.contextual_cue_score)


def _redundancy_group(candidate: EvidenceCandidate | SlotCandidate) -> str:
    if isinstance(candidate, SlotCandidate):
        return str(candidate.redundancy_group).strip()
    return str(candidate.source_key).strip()


def _source_quality(candidate: EvidenceCandidate | SlotCandidate) -> float:
    return float(getattr(candidate, "source_quality", 0.0))


def select_evidence(
    candidates: Iterable[EvidenceCandidate | SlotCandidate],
    slots: Iterable[EvidenceSlot],
    budget: int,
    *,
    required_gain: float = 10.0,
    optional_gain: float = 2.0,
    relevance_gain: float = 1.0,
    cue_gain: float = 0.5,
    redundancy_penalty: float = 0.5,
) -> CoverageSelection:
    """Greedy set-cover selector shared by base and contextual retrieval."""
    slot_list = list(slots)
    required = {item.slot_id for item in slot_list if item.required}
    remaining = list(candidates)
    selected: list[EvidenceCandidate | SlotCandidate] = []
    covered: set[str] = set()
    while remaining and len(selected) < max(0, int(budget)):
        def score(item: EvidenceCandidate | SlotCandidate) -> tuple[float, float, int]:
            slot_ids = set(_slot_ids(item))
            required_gain_value = len((slot_ids & required) - covered)
            optional_gain_value = len(slot_ids - required - covered)
            # A contextual edge is a residual repair hint, not a global rank
            # boost.  Once its slot is already covered it has no special
            # privilege over ordinary direct evidence.
            contextual_bonus = (
                cue_gain * _contextual_score(item)
                if slot_ids - covered
                else 0.0
            )
            redundancy = float(getattr(item, "redundancy", 0.0))
            group = _redundancy_group(item)
            if group and any(_redundancy_group(old) == group for old in selected):
                redundancy += 1.0
            value = (
                required_gain * required_gain_value
                + optional_gain * optional_gain_value
                + relevance_gain * _direct_score(item)
                + contextual_bonus
                + _source_quality(item)
                - redundancy_penalty * redundancy
            )
            return value, _direct_score(item), -int(item.episode_id)

        best = max(remaining, key=score)
        remaining.remove(best)
        if score(best)[0] <= 0:
            break
        selected.append(best)
        covered.update(_slot_ids(best))
    return CoverageSelection(
        selected=tuple(selected),
        covered_slots=frozenset(covered),
        missing_required=frozenset(required - covered),
    )


def treatment_masked_delta(
    treatment: CoverageSelection,
    masked: CoverageSelection,
) -> dict[str, object]:
    """Compare two selections made from the same candidate universe."""
    treatment_ids = {item.episode_id for item in treatment.selected}
    masked_ids = {item.episode_id for item in masked.selected}
    new_slots = treatment.covered_slots - masked.covered_slots
    lost_slots = masked.covered_slots - treatment.covered_slots
    return {
        "treatment_episode_ids": sorted(treatment_ids),
        "masked_episode_ids": sorted(masked_ids),
        "new_slots": sorted(new_slots),
        "new_slot_count": len(new_slots),
        "lost_slots": sorted(lost_slots),
        "harm": bool(lost_slots),
    }


def strict_contextual_attribution(
    candidates: Iterable[EvidenceCandidate | SlotCandidate],
    slots: Iterable[EvidenceSlot],
    budget: int,
) -> dict[str, object]:
    """Run local Treatment/Masked, single-edge, and leave-one-out replays.

    Only edges that actually place an Episode in Treatment are attributed.
    This is intentionally deterministic and consumes no model or database
    resource, so a later background worker can decide whether an observation
    is safe to persist after the user-visible answer guard succeeds.
    """
    candidate_list = list(candidates)
    slot_list = list(slots)
    masked_candidates = [
        item
        for item in candidate_list
        if getattr(item, "contextual_edge_id", None) is None
    ]
    masked = select_evidence(masked_candidates, slot_list, budget)
    treatment = select_evidence(candidate_list, slot_list, budget)
    delta = treatment_masked_delta(treatment, masked)
    selected_edge_ids = {
        int(getattr(item, "contextual_edge_id"))
        for item in treatment.selected
        if getattr(item, "contextual_edge_id", None) is not None
    }
    rows: list[dict[str, object]] = []
    for edge_id in sorted(selected_edge_ids):
        only_edge = [
            item
            for item in candidate_list
            if getattr(item, "contextual_edge_id", None) in {None, edge_id}
        ]
        without_edge = [
            item
            for item in candidate_list
            if getattr(item, "contextual_edge_id", None) != edge_id
        ]
        single = select_evidence(only_edge, slot_list, budget)
        leave_one_out = select_evidence(without_edge, slot_list, budget)
        single_delta = treatment_masked_delta(single, masked)
        loo_delta = treatment_masked_delta(treatment, leave_one_out)
        rows.append(
            {
                "association_id": edge_id,
                "selected": True,
                "single_edge_new_slots": single_delta["new_slots"],
                "leave_one_out_new_slots": loo_delta["new_slots"],
                "sufficient": bool(single_delta["new_slots"]),
                "necessary": bool(loo_delta["new_slots"]),
                "harm": bool(single_delta["harm"] or loo_delta["harm"]),
            }
        )
    return {
        "masked": masked,
        "treatment": treatment,
        "delta": delta,
        "edges": rows,
    }
