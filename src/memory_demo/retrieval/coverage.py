from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Iterable, Literal

from memory_demo.types import (
    CandidateAggregate,
    CandidateContribution,
    ClauseRequirement,
    EvidenceSelectionBudget,
    EvidenceSlot,
    SlotCandidate,
    SourceFactRef,
)


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


# ---------------------------------------------------------------------------
# V3 request-local contribution core
# ---------------------------------------------------------------------------
#
# The legacy selector above intentionally remains unchanged while normal query
# paths are migrated.  The objects/functions below form the one pure selection
# implementation that T10/T11 will share for normal selection, masks and
# leave-one-out runs.  They do not read repositories, invoke models, or infer
# factual support from a relevance score.


@dataclass(frozen=True, slots=True)
class ContributionCoverageState:
    """Coverage observed from a fixed set of candidate aggregates."""

    covered_required_clauses: frozenset[str]
    covered_optional_clauses: frozenset[str]
    incomplete_joint_clauses: frozenset[str]
    supported_components: tuple[tuple[str, tuple[str, ...]], ...]
    supporting_source_facts: tuple[SourceFactRef, ...]

    def components_for(self, slot_id: str) -> frozenset[str]:
        wanted = str(slot_id).strip()
        for item_slot_id, components in self.supported_components:
            if item_slot_id == wanted:
                return frozenset(components)
        return frozenset()

    @property
    def covered_clauses(self) -> frozenset[str]:
        return self.covered_required_clauses | self.covered_optional_clauses


@dataclass(frozen=True, slots=True)
class ContributionSelectionDecision:
    """One deterministic selector observation, suitable for a trace stage."""

    iteration: int
    chosen_episode_id: int | None
    chosen_contribution_ids: tuple[str, ...]
    covered_required_before: tuple[str, ...]
    newly_covered_required_clauses: tuple[str, ...]
    joint_incomplete_clauses: tuple[str, ...]
    remaining_episode_budget: int
    remaining_source_fact_budget: int | None
    remaining_delivery_token_budget: int | None
    score_components: tuple[tuple[str, float], ...] = ()
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ContributionCoverageSelection:
    """Result of contribution-aware, source-bound evidence selection."""

    selected: tuple[CandidateAggregate, ...]
    selected_episode_ids: tuple[int, ...]
    selected_contribution_ids: tuple[str, ...]
    covered_required_clauses: frozenset[str]
    missing_required_clauses: frozenset[str]
    covered_optional_clauses: frozenset[str]
    incomplete_joint_clauses: frozenset[str]
    budget: EvidenceSelectionBudget
    budget_exhausted: bool
    episode_budget_exhausted: bool
    source_fact_budget_exhausted: bool
    delivery_token_budget_exhausted: bool
    delivery_loss: bool
    delivery_loss_clauses: frozenset[str]
    actual_delivery_episode_count: int
    actual_delivery_source_fact_count: int
    actual_delivery_token_cost: int
    decisions: tuple[ContributionSelectionDecision, ...]
    stop_reason: str
    selection_strategy: Literal["greedy", "oracle"] = "greedy"

    @property
    def covered_clauses(self) -> frozenset[str]:
        """Compatibility-friendly shorthand for required plus optional."""

        return self.covered_required_clauses | self.covered_optional_clauses

    @property
    def missing_required(self) -> frozenset[str]:
        return self.missing_required_clauses

    @property
    def delivery_loss_due_to_budget(self) -> bool:
        return self.delivery_loss and self.budget_exhausted


def aggregate_contributions(
    contributions: Iterable[CandidateContribution],
) -> tuple[CandidateAggregate, ...]:
    """Merge only by Episode while retaining every base/edge contribution.

    This is deliberately not a "best source" reducer.  Contributions are
    collected before any ranking decision so that a later contribution-level
    mask can remove an edge route without erasing a separate dense/sparse base
    route to the same Episode.
    """

    grouped: dict[int, list[CandidateContribution]] = {}
    for contribution in contributions:
        if not isinstance(contribution, CandidateContribution):
            raise TypeError("aggregate_contributions expects CandidateContribution values")
        grouped.setdefault(contribution.episode_id, []).append(contribution)
    return tuple(
        CandidateAggregate(episode_id=episode_id, contributions=tuple(items))
        for episode_id, items in sorted(grouped.items())
    )


# Verbose alias makes it unambiguous that this is the T10 aggregate operation.
aggregate_candidate_contributions = aggregate_contributions


def _normalise_aggregates(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
) -> tuple[CandidateAggregate, ...]:
    grouped_contributions: dict[int, list[CandidateContribution]] = {}
    grouped_facts: dict[int, list[SourceFactRef]] = {}
    grouped_spans: dict[int, list[str]] = {}
    grouped_token_cost: dict[int, int] = {}
    for candidate in candidates:
        if isinstance(candidate, CandidateContribution):
            aggregate = CandidateAggregate(
                episode_id=candidate.episode_id,
                contributions=(candidate,),
            )
        elif isinstance(candidate, CandidateAggregate):
            aggregate = candidate
        else:
            raise TypeError(
                "contribution selector expects CandidateContribution or CandidateAggregate values"
            )
        episode_id = aggregate.episode_id
        grouped_contributions.setdefault(episode_id, []).extend(
            aggregate.contributions
        )
        grouped_facts.setdefault(episode_id, []).extend(aggregate.source_facts)
        grouped_spans.setdefault(episode_id, []).extend(aggregate.source_span_refs)
        grouped_token_cost[episode_id] = max(
            grouped_token_cost.get(episode_id, 0),
            aggregate.delivery_token_cost,
        )
    return tuple(
        CandidateAggregate(
            episode_id=episode_id,
            contributions=tuple(grouped_contributions[episode_id]),
            source_facts=tuple(grouped_facts[episode_id]),
            source_span_refs=tuple(grouped_spans[episode_id]),
            delivery_token_cost=grouped_token_cost[episode_id],
        )
        for episode_id in sorted(grouped_contributions)
    )


def _normalise_requirements(
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
) -> tuple[ClauseRequirement, ...]:
    by_slot: dict[str, ClauseRequirement] = {}
    for item in requirements:
        requirement = (
            ClauseRequirement.from_evidence_slot(item)
            if isinstance(item, EvidenceSlot)
            else item
        )
        if not isinstance(requirement, ClauseRequirement):
            raise TypeError(
                "contribution selector requirements must be EvidenceSlot or ClauseRequirement"
            )
        previous = by_slot.get(requirement.slot_id)
        if previous is not None and previous != requirement:
            raise ValueError("duplicate clause requirement slot_id is ambiguous")
        by_slot.setdefault(requirement.slot_id, requirement)
    return tuple(by_slot[slot_id] for slot_id in sorted(by_slot))


def contribution_coverage_state(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
) -> ContributionCoverageState:
    """Compute source-bound alternative/joint coverage with no ranking input.

    Only a ``ClauseSupport`` whose verifier status is ``verified`` or
    ``source_bound`` *and* carries a :class:`SourceFactRef` can participate.
    This is the hard boundary that prevents relevance-only model estimates from
    satisfying a required production clause.
    """

    aggregates = _normalise_aggregates(candidates)
    requirement_list = _normalise_requirements(requirements)
    required_slot_ids = {
        requirement.slot_id for requirement in requirement_list if requirement.required
    }
    component_map: dict[str, set[str]] = {
        requirement.slot_id: set() for requirement in requirement_list
    }
    fact_map: dict[tuple[str, tuple[str, ...], str, str], SourceFactRef] = {}
    requirements_by_slot = {
        requirement.slot_id: requirement for requirement in requirement_list
    }
    for aggregate in aggregates:
        for support in aggregate.clause_supports:
            requirement = requirements_by_slot.get(support.slot_id)
            if requirement is None:
                continue
            if support.support_mode != requirement.support_mode:
                # A mapper cannot silently turn a joint clause into an
                # alternative merely by re-labelling a support row.
                continue
            if support.clause_id not in requirement.clause_ids:
                continue
            if not support.is_required_coverage:
                continue
            component_map[requirement.slot_id].add(support.clause_id)
            assert support.source_fact is not None  # narrows property contract
            fact_map.setdefault(
                support.source_fact.identity_key,
                support.source_fact,
            )

    covered_required: set[str] = set()
    covered_optional: set[str] = set()
    incomplete_joint: set[str] = set()
    for requirement in requirement_list:
        components = component_map[requirement.slot_id]
        complete = (
            bool(components)
            if requirement.support_mode == "alternative"
            else set(requirement.clause_ids).issubset(components)
        )
        if complete:
            if requirement.required:
                covered_required.add(requirement.slot_id)
            else:
                covered_optional.add(requirement.slot_id)
        elif requirement.support_mode == "joint" and components:
            incomplete_joint.add(requirement.slot_id)
    return ContributionCoverageState(
        covered_required_clauses=frozenset(covered_required),
        covered_optional_clauses=frozenset(covered_optional),
        incomplete_joint_clauses=frozenset(incomplete_joint),
        supported_components=tuple(
            (slot_id, tuple(sorted(components)))
            for slot_id, components in sorted(component_map.items())
        ),
        supporting_source_facts=tuple(
            fact_map[key] for key in sorted(fact_map, key=lambda key: key)
        ),
    )


def _aggregate_fact_keys(
    aggregates: Iterable[CandidateAggregate],
) -> set[tuple[str, tuple[str, ...], str, str]]:
    return {
        fact.identity_key
        for aggregate in aggregates
        for fact in aggregate.source_facts
    }


def _aggregate_token_cost(aggregates: Iterable[CandidateAggregate]) -> int:
    return sum(int(item.delivery_token_cost) for item in aggregates)


def _fits_explicit_budget(
    selected: tuple[CandidateAggregate, ...],
    candidate: CandidateAggregate,
    budget: EvidenceSelectionBudget,
) -> bool:
    if len(selected) >= budget.episode_limit:
        return False
    selected_facts = _aggregate_fact_keys(selected)
    candidate_facts = _aggregate_fact_keys((candidate,))
    if (
        budget.source_fact_limit is not None
        and len(selected_facts | candidate_facts) > budget.source_fact_limit
    ):
        return False
    if (
        budget.delivery_token_limit is not None
        and _aggregate_token_cost((*selected, candidate))
        > budget.delivery_token_limit
    ):
        return False
    return True


def _score_candidate_step(
    candidate: CandidateAggregate,
    selected: tuple[CandidateAggregate, ...],
    requirements: tuple[ClauseRequirement, ...],
) -> tuple[tuple[float, ...], dict[str, float], ContributionCoverageState]:
    before = contribution_coverage_state(selected, requirements)
    after = contribution_coverage_state((*selected, candidate), requirements)
    newly_required = after.covered_required_clauses - before.covered_required_clauses
    newly_optional = after.covered_optional_clauses - before.covered_optional_clauses
    joint_progress = 0.0
    for requirement in requirements:
        if requirement.support_mode != "joint":
            continue
        denominator = max(1, len(requirement.clause_ids))
        before_components = before.components_for(requirement.slot_id)
        after_components = after.components_for(requirement.slot_id)
        if set(requirement.clause_ids).issubset(before_components):
            continue
        joint_progress += max(
            0.0,
            (len(after_components) - len(before_components)) / denominator,
        )
    selected_facts = _aggregate_fact_keys(selected)
    candidate_facts = _aggregate_fact_keys((candidate,))
    new_fact_count = len(candidate_facts - selected_facts)
    components = {
        "new_required_clause_count": float(len(newly_required)),
        "joint_progress": float(joint_progress),
        "new_optional_clause_count": float(len(newly_optional)),
        "new_source_fact_count": float(new_fact_count),
        "delivery_token_cost": float(candidate.delivery_token_cost),
        "relevance_score": float(candidate.relevance_score),
    }
    # Required completions dominate.  Joint component progress comes before
    # rank relevance so a valid prerequisite cannot be abandoned merely
    # because it has zero *completed-clause* marginal gain by itself.
    rank_key = (
        components["new_required_clause_count"],
        components["joint_progress"],
        components["new_optional_clause_count"],
        -components["new_source_fact_count"],
        -components["delivery_token_cost"],
        components["relevance_score"],
        -float(candidate.episode_id),
    )
    return rank_key, components, after


def _has_meaningful_step(components: dict[str, float]) -> bool:
    return bool(
        components["new_required_clause_count"] > 0
        or components["joint_progress"] > 0
        or components["new_optional_clause_count"] > 0
        or components["relevance_score"] > 0
    )


def _budget_flags(
    selected: tuple[CandidateAggregate, ...],
    all_aggregates: tuple[CandidateAggregate, ...],
    requirements: tuple[ClauseRequirement, ...],
    budget: EvidenceSelectionBudget,
    missing_required: frozenset[str],
) -> tuple[bool, bool, bool, bool]:
    selected_ids = {item.episode_id for item in selected}
    remaining = tuple(
        item for item in all_aggregates if item.episode_id not in selected_ids
    )
    useful_remaining: list[CandidateAggregate] = []
    for candidate in remaining:
        _rank, components, _state = _score_candidate_step(
            candidate, selected, requirements
        )
        if (
            components["new_required_clause_count"] > 0
            or components["joint_progress"] > 0
        ):
            useful_remaining.append(candidate)
    episode_exhausted = bool(
        missing_required
        and useful_remaining
        and len(selected) >= budget.episode_limit
    )
    selected_facts = _aggregate_fact_keys(selected)
    source_fact_exhausted = bool(
        missing_required
        and budget.source_fact_limit is not None
        and any(
            len(selected_facts | _aggregate_fact_keys((candidate,)))
            > budget.source_fact_limit
            for candidate in useful_remaining
        )
    )
    selected_tokens = _aggregate_token_cost(selected)
    token_exhausted = bool(
        missing_required
        and budget.delivery_token_limit is not None
        and any(
            selected_tokens + candidate.delivery_token_cost
            > budget.delivery_token_limit
            for candidate in useful_remaining
        )
    )
    return (
        episode_exhausted or source_fact_exhausted or token_exhausted,
        episode_exhausted,
        source_fact_exhausted,
        token_exhausted,
    )


def _build_contribution_selection(
    *,
    selected: tuple[CandidateAggregate, ...],
    all_aggregates: tuple[CandidateAggregate, ...],
    requirements: tuple[ClauseRequirement, ...],
    budget: EvidenceSelectionBudget,
    decisions: tuple[ContributionSelectionDecision, ...],
    stop_reason: str,
    strategy: Literal["greedy", "oracle"],
) -> ContributionCoverageSelection:
    state = contribution_coverage_state(selected, requirements)
    all_state = contribution_coverage_state(all_aggregates, requirements)
    required_ids = {
        requirement.slot_id for requirement in requirements if requirement.required
    }
    missing = frozenset(required_ids - state.covered_required_clauses)
    available_but_undelivered = frozenset(
        missing & all_state.covered_required_clauses
    )
    budget_exhausted, episode_exhausted, source_fact_exhausted, token_exhausted = (
        _budget_flags(selected, all_aggregates, requirements, budget, missing)
    )
    selected_episode_ids = tuple(item.episode_id for item in selected)
    selected_contribution_ids = tuple(
        contribution_id
        for item in selected
        for contribution_id in item.contribution_ids
    )
    actual_fact_count = len(_aggregate_fact_keys(selected))
    actual_tokens = _aggregate_token_cost(selected)
    # These are internal invariants.  A caller that sees a result can trust
    # its delivered set never exceeded the explicit request limits.
    if len(selected_episode_ids) > budget.episode_limit:
        raise AssertionError("selector exceeded episode budget")
    if (
        budget.source_fact_limit is not None
        and actual_fact_count > budget.source_fact_limit
    ):
        raise AssertionError("selector exceeded source fact budget")
    if (
        budget.delivery_token_limit is not None
        and actual_tokens > budget.delivery_token_limit
    ):
        raise AssertionError("selector exceeded delivery token budget")
    return ContributionCoverageSelection(
        selected=selected,
        selected_episode_ids=selected_episode_ids,
        selected_contribution_ids=selected_contribution_ids,
        covered_required_clauses=state.covered_required_clauses,
        missing_required_clauses=missing,
        covered_optional_clauses=state.covered_optional_clauses,
        incomplete_joint_clauses=state.incomplete_joint_clauses,
        budget=budget,
        budget_exhausted=budget_exhausted,
        episode_budget_exhausted=episode_exhausted,
        source_fact_budget_exhausted=source_fact_exhausted,
        delivery_token_budget_exhausted=token_exhausted,
        delivery_loss=bool(available_but_undelivered),
        delivery_loss_clauses=available_but_undelivered,
        actual_delivery_episode_count=len(selected_episode_ids),
        actual_delivery_source_fact_count=actual_fact_count,
        actual_delivery_token_cost=actual_tokens,
        decisions=decisions,
        stop_reason=stop_reason,
        selection_strategy=strategy,
    )


def select_contribution_evidence(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
) -> ContributionCoverageSelection:
    """Select a request-bounded, provenance-preserving evidence set.

    The budget is an explicit request input.  In particular, it is never
    shrunk to the size of a base candidate pool before contextual candidates
    arrive.  The selector considers whole Episode aggregates, so an Episode
    supporting multiple clauses occupies one delivery position while retaining
    every contribution that justified it.
    """

    if not isinstance(budget, EvidenceSelectionBudget):
        raise TypeError("v3 contribution selection requires EvidenceSelectionBudget")
    aggregates = _normalise_aggregates(candidates)
    requirement_list = _normalise_requirements(requirements)
    required_slot_ids = {
        requirement.slot_id
        for requirement in requirement_list
        if requirement.required
    }
    selected: tuple[CandidateAggregate, ...] = ()
    remaining = list(aggregates)
    decisions: list[ContributionSelectionDecision] = []
    stop_reason = "candidate_pool_exhausted"

    while remaining and len(selected) < budget.episode_limit:
        feasible = [
            item
            for item in remaining
            if _fits_explicit_budget(selected, item, budget)
        ]
        if not feasible:
            stop_reason = "delivery_budget_blocked"
            break
        scored = [
            (_score_candidate_step(item, selected, requirement_list), item)
            for item in feasible
        ]
        (rank_key, score_components, after_state), chosen = max(
            scored,
            key=lambda pair: pair[0][0],
        )
        if not _has_meaningful_step(score_components):
            stop_reason = "no_positive_marginal_gain"
            break
        before_state = contribution_coverage_state(selected, requirement_list)
        selected = (*selected, chosen)
        remaining.remove(chosen)
        selected_facts = _aggregate_fact_keys(selected)
        selected_tokens = _aggregate_token_cost(selected)
        decisions.append(
            ContributionSelectionDecision(
                iteration=len(decisions),
                chosen_episode_id=chosen.episode_id,
                chosen_contribution_ids=chosen.contribution_ids,
                covered_required_before=tuple(
                    sorted(before_state.covered_required_clauses)
                ),
                newly_covered_required_clauses=tuple(
                    sorted(
                        after_state.covered_required_clauses
                        - before_state.covered_required_clauses
                    )
                ),
                joint_incomplete_clauses=tuple(
                    sorted(after_state.incomplete_joint_clauses)
                ),
                remaining_episode_budget=max(0, budget.episode_limit - len(selected)),
                remaining_source_fact_budget=(
                    None
                    if budget.source_fact_limit is None
                    else max(0, budget.source_fact_limit - len(selected_facts))
                ),
                remaining_delivery_token_budget=(
                    None
                    if budget.delivery_token_limit is None
                    else max(0, budget.delivery_token_limit - selected_tokens)
                ),
                score_components=tuple(
                    sorted(
                        {
                            **score_components,
                            "rank_key_required": rank_key[0],
                            "rank_key_joint_progress": rank_key[1],
                        }.items()
                    )
                ),
            )
        )
        if not required_slot_ids.difference(after_state.covered_required_clauses):
            stop_reason = "all_required_clauses_covered"
            break
    else:
        if len(selected) >= budget.episode_limit and remaining:
            stop_reason = "episode_budget_reached"
        elif not remaining:
            stop_reason = "candidate_pool_exhausted"

    if not decisions and not aggregates:
        stop_reason = "no_candidates"
    final_state = contribution_coverage_state(selected, requirement_list)
    decisions.append(
        ContributionSelectionDecision(
            iteration=len(decisions),
            chosen_episode_id=None,
            chosen_contribution_ids=(),
            covered_required_before=tuple(sorted(final_state.covered_required_clauses)),
            newly_covered_required_clauses=(),
            joint_incomplete_clauses=tuple(sorted(final_state.incomplete_joint_clauses)),
            remaining_episode_budget=max(0, budget.episode_limit - len(selected)),
            remaining_source_fact_budget=(
                None
                if budget.source_fact_limit is None
                else max(0, budget.source_fact_limit - len(_aggregate_fact_keys(selected)))
            ),
            remaining_delivery_token_budget=(
                None
                if budget.delivery_token_limit is None
                else max(0, budget.delivery_token_limit - _aggregate_token_cost(selected))
            ),
            stop_reason=stop_reason,
        )
    )
    return _build_contribution_selection(
        selected=selected,
        all_aggregates=aggregates,
        requirements=requirement_list,
        budget=budget,
        decisions=tuple(decisions),
        stop_reason=stop_reason,
        strategy="greedy",
    )


# More discoverable name for callers that operate on already-merged candidates.
select_candidate_aggregates = select_contribution_evidence


def _selection_is_within_budget(
    selected: tuple[CandidateAggregate, ...],
    budget: EvidenceSelectionBudget,
) -> bool:
    if len(selected) > budget.episode_limit:
        return False
    if (
        budget.source_fact_limit is not None
        and len(_aggregate_fact_keys(selected)) > budget.source_fact_limit
    ):
        return False
    return not (
        budget.delivery_token_limit is not None
        and _aggregate_token_cost(selected) > budget.delivery_token_limit
    )


def _oracle_key(
    selected: tuple[CandidateAggregate, ...],
    requirements: tuple[ClauseRequirement, ...],
) -> tuple[object, ...]:
    state = contribution_coverage_state(selected, requirements)
    # Smaller evidence sets/fact sets win after equal coverage.  Relevance is
    # only a late tie-breaker and cannot convert an unverified mapping into
    # coverage.  The final negative ids make ties deterministic.
    return (
        len(state.covered_required_clauses),
        len(state.covered_optional_clauses),
        -len(selected),
        -len(_aggregate_fact_keys(selected)),
        -_aggregate_token_cost(selected),
        round(sum(item.relevance_score for item in selected), 12),
        tuple(-item.episode_id for item in selected),
    )


def exhaustive_contribution_oracle(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
    *,
    max_candidates: int = 12,
) -> ContributionCoverageSelection:
    """Exact small-set oracle for selector regression/property tests.

    This deliberately refuses large universes instead of quietly doing an
    exponential production search.  It evaluates the same source-bound
    coverage function and the same request budget as the greedy selector.
    """

    if not isinstance(budget, EvidenceSelectionBudget):
        raise TypeError("v3 contribution oracle requires EvidenceSelectionBudget")
    aggregates = _normalise_aggregates(candidates)
    requirements_list = _normalise_requirements(requirements)
    if int(max_candidates) < 0:
        raise ValueError("oracle max_candidates cannot be negative")
    if len(aggregates) > int(max_candidates):
        raise ValueError(
            "candidate universe exceeds exhaustive oracle max_candidates"
        )

    best: tuple[CandidateAggregate, ...] = ()
    best_key = _oracle_key(best, requirements_list)
    upper = min(len(aggregates), budget.episode_limit)
    for size in range(1, upper + 1):
        for combination in combinations(aggregates, size):
            selected = tuple(combination)
            if not _selection_is_within_budget(selected, budget):
                continue
            key = _oracle_key(selected, requirements_list)
            if key > best_key:
                best = selected
                best_key = key

    state = contribution_coverage_state(best, requirements_list)
    decisions = (
        ContributionSelectionDecision(
            iteration=0,
            chosen_episode_id=None,
            chosen_contribution_ids=tuple(
                contribution_id
                for item in best
                for contribution_id in item.contribution_ids
            ),
            covered_required_before=(),
            newly_covered_required_clauses=tuple(
                sorted(state.covered_required_clauses)
            ),
            joint_incomplete_clauses=tuple(sorted(state.incomplete_joint_clauses)),
            remaining_episode_budget=max(0, budget.episode_limit - len(best)),
            remaining_source_fact_budget=(
                None
                if budget.source_fact_limit is None
                else max(0, budget.source_fact_limit - len(_aggregate_fact_keys(best)))
            ),
            remaining_delivery_token_budget=(
                None
                if budget.delivery_token_limit is None
                else max(0, budget.delivery_token_limit - _aggregate_token_cost(best))
            ),
            stop_reason="exhaustive_oracle",
        ),
    )
    return _build_contribution_selection(
        selected=best,
        all_aggregates=aggregates,
        requirements=requirements_list,
        budget=budget,
        decisions=decisions,
        stop_reason="exhaustive_oracle",
        strategy="oracle",
    )


exhaustive_selection_oracle = exhaustive_contribution_oracle


def compare_greedy_to_oracle(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
    *,
    max_candidates: int = 12,
) -> dict[str, object]:
    """Return a compact deterministic regression record for a small fixture."""

    # Materialise once: generator inputs must feed both algorithms identically.
    candidate_list = tuple(candidates)
    requirement_list = tuple(requirements)
    greedy = select_contribution_evidence(candidate_list, requirement_list, budget)
    oracle = exhaustive_contribution_oracle(
        candidate_list,
        requirement_list,
        budget,
        max_candidates=max_candidates,
    )
    greedy_required = len(greedy.covered_required_clauses)
    oracle_required = len(oracle.covered_required_clauses)
    return {
        "greedy_selected_episode_ids": list(greedy.selected_episode_ids),
        "oracle_selected_episode_ids": list(oracle.selected_episode_ids),
        "greedy_covered_required_clauses": sorted(
            greedy.covered_required_clauses
        ),
        "oracle_covered_required_clauses": sorted(
            oracle.covered_required_clauses
        ),
        "greedy_required_count": greedy_required,
        "oracle_required_count": oracle_required,
        "greedy_loss": greedy_required < oracle_required,
        "greedy_delivery_loss": greedy.delivery_loss,
    }


# ---------------------------------------------------------------------------
# V3 contribution-level counterfactuals
# ---------------------------------------------------------------------------
#
# These helpers deliberately operate *below* the Episode aggregate.  A
# contextual route is one contribution among potentially several routes to the
# same Episode; removing that route must not accidentally remove a dense,
# sparse, or source-bound base route which happens to lead to the same result.
# They are pure request-local functions so normal selection and every
# counterfactual use the exact same selector and explicit budget.


@dataclass(frozen=True, slots=True)
class ContributionCounterfactualDelta:
    """Deterministic comparison of a full treatment and one masked replay.

    ``gained_required_clauses`` are clauses present in treatment but absent
    after the mask.  ``lost_required_clauses`` are the reverse: clauses that
    the masked replay can deliver but treatment cannot.  The latter is harm
    even when the treatment also gains another clause, so ``classification``
    gives harm priority and ``pure_success`` cannot be true in a mixed run.
    """

    treatment: ContributionCoverageSelection
    masked: ContributionCoverageSelection
    masked_contribution_ids: tuple[str, ...]
    masked_edge_ids: tuple[int, ...]
    gained_required_clauses: frozenset[str]
    lost_required_clauses: frozenset[str]
    gained_optional_clauses: frozenset[str]
    lost_optional_clauses: frozenset[str]
    classification: Literal["success", "harmful", "neutral"]
    mixed: bool
    pure_success: bool
    harm: bool

    @property
    def treatment_episode_ids(self) -> tuple[int, ...]:
        return self.treatment.selected_episode_ids

    @property
    def masked_episode_ids(self) -> tuple[int, ...]:
        return self.masked.selected_episode_ids

    @property
    def new_required_clauses(self) -> frozenset[str]:
        """Compatibility-friendly name for treatment's required gains."""

        return self.gained_required_clauses

    @property
    def has_harm(self) -> bool:
        return self.harm

    def as_dict(self) -> dict[str, object]:
        """Return a stable primitive summary for receipts and regression tests."""

        return {
            "treatment_episode_ids": list(self.treatment_episode_ids),
            "masked_episode_ids": list(self.masked_episode_ids),
            "masked_contribution_ids": list(self.masked_contribution_ids),
            "masked_edge_ids": list(self.masked_edge_ids),
            "gained_required_clauses": sorted(self.gained_required_clauses),
            "lost_required_clauses": sorted(self.lost_required_clauses),
            "gained_optional_clauses": sorted(self.gained_optional_clauses),
            "lost_optional_clauses": sorted(self.lost_optional_clauses),
            "classification": self.classification,
            "mixed": self.mixed,
            "pure_success": self.pure_success,
            "harm": self.harm,
        }


# A comparison is the public counterfactual result; retain the longer name as
# an alias for callers whose code reads more naturally with it.
ContributionCounterfactualComparison = ContributionCounterfactualDelta


@dataclass(frozen=True, slots=True)
class ContextualEdgeCounterfactual:
    """Single-edge and leave-one-out results for one contextual edge."""

    edge_id: int
    contribution_ids: tuple[str, ...]
    selected_in_treatment: bool
    single_edge: ContributionCounterfactualDelta
    leave_one_out: ContributionCounterfactualDelta
    sufficient: bool
    necessary: bool
    harmful: bool

    @property
    def classification(self) -> Literal["success", "harmful", "neutral"]:
        """Classify the edge with the same harm-first rule as each replay."""

        if self.harmful:
            return "harmful"
        if self.sufficient or self.necessary:
            return "success"
        return "neutral"


@dataclass(frozen=True, slots=True)
class ContributionContextualAttribution:
    """Treatment/masked and per-edge counterfactuals from one fixed universe."""

    treatment: ContributionCoverageSelection
    masked: ContributionCoverageSelection
    treatment_vs_masked: ContributionCounterfactualDelta
    edges: tuple[ContextualEdgeCounterfactual, ...]
    candidate_universe_fingerprint: str
    requirements_fingerprint: str
    budget_fingerprint: str
    input_fingerprint: str

    @property
    def edge_ids(self) -> tuple[int, ...]:
        return tuple(item.edge_id for item in self.edges)


ContributionAttributionResult = ContributionContextualAttribution


def _opaque_fingerprint(payload: object) -> str:
    """Hash canonical internal inputs without exposing their raw identifiers."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _contribution_universe_fingerprint(
    aggregates: tuple[CandidateAggregate, ...],
) -> str:
    """Fingerprint selector-relevant inputs, excluding source paths and text."""

    payload: list[dict[str, object]] = []
    for aggregate in aggregates:
        contributions: list[dict[str, object]] = []
        for contribution in aggregate.contributions:
            supports = [
                {
                    "slot_id": support.slot_id,
                    "clause_id": support.clause_id,
                    "support_mode": support.support_mode,
                    "verification_status": support.verification_status,
                    "source_fact_id": (
                        support.source_fact.fact_id
                        if support.source_fact is not None
                        else None
                    ),
                }
                for support in contribution.clause_supports
            ]
            contributions.append(
                {
                    "contribution_id": contribution.contribution_id,
                    "slot_id": contribution.slot_id,
                    "lane": contribution.lane,
                    "edge_id": contribution.edge_id,
                    "parent_contribution_ids": list(
                        contribution.parent_contribution_ids
                    ),
                    "rank_features": list(contribution.rank_features),
                    "source_fact_ids": [
                        fact.fact_id for fact in contribution.source_facts
                    ],
                    "clause_supports": supports,
                    "delivery_token_cost": contribution.delivery_token_cost,
                }
            )
        payload.append(
            {
                "episode_id": aggregate.episode_id,
                "source_fact_ids": [fact.fact_id for fact in aggregate.source_facts],
                "delivery_token_cost": aggregate.delivery_token_cost,
                "contributions": contributions,
            }
        )
    return _opaque_fingerprint(payload)


def _requirements_fingerprint(
    requirements: tuple[ClauseRequirement, ...],
) -> str:
    return _opaque_fingerprint(
        [
            {
                "slot_id": requirement.slot_id,
                "clause_ids": list(requirement.clause_ids),
                "support_mode": requirement.support_mode,
                "required": requirement.required,
            }
            for requirement in requirements
        ]
    )


def _budget_fingerprint(budget: EvidenceSelectionBudget) -> str:
    return _opaque_fingerprint(
        {
            "episode_limit": budget.episode_limit,
            "source_fact_limit": budget.source_fact_limit,
            "delivery_token_limit": budget.delivery_token_limit,
        }
    )


def _normalise_mask_contribution_ids(
    contribution_ids: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if contribution_ids is None:
        return ()
    values: Iterable[str]
    if isinstance(contribution_ids, str):
        values = (contribution_ids,)
    else:
        values = contribution_ids
    result: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            raise ValueError("masked contribution ids must be non-empty")
        result.add(value)
    return tuple(sorted(result))


def _normalise_mask_edge_ids(
    edge_ids: Iterable[int] | int | None,
) -> tuple[int, ...]:
    if edge_ids is None:
        return ()
    if isinstance(edge_ids, bool):
        raise TypeError("masked edge ids must be positive integers")
    values: Iterable[int]
    if isinstance(edge_ids, int):
        values = (edge_ids,)
    else:
        values = edge_ids  # type: ignore[assignment]
    result: set[int] = set()
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise TypeError("masked edge ids must be positive integers")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("masked edge ids must be positive integers") from exc
        if value <= 0:
            raise ValueError("masked edge ids must be positive")
        result.add(value)
    return tuple(sorted(result))


def mask_contribution_aggregates(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    *,
    contribution_ids: Iterable[str] | str | None = (),
    edge_ids: Iterable[int] | int | None = (),
) -> tuple[CandidateAggregate, ...]:
    """Mask exact contribution and/or edge IDs, then re-aggregate survivors.

    An edge mask removes only contributions whose ``edge_id`` exactly matches.
    It does not remove an entire Episode, any other edge, a shared cue, or an
    independent base contribution.  Rebuilding aggregates from survivors is
    intentional: stale source facts, spans, and rank features from a masked
    route cannot leak into the replay's coverage or delivery budget.
    """

    masked_contribution_ids = frozenset(
        _normalise_mask_contribution_ids(contribution_ids)
    )
    masked_edge_ids = frozenset(_normalise_mask_edge_ids(edge_ids))
    aggregates = _normalise_aggregates(candidates)
    survivors: list[CandidateContribution] = []
    for aggregate in aggregates:
        for contribution in aggregate.contributions:
            if contribution.contribution_id in masked_contribution_ids:
                continue
            if contribution.edge_id in masked_edge_ids:
                continue
            survivors.append(contribution)
    return aggregate_contributions(survivors)


# The noun phrase is useful at call sites that already have a contribution
# list, while the implementation name makes clear that it returns aggregates.
mask_candidate_contributions = mask_contribution_aggregates


def select_masked_contribution_evidence(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
    *,
    contribution_ids: Iterable[str] | str | None = (),
    edge_ids: Iterable[int] | int | None = (),
) -> ContributionCoverageSelection:
    """Run the normal V3 selector after an exact contribution-level mask."""

    candidate_list = tuple(candidates)
    requirement_list = tuple(requirements)
    masked = mask_contribution_aggregates(
        candidate_list,
        contribution_ids=contribution_ids,
        edge_ids=edge_ids,
    )
    return select_contribution_evidence(masked, requirement_list, budget)


def compare_contribution_selections(
    treatment: ContributionCoverageSelection,
    masked: ContributionCoverageSelection,
    *,
    masked_contribution_ids: Iterable[str] | str | None = (),
    masked_edge_ids: Iterable[int] | int | None = (),
) -> ContributionCounterfactualDelta:
    """Compare two V3 selections with a harm-first required-coverage label."""

    if not isinstance(treatment, ContributionCoverageSelection):
        raise TypeError("treatment must be a ContributionCoverageSelection")
    if not isinstance(masked, ContributionCoverageSelection):
        raise TypeError("masked must be a ContributionCoverageSelection")
    normalised_contribution_ids = _normalise_mask_contribution_ids(
        masked_contribution_ids
    )
    normalised_edge_ids = _normalise_mask_edge_ids(masked_edge_ids)
    gained_required = (
        treatment.covered_required_clauses - masked.covered_required_clauses
    )
    lost_required = (
        masked.covered_required_clauses - treatment.covered_required_clauses
    )
    gained_optional = (
        treatment.covered_optional_clauses - masked.covered_optional_clauses
    )
    lost_optional = (
        masked.covered_optional_clauses - treatment.covered_optional_clauses
    )
    harm = bool(lost_required)
    pure_success = bool(gained_required) and not harm
    classification: Literal["success", "harmful", "neutral"]
    if harm:
        classification = "harmful"
    elif pure_success:
        classification = "success"
    else:
        classification = "neutral"
    return ContributionCounterfactualDelta(
        treatment=treatment,
        masked=masked,
        masked_contribution_ids=normalised_contribution_ids,
        masked_edge_ids=normalised_edge_ids,
        gained_required_clauses=frozenset(gained_required),
        lost_required_clauses=frozenset(lost_required),
        gained_optional_clauses=frozenset(gained_optional),
        lost_optional_clauses=frozenset(lost_optional),
        classification=classification,
        mixed=bool(gained_required and lost_required),
        pure_success=pure_success,
        harm=harm,
    )


def run_contribution_counterfactual(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
    *,
    contribution_ids: Iterable[str] | str | None = (),
    edge_ids: Iterable[int] | int | None = (),
) -> ContributionCounterfactualDelta:
    """Run treatment and a specified masked replay with the same V3 selector."""

    candidate_list = tuple(candidates)
    requirement_list = tuple(requirements)
    normalised_contribution_ids = _normalise_mask_contribution_ids(contribution_ids)
    normalised_edge_ids = _normalise_mask_edge_ids(edge_ids)
    treatment = select_contribution_evidence(candidate_list, requirement_list, budget)
    masked = select_masked_contribution_evidence(
        candidate_list,
        requirement_list,
        budget,
        contribution_ids=normalised_contribution_ids,
        edge_ids=normalised_edge_ids,
    )
    return compare_contribution_selections(
        treatment,
        masked,
        masked_contribution_ids=normalised_contribution_ids,
        masked_edge_ids=normalised_edge_ids,
    )


# Short form for use in request-local selector wiring.
contribution_counterfactual = run_contribution_counterfactual


def contribution_contextual_attribution(
    candidates: Iterable[CandidateContribution | CandidateAggregate],
    requirements: Iterable[EvidenceSlot | ClauseRequirement],
    budget: EvidenceSelectionBudget,
) -> ContributionContextualAttribution:
    """Evaluate all contextual edges with treatment, baseline, and LOO runs.

    The masked baseline removes every *identified edge contribution*, while
    each single-edge replay restores only that edge alongside independent base
    contributions.  The leave-one-out replay removes exactly one edge and
    leaves all alternatives intact.  Every branch uses
    :func:`select_contribution_evidence`; this function introduces no separate
    ranking or mutable cache.
    """

    if not isinstance(budget, EvidenceSelectionBudget):
        raise TypeError("v3 contribution attribution requires EvidenceSelectionBudget")
    candidate_list = tuple(candidates)
    requirement_list = _normalise_requirements(requirements)
    aggregates = _normalise_aggregates(candidate_list)
    edge_to_contribution_ids: dict[int, set[str]] = {}
    for aggregate in aggregates:
        for contribution in aggregate.contributions:
            if contribution.edge_id is not None:
                edge_to_contribution_ids.setdefault(contribution.edge_id, set()).add(
                    contribution.contribution_id
                )
    edge_ids = tuple(sorted(edge_to_contribution_ids))
    candidate_universe_fingerprint = _contribution_universe_fingerprint(aggregates)
    requirements_fingerprint = _requirements_fingerprint(requirement_list)
    budget_fingerprint = _budget_fingerprint(budget)
    input_fingerprint = _opaque_fingerprint(
        {
            "candidate_universe_fingerprint": candidate_universe_fingerprint,
            "requirements_fingerprint": requirements_fingerprint,
            "budget_fingerprint": budget_fingerprint,
            "baseline_mask_edge_ids": edge_ids,
        }
    )
    treatment = select_contribution_evidence(aggregates, requirement_list, budget)
    masked = select_masked_contribution_evidence(
        aggregates,
        requirement_list,
        budget,
        edge_ids=edge_ids,
    )
    treatment_vs_masked = compare_contribution_selections(
        treatment,
        masked,
        masked_edge_ids=edge_ids,
    )
    selected_contribution_ids = frozenset(treatment.selected_contribution_ids)
    rows: list[ContextualEdgeCounterfactual] = []
    all_edge_ids = frozenset(edge_ids)
    for edge_id in edge_ids:
        single_edge_mask = tuple(sorted(all_edge_ids - {edge_id}))
        single_edge_selection = select_masked_contribution_evidence(
            aggregates,
            requirement_list,
            budget,
            edge_ids=single_edge_mask,
        )
        single_edge = compare_contribution_selections(
            single_edge_selection,
            masked,
            masked_edge_ids=edge_ids,
        )
        leave_one_out_selection = select_masked_contribution_evidence(
            aggregates,
            requirement_list,
            budget,
            edge_ids=(edge_id,),
        )
        leave_one_out = compare_contribution_selections(
            treatment,
            leave_one_out_selection,
            masked_edge_ids=(edge_id,),
        )
        sufficient = single_edge.pure_success
        necessary = leave_one_out.pure_success
        rows.append(
            ContextualEdgeCounterfactual(
                edge_id=edge_id,
                contribution_ids=tuple(sorted(edge_to_contribution_ids[edge_id])),
                selected_in_treatment=bool(
                    selected_contribution_ids
                    & frozenset(edge_to_contribution_ids[edge_id])
                ),
                single_edge=single_edge,
                leave_one_out=leave_one_out,
                sufficient=sufficient,
                necessary=necessary,
                harmful=bool(single_edge.harm or leave_one_out.harm),
            )
        )
    return ContributionContextualAttribution(
        treatment=treatment,
        masked=masked,
        treatment_vs_masked=treatment_vs_masked,
        edges=tuple(rows),
        candidate_universe_fingerprint=candidate_universe_fingerprint,
        requirements_fingerprint=requirements_fingerprint,
        budget_fingerprint=budget_fingerprint,
        input_fingerprint=input_fingerprint,
    )
