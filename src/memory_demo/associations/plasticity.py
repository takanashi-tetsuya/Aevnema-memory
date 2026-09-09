from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable, Mapping

from memory_demo.types import (
    ContextualRecallCandidate,
    ContextualUtilityLedgerObservation,
    ContextualUtilityObservation,
    LearningAnchor,
    LearningCandidate,
    LearningCandidatePlan,
    LearningCandidateRejection,
    RecallLearningEvent,
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
    contextual_anchor_ids: Iterable[int] = (),
    initial_candidate_episode_ids: Iterable[int] | None = None,
    initial_selected_episode_ids: Iterable[int] | None = None,
    final_selected_episode_ids: Iterable[int] | None = None,
    initial_delivered_episode_ids: Iterable[int] | None = None,
    final_delivered_episode_ids: Iterable[int] | None = None,
    creation_reason: str = "",
    costly_success: bool = False,
    costly_stage_refs: Iterable[str] = (),
) -> list[ContextualRecallCandidate]:
    """Derive legacy candidate records without silently excluding base targets.

    This compatibility helper cannot infer a dropped target from the old
    three-set signature.  Callers can provide the initial delivery set or an
    explicit creation reason.  A base candidate is therefore accepted only
    when there is an auditable recovery/reuse reason, never merely because it
    happened to be present in the final selected list.
    """

    base = {int(value) for value in base_episode_ids}
    initial_candidates = (
        {int(value) for value in initial_candidate_episode_ids}
        if initial_candidate_episode_ids is not None
        else set(base)
    )
    initial_delivered = (
        {int(value) for value in initial_delivered_episode_ids}
        if initial_delivered_episode_ids is not None
        else set()
    )
    initial_selected = (
        {int(value) for value in initial_selected_episode_ids}
        if initial_selected_episode_ids is not None
        else set()
    )
    selected = (
        {int(value) for value in final_selected_episode_ids}
        if final_selected_episode_ids is not None
        else {int(value) for value in selected_episode_ids}
    )
    delivered = (
        {int(value) for value in final_delivered_episode_ids}
        if final_delivered_episode_ids is not None
        else set(selected)
    )
    contextual = {int(value) for value in contextual_episode_ids}
    contextual_anchors = {int(value) for value in contextual_anchor_ids}
    explicit_reason = str(creation_reason or "").strip()
    valid_reasons = {
        "candidate_missing",
        "candidate_present_dropped",
        "source_only",
        "clarification_recovery",
        "costly_success_reuse",
    }
    if explicit_reason and explicit_reason not in valid_reasons:
        raise ValueError("creation_reason is invalid")
    costly = bool(
        costly_success
        or tuple(
            str(value).strip()
            for value in costly_stage_refs
            if str(value).strip()
        )
    )
    candidate_limit = min(4, max(0, int(max_candidates)))
    normalized_anchor_id = int(anchor_id)
    if (
        candidate_limit == 0
        or normalized_anchor_id <= 0
        or anchor_type not in {"episode", "concept"}
        or (
            anchor_type == "episode"
            and normalized_anchor_id in (contextual_anchors | contextual)
        )
    ):
        return []
    result: list[ContextualRecallCandidate] = []
    for target in dict.fromkeys(int(value) for value in target_episode_ids):
        if target <= 0 or target in contextual or target not in selected or target not in delivered:
            continue
        if explicit_reason:
            reason = explicit_reason
        elif target not in initial_candidates:
            reason = "candidate_missing"
        elif (
            initial_selected_episode_ids is not None
            and target not in initial_selected
        ):
            reason = "candidate_present_dropped"
        elif (
            initial_delivered_episode_ids is not None
            and target not in initial_delivered
        ):
            reason = "candidate_present_dropped"
        elif costly:
            reason = "costly_success_reuse"
        else:
            # The legacy signature does not prove that a base-present target
            # was dropped or that a correct path was costly.  Do not invent a
            # learning reason from presence alone.
            continue
        if anchor_type == "episode" and normalized_anchor_id == target:
            continue
        result.append(
            ContextualRecallCandidate(
                anchor_type=anchor_type,
                anchor_id=normalized_anchor_id,
                target_episode_id=target,
                context_query_id=str(context_query_id),
                need_query_id=str(need_query_id),
                slot_id=str(slot_id),
                source_request_hash=str(source_request_hash),
                reason=reason,
            )
        )
        if len(result) >= candidate_limit:
            break
    return result


def select_independent_learning_anchors(
    anchors: Iterable[LearningAnchor],
    *,
    contextual_episode_ids: Iterable[int] = (),
    target_episode_ids: Iterable[int] = (),
    max_anchors: int = 4,
) -> tuple[LearningAnchor, ...]:
    """Pick at most four deterministic, independently-found base anchors.

    The cap is applied before target pairing, preventing a selected-all
    Cartesian-product of candidate endpoints.  Episode anchors observed only
    through same-round contextual expansion are excluded even if their lane
    were accidentally labelled as a base lane.
    """

    limit = min(4, max(0, int(max_anchors)))
    if limit == 0:
        return ()
    contextual = {int(value) for value in contextual_episode_ids}
    targets = {int(value) for value in target_episode_ids}
    eligible: list[LearningAnchor] = []
    for anchor in anchors:
        if not isinstance(anchor, LearningAnchor):
            raise TypeError("learning anchors must be LearningAnchor instances")
        if not anchor.is_eligible_base_anchor:
            continue
        if anchor.anchor_type == "episode" and (
            anchor.anchor_id in contextual or anchor.anchor_id in targets
        ):
            continue
        eligible.append(anchor)

    # One physical endpoint may have several lane contributions, but it is one
    # anchor for a single learning event.  Keep its strongest deterministic
    # base route; preserve source provenance on that route in the candidate.
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item.activation),
            item.anchor_type,
            int(item.anchor_id),
            item.contribution_id,
        ),
    )
    selected: list[LearningAnchor] = []
    seen_endpoints: set[tuple[str, int]] = set()
    for anchor in ranked:
        endpoint = (anchor.anchor_type, anchor.anchor_id)
        if endpoint in seen_endpoints:
            continue
        seen_endpoints.add(endpoint)
        selected.append(anchor)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _candidate_id(
    event: RecallLearningEvent,
    anchor: LearningAnchor,
    target_episode_id: int,
    reason: str,
) -> str:
    payload = "\0".join(
        (
            event.request_id,
            event.request_hash,
            event.domain,
            anchor.anchor_type,
            str(anchor.anchor_id),
            anchor.contribution_id,
            str(int(target_episode_id)),
            str(reason),
            event.context_query_id,
            event.need_query_id,
            event.slot_id,
        )
    ).encode("utf-8")
    return "learning-candidate:sha256:" + hashlib.sha256(payload).hexdigest()


def _event_target_rejection(
    event: RecallLearningEvent,
    target_episode_id: int,
) -> str | None:
    target = int(target_episode_id)
    if event.cancelled:
        return "request_cancelled"
    if not event.answer_guard_passed:
        return "answer_guard_failed"
    if target not in event.final_delivered_episode_ids:
        return "target_not_delivered"
    if target in event.contextual_expansion_episode_ids:
        return "target_returned_by_contextual_expansion"
    if not event.source_facts:
        return "source_evidence_missing"
    if event.verification_status not in {"verified", "source_bound"}:
        return "verification_not_source_bound"
    if not event.verification_refs:
        return "verification_reference_missing"
    if not event.context_query_id or not event.need_query_id or not event.slot_id:
        return "query_or_slot_reference_missing"
    if not event.context_vector_ref or not event.need_vector_ref:
        return "vector_reference_missing"
    if not event.target_provenance_refs:
        return "target_provenance_missing"
    if event.reason_for_target(target) is None:
        return "no_recovery_or_costly_reuse_reason"
    return None


def plan_learning_candidates(
    event: RecallLearningEvent,
    anchors: Iterable[LearningAnchor],
    *,
    max_candidates: int = 4,
    max_anchors: int = 4,
) -> LearningCandidatePlan:
    """Create a bounded, side-effect-free learning candidate plan.

    The output is intentionally only a plan.  T13 owns durable cue/edge
    creation, a transaction receipt, and index publication.  Consequently no
    repository, model, vector index, or network call occurs here.
    """

    if not isinstance(event, RecallLearningEvent):
        raise TypeError("event must be a RecallLearningEvent")
    anchor_rows = tuple(anchors)
    contextual = set(event.contextual_expansion_episode_ids)
    anchor_rejections: list[LearningCandidateRejection] = []
    for anchor in anchor_rows:
        if not isinstance(anchor, LearningAnchor):
            raise TypeError("learning anchors must be LearningAnchor instances")
        if anchor.is_contextual:
            anchor_rejections.append(
                LearningCandidateRejection(
                    code="contextual_anchor_not_independent",
                    anchor_id=anchor.anchor_id,
                )
            )
        elif anchor.anchor_type == "episode" and anchor.anchor_id in contextual:
            anchor_rejections.append(
                LearningCandidateRejection(
                    code="anchor_returned_by_contextual_expansion",
                    anchor_id=anchor.anchor_id,
                )
            )
        elif not anchor.is_eligible_base_anchor:
            anchor_rejections.append(
                LearningCandidateRejection(
                    code="anchor_missing_independent_base_provenance",
                    anchor_id=anchor.anchor_id,
                )
            )

    selected_anchors = select_independent_learning_anchors(
        anchor_rows,
        contextual_episode_ids=event.contextual_expansion_episode_ids,
        target_episode_ids=event.target_episode_ids,
        max_anchors=max_anchors,
    )
    candidate_limit = min(4, max(0, int(max_candidates)))
    candidates: list[LearningCandidate] = []
    rejections = list(anchor_rejections)
    for target in sorted(event.target_episode_ids):
        rejection = _event_target_rejection(event, target)
        if rejection is not None:
            rejections.append(
                LearningCandidateRejection(code=rejection, target_episode_id=target)
            )
            continue
        if not selected_anchors:
            rejections.append(
                LearningCandidateRejection(
                    code="no_independent_base_anchor",
                    target_episode_id=target,
                )
            )
            continue
        if len(candidates) >= candidate_limit:
            rejections.append(
                LearningCandidateRejection(
                    code="candidate_limit_reached",
                    target_episode_id=target,
                )
            )
            continue
        reason = event.reason_for_target(target)
        assert reason is not None  # guarded by _event_target_rejection
        for anchor in selected_anchors:
            if len(candidates) >= candidate_limit:
                break
            candidates.append(
                LearningCandidate(
                    candidate_id=_candidate_id(event, anchor, target, reason),
                    anchor_type=anchor.anchor_type,
                    anchor_id=anchor.anchor_id,
                    target_episode_id=target,
                    reason=reason,
                    request_id=event.request_id,
                    request_hash=event.request_hash,
                    source_request_hash=event.source_request_hash,
                    context_query_id=event.context_query_id,
                    need_query_id=event.need_query_id,
                    slot_id=event.slot_id,
                    context_vector_ref=event.context_vector_ref,
                    need_vector_ref=event.need_vector_ref,
                    anchor_vector_ref=anchor.vector_ref,
                    source_facts=(*anchor.source_facts, *event.source_facts),
                    verification_refs=event.verification_refs,
                    verification_status=event.verification_status,
                    anchor_contribution_id=anchor.contribution_id,
                    anchor_provenance_refs=anchor.provenance_refs,
                    target_provenance_refs=event.target_provenance_refs,
                    costly_stage_refs=event.costly_stage_refs,
                )
            )
    # Rejections are diagnostic output too: make them independent of the
    # caller's input ordering and collapse repeated invalid anchor records.
    rejection_by_key = {
        (item.code, item.target_episode_id, item.anchor_id): item
        for item in rejections
    }
    ordered_rejections = tuple(
        rejection_by_key[key]
        for key in sorted(
            rejection_by_key,
            key=lambda key: (key[0], key[1], key[2]),
        )
    )
    return LearningCandidatePlan(
        request_id=event.request_id,
        selected_anchors=selected_anchors,
        candidates=tuple(candidates),
        rejections=ordered_rejections,
    )


def derive_learning_candidates(
    event: RecallLearningEvent,
    anchors: Iterable[LearningAnchor],
    *,
    max_candidates: int = 4,
    max_anchors: int = 4,
) -> list[LearningCandidate]:
    """Compatibility-friendly list view of :func:`plan_learning_candidates`."""

    return list(
        plan_learning_candidates(
            event,
            anchors,
            max_candidates=max_candidates,
            max_anchors=max_anchors,
        ).candidates
    )


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
    # A required loss is never offset by a gain in a different slot.  The
    # legacy observation type has no separate ``mixed`` state, so retain the
    # conservative harmful label for both loss-only and gain-plus-loss cases.
    if lost:
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
    # T13 creation is deliberately owned by MemoryApplication: it is the
    # only layer with both the durable repository and the RAM cue indexes.
    # Keep this optional so historical repository-only callers stay exactly
    # read-only with respect to V3 candidate plans.
    creation_finalizer: Callable[..., object] | None = None

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

    def record_utility_ledger(
        self,
        observations: Iterable[ContextualUtilityLedgerObservation],
    ):
        """Append only independently observed, already-typed V15 utility.

        This deliberately has no conversion from a creation candidate, a
        relevance score, or the old mutable observation shape.  Callers must
        first construct source-bound counterfactual evidence for a later
        recall round.
        """

        values = tuple(observations or ())
        if any(
            not isinstance(value, ContextualUtilityLedgerObservation)
            for value in values
        ):
            raise TypeError("V15 utility ledger requires typed observations")
        if not values:
            return ()
        return tuple(self.repository.record_contextual_utility_ledger(values))

    def apply_event(
        self,
        event,
        *,
        plan: LearningCandidatePlan | None = None,
        cue_materializations: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        """Apply only independent post-creation utility observations.

        A typed V3 recall event may now be handed to the injected application
        facade together with its already-derived plan/materializations.  This
        class never falls back to ``repository.create_contextual`` for that
        path: doing so would bypass receipt atomicity and RAM publication.
        Legacy ``PlasticityEvent`` handling remains unchanged.
        """

        if isinstance(event, RecallLearningEvent):
            if not event.answer_guard_passed:
                return {
                    "created": [],
                    "utility": {"updated": 0},
                    "skipped": "guard",
                }
            if plan is None:
                return {
                    "created": [],
                    "utility": {"updated": 0},
                    "pending_creation": True,
                    "skipped": "plan_required",
                }
            if self.creation_finalizer is None:
                return {
                    "created": [],
                    "utility": {"updated": 0},
                    "pending_creation": bool(plan.candidates),
                    "skipped": "application_finalizer_required",
                }
            try:
                finalized = self.creation_finalizer(
                    event,
                    plan,
                    dict(cue_materializations or {}),
                )
            except Exception:
                # The injected facade owns the DB transaction.  A failure
                # must not be re-routed to this legacy repository API.
                return {
                    "created": [],
                    "utility": {"updated": 0},
                    "skipped": "application_finalizer_failed",
                }
            receipts = (
                finalized.get("receipts", ())
                if isinstance(finalized, Mapping)
                else finalized
            )
            if not isinstance(receipts, (list, tuple)):
                receipts = ()
            return {
                "created": [
                    int(item.get("association_id", 0) or 0)
                    for item in receipts
                    if isinstance(item, Mapping)
                    and int(item.get("association_id", 0) or 0) > 0
                ],
                # A creation receipt is never a utility observation or an
                # independent success, including costly-success reuse.
                "utility": {"updated": 0},
                "receipts": list(receipts),
                "creation_observations_ignored": 0,
            }

        if not bool(getattr(event, "answer_guard_passed", False)):
            return {"created": [], "utility": {"updated": 0}, "skipped": "guard"}
        candidates = tuple(getattr(event, "candidates", ()) or ())
        creation_request_hash = str(getattr(event, "request_hash", "") or "").strip()
        observations = tuple(getattr(event, "observations", ()) or ())
        ledger_observations = tuple(
            getattr(event, "ledger_observations", ()) or ()
        )
        if any(
            not isinstance(value, ContextualUtilityLedgerObservation)
            for value in ledger_observations
        ):
            raise TypeError("V15 utility ledger requires typed observations")
        independent_observations = tuple(
            observation
            for observation in observations
            if not (
                candidates
                and creation_request_hash
                and str(observation.query_hash or "") == creation_request_hash
            )
        )
        ignored_creation_observations = len(observations) - len(independent_observations)
        utility = (
            self.update(independent_observations)
            if independent_observations
            else {"updated": 0}
        )
        # A creation round is never allowed to self-confirm an edge.  Do not
        # attempt to reinterpret a typed ledger row with the same event: it
        # must arrive through a later independent PlasticityEvent instead.
        ledger_receipts = (
            ()
            if candidates
            else self.record_utility_ledger(ledger_observations)
        )
        return {
            "created": [],
            "utility": utility,
            "utility_ledger": list(ledger_receipts),
            "candidate_count": len(candidates),
            "pending_creation": bool(candidates),
            "creation_observations_ignored": ignored_creation_observations,
            "creation_ledger_observations_ignored": (
                len(ledger_observations) if candidates else 0
            ),
        }
