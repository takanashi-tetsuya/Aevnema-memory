from __future__ import annotations

from collections import Counter
from copy import deepcopy
from statistics import mean
from typing import Any

from memory_demo.association_overlay import AssociationDelta
from benchmarks.support.stage5 import score_evidence_retrieval


def replay_bundle_variant(
    bundle: dict[str, Any],
    *,
    graph_max_hops: int,
    answer_episode_limit: int,
) -> dict[str, Any]:
    """Return a replay-compatible budget/graph ablation of one frozen plan.

    Query planning, embeddings, dense/sparse rankings and the LLM rerank are kept
    byte-for-byte identical.  Only deterministic graph traversal and final
    evidence capacity may change.
    """
    if graph_max_hops < 0:
        raise ValueError("graph_max_hops must be non-negative")
    if answer_episode_limit <= 0:
        raise ValueError("answer_episode_limit must be positive")
    variant = deepcopy(bundle)
    configuration = variant.setdefault("configuration", {})
    configuration["graph_max_hops"] = int(graph_max_hops)
    configuration["answer_episode_limit"] = int(answer_episode_limit)
    # Backward-compatible upgrade for pilot bundles frozen before the learned
    # bridge lane became part of replay configuration.
    configuration.setdefault("learned_bridge_slots", 2)
    configuration.setdefault("learned_bridge_min_query_relevance", 0.03)
    configuration.setdefault("learned_bridge_duplicate_threshold", 0.05)
    return variant


def hard_group_score(result: dict[str, Any], episode_ids: list[int]) -> dict[str, Any]:
    allowed = {int(value) for value in episode_ids}
    selected = [
        int(value) for value in result.get("episode_ids", [])
        if int(value) in allowed
    ]
    candidates = [
        int(value) for value in result.get("candidate_episode_ids", [])
        if int(value) in allowed
    ]
    return {
        "alternatives": sorted(allowed),
        "selected": bool(selected),
        "selected_matches": selected,
        "candidate": bool(candidates),
        "candidate_matches": candidates,
    }


def association_utility_diagnostics(
    treatment: dict[str, Any],
    masked: dict[str, Any],
    delta: AssociationDelta,
    criterion: dict[str, Any],
    hard_group: list[int],
) -> dict[str, Any]:
    """Explain whether learned rows caused useful evidence-set changes.

    Edge counts are deliberately absent from the utility verdict.  A learned
    edge is useful only when it is traversed and masking it reduces registered
    evidence coverage or the pre-registered hard group.
    """
    changed_ids = delta.created_ids | set(delta.reinforced_before)
    treatment_paths = [
        dict(path)
        for path in treatment.get("association_paths", [])
        if int(path.get("association_id", -1)) in changed_ids
    ]
    treatment_score = score_evidence_retrieval(treatment, criterion)
    masked_score = score_evidence_retrieval(masked, criterion)
    treatment_hard = hard_group_score(treatment, hard_group)
    masked_hard = hard_group_score(masked, hard_group)
    treatment_ids = {int(value) for value in treatment.get("episode_ids", [])}
    masked_ids = {int(value) for value in masked.get("episode_ids", [])}
    recall_delta = (
        float(treatment_score["recall_at_30"])
        - float(masked_score["recall_at_30"])
    )
    hard_group_gain = treatment_hard["selected"] and not masked_hard["selected"]
    return {
        "learned_path_used": bool(treatment_paths),
        "learned_path_ids": sorted(
            {int(path["association_id"]) for path in treatment_paths}
        ),
        "learned_paths": treatment_paths,
        "selected_recall_delta_t_minus_m": recall_delta,
        "matched_group_delta_t_minus_m": (
            int(treatment_score["matched_group_count"])
            - int(masked_score["matched_group_count"])
        ),
        "hard_group_treatment": treatment_hard,
        "hard_group_masked": masked_hard,
        "hard_group_gain": hard_group_gain,
        "treatment_only_episode_ids": sorted(treatment_ids - masked_ids),
        "masked_only_episode_ids": sorted(masked_ids - treatment_ids),
        "causal_utility_observed": bool(treatment_paths and recall_delta > 0.0),
    }


def probation_promotion_decision(
    diagnostic: dict[str, Any],
    delta: AssociationDelta,
) -> dict[str, Any]:
    """Decide which cross-query probation rows have earned persistence.

    A row proposed while answering Q1 is only a *candidate memory*.  It earns
    promotion after a separate Q2 when masking the Q1 delta removes the
    preregistered hard evidence group and lowers registered evidence recall.
    Rows that were not actually traversed by Q2 are never promoted.
    """
    candidate_ids = sorted(delta.created_ids | set(delta.reinforced_before))
    used_ids = sorted(
        {
            int(value)
            for key in ("learned_path_ids", "learned_cue_ids")
            for value in diagnostic.get(key, [])
        }.intersection(candidate_ids)
    )
    utility_observed = bool(diagnostic.get("causal_utility_observed"))
    hard_group_gain = bool(diagnostic.get("hard_group_gain"))
    promote_ids = used_ids if utility_observed and hard_group_gain else []
    if promote_ids:
        reason = "q2_masking_removed_hard_group_and_reduced_registered_recall"
    elif not used_ids:
        reason = "q2_did_not_traverse_probation_rows"
    elif not hard_group_gain:
        reason = "q2_hard_group_did_not_depend_on_probation_rows"
    else:
        reason = "q2_registered_recall_did_not_improve"
    return {
        "policy": "cross_query_probation_v1",
        "candidate_association_ids": candidate_ids,
        "q2_used_candidate_ids": used_ids,
        "promote_association_ids": promote_ids,
        "reject_association_ids": sorted(set(candidate_ids) - set(promote_ids)),
        "promotion_earned": bool(promote_ids),
        "reason": reason,
    }


def minimal_probation_promotion_decision(
    delta: AssociationDelta,
    *,
    individually_sufficient_ids: list[int],
    individually_necessary_ids: list[int],
) -> dict[str, Any]:
    """Choose the smallest evidence-safe promotion after per-row ablation.

    Necessary rows are retained together.  When several rows are redundant
    substitutes and each is sufficient alone, retain only the highest-quality
    direct row so one successful batch cannot promote a cloud of duplicates.
    """
    changed_rows = {
        int(item["id"]): dict(item.get("after") or {})
        for item in [*delta.created, *delta.reinforced]
    }
    changed_ids = set(changed_rows)
    sufficient = sorted(
        changed_ids.intersection(int(value) for value in individually_sufficient_ids)
    )
    necessary = sorted(
        changed_ids.intersection(int(value) for value in individually_necessary_ids)
    )

    def quality(association_id: int) -> tuple[float, float, float, int]:
        row = changed_rows[association_id]
        generation = max(0, int(row.get("generation", 0)))
        return (
            -float(generation),
            float(row.get("confidence", 0.0)),
            float(row.get("weight", 0.0)),
            -association_id,
        )

    if necessary:
        promoted = necessary
        reason = "leave_one_out_identified_necessary_rows"
    elif sufficient:
        promoted = [max(sufficient, key=quality)]
        reason = "retained_best_single_sufficient_row_and_rejected_redundancy"
    else:
        promoted = []
        reason = "no_individual_row_proved_cross_query_utility"
    return {
        "policy": "cross_query_probation_minimal_v2",
        "candidate_association_ids": sorted(changed_ids),
        "individually_sufficient_ids": sufficient,
        "individually_necessary_ids": necessary,
        "promote_association_ids": promoted,
        "reject_association_ids": sorted(changed_ids.difference(promoted)),
        "promotion_earned": bool(promoted),
        "reason": reason,
    }


def normalize_answer_pair_judgment(
    payload: Any,
    *,
    label_to_arm: dict[str, str],
) -> dict[str, Any]:
    """Normalize one blind A/B judgment without trusting model arithmetic."""
    value = payload if isinstance(payload, dict) else {}
    raw_scores = value.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for label in ("A", "B"):
        raw_dimensions = raw_scores.get(label, {})
        if not isinstance(raw_dimensions, dict):
            raw_dimensions = {}
        dimensions: dict[str, int] = {}
        for key in (
            "evidence_completeness",
            "reasoning_coherence",
            "fact_boundary",
            "uncertainty_calibration",
        ):
            try:
                score = int(raw_dimensions.get(key, 0))
            except (TypeError, ValueError):
                score = 0
            dimensions[key] = max(0, min(4, score))
        scores[label] = dimensions
        totals[label] = sum(dimensions.values())
    if totals["A"] > totals["B"]:
        winning_label = "A"
    elif totals["B"] > totals["A"]:
        winning_label = "B"
    else:
        winning_label = "tie"
    winning_arm = label_to_arm.get(winning_label, "tie")
    arm_totals = {
        label_to_arm[label]: totals[label]
        for label in ("A", "B")
        if label in label_to_arm
    }
    return {
        "scores": scores,
        "totals": totals,
        "arm_totals": arm_totals,
        "winning_label": winning_label,
        "winning_arm": winning_arm,
        "score_delta_t2_minus_m2": (
            arm_totals.get("T2", 0) - arm_totals.get("M2", 0)
        ),
        "reason": str(value.get("reason", "")).strip(),
        "unsupported_claims": value.get("unsupported_claims", {}),
    }


def summarize_answer_pair_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    judgments = [
        judgment
        for trial in trials
        for judgment in trial.get("judgments", [])
        if isinstance(judgment, dict)
    ]
    deltas = [
        int(judgment.get("score_delta_t2_minus_m2", 0))
        for judgment in judgments
    ]
    winners = Counter(
        str(judgment.get("winning_arm", "tie")) for judgment in judgments
    )
    trial_votes: list[str] = []
    for trial in trials:
        votes = Counter(
            str(item.get("winning_arm", "tie"))
            for item in trial.get("judgments", [])
            if isinstance(item, dict)
        )
        if votes["T2"] > votes["M2"]:
            trial_votes.append("T2")
        elif votes["M2"] > votes["T2"]:
            trial_votes.append("M2")
        else:
            trial_votes.append("tie")
    return {
        "trials": len(trials),
        "judgments": len(judgments),
        "judge_wins": {
            "T2": winners["T2"],
            "M2": winners["M2"],
            "tie": winners["tie"],
        },
        "trial_wins": {
            "T2": trial_votes.count("T2"),
            "M2": trial_votes.count("M2"),
            "tie": trial_votes.count("tie"),
        },
        "score_deltas_t2_minus_m2": deltas,
        "mean_score_delta_t2_minus_m2": mean(deltas) if deltas else None,
    }
