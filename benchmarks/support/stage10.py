from __future__ import annotations

from typing import Any

from memory_demo.association_overlay import AssociationDelta
from benchmarks.support.stage5 import score_evidence_retrieval


def _delta_episode_endpoint_ids(delta: AssociationDelta) -> set[int]:
    endpoint_ids: set[int] = set()
    for change in [*delta.created, *delta.reinforced]:
        row = change.get("after")
        if not isinstance(row, dict):
            continue
        for prefix in ("from", "to"):
            if str(row.get(f"{prefix}_type", "")) != "episode":
                continue
            try:
                endpoint_ids.add(int(row[f"{prefix}_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return endpoint_ids


def positive_actionable_opportunity(
    masked: dict[str, Any],
    delta: AssociationDelta,
    criterion: dict[str, Any],
) -> dict[str, Any]:
    """Identify missing evidence groups that the learned delta can reach.

    A plain recall miss is not an Association opportunity.  It is actionable
    only when a learned/reinforced edge has an Episode endpoint inside that
    missing group, so enabling the delta could in principle fill the slot.
    """
    selected_ids = {int(value) for value in masked.get("episode_ids", [])}
    delta_endpoint_ids = _delta_episode_endpoint_ids(delta)
    missing_group_indexes: list[int] = []
    actionable_group_indexes: list[int] = []
    actionable_episode_ids: set[int] = set()
    for index, raw_group in enumerate(
        criterion.get("required_episode_groups", [])
    ):
        group_ids = {int(value) for value in raw_group}
        if selected_ids.intersection(group_ids):
            continue
        missing_group_indexes.append(index)
        reachable = group_ids.intersection(delta_endpoint_ids)
        if reachable:
            actionable_group_indexes.append(index)
            actionable_episode_ids.update(reachable)
    return {
        "actionable": bool(actionable_group_indexes),
        "missing_group_indexes": missing_group_indexes,
        "actionable_group_indexes": actionable_group_indexes,
        "actionable_episode_ids": sorted(actionable_episode_ids),
        "delta_episode_endpoint_ids": sorted(delta_endpoint_ids),
    }


def hard_negative_diagnostics(
    treatment: dict[str, Any],
    graph: dict[str, Any],
    delta: AssociationDelta,
    criterion: dict[str, Any],
) -> dict[str, Any]:
    """Measure whether a semantically close cue contaminates final evidence."""
    changed_ids = delta.created_ids | set(delta.reinforced_before)
    treatment_cues = {
        int(value) for value in treatment.get("association_cue_ids", [])
    }
    treatment_paths = treatment.get("association_paths", [])
    bridge_ids = {
        int(path["association_id"])
        for path in treatment_paths
        if path.get("learned_bridge_slot_used")
        and "association_id" in path
    }
    target_bridge_ids = sorted(bridge_ids.intersection(changed_ids))
    treatment_ids = {
        int(value) for value in treatment.get("episode_ids", [])
    }
    graph_ids = {int(value) for value in graph.get("episode_ids", [])}
    forbidden_ids = {
        int(value) for value in criterion.get("forbidden_episode_ids", [])
    }
    treatment_only = treatment_ids - graph_ids
    forbidden_intrusions = sorted(treatment_only.intersection(forbidden_ids))
    treatment_score = score_evidence_retrieval(treatment, criterion)
    graph_score = score_evidence_retrieval(graph, criterion)
    matched_delta = (
        int(treatment_score["matched_group_count"])
        - int(graph_score["matched_group_count"])
    )
    return {
        "target_cue_candidate_ids": sorted(
            treatment_cues.intersection(changed_ids)
        ),
        "target_cue_candidate_seen": bool(
            treatment_cues.intersection(changed_ids)
        ),
        "target_bridge_path_ids": target_bridge_ids,
        "target_bridge_path_used": bool(target_bridge_ids),
        "treatment_only_episode_ids": sorted(treatment_only),
        "forbidden_intrusion_episode_ids": forbidden_intrusions,
        "matched_group_delta_s_minus_g": matched_delta,
        "required_recall_non_regression": matched_delta >= 0,
        "answer_contained": bool(matched_delta >= 0 and not forbidden_intrusions),
    }
