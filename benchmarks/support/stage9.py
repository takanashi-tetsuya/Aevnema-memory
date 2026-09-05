from __future__ import annotations

from copy import deepcopy
from typing import Any

from memory_demo.association_overlay import AssociationDelta
from benchmarks.support.stage5 import score_evidence_retrieval
from benchmarks.support.stage8 import replay_bundle_variant


_CUE_CONFIGURATION_KEYS = (
    "association_cue_enabled",
    "association_cue_top_k",
    "association_cue_min_similarity",
    "association_cue_rrf_weight",
)

_CUE_OPTIONAL_CONFIGURATION_KEYS = (
    "association_cue_semantic_gate_enabled",
    "association_cue_semantic_gate_max_selected",
)


def association_cue_replay_variant(
    bundle: dict[str, Any],
    *,
    enabled: bool,
    graph_max_hops: int,
    answer_episode_limit: int,
) -> dict[str, Any]:
    """Toggle only the frozen Association cue lane and graph/budget controls."""
    variant = replay_bundle_variant(
        bundle,
        graph_max_hops=graph_max_hops,
        answer_episode_limit=answer_episode_limit,
    )
    configuration = variant["configuration"]
    if enabled:
        source = bundle.get("configuration", {})
        for key in _CUE_CONFIGURATION_KEYS:
            if key not in source:
                raise ValueError(f"cue-enabled bundle is missing {key}")
            configuration[key] = source[key]
        for key in _CUE_OPTIONAL_CONFIGURATION_KEYS:
            if key in source:
                configuration[key] = source[key]
            else:
                configuration.pop(key, None)
    else:
        for key in (*_CUE_CONFIGURATION_KEYS, *_CUE_OPTIONAL_CONFIGURATION_KEYS):
            configuration.pop(key, None)
        variant["final_seed_hits"] = deepcopy(
            variant.get("base_final_seed_hits", variant["final_seed_hits"])
        )
    return variant


def association_cue_utility_diagnostics(
    treatment: dict[str, Any],
    masked: dict[str, Any],
    delta: AssociationDelta,
    criterion: dict[str, Any],
) -> dict[str, Any]:
    changed_ids = delta.created_ids | set(delta.reinforced_before)
    treatment_cues = {
        int(value) for value in treatment.get("association_cue_ids", [])
    }
    masked_cues = {
        int(value) for value in masked.get("association_cue_ids", [])
    }
    used_delta_cues = sorted(treatment_cues.intersection(changed_ids))
    treatment_score = score_evidence_retrieval(treatment, criterion)
    masked_score = score_evidence_retrieval(masked, criterion)
    treatment_ids = {int(value) for value in treatment.get("episode_ids", [])}
    masked_ids = {int(value) for value in masked.get("episode_ids", [])}
    matched_delta = (
        int(treatment_score["matched_group_count"])
        - int(masked_score["matched_group_count"])
    )
    return {
        "learned_cue_used": bool(used_delta_cues),
        "learned_cue_ids": used_delta_cues,
        "treatment_cue_ids": sorted(treatment_cues),
        "masked_cue_ids": sorted(masked_cues),
        "matched_group_delta_s_minus_sm": matched_delta,
        "selected_recall_delta_s_minus_sm": (
            float(treatment_score["recall_at_30"])
            - float(masked_score["recall_at_30"])
        ),
        "treatment_only_episode_ids": sorted(treatment_ids - masked_ids),
        "masked_only_episode_ids": sorted(masked_ids - treatment_ids),
        "causal_utility_observed": bool(used_delta_cues and matched_delta > 0),
    }
