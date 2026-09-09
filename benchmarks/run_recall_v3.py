from __future__ import annotations

"""Offline, fail-closed preflight for a v3 recall experiment.

This module intentionally *does not* import the application, a model client,
``requests``, or ``sqlite3``.  Its in-memory JSON preflight only reads the
documents supplied by the caller and may write one JSON report.  The formal CLI
also invokes the evaluator-only Source-gold validator against a frozen local
Source root; neither path can open a network connection, invoke a model, or
write a knowledge-base database.

Minimal run-spec contract (``aevnema.recall-v3.run-spec.v1``)::

    {
      "schema": "aevnema.recall-v3.run-spec.v1",
      "safety": {
        "network": "forbidden",
        "model_calls": "forbidden",
        "database_writes": "forbidden"
      },
      "arms": [{
        "arm_id": "contextual-input-frozen",
        "primary_score": true,
        "evaluation_role": "treatment",
        "requires_contextual": true,
        "replay_mode": "input_frozen",
        "frozen_input": {"artifact_id": "request-set-v1"},
        "execution_modules": {"matcher": true, "selector": true},
        "snapshot": {
          "database_sha256": "sha256:...",
          "source_manifest_sha256": "sha256:...",
          "scope_sha256": "sha256:...",
          "embedding_space": {
            "id": "bge-m3/normalized-v1", "model_id": "bge-m3",
            "preprocess_version": "v1", "dimension": 1024,
            "normalized": true
          }
        },
        "contextual": {
          "matcher": {"enabled": true, "implementation": "v3-matcher"},
          "cues": [{"cue_id": "cue-1", "vector_binding_id": "cue-v"}],
          "edges": [{"edge_id": "edge-1", "cue_id": "cue-1",
                     "target_ref": "episode:1"}],
          "query_refs": [{"query_ref_id": "whole", "role": "whole",
                          "vector_binding_id": "query-v"}],
          "vector_bindings": [
            {"binding_id": "cue-v", "embedding_space_id": "bge-m3/normalized-v1",
             "dimension": 1024, "normalized": true},
            {"binding_id": "query-v", "embedding_space_id": "bge-m3/normalized-v1",
             "dimension": 1024, "normalized": true}
          ]
        },
        "candidate_inventory": {"state": "known_empty", "count": 0}
      }]
    }

``known_empty`` is deliberately a valid outcome: it says an otherwise complete
mechanism has no candidates for this arm.  It is not confused with an empty
contextual mechanism (no matcher/cue/edge/query-vector closure), which is an
``invalid_setup`` before a paid or mutable stage can start.

The gold and split files are evaluator-only inputs.  The pure
``preflight_run()`` API checks their JSON readiness only, which keeps it useful
for synthetic unit tests.  The formal CLI additionally requires a frozen
Source root and a frozen source manifest, then fails closed unless every
approved atom physically resolves to its declared Source span.  This rejects
the 2026-09-05 draft by design.
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from benchmarks.validate_source_gold import (
    SOURCE_GOLD_VALIDATION_SCHEMA,
    _direct_local_path_preflight,
    _report_output_preflight,
    _safe_report_output_path,
    validate_source_gold_bytes,
)


RUN_SPEC_SCHEMA = "aevnema.recall-v3.run-spec.v1"
PREFLIGHT_SCHEMA = "aevnema.recall-v3.preflight.v1"
REPLAY_RESULT_SCHEMA = "aevnema.recall-v3.replay-result.v1"
FROZEN_INPUT_SCHEMA = "aevnema.recall-v3.input-frozen.v1"
REQUEST_RECORD_SCHEMA = "aevnema.recall-v3.request-record.v1"
REQUEST_MANIFEST_SCHEMA = "aevnema.recall-v3.request-manifest.v1"
VECTOR_MANIFEST_SCHEMA = "aevnema.recall-v3.vector-manifest.v1"
DATABASE_MANIFEST_SCHEMA = "aevnema.recall-v3.database-manifest.v1"
_APPROVED_STATUSES = {
    "accepted",
    "accepted_for_scoring",
    "approved",
    "approved_for_scoring",
}
_DRAFT_STATUSES = {
    "draft",
    "draft_pending_source_span_review",
    "pending_source_span_review",
}
_REPLAY_MODES = {"live", "input_frozen", "candidate_frozen"}
_FROZEN_REPLAY_MODES = {"input_frozen", "candidate_frozen"}
_CANDIDATE_STATES = {"not_materialized", "known_empty", "nonempty"}


class PreflightInputError(ValueError):
    """Raised when a JSON input cannot be safely parsed as a JSON document."""


def _issue(
    code: str,
    message: str,
    *,
    path: str,
    scope: str = "setup",
) -> dict[str, str]:
    return {"scope": scope, "code": code, "path": path, "message": message}


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: object) -> bool:
    # bool is an int subclass but is never a valid dimension/count here.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalise_modules(value: object) -> dict[str, bool]:
    """Return a compact module bitmap without treating unknown values as true."""
    if not _is_mapping(value):
        return {}
    result: dict[str, bool] = {}
    for name, setting in value.items():
        if isinstance(setting, bool):
            result[str(name)] = setting
        elif _is_mapping(setting):
            enabled = setting.get("enabled")
            result[str(name)] = enabled is True
        else:
            result[str(name)] = False
    return result


def _iter_nodes(value: object, path: str = "$") -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield every JSON object with a JSONPath-like location."""
    if _is_mapping(value):
        typed = dict(value)
        yield path, typed
        for key, child in typed.items():
            yield from _iter_nodes(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_nodes(child, f"{path}[{index}]")


def _json_bytes(document: object) -> bytes:
    """Stable document encoding used only for local input identity reporting."""
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _document_digest(document: object) -> str:
    return "sha256:" + sha256(_json_bytes(document)).hexdigest()


def _document_status(document: object) -> object:
    if not _is_mapping(document):
        return None
    return document.get("status")


def _validate_gold(gold: object) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate source-gold readiness without exposing question/evidence text."""
    issues: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "status": _document_status(gold),
        "eligible_for_scoring": False,
        "family_ids": [],
        "digest": _document_digest(gold),
    }
    if not _is_mapping(gold):
        issues.append(
            _issue("gold_not_object", "gold must be a JSON object", path="gold")
        )
        return issues, summary

    status = gold.get("status")
    if status in _DRAFT_STATUSES:
        issues.append(
            _issue(
                "gold_draft_not_accepted",
                "draft or pending Source gold cannot score a v3 experiment",
                path="gold.status",
                scope="gold",
            )
        )
    elif status not in _APPROVED_STATUSES:
        issues.append(
            _issue(
                "gold_not_approved",
                "gold.status must explicitly approve scoring",
                path="gold.status",
                scope="gold",
            )
        )

    families = gold.get("families")
    if not isinstance(families, list) or not families:
        issues.append(
            _issue(
                "gold_missing_families",
                "gold must contain at least one reviewed family",
                path="gold.families",
                scope="gold",
            )
        )
        return issues, summary

    family_ids: list[str] = []
    for index, family in enumerate(families):
        family_path = f"gold.families[{index}]"
        if not _is_mapping(family) or not _is_nonempty_string(family.get("family_id")):
            issues.append(
                _issue(
                    "gold_family_id_missing",
                    "each gold family needs a non-empty family_id",
                    path=f"{family_path}.family_id",
                    scope="gold",
                )
            )
            continue
        family_ids.append(str(family["family_id"]))

    if len(family_ids) != len(set(family_ids)):
        issues.append(
            _issue(
                "gold_family_id_duplicate",
                "gold family_id values must be unique",
                path="gold.families",
                scope="gold",
            )
        )

    # A false flag anywhere in a source-gold document is an explicit refusal
    # to score.  A missing flag on a scoring node is also fail-closed below.
    false_paths = [
        path
        for path, node in _iter_nodes(gold, "gold")
        if node.get("usable_for_scoring") is False
    ]
    for path in false_paths:
        issues.append(
            _issue(
                "gold_node_not_usable_for_scoring",
                "a source-gold node explicitly forbids scoring",
                path=f"{path}.usable_for_scoring",
                scope="gold",
            )
        )

    # The source-gold builder uses these node names.  Requiring explicit true
    # values makes it impossible to accidentally grade a partly reviewed set.
    scoring_nodes: list[tuple[str, Mapping[str, Any]]] = [("gold", dict(gold))]
    for family_index, family in enumerate(families):
        if not _is_mapping(family):
            continue
        family_path = f"gold.families[{family_index}]"
        scoring_nodes.append((family_path, dict(family)))
        groups = family.get("claim_groups")
        if not isinstance(groups, list) or not groups:
            issues.append(
                _issue(
                    "gold_missing_claim_groups",
                    "each family needs reviewed claim_groups",
                    path=f"{family_path}.claim_groups",
                    scope="gold",
                )
            )
            continue
        for group_index, group in enumerate(groups):
            if not _is_mapping(group):
                issues.append(
                    _issue(
                        "gold_claim_group_not_object",
                        "each claim group must be an object",
                        path=f"{family_path}.claim_groups[{group_index}]",
                        scope="gold",
                    )
                )
                continue
            group_path = f"{family_path}.claim_groups[{group_index}]"
            scoring_nodes.append((group_path, dict(group)))
            atoms = group.get("evidence_atoms")
            if not isinstance(atoms, list) or not atoms:
                issues.append(
                    _issue(
                        "gold_missing_evidence_atoms",
                        "each claim group needs source evidence atoms",
                        path=f"{group_path}.evidence_atoms",
                        scope="gold",
                    )
                )
                continue
            for atom_index, atom in enumerate(atoms):
                if not _is_mapping(atom):
                    issues.append(
                        _issue(
                            "gold_evidence_atom_not_object",
                            "each evidence atom must be an object",
                            path=f"{group_path}.evidence_atoms[{atom_index}]",
                            scope="gold",
                        )
                    )
                    continue
                scoring_nodes.append(
                    (f"{group_path}.evidence_atoms[{atom_index}]", dict(atom))
                )

    for path, node in scoring_nodes:
        if node.get("usable_for_scoring") is not True:
            issues.append(
                _issue(
                    "gold_usability_not_explicit",
                    "every scoring node must explicitly set usable_for_scoring to true",
                    path=f"{path}.usable_for_scoring",
                    scope="gold",
                )
            )

    summary["family_ids"] = sorted(set(family_ids))
    summary["eligible_for_scoring"] = not issues
    return issues, summary


def _validate_split(
    split: object, gold_family_ids: Sequence[str]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate split closure against the reviewed source-gold family set."""
    issues: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "status": _document_status(split),
        "eligible_for_scoring": False,
        "assigned_family_ids": [],
        "digest": _document_digest(split),
    }
    if not _is_mapping(split):
        issues.append(
            _issue("split_not_object", "split must be a JSON object", path="split")
        )
        return issues, summary

    status = split.get("status")
    if status in _DRAFT_STATUSES:
        issues.append(
            _issue(
                "split_draft_not_accepted",
                "draft or pending split cannot score a v3 experiment",
                path="split.status",
                scope="split",
            )
        )
    elif status not in _APPROVED_STATUSES:
        issues.append(
            _issue(
                "split_not_approved",
                "split.status must explicitly approve scoring",
                path="split.status",
                scope="split",
            )
        )

    assignments = split.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        issues.append(
            _issue(
                "split_missing_assignments",
                "split must assign every scored gold family",
                path="split.assignments",
                scope="split",
            )
        )
        return issues, summary

    assigned: list[str] = []
    for index, assignment in enumerate(assignments):
        path = f"split.assignments[{index}]"
        if not _is_mapping(assignment) or not _is_nonempty_string(
            assignment.get("family_id")
        ):
            issues.append(
                _issue(
                    "split_family_id_missing",
                    "each split assignment needs a non-empty family_id",
                    path=f"{path}.family_id",
                    scope="split",
                )
            )
            continue
        if not _is_nonempty_string(assignment.get("split")):
            issues.append(
                _issue(
                    "split_label_missing",
                    "each split assignment needs a non-empty split label",
                    path=f"{path}.split",
                    scope="split",
                )
            )
        assigned.append(str(assignment["family_id"]))

    if len(assigned) != len(set(assigned)):
        issues.append(
            _issue(
                "split_family_id_duplicate",
                "a family may appear in exactly one split assignment",
                path="split.assignments",
                scope="split",
            )
        )
    missing = sorted(set(gold_family_ids) - set(assigned))
    unexpected = sorted(set(assigned) - set(gold_family_ids))
    if missing:
        issues.append(
            _issue(
                "split_missing_gold_family",
                "reviewed gold families are missing from the split",
                path="split.assignments",
                scope="split",
            )
        )
    if unexpected:
        issues.append(
            _issue(
                "split_unknown_gold_family",
                "split references a family absent from gold",
                path="split.assignments",
                scope="split",
            )
        )

    summary["assigned_family_ids"] = sorted(set(assigned))
    summary["eligible_for_scoring"] = not issues
    return issues, summary


def _validate_safety(spec: object) -> list[dict[str, str]]:
    if not _is_mapping(spec):
        return [_issue("spec_not_object", "run spec must be a JSON object", path="spec")]
    safety = spec.get("safety")
    if not _is_mapping(safety):
        return [
            _issue(
                "safety_policy_missing",
                "run spec must explicitly declare local-only safety policy",
                path="spec.safety",
            )
        ]
    issues: list[dict[str, str]] = []
    for key in ("network", "model_calls", "database_writes"):
        if safety.get(key) != "forbidden":
            issues.append(
                _issue(
                    f"{key}_not_forbidden",
                    f"spec.safety.{key} must be 'forbidden' for local preflight",
                    path=f"spec.safety.{key}",
                )
            )
    return issues


def _mapping_items(
    closure: Mapping[str, Any], key: str, *, path: str, issues: list[dict[str, str]], code: str
) -> list[Mapping[str, Any]]:
    value = closure.get(key)
    if not isinstance(value, list) or not value:
        issues.append(
            _issue(
                code,
                f"contextual closure needs a non-empty {key} list",
                path=f"{path}.{key}",
                scope="arm",
            )
        )
        return []
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not _is_mapping(item):
            issues.append(
                _issue(
                    f"{code}_invalid_entry",
                    f"every {key} entry must be an object",
                    path=f"{path}.{key}[{index}]",
                    scope="arm",
                )
            )
            continue
        result.append(dict(item))
    return result


def _validate_contextual_closure(
    arm: Mapping[str, Any], *, arm_path: str, issues: list[dict[str, str]]
) -> dict[str, bool]:
    """Check a declared v3 contextual matcher closure, without running it."""
    modules = _normalise_modules(arm.get("execution_modules"))
    if modules.get("matcher") is not True:
        issues.append(
            _issue(
                "matcher_unavailable",
                "a contextual arm must declare that the v3 matcher will run",
                path=f"{arm_path}.execution_modules.matcher",
                scope="arm",
            )
        )
    if modules.get("selector") is not True:
        issues.append(
            _issue(
                "selector_unavailable",
                "a contextual arm must declare that the selector will run",
                path=f"{arm_path}.execution_modules.selector",
                scope="arm",
            )
        )

    snapshot = arm.get("snapshot")
    if not _is_mapping(snapshot):
        issues.append(
            _issue(
                "snapshot_provenance_missing",
                "contextual arm needs a snapshot provenance object",
                path=f"{arm_path}.snapshot",
                scope="arm",
            )
        )
        snapshot = {}
    snapshot = dict(snapshot)
    for field in ("database_sha256", "source_manifest_sha256", "scope_sha256"):
        if not _is_nonempty_string(snapshot.get(field)):
            issues.append(
                _issue(
                    "snapshot_provenance_missing",
                    f"contextual arm snapshot needs {field}",
                    path=f"{arm_path}.snapshot.{field}",
                    scope="arm",
                )
            )
    embedding_space = snapshot.get("embedding_space")
    if not _is_mapping(embedding_space):
        issues.append(
            _issue(
                "embedding_space_missing",
                "contextual arm snapshot needs embedding-space provenance",
                path=f"{arm_path}.snapshot.embedding_space",
                scope="arm",
            )
        )
        embedding_space = {}
    embedding_space = dict(embedding_space)
    for field in ("id", "model_id", "preprocess_version"):
        if not _is_nonempty_string(embedding_space.get(field)):
            issues.append(
                _issue(
                    "embedding_space_missing",
                    f"embedding_space needs {field}",
                    path=f"{arm_path}.snapshot.embedding_space.{field}",
                    scope="arm",
                )
            )
    if not _is_positive_int(embedding_space.get("dimension")):
        issues.append(
            _issue(
                "embedding_dimension_invalid",
                "embedding_space.dimension must be a positive integer",
                path=f"{arm_path}.snapshot.embedding_space.dimension",
                scope="arm",
            )
        )
    if embedding_space.get("normalized") is not True:
        issues.append(
            _issue(
                "embedding_normalization_missing",
                "embedding_space.normalized must be true",
                path=f"{arm_path}.snapshot.embedding_space.normalized",
                scope="arm",
            )
        )

    closure = arm.get("contextual")
    if not _is_mapping(closure):
        issues.append(
            _issue(
                "contextual_closure_missing",
                "requires_contextual=true needs a contextual closure declaration",
                path=f"{arm_path}.contextual",
                scope="arm",
            )
        )
        return modules
    closure = dict(closure)
    matcher = closure.get("matcher")
    if not _is_mapping(matcher) or matcher.get("enabled") is not True or not _is_nonempty_string(
        matcher.get("implementation")
    ):
        issues.append(
            _issue(
                "contextual_disabled",
                "contextual.matcher must be enabled and identify its implementation",
                path=f"{arm_path}.contextual.matcher",
                scope="arm",
            )
        )

    cues = _mapping_items(
        closure,
        "cues",
        path=f"{arm_path}.contextual",
        issues=issues,
        code="missing_cue_prototype",
    )
    edges = _mapping_items(
        closure,
        "edges",
        path=f"{arm_path}.contextual",
        issues=issues,
        code="no_eligible_contextual_edges",
    )
    query_refs = _mapping_items(
        closure,
        "query_refs",
        path=f"{arm_path}.contextual",
        issues=issues,
        code="missing_query_ref",
    )
    vector_bindings = _mapping_items(
        closure,
        "vector_bindings",
        path=f"{arm_path}.contextual",
        issues=issues,
        code="missing_vector_binding",
    )

    binding_ids: set[str] = set()
    expected_space = embedding_space.get("id")
    expected_dimension = embedding_space.get("dimension")
    for index, binding in enumerate(vector_bindings):
        path = f"{arm_path}.contextual.vector_bindings[{index}]"
        binding_id = binding.get("binding_id")
        if not _is_nonempty_string(binding_id):
            issues.append(
                _issue(
                    "vector_binding_id_missing",
                    "a vector binding needs binding_id",
                    path=f"{path}.binding_id",
                    scope="arm",
                )
            )
            continue
        binding_ids.add(str(binding_id))
        if binding.get("embedding_space_id") != expected_space:
            issues.append(
                _issue(
                    "cue_dimension_model_mismatch",
                    "vector binding embedding space must equal snapshot embedding space",
                    path=f"{path}.embedding_space_id",
                    scope="arm",
                )
            )
        if binding.get("dimension") != expected_dimension:
            issues.append(
                _issue(
                    "cue_dimension_model_mismatch",
                    "vector binding dimension must equal snapshot embedding dimension",
                    path=f"{path}.dimension",
                    scope="arm",
                )
            )
        if binding.get("normalized") is not True:
            issues.append(
                _issue(
                    "vector_binding_not_normalized",
                    "every vector binding must be explicitly normalized",
                    path=f"{path}.normalized",
                    scope="arm",
                )
            )

    cue_ids: set[str] = set()
    for index, cue in enumerate(cues):
        path = f"{arm_path}.contextual.cues[{index}]"
        cue_id = cue.get("cue_id")
        if not _is_nonempty_string(cue_id):
            issues.append(
                _issue(
                    "cue_id_missing",
                    "a cue needs cue_id",
                    path=f"{path}.cue_id",
                    scope="arm",
                )
            )
        else:
            cue_ids.add(str(cue_id))
        if cue.get("vector_binding_id") not in binding_ids:
            issues.append(
                _issue(
                    "cue_vector_binding_missing",
                    "cue must refer to a declared vector binding",
                    path=f"{path}.vector_binding_id",
                    scope="arm",
                )
            )
        _check_provenance_closure(
            cue,
            snapshot,
            path=path,
            issues=issues,
        )

    for index, edge in enumerate(edges):
        path = f"{arm_path}.contextual.edges[{index}]"
        if not _is_nonempty_string(edge.get("edge_id")):
            issues.append(
                _issue(
                    "edge_id_missing",
                    "an edge needs edge_id",
                    path=f"{path}.edge_id",
                    scope="arm",
                )
            )
        if edge.get("cue_id") not in cue_ids:
            issues.append(
                _issue(
                    "edge_cue_closure_missing",
                    "edge must refer to a declared cue",
                    path=f"{path}.cue_id",
                    scope="arm",
                )
            )
        if not _is_nonempty_string(edge.get("target_ref")):
            issues.append(
                _issue(
                    "edge_target_missing",
                    "edge must identify a target reference",
                    path=f"{path}.target_ref",
                    scope="arm",
                )
            )
        _check_provenance_closure(
            edge,
            snapshot,
            path=path,
            issues=issues,
        )

    has_whole_query = False
    for index, query_ref in enumerate(query_refs):
        path = f"{arm_path}.contextual.query_refs[{index}]"
        if not _is_nonempty_string(query_ref.get("query_ref_id")):
            issues.append(
                _issue(
                    "query_ref_id_missing",
                    "query ref needs query_ref_id",
                    path=f"{path}.query_ref_id",
                    scope="arm",
                )
            )
        if query_ref.get("role") == "whole":
            has_whole_query = True
        if query_ref.get("vector_binding_id") not in binding_ids:
            issues.append(
                _issue(
                    "missing_query_ref_vector_binding",
                    "query ref must refer to a declared vector binding",
                    path=f"{path}.vector_binding_id",
                    scope="arm",
                )
            )
        _check_provenance_closure(
            query_ref,
            snapshot,
            path=path,
            issues=issues,
        )
    if not has_whole_query:
        issues.append(
            _issue(
                "no_whole_vector",
                "contextual closure needs a whole-query vector reference",
                path=f"{arm_path}.contextual.query_refs",
                scope="arm",
            )
        )
    return modules


def _check_provenance_closure(
    item: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    path: str,
    issues: list[dict[str, str]],
) -> None:
    """Require cue/edge/query declarations to belong to the same snapshot/scope."""
    for key in ("source_manifest_sha256", "scope_sha256"):
        expected = snapshot.get(key)
        actual = item.get(key)
        if actual is None:
            # The arm-level snapshot is the default provenance declaration.  It
            # is sufficient unless an item tries to name a different one.
            continue
        if actual != expected:
            issues.append(
                _issue(
                    "contextual_provenance_mismatch",
                    f"{key} must match the arm snapshot",
                    path=f"{path}.{key}",
                    scope="arm",
                )
            )


def _legacy_markers(arm: Mapping[str, Any]) -> list[str]:
    """Find known legacy replay declarations without scanning free-form text."""
    markers: list[str] = []
    if arm.get("frozen_plan") not in (None, False, ""):
        markers.append("frozen_plan")
    if arm.get("legacy_pipeline_replay") is True:
        markers.append("legacy_pipeline_replay")
    replay_mode = arm.get("replay_mode")
    if replay_mode in {"legacy_pipeline_replay", "frozen_plan"}:
        markers.append(str(replay_mode))
    replay = arm.get("replay")
    if _is_mapping(replay):
        mode = replay.get("mode")
        if mode in {"legacy_pipeline_replay", "frozen_plan"}:
            markers.append(str(mode))
        if replay.get("frozen_plan") not in (None, False, ""):
            markers.append("replay.frozen_plan")
    return sorted(set(markers))


def _validate_arm(arm: object, index: int) -> dict[str, Any]:
    path = f"spec.arms[{index}]"
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not _is_mapping(arm):
        return {
            "arm_id": f"arm-{index}",
            "status": "invalid_setup",
            "execution_state": "not_executable",
            "issues": [
                _issue(
                    "arm_not_object", "every arm must be a JSON object", path=path, scope="arm"
                )
            ],
            "warnings": [],
            "module_bitmap": {},
            "candidate_state": None,
            "primary_score": None,
            "requires_contextual": None,
            "replay_mode": None,
        }
    arm = dict(arm)
    arm_id = arm.get("arm_id")
    if not _is_nonempty_string(arm_id):
        issues.append(
            _issue(
                "arm_id_missing",
                "each arm needs a non-empty arm_id",
                path=f"{path}.arm_id",
                scope="arm",
            )
        )
        arm_id = f"arm-{index}"

    primary_score = arm.get("primary_score")
    if not isinstance(primary_score, bool):
        issues.append(
            _issue(
                "primary_score_missing",
                "each arm must explicitly declare primary_score as true or false",
                path=f"{path}.primary_score",
                scope="arm",
            )
        )
        primary_score = False
    requires_contextual = arm.get("requires_contextual")
    if not isinstance(requires_contextual, bool):
        issues.append(
            _issue(
                "requires_contextual_missing",
                "each arm must explicitly declare requires_contextual",
                path=f"{path}.requires_contextual",
                scope="arm",
            )
        )
        requires_contextual = False

    replay_mode = arm.get("replay_mode")
    legacy_markers = _legacy_markers(arm)
    if replay_mode not in _REPLAY_MODES:
        if not (legacy_markers and primary_score is False):
            issues.append(
                _issue(
                    "replay_mode_invalid",
                    "replay_mode must be live, input_frozen, or candidate_frozen",
                    path=f"{path}.replay_mode",
                    scope="arm",
                )
            )
    if replay_mode in _FROZEN_REPLAY_MODES:
        freeze_field = "frozen_input" if replay_mode == "input_frozen" else "frozen_candidates"
        frozen_artifact = arm.get(freeze_field)
        if not _is_mapping(frozen_artifact) or not _is_nonempty_string(
            frozen_artifact.get("artifact_id")
        ):
            issues.append(
                _issue(
                    "frozen_input_artifact_missing",
                    f"{replay_mode} needs {freeze_field}.artifact_id",
                    path=f"{path}.{freeze_field}.artifact_id",
                    scope="arm",
                )
            )

    if legacy_markers:
        if primary_score:
            issues.append(
                _issue(
                    "legacy_frozen_plan_not_allowed",
                    "legacy frozen_plan / legacy_pipeline_replay cannot produce a v3 primary score",
                    path=path,
                    scope="arm",
                )
            )
        else:
            warnings.append(
                _issue(
                    "legacy_replay_diagnostic_only",
                    "legacy replay is retained only as a non-primary diagnostic",
                    path=path,
                    scope="arm",
                )
            )

    evaluation_role = arm.get("evaluation_role", "treatment")
    shadow = arm.get("shadow", False)
    if not isinstance(shadow, bool):
        issues.append(
            _issue(
                "shadow_flag_invalid",
                "shadow must be a boolean when supplied",
                path=f"{path}.shadow",
                scope="arm",
            )
        )
    elif shadow and evaluation_role != "shadow":
        issues.append(
            _issue(
                "shadow_role_mismatch",
                "shadow=true is valid only for evaluation_role='shadow'",
                path=f"{path}.evaluation_role",
                scope="arm",
            )
        )
    elif evaluation_role == "shadow" and primary_score:
        issues.append(
            _issue(
                "shadow_arm_cannot_be_primary",
                "a shadow arm cannot be counted as a primary treatment score",
                path=f"{path}.primary_score",
                scope="arm",
            )
        )

    module_bitmap = _normalise_modules(arm.get("execution_modules"))
    if requires_contextual:
        module_bitmap = _validate_contextual_closure(arm, arm_path=path, issues=issues)

    inventory = arm.get("candidate_inventory", {"state": "not_materialized"})
    candidate_state: str | None = None
    if not _is_mapping(inventory):
        issues.append(
            _issue(
                "candidate_inventory_invalid",
                "candidate_inventory must be an object when supplied",
                path=f"{path}.candidate_inventory",
                scope="arm",
            )
        )
    else:
        candidate_state_value = inventory.get("state", "not_materialized")
        if candidate_state_value not in _CANDIDATE_STATES:
            issues.append(
                _issue(
                    "candidate_state_invalid",
                    "candidate_inventory.state is invalid",
                    path=f"{path}.candidate_inventory.state",
                    scope="arm",
                )
            )
        else:
            candidate_state = str(candidate_state_value)
            count = inventory.get("count")
            if candidate_state == "known_empty":
                if count != 0:
                    issues.append(
                        _issue(
                            "candidate_count_inconsistent",
                            "known_empty candidate inventory must have count 0",
                            path=f"{path}.candidate_inventory.count",
                            scope="arm",
                        )
                    )
            elif candidate_state == "nonempty":
                if not _is_positive_int(count):
                    issues.append(
                        _issue(
                            "candidate_count_inconsistent",
                            "nonempty candidate inventory needs a positive integer count",
                            path=f"{path}.candidate_inventory.count",
                            scope="arm",
                        )
                    )
            elif count is not None and not _is_nonnegative_int(count):
                issues.append(
                    _issue(
                        "candidate_count_invalid",
                        "candidate count must be a non-negative integer",
                        path=f"{path}.candidate_inventory.count",
                        scope="arm",
                    )
                )

    status = "invalid_setup" if issues else "ready"
    if status == "invalid_setup":
        execution_state = "not_executable"
    elif legacy_markers:
        execution_state = "diagnostic_only"
    elif candidate_state == "known_empty":
        # This is intentionally successful setup: the runtime may record an
        # empty match, but it must never pay for a missing mechanism.
        execution_state = "valid_no_hit"
    else:
        execution_state = "ready_to_execute"
    return {
        "arm_id": str(arm_id),
        "status": status,
        "execution_state": execution_state,
        "issues": issues,
        "warnings": warnings,
        "module_bitmap": module_bitmap,
        "candidate_state": candidate_state,
        "primary_score": primary_score,
        "requires_contextual": requires_contextual,
        "replay_mode": replay_mode,
    }


def preflight_run(
    run_spec: object,
    gold: object,
    split: object,
    *,
    input_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, structured preflight report without side effects.

    ``input_digests`` is optional because unit callers may hold documents only
    in memory.  The CLI always supplies raw-file SHA-256 values.  If a spec
    declares ``input_digests``, it is checked against the supplied documents.
    """
    global_issues: list[dict[str, str]] = []
    global_warnings: list[dict[str, str]] = []
    if not _is_mapping(run_spec):
        global_issues.append(
            _issue("spec_not_object", "run spec must be a JSON object", path="spec")
        )
        spec_mapping: dict[str, Any] = {}
    else:
        spec_mapping = dict(run_spec)
        if spec_mapping.get("schema") != RUN_SPEC_SCHEMA:
            global_issues.append(
                _issue(
                    "run_spec_schema_invalid",
                    f"spec.schema must be {RUN_SPEC_SCHEMA!r}",
                    path="spec.schema",
                )
            )
        global_issues.extend(_validate_safety(spec_mapping))

    gold_issues, gold_summary = _validate_gold(gold)
    split_issues, split_summary = _validate_split(split, gold_summary["family_ids"])
    global_issues.extend(gold_issues)
    global_issues.extend(split_issues)

    actual_digests = {
        "spec": _document_digest(run_spec),
        "gold": _document_digest(gold),
        "split": _document_digest(split),
    }
    if input_digests:
        # Raw-byte hashes, when supplied by the CLI, are stronger than the
        # canonical JSON identity used by an in-memory caller.
        actual_digests.update({str(key): str(value) for key, value in input_digests.items()})
    declared_digests = spec_mapping.get("input_digests")
    if declared_digests is not None:
        if not _is_mapping(declared_digests):
            global_issues.append(
                _issue(
                    "input_digests_invalid",
                    "spec.input_digests must be an object when supplied",
                    path="spec.input_digests",
                )
            )
        else:
            for key in ("gold", "split"):
                declared = declared_digests.get(key)
                actual = actual_digests.get(key)
                if _is_nonempty_string(declared) and actual is not None and declared != actual:
                    global_issues.append(
                        _issue(
                            "input_digest_mismatch",
                            f"declared {key} digest does not match the loaded document",
                            path=f"spec.input_digests.{key}",
                        )
                    )

    raw_arms = spec_mapping.get("arms")
    arm_reports: list[dict[str, Any]] = []
    if not isinstance(raw_arms, list) or not raw_arms:
        global_issues.append(
            _issue(
                "arms_missing",
                "run spec must contain at least one arm",
                path="spec.arms",
            )
        )
    else:
        arm_reports = [_validate_arm(arm, index) for index, arm in enumerate(raw_arms)]
        arm_ids = [arm["arm_id"] for arm in arm_reports]
        if len(arm_ids) != len(set(arm_ids)):
            global_issues.append(
                _issue(
                    "arm_id_duplicate",
                    "arm_id values must be unique",
                    path="spec.arms",
                )
            )
        if not any(arm["primary_score"] is True for arm in arm_reports):
            global_issues.append(
                _issue(
                    "primary_scoring_arm_missing",
                    "at least one non-shadow arm must be declared primary_score=true",
                    path="spec.arms",
                )
            )

    arm_issues = [issue for arm in arm_reports for issue in arm["issues"]]
    all_issues = [*global_issues, *arm_issues]
    can_execute = not all_issues
    primary_arm_ids = [
        arm["arm_id"]
        for arm in arm_reports
        if arm["primary_score"] is True and arm["status"] == "ready"
    ]
    valid_no_hit_arm_ids = [
        arm["arm_id"]
        for arm in arm_reports
        if arm["status"] == "ready" and arm["execution_state"] == "valid_no_hit"
    ]
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ready" if can_execute else "invalid_setup",
        "can_execute": can_execute,
        "local_only": True,
        "network_calls": 0,
        "model_calls": 0,
        "database_writes": 0,
        "inputs": {
            "digests": actual_digests,
            "gold": gold_summary,
            "split": split_summary,
        },
        "arms": arm_reports,
        "primary_scoring_arm_ids": primary_arm_ids,
        "valid_no_hit_arm_ids": valid_no_hit_arm_ids,
        "issues": all_issues,
        "warnings": global_warnings,
    }
    return report


def _normalise_sha256_hex(value: object) -> str | None:
    """Accept a SHA-256 digest with or without its conventional prefix."""

    if not _is_nonempty_string(value):
        return None
    digest = str(value).strip()
    if digest.lower().startswith("sha256:"):
        digest = digest[7:]
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        return None
    return digest.lower()


def _formal_source_manifest_binding_issues(
    run_spec: object,
    *,
    frozen_source_manifest_sha256: str | None,
) -> list[dict[str, str]]:
    """Bind every formal arm to the same manifest the gold validator read.

    The evaluator-only physical check proves individual Source spans.  A
    formal replay must still use that same frozen source scope; otherwise an
    approved gold document could be checked against one corpus and evaluated
    against another.  This helper remains JSON-only and exposes no source
    contents.
    """

    issues: list[dict[str, str]] = []
    expected = _normalise_sha256_hex(frozen_source_manifest_sha256)
    if expected is None:
        return [
            _issue(
                "formal_source_manifest_digest_invalid",
                "formal Source-gold preflight requires a valid frozen source manifest digest",
                path="frozen_source_manifest",
                scope="gold_physical",
            )
        ]
    if not _is_mapping(run_spec):
        return issues

    arms = run_spec.get("arms")
    if not isinstance(arms, list):
        return issues
    for index, arm in enumerate(arms):
        path = f"spec.arms[{index}].snapshot.source_manifest_sha256"
        if not _is_mapping(arm):
            continue
        snapshot = arm.get("snapshot")
        if not _is_mapping(snapshot):
            issues.append(
                _issue(
                    "formal_source_manifest_binding_missing",
                    "every formal arm needs a snapshot bound to the frozen source manifest",
                    path=path,
                    scope="gold_physical",
                )
            )
            continue
        if _normalise_sha256_hex(snapshot.get("source_manifest_sha256")) != expected:
            issues.append(
                _issue(
                    "formal_source_manifest_binding_mismatch",
                    "formal arm source_manifest_sha256 must match the physically validated frozen source manifest",
                    path=path,
                    scope="gold_physical",
                )
            )

    sqlite_preflight = run_spec.get("sqlite_preflight")
    if _is_mapping(sqlite_preflight) and (
        _normalise_sha256_hex(sqlite_preflight.get("source_manifest_sha256"))
        != expected
    ):
        issues.append(
            _issue(
                "formal_source_manifest_binding_mismatch",
                "sqlite preflight source_manifest_sha256 must match the physically validated frozen source manifest",
                path="spec.sqlite_preflight.source_manifest_sha256",
                scope="gold_physical",
            )
        )
    return issues


_FORMAL_SAFE_ISSUE_SCOPES = frozenset(
    {"setup", "gold", "split", "arm", "gold_physical", "source_manifest", "atom"}
)
_FORMAL_SAFE_ARM_STATUSES = frozenset({"ready", "invalid_setup"})
_FORMAL_SAFE_EXECUTION_STATES = frozenset(
    {"not_executable", "diagnostic_only", "valid_no_hit", "ready_to_execute"}
)


def _formal_status_category(value: object) -> str:
    """Classify an artifact status without echoing an untrusted string."""

    if isinstance(value, str) and value in _APPROVED_STATUSES:
        return "approved"
    if isinstance(value, str) and value in _DRAFT_STATUSES:
        return "draft"
    if value is None:
        return "missing"
    return "other"


def _is_safe_enum(value: object, allowed: frozenset[str] | set[str]) -> bool:
    """Membership guard that cannot be tripped by a JSON list/object value."""

    return isinstance(value, str) and value in allowed


def _safe_digest_label(value: object) -> str | None:
    digest = _normalise_sha256_hex(value)
    return f"sha256:{digest}" if digest is not None else None


def _safe_issue_summaries(value: object) -> list[dict[str, str]]:
    """Keep formal reports identifier-only even if a future validator changes."""

    if not isinstance(value, list):
        return []
    summaries: list[dict[str, str]] = []
    for item in value:
        if not _is_mapping(item):
            continue
        code = item.get("code")
        if (
            not isinstance(code, str)
            or not code.isascii()
            or not code.replace("_", "").isalnum()
            or len(code) > 120
        ):
            continue
        scope = item.get("scope")
        summary = {"code": code}
        if _is_safe_enum(scope, _FORMAL_SAFE_ISSUE_SCOPES):
            summary["scope"] = str(scope)
        summaries.append(summary)
    return summaries


def _safe_nonnegative_count(value: object) -> int:
    return value if _is_nonnegative_int(value) else 0


def _sanitize_physical_source_gold_report(value: object) -> dict[str, Any]:
    """Return the fixed-shape, text-free subset of physical-gold diagnostics."""

    raw = dict(value) if _is_mapping(value) else {}
    raw_counts = raw.get("counts")
    counts = dict(raw_counts) if _is_mapping(raw_counts) else {}
    raw_inputs = raw.get("inputs")
    inputs = dict(raw_inputs) if _is_mapping(raw_inputs) else {}
    safe_inputs: dict[str, object] = {
        "gold_status_category": _formal_status_category(
            inputs.get("gold_status_category")
        ),
        "split_status_category": _formal_status_category(
            inputs.get("split_status_category")
        ),
    }
    for key in (
        "gold_manifest_sha256",
        "split_manifest_sha256",
        "frozen_source_manifest_sha256",
    ):
        if digest := _safe_digest_label(inputs.get(key)):
            safe_inputs[key] = digest
    return {
        "schema": SOURCE_GOLD_VALIDATION_SCHEMA,
        "status": raw.get("status")
        if _is_safe_enum(raw.get("status"), {"ready", "draft", "invalid"})
        else "invalid",
        "scoring_eligible": raw.get("scoring_eligible") is True,
        "counts": {
            "families": _safe_nonnegative_count(counts.get("families")),
            "claim_groups": _safe_nonnegative_count(counts.get("claim_groups")),
            "evidence_atoms": _safe_nonnegative_count(counts.get("evidence_atoms")),
            "frozen_sources": _safe_nonnegative_count(counts.get("frozen_sources")),
        },
        "inputs": safe_inputs,
        "issues": _safe_issue_summaries(raw.get("issues")),
        "diagnostics": _safe_issue_summaries(raw.get("diagnostics")),
    }


def _sanitize_formal_preflight_report(
    report: object, physical: object
) -> dict[str, Any]:
    """Remove all caller-controlled strings from the formal CLI report.

    The structural in-memory preflight remains useful for test authors and
    local editors.  Formal output is an audit artifact, however, so it must
    not re-emit arbitrary family IDs, arm IDs, status labels, mapping keys,
    paths, or free-form messages that could contain Source text.
    """

    raw = dict(report) if _is_mapping(report) else {}
    raw_inputs = raw.get("inputs")
    inputs = dict(raw_inputs) if _is_mapping(raw_inputs) else {}
    raw_digests = inputs.get("digests")
    digests = dict(raw_digests) if _is_mapping(raw_digests) else {}
    safe_digests: dict[str, str] = {}
    for key in ("spec", "gold", "split"):
        if digest := _safe_digest_label(digests.get(key)):
            safe_digests[key] = digest

    raw_gold = inputs.get("gold")
    gold = dict(raw_gold) if _is_mapping(raw_gold) else {}
    raw_split = inputs.get("split")
    split = dict(raw_split) if _is_mapping(raw_split) else {}
    safe_arms: list[dict[str, Any]] = []
    raw_arms = raw.get("arms")
    if isinstance(raw_arms, list):
        for index, raw_arm in enumerate(raw_arms):
            arm = dict(raw_arm) if _is_mapping(raw_arm) else {}
            safe_arms.append(
                {
                    "arm_index": index,
                    "status": arm.get("status")
                    if _is_safe_enum(arm.get("status"), _FORMAL_SAFE_ARM_STATUSES)
                    else "invalid_setup",
                    "execution_state": arm.get("execution_state")
                    if _is_safe_enum(
                        arm.get("execution_state"), _FORMAL_SAFE_EXECUTION_STATES
                    )
                    else "not_executable",
                    "issues": _safe_issue_summaries(arm.get("issues")),
                    "warnings": _safe_issue_summaries(arm.get("warnings")),
                    "candidate_state": arm.get("candidate_state")
                    if _is_safe_enum(arm.get("candidate_state"), _CANDIDATE_STATES)
                    else None,
                    "primary_score": arm.get("primary_score")
                    if isinstance(arm.get("primary_score"), bool)
                    else None,
                    "requires_contextual": arm.get("requires_contextual")
                    if isinstance(arm.get("requires_contextual"), bool)
                    else None,
                    "replay_mode": arm.get("replay_mode")
                    if _is_safe_enum(arm.get("replay_mode"), _REPLAY_MODES)
                    else None,
                }
            )
    primary_indexes = [
        arm["arm_index"]
        for arm in safe_arms
        if arm["status"] == "ready" and arm["primary_score"] is True
    ]
    valid_no_hit_indexes = [
        arm["arm_index"]
        for arm in safe_arms
        if arm["status"] == "ready" and arm["execution_state"] == "valid_no_hit"
    ]
    status = "ready" if raw.get("status") == "ready" else "invalid_setup"
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "can_execute": status == "ready" and raw.get("can_execute") is True,
        "formal_output": True,
        "local_only": True,
        "network_calls": _safe_nonnegative_count(raw.get("network_calls")),
        "model_calls": _safe_nonnegative_count(raw.get("model_calls")),
        "database_writes": _safe_nonnegative_count(raw.get("database_writes")),
        "inputs": {
            "digests": safe_digests,
            "gold": {
                "status_category": _formal_status_category(gold.get("status")),
                "eligible_for_scoring": gold.get("eligible_for_scoring") is True,
            },
            "split": {
                "status_category": _formal_status_category(split.get("status")),
                "eligible_for_scoring": split.get("eligible_for_scoring") is True,
            },
        },
        "arms": safe_arms,
        "primary_scoring_arm_indexes": primary_indexes,
        "valid_no_hit_arm_indexes": valid_no_hit_indexes,
        "issues": _safe_issue_summaries(raw.get("issues")),
        "warnings": _safe_issue_summaries(raw.get("warnings")),
        "source_gold_physical": _sanitize_physical_source_gold_report(physical),
    }


def preflight_formal_run_bytes(
    run_spec_bytes: bytes,
    *,
    gold_bytes: bytes,
    split_bytes: bytes,
    source_root: str | Path,
    frozen_source_manifest_bytes: bytes,
) -> dict[str, Any]:
    """Run byte-bound JSON preflight plus the non-bypassable Source-gold gate.

    This is the only formal preflight entry point that can return ``ready``.
    It binds the run spec, gold, split, and frozen-source manifest to the
    exact bytes that it parses.  ``preflight_run`` remains an in-memory
    structural/synthetic API; it is not an official scoring gate.
    """

    try:
        if not all(
            isinstance(value, bytes)
            for value in (
                run_spec_bytes,
                gold_bytes,
                split_bytes,
                frozen_source_manifest_bytes,
            )
        ):
            raise PreflightInputError("formal inputs must be exact bytes")
        run_spec, spec_digest = _parse_json_bytes(run_spec_bytes, label="spec")
        gold, gold_digest = _parse_json_bytes(gold_bytes, label="gold")
        split, split_digest = _parse_json_bytes(split_bytes, label="split")
        _frozen_source_manifest, frozen_source_manifest_digest = _parse_json_bytes(
            frozen_source_manifest_bytes,
            label="frozen source manifest",
        )
    except PreflightInputError:
        report = _error_report("formal gold or split input is not valid strict JSON")
        physical = {
            "schema": SOURCE_GOLD_VALIDATION_SCHEMA,
            "status": "invalid",
            "scoring_eligible": False,
            "counts": {},
            "inputs": {},
            "issues": [
                _issue(
                    "source_gold_physical_input_invalid",
                    "formal Source-gold validation requires valid strict JSON artifacts",
                    path="source_gold_physical",
                    scope="gold_physical",
                )
            ],
            "diagnostics": [],
        }
        return _sanitize_formal_preflight_report(report, physical)

    try:
        report = preflight_run(
            run_spec,
            gold,
            split,
            input_digests={
                "spec": spec_digest,
                "gold": gold_digest,
                "split": split_digest,
            },
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        report = _error_report("formal run-spec cannot be normalized safely")
    try:
        physical = validate_source_gold_bytes(
            gold_bytes,
            split_bytes,
            source_root=source_root,
            frozen_source_manifest_bytes=frozen_source_manifest_bytes,
        )
    except Exception:
        # The validation tool is intentionally local and deterministic, but a
        # malformed caller object or an unexpected filesystem failure must
        # fail closed without serializing an exception that could contain a
        # source path or source text.
        physical: dict[str, Any] = {
            "schema": SOURCE_GOLD_VALIDATION_SCHEMA,
            "status": "invalid",
            "scoring_eligible": False,
            "counts": {},
            "inputs": {},
            "issues": [
                _issue(
                    "source_gold_physical_validator_error",
                    "physical Source-gold validation could not complete",
                    path="source_gold_physical",
                    scope="gold_physical",
                )
            ],
            "diagnostics": [],
        }

    # ``validate_source_gold_bytes`` owns the safe, identifier/hash-only
    # contents of this nested report and derives all identities from the exact
    # bytes it parsed.  Do not accept an injected report from a caller.
    report["source_gold_physical"] = physical
    physical_ready = (
        isinstance(physical, Mapping)
        and physical.get("schema") == SOURCE_GOLD_VALIDATION_SCHEMA
        and physical.get("scoring_eligible") is True
    )
    formal_issues = _formal_source_manifest_binding_issues(
        run_spec,
        frozen_source_manifest_sha256=frozen_source_manifest_digest,
    )
    if not physical_ready:
        formal_issues.append(
            _issue(
                "source_gold_physical_not_ready",
                "formal execution requires Source-gold validation against frozen local Source files",
                path="source_gold_physical",
                scope="gold_physical",
            )
        )
    if formal_issues:
        report["issues"] = [*report["issues"], *formal_issues]
        report["status"] = "invalid_setup"
        report["can_execute"] = False
    return _sanitize_formal_preflight_report(report, physical)


def preflight_formal_run(
    run_spec: object,
    *,
    gold_bytes: bytes,
    split_bytes: bytes,
    source_root: str | Path,
    frozen_source_manifest_bytes: bytes,
) -> dict[str, Any]:
    """Diagnostic object wrapper that deliberately cannot authorize scoring.

    A caller-held object cannot prove which raw run-spec artifact produced it.
    Formal callers must use :func:`preflight_formal_run_bytes`; this wrapper is
    retained only for synthetic diagnostics and always fails closed.
    """

    try:
        result = preflight_formal_run_bytes(
            _json_bytes(run_spec),
            gold_bytes=gold_bytes,
            split_bytes=split_bytes,
            source_root=source_root,
            frozen_source_manifest_bytes=frozen_source_manifest_bytes,
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        result = _sanitize_formal_preflight_report(
            _error_report("in-memory run spec cannot be normalized safely"),
            {},
        )
    result["issues"] = [
        *result["issues"],
        {"scope": "setup", "code": "formal_run_spec_raw_binding_unavailable"},
    ]
    result["status"] = "invalid_setup"
    result["can_execute"] = False
    return result


def _replay_issue(code: str, message: str, *, path: str) -> dict[str, str]:
    """Create an execution-layer issue without exposing frozen request text."""

    return _issue(code, message, path=path, scope="replay")


def _is_sha256_digest(value: object) -> bool:
    if not _is_nonempty_string(value):
        return False
    text = str(value)
    return (
        text.startswith("sha256:")
        and len(text) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _database_identity(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Hash the frozen SQLite file and its journal sidecars without opening it.

    The SQLite preflight owns semantic/database checks.  This small identity
    snapshot brackets execution so an injected executor cannot quietly turn a
    replay into a mutable run.
    """

    issues: list[dict[str, str]] = []

    def digest_file(candidate: Path, *, required: bool) -> str | None:
        if not candidate.exists():
            if required:
                issues.append(
                    _replay_issue(
                        "database_identity_unavailable",
                        "frozen database is not readable",
                        path="database",
                    )
                )
            return None
        if not candidate.is_file():
            issues.append(
                _replay_issue(
                    "database_identity_unavailable",
                    "frozen database identity target is not a regular file",
                    path="database",
                )
            )
            return None
        try:
            return "sha256:" + sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            issues.append(
                _replay_issue(
                    "database_identity_unavailable",
                    "frozen database bytes could not be read",
                    path="database",
                )
            )
            return None

    main_digest = digest_file(path, required=True)
    sidecars = {
        suffix: digest_file(path.with_name(path.name + suffix), required=False)
        for suffix in ("-wal", "-shm", "-journal")
    }
    if main_digest is None:
        return None, issues
    return {"database_sha256": main_digest, "sidecars": sidecars}, issues


def _record_issue(
    issues: list[dict[str, str]],
    code: str,
    message: str,
    *,
    path: str,
) -> None:
    issues.append(_replay_issue(code, message, path=path))


def _validate_request_record(
    record: object,
    *,
    path: str,
    snapshot: Mapping[str, Any],
    sqlite_config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Validate the minimal, versioned input-frozen record contract.

    Actual vector values intentionally stay outside JSON.  A local injected
    resolver supplies the already-frozen ``QueryVectorBundle`` identified by
    this manifest; this module neither embeds text nor contacts a provider.
    """

    issues: list[dict[str, str]] = []
    if not _is_mapping(record):
        _record_issue(issues, "request_record_not_object", "record must be an object", path=path)
        return None, issues
    normalized = dict(record)
    if normalized.get("schema") != REQUEST_RECORD_SCHEMA:
        _record_issue(
            issues,
            "request_record_schema_invalid",
            f"record.schema must be {REQUEST_RECORD_SCHEMA!r}",
            path=f"{path}.schema",
        )
    if not _is_nonempty_string(normalized.get("record_id")):
        _record_issue(
            issues,
            "request_record_id_missing",
            "record_id is required",
            path=f"{path}.record_id",
        )

    request = normalized.get("request")
    if not _is_mapping(request):
        _record_issue(
            issues,
            "request_manifest_missing",
            "record needs a versioned frozen request object",
            path=f"{path}.request",
        )
        request = {}
    request = dict(request)
    if request.get("schema") != REQUEST_MANIFEST_SCHEMA:
        _record_issue(
            issues,
            "request_manifest_schema_invalid",
            f"request.schema must be {REQUEST_MANIFEST_SCHEMA!r}",
            path=f"{path}.request.schema",
        )
    expected_request_digest = _document_digest(request)
    if normalized.get("request_sha256") != expected_request_digest:
        _record_issue(
            issues,
            "request_manifest_digest_mismatch",
            "request_sha256 must equal the canonical frozen request digest",
            path=f"{path}.request_sha256",
        )
    if not _is_nonempty_string(request.get("question")):
        _record_issue(
            issues,
            "request_question_missing",
            "frozen request needs a non-empty question",
            path=f"{path}.request.question",
        )
    if not _is_mapping(request.get("intent_override")):
        _record_issue(
            issues,
            "request_intent_missing",
            "frozen request needs an explicit intent_override object",
            path=f"{path}.request.intent_override",
        )
    followups = request.get("followup_queries_override")
    if not isinstance(followups, list) or not all(
        _is_nonempty_string(value) for value in followups
    ):
        _record_issue(
            issues,
            "request_followups_missing",
            "frozen request needs an explicit followup_queries_override list",
            path=f"{path}.request.followup_queries_override",
        )
    if not _is_nonempty_string(request.get("contextual_domain")):
        _record_issue(
            issues,
            "request_contextual_domain_missing",
            "frozen request needs contextual_domain",
            path=f"{path}.request.contextual_domain",
        )
    if not _is_nonempty_string(request.get("contextual_evaluation_as_of")):
        _record_issue(
            issues,
            "request_evaluation_as_of_missing",
            "frozen request needs contextual_evaluation_as_of",
            path=f"{path}.request.contextual_evaluation_as_of",
        )
    elif request.get("contextual_evaluation_as_of") != sqlite_config.get("evaluation_as_of"):
        _record_issue(
            issues,
            "request_evaluation_as_of_mismatch",
            "record evaluation time must equal the frozen SQLite snapshot time",
            path=f"{path}.request.contextual_evaluation_as_of",
        )
    if request.get("frozen_plan") not in (None, False, "") or request.get(
        "legacy_pipeline_replay"
    ) is True:
        _record_issue(
            issues,
            "legacy_frozen_pipeline_bypass",
            "v3 input replay may not provide a legacy frozen plan/pipeline",
            path=f"{path}.request",
        )

    vector = normalized.get("vector_manifest")
    if not _is_mapping(vector):
        _record_issue(
            issues,
            "vector_manifest_missing",
            "record needs a versioned vector manifest",
            path=f"{path}.vector_manifest",
        )
        vector = {}
    vector = dict(vector)
    if vector.get("schema") != VECTOR_MANIFEST_SCHEMA:
        _record_issue(
            issues,
            "vector_manifest_schema_invalid",
            f"vector_manifest.schema must be {VECTOR_MANIFEST_SCHEMA!r}",
            path=f"{path}.vector_manifest.schema",
        )
    for field in ("artifact_id", "artifact_sha256", "embedding_space_id"):
        value = vector.get(field)
        if not _is_nonempty_string(value) or (
            field == "artifact_sha256" and not _is_sha256_digest(value)
        ):
            _record_issue(
                issues,
                "vector_manifest_incomplete",
                f"vector manifest needs a valid {field}",
                path=f"{path}.vector_manifest.{field}",
            )
    expected_space = (
        snapshot.get("embedding_space", {}).get("id")
        if _is_mapping(snapshot.get("embedding_space"))
        else None
    )
    expected_dimension = (
        snapshot.get("embedding_space", {}).get("dimension")
        if _is_mapping(snapshot.get("embedding_space"))
        else None
    )
    if vector.get("embedding_space_id") != expected_space:
        _record_issue(
            issues,
            "vector_manifest_space_mismatch",
            "vector manifest embedding space must equal the arm snapshot",
            path=f"{path}.vector_manifest.embedding_space_id",
        )
    if vector.get("dimension") != expected_dimension or not _is_positive_int(
        vector.get("dimension")
    ):
        _record_issue(
            issues,
            "vector_manifest_dimension_mismatch",
            "vector manifest dimension must equal the arm snapshot",
            path=f"{path}.vector_manifest.dimension",
        )
    if vector.get("normalized") is not True:
        _record_issue(
            issues,
            "vector_manifest_not_normalized",
            "vector manifest must explicitly require normalized vectors",
            path=f"{path}.vector_manifest.normalized",
        )
    bindings = vector.get("binding_ids")
    if not isinstance(bindings, list) or not bindings or not all(
        _is_nonempty_string(value) for value in bindings
    ):
        _record_issue(
            issues,
            "vector_manifest_bindings_missing",
            "vector manifest needs non-empty binding_ids",
            path=f"{path}.vector_manifest.binding_ids",
        )

    database = normalized.get("database_manifest")
    if not _is_mapping(database):
        _record_issue(
            issues,
            "database_manifest_missing",
            "record needs a versioned database manifest",
            path=f"{path}.database_manifest",
        )
        database = {}
    database = dict(database)
    if database.get("schema") != DATABASE_MANIFEST_SCHEMA:
        _record_issue(
            issues,
            "database_manifest_schema_invalid",
            f"database_manifest.schema must be {DATABASE_MANIFEST_SCHEMA!r}",
            path=f"{path}.database_manifest.schema",
        )
    for field in ("database_sha256", "source_manifest_sha256", "scope_sha256"):
        value = database.get(field)
        if not _is_sha256_digest(value):
            _record_issue(
                issues,
                "database_manifest_incomplete",
                f"database manifest needs a valid {field}",
                path=f"{path}.database_manifest.{field}",
            )
        elif value != snapshot.get(field):
            _record_issue(
                issues,
                "database_manifest_snapshot_mismatch",
                f"database manifest {field} must equal the arm snapshot",
                path=f"{path}.database_manifest.{field}",
            )
    if database.get("schema_version") != sqlite_config.get("schema_version"):
        _record_issue(
            issues,
            "database_manifest_schema_version_mismatch",
            "database manifest schema_version must equal sqlite_preflight",
            path=f"{path}.database_manifest.schema_version",
        )

    return normalized, issues


def _validate_input_frozen_arm(
    arm: Mapping[str, Any],
    *,
    path: str,
    sqlite_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return validated records for the supported, non-legacy frozen mode."""

    issues: list[dict[str, str]] = []
    frozen = arm.get("frozen_input")
    if not _is_mapping(frozen):
        _record_issue(
            issues,
            "frozen_input_manifest_missing",
            "input_frozen replay needs a versioned frozen_input manifest",
            path=f"{path}.frozen_input",
        )
        return [], issues
    frozen = dict(frozen)
    if frozen.get("schema") != FROZEN_INPUT_SCHEMA:
        _record_issue(
            issues,
            "frozen_input_schema_invalid",
            f"frozen_input.schema must be {FROZEN_INPUT_SCHEMA!r}",
            path=f"{path}.frozen_input.schema",
        )
    if not _is_nonempty_string(frozen.get("artifact_id")):
        _record_issue(
            issues,
            "frozen_input_artifact_missing",
            "frozen input needs artifact_id",
            path=f"{path}.frozen_input.artifact_id",
        )
    records = frozen.get("records")
    if not isinstance(records, list) or not records:
        _record_issue(
            issues,
            "frozen_input_records_missing",
            "frozen input needs at least one request record",
            path=f"{path}.frozen_input.records",
        )
        return [], issues
    if frozen.get("artifact_sha256") != _document_digest(records):
        _record_issue(
            issues,
            "frozen_input_digest_mismatch",
            "artifact_sha256 must equal the canonical request-record digest",
            path=f"{path}.frozen_input.artifact_sha256",
        )

    snapshot = arm.get("snapshot")
    snapshot = dict(snapshot) if _is_mapping(snapshot) else {}
    normalized: list[dict[str, Any]] = []
    record_ids: list[str] = []
    for index, record in enumerate(records):
        parsed, record_issues = _validate_request_record(
            record,
            path=f"{path}.frozen_input.records[{index}]",
            snapshot=snapshot,
            sqlite_config=sqlite_config,
        )
        issues.extend(record_issues)
        if parsed is not None:
            normalized.append(parsed)
            if _is_nonempty_string(parsed.get("record_id")):
                record_ids.append(str(parsed["record_id"]))
    if len(record_ids) != len(set(record_ids)):
        _record_issue(
            issues,
            "request_record_id_duplicate",
            "record_id values must be unique within one frozen input artifact",
            path=f"{path}.frozen_input.records",
        )
    return normalized, issues


def _actual_module_bitmap(result: Mapping[str, Any]) -> dict[str, bool]:
    """Derive execution evidence from the returned receipt, never the spec."""

    contextual = result.get("contextual_association")
    contextual = dict(contextual) if _is_mapping(contextual) else {}
    slot_trace = result.get("evidence_slot_trace")
    slot_trace = dict(slot_trace) if _is_mapping(slot_trace) else {}
    selector_trace = slot_trace.get("slot_selector_v3")
    selector_trace = dict(selector_trace) if _is_mapping(selector_trace) else {}
    backend = str(contextual.get("backend", ""))
    contribution = (
        backend.endswith("_contribution_selector_v3")
        and contextual.get("compatibility_projection") == "v3_contribution_selector"
    ) or selector_trace.get("compatibility_projection") == "v3_contribution_selector"
    gate = contextual.get("target_gate")
    target_gate = bool(
        _is_mapping(gate)
        and dict(gate).get("stage") == "target_checked_before_endpoint_cap_v1"
    )
    selector = _is_mapping(contextual.get("contribution_selector")) or _is_mapping(
        selector_trace.get("treatment_selector")
    )
    legacy = bool(result.get("query_plan_frozen") is True) or (
        contextual.get("compatibility_projection")
        == "contextual_slot_hits_pending_t10_contribution_selector"
    )
    return {
        "query_engine": True,
        "v3_contribution_mechanism": bool(contribution),
        "target_gate": target_gate,
        "contribution_selector": bool(selector),
        "legacy_frozen_pipeline": legacy,
    }


def _safe_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep identifiers/counts only; a replay receipt must not echo source text."""

    def ids(name: str) -> list[int]:
        value = result.get(name)
        if not isinstance(value, list):
            return []
        return [int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)]

    contextual = result.get("contextual_association")
    contextual = dict(contextual) if _is_mapping(contextual) else {}
    return {
        "episode_ids": ids("episode_ids"),
        "candidate_episode_ids": ids("candidate_episode_ids"),
        "association_ids": ids("association_ids"),
        "contextual_backend": str(contextual.get("backend", "")),
        "contextual_selected_count": int(
            contextual.get("selected_count", 0)
            if isinstance(contextual.get("selected_count", 0), int)
            else 0
        ),
    }


def execute_replay(
    run_spec: object,
    gold: object,
    split: object,
    *,
    executor: object,
    database: str | Path,
    source_manifest: str | Path,
    scope_manifest: str | Path,
    vector_bundle_resolver: Callable[[Mapping[str, Any]], object] | None = None,
    trace_bridge_factory: Callable[[Mapping[str, Any]], object] | None = None,
    input_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute versioned ``input_frozen`` V3 records through ``executor.query``.

    This is intentionally a small adapter, not a second retrieval pipeline.
    The caller injects a local, pre-frozen vector resolver and a QueryEngine-
    compatible executor.  Each request is sent through ``query`` with strict
    vectors and no ``frozen_plan`` so the live V3 contribution selector and
    target gate remain in the execution path.  ``candidate_frozen`` is
    deliberately not accepted yet: the current engine's candidate freeze is
    its legacy ``frozen_plan`` path, which bypasses that V3 path.

    The function never grades gold.  In particular it does not invoke the
    executor if source gold/split is draft or non-usable, and it records only
    safe identifiers/counts from an execution result.
    """

    json_preflight = preflight_run(
        run_spec, gold, split, input_digests=input_digests
    )
    report: dict[str, Any] = {
        "schema": REPLAY_RESULT_SCHEMA,
        "status": "invalid_setup",
        "can_execute": False,
        "local_only": True,
        "network_calls": 0,
        "model_calls": 0,
        "database_writes": 0,
        "json_preflight": json_preflight,
        "sqlite_preflight": None,
        "database_identity": {"before": None, "after": None, "immutable": False},
        "arms": [],
        "scoring": {
            "eligible_source_gold": bool(
                json_preflight.get("inputs", {}).get("gold", {}).get(
                    "eligible_for_scoring", False
                )
            ),
            "performed": False,
            "reason": "replay_executor_does_not_score_gold",
        },
        "issues": [],
        "warnings": [],
    }
    issues: list[dict[str, str]] = []
    if json_preflight.get("status") != "ready":
        # Do not touch a database or executor when evaluator inputs are not
        # approved.  This makes draft gold an unambiguous no-score outcome.
        issues.append(
            _replay_issue(
                "json_preflight_not_ready",
                "V3 JSON/gold/split preflight must be ready before replay",
                path="json_preflight",
            )
        )
        report["issues"] = issues
        report["scoring"]["reason"] = "source_gold_or_split_not_usable"
        return report

    # Keep the SQLite-specific dependency lazy: ordinary JSON preflight remains
    # usable in environments intentionally lacking SQLite support.
    from benchmarks.support.recall_v3_sqlite_preflight import preflight_sqlite_evidence

    sqlite_preflight = preflight_sqlite_evidence(
        run_spec,
        database,
        source_manifest=source_manifest,
        scope_manifest=scope_manifest,
    )
    report["sqlite_preflight"] = sqlite_preflight
    if sqlite_preflight.get("status") != "ready":
        issues.append(
            _replay_issue(
                "sqlite_preflight_not_ready",
                "frozen SQLite/source closure is unsafe for replay",
                path="sqlite_preflight",
            )
        )
        report["issues"] = issues
        return report

    database_path = Path(database)
    before, identity_issues = _database_identity(database_path)
    issues.extend(identity_issues)
    report["database_identity"]["before"] = before
    expected_database = (
        sqlite_preflight.get("database", {}).get("database_sha256_after")
        if _is_mapping(sqlite_preflight.get("database"))
        else None
    )
    if before is None or before.get("database_sha256") != expected_database:
        issues.append(
            _replay_issue(
                "database_identity_not_pinned",
                "database identity changed after SQLite preflight and before execution",
                path="database",
            )
        )

    query = getattr(executor, "query", None)
    if not callable(query):
        issues.append(
            _replay_issue(
                "query_engine_executor_missing",
                "executor must provide a QueryEngine-compatible query method",
                path="executor",
            )
        )
    if vector_bundle_resolver is None:
        issues.append(
            _replay_issue(
                "vector_bundle_resolver_missing",
                "input-frozen replay needs a local frozen-vector resolver",
                path="vector_bundle_resolver",
            )
        )

    spec = dict(run_spec) if _is_mapping(run_spec) else {}
    sqlite_config = (
        dict(spec.get("sqlite_preflight"))
        if _is_mapping(spec.get("sqlite_preflight"))
        else {}
    )
    raw_arms = spec.get("arms")
    plans: list[tuple[str, int, Mapping[str, Any], list[dict[str, Any]]]] = []
    if not isinstance(raw_arms, list):
        issues.append(
            _replay_issue("arms_missing", "run spec arms are unavailable", path="spec.arms")
        )
    else:
        for index, raw_arm in enumerate(raw_arms):
            arm_path = f"spec.arms[{index}]"
            if not _is_mapping(raw_arm):
                continue
            arm = dict(raw_arm)
            arm_id = str(arm.get("arm_id") or f"arm-{index}")
            replay_mode = arm.get("replay_mode")
            legacy = _legacy_markers(arm)
            arm_result: dict[str, Any] = {
                "arm_id": arm_id,
                "replay_mode": replay_mode,
                "records": [],
            }
            if legacy:
                arm_result.update(
                    {
                        "status": "legacy_diagnostic_only",
                        "execution_label": "legacy_replay_not_v3",
                        "legacy_markers": legacy,
                    }
                )
                report["arms"].append(arm_result)
                continue
            if replay_mode == "live":
                arm_result.update(
                    {
                        "status": "not_executed",
                        "execution_label": "live_arm_outside_frozen_replay_executor",
                    }
                )
                report["arms"].append(arm_result)
                continue
            if replay_mode == "candidate_frozen":
                arm_result.update(
                    {
                        "status": "invalid_setup",
                        "execution_label": "candidate_frozen_not_supported_without_v3_native_injection",
                    }
                )
                arm_result["issues"] = [
                    _replay_issue(
                        "candidate_frozen_v3_bypass_risk",
                        "current candidate freeze maps to legacy frozen_plan and would bypass V3 selection",
                        path=f"{arm_path}.replay_mode",
                    )
                ]
                issues.extend(arm_result["issues"])
                report["arms"].append(arm_result)
                continue
            if replay_mode != "input_frozen":
                arm_result.update({"status": "invalid_setup", "execution_label": "unsupported_replay_mode"})
                arm_result["issues"] = [
                    _replay_issue(
                        "unsupported_replay_mode",
                        "only versioned input_frozen replay is executable here",
                        path=f"{arm_path}.replay_mode",
                    )
                ]
                issues.extend(arm_result["issues"])
                report["arms"].append(arm_result)
                continue
            if arm.get("requires_contextual") is not True:
                arm_result.update({"status": "invalid_setup", "execution_label": "v3_contextual_required"})
                arm_result["issues"] = [
                    _replay_issue(
                        "v3_contextual_required",
                        "frozen V3 replay must require the contextual contribution path",
                        path=f"{arm_path}.requires_contextual",
                    )
                ]
                issues.extend(arm_result["issues"])
                report["arms"].append(arm_result)
                continue
            records, arm_issues = _validate_input_frozen_arm(
                arm, path=arm_path, sqlite_config=sqlite_config
            )
            arm_result["issues"] = arm_issues
            if arm_issues:
                arm_result.update({"status": "invalid_setup", "execution_label": "frozen_manifest_invalid"})
                issues.extend(arm_issues)
            else:
                arm_result.update({"status": "planned", "execution_label": "input_frozen_v3"})
                plans.append((arm_id, len(report["arms"]), arm, records))
            report["arms"].append(arm_result)

    if not plans:
        issues.append(
            _replay_issue(
                "no_input_frozen_v3_records",
                "no executable versioned input_frozen V3 records were supplied",
                path="spec.arms",
            )
        )
    if issues:
        report["issues"] = issues
        return report

    # All static validation has completed.  Only now can the injected engine
    # receive a request; this avoids partial benchmark execution from an
    # incomplete artifact set.
    assert callable(query)
    assert vector_bundle_resolver is not None
    for _arm_id, arm_index, _arm, records in plans:
        arm_result = report["arms"][arm_index]
        arm_result["status"] = "executing"
        for record in records:
            record_id = str(record["record_id"])
            record_result: dict[str, Any] = {
                "record_id": record_id,
                "record_sha256": _document_digest(record),
                "status": "invalid_setup",
                "executed_modules": {},
            }
            request = dict(record["request"])
            try:
                bundle = vector_bundle_resolver(record)
                if bundle is None:
                    raise ValueError("frozen vector resolver returned no bundle")
                kwargs: dict[str, Any] = {
                    "generate_answer": False,
                    "intent_override": dict(request["intent_override"]),
                    "followup_queries_override": list(
                        request["followup_queries_override"]
                    ),
                    "query_vector_bundle": bundle,
                    "strict_vector_bundle": True,
                    "contextual_domain": str(request["contextual_domain"]),
                    "contextual_evaluation_as_of": str(
                        request["contextual_evaluation_as_of"]
                    ),
                }
                if trace_bridge_factory is not None:
                    kwargs["trace_bridge"] = trace_bridge_factory(record)
                # Do not pass ``frozen_plan``: the current engine's frozen-plan
                # branch explicitly skips contextual V3 selection.
                raw_result = query(str(request["question"]), **kwargs)
            except Exception as exc:
                record_result["status"] = "execution_failed"
                record_result["issues"] = [
                    _replay_issue(
                        "executor_query_failed",
                        f"QueryEngine-compatible executor raised {type(exc).__name__}",
                        path=f"arm:{_arm_id}/record:{record_id}",
                    )
                ]
                issues.extend(record_result["issues"])
                arm_result["records"].append(record_result)
                continue
            if not _is_mapping(raw_result):
                record_result["issues"] = [
                    _replay_issue(
                        "executor_result_invalid",
                        "executor query result must be an object with V3 trace evidence",
                        path=f"arm:{_arm_id}/record:{record_id}",
                    )
                ]
                issues.extend(record_result["issues"])
                arm_result["records"].append(record_result)
                continue
            result = dict(raw_result)
            bitmap = _actual_module_bitmap(result)
            record_result["executed_modules"] = bitmap
            mechanism_issues: list[dict[str, str]] = []
            required = (
                ("v3_contribution_mechanism", "v3_contribution_mechanism_absent"),
                ("target_gate", "v3_target_gate_absent"),
                ("contribution_selector", "v3_selector_absent"),
            )
            for module, code in required:
                if not bitmap[module]:
                    mechanism_issues.append(
                        _replay_issue(
                            code,
                            f"executed result did not prove {module}",
                            path=f"arm:{_arm_id}/record:{record_id}",
                        )
                    )
            if bitmap["legacy_frozen_pipeline"]:
                mechanism_issues.append(
                    _replay_issue(
                        "legacy_frozen_pipeline_bypass",
                        "executed result reports a legacy frozen pipeline rather than V3 selection",
                        path=f"arm:{_arm_id}/record:{record_id}",
                    )
                )
            record_result["result"] = _safe_result_summary(result)
            if mechanism_issues:
                record_result["status"] = "invalid_setup"
                record_result["issues"] = mechanism_issues
                issues.extend(mechanism_issues)
            else:
                record_result["status"] = "executed"
            arm_result["records"].append(record_result)
        arm_result["status"] = (
            "invalid_setup"
            if any(item.get("status") != "executed" for item in arm_result["records"])
            else "executed"
        )

    after, identity_issues = _database_identity(database_path)
    issues.extend(identity_issues)
    report["database_identity"]["after"] = after
    immutable = before is not None and before == after
    report["database_identity"]["immutable"] = immutable
    if not immutable:
        issues.append(
            _replay_issue(
                "database_changed_during_replay",
                "database or SQLite sidecar identity changed during replay",
                path="database",
            )
        )

    report["issues"] = issues
    report["status"] = "executed" if not issues else "invalid_setup"
    report["can_execute"] = report["status"] == "executed"
    return report


# A descriptive spelling for callers that want to distinguish the executable
# layer from the preflight-only ``preflight_run`` API.
execute_frozen_replay = execute_replay


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> object:
    raise PreflightInputError(f"non-standard JSON number: {value}")


def _parse_finite_json_float(value: str) -> float:
    """Decode a JSON float but reject overflow to a non-finite value."""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise PreflightInputError("non-finite JSON number")
    return parsed


def _parse_json_bytes(raw: bytes, *, label: str) -> tuple[object, str]:
    """Strictly parse one exact JSON byte payload and return its raw digest."""

    if not isinstance(raw, bytes):
        raise PreflightInputError(f"{label} must be exact bytes")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_json_float,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        PreflightInputError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise PreflightInputError(f"invalid strict JSON for {label}") from exc
    return document, "sha256:" + sha256(raw).hexdigest()


def load_json_document_bytes(path: Path) -> tuple[object, str, bytes]:
    """Read/strictly parse one JSON file while preserving its exact bytes."""

    if path.suffix.lower() != ".json":
        raise PreflightInputError(f"preflight inputs must be .json files: {path.name}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightInputError(f"cannot read JSON input {path.name}: {exc}") from exc
    try:
        document, digest = _parse_json_bytes(raw, label=path.name)
    except PreflightInputError as exc:
        raise PreflightInputError(f"invalid JSON in {path.name}") from exc
    return document, digest, raw


def load_json_document(path: Path) -> tuple[object, str]:
    """Read one JSON document; reject duplicate keys and NaN/Infinity tokens."""

    document, digest, _ = load_json_document_bytes(path)
    return document, digest


def _validate_formal_report_output_path(
    output: Path,
    *,
    input_paths: Sequence[Path],
    source_root: Path,
) -> None:
    """Reject formal report writes that could mutate frozen inputs/sources."""

    if output.suffix.lower() != ".json" or not _safe_report_output_path(
        output,
        artifact_paths=input_paths,
        source_root=source_root,
    ):
        raise PreflightInputError("formal report output path is unsafe")


def write_preflight(
    report: Mapping[str, Any],
    output: Path,
    *,
    input_paths: Sequence[Path],
    source_root: Path,
) -> bool:
    """Atomically write a report only while it remains outside frozen inputs.

    The shared Source-gold guard rechecks resolved paths and physical file
    identities before writing and again before replacement.  Every caller
    spelling is first fixed to a verified absolute direct-local path, so a
    CWD change after preflight cannot retarget the temporary file or replace.
    """

    direct_output, output_rejection = _report_output_preflight(output)
    direct_source_root, source_root_rejection = _direct_local_path_preflight(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if (
        output_rejection is not None
        or direct_output is None
        or source_root_rejection is not None
        or direct_source_root is None
        or direct_output.suffix.lower() != ".json"
    ):
        return False
    direct_input_paths: list[Path] = []
    for input_path in input_paths:
        direct_input, input_rejection = _direct_local_path_preflight(
            input_path,
            prefix="artifact_input",
            expect_directory=False,
        )
        if input_rejection is not None or direct_input is None:
            return False
        direct_input_paths.append(direct_input)

    if not _safe_report_output_path(
        direct_output,
        artifact_paths=direct_input_paths,
        source_root=direct_source_root,
    ):
        return False
    # Formal reports require an already-existing direct-local parent.  Do not
    # create a directory after the guard: a missing/replaced parent must fail
    # closed rather than becoming an implicit filesystem mutation.
    if not _safe_report_output_path(
        direct_output,
        artifact_paths=direct_input_paths,
        source_root=direct_source_root,
    ):
        return False
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=direct_output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not _safe_report_output_path(
            direct_output,
            artifact_paths=direct_input_paths,
            source_root=direct_source_root,
        ):
            return False
        os.replace(temporary, direct_output)
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _error_report(message: str) -> dict[str, Any]:
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "invalid_setup",
        "can_execute": False,
        "local_only": True,
        "network_calls": 0,
        "model_calls": 0,
        "database_writes": 0,
        "inputs": {},
        "arms": [],
        "primary_scoring_arm_ids": [],
        "valid_no_hit_arm_ids": [],
        "issues": [_issue("input_load_failed", message, path="cli")],
        "warnings": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local-only, no-model v3 recall experiment preflight."
    )
    parser.add_argument("--spec", type=Path, required=True, help="v3 run-spec JSON")
    parser.add_argument("--gold", type=Path, required=True, help="reviewed Source-gold JSON")
    parser.add_argument("--split", type=Path, required=True, help="reviewed split JSON")
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root directory containing the frozen corpus-relative Source JSON files",
    )
    parser.add_argument(
        "--frozen-source-manifest",
        type=Path,
        required=True,
        help="frozen source-key / source-file-hash JSON manifest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="preflight.json output path, or '-' for stdout",
    )
    args = parser.parse_args(argv)
    local_input_checks = (
        ("spec_input", args.spec, False),
        ("gold_input", args.gold, False),
        ("split_input", args.split, False),
        ("frozen_source_manifest_input", args.frozen_source_manifest, False),
        ("source_root", args.source_root, True),
    )
    verified_inputs: dict[str, Path] = {}
    for prefix, path, expect_directory in local_input_checks:
        direct_path, rejection = _direct_local_path_preflight(
            path,
            prefix=prefix,
            expect_directory=expect_directory,
        )
        if rejection is not None or direct_path is None:
            # Reject path classes that could trigger filesystem network I/O
            # before resolving, opening, or reporting on any input.  Do not
            # echo the original path.
            print(
                json.dumps(
                    _sanitize_formal_preflight_report(
                        _error_report("formal inputs must be direct local paths"), {}
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        verified_inputs[prefix] = direct_path

    spec_path = verified_inputs["spec_input"]
    gold_path = verified_inputs["gold_input"]
    split_path = verified_inputs["split_input"]
    frozen_source_manifest_path = verified_inputs["frozen_source_manifest_input"]
    source_root = verified_inputs["source_root"]
    output_path: Path | None = None
    if args.output != "-":
        output_path, output_rejection = _report_output_preflight(Path(args.output))
        if output_rejection is not None or output_path is None:
            print(
                json.dumps(
                    _sanitize_formal_preflight_report(
                        _error_report("formal report output path is unsafe"), {}
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
    if output_path is not None:
        try:
            _validate_formal_report_output_path(
                output_path,
                input_paths=(
                    spec_path,
                    gold_path,
                    split_path,
                    frozen_source_manifest_path,
                ),
                source_root=source_root,
            )
        except PreflightInputError:
            # Do not write an error report to a location that could itself be
            # a frozen input/source.  Stdout is the only safe fallback.
            print(
                json.dumps(
                    _sanitize_formal_preflight_report(
                        _error_report("formal report output path is unsafe"), {}
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    try:
        _spec, _spec_digest, spec_bytes = load_json_document_bytes(spec_path)
        _gold, _gold_digest, gold_bytes = load_json_document_bytes(gold_path)
        _split, _split_digest, split_bytes = load_json_document_bytes(split_path)
        _frozen, _frozen_digest, frozen_source_manifest_bytes = load_json_document_bytes(
            frozen_source_manifest_path
        )
        report = preflight_formal_run_bytes(
            spec_bytes,
            gold_bytes=gold_bytes,
            split_bytes=split_bytes,
            source_root=source_root,
            frozen_source_manifest_bytes=frozen_source_manifest_bytes,
        )
    except PreflightInputError:
        report = _sanitize_formal_preflight_report(
            _error_report("formal input cannot be read as strict JSON"), {}
        )
    if args.output == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        assert output_path is not None
        try:
            written = write_preflight(
                report,
                output_path,
                input_paths=(
                    spec_path,
                    gold_path,
                    split_path,
                    frozen_source_manifest_path,
                ),
                source_root=source_root,
            )
        except OSError:
            written = False
        if not written:
            # A failed atomic report write must not be retried against an
            # alternate path implicitly.  Emit a text-safe failure on stdout.
            print(
                json.dumps(
                    _sanitize_formal_preflight_report(
                        _error_report("formal report could not be written"), {}
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
