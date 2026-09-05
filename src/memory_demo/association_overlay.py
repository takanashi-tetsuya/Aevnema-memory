from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from typing import Any, Iterable

from memory_demo.repositories import AssociationRepository
from memory_demo.types import AssociationDraft


@dataclass(slots=True)
class AssociationDelta:
    """Complete row-level changes caused by one query-time growth step."""

    created: list[dict[str, Any]]
    reinforced: list[dict[str, Any]]

    @classmethod
    def capture(
        cls,
        before: dict[int, dict[str, Any]],
        after: dict[int, dict[str, Any]],
        created_ids: Iterable[int],
        reinforced_ids: Iterable[int],
    ) -> "AssociationDelta":
        created: list[dict[str, Any]] = []
        reinforced: list[dict[str, Any]] = []
        created_id_order = list(
            dict.fromkeys(int(value) for value in created_ids)
        )
        created_id_set = set(created_id_order)
        for association_id in created_id_order:
            if association_id in before:
                raise ValueError(
                    f"created association already existed before growth: {association_id}"
                )
            if association_id not in after:
                raise ValueError(
                    f"created association missing after growth: {association_id}"
                )
            created.append(
                {"id": association_id, "before": None, "after": after[association_id]}
            )
        # A multi-round query may create an edge in round one and reinforce it
        # in round two. It is still a creation relative to the pre-query state;
        # the created snapshot above already stores its final after-row.
        for association_id in dict.fromkeys(int(value) for value in reinforced_ids):
            if association_id in created_id_set:
                continue
            if association_id not in before or association_id not in after:
                raise ValueError(
                    f"reinforced association lacks a complete snapshot: {association_id}"
                )
            reinforced.append(
                {
                    "id": association_id,
                    "before": before[association_id],
                    "after": after[association_id],
                }
            )
        return cls(created=created, reinforced=reinforced)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssociationDelta":
        return cls(
            created=[dict(item) for item in value.get("created", [])],
            reinforced=[dict(item) for item in value.get("reinforced", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"created": self.created, "reinforced": self.reinforced}

    @property
    def created_ids(self) -> set[int]:
        return {int(item["id"]) for item in self.created}

    @property
    def reinforced_before(self) -> dict[int, dict[str, Any]]:
        return {
            int(item["id"]): dict(item["before"])
            for item in self.reinforced
        }


class AssociationOverlay:
    """Read-only counterfactual view over an AssociationRepository.

    Newly created rows may be hidden and reinforced rows replaced with their
    full pre-treatment snapshot.  No operation on this class writes SQLite.
    """

    def __init__(
        self,
        repository: AssociationRepository,
        hidden_ids: Iterable[int] = (),
        restored_rows: dict[int, dict[str, Any]] | None = None,
        hidden_context_cue_ids: Iterable[int] = (),
        hidden_need_cue_ids: Iterable[int] = (),
    ):
        self.repository = repository
        self.hidden_ids = {int(value) for value in hidden_ids}
        self.hidden_context_cue_ids = {int(value) for value in hidden_context_cue_ids}
        self.hidden_need_cue_ids = {int(value) for value in hidden_need_cue_ids}
        self.restored_rows = {
            int(key): dict(value) for key, value in (restored_rows or {}).items()
        }

    @classmethod
    def from_delta(
        cls, repository: AssociationRepository, delta: AssociationDelta
    ) -> "AssociationOverlay":
        rows = [
            item.get("after", {})
            for item in delta.created
            if isinstance(item.get("after"), dict)
        ]
        return cls(
            repository,
            delta.created_ids,
            delta.reinforced_before,
            hidden_context_cue_ids=(
                int(row["context_cue_id"])
                for row in rows
                if row.get("context_cue_id") is not None
            ),
            hidden_need_cue_ids=(
                int(row["need_cue_id"])
                for row in rows
                if row.get("need_cue_id") is not None
            ),
        )

    def _visible(self, row) -> dict[str, Any] | None:
        association_id = int(row["id"])
        if association_id in self.hidden_ids:
            return None
        return dict(self.restored_rows.get(association_id, dict(row)))

    def neighbors(self, node_type, node_id: int, limit: int = 100):
        # Fetch all incident rows before filtering so hidden high-weight rows do
        # not reduce the requested visible fan-out.
        rows = self.repository.neighbors(node_type, node_id, limit=2_147_483_647)
        visible = [item for row in rows if (item := self._visible(row)) is not None]
        visible.sort(
            key=lambda item: (float(item["weight"]), float(item["confidence"])),
            reverse=True,
        )
        return visible[:limit]

    def neighbors_many(self, node_type, node_ids, limit: int = 100):
        requested = list(dict.fromkeys(int(value) for value in node_ids))
        grouped = self.repository.neighbors_many(
            node_type, requested, limit=2_147_483_647
        )
        result = {}
        for node_id in requested:
            visible = [
                item
                for row in grouped.get(node_id, [])
                if (item := self._visible(row)) is not None
            ]
            visible.sort(
                key=lambda item: (
                    float(item["weight"]),
                    float(item["confidence"]),
                ),
                reverse=True,
            )
            result[node_id] = visible[:limit]
        return result

    def get(self, association_id: int):
        if int(association_id) in self.hidden_ids:
            return None
        restored = self.restored_rows.get(int(association_id))
        if restored is not None:
            return dict(restored)
        row = self.repository.get(int(association_id))
        return dict(row) if row is not None else None

    def list_temporal(self, timeline_scope: str):
        rows = self.repository.list_temporal(timeline_scope)
        visible = [item for row in rows if (item := self._visible(row)) is not None]
        visible.sort(key=lambda item: float(item["weight"]), reverse=True)
        return visible

    def mark_used(self, association_ids: list[int]) -> None:
        # Counterfactual replay and masked full queries must leave the treatment
        # database byte-for-byte unchanged at the Association layer.
        return None

    def get_contextual_for_prototypes(self, context_ids, need_ids, **kwargs):
        rows = self.repository.get_contextual_for_prototypes(
            context_ids, need_ids, **kwargs
        )
        return [
            row
            for row in rows
            if int(row["id"]) not in self.hidden_ids
            and int(row["context_cue_id"]) not in self.hidden_context_cue_ids
            and int(row["need_cue_id"]) not in self.hidden_need_cue_ids
        ]

    def record_utility(self, observations) -> dict[str, int]:
        return {"updated": 0, "successes": 0, "noops": 0, "harms": 0}

    def stats(self) -> dict[str, int]:
        rows = self.repository.snapshot()
        visible = [
            self.restored_rows.get(association_id, row)
            for association_id, row in rows.items()
            if association_id not in self.hidden_ids
        ]
        return {
            "edges": len(visible),
            "negative": sum(int(row["polarity"]) < 0 for row in visible),
        }

    def upsert(self, *_args, **_kwargs):
        raise RuntimeError("AssociationOverlay is read-only")

    def delete(self, *_args, **_kwargs):
        raise RuntimeError("AssociationOverlay is read-only")


class StagedAssociationOverlay:
    """Mutable in-memory Association view committed only after utility review.

    Growth can traverse newly proposed rows and provisional reinforcements, but
    SQLite remains unchanged until ``commit``. Created rows use negative IDs
    inside the query; the commit mapping replaces them with durable IDs.
    """

    def __init__(self, repository: AssociationRepository):
        self.repository = repository
        self.weights = repository.weights
        self._rows: dict[int, dict[str, Any]] = {}
        self._drafts: dict[int, list[AssociationDraft]] = {}
        self._created_ids: set[int] = set()
        self._reinforced_before: dict[int, dict[str, Any]] = {}
        self._deleted_ids: set[int] = set()
        self._used_ids: list[int] = []
        self._next_id = -1

    @staticmethod
    def _equivalent(row: dict[str, Any], draft: AssociationDraft) -> bool:
        if (
            str(row["relation_type"]) != draft.relation_type
            or str(row["relation_key"]) != draft.relation_key
            or int(row["polarity"]) != int(draft.polarity)
        ):
            return False
        left = (str(row["from_type"]), int(row["from_id"]))
        right = (str(row["to_type"]), int(row["to_id"]))
        proposed_left = (draft.from_type, int(draft.from_id))
        proposed_right = (draft.to_type, int(draft.to_id))
        if draft.relation_type == "semantic":
            return {left, right} == {proposed_left, proposed_right}
        return left == proposed_left and right == proposed_right

    def find_exact_id(self, draft: AssociationDraft) -> int | None:
        for association_id, row in self._rows.items():
            if association_id not in self._deleted_ids and self._equivalent(row, draft):
                return association_id
        durable_id = self.repository.find_exact_id(draft)
        if durable_id is not None and durable_id not in self._deleted_ids:
            return int(durable_id)
        return None

    def get(self, association_id: int):
        value = int(association_id)
        if value in self._deleted_ids:
            return None
        if value in self._rows:
            return dict(self._rows[value])
        row = self.repository.get(value)
        return dict(row) if row is not None else None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _new_row(self, association_id: int, draft: AssociationDraft) -> dict[str, Any]:
        now = self._now()
        return {
            "id": association_id,
            "from_type": draft.from_type,
            "from_id": int(draft.from_id),
            "to_type": draft.to_type,
            "to_id": int(draft.to_id),
            "relation_type": draft.relation_type,
            "relation_key": draft.relation_key,
            "relation_text": draft.relation_text,
            "polarity": int(draft.polarity),
            "weight": max(0.0, min(1.0, float(draft.weight))),
            "confidence": max(0.0, min(1.0, float(draft.confidence))),
            "generation": int(draft.generation),
            "evidence_count": 1,
            "claim_level": draft.claim_level,
            "audit_status": draft.audit_status,
            "evidence_json": draft.evidence_json,
            "audit_json": draft.audit_json,
            "created_reason": draft.created_reason,
            "association_mode": "semantic",
            "context_cue_id": None,
            "need_cue_id": None,
            "utility_weight": 0.0,
            "utility_successes": 0,
            "utility_noops": 0,
            "utility_harms": 0,
            "distinct_query_count": 0,
            "lifecycle_state": "active",
            "expires_at": None,
            "last_evaluated_at": None,
            "source_request_hash": "",
            "utility_query_hashes": "[]",
            "last_used": None,
            "use_count": 0,
            "created_at": now,
            "updated_at": now,
        }

    def _reinforced_row(
        self, row: dict[str, Any], draft: AssociationDraft
    ) -> dict[str, Any]:
        result = dict(row)
        signal = max(0.0, min(1.0, float(draft.weight)))
        result["weight"] = min(
            1.0,
            float(row["weight"])
            + self.weights.learning_rate * signal * (1.0 - float(row["weight"])),
        )
        result["confidence"] = max(
            float(row["confidence"]), float(draft.confidence)
        )
        current_level = str(row["claim_level"])
        current_generation = int(row["generation"])
        result["claim_level"] = AssociationRepository._stronger_claim_level(
            current_level, draft.claim_level
        )
        if not (
            AssociationRepository._claim_rank(current_level)
            > AssociationRepository._claim_rank(draft.claim_level)
            or (
                AssociationRepository._claim_rank(current_level)
                == AssociationRepository._claim_rank(draft.claim_level)
                and current_generation < int(draft.generation)
            )
        ):
            result["relation_text"] = draft.relation_text
        result["generation"] = min(current_generation, int(draft.generation))
        result["audit_status"] = (
            "dual_accepted"
            if "dual_accepted" in (str(row["audit_status"]), draft.audit_status)
            else "not_required"
        )
        result["evidence_json"] = AssociationRepository._merge_json_arrays(
            str(row["evidence_json"]), draft.evidence_json
        )
        result["audit_json"] = AssociationRepository._merge_json_arrays(
            str(row["audit_json"]), draft.audit_json
        )
        existing_reason = str(row.get("created_reason", "") or "").strip()
        proposed_reason = str(draft.created_reason or "").strip()
        if proposed_reason and proposed_reason not in existing_reason:
            result["created_reason"] = " | ".join(
                value for value in (existing_reason, proposed_reason) if value
            )
        result["evidence_count"] = int(row.get("evidence_count", 1)) + 1
        result["updated_at"] = self._now()
        return result

    def upsert(self, draft: AssociationDraft) -> int:
        if draft.from_type == draft.to_type and draft.from_id == draft.to_id:
            raise ValueError("self associations are not allowed")
        if isinstance(draft.generation, bool) or int(draft.generation) < 0:
            raise ValueError("generation must be a non-negative integer")
        existing_id = self.find_exact_id(draft)
        if existing_id is None:
            association_id = self._next_id
            self._next_id -= 1
            self._rows[association_id] = self._new_row(association_id, draft)
            self._drafts[association_id] = [draft]
            self._created_ids.add(association_id)
            return association_id
        association_id = int(existing_id)
        existing = self.get(association_id)
        if existing is None:
            raise ValueError("equivalent association disappeared from staged view")
        if association_id >= 0 and association_id not in self._reinforced_before:
            self._reinforced_before[association_id] = dict(existing)
        self._rows[association_id] = self._reinforced_row(existing, draft)
        self._drafts.setdefault(association_id, []).append(draft)
        return association_id

    def neighbors(self, node_type, node_id: int, limit: int = 100):
        rows: dict[int, dict[str, Any]] = {}
        for row in self.repository.neighbors(
            node_type, node_id, limit=2_147_483_647
        ):
            association_id = int(row["id"])
            if association_id in self._deleted_ids:
                continue
            rows[association_id] = dict(self._rows.get(association_id, dict(row)))
        for association_id, row in self._rows.items():
            if association_id in self._deleted_ids:
                continue
            if (
                str(row["from_type"]) == node_type
                and int(row["from_id"]) == int(node_id)
            ) or (
                str(row["to_type"]) == node_type
                and int(row["to_id"]) == int(node_id)
            ):
                rows[association_id] = dict(row)
        values = sorted(
            rows.values(),
            key=lambda item: (float(item["weight"]), float(item["confidence"])),
            reverse=True,
        )
        return values[:limit]

    def neighbors_many(self, node_type, node_ids, limit: int = 100):
        requested = list(dict.fromkeys(int(value) for value in node_ids))
        durable = self.repository.neighbors_many(
            node_type, requested, limit=2_147_483_647
        )
        result: dict[int, list[dict[str, Any]]] = {}
        for node_id in requested:
            rows: dict[int, dict[str, Any]] = {}
            for row in durable.get(node_id, []):
                association_id = int(row["id"])
                if association_id in self._deleted_ids:
                    continue
                rows[association_id] = dict(
                    self._rows.get(association_id, dict(row))
                )
            for association_id, row in self._rows.items():
                if association_id in self._deleted_ids:
                    continue
                if (
                    str(row["from_type"]) == node_type
                    and int(row["from_id"]) == node_id
                ) or (
                    str(row["to_type"]) == node_type
                    and int(row["to_id"]) == node_id
                ):
                    rows[association_id] = dict(row)
            result[node_id] = sorted(
                rows.values(),
                key=lambda item: (
                    float(item["weight"]),
                    float(item["confidence"]),
                ),
                reverse=True,
            )[:limit]
        return result

    def list_temporal(self, timeline_scope: str):
        rows = {
            int(row["id"]): dict(row)
            for row in self.repository.list_temporal(timeline_scope)
            if int(row["id"]) not in self._deleted_ids
        }
        for association_id, row in self._rows.items():
            if association_id in self._deleted_ids:
                continue
            if str(row["relation_type"]) == "temporal":
                rows[association_id] = dict(row)
        return sorted(rows.values(), key=lambda item: float(item["weight"]), reverse=True)

    def concept_reachable_episode_counts(self, concept_ids: list[int]):
        return self.repository.concept_reachable_episode_counts(concept_ids)

    def list_contextual(self, state: str | None = None):
        rows = {
            int(row["id"]): dict(row)
            for row in self.repository.list_contextual(state)
            if int(row["id"]) not in self._deleted_ids
        }
        for association_id, row in self._rows.items():
            if association_id in self._deleted_ids:
                continue
            if str(row.get("association_mode", "")) != "contextual_recall":
                continue
            if state and str(row.get("lifecycle_state", "")) != str(state):
                continue
            rows[association_id] = dict(row)
        return sorted(rows.values(), key=lambda item: int(item["id"]))

    def get_contextual_for_prototypes(self, context_ids, need_ids, **kwargs):
        rows = self.repository.get_contextual_for_prototypes(
            context_ids, need_ids, **kwargs
        )
        visible = {
            int(row["id"]): dict(row)
            for row in rows
            if int(row["id"]) not in self._deleted_ids
        }
        context_set = {int(value) for value in context_ids}
        need_set = {int(value) for value in need_ids}
        for association_id, row in self._rows.items():
            if association_id in self._deleted_ids:
                continue
            if (
                str(row.get("association_mode", "")) == "contextual_recall"
                and int(row.get("context_cue_id", -1)) in context_set
                and int(row.get("need_cue_id", -1)) in need_set
            ):
                visible[association_id] = dict(row)
        return sorted(
            visible.values(),
            key=lambda item: (
                -float(item.get("utility_weight", 0.0)),
                int(item["id"]),
            ),
        )[: int(kwargs.get("limit", 1000))]

    def record_utility(self, _observations):
        return {"updated": 0, "successes": 0, "noops": 0, "harms": 0}

    def snapshot(self, association_ids: list[int] | None = None) -> dict[int, dict]:
        if association_ids is None:
            rows = self.repository.snapshot()
            rows.update(
                {
                    association_id: dict(row)
                    for association_id, row in self._rows.items()
                    if association_id not in self._deleted_ids
                }
            )
            for association_id in self._deleted_ids:
                rows.pop(association_id, None)
            return rows
        result: dict[int, dict] = {}
        for association_id in dict.fromkeys(int(value) for value in association_ids):
            row = self.get(association_id)
            if row is not None:
                result[association_id] = dict(row)
        return result

    def delete(self, association_id: int) -> bool:
        value = int(association_id)
        if value in self._created_ids:
            existed = value in self._rows
            self._rows.pop(value, None)
            self._drafts.pop(value, None)
            self._created_ids.discard(value)
            self._deleted_ids.add(value)
            return existed
        if value in self._rows:
            self._rows.pop(value, None)
            self._drafts.pop(value, None)
            self._deleted_ids.add(value)
            return True
        return False

    def restore_rows(self, rows: dict[int, dict]) -> list[int]:
        restored: list[int] = []
        for association_id in rows:
            value = int(association_id)
            if value in self._rows or value in self._drafts:
                self._rows.pop(value, None)
                self._drafts.pop(value, None)
                self._deleted_ids.discard(value)
                restored.append(value)
        return restored

    def mark_used(self, association_ids: list[int]) -> None:
        self._used_ids.extend(int(value) for value in association_ids)

    def stats(self) -> dict[str, int]:
        rows = self.snapshot()
        return {
            "edges": len(rows),
            "negative": sum(int(row["polarity"]) < 0 for row in rows.values()),
        }

    @staticmethod
    def _remap_json_ids(raw: str, mapping: dict[int, int]) -> str:
        try:
            payload = json.loads(raw or "[]")
        except (TypeError, json.JSONDecodeError):
            return raw

        def visit(value, parent: dict | None = None, key: str | None = None):
            if isinstance(value, dict):
                return {item_key: visit(item, value, item_key) for item_key, item in value.items()}
            if isinstance(value, list):
                return [visit(item, parent, key) for item in value]
            if isinstance(value, int):
                if key in {"association_id"}:
                    return mapping.get(value, value)
                if key == "id" and parent and parent.get("type") == "association":
                    return mapping.get(value, value)
                if key == "premise_association_ids":
                    return mapping.get(value, value)
            return value

        return json.dumps(visit(payload), ensure_ascii=False)

    def commit(self) -> dict[int, int]:
        """Commit the post-gate staged rows and return temporary→durable IDs."""
        mapping: dict[int, int] = {}
        ordered = sorted(
            self._drafts,
            key=lambda association_id: (
                int(self._rows.get(association_id, {}).get("generation", 0)),
                association_id >= 0,
                association_id,
            ),
        )
        for association_id in ordered:
            if association_id in self._deleted_ids:
                continue
            durable_id: int | None = None
            for draft in self._drafts.get(association_id, []):
                committed_draft = replace(
                    draft,
                    evidence_json=self._remap_json_ids(draft.evidence_json, mapping),
                    audit_json=self._remap_json_ids(draft.audit_json, mapping),
                )
                durable_id = self.repository.upsert(committed_draft)
            if durable_id is not None:
                mapping[association_id] = int(durable_id)
        used = [mapping.get(value, value) for value in self._used_ids]
        self.repository.mark_used(list(dict.fromkeys(used)))
        return mapping
