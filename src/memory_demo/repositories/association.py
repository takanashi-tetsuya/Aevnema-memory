from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping

from memory_demo.config import WeightConfig
from memory_demo.database import Database, utc_now
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.types import (
    AssociationDraft,
    ContextualRevisitContract,
    ContextualRevisitContractLookup,
    ContextualRevisitContractReceipt,
    ContextualRevisitRuntimeManifest,
    ContextualRevisitRuntimeManifestLookup,
    ContextualRevisitRuntimeManifestReceipt,
    ContextualRevisitRuntimeSeed,
    ContextualRevisitSlotNeedBinding,
    ContextualRestrictedRewriteGuard,
    ContextualRestrictedRewriteGuardDraft,
    ContextualRestrictedRewriteGuardLookup,
    ContextualUtilityLedgerObservation,
    ContextualUtilityLedgerReceipt,
    NodeType,
    SourceFactRef,
)


def normalize_evaluation_as_of(value: str | None) -> str:
    """Return the canonical UTC instant used by a read-only v3 lookup.

    The contextual matcher must receive this value from its request/replay
    contract.  Keeping it explicit prevents a historical replay from quietly
    depending on ``utc_now()`` while expiry filtering a contextual edge.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("evaluation_as_of is required")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as error:
        raise ValueError("evaluation_as_of must be an ISO-8601 UTC instant") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("evaluation_as_of must be UTC")
    return parsed.astimezone(timezone.utc).isoformat()


class TemporalCycleError(ValueError):
    """Raised when a positive temporal edge would make chronology cyclic."""

    def __init__(self, earlier_id: int, later_id: int):
        self.earlier_id = int(earlier_id)
        self.later_id = int(later_id)
        super().__init__(
            f"temporal edge {self.earlier_id} -> {self.later_id} would close a cycle: "
            f"{self.later_id} already reaches {self.earlier_id}"
        )


class RestrictedRewriteReadyBindingError(ValueError):
    """A V22 guarded receipt may not publish without its private signer."""


_RESTRICTED_REWRITE_GUARD_V5 = "contextual-restricted-rewrite-guard-v5"
_RESTRICTED_REWRITE_LEGACY_SIGNATURE_VERSIONS = frozenset(
    f"contextual-restricted-rewrite-guard-v{version}" for version in range(1, 5)
)


class AssociationRepository:
    def __init__(
        self,
        db: Database,
        weights: WeightConfig | None = None,
        *,
        contextual_noop_decay: float = 0.98,
        contextual_harm_multiplier: float = 0.50,
        contextual_positive_rate: float = 0.20,
        contextual_min_distinct_successes: int = 2,
        contextual_promotion_enabled: bool = False,
    ):
        self.db = db
        self.weights = weights or WeightConfig()
        self.contextual_noop_decay = max(0.0, min(1.0, float(contextual_noop_decay)))
        self.contextual_harm_multiplier = max(
            0.0, min(1.0, float(contextual_harm_multiplier))
        )
        self.contextual_positive_rate = max(
            0.0, min(1.0, float(contextual_positive_rate))
        )
        self.contextual_min_distinct_successes = max(
            1, int(contextual_min_distinct_successes)
        )
        self.contextual_promotion_enabled = bool(contextual_promotion_enabled)
        self._concept_reach_cache: dict[int, int] = {}

    @staticmethod
    def _find_equivalent(connection, draft: AssociationDraft):
        """Find the durable edge represented by a draft.

        Most relations are directional. Semantic relations with the same stable
        key are treated as undirected for deduplication because query-time LLMs
        can legitimately describe the same thematic bridge from either end.
        Different semantic keys remain distinct: a contrast and an evidence
        bridge between the same nodes may both carry useful information.
        """
        if draft.relation_type == "semantic":
            return connection.execute(
                """
                SELECT * FROM association
                WHERE relation_type = ? AND relation_key = ? AND polarity = ?
                  AND (
                    (from_type = ? AND from_id = ? AND to_type = ? AND to_id = ?)
                    OR
                    (from_type = ? AND from_id = ? AND to_type = ? AND to_id = ?)
                  )
                ORDER BY id
                LIMIT 1
                """,
                (
                    draft.relation_type,
                    draft.relation_key,
                    draft.polarity,
                    draft.from_type,
                    draft.from_id,
                    draft.to_type,
                    draft.to_id,
                    draft.to_type,
                    draft.to_id,
                    draft.from_type,
                    draft.from_id,
                ),
            ).fetchone()
        return connection.execute(
            """
            SELECT * FROM association
            WHERE from_type = ? AND from_id = ? AND to_type = ? AND to_id = ?
              AND relation_type = ? AND relation_key = ? AND polarity = ?
            ORDER BY id
            LIMIT 1
            """,
            (
                draft.from_type,
                draft.from_id,
                draft.to_type,
                draft.to_id,
                draft.relation_type,
                draft.relation_key,
                draft.polarity,
            ),
        ).fetchone()

    @staticmethod
    def _node_exists(connection, node_type: NodeType, node_id: int) -> bool:
        table = "episode" if node_type == "episode" else "concept"
        return connection.execute(
            f"SELECT 1 FROM {table} WHERE id = ?", (node_id,)
        ).fetchone() is not None

    @staticmethod
    def _claim_rank(value: str) -> int:
        return {
            "retrieval_only": 0,
            "historical_context": 1,
            "supported_inference": 2,
            "direct_fact": 3,
        }.get(str(value), 0)

    @classmethod
    def _stronger_claim_level(cls, current: str, proposed: str) -> str:
        return (
            current
            if cls._claim_rank(current) >= cls._claim_rank(proposed)
            else proposed
        )

    @staticmethod
    def _merge_json_arrays(current: str, proposed: str) -> str:
        merged: list = []
        fingerprints: set[str] = set()
        for raw in (current, proposed):
            try:
                values = json.loads(raw or "[]")
            except (TypeError, json.JSONDecodeError):
                values = []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                fingerprint = json.dumps(
                    value, ensure_ascii=False, sort_keys=True
                )
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    merged.append(value)
        return json.dumps(merged, ensure_ascii=False)

    @staticmethod
    def _positive_temporal_endpoints(
        draft: AssociationDraft,
    ) -> tuple[int, int] | None:
        """Return canonical ``earlier, later`` endpoints for ordering edges."""
        if (
            draft.from_type != "episode"
            or draft.to_type != "episode"
            or draft.relation_type != "temporal"
            or int(draft.polarity) <= 0
        ):
            return None
        key = str(draft.relation_key).casefold()
        if key == "after" or key.endswith("_after"):
            return int(draft.to_id), int(draft.from_id)
        if key in {"before", "precedes"} or key.endswith("_before"):
            return int(draft.from_id), int(draft.to_id)
        return None

    @staticmethod
    def _temporal_reaches(
        connection, start_id: int, target_id: int
    ) -> bool:
        """Check positive temporal reachability inside the current transaction.

        ``UNION`` deduplicates visited nodes, so this remains safe when auditing
        a legacy database that already contains a cycle.
        """
        row = connection.execute(
            """
            WITH RECURSIVE temporal_edges(earlier_id, later_id) AS (
                SELECT from_id, to_id
                FROM association
                WHERE from_type = 'episode' AND to_type = 'episode'
                  AND relation_type = 'temporal' AND polarity > 0
                  AND (
                      LOWER(relation_key) IN ('before', 'precedes')
                      OR SUBSTR(LOWER(relation_key), -7) = '_before'
                  )
                UNION ALL
                SELECT to_id, from_id
                FROM association
                WHERE from_type = 'episode' AND to_type = 'episode'
                  AND relation_type = 'temporal' AND polarity > 0
                  AND (
                      LOWER(relation_key) = 'after'
                      OR SUBSTR(LOWER(relation_key), -6) = '_after'
                  )
            ), reachable(node_id) AS (
                SELECT ?
                UNION
                SELECT edge.later_id
                FROM temporal_edges AS edge
                JOIN reachable ON edge.earlier_id = reachable.node_id
            )
            SELECT 1 FROM reachable WHERE node_id = ? LIMIT 1
            """,
            (int(start_id), int(target_id)),
        ).fetchone()
        return row is not None

    @staticmethod
    def _validate_upsert_draft(draft: AssociationDraft) -> int:
        """Validate the public upsert contract and return its generation.

        Keeping this validation outside the transaction wrapper lets a staged
        query commit several dependent drafts in *one* durable transaction
        without creating a weaker second implementation of association
        semantics.
        """

        if draft.from_type == draft.to_type and draft.from_id == draft.to_id:
            raise ValueError("self associations are not allowed")
        if isinstance(draft.generation, bool):
            raise ValueError("generation must be a non-negative integer")
        generation = int(draft.generation)
        if generation < 0 or generation != draft.generation:
            raise ValueError("generation must be a non-negative integer")
        return generation

    def _upsert_in_transaction(
        self,
        connection,
        draft: AssociationDraft,
        *,
        now: str | None = None,
    ) -> int:
        """Apply one draft using an already-open write transaction.

        This is deliberately an internal primitive for
        :class:`StagedAssociationOverlay`: a staged commit must not open one
        SQLite transaction per draft, because a later failure would otherwise
        leave a partially learned graph behind.
        """

        generation = self._validate_upsert_draft(draft)
        timestamp = now or utc_now()
        if not self._node_exists(connection, draft.from_type, draft.from_id):
            raise ValueError("from node does not exist")
        if not self._node_exists(connection, draft.to_type, draft.to_id):
            raise ValueError("to node does not exist")
        existing = self._find_equivalent(connection, draft)
        if existing is None:
            endpoints = self._positive_temporal_endpoints(draft)
            if endpoints is not None:
                earlier_id, later_id = endpoints
                if self._temporal_reaches(connection, later_id, earlier_id):
                    raise TemporalCycleError(earlier_id, later_id)
            cursor = connection.execute(
                """
                INSERT INTO association(
                    from_type, from_id, to_type, to_id,
                    relation_type, relation_key, relation_text, polarity,
                    weight, confidence, generation, claim_level, audit_status,
                    evidence_json, audit_json, created_reason, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.from_type,
                    draft.from_id,
                    draft.to_type,
                    draft.to_id,
                    draft.relation_type,
                    draft.relation_key,
                    draft.relation_text,
                    draft.polarity,
                    max(0.0, min(1.0, draft.weight)),
                    max(0.0, min(1.0, draft.confidence)),
                    generation,
                    draft.claim_level,
                    draft.audit_status,
                    draft.evidence_json,
                    draft.audit_json,
                    draft.created_reason,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)
        signal = max(0.0, min(1.0, draft.weight))
        reinforced = float(existing["weight"]) + self.weights.learning_rate * signal * (
            1.0 - float(existing["weight"])
        )
        current_level = str(existing["claim_level"])
        current_generation = int(existing["generation"])
        merged_level = self._stronger_claim_level(
            current_level, draft.claim_level
        )
        relation_text = (
            str(existing["relation_text"])
            if self._claim_rank(current_level)
            > self._claim_rank(draft.claim_level)
            or (
                self._claim_rank(current_level)
                == self._claim_rank(draft.claim_level)
                and current_generation < generation
            )
            else draft.relation_text
        )
        audit_status = (
            "dual_accepted"
            if "dual_accepted"
            in (str(existing["audit_status"]), draft.audit_status)
            else "not_required"
        )
        existing_reason = str(existing["created_reason"] or "").strip()
        proposed_reason = str(draft.created_reason or "").strip()
        created_reason = existing_reason
        if proposed_reason and proposed_reason not in existing_reason:
            created_reason = " | ".join(
                value for value in (existing_reason, proposed_reason) if value
            )
        connection.execute(
            """
            UPDATE association SET
                relation_text = ?, weight = ?,
                confidence = MAX(confidence, ?),
                generation = MIN(generation, ?),
                claim_level = ?, audit_status = ?,
                evidence_json = ?, audit_json = ?,
                created_reason = ?,
                evidence_count = evidence_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (
                relation_text,
                min(1.0, reinforced),
                draft.confidence,
                generation,
                merged_level,
                audit_status,
                self._merge_json_arrays(
                    str(existing["evidence_json"]), draft.evidence_json
                ),
                self._merge_json_arrays(
                    str(existing["audit_json"]), draft.audit_json
                ),
                created_reason,
                timestamp,
                existing["id"],
            ),
        )
        return int(existing["id"])

    @staticmethod
    def _mark_used_in_transaction(connection, association_ids: list[int]) -> None:
        if not association_ids:
            return
        placeholders = ",".join("?" for _ in association_ids)
        timestamp = utc_now()
        connection.execute(
            f"""
            UPDATE association
            SET use_count = use_count + 1, last_used = ?, updated_at = ?
            WHERE id IN ({placeholders})
            """,
            [timestamp, timestamp, *association_ids],
        )

    def upsert(self, draft: AssociationDraft) -> int:
        self._validate_upsert_draft(draft)
        # A new topology edge can change both the directly touched Concept and
        # one-hop neighbors. Clearing this small derived cache is safer than
        # attempting partial invalidation.
        self._concept_reach_cache.clear()
        with self.db.transaction() as connection:
            return self._upsert_in_transaction(connection, draft)

    def insert_new_non_temporal_many(
        self, drafts: list[AssociationDraft]
    ) -> list[int]:
        """Insert known-new direct edges in one SQLite transaction.

        Importing a newly allocated Episode cannot reinforce a pre-existing
        Episode-to-Concept edge.  The general ``upsert`` path performs several
        reads and opens one transaction per edge; this narrower contract keeps
        those checks out of the hot import loop while retaining database
        uniqueness as a final guard.
        """

        if not drafts:
            return []
        seen: set[tuple[str, int, str, int, str, str, int]] = set()
        for draft in drafts:
            if draft.from_type == draft.to_type and draft.from_id == draft.to_id:
                raise ValueError("self associations are not allowed")
            if self._positive_temporal_endpoints(draft) is not None:
                raise ValueError("temporal edges must use cycle-audited upsert")
            if isinstance(draft.generation, bool):
                raise ValueError("generation must be a non-negative integer")
            generation = int(draft.generation)
            if generation < 0 or generation != draft.generation:
                raise ValueError("generation must be a non-negative integer")
            key = (
                draft.from_type,
                int(draft.from_id),
                draft.to_type,
                int(draft.to_id),
                draft.relation_type,
                draft.relation_key,
                int(draft.polarity),
            )
            if key in seen:
                raise ValueError("duplicate association in known-new batch")
            seen.add(key)

        self._concept_reach_cache.clear()
        now = utc_now()
        ids: list[int] = []
        with self.db.transaction() as connection:
            node_ids: dict[str, set[int]] = {"episode": set(), "concept": set()}
            for draft in drafts:
                node_ids[draft.from_type].add(int(draft.from_id))
                node_ids[draft.to_type].add(int(draft.to_id))
            for node_type, expected in node_ids.items():
                if not expected:
                    continue
                placeholders = ",".join("?" for _ in expected)
                table = "episode" if node_type == "episode" else "concept"
                found = {
                    int(row[0])
                    for row in connection.execute(
                        f"SELECT id FROM {table} WHERE id IN ({placeholders})",
                        sorted(expected),
                    )
                }
                if found != expected:
                    raise ValueError(f"one or more {node_type} nodes do not exist")

            for draft in drafts:
                cursor = connection.execute(
                    """
                    INSERT INTO association(
                        from_type, from_id, to_type, to_id,
                        relation_type, relation_key, relation_text, polarity,
                        weight, confidence, generation, claim_level, audit_status,
                        evidence_json, audit_json, created_reason, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        draft.from_type,
                        draft.from_id,
                        draft.to_type,
                        draft.to_id,
                        draft.relation_type,
                        draft.relation_key,
                        draft.relation_text,
                        draft.polarity,
                        max(0.0, min(1.0, draft.weight)),
                        max(0.0, min(1.0, draft.confidence)),
                        int(draft.generation),
                        draft.claim_level,
                        draft.audit_status,
                        draft.evidence_json,
                        draft.audit_json,
                        draft.created_reason,
                        now,
                        now,
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def find_exact_id(self, draft: AssociationDraft) -> int | None:
        """Return an existing edge with the same durable fingerprint, if any."""
        with self.db.connection() as connection:
            row = self._find_equivalent(connection, draft)
        return int(row["id"]) if row is not None else None

    def neighbors(self, node_type: NodeType, node_id: int, limit: int = 100):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT * FROM association
                WHERE (from_type = ? AND from_id = ?)
                   OR (to_type = ? AND to_id = ?)
                ORDER BY weight DESC, confidence DESC
                LIMIT ?
                """,
                (node_type, node_id, node_type, node_id, limit),
            ).fetchall()

    def neighbors_many(
        self,
        node_type: NodeType,
        node_ids,
        limit: int = 100,
    ) -> dict[int, list]:
        """Return incident rows for many same-type nodes with one connection.

        Query finalization and chronology often inspect the same bounded set of
        Episodes. Opening one SQLite connection per Episode is especially
        expensive when a Linux process reads a database from a mounted Windows
        filesystem, so group and apply the per-node limit in memory instead.
        """
        requested = list(dict.fromkeys(int(value) for value in node_ids))
        result = {node_id: [] for node_id in requested}
        if not requested or limit <= 0:
            return result
        placeholders = ",".join("?" for _ in requested)
        with self.db.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM association
                WHERE (from_type = ? AND from_id IN ({placeholders}))
                   OR (to_type = ? AND to_id IN ({placeholders}))
                ORDER BY weight DESC, confidence DESC
                """,
                [node_type, *requested, node_type, *requested],
            ).fetchall()
        requested_set = set(requested)
        for row in rows:
            endpoints: set[int] = set()
            if str(row["from_type"]) == node_type:
                endpoints.add(int(row["from_id"]))
            if str(row["to_type"]) == node_type:
                endpoints.add(int(row["to_id"]))
            for endpoint in endpoints.intersection(requested_set):
                if len(result[endpoint]) < limit:
                    result[endpoint].append(row)
        return result

    def concept_reachable_episode_counts(
        self, concept_ids: list[int]
    ) -> dict[int, int]:
        """Count Episodes reachable directly or through one Concept neighbor."""
        requested_ids = list(dict.fromkeys(int(value) for value in concept_ids))
        if not requested_ids:
            return {}
        ids = [
            concept_id
            for concept_id in requested_ids
            if concept_id not in self._concept_reach_cache
        ]
        if not ids:
            return {
                concept_id: self._concept_reach_cache[concept_id]
                for concept_id in requested_ids
            }
        placeholders = ",".join("?" for _ in ids)
        with self.db.connection() as connection:
            first_hop = connection.execute(
                f"""
                SELECT from_type, from_id, to_type, to_id
                FROM association
                WHERE (from_type = 'concept' AND from_id IN ({placeholders}))
                   OR (to_type = 'concept' AND to_id IN ({placeholders}))
                """,
                [*ids, *ids],
            ).fetchall()
            reachable: dict[int, set[int]] = {concept_id: set() for concept_id in ids}
            roots_by_neighbor: dict[int, set[int]] = {
                concept_id: {concept_id} for concept_id in ids
            }
            requested = set(ids)
            for row in first_hop:
                from_type = str(row["from_type"])
                to_type = str(row["to_type"])
                from_id = int(row["from_id"])
                to_id = int(row["to_id"])
                root_id = from_id if from_type == "concept" else to_id
                if root_id not in requested:
                    continue
                if from_type == "episode":
                    reachable[root_id].add(from_id)
                elif to_type == "episode":
                    reachable[root_id].add(to_id)
                elif from_type == to_type == "concept":
                    neighbor_id = to_id if from_id == root_id else from_id
                    roots_by_neighbor.setdefault(neighbor_id, set()).add(root_id)
            neighbor_ids = list(roots_by_neighbor)
            if neighbor_ids:
                neighbor_placeholders = ",".join("?" for _ in neighbor_ids)
                episode_rows = connection.execute(
                    f"""
                    SELECT from_type, from_id, to_type, to_id
                    FROM association
                    WHERE (
                        from_type = 'concept'
                        AND from_id IN ({neighbor_placeholders})
                        AND to_type = 'episode'
                    ) OR (
                        to_type = 'concept'
                        AND to_id IN ({neighbor_placeholders})
                        AND from_type = 'episode'
                    )
                    """,
                    [*neighbor_ids, *neighbor_ids],
                ).fetchall()
                for row in episode_rows:
                    neighbor_id = (
                        int(row["from_id"])
                        if str(row["from_type"]) == "concept"
                        else int(row["to_id"])
                    )
                    episode_id = (
                        int(row["from_id"])
                        if str(row["from_type"]) == "episode"
                        else int(row["to_id"])
                    )
                    for root_id in roots_by_neighbor.get(neighbor_id, ()):
                        reachable[root_id].add(episode_id)
        computed = {
            concept_id: len(episode_ids)
            for concept_id, episode_ids in reachable.items()
        }
        self._concept_reach_cache.update(computed)
        return {
            concept_id: self._concept_reach_cache.get(concept_id, 0)
            for concept_id in requested_ids
        }

    def get(self, association_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                "SELECT * FROM association WHERE id = ?", (association_id,)
            ).fetchone()

    def list_cue_candidates(self):
        """Return audited learned relation text eligible for cue indexing."""
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT
                    a.*,
                    source_episode.participants_json AS from_participants_json,
                    target_episode.participants_json AS to_participants_json
                FROM association AS a
                LEFT JOIN episode AS source_episode
                  ON a.from_type = 'episode'
                 AND a.from_id = source_episode.id
                LEFT JOIN episode AS target_episode
                  ON a.to_type = 'episode'
                 AND a.to_id = target_episode.id
                WHERE a.audit_status = 'dual_accepted'
                  AND a.relation_key <> 'involves'
                  AND a.relation_text <> ''
                  AND a.created_reason LIKE '%查询中自主增长：%'
                ORDER BY a.id
                """
            ).fetchall()

    def store_cue_embedding(
        self,
        association_id: int,
        relation_text: str,
        cue_text: str,
        embedding: bytes,
    ) -> None:
        """Persist the vector together with the exact text it represents."""

        with self.db.transaction() as connection:
            connection.execute(
                    """
                    UPDATE association
                SET cue_embedding = ?, cue_embedding_text = ?
                WHERE id = ? AND relation_text = ?
                """,
                (
                    embedding,
                    cue_text,
                    int(association_id),
                    relation_text,
                ),
            )

    # ---- Contextual association v1 -------------------------------------
    # These methods are deliberately repository-level and batch-oriented.  A
    # matcher receives only vectors and rows; it never reaches a model client.

    @staticmethod
    def _cue_blob(vector, dimension: int) -> bytes:
        array = Database.validate_float32_vector(
            vector, dimension, require_normalized=True
        )
        return array.tobytes()

    @staticmethod
    def _default_embedding_space_id(model_id: str, dimension: int) -> str:
        """Stable legacy-compatible full identity for a float32 cue space."""

        payload = json.dumps(
            {
                "model_id": str(model_id),
                "dimension": int(dimension),
                "dtype": "float32",
                "normalization": "l2_float32_v1",
                "preprocessing": "legacy_contextual_cue_v1",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "contextual-cue-space:sha256:" + hashlib.sha256(payload).hexdigest()

    @classmethod
    def _normalize_embedding_space_id(
        cls,
        *,
        model_id: str,
        dimension: int,
        embedding_space_id: str = "",
        embedding_space=None,
    ) -> str:
        supplied = str(embedding_space_id or "").strip()
        object_id = str(
            getattr(embedding_space, "canonical_id", "") or ""
        ).strip()
        if supplied and object_id and supplied != object_id:
            raise ValueError("embedding space identity conflicts with object")
        return supplied or object_id or cls._default_embedding_space_id(
            model_id, dimension
        )

    def _get_or_create_cue_prototype_in_transaction(
        self,
        connection,
        *,
        domain: str,
        cue_kind: str,
        model_id: str,
        dimension: int,
        vector,
        text_hash: str,
        embedding_space_id: str = "",
        display_text: str = "",
        source_request_hash: str = "",
    ) -> int:
        domain = str(domain or "").strip()
        cue_kind = str(cue_kind or "").strip().casefold()
        model_id = str(model_id or "").strip()
        dimension = int(dimension)
        if not domain or cue_kind not in {"context", "need"} or not model_id:
            raise ValueError("domain, cue_kind and model_id are required")
        if dimension <= 0 or not str(text_hash or "").strip():
            raise ValueError("cue dimension and text_hash are required")
        space_id = self._normalize_embedding_space_id(
            model_id=model_id,
            dimension=dimension,
            embedding_space_id=embedding_space_id,
        )
        blob = self._cue_blob(vector, dimension)
        row = connection.execute(
            """
            SELECT id, dimension, dtype, vector_blob, embedding_space_id
            FROM association_cue_prototype
            WHERE domain = ? AND cue_kind = ? AND model_id = ? AND text_hash = ?
            """,
            (domain, cue_kind, model_id, str(text_hash)),
        ).fetchone()
        if row is not None:
            if (
                int(row["dimension"]) != dimension
                or str(row["dtype"]) != "float32"
            ):
                raise ValueError("existing cue prototype has incompatible shape")
            stored_space_id = str(row["embedding_space_id"] or "").strip()
            if stored_space_id and stored_space_id != space_id:
                raise ValueError("existing cue prototype has a different embedding space")
            if not stored_space_id and space_id != self._default_embedding_space_id(
                model_id, dimension
            ):
                # A v13 cue lacks full space metadata.  It is safe only for
                # the deterministic legacy identity, never for a claimed
                # revision/preprocessing-specific v3 vector space.
                raise ValueError("existing cue prototype has unknown embedding space")
            if bytes(row["vector_blob"]) != blob:
                raise ValueError("cue text hash already has a different vector")
            return int(row["id"])
        cursor = connection.execute(
            """
            INSERT INTO association_cue_prototype(
                domain, cue_kind, model_id, dimension, dtype, vector_blob,
                text_hash, embedding_space_id, display_text, source_request_hash,
                created_at
            ) VALUES(?, ?, ?, ?, 'float32', ?, ?, ?, ?, ?, ?)
            """,
            (
                domain,
                cue_kind,
                model_id,
                dimension,
                blob,
                str(text_hash),
                space_id,
                str(display_text or "")[:500],
                str(source_request_hash or ""),
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)

    def get_or_create_cue_prototype(
        self,
        *,
        domain: str,
        cue_kind: str,
        model_id: str,
        dimension: int,
        vector,
        text_hash: str,
        embedding_space_id: str = "",
        display_text: str = "",
        source_request_hash: str = "",
    ) -> int:
        with self.db.transaction() as connection:
            return self._get_or_create_cue_prototype_in_transaction(
                connection,
                domain=domain,
                cue_kind=cue_kind,
                model_id=model_id,
                dimension=dimension,
                vector=vector,
                text_hash=text_hash,
                embedding_space_id=embedding_space_id,
                display_text=display_text,
                source_request_hash=source_request_hash,
            )

    def list_cue_prototypes(
        self, *, domain: str | None = None, cue_kind: str | None = None
    ) -> list:
        clauses: list[str] = []
        params: list[object] = []
        if domain:
            clauses.append("domain = ?")
            params.append(str(domain))
        if cue_kind:
            normalized = str(cue_kind).casefold()
            if normalized not in {"context", "need"}:
                raise ValueError("cue_kind must be context or need")
            clauses.append("cue_kind = ?")
            params.append(normalized)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.db.connection() as connection:
            return connection.execute(
                f"SELECT * FROM association_cue_prototype{where} ORDER BY id",
                params,
            ).fetchall()

    def load_contextual_cue_pair(
        self,
        *,
        context_cue_id: int,
        need_cue_id: int,
        domain: str,
    ) -> tuple:
        """Load exactly one domain-bound context/need pair for local revisit.

        This deliberately avoids using ``list_cue_prototypes`` on a hot Q2
        path: an automatic revisit needs two current blobs, never every cue
        vector in a large imported domain.  Callers still must validate all
        returned metadata and durable manifest fingerprints themselves.
        """

        def positive_id(value: object, field_name: str) -> int:
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be a positive integer")
            try:
                normalized = int(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"{field_name} must be a positive integer"
                ) from error
            if normalized <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            return normalized

        context_id = positive_id(context_cue_id, "context_cue_id")
        need_id = positive_id(need_cue_id, "need_cue_id")
        if context_id == need_id:
            raise ValueError("context and need cue ids must differ")
        normalized_domain = str(domain or "").strip()
        if (
            not normalized_domain
            or "\n" in normalized_domain
            or "\r" in normalized_domain
            or "," in normalized_domain
        ):
            raise ValueError("cue pair domain is invalid")
        with self.db.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM association_cue_prototype
                WHERE domain = ? AND id IN (?, ?)
                ORDER BY id
                """,
                (normalized_domain, context_id, need_id),
            ).fetchall()
        return tuple(rows)

    # Explicit names used by the contextual-association maintenance API.
    insert_cue_prototype = get_or_create_cue_prototype

    def _create_contextual_in_transaction(
        self,
        connection,
        candidate,
        *,
        context_cue_id: int,
        need_cue_id: int,
        utility_weight: float = 0.20,
        probation_ttl: int = 2_592_000,
        expected_domain: str | None = None,
        expected_model_id: str | None = None,
        expected_dimension: int | None = None,
        expected_embedding_space_id: str | None = None,
    ) -> tuple[int, bool]:
        """Create/reuse one contextual edge inside an existing transaction."""

        if candidate.anchor_type not in {"episode", "concept"}:
            raise ValueError("invalid contextual anchor type")
        if int(candidate.target_episode_id) <= 0:
            raise ValueError("contextual target must be an Episode")
        if (
            int(candidate.anchor_id) == int(candidate.target_episode_id)
            and candidate.anchor_type == "episode"
        ):
            raise ValueError("contextual self-edge is not allowed")
        utility_weight = max(0.0, min(1.0, float(utility_weight)))
        expires_at = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + max(0, int(probation_ttl)),
            timezone.utc,
        ).isoformat()

        context = connection.execute(
            """
            SELECT id, domain, cue_kind, model_id, dimension, dtype,
                   embedding_space_id
            FROM association_cue_prototype
            WHERE id = ?
            """,
            (int(context_cue_id),),
        ).fetchone()
        need = connection.execute(
            """
            SELECT id, domain, cue_kind, model_id, dimension, dtype,
                   embedding_space_id
            FROM association_cue_prototype
            WHERE id = ?
            """,
            (int(need_cue_id),),
        ).fetchone()
        if (
            context is None
            or need is None
            or str(context["cue_kind"]) != "context"
            or str(need["cue_kind"]) != "need"
            or str(context["domain"]) != str(need["domain"])
            or str(context["model_id"]) != str(need["model_id"])
            or int(context["dimension"]) != int(need["dimension"])
            or str(context["dtype"]) != "float32"
            or str(need["dtype"]) != "float32"
            or str(context["embedding_space_id"] or "")
            != str(need["embedding_space_id"] or "")
        ):
            raise ValueError("context and need cues must share one valid model space")
        if expected_domain is not None and str(context["domain"]) != str(expected_domain):
            raise ValueError("contextual cue domain does not match creation request")
        if (
            expected_model_id is not None
            and str(context["model_id"]) != str(expected_model_id)
        ):
            raise ValueError("contextual cue model does not match creation request")
        if (
            expected_dimension is not None
            and int(context["dimension"]) != int(expected_dimension)
        ):
            raise ValueError("contextual cue dimension does not match creation request")
        if (
            expected_embedding_space_id is not None
            and str(context["embedding_space_id"] or "")
            != str(expected_embedding_space_id)
        ):
            raise ValueError("contextual cue embedding space does not match creation request")

        table = "episode" if candidate.anchor_type == "episode" else "concept"
        if connection.execute(
            f"SELECT 1 FROM {table} WHERE id = ?", (int(candidate.anchor_id),)
        ).fetchone() is None:
            raise ValueError("contextual anchor does not exist")
        if candidate.anchor_type == "episode":
            anchor_row = connection.execute(
                "SELECT generation FROM episode WHERE id = ?",
                (int(candidate.anchor_id),),
            ).fetchone()
            if anchor_row is None or int(anchor_row["generation"]) != 0:
                raise ValueError("contextual anchor Episode must be generation 0")
        target_row = connection.execute(
            """
            SELECT e.generation, e.source_id, s.id AS source_exists
            FROM episode e
            LEFT JOIN source s ON s.id = e.source_id
            WHERE e.id = ?
            """,
            (int(candidate.target_episode_id),),
        ).fetchone()
        if target_row is None or target_row["source_exists"] is None:
            raise ValueError("contextual target Episode/source does not exist")
        if int(target_row["generation"]) != 0:
            raise ValueError("contextual target must be a generation-0 Episode")

        existing = connection.execute(
            """
            SELECT id, context_cue_id, need_cue_id FROM association
            WHERE from_type = ? AND from_id = ? AND to_type = 'episode'
              AND to_id = ? AND relation_type = 'retrieval'
              AND relation_key = 'contextual_recall' AND polarity = 1
              AND association_mode = 'contextual_recall'
            """,
            (
                candidate.anchor_type,
                int(candidate.anchor_id),
                int(candidate.target_episode_id),
            ),
        ).fetchone()
        if existing is not None:
            if (
                int(existing["context_cue_id"] or 0) != int(context_cue_id)
                or int(existing["need_cue_id"] or 0) != int(need_cue_id)
            ):
                # v13's association unique key does not include cue IDs.  Do
                # not silently bind a different cue pair to an existing edge.
                raise ValueError("contextual endpoint already has a different cue pair")
            return int(existing["id"]), True
        now = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO association(
                from_type, from_id, to_type, to_id, relation_type, relation_key,
                relation_text, polarity, weight, confidence, generation,
                claim_level, audit_status, evidence_json, audit_json,
                created_reason, association_mode, context_cue_id, need_cue_id,
                utility_weight, lifecycle_state, expires_at, source_request_hash,
                created_at, updated_at
            ) VALUES(?, ?, 'episode', ?, 'retrieval', 'contextual_recall',
                '', 1, ?, 0.0, 0, 'retrieval_only', 'not_required', '[]', '[]',
                ?, 'contextual_recall', ?, ?, ?, 'probation', ?, ?, ?, ?)
            """,
            (
                candidate.anchor_type,
                int(candidate.anchor_id),
                int(candidate.target_episode_id),
                utility_weight,
                str(candidate.reason or "recovered_missing_evidence"),
                int(context_cue_id),
                int(need_cue_id),
                utility_weight,
                expires_at,
                str(candidate.source_request_hash or ""),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid), False

    def create_contextual(
        self,
        candidate,
        *,
        context_cue_id: int,
        need_cue_id: int,
        utility_weight: float = 0.20,
        probation_ttl: int = 2_592_000,
    ) -> int:
        """Persist one retrieval-only edge after local candidate validation."""

        with self.db.transaction() as connection:
            association_id, _reused = self._create_contextual_in_transaction(
                connection,
                candidate,
                context_cue_id=context_cue_id,
                need_cue_id=need_cue_id,
                utility_weight=utility_weight,
                probation_ttl=probation_ttl,
            )
            return association_id

    create_contextual_association = create_contextual

    # ---- Contextual creation receipts (v14) ---------------------------
    # A receipt is deliberately one candidate/edge per unique
    # ``creation_request_id``.  Event-level callers pass every candidate's
    # opaque candidate_id as that key through ``finalize_recall_event``; the
    # whole event remains one SQLite transaction, without collapsing a
    # multi-candidate plan into one edge.

    @staticmethod
    def _opaque_text_tuple(value) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        return tuple(
            dict.fromkeys(
                str(item).strip() for item in value if str(item).strip()
            )
        )

    @staticmethod
    def _source_fact_payload(fact) -> dict[str, object]:
        revision = str(getattr(fact, "source_revision_id", "") or "").strip()
        raw_span_hash = str(getattr(fact, "raw_span_hash", "") or "").strip()
        span_hash = str(getattr(fact, "span_hash", "") or "").strip()
        raw_span = getattr(fact, "record_span", ())
        if isinstance(raw_span, str):
            raw_span = (raw_span,)
        record_span = tuple(
            str(value).strip() for value in (raw_span or ()) if str(value).strip()
        )
        if not revision or not raw_span_hash or (not record_span and not span_hash):
            raise ValueError("learning candidate contains an incomplete source fact")
        # source_key/raw text are intentionally absent: the durable receipt
        # contains only a revision + span identity suitable for an audit join.
        return {
            "source_revision_id": revision,
            "record_span": record_span,
            "span_hash": span_hash,
            "raw_span_hash": raw_span_hash,
        }

    @staticmethod
    def _row_text(row, field_name: str) -> str:
        """Read a possibly extension-provided SQLite column without guessing."""

        try:
            return str(row[field_name] or "").strip()
        except (KeyError, IndexError, TypeError):
            return ""

    @classmethod
    def _explicit_row_revision(cls, row, *, label: str) -> tuple[str, str, str | None]:
        """Mirror the source-version contract used to build SourceFactRef."""

        keys = set(row.keys()) if hasattr(row, "keys") else set()
        for field_name in (
            "source_version",
            "source_revision",
            "source_hash",
            "source_content_sha256",
            "content_hash",
            "revision",
            "version",
        ):
            if field_name not in keys:
                continue
            value = cls._row_text(row, field_name)
            if not value:
                return "", f"{label}.{field_name}", "source revision is blank"
            return value, f"{label}.{field_name}", None
        return "", "", None

    @staticmethod
    def _source_fact_locator(
        fact,
    ) -> tuple[int, str, tuple[tuple[int, int], ...]]:
        """Parse only the durable locator format emitted by the V3 engine."""

        source_ids: list[int] = []
        spans: set[tuple[int, int]] = set()
        coordinate_basis = ""
        for item in tuple(getattr(fact, "record_span", ()) or ()):
            value = str(item or "").strip()
            source_match = re.fullmatch(r"source-row:([1-9][0-9]*)", value)
            if source_match:
                source_ids.append(int(source_match.group(1)))
                continue
            span_match = re.fullmatch(r"lines:([1-9][0-9]*)-([1-9][0-9]*)", value)
            reasoning_view_match = re.fullmatch(
                r"reasoning-view-nonempty-lines:([1-9][0-9]*)-([1-9][0-9]*)",
                value,
            )
            if span_match or reasoning_view_match:
                matched = span_match or reasoning_view_match
                assert matched is not None
                current_basis = (
                    "raw_source_lines_v1"
                    if span_match
                    else "reasoning_view_nonempty_lines_v1"
                )
                if coordinate_basis and coordinate_basis != current_basis:
                    raise ValueError("source fact mixes coordinate bases")
                coordinate_basis = current_basis
                start, end = int(matched.group(1)), int(matched.group(2))
                if end < start:
                    raise ValueError("source fact line span is invalid")
                spans.add((start, end))
                continue
            raise ValueError("source fact locator is not a V3 source span")
        if len(set(source_ids)) != 1 or not spans or not coordinate_basis:
            raise ValueError("source fact needs one source row and one line span")
        return source_ids[0], coordinate_basis, tuple(sorted(spans))

    def _current_source_revision_for_episode(
        self,
        *,
        source_id: int,
        source,
        raw_source: str,
        episode,
    ) -> str | None:
        """Rebuild the engine's source-revision identity for one Episode."""

        source_version, source_basis, source_error = self._explicit_row_revision(
            source, label="source"
        )
        episode_version, episode_basis, episode_error = self._explicit_row_revision(
            episode, label="episode"
        )
        if source_error is not None or episode_error is not None:
            return None
        if (
            episode_version
            and source_version
            and episode_basis.rsplit(".", 1)[-1]
            == source_basis.rsplit(".", 1)[-1]
            and episode_version != source_version
        ):
            return None
        revision_marker = source_version or episode_version
        if not revision_marker:
            try:
                revision_marker = normalize_evaluation_as_of(
                    self._row_text(episode, "updated_at")
                )
            except ValueError:
                return None
        source_key = self._row_text(episode, "source_key")
        source_revision_material = json.dumps(
            {
                "source_id": int(source_id),
                "source_key_sha256": hashlib.sha256(
                    source_key.encode("utf-8")
                ).hexdigest(),
                "raw_source_sha256": hashlib.sha256(
                    raw_source.encode("utf-8")
                ).hexdigest(),
                "revision_marker": revision_marker,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "source-revision:sha256:" + hashlib.sha256(
            source_revision_material
        ).hexdigest()

    def _verify_current_source_fact_in_transaction(self, connection, fact) -> tuple[dict[str, object], int]:
        """Fail closed unless a SourceFactRef still names current source truth.

        A V3 fact is not just an opaque assertion.  Its locator must resolve
        against the current raw source, its exact spans must reconstruct both
        hashes, and its revision identity must still match an eligible
        generation-zero Episode.  Nothing from the raw source is returned or
        stored in the receipt.
        """

        payload = self._source_fact_payload(fact)
        if not str(payload["span_hash"] or "").strip():
            raise ValueError("v3 source fact needs a source-span hash")
        source_id, coordinate_basis, spans = self._source_fact_locator(fact)
        source = connection.execute(
            "SELECT * FROM source WHERE id = ?", (source_id,)
        ).fetchone()
        if source is None:
            raise ValueError("source fact source row no longer exists")
        try:
            # Do not trim this string: the raw-source and reasoning-view
            # locators below are both derived from the persisted string.
            raw_source = str(source["raw_text"] or "")
        except (KeyError, IndexError, TypeError):
            raw_source = ""
        if not raw_source.strip():
            raise ValueError("source fact source row is empty")
        if coordinate_basis == "reasoning_view_nonempty_lines_v1":
            lines = MemoryExtractor._single_pass_source_lines(
                MemoryExtractor.compact_source_for_reasoning(raw_source)
            )[0]
        else:
            lines = raw_source.splitlines()
        if not lines or any(end > len(lines) for _start, end in spans):
            raise ValueError("source fact line span is outside current source")
        expected_span_hash = "sha256:" + hashlib.sha256(
            json.dumps(spans, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_raw_span_hash = "sha256:" + hashlib.sha256(
            "\x1e".join(
                "\n".join(lines[start - 1 : end]) for start, end in spans
            ).encode("utf-8")
        ).hexdigest()
        if (
            str(payload["span_hash"]) != expected_span_hash
            or str(payload["raw_span_hash"]) != expected_raw_span_hash
        ):
            raise ValueError("source fact span no longer matches current source")

        episodes = connection.execute(
            "SELECT * FROM episode WHERE source_id = ? ORDER BY id", (source_id,)
        ).fetchall()
        supplied_source_key = str(getattr(fact, "source_key", "") or "").strip()
        for episode in episodes:
            if supplied_source_key and self._row_text(episode, "source_key") != supplied_source_key:
                continue
            try:
                generation = int(episode["generation"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if generation != 0:
                continue
            origin = self._row_text(episode, "evidence_origin").casefold()
            if origin not in {"source", "mixed"}:
                continue
            expected_revision = self._current_source_revision_for_episode(
                source_id=source_id,
                source=source,
                raw_source=raw_source,
                episode=episode,
            )
            if not expected_revision:
                continue
            if str(payload["source_revision_id"]) == expected_revision:
                return payload, source_id
        raise ValueError("source fact revision no longer matches current source")

    @staticmethod
    def _canonical_json(value) -> str:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _candidate_closure_in_transaction(self, connection, candidate) -> dict[str, object]:
        """Validate and serialize source/verification closure without text."""

        strict = hasattr(candidate, "source_facts")
        target_row = connection.execute(
            """
            SELECT e.id, e.source_id, s.id AS source_exists,
                   length(COALESCE(s.raw_text, '')) AS source_text_length
            FROM episode e
            LEFT JOIN source s ON s.id = e.source_id
            WHERE e.id = ?
            """,
            (int(candidate.target_episode_id),),
        ).fetchone()
        if target_row is None or target_row["source_exists"] is None:
            raise ValueError("learning candidate target has no durable Source")
        if strict:
            facts = tuple(getattr(candidate, "source_facts", ()) or ())
            if not facts:
                raise ValueError("v3 learning candidate needs source facts")
            verified_facts = [
                self._verify_current_source_fact_in_transaction(connection, fact)
                for fact in facts
            ]
            target_episode = connection.execute(
                "SELECT * FROM episode WHERE id = ?",
                (int(candidate.target_episode_id),),
            ).fetchone()
            target_source = connection.execute(
                "SELECT * FROM source WHERE id = ?",
                (int(target_row["source_id"]),),
            ).fetchone()
            if target_episode is None or target_source is None:
                raise ValueError("learning candidate target has no current Source")
            try:
                target_raw_source = str(target_source["raw_text"] or "")
            except (KeyError, IndexError, TypeError):
                target_raw_source = ""
            target_revision = self._current_source_revision_for_episode(
                source_id=int(target_row["source_id"]),
                source=target_source,
                raw_source=target_raw_source,
                episode=target_episode,
            )
            if not target_revision or not any(
                source_id == int(target_row["source_id"])
                and str(payload["source_revision_id"]) == target_revision
                for payload, source_id in verified_facts
            ):
                raise ValueError("learning candidate lacks current target source closure")
            fact_payloads = [payload for payload, _source_id in verified_facts]
            fact_payloads = sorted(
                {
                    self._canonical_json(item): item for item in fact_payloads
                }.values(),
                key=self._canonical_json,
            )
            verification_status = str(
                getattr(candidate, "verification_status", "") or ""
            ).strip()
            verification_refs = self._opaque_text_tuple(
                getattr(candidate, "verification_refs", ())
            )
            anchor_provenance = self._opaque_text_tuple(
                getattr(candidate, "anchor_provenance_refs", ())
            )
            target_provenance = self._opaque_text_tuple(
                getattr(candidate, "target_provenance_refs", ())
            )
            if verification_status not in {"verified", "source_bound"}:
                raise ValueError("v3 learning candidate must be source verified")
            if not verification_refs or not anchor_provenance or not target_provenance:
                raise ValueError("v3 learning candidate has incomplete provenance")
            if int(target_row["source_text_length"] or 0) <= 0:
                raise ValueError("v3 learning candidate target Source is empty")
            return {
                "strict": True,
                "source_facts": fact_payloads,
                "verification_refs": verification_refs,
                "verification_status": verification_status,
                "anchor_provenance": anchor_provenance,
                "target_provenance": target_provenance,
            }

        # ContextualRecallCandidate predates source-fact closure.  Keep its
        # callers operational, but represent the absence explicitly and never
        # mark this compatibility path ready for revisit.
        return {
            "strict": False,
            "source_facts": [
                {
                    "legacy_endpoint_source_id": int(target_row["source_id"]),
                    "target_episode_id": int(candidate.target_episode_id),
                }
            ],
            "verification_refs": ("legacy_endpoint_validation",),
            "verification_status": "legacy_compatibility",
            "anchor_provenance": ("legacy_contextual_candidate",),
            "target_provenance": ("legacy_contextual_candidate",),
        }

    @staticmethod
    def _candidate_audit_refs(candidate, *, strict: bool) -> dict[str, str]:
        """Keep opaque request/vector references with the immutable receipt."""

        fields = (
            "candidate_id",
            "request_id",
            "request_hash",
            "source_request_hash",
            "context_query_id",
            "need_query_id",
            "slot_id",
            "context_vector_ref",
            "need_vector_ref",
            "anchor_vector_ref",
            "anchor_contribution_id",
        )
        values = {
            field_name: str(getattr(candidate, field_name, "") or "").strip()
            for field_name in fields
        }
        if strict and any(not value for value in values.values()):
            raise ValueError("v3 learning candidate has incomplete audit references")
        return values

    @staticmethod
    def _candidate_endpoints(candidate) -> dict[str, object]:
        anchor_type = str(getattr(candidate, "anchor_type", "") or "").strip()
        if anchor_type not in {"episode", "concept"}:
            raise ValueError("learning candidate anchor type is invalid")
        anchor_id = int(getattr(candidate, "anchor_id", 0) or 0)
        target_id = int(getattr(candidate, "target_episode_id", 0) or 0)
        if anchor_id <= 0 or target_id <= 0:
            raise ValueError("learning candidate endpoint IDs must be positive")
        if anchor_type == "episode" and anchor_id == target_id:
            raise ValueError("learning candidate cannot create an episode self-edge")
        return {
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "target_episode_id": target_id,
            "reason": str(getattr(candidate, "reason", "") or "").strip(),
        }

    def _creation_identifiers(
        self,
        candidate,
        endpoints: Mapping[str, object],
        *,
        creation_request_id: str = "",
        creation_request_hash: str = "",
        context_text_hash: str,
        need_text_hash: str,
    ) -> tuple[str, str]:
        request_id = str(creation_request_id or "").strip()
        if not request_id:
            request_id = str(getattr(candidate, "candidate_id", "") or "").strip()
        if not request_id:
            request_id = str(getattr(candidate, "request_id", "") or "").strip()
        if not request_id:
            payload = self._canonical_json(
                {
                    "legacy": True,
                    **endpoints,
                    "context_text_hash": str(context_text_hash),
                    "need_text_hash": str(need_text_hash),
                    "source_request_hash": str(
                        getattr(candidate, "source_request_hash", "") or ""
                    ),
                }
            ).encode("utf-8")
            request_id = "legacy-creation:sha256:" + hashlib.sha256(payload).hexdigest()
        request_hash = str(creation_request_hash or "").strip()
        if not request_hash:
            request_hash = str(getattr(candidate, "request_hash", "") or "").strip()
        if not request_hash:
            request_hash = str(
                getattr(candidate, "source_request_hash", "") or ""
            ).strip()
        if not request_hash:
            request_hash = "creation-request:sha256:" + hashlib.sha256(
                request_id.encode("utf-8")
            ).hexdigest()
        return request_id, request_hash

    @staticmethod
    def _receipt_dict(row, *, idempotent: bool = False, edge_reused: bool = False) -> dict[str, object]:
        status = str(row["status"])
        return {
            "receipt_id": int(row["id"]),
            "status": status,
            "ready_for_revisit": status == "ready",
            "association_id": int(row["association_id"]),
            "context_cue_id": int(row["context_cue_id"]),
            "need_cue_id": int(row["need_cue_id"]),
            "domain": str(row["domain"]),
            "model_id": str(row["model_id"]),
            "embedding_space_id": str(row["embedding_space_id"]),
            "dimension": int(row["dimension"]),
            "dtype": str(row["dtype"]),
            "creation_request_id": str(row["creation_request_id"]),
            "creation_request_hash": str(row["creation_request_hash"]),
            "source_request_hash": str(row["source_request_hash"]),
            "context_vector_ref": str(row["context_vector_ref"]),
            "need_vector_ref": str(row["need_vector_ref"]),
            "anchor_vector_ref": str(row["anchor_vector_ref"]),
            "anchor_contribution_id": str(row["anchor_contribution_id"]),
            "index_epoch": int(row["ready_index_epoch"] or 0),
            "ready_index_epoch": int(row["ready_index_epoch"] or 0),
            "ready_at": str(row["ready_at"] or ""),
            "verification_status": str(row["verification_status"]),
            "durable_artifact_hash": str(row["durable_artifact_hash"]),
            "idempotent": bool(idempotent),
            "edge_reused": bool(edge_reused),
        }

    def _finalize_contextual_creation_in_transaction(
        self,
        connection,
        candidate,
        *,
        domain: str,
        model_id: str,
        dimension: int,
        context_vector,
        need_vector,
        context_text_hash: str,
        need_text_hash: str,
        embedding_space_id: str = "",
        embedding_space=None,
        context_display_text: str = "",
        need_display_text: str = "",
        creation_request_id: str = "",
        creation_request_hash: str = "",
        utility_weight: float = 0.20,
        probation_ttl: int = 2_592_000,
        revisit_runtime_seed: ContextualRevisitRuntimeSeed | None = None,
        restricted_rewrite_guard: ContextualRestrictedRewriteGuardDraft | None = None,
        restricted_rewrite_manifest_binding_signer: Callable[
            [
                ContextualRestrictedRewriteGuardDraft,
                ContextualRevisitRuntimeSeed,
                str,
            ],
            str,
        ]
        | None = None,
    ) -> dict[str, object]:
        """Atomically materialize two cues, one edge, and one receipt.

        ``revisit_runtime_seed`` and ``restricted_rewrite_guard`` are
        in-process Q1 capabilities emitted by QueryEngine and carried through
        MemoryApplication's typed materialization boundary.  Neither is an
        external request field or a repair mechanism for an old receipt.
        """

        normalized_domain = str(domain or "").strip()
        normalized_model = str(model_id or "").strip()
        normalized_dimension = int(dimension)
        if not normalized_domain or not normalized_model or normalized_dimension <= 0:
            raise ValueError("domain, model_id and positive dimension are required")
        if not str(context_text_hash or "").strip() or not str(need_text_hash or "").strip():
            raise ValueError("context and need text hashes are required")
        if restricted_rewrite_guard is not None:
            if not isinstance(
                restricted_rewrite_guard, ContextualRestrictedRewriteGuardDraft
            ):
                raise TypeError(
                    "restricted rewrite guard must use the typed T16 draft"
                )
            if not isinstance(revisit_runtime_seed, ContextualRevisitRuntimeSeed):
                raise ValueError(
                    "restricted rewrite guard requires a typed V17 runtime seed"
                )
            if not callable(restricted_rewrite_manifest_binding_signer):
                raise ValueError(
                    "restricted rewrite guard requires a process-only manifest signer"
                )
            if (
                str(restricted_rewrite_guard.creation_request_id)
                != str(revisit_runtime_seed.creation_request_id)
                or str(restricted_rewrite_guard.domain)
                != str(revisit_runtime_seed.domain)
                or str(restricted_rewrite_guard.context_scope_hash)
                != str(revisit_runtime_seed.context_scope_hash)
            ):
                raise ValueError(
                    "restricted rewrite guard does not bind the runtime seed"
                )
        endpoints = self._candidate_endpoints(candidate)
        closure = self._candidate_closure_in_transaction(connection, candidate)
        audit_refs = self._candidate_audit_refs(
            candidate, strict=bool(closure["strict"])
        )
        space_id = self._normalize_embedding_space_id(
            model_id=normalized_model,
            dimension=normalized_dimension,
            embedding_space_id=embedding_space_id,
            embedding_space=embedding_space,
        )
        request_id, request_hash = self._creation_identifiers(
            candidate,
            endpoints,
            creation_request_id=creation_request_id,
            creation_request_hash=creation_request_hash,
            context_text_hash=context_text_hash,
            need_text_hash=need_text_hash,
        )
        fingerprint_payload = {
            "endpoints": endpoints,
            "domain": normalized_domain,
            "model_id": normalized_model,
            "dimension": normalized_dimension,
            "dtype": "float32",
            "embedding_space_id": space_id,
            "context_text_hash": str(context_text_hash),
            "need_text_hash": str(need_text_hash),
            "closure": closure,
            "creation_request_id": request_id,
            "creation_request_hash": request_hash,
            "audit_refs": audit_refs,
        }
        fingerprint = hashlib.sha256(
            self._canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        existing_receipt = connection.execute(
            "SELECT * FROM contextual_creation_receipt WHERE creation_request_id = ?",
            (request_id,),
        ).fetchone()
        if existing_receipt is not None:
            if str(existing_receipt["candidate_fingerprint"]) != fingerprint:
                raise ValueError("creation request id was reused for different input")
            if revisit_runtime_seed is not None:
                self._assert_runtime_seed_matches_existing_receipt_in_transaction(
                    connection,
                    receipt=existing_receipt,
                    seed=revisit_runtime_seed,
                )
            if restricted_rewrite_guard is not None:
                self._assert_restricted_rewrite_guard_matches_existing_receipt_in_transaction(
                    connection,
                    receipt=existing_receipt,
                    seed=revisit_runtime_seed,
                    guard=restricted_rewrite_guard,
                    manifest_binding_signer=restricted_rewrite_manifest_binding_signer,
                )
            return self._receipt_dict(existing_receipt, idempotent=True, edge_reused=True)

        context_cue_id = self._get_or_create_cue_prototype_in_transaction(
            connection,
            domain=normalized_domain,
            cue_kind="context",
            model_id=normalized_model,
            dimension=normalized_dimension,
            vector=context_vector,
            text_hash=context_text_hash,
            embedding_space_id=space_id,
            display_text=context_display_text,
            source_request_hash=str(getattr(candidate, "source_request_hash", "") or request_hash),
        )
        need_cue_id = self._get_or_create_cue_prototype_in_transaction(
            connection,
            domain=normalized_domain,
            cue_kind="need",
            model_id=normalized_model,
            dimension=normalized_dimension,
            vector=need_vector,
            text_hash=need_text_hash,
            embedding_space_id=space_id,
            display_text=need_display_text,
            source_request_hash=str(getattr(candidate, "source_request_hash", "") or request_hash),
        )
        association_id, edge_reused = self._create_contextual_in_transaction(
            connection,
            candidate,
            context_cue_id=context_cue_id,
            need_cue_id=need_cue_id,
            utility_weight=utility_weight,
            probation_ttl=probation_ttl,
            expected_domain=normalized_domain,
            expected_model_id=normalized_model,
            expected_dimension=normalized_dimension,
            expected_embedding_space_id=space_id,
        )
        status = (
            "committed_pending_index"
            if bool(closure["strict"])
            else "legacy_pending_verification"
        )
        now = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO contextual_creation_receipt(
                creation_request_id, creation_request_hash, candidate_fingerprint,
                association_id, context_cue_id, need_cue_id, domain, model_id,
                embedding_space_id, dimension, dtype, source_facts_json,
                verification_refs_json, verification_status,
                anchor_provenance_json, target_provenance_json, status,
                source_request_hash, context_vector_ref, need_vector_ref,
                anchor_vector_ref, anchor_contribution_id, durable_artifact_hash,
                created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'float32', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                request_hash,
                fingerprint,
                association_id,
                context_cue_id,
                need_cue_id,
                normalized_domain,
                normalized_model,
                space_id,
                normalized_dimension,
                self._canonical_json(closure["source_facts"]),
                self._canonical_json(closure["verification_refs"]),
                str(closure["verification_status"]),
                self._canonical_json(closure["anchor_provenance"]),
                self._canonical_json(closure["target_provenance"]),
                status,
                audit_refs["source_request_hash"],
                audit_refs["context_vector_ref"],
                audit_refs["need_vector_ref"],
                audit_refs["anchor_vector_ref"],
                audit_refs["anchor_contribution_id"],
                "receipt:sha256:" + fingerprint,
                now,
            ),
        )
        row = connection.execute(
            "SELECT * FROM contextual_creation_receipt WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        if revisit_runtime_seed is not None:
            self._create_contextual_revisit_runtime_seed_in_transaction(
                connection,
                receipt=row,
                seed=revisit_runtime_seed,
            )
        if restricted_rewrite_guard is not None:
            self._create_contextual_restricted_rewrite_guard_in_transaction(
                connection,
                receipt=row,
                seed=revisit_runtime_seed,
                guard=restricted_rewrite_guard,
                manifest_binding_signer=restricted_rewrite_manifest_binding_signer,
            )
        return self._receipt_dict(row, edge_reused=edge_reused)

    def finalize_contextual_creation(self, candidate, **kwargs) -> dict[str, object]:
        """Public one-candidate atomic creation boundary.

        A V3 ``LearningCandidate`` is accepted only with its verified source
        closure.  A legacy ``ContextualRecallCandidate`` receives a durable
        compatibility receipt but remains ``legacy_pending_verification``.
        """

        with self.db.transaction() as connection:
            return self._finalize_contextual_creation_in_transaction(
                connection, candidate, **kwargs
            )

    def finalize_recall_event(
        self,
        event,
        plan,
        *,
        cue_materializations: Mapping[str, Mapping[str, object]],
        utility_weight: float = 0.20,
        probation_ttl: int = 2_592_000,
    ) -> list[dict[str, object]]:
        """Atomically persist every candidate in one bounded learning plan.

        Each plan candidate maps to one immutable receipt keyed by its
        candidate ID.  The loop is intentionally inside one transaction so a
        missing materialization or one invalid edge rolls back every cue/edge
        created for the event.  This is a trusted in-process application
        boundary, not a request-payload decoder: a V17
        ``revisit_runtime_seed`` may originate only from the current Q1
        engine projection and must never be accepted from an untrusted API
        client.
        """

        request_id = str(getattr(event, "request_id", "") or "").strip()
        request_hash = str(getattr(event, "request_hash", "") or "").strip()
        event_domain = str(getattr(event, "domain", "") or "").strip()
        candidates = tuple(getattr(plan, "candidates", ()) or ())
        if not request_id or not request_hash or not event_domain:
            raise ValueError("recall event request_id, request_hash and domain are required")
        if bool(getattr(event, "cancelled", False)):
            raise ValueError("cancelled recall event cannot create contextual edges")
        if not bool(getattr(event, "answer_guard_passed", False)):
            raise ValueError("recall event answer guard did not pass")
        if str(getattr(plan, "request_id", "") or "").strip() != request_id:
            raise ValueError("learning candidate plan request_id does not match event")

        target_ids = {
            int(value) for value in tuple(getattr(event, "target_episode_ids", ()) or ())
        }
        delivered_ids = {
            int(value)
            for value in tuple(getattr(event, "final_delivered_episode_ids", ()) or ())
        }
        contextual_ids = {
            int(value)
            for value in tuple(
                getattr(event, "contextual_expansion_episode_ids", ()) or ()
            )
        }
        expected_reason = getattr(event, "reason_for_target", None)
        prepared: list[tuple[object, Mapping[str, object]]] = []
        seen_candidate_ids: set[str] = set()
        for candidate in candidates:
            candidate_id = str(getattr(candidate, "candidate_id", "") or "").strip()
            if not candidate_id or candidate_id in seen_candidate_ids:
                raise ValueError("event learning candidate_id must be unique")
            seen_candidate_ids.add(candidate_id)
            if str(getattr(candidate, "request_id", "") or "").strip() != request_id:
                raise ValueError("candidate request_id does not match event")
            if str(getattr(candidate, "request_hash", "") or "").strip() != request_hash:
                raise ValueError("candidate request_hash does not match event")
            target_id = int(getattr(candidate, "target_episode_id", 0) or 0)
            if (
                target_id not in target_ids
                or target_id not in delivered_ids
                or target_id in contextual_ids
            ):
                raise ValueError("candidate target is not eligible in this recall event")
            if callable(expected_reason) and str(
                getattr(candidate, "reason", "") or ""
            ) != str(expected_reason(target_id) or ""):
                raise ValueError("candidate reason does not match recall event")
            materialization = cue_materializations.get(candidate_id)
            if not isinstance(materialization, Mapping):
                raise ValueError("candidate cue materialization is required")
            materialization_domain = str(
                materialization.get("domain", event_domain) or ""
            ).strip()
            if materialization_domain != event_domain:
                raise ValueError("candidate materialization domain does not match event")
            runtime_seed = materialization.get("revisit_runtime_seed")
            if runtime_seed is not None:
                if not isinstance(runtime_seed, ContextualRevisitRuntimeSeed):
                    raise TypeError(
                        "candidate revisit_runtime_seed must use the V17 typed record"
                    )
                if str(runtime_seed.creation_request_id) != candidate_id:
                    raise ValueError(
                        "candidate runtime seed creation request id does not match"
                    )
            restricted_rewrite_guard = materialization.get(
                "restricted_rewrite_guard"
            )
            if restricted_rewrite_guard is not None:
                if not isinstance(
                    restricted_rewrite_guard, ContextualRestrictedRewriteGuardDraft
                ):
                    raise TypeError(
                        "candidate restricted rewrite guard must use the T16 typed record"
                    )
                if not isinstance(runtime_seed, ContextualRevisitRuntimeSeed):
                    raise ValueError(
                        "candidate restricted rewrite guard requires a V17 runtime seed"
                    )
                if not callable(
                    materialization.get(
                        "restricted_rewrite_manifest_binding_signer"
                    )
                ):
                    raise ValueError(
                        "candidate restricted rewrite guard requires a process-only manifest signer"
                    )
                if (
                    str(restricted_rewrite_guard.creation_request_id) != candidate_id
                    or str(restricted_rewrite_guard.domain) != event_domain
                    or str(restricted_rewrite_guard.domain)
                    != str(runtime_seed.domain)
                    or str(restricted_rewrite_guard.context_scope_hash)
                    != str(runtime_seed.context_scope_hash)
                ):
                    raise ValueError(
                        "candidate restricted rewrite guard does not bind event/runtime seed"
                    )
            prepared.append((candidate, materialization))

        receipts: list[dict[str, object]] = []
        with self.db.transaction() as connection:
            for candidate, materialization in prepared:
                candidate_id = str(getattr(candidate, "candidate_id", "") or "").strip()
                receipts.append(
                    self._finalize_contextual_creation_in_transaction(
                        connection,
                        candidate,
                        domain=event_domain,
                        model_id=str(materialization.get("model_id", "")),
                        dimension=int(materialization.get("dimension", 0)),
                        context_vector=materialization.get("context_vector"),
                        need_vector=materialization.get("need_vector"),
                        context_text_hash=str(materialization.get("context_text_hash", "")),
                        need_text_hash=str(materialization.get("need_text_hash", "")),
                        embedding_space_id=str(materialization.get("embedding_space_id", "")),
                        embedding_space=materialization.get("embedding_space"),
                        context_display_text=str(materialization.get("context_display_text", "")),
                        need_display_text=str(materialization.get("need_display_text", "")),
                        creation_request_id=candidate_id,
                        creation_request_hash=request_hash,
                        utility_weight=utility_weight,
                        probation_ttl=probation_ttl,
                        revisit_runtime_seed=materialization.get("revisit_runtime_seed"),
                        restricted_rewrite_guard=materialization.get(
                            "restricted_rewrite_guard"
                        ),
                        restricted_rewrite_manifest_binding_signer=materialization.get(
                            "restricted_rewrite_manifest_binding_signer"
                        ),
                    )
                )
        return receipts

    def get_contextual_creation_receipt(self, receipt_id: int) -> dict[str, object] | None:
        with self.db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                (int(receipt_id),),
            ).fetchone()
        return self._receipt_dict(row) if row is not None else None

    def get_contextual_index_publication(self) -> dict[str, object]:
        """Return the durable local index publication checkpoint."""

        with self.db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM contextual_index_publication WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("contextual index publication state is missing")
        return {
            "index_epoch": int(row["index_epoch"]),
            "embedding_space_id": str(row["embedding_space_id"] or ""),
            "context_cue_count": int(row["context_cue_count"]),
            "need_cue_count": int(row["need_cue_count"]),
            "published_at": str(row["published_at"] or ""),
        }

    def mark_contextual_receipt_ready(
        self,
        receipt_id: int,
        *,
        context_cue_count: int,
        need_cue_count: int,
        expected_context_cue_id: int | None = None,
        expected_need_cue_id: int | None = None,
        restricted_rewrite_ready_manifest_signer: Callable[
            [ContextualRestrictedRewriteGuard, ContextualRevisitRuntimeManifest],
            str,
        ]
        | None = None,
    ) -> dict[str, object]:
        """Publish one committed V3 receipt only after its RAM cues exist."""

        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                (int(receipt_id),),
            ).fetchone()
            if row is None:
                raise ValueError("contextual creation receipt does not exist")
            if (
                expected_context_cue_id is not None
                and int(row["context_cue_id"]) != int(expected_context_cue_id)
            ):
                raise ValueError("receipt context cue guard failed")
            if (
                expected_need_cue_id is not None
                and int(row["need_cue_id"]) != int(expected_need_cue_id)
            ):
                raise ValueError("receipt need cue guard failed")
            if str(row["status"]) == "ready":
                # A raw database writer may try to claim the receipt is ready
                # before the one-shot V22 guard transition.  Do not present a
                # ready+blank sidecar as a harmless idempotent publication.
                self._ready_v22_restricted_rewrite_guard_in_transaction(
                    connection, int(receipt_id)
                )
                return self._receipt_dict(row, idempotent=True)
            if str(row["status"]) != "committed_pending_index":
                return self._receipt_dict(row, idempotent=True)
            pending_rewrite_guard = (
                self._pending_v22_restricted_rewrite_guard_in_transaction(
                    connection, int(receipt_id)
                )
            )
            if (
                pending_rewrite_guard is not None
                and not callable(restricted_rewrite_ready_manifest_signer)
            ):
                # Do this before the receipt/publication state changes.  A
                # later retry must not be able to sign an already-ready blank
                # sidecar whose rows could have been database-tampered.
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite publication requires a process-only ready signer"
                )
            state = connection.execute(
                "SELECT * FROM contextual_index_publication WHERE singleton = 1"
            ).fetchone()
            if state is None:
                raise RuntimeError("contextual index publication state is missing")
            receipt_space = str(row["embedding_space_id"] or "").strip()
            state_space = str(state["embedding_space_id"] or "").strip()
            if state_space and state_space != receipt_space:
                raise ValueError("contextual index is published for another embedding space")
            epoch = int(state["index_epoch"]) + 1
            now = utc_now()
            connection.execute(
                """
                UPDATE contextual_index_publication
                SET index_epoch = ?, embedding_space_id = ?,
                    context_cue_count = ?, need_cue_count = ?, published_at = ?
                WHERE singleton = 1
                """,
                (
                    epoch,
                    receipt_space,
                    max(0, int(context_cue_count)),
                    max(0, int(need_cue_count)),
                    now,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE contextual_creation_receipt
                SET status = 'ready', ready_index_epoch = ?, ready_at = ?
                WHERE id = ? AND status = 'committed_pending_index'
                """,
                (epoch, now, int(receipt_id)),
            )
            if int(cursor.rowcount) != 1:
                raise RuntimeError("contextual receipt readiness transition failed")
            ready = connection.execute(
                "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                (int(receipt_id),),
            ).fetchone()
            assert ready is not None
            # A V17 manifest is an optional automatic-revisit optimization,
            # never a prerequisite for publishing an otherwise valid cue/edge
            # receipt.  Source revision or cue integrity may legitimately
            # drift between Q1 and this local RAM transition.  Isolate its
            # contract/manifest promotion so such a stale seed remains
            # pending and fails closed without rolling the normal receipt
            # back to pending (or leaving a half-created V16 contract).
            savepoint = "revisit_manifest_mark_ready"
            connection.execute(f"SAVEPOINT {savepoint}")
            try:
                promotion = self._promote_contextual_revisit_runtime_manifest_in_transaction(
                    connection,
                    int(receipt_id),
                    restricted_rewrite_ready_manifest_signer=(
                        restricted_rewrite_ready_manifest_signer
                    ),
                    _allow_restricted_rewrite_ready_transition=True,
                )
                if pending_rewrite_guard is not None:
                    completed_guard = (
                        self._ready_v22_restricted_rewrite_guard_in_transaction(
                            connection, int(receipt_id)
                        )
                    )
                    if (
                        promotion is None
                        or promotion.state != "ready"
                        or completed_guard is None
                    ):
                        raise RestrictedRewriteReadyBindingError(
                            "guarded receipt did not atomically bind its ready manifest"
                        )
            except RestrictedRewriteReadyBindingError:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                # The outer transaction must also roll receipt/publication
                # changes back; do not turn a missing/bad signer into a
                # permanently ready-but-blank V22 sidecar.
                raise
            except (ValueError, sqlite3.IntegrityError) as error:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                if pending_rewrite_guard is not None:
                    # A plain V17 manifest is an optional optimization, but a
                    # V22 guard is part of the same publication invariant.  A
                    # stale source/cue, expiry, contract error, or trigger
                    # failure must roll the receipt/publication checkpoint
                    # back rather than commit a ready+blank guard.
                    raise RestrictedRewriteReadyBindingError(
                        "guarded receipt could not atomically bind its ready manifest"
                    ) from error
            else:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            return self._receipt_dict(ready)

    def reconcile_contextual_receipts_after_index_rebuild(
        self,
        *,
        context_cue_ids,
        need_cue_ids,
        context_cue_count: int,
        need_cue_count: int,
        embedding_space_id: str = "",
        restricted_rewrite_ready_manifest_signer: Callable[
            [ContextualRestrictedRewriteGuard, ContextualRevisitRuntimeManifest],
            str,
        ]
        | None = None,
    ) -> dict[str, int]:
        """Mark only fully loaded pending receipts ready after an offline rebuild."""

        contexts = {int(value) for value in context_cue_ids}
        needs = {int(value) for value in need_cue_ids}
        with self.db.transaction() as connection:
            state = connection.execute(
                "SELECT * FROM contextual_index_publication WHERE singleton = 1"
            ).fetchone()
            if state is None:
                raise RuntimeError("contextual index publication state is missing")
            state_space = str(state["embedding_space_id"] or "").strip()
            requested_space = str(embedding_space_id or "").strip()
            if state_space and requested_space and state_space != requested_space:
                raise ValueError("contextual index rebuild uses another embedding space")
            selected_space = state_space or requested_space
            epoch = int(state["index_epoch"]) + 1
            now = utc_now()
            connection.execute(
                """
                UPDATE contextual_index_publication
                SET index_epoch = ?, embedding_space_id = ?,
                    context_cue_count = ?, need_cue_count = ?, published_at = ?
                WHERE singleton = 1
                """,
                (
                    epoch,
                    selected_space,
                    max(0, int(context_cue_count)),
                    max(0, int(need_cue_count)),
                    now,
                ),
            )
            if selected_space:
                pending = connection.execute(
                    """
                    SELECT id, context_cue_id, need_cue_id
                    FROM contextual_creation_receipt
                    WHERE status = 'committed_pending_index'
                      AND embedding_space_id = ?
                    ORDER BY id
                    """,
                    (selected_space,),
                ).fetchall()
            else:
                pending = []
            ready_ids = [
                int(row["id"])
                for row in pending
                if int(row["context_cue_id"]) in contexts
                and int(row["need_cue_id"]) in needs
            ]
            runtime_manifest_promoted = 0
            runtime_manifest_rejected = 0
            unguarded_ready_ids: list[int] = []
            guarded_ready_ids: list[int] = []
            for ready_id in ready_ids:
                try:
                    pending_guard = (
                        self._pending_v22_restricted_rewrite_guard_in_transaction(
                            connection, ready_id
                        )
                    )
                except RestrictedRewriteReadyBindingError:
                    runtime_manifest_rejected += 1
                    continue
                if pending_guard is None:
                    unguarded_ready_ids.append(ready_id)
                elif callable(restricted_rewrite_ready_manifest_signer):
                    guarded_ready_ids.append(ready_id)
                else:
                    # Leave a guarded receipt fully pending.  It may be
                    # retried only by a future trusted rebuild that supplies
                    # the process-only signer, never after raw rows are
                    # already marked ready.
                    runtime_manifest_rejected += 1
            reconciled = 0
            if unguarded_ready_ids:
                placeholders = ",".join("?" for _ in unguarded_ready_ids)
                cursor = connection.execute(
                    f"""
                    UPDATE contextual_creation_receipt
                    SET status = 'ready', ready_index_epoch = ?, ready_at = ?
                    WHERE status = 'committed_pending_index'
                      AND id IN ({placeholders})
                    """,
                    (epoch, now, *unguarded_ready_ids),
                )
                reconciled += int(cursor.rowcount)
                # Rebuild is the other normal publication route.  A malformed
                # or stale optional V17 seed must not roll an otherwise-good
                # batch of cue receipts back.  Savepoint each promotion so no
                # failed manifest can leave a partial V16 contract behind or
                # block its healthy neighbours.
                for index, ready_id in enumerate(unguarded_ready_ids):
                    savepoint = f"revisit_manifest_rebuild_{index}"
                    connection.execute(f"SAVEPOINT {savepoint}")
                    try:
                        promotion = (
                            self._promote_contextual_revisit_runtime_manifest_in_transaction(
                                connection, ready_id
                            )
                        )
                    except (ValueError, sqlite3.IntegrityError):
                        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                        runtime_manifest_rejected += 1
                        continue
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    if promotion is not None and promotion.state == "ready":
                        runtime_manifest_promoted += int(promotion.promoted)
            # V22 guarded rows transition one at a time so signer failure can
            # roll the receipt back to pending in the same savepoint.  A bulk
            # status update before signing would create an unsafe ready+blank
            # state that a later trusted process must never repair.
            for index, ready_id in enumerate(guarded_ready_ids):
                savepoint = f"revisit_manifest_guarded_rebuild_{index}"
                connection.execute(f"SAVEPOINT {savepoint}")
                try:
                    cursor = connection.execute(
                        """
                        UPDATE contextual_creation_receipt
                        SET status = 'ready', ready_index_epoch = ?, ready_at = ?
                        WHERE id = ? AND status = 'committed_pending_index'
                        """,
                        (epoch, now, ready_id),
                    )
                    if int(cursor.rowcount) != 1:
                        raise RestrictedRewriteReadyBindingError(
                            "guarded receipt readiness transition lost its pending row"
                        )
                    promotion = (
                        self._promote_contextual_revisit_runtime_manifest_in_transaction(
                            connection,
                            ready_id,
                            restricted_rewrite_ready_manifest_signer=(
                                restricted_rewrite_ready_manifest_signer
                            ),
                            _allow_restricted_rewrite_ready_transition=True,
                        )
                    )
                    if promotion is None or promotion.state != "ready":
                        raise RestrictedRewriteReadyBindingError(
                            "guarded receipt did not produce a ready runtime manifest"
                        )
                    completed_guard = (
                        self._ready_v22_restricted_rewrite_guard_in_transaction(
                            connection, ready_id
                        )
                    )
                    if completed_guard is None:
                        raise RestrictedRewriteReadyBindingError(
                            "guarded receipt did not atomically bind its ready manifest"
                        )
                except (
                    RestrictedRewriteReadyBindingError,
                    ValueError,
                    sqlite3.IntegrityError,
                ):
                    connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    runtime_manifest_rejected += 1
                    continue
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                reconciled += 1
                runtime_manifest_promoted += int(promotion.promoted)
            return {
                "index_epoch": epoch,
                "reconciled": reconciled,
                "pending": len(pending) - reconciled,
                "runtime_manifest_promoted": runtime_manifest_promoted,
                "runtime_manifest_rejected": runtime_manifest_rejected,
            }

    def list_contextual(self, state: str | None = None) -> list:
        clauses = ["association_mode = 'contextual_recall'"]
        params: list[object] = []
        if state:
            clauses.append("lifecycle_state = ?")
            params.append(str(state))
        with self.db.connection() as connection:
            return connection.execute(
                "SELECT * FROM association WHERE " + " AND ".join(clauses) + " ORDER BY id",
                params,
            ).fetchall()

    def get_contextual_for_prototypes(
        self,
        context_ids,
        need_ids,
        *,
        domain: str | None = None,
        limit: int | None = 1000,
        anchor_episode_ids=None,
        evaluation_as_of: str | None = None,
    ) -> list:
        """Read contextual edges, with an explicit anchor-first v3 mode.

        The historical positional API remains for old callers that already
        have global context/need prototype IDs.  V3 callers must pass both
        ``anchor_episode_ids`` and ``evaluation_as_of``.  That path filters
        active Episode anchors in SQL *before* any ordering or optional limit,
        and never reaches ``utc_now()``.
        """

        is_v3 = anchor_episode_ids is not None or evaluation_as_of is not None
        if is_v3 and (anchor_episode_ids is None or evaluation_as_of is None):
            raise ValueError(
                "anchor_episode_ids and evaluation_as_of are both required for v3 lookup"
            )

        contexts = (
            None
            if context_ids is None
            else sorted({int(value) for value in context_ids})
        )
        needs = (
            None if need_ids is None else sorted({int(value) for value in need_ids})
        )
        if contexts == [] or needs == []:
            return []

        clauses = [
            "a.association_mode = 'contextual_recall'",
            "cp.cue_kind = 'context'",
            "np.cue_kind = 'need'",
            "a.lifecycle_state IN ('probation', 'active')",
            "cp.domain = np.domain",
            # New receipt-backed edges become query-visible only after the
            # exact committed receipt records RAM-index publication.  Old
            # v13 edges have no receipt and remain readable for compatibility.
            "(NOT EXISTS (SELECT 1 FROM contextual_creation_receipt r "
            "WHERE r.association_id = a.id) OR EXISTS (SELECT 1 FROM "
            "contextual_creation_receipt r WHERE r.association_id = a.id "
            "AND r.status = 'ready'))",
        ]
        params: list[object] = []
        if contexts is not None:
            context_ph = ",".join("?" for _ in contexts)
            clauses.append(f"a.context_cue_id IN ({context_ph})")
            params.extend(contexts)
        if needs is not None:
            need_ph = ",".join("?" for _ in needs)
            clauses.append(f"a.need_cue_id IN ({need_ph})")
            params.extend(needs)
        if domain:
            clauses.append("cp.domain = ?")
            params.append(str(domain))

        if is_v3:
            # Validate the frozen/request timestamp even if the caller has no
            # anchors. A malformed replay contract is not a valid no-hit.
            as_of = normalize_evaluation_as_of(evaluation_as_of)
            anchors_set: set[int] = set()
            for value in anchor_episode_ids:
                try:
                    anchor_id = int(value)
                except (TypeError, ValueError):
                    continue
                if anchor_id > 0:
                    anchors_set.add(anchor_id)
            anchors = sorted(anchors_set)
            if not anchors:
                return []
            anchor_ph = ",".join("?" for _ in anchors)
            # These clauses intentionally appear before the final ORDER/LIMIT.
            # They stop an unrelated high-utility edge from consuming a global
            # result budget before the matcher can score its actual anchor.
            clauses.extend(
                [
                    "a.from_type = 'episode'",
                    f"a.from_id IN ({anchor_ph})",
                ]
            )
            params.extend(anchors)
        else:
            # Legacy/non-v3 compatibility only. New matcher calls always use
            # the explicit branch above, so replay never gets a hidden clock.
            as_of = utc_now()
        clauses.append("(a.expires_at IS NULL OR a.expires_at > ?)")
        params.append(as_of)

        limit_clause = ""
        if limit is not None:
            normalized_limit = max(1, int(limit))
            limit_clause = " LIMIT ?"
            params.append(normalized_limit)
        order_by = "a.id" if is_v3 else "a.utility_weight DESC, a.id"
        with self.db.connection() as connection:
            return connection.execute(
                f"""
                SELECT a.*, cp.domain AS cue_domain
                FROM association a
                JOIN association_cue_prototype cp ON cp.id = a.context_cue_id
                JOIN association_cue_prototype np ON np.id = a.need_cue_id
                WHERE {' AND '.join(clauses)}
                ORDER BY {order_by}{limit_clause}
                """,
                params,
            ).fetchall()

    @staticmethod
    def _utility_ledger_payload(
        observation: ContextualUtilityLedgerObservation,
    ) -> dict[str, object]:
        """Return one canonical primitive payload for equality and insertion."""

        return {
            "observation_id": observation.observation_id,
            "family_id": observation.family_id,
            "association_id": int(observation.association_id),
            "creation_receipt_id": int(observation.creation_receipt_id),
            "evaluation_as_of": normalize_evaluation_as_of(observation.evaluation_as_of),
            "candidate_universe_fingerprint": observation.candidate_universe_fingerprint,
            "requirements_fingerprint": observation.requirements_fingerprint,
            "budget_fingerprint": observation.budget_fingerprint,
            "input_fingerprint": observation.input_fingerprint,
            "treatment_fingerprint": observation.treatment_fingerprint,
            "masked_fingerprint": observation.masked_fingerprint,
            "single_edge_fingerprint": observation.single_edge_fingerprint,
            "leave_one_out_fingerprint": observation.leave_one_out_fingerprint,
            "factual_support_verified": int(observation.factual_support_verified),
            "treatment_episode_count": int(observation.treatment_episode_count),
            "masked_episode_count": int(observation.masked_episode_count),
            "treatment_required_count": int(observation.treatment_required_count),
            "masked_required_count": int(observation.masked_required_count),
            "treatment_gain_count": int(observation.treatment_gain_count),
            "treatment_loss_count": int(observation.treatment_loss_count),
            "single_edge_gain_count": int(observation.single_edge_gain_count),
            "single_edge_loss_count": int(observation.single_edge_loss_count),
            "leave_one_out_gain_count": int(observation.leave_one_out_gain_count),
            "leave_one_out_loss_count": int(observation.leave_one_out_loss_count),
            "recall_gain": int(observation.recall_gain),
            "work_metric": str(observation.work_metric),
            "treatment_work": observation.treatment_work,
            "masked_work": observation.masked_work,
            "work_saved": observation.work_saved,
            "provider_receipt_refs_json": json.dumps(
                list(observation.provider_receipt_refs),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "harm": int(observation.harm),
            "is_shadow": int(observation.is_shadow),
            "outcome": str(observation.outcome),
        }

    @staticmethod
    def _utility_ledger_row_matches(row, payload: Mapping[str, object]) -> bool:
        """Compare every immutable logical column, not a subset of a retry."""

        for field_name, expected in payload.items():
            actual = row[field_name]
            if actual is None or expected is None:
                if actual is not expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _utility_ledger_receipt(row, *, idempotent: bool) -> ContextualUtilityLedgerReceipt:
        return ContextualUtilityLedgerReceipt(
            ledger_id=int(row["id"]),
            observation_id=str(row["observation_id"]),
            family_id=str(row["family_id"]),
            association_id=int(row["association_id"]),
            creation_receipt_id=int(row["creation_receipt_id"]),
            idempotent=bool(idempotent),
            # This is only a read-only eligibility fact.  v15 never changes an
            # Association lifecycle from a utility observation.
            promotion_eligible=(
                not bool(row["is_shadow"])
                and not bool(row["harm"])
                and bool(row["factual_support_verified"])
                and int(row["recall_gain"]) > 0
            ),
        )

    @staticmethod
    def _canonical_utility_receipt_in_transaction(
        connection,
        observation: ContextualUtilityLedgerObservation,
    ):
        """Require the stable earliest ready source-bound origin for an edge."""

        receipt = connection.execute(
            """
            SELECT r.*
            FROM contextual_creation_receipt AS r
            JOIN association AS a ON a.id = r.association_id
            JOIN association_cue_prototype AS context_cue
                ON context_cue.id = r.context_cue_id
            JOIN association_cue_prototype AS need_cue
                ON need_cue.id = r.need_cue_id
            WHERE r.id = COALESCE(
                (
                    SELECT MIN(existing_contract.creation_receipt_id)
                    FROM contextual_revisit_contract AS existing_contract
                    WHERE existing_contract.association_id = ?
                ),
                (
                    SELECT MIN(existing_ledger.creation_receipt_id)
                    FROM contextual_utility_ledger AS existing_ledger
                    WHERE existing_ledger.association_id = ?
                ),
                (
                    SELECT MIN(candidate_receipt.id)
                    FROM contextual_creation_receipt AS candidate_receipt
                    WHERE candidate_receipt.association_id = ?
                      AND candidate_receipt.status = 'ready'
                      AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                )
            )
              AND a.association_mode = 'contextual_recall'
              AND a.context_cue_id = r.context_cue_id
              AND a.need_cue_id = r.need_cue_id
              AND context_cue.cue_kind = 'context'
              AND need_cue.cue_kind = 'need'
              AND context_cue.domain = r.domain
              AND need_cue.domain = r.domain
              AND context_cue.model_id = r.model_id
              AND need_cue.model_id = r.model_id
              AND context_cue.embedding_space_id = r.embedding_space_id
              AND need_cue.embedding_space_id = r.embedding_space_id
              AND context_cue.dimension = r.dimension
              AND need_cue.dimension = r.dimension
              AND context_cue.dtype = r.dtype
              AND need_cue.dtype = r.dtype
              AND r.status = 'ready'
              AND r.verification_status IN ('verified', 'source_bound')
            """,
            (
                int(observation.association_id),
                int(observation.association_id),
                int(observation.association_id),
            ),
        ).fetchone()
        if receipt is None:
            raise ValueError(
                "utility ledger needs a canonical ready source-bound creation receipt"
            )
        if int(receipt["id"]) != int(observation.creation_receipt_id):
            raise ValueError("utility ledger creation receipt is not canonical for edge")
        return receipt

    def _record_utility_ledger_in_transaction(
        self,
        connection,
        observation: ContextualUtilityLedgerObservation,
    ) -> ContextualUtilityLedgerReceipt:
        payload = self._utility_ledger_payload(observation)
        existing = connection.execute(
            "SELECT * FROM contextual_utility_ledger WHERE observation_id = ?",
            (str(payload["observation_id"]),),
        ).fetchone()
        if existing is not None:
            if not self._utility_ledger_row_matches(existing, payload):
                raise ValueError("utility observation id was reused for different input")
            return self._utility_ledger_receipt(existing, idempotent=True)
        # Exact retries must remain valid even if an older pending receipt
        # became ready later.  New observations still pass the stable
        # canonical-origin check below.
        self._canonical_utility_receipt_in_transaction(connection, observation)
        family_existing = connection.execute(
            """
            SELECT * FROM contextual_utility_ledger
            WHERE family_id = ? AND association_id = ?
            """,
            (str(payload["family_id"]), int(payload["association_id"])),
        ).fetchone()
        if family_existing is not None:
            # A fresh random observation ID is not permission to count the
            # same counterfactual family again.  The caller must create a new
            # independent family with its own immutable selector snapshot.
            raise ValueError("utility observation family is already recorded for edge")

        fields = tuple(payload)
        placeholders = ", ".join("?" for _ in fields)
        try:
            cursor = connection.execute(
                f"""
                INSERT INTO contextual_utility_ledger(
                    {', '.join(fields)}, created_at
                ) VALUES({placeholders}, ?)
                """,
                [*(payload[field_name] for field_name in fields), utc_now()],
            )
        except sqlite3.IntegrityError as error:
            # A second Database instance may have inserted while this writer
            # was waiting.  Resolve an exact retry; every other uniqueness
            # conflict stays fail-closed.
            existing = connection.execute(
                "SELECT * FROM contextual_utility_ledger WHERE observation_id = ?",
                (str(payload["observation_id"]),),
            ).fetchone()
            if existing is not None and self._utility_ledger_row_matches(existing, payload):
                return self._utility_ledger_receipt(existing, idempotent=True)
            raise ValueError("utility ledger uniqueness conflict") from error
        row = connection.execute(
            "SELECT * FROM contextual_utility_ledger WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return self._utility_ledger_receipt(row, idempotent=False)

    def record_contextual_utility_ledger(
        self,
        observations,
    ) -> tuple[ContextualUtilityLedgerReceipt, ...]:
        """Append only fully typed V15 utility observations.

        The method intentionally has no weight, counter, lifecycle, source
        request hash, or promotion side effect.  SQLite triggers repeat the
        source-bound and append-only checks so direct SQL cannot bypass this
        repository boundary.
        """

        values = tuple(observations or ())
        if any(
            not isinstance(value, ContextualUtilityLedgerObservation)
            for value in values
        ):
            raise TypeError("utility ledger observations must use the V15 typed record")
        if not values:
            return ()
        with self.db.transaction() as connection:
            return tuple(
                self._record_utility_ledger_in_transaction(connection, value)
                for value in values
            )

    def list_promotable_contextual_utility(self) -> list[dict[str, object]]:
        """Return evidence rows a future explicit promoter may inspect.

        Shadow observations are durable diagnostics only; they are never
        returned here.  This repository deliberately does not perform a
        promotion write in v15.
        """

        with self.db.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM contextual_utility_ledger
                WHERE is_shadow = 0 AND harm = 0
                  AND factual_support_verified = 1 AND recall_gain > 0
                ORDER BY id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def contextual_revisit_ready_publication_fingerprint(receipt: Mapping[str, object]) -> str:
        """Derive the immutable ready-publication fingerprint for a receipt.

        The material deliberately excludes query/source/answer prose and all
        vector bytes.  It covers the receipt identity, bound cue/edge IDs,
        embedding compatibility metadata, and the immutable ready transition.
        """

        def value(*names: str) -> object:
            for name in names:
                try:
                    return receipt[name]
                except KeyError:
                    continue
            raise KeyError(names[0])

        try:
            ready_epoch = int(value("ready_index_epoch", "index_epoch"))
            receipt_id = int(value("id", "receipt_id"))
            association_id = int(value("association_id"))
            context_cue_id = int(value("context_cue_id"))
            need_cue_id = int(value("need_cue_id"))
            dimension = int(value("dimension"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("revisit contract receipt has invalid ready metadata") from error
        if min(
            ready_epoch,
            receipt_id,
            association_id,
            context_cue_id,
            need_cue_id,
            dimension,
        ) <= 0:
            raise ValueError("revisit contract receipt has invalid ready metadata")
        payload = {
            "association_id": association_id,
            "context_cue_id": context_cue_id,
            "creation_receipt_id": receipt_id,
            "dimension": dimension,
            "domain": str(value("domain") or ""),
            "dtype": str(value("dtype") or ""),
            "durable_artifact_hash": str(value("durable_artifact_hash") or ""),
            "embedding_space_id": str(value("embedding_space_id") or ""),
            "model_id": str(value("model_id") or ""),
            "need_cue_id": need_cue_id,
            "ready_at": str(value("ready_at") or ""),
            "ready_index_epoch": ready_epoch,
            "status": str(value("status") or ""),
            "verification_status": str(value("verification_status") or ""),
            "version": "contextual-ready-publication-v1",
        }
        material = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "revisit-publication:sha256:" + hashlib.sha256(material).hexdigest()

    @staticmethod
    def _revisit_contract_payload(
        contract: ContextualRevisitContract,
    ) -> dict[str, object]:
        return contract.storage_payload()

    @staticmethod
    def _revisit_contract_row_matches(row, payload: Mapping[str, object]) -> bool:
        """Require an idempotent retry to match every immutable input."""

        for field_name, expected in payload.items():
            actual = row[field_name]
            if actual is None or expected is None:
                if actual is not expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _revisit_contract_receipt(
        row,
        *,
        idempotent: bool,
    ) -> ContextualRevisitContractReceipt:
        return ContextualRevisitContractReceipt(
            contract_id=int(row["id"]),
            creation_receipt_id=int(row["creation_receipt_id"]),
            association_id=int(row["association_id"]),
            contract_fingerprint=str(row["contract_fingerprint"]),
            idempotent=bool(idempotent),
        )

    def _canonical_revisit_receipt_in_transaction(
        self,
        connection,
        contract: ContextualRevisitContract,
    ):
        """Verify the canonical ready source-bound origin and publication."""

        receipt = connection.execute(
            """
            SELECT r.*
            FROM contextual_creation_receipt AS r
            JOIN association AS a ON a.id = r.association_id
            JOIN association_cue_prototype AS context_cue
                ON context_cue.id = r.context_cue_id
            JOIN association_cue_prototype AS need_cue
                ON need_cue.id = r.need_cue_id
            WHERE r.id = COALESCE(
                (
                    SELECT MIN(existing_contract.creation_receipt_id)
                    FROM contextual_revisit_contract AS existing_contract
                    WHERE existing_contract.association_id = ?
                ),
                (
                    SELECT MIN(existing_ledger.creation_receipt_id)
                    FROM contextual_utility_ledger AS existing_ledger
                    WHERE existing_ledger.association_id = ?
                ),
                (
                    SELECT MIN(candidate_receipt.id)
                    FROM contextual_creation_receipt AS candidate_receipt
                    WHERE candidate_receipt.association_id = ?
                      AND candidate_receipt.status = 'ready'
                      AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                )
            )
              AND a.association_mode = 'contextual_recall'
              AND a.context_cue_id = r.context_cue_id
              AND a.need_cue_id = r.need_cue_id
              AND context_cue.cue_kind = 'context'
              AND need_cue.cue_kind = 'need'
              AND context_cue.domain = r.domain
              AND need_cue.domain = r.domain
              AND context_cue.model_id = r.model_id
              AND need_cue.model_id = r.model_id
              AND context_cue.embedding_space_id = r.embedding_space_id
              AND need_cue.embedding_space_id = r.embedding_space_id
              AND context_cue.dimension = r.dimension
              AND need_cue.dimension = r.dimension
              AND context_cue.dtype = r.dtype
              AND need_cue.dtype = r.dtype
              AND r.status = 'ready'
              AND r.verification_status IN ('verified', 'source_bound')
            """,
            (
                int(contract.association_id),
                int(contract.association_id),
                int(contract.association_id),
            ),
        ).fetchone()
        if receipt is None:
            raise ValueError(
                "revisit contract needs a canonical ready source-bound creation receipt"
            )
        if int(receipt["id"]) != int(contract.creation_receipt_id):
            raise ValueError("revisit contract creation receipt is not canonical for edge")
        expected_fields = {
            "id": int(contract.creation_receipt_id),
            "association_id": int(contract.association_id),
            "context_cue_id": int(contract.context_cue_id),
            "need_cue_id": int(contract.need_cue_id),
            "domain": str(contract.domain),
            "model_id": str(contract.model_id),
            "embedding_space_id": str(contract.embedding_space_id),
            "dimension": int(contract.dimension),
            "dtype": str(contract.dtype),
            "ready_index_epoch": int(contract.ready_index_epoch),
        }
        if any(receipt[field_name] != expected for field_name, expected in expected_fields.items()):
            raise ValueError("revisit contract does not match its creation receipt")
        publication = connection.execute(
            """
            SELECT index_epoch, embedding_space_id
            FROM contextual_index_publication
            WHERE singleton = 1
            """
        ).fetchone()
        if (
            publication is None
            or str(publication["embedding_space_id"] or "")
            != str(receipt["embedding_space_id"])
            or int(publication["index_epoch"])
            < int(receipt["ready_index_epoch"])
        ):
            raise ValueError("revisit contract receipt is not in a ready publication")
        expected_publication = self.contextual_revisit_ready_publication_fingerprint(
            dict(receipt)
        )
        if contract.ready_publication_fingerprint != expected_publication:
            raise ValueError("revisit contract ready publication fingerprint mismatch")
        return receipt

    @staticmethod
    def _revisit_contract_from_row(row) -> ContextualRevisitContract:
        """Rebuild a typed contract only if durable JSON is canonical/redacted."""

        try:
            raw_bindings = json.loads(str(row["slot_need_bindings_json"]))
            if not isinstance(raw_bindings, list):
                raise ValueError
            bindings = tuple(
                ContextualRevisitSlotNeedBinding(
                    slot_id=item["slot_id"],
                    need_query_id=item["need_query_id"],
                    need_hash=item["need_hash"],
                )
                for item in raw_bindings
                if isinstance(item, dict)
            )
            if len(bindings) != len(raw_bindings):
                raise ValueError
            contract = ContextualRevisitContract(
                creation_receipt_id=int(row["creation_receipt_id"]),
                association_id=int(row["association_id"]),
                context_cue_id=int(row["context_cue_id"]),
                need_cue_id=int(row["need_cue_id"]),
                domain=str(row["domain"]),
                model_id=str(row["model_id"]),
                embedding_space_id=str(row["embedding_space_id"]),
                dimension=int(row["dimension"]),
                dtype=str(row["dtype"]),
                context_hash=str(row["context_hash"]),
                slot_need_bindings=bindings,
                requirements_fingerprint=str(row["requirements_fingerprint"]),
                source_closure_fingerprint=str(row["source_closure_fingerprint"]),
                retrieval_policy_fingerprint=str(row["retrieval_policy_fingerprint"]),
                budget_fingerprint=str(row["budget_fingerprint"]),
                anchor_manifest_fingerprint=str(row["anchor_manifest_fingerprint"]),
                source_fact_roles_fingerprint=str(row["source_fact_roles_fingerprint"]),
                source_fact_refs_fingerprint=str(row["source_fact_refs_fingerprint"]),
                ready_index_epoch=int(row["ready_index_epoch"]),
                ready_publication_fingerprint=str(
                    row["ready_publication_fingerprint"]
                ),
                contract_version=str(row["contract_version"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("stored contextual revisit contract is invalid") from error
        if contract.slot_need_bindings_json != str(row["slot_need_bindings_json"]):
            raise ValueError("stored contextual revisit bindings are not canonical")
        if contract.contract_fingerprint != str(row["contract_fingerprint"]):
            raise ValueError("stored contextual revisit contract fingerprint mismatch")
        return contract

    def _create_contextual_revisit_contract_in_transaction(
        self,
        connection,
        contract: ContextualRevisitContract,
    ) -> ContextualRevisitContractReceipt:
        payload = self._revisit_contract_payload(contract)
        existing = connection.execute(
            """
            SELECT * FROM contextual_revisit_contract
            WHERE creation_receipt_id = ?
            """,
            (int(contract.creation_receipt_id),),
        ).fetchone()
        if existing is not None:
            if not self._revisit_contract_row_matches(existing, payload):
                raise ValueError(
                    "creation receipt is already bound to a different revisit contract"
                )
            return self._revisit_contract_receipt(existing, idempotent=True)

        fingerprint_existing = connection.execute(
            """
            SELECT * FROM contextual_revisit_contract
            WHERE contract_fingerprint = ?
            """,
            (str(contract.contract_fingerprint),),
        ).fetchone()
        if fingerprint_existing is not None:
            raise ValueError("contextual revisit contract fingerprint collision")

        self._canonical_revisit_receipt_in_transaction(connection, contract)
        fields = tuple(payload)
        placeholders = ", ".join("?" for _ in fields)
        try:
            cursor = connection.execute(
                f"""
                INSERT INTO contextual_revisit_contract(
                    {', '.join(fields)}, created_at
                ) VALUES({placeholders}, ?)
                """,
                [*(payload[field_name] for field_name in fields), utc_now()],
            )
        except sqlite3.IntegrityError as error:
            # A second Database instance can win after this local connection
            # waited for SQLite's writer lock.  Only byte-for-byte equality is
            # an idempotent retry; every divergent collision stays closed.
            existing = connection.execute(
                """
                SELECT * FROM contextual_revisit_contract
                WHERE creation_receipt_id = ?
                """,
                (int(contract.creation_receipt_id),),
            ).fetchone()
            if existing is not None and self._revisit_contract_row_matches(
                existing, payload
            ):
                return self._revisit_contract_receipt(existing, idempotent=True)
            raise ValueError("contextual revisit contract uniqueness conflict") from error
        row = connection.execute(
            "SELECT * FROM contextual_revisit_contract WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return self._revisit_contract_receipt(row, idempotent=False)

    def create_contextual_revisit_contract(
        self,
        contract: ContextualRevisitContract,
    ) -> ContextualRevisitContractReceipt:
        """Persist one typed v16 contract without any query-path side effect.

        This is intentionally not a fast path: it stores a future-use input
        contract only after the linked source-bound creation receipt is ready.
        Historical v14/v15 receipts are never backfilled or inferred here.
        """

        if not isinstance(contract, ContextualRevisitContract):
            raise TypeError("revisit contracts must use the v16 typed record")
        with self.db.transaction() as connection:
            return self._create_contextual_revisit_contract_in_transaction(
                connection, contract
            )

    def load_contextual_revisit_contract(
        self,
        creation_receipt_id: int,
    ) -> ContextualRevisitContract | None:
        """Return a strictly valid v16 contract or a fail-closed cache miss.

        v14/v15 receipts intentionally return ``None`` because their original
        selector/closure inputs cannot be reconstructed safely.  A malformed,
        stale, noncanonical, or no-longer-published durable row also returns a
        miss rather than a partially trusted contract.
        """

        if isinstance(creation_receipt_id, bool):
            raise TypeError("creation_receipt_id must be a positive integer")
        try:
            normalized_receipt_id = int(creation_receipt_id)
        except (TypeError, ValueError) as error:
            raise TypeError("creation_receipt_id must be a positive integer") from error
        if normalized_receipt_id <= 0:
            raise ValueError("creation_receipt_id must be a positive integer")
        with self.db.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM contextual_revisit_contract
                WHERE creation_receipt_id = ?
                """,
                (normalized_receipt_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                contract = self._revisit_contract_from_row(row)
                self._canonical_revisit_receipt_in_transaction(connection, contract)
            except ValueError:
                return None
            return contract

    def find_contextual_revisit_contract(
        self,
        lookup: ContextualRevisitContractLookup,
    ) -> ContextualRevisitContract | None:
        """Find exactly one still-valid contract by its redacted input key.

        A candidate lookup is deliberately not a fuzzy search.  No matching
        row, malformed/stale row, or more than one matching durable contract
        is an ambiguity-safe miss.  The API returns no prose or vector data.
        """

        if not isinstance(lookup, ContextualRevisitContractLookup):
            raise TypeError("revisit contract lookup must use the v16 typed record")
        with self.db.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM contextual_revisit_contract
                WHERE domain = ?
                  AND context_hash = ?
                  AND requirements_fingerprint = ?
                  AND source_closure_fingerprint = ?
                  AND retrieval_policy_fingerprint = ?
                  AND budget_fingerprint = ?
                  AND model_id = ?
                  AND embedding_space_id = ?
                  AND dimension = ?
                  AND dtype = ?
                  AND slot_need_bindings_json = ?
                  AND anchor_manifest_fingerprint = ?
                  AND contract_version = ?
                ORDER BY id
                LIMIT 2
                """,
                (
                    lookup.domain,
                    lookup.context_hash,
                    lookup.requirements_fingerprint,
                    lookup.source_closure_fingerprint,
                    lookup.retrieval_policy_fingerprint,
                    lookup.budget_fingerprint,
                    lookup.model_id,
                    lookup.embedding_space_id,
                    int(lookup.dimension),
                    lookup.dtype,
                    lookup.slot_need_bindings_json,
                    lookup.anchor_manifest_fingerprint,
                    lookup.contract_version,
                ),
            ).fetchall()
            if len(rows) != 1:
                return None
            try:
                contract = self._revisit_contract_from_row(rows[0])
                self._canonical_revisit_receipt_in_transaction(connection, contract)
            except ValueError:
                return None
            return contract

    # ---- Restart-safe public exact revisit manifests (v17) ------------

    @staticmethod
    def _runtime_manifest_select_sql() -> str:
        """The one projection used to rebuild a typed manifest from SQLite."""

        return """
            SELECT manifest.*, receipt.creation_request_id AS runtime_creation_request_id
            FROM contextual_revisit_runtime_manifest AS manifest
            JOIN contextual_creation_receipt AS receipt
              ON receipt.id = manifest.creation_receipt_id
        """

    @staticmethod
    def _cue_vector_fingerprint(value: object) -> str:
        """Hash a current cue blob without returning or copying it into a manifest."""

        try:
            blob = bytes(value)
        except (TypeError, ValueError) as error:
            raise ValueError("contextual runtime manifest cue vector is unavailable") from error
        if not blob:
            raise ValueError("contextual runtime manifest cue vector is empty")
        return "cue-vector:sha256:" + hashlib.sha256(blob).hexdigest()

    @staticmethod
    def _runtime_manifest_row_value_matches(actual: object, expected: object) -> bool:
        if actual is None or expected is None:
            return actual is expected
        return actual == expected

    @staticmethod
    def _runtime_manifest_from_row(row) -> ContextualRevisitRuntimeManifest:
        """Rebuild a V17 row only when every stored safe field is canonical."""

        try:
            raw_bindings = json.loads(str(row["slot_need_bindings_json"]))
            if not isinstance(raw_bindings, list) or len(raw_bindings) != 1:
                raise ValueError
            binding_item = raw_bindings[0]
            if not isinstance(binding_item, dict):
                raise ValueError
            binding = ContextualRevisitSlotNeedBinding(
                slot_id=binding_item["slot_id"],
                need_query_id=binding_item["need_query_id"],
                need_hash=binding_item["need_hash"],
            )
            seed = ContextualRevisitRuntimeSeed(
                creation_request_id=str(row["runtime_creation_request_id"]),
                domain=str(row["domain"]),
                context_scope_hash=str(row["context_scope_hash"]),
                context_hash=str(row["context_hash"]),
                source_request_hash=str(row["source_request_hash"]),
                context_cue_text_hash=str(row["context_cue_text_hash"]),
                need_cue_text_hash=str(row["need_cue_text_hash"]),
                slot_need_bindings=(binding,),
                requirements_fingerprint=str(row["requirements_fingerprint"]),
                source_closure_fingerprint=str(row["source_closure_fingerprint"]),
                retrieval_policy_fingerprint=str(row["retrieval_policy_fingerprint"]),
                budget_fingerprint=str(row["budget_fingerprint"]),
                anchor_manifest_fingerprint=str(row["anchor_manifest_fingerprint"]),
                source_fact_refs_fingerprint=str(
                    row["source_fact_refs_fingerprint"]
                ),
                anchor_episode_id=int(row["anchor_episode_id"]),
                anchor_activation=float(row["anchor_activation"]),
                anchor_source_fact_id=str(row["anchor_source_fact_id"]),
                target_episode_id=int(row["target_episode_id"]),
                target_source_fact_id=str(row["target_source_fact_id"]),
                target_mapping_ref=str(row["target_mapping_ref"]),
                runtime_slot_ref=str(row["runtime_slot_ref"]),
                runtime_query_ref=str(row["runtime_query_ref"]),
                runtime_clause_ref=str(row["runtime_clause_ref"]),
                endpoint_limit=int(row["endpoint_limit"]),
                episode_limit=int(row["episode_limit"]),
                source_fact_limit=(
                    None
                    if row["source_fact_limit"] is None
                    else int(row["source_fact_limit"])
                ),
                delivery_token_limit=(
                    None
                    if row["delivery_token_limit"] is None
                    else int(row["delivery_token_limit"])
                ),
                support_mode=str(row["support_mode"]),
                contract_version=str(row["contract_version"]),
                manifest_version=str(row["manifest_version"]),
            )
            manifest = ContextualRevisitRuntimeManifest(
                manifest_id=int(row["id"]),
                creation_receipt_id=int(row["creation_receipt_id"]),
                association_id=int(row["association_id"]),
                context_cue_id=int(row["context_cue_id"]),
                need_cue_id=int(row["need_cue_id"]),
                model_id=str(row["model_id"]),
                embedding_space_id=str(row["embedding_space_id"]),
                dimension=int(row["dimension"]),
                dtype=str(row["dtype"]),
                seed=seed,
                context_cue_vector_fingerprint=str(
                    row["context_cue_vector_fingerprint"]
                ),
                need_cue_vector_fingerprint=str(
                    row["need_cue_vector_fingerprint"]
                ),
                source_fact_roles_fingerprint=str(
                    row["source_fact_roles_fingerprint"]
                ),
                not_before_at=str(row["not_before_at"]),
                expires_at=str(row["expires_at"]),
                binding_fingerprint=str(row["binding_fingerprint"]),
                state=str(row["state"]),
                ready_index_epoch=(
                    None
                    if row["ready_index_epoch"] is None
                    else int(row["ready_index_epoch"])
                ),
                ready_at=str(row["ready_at"] or ""),
                ready_publication_fingerprint=str(
                    row["ready_publication_fingerprint"] or ""
                ),
                contract_id=(
                    None if row["contract_id"] is None else int(row["contract_id"])
                ),
                contract_fingerprint=str(row["contract_fingerprint"] or ""),
                manifest_fingerprint=str(row["manifest_fingerprint"] or ""),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("stored contextual runtime manifest is invalid") from error

        if seed.slot_need_bindings_json != str(row["slot_need_bindings_json"]):
            raise ValueError("stored contextual runtime manifest binding is not canonical")
        for field_name, expected in manifest.storage_payload().items():
            if not AssociationRepository._runtime_manifest_row_value_matches(
                row[field_name], expected
            ):
                raise ValueError(
                    f"stored contextual runtime manifest {field_name} mismatch"
                )
        if str(row["created_at"] or "") != manifest.not_before_at:
            raise ValueError("stored contextual runtime manifest creation time mismatch")
        return manifest

    @staticmethod
    def _runtime_manifest_receipt(
        manifest: ContextualRevisitRuntimeManifest,
        *,
        idempotent: bool,
        promoted: bool,
    ) -> ContextualRevisitRuntimeManifestReceipt:
        return ContextualRevisitRuntimeManifestReceipt(
            manifest_id=manifest.manifest_id,
            creation_receipt_id=manifest.creation_receipt_id,
            association_id=manifest.association_id,
            state=manifest.state,
            seed_fingerprint=manifest.seed.seed_fingerprint,
            manifest_fingerprint=manifest.manifest_fingerprint,
            contract_id=manifest.contract_id,
            contract_fingerprint=manifest.contract_fingerprint,
            idempotent=idempotent,
            promoted=promoted,
        )

    @staticmethod
    def _runtime_seed_source_facts_from_receipt(receipt) -> dict[str, SourceFactRef]:
        """Decode only existing redacted fact locators; never source text."""

        try:
            raw_facts = json.loads(str(receipt["source_facts_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("runtime manifest receipt source closure is invalid") from error
        if not isinstance(raw_facts, list) or not raw_facts:
            raise ValueError("runtime manifest receipt source closure is empty")
        facts: dict[str, SourceFactRef] = {}
        for payload in raw_facts:
            if not isinstance(payload, Mapping):
                raise ValueError("runtime manifest receipt source fact is invalid")
            try:
                fact = SourceFactRef(
                    source_revision_id=str(payload["source_revision_id"]),
                    record_span=tuple(payload.get("record_span", ()) or ()),
                    span_hash=str(payload.get("span_hash", "")),
                    raw_span_hash=str(payload["raw_span_hash"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("runtime manifest receipt source fact is invalid") from error
            if fact.fact_id in facts:
                raise ValueError("runtime manifest receipt source fact is ambiguous")
            facts[fact.fact_id] = fact
        return facts

    def _validate_runtime_seed_source_proofs_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
    ) -> None:
        """Bind seed fact IDs to both current endpoint sources before persistence."""

        facts = self._runtime_seed_source_facts_from_receipt(receipt)
        required = {
            int(seed.anchor_episode_id): str(seed.anchor_source_fact_id),
            int(seed.target_episode_id): str(seed.target_source_fact_id),
        }
        if len(required) != 2 or any(fact_id not in facts for fact_id in required.values()):
            raise ValueError("runtime manifest source proof is absent from receipt closure")
        # The V17 public lane deliberately has one anchor and one target
        # proof.  A Q1 receipt may legitimately contain *additional* source
        # facts used by the normal planner, so its whole closure cannot be
        # substituted for this narrow runtime closure.  Recompute exactly the
        # two seed-bound fact identities (deduplicated when both endpoints
        # share a fact); each still has to occur in the authoritative receipt
        # and verify against its current Source below.
        normalised_facts = tuple(
            sorted(
                {facts[fact_id] for fact_id in required.values()},
                key=lambda item: (
                    item.identity_key[0],
                    item.identity_key[1],
                    item.identity_key[2],
                    item.identity_key[3],
                ),
            )
        )
        closure_payload = [
            {
                "source_revision_id": item.source_revision_id,
                "record_span": list(item.record_span),
                "span_hash": item.span_hash,
                "raw_span_hash": item.raw_span_hash,
            }
            for item in normalised_facts
        ]
        expected_closure_fingerprint = (
            "revisit-source-closure:sha256:"
            + hashlib.sha256(
                self._canonical_json(closure_payload).encode("utf-8")
            ).hexdigest()
        )
        expected_fact_refs_fingerprint = (
            "revisit-source-fact-refs:sha256:"
            + hashlib.sha256(
                self._canonical_json([item.fact_id for item in normalised_facts]).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        if (
            seed.source_closure_fingerprint != expected_closure_fingerprint
            or seed.source_fact_refs_fingerprint != expected_fact_refs_fingerprint
        ):
            raise ValueError(
                "runtime manifest source closure fingerprint does not match receipt"
            )
        placeholders = ",".join("?" for _ in required)
        rows = connection.execute(
            f"""
            SELECT id, source_id, generation
            FROM episode
            WHERE id IN ({placeholders})
            """,
            tuple(sorted(required)),
        ).fetchall()
        endpoints = {int(row["id"]): row for row in rows}
        if set(endpoints) != set(required):
            raise ValueError("runtime manifest endpoint episode is missing")
        for episode_id, fact_id in required.items():
            endpoint = endpoints[episode_id]
            if int(endpoint["generation"]) != 0:
                raise ValueError("runtime manifest endpoint must remain generation 0")
            _payload, source_id = self._verify_current_source_fact_in_transaction(
                connection, facts[fact_id]
            )
            if int(source_id) != int(endpoint["source_id"]):
                raise ValueError("runtime manifest source proof is bound to another episode")

    def _bound_runtime_manifest_from_seed_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
        manifest_id: int,
        state: str = "pending_index",
        ready_index_epoch: int | None = None,
        ready_at: str = "",
        ready_publication_fingerprint: str = "",
        contract_id: int | None = None,
        contract_fingerprint: str = "",
        manifest_fingerprint: str = "",
    ) -> ContextualRevisitRuntimeManifest:
        """Attach a pre-receipt seed to authoritative cue/edge/receipt rows."""

        if not isinstance(seed, ContextualRevisitRuntimeSeed):
            raise TypeError("runtime manifests must use the V17 typed seed")
        if str(receipt["creation_request_id"] or "") != seed.creation_request_id:
            raise ValueError("runtime manifest seed creation request does not match receipt")
        if str(receipt["source_request_hash"] or "") != seed.source_request_hash:
            raise ValueError("runtime manifest seed source request does not match receipt")
        if str(receipt["domain"] or "") != seed.domain:
            raise ValueError("runtime manifest seed domain does not match receipt")
        if str(receipt["verification_status"] or "") not in {
            "verified",
            "source_bound",
        }:
            raise ValueError("runtime manifest receipt is not source-bound")
        receipt_status = str(receipt["status"] or "")
        if state == "pending_index" and receipt_status not in {
            "committed_pending_index",
            "ready",
        }:
            raise ValueError("pending runtime manifest needs a publishable creation receipt")
        if state == "ready" and receipt_status != "ready":
            raise ValueError("ready runtime manifest needs a ready creation receipt")
        row = connection.execute(
            """
            SELECT edge.*, context_cue.domain AS context_domain,
                   context_cue.cue_kind AS context_kind,
                   context_cue.model_id AS context_model_id,
                   context_cue.dimension AS context_dimension,
                   context_cue.dtype AS context_dtype,
                   context_cue.embedding_space_id AS context_space_id,
                   context_cue.text_hash AS context_text_hash,
                   context_cue.vector_blob AS context_vector_blob,
                   need_cue.domain AS need_domain,
                   need_cue.cue_kind AS need_kind,
                   need_cue.model_id AS need_model_id,
                   need_cue.dimension AS need_dimension,
                   need_cue.dtype AS need_dtype,
                   need_cue.embedding_space_id AS need_space_id,
                   need_cue.text_hash AS need_text_hash,
                   need_cue.vector_blob AS need_vector_blob
            FROM association AS edge
            JOIN association_cue_prototype AS context_cue
              ON context_cue.id = edge.context_cue_id
            JOIN association_cue_prototype AS need_cue
              ON need_cue.id = edge.need_cue_id
            WHERE edge.id = ?
            """,
            (int(receipt["association_id"]),),
        ).fetchone()
        if row is None:
            raise ValueError("runtime manifest association is missing")
        if (
            str(row["association_mode"] or "") != "contextual_recall"
            or str(row["from_type"] or "") != "episode"
            or int(row["from_id"] or 0) != int(seed.anchor_episode_id)
            or str(row["to_type"] or "") != "episode"
            or int(row["to_id"] or 0) != int(seed.target_episode_id)
            or int(row["context_cue_id"] or 0) != int(receipt["context_cue_id"])
            or int(row["need_cue_id"] or 0) != int(receipt["need_cue_id"])
            or not str(row["expires_at"] or "").strip()
        ):
            raise ValueError("runtime manifest association binding is invalid")
        expected_metadata = (
            ("context", "context_kind"),
            ("need", "need_kind"),
        )
        for expected_kind, kind_field in expected_metadata:
            prefix = "context" if expected_kind == "context" else "need"
            if (
                str(row[kind_field] or "") != expected_kind
                or str(row[f"{prefix}_domain"] or "") != str(receipt["domain"])
                or str(row[f"{prefix}_model_id"] or "") != str(receipt["model_id"])
                or int(row[f"{prefix}_dimension"] or 0)
                != int(receipt["dimension"])
                or str(row[f"{prefix}_dtype"] or "") != str(receipt["dtype"])
                or str(row[f"{prefix}_space_id"] or "")
                != str(receipt["embedding_space_id"])
            ):
                raise ValueError("runtime manifest cue metadata does not match receipt")
        if (
            str(row["context_text_hash"] or "") != seed.context_cue_text_hash
            or str(row["need_text_hash"] or "") != seed.need_cue_text_hash
        ):
            raise ValueError("runtime manifest cue text hash does not match seed")
        self._validate_runtime_seed_source_proofs_in_transaction(
            connection, receipt=receipt, seed=seed
        )
        not_before_at = str(receipt["created_at"] or "")
        expires_at = str(row["expires_at"] or "")
        context_vector_fingerprint = self._cue_vector_fingerprint(
            row["context_vector_blob"]
        )
        need_vector_fingerprint = self._cue_vector_fingerprint(row["need_vector_blob"])
        roles_fingerprint = seed.source_fact_roles_fingerprint(
            int(receipt["association_id"])
        )
        binding_fingerprint = (
            ContextualRevisitRuntimeManifest.expected_binding_fingerprint(
                creation_receipt_id=int(receipt["id"]),
                association_id=int(receipt["association_id"]),
                context_cue_id=int(receipt["context_cue_id"]),
                need_cue_id=int(receipt["need_cue_id"]),
                model_id=str(receipt["model_id"]),
                embedding_space_id=str(receipt["embedding_space_id"]),
                dimension=int(receipt["dimension"]),
                dtype=str(receipt["dtype"]),
                seed=seed,
                context_cue_vector_fingerprint=context_vector_fingerprint,
                need_cue_vector_fingerprint=need_vector_fingerprint,
                source_fact_roles_fingerprint=roles_fingerprint,
                not_before_at=not_before_at,
                expires_at=expires_at,
            )
        )
        return ContextualRevisitRuntimeManifest(
            manifest_id=manifest_id,
            creation_receipt_id=int(receipt["id"]),
            association_id=int(receipt["association_id"]),
            context_cue_id=int(receipt["context_cue_id"]),
            need_cue_id=int(receipt["need_cue_id"]),
            model_id=str(receipt["model_id"]),
            embedding_space_id=str(receipt["embedding_space_id"]),
            dimension=int(receipt["dimension"]),
            dtype=str(receipt["dtype"]),
            seed=seed,
            context_cue_vector_fingerprint=context_vector_fingerprint,
            need_cue_vector_fingerprint=need_vector_fingerprint,
            source_fact_roles_fingerprint=roles_fingerprint,
            not_before_at=not_before_at,
            expires_at=expires_at,
            binding_fingerprint=binding_fingerprint,
            state=state,
            ready_index_epoch=ready_index_epoch,
            ready_at=ready_at,
            ready_publication_fingerprint=ready_publication_fingerprint,
            contract_id=contract_id,
            contract_fingerprint=contract_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
        )

    def _load_runtime_manifest_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
    ) -> ContextualRevisitRuntimeManifest | None:
        row = connection.execute(
            self._runtime_manifest_select_sql()
            + " WHERE manifest.creation_receipt_id = ?",
            (int(creation_receipt_id),),
        ).fetchone()
        if row is None:
            return None
        return self._runtime_manifest_from_row(row)

    def _assert_runtime_seed_matches_existing_receipt_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
    ) -> None:
        """Make a retry exact without retroactively backfilling legacy rows."""

        if str(receipt["creation_request_id"] or "") != seed.creation_request_id:
            raise ValueError("runtime manifest seed creation request does not match receipt")
        manifest = self._load_runtime_manifest_in_transaction(
            connection, int(receipt["id"])
        )
        if manifest is None:
            # This is an older receipt or an earlier software version.  It is
            # intentionally not backfilled outside the original Q1 commit.
            return
        if manifest.seed != seed:
            raise ValueError("creation receipt is already bound to a different runtime seed")

    @staticmethod
    def _restricted_rewrite_guard_from_row(row) -> ContextualRestrictedRewriteGuard:
        """Rebuild one redacted V22 sidecar, rejecting any non-canonical row."""

        try:
            return ContextualRestrictedRewriteGuard(
                creation_receipt_id=int(row["creation_receipt_id"]),
                association_id=int(row["association_id"]),
                domain=str(row["domain"]),
                context_scope_hash=str(row["context_scope_hash"]),
                rewrite_commitment=str(row["rewrite_commitment"]),
                binding_commitment=str(row["binding_commitment"]),
                manifest_binding_commitment=str(
                    row["manifest_binding_commitment"]
                ),
                ready_manifest_commitment=str(
                    row["ready_manifest_commitment"]
                ),
                context_cue_vector_fingerprint=str(
                    row["context_cue_vector_fingerprint"]
                ),
                need_cue_vector_fingerprint=str(
                    row["need_cue_vector_fingerprint"]
                ),
                model_id=str(row["model_id"]),
                embedding_space_id=str(row["embedding_space_id"]),
                dimension=int(row["dimension"]),
                dtype=str(row["dtype"]),
                commitment_key_id=str(row["commitment_key_id"]),
                grammar_version=str(row["grammar_version"]),
                seed_fingerprint=str(row["seed_fingerprint"]),
                signature_version=str(row["signature_version"]),
                created_at=str(row["created_at"]),
                guard_fingerprint=str(row["guard_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("stored restricted rewrite guard is invalid") from error

    def _load_restricted_rewrite_guard_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
    ) -> ContextualRestrictedRewriteGuard | None:
        row = connection.execute(
            """
            SELECT *
            FROM contextual_restricted_rewrite_guard
            WHERE creation_receipt_id = ?
            """,
            (int(creation_receipt_id),),
        ).fetchone()
        if row is None:
            return None
        return self._restricted_rewrite_guard_from_row(row)

    def _v22_restricted_rewrite_guard_or_legacy_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
    ) -> ContextualRestrictedRewriteGuard | None:
        """Return a typed V5 guard, or ``None`` only for known legacy rows."""

        try:
            guard = self._load_restricted_rewrite_guard_in_transaction(
                connection, int(creation_receipt_id)
            )
        except ValueError as error:
            row = connection.execute(
                """
                SELECT signature_version
                FROM contextual_restricted_rewrite_guard
                WHERE creation_receipt_id = ?
                """,
                (int(creation_receipt_id),),
            ).fetchone()
            signature_version = str(row["signature_version"] or "") if row else ""
            if signature_version in _RESTRICTED_REWRITE_LEGACY_SIGNATURE_VERSIONS:
                # V21 and older rows are deliberately unreadable under the V22
                # typed version check.  They cannot acquire a new ready HMAC.
                return None
            # Do not silently downgrade a malformed self-described V5 (or an
            # unrecognised future) sidecar into an unguarded V17 publication.
            # The receipt must stay pending until an operator diagnoses it.
            raise RestrictedRewriteReadyBindingError(
                "restricted rewrite guard has an invalid V22/future signature"
            ) from error
        return guard

    def _pending_v22_restricted_rewrite_guard_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
    ) -> ContextualRestrictedRewriteGuard | None:
        """Return a fresh V22 pending guard without reviving older sidecars."""

        guard = self._v22_restricted_rewrite_guard_or_legacy_in_transaction(
            connection, int(creation_receipt_id)
        )
        if guard is None:
            return None
        if guard.ready_manifest_commitment:
            raise RestrictedRewriteReadyBindingError(
                "pending restricted rewrite guard is already ready-bound"
            )
        return guard

    def _ready_v22_restricted_rewrite_guard_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
    ) -> ContextualRestrictedRewriteGuard | None:
        """Require the one-shot ready HMAC on every readable V5 sidecar."""

        guard = self._v22_restricted_rewrite_guard_or_legacy_in_transaction(
            connection, int(creation_receipt_id)
        )
        if guard is not None and not guard.ready_manifest_commitment:
            raise RestrictedRewriteReadyBindingError(
                "ready restricted rewrite guard is missing its ready HMAC"
            )
        return guard

    @staticmethod
    def _restricted_rewrite_guard_expected(
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
        guard: ContextualRestrictedRewriteGuardDraft,
        manifest: ContextualRevisitRuntimeManifest,
        manifest_binding_signer: Callable[
            [
                ContextualRestrictedRewriteGuardDraft,
                ContextualRevisitRuntimeSeed,
                str,
            ],
            str,
        ],
    ) -> ContextualRestrictedRewriteGuard:
        """Bind a trusted draft to one canonical manifest inside Q1's transaction."""

        if not isinstance(seed, ContextualRevisitRuntimeSeed):
            raise TypeError("restricted rewrite guard requires a runtime seed")
        if not isinstance(guard, ContextualRestrictedRewriteGuardDraft):
            raise TypeError("restricted rewrite guard requires the typed T16 draft")
        if not isinstance(manifest, ContextualRevisitRuntimeManifest):
            raise TypeError("restricted rewrite guard requires a typed runtime manifest")
        if not callable(manifest_binding_signer):
            raise TypeError("restricted rewrite guard requires a process-only manifest signer")
        if (
            str(receipt["creation_request_id"] or "") != guard.creation_request_id
            or str(receipt["domain"] or "") != guard.domain
            or str(seed.creation_request_id) != guard.creation_request_id
            or str(seed.domain) != guard.domain
            or str(seed.context_scope_hash) != guard.context_scope_hash
        ):
            raise ValueError("restricted rewrite guard does not bind the receipt seed")
        try:
            manifest_binding_commitment = manifest_binding_signer(
                guard,
                seed,
                str(manifest.binding_fingerprint),
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise ValueError(
                "restricted rewrite manifest signer rejected the canonical pending binding"
            ) from error
        return guard.bind(
            creation_receipt_id=int(receipt["id"]),
            association_id=int(receipt["association_id"]),
            seed_fingerprint=seed.seed_fingerprint,
            created_at=str(receipt["created_at"]),
            manifest_binding_commitment=manifest_binding_commitment,
        )

    def _assert_restricted_rewrite_guard_matches_existing_receipt_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
        guard: ContextualRestrictedRewriteGuardDraft,
        manifest_binding_signer: Callable[
            [
                ContextualRestrictedRewriteGuardDraft,
                ContextualRevisitRuntimeSeed,
                str,
            ],
            str,
        ],
    ) -> None:
        """Make a retry exact without backfilling a pre-V21 sidecar."""

        manifest = self._load_runtime_manifest_in_transaction(
            connection, int(receipt["id"])
        )
        if manifest is None:
            # Older software created this receipt.  It deliberately never
            # acquires a fresh rewrite authorization during a retry.
            return
        if manifest.seed != seed:
            raise ValueError("creation receipt runtime seed no longer matches guard")
        self._validate_runtime_manifest_in_transaction(
            connection, manifest, require_ready=False
        )
        expected = self._restricted_rewrite_guard_expected(
            receipt=receipt,
            seed=seed,
            guard=guard,
            manifest=manifest,
            manifest_binding_signer=manifest_binding_signer,
        )
        existing = self._load_restricted_rewrite_guard_in_transaction(
            connection, int(receipt["id"])
        )
        if existing is None:
            # V17 rows do not have the necessary original parser commitment.
            # A retry must remain a normal exact-only receipt, not a backfill.
            return
        if existing != expected:
            raise ValueError("creation receipt is already bound to another rewrite guard")

    def _create_contextual_restricted_rewrite_guard_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
        guard: ContextualRestrictedRewriteGuardDraft,
        manifest_binding_signer: Callable[
            [
                ContextualRestrictedRewriteGuardDraft,
                ContextualRevisitRuntimeSeed,
                str,
            ],
            str,
        ],
    ) -> ContextualRestrictedRewriteGuard:
        """Create one sidecar immediately after its pending V17 seed.

        The insert remains inside the outer Q1 transaction.  If its HMAC
        metadata or its manifest binding is inconsistent, cues, edge, receipt,
        manifest and sidecar all roll back together.
        """

        if str(receipt["status"] or "") != "committed_pending_index":
            raise ValueError("restricted rewrite guard requires a pending receipt")
        manifest = self._load_runtime_manifest_in_transaction(
            connection, int(receipt["id"])
        )
        if (
            manifest is None
            or manifest.state != "pending_index"
            or manifest.seed != seed
            or int(manifest.association_id) != int(receipt["association_id"])
            or str(manifest.seed.context_scope_hash)
            != str(guard.context_scope_hash)
            or str(manifest.seed.seed_fingerprint)
            != str(seed.seed_fingerprint)
            or str(manifest.context_cue_vector_fingerprint)
            != str(guard.context_cue_vector_fingerprint)
            or str(manifest.need_cue_vector_fingerprint)
            != str(guard.need_cue_vector_fingerprint)
            or str(manifest.model_id) != str(guard.model_id)
            or str(manifest.embedding_space_id)
            != str(guard.embedding_space_id)
            or int(manifest.dimension) != int(guard.dimension)
            or str(manifest.dtype) != str(guard.dtype)
        ):
            raise ValueError("restricted rewrite guard requires its pending manifest")
        self._validate_runtime_manifest_in_transaction(
            connection, manifest, require_ready=False
        )
        expected = self._restricted_rewrite_guard_expected(
            receipt=receipt,
            seed=seed,
            guard=guard,
            manifest=manifest,
            manifest_binding_signer=manifest_binding_signer,
        )
        existing = self._load_restricted_rewrite_guard_in_transaction(
            connection, int(receipt["id"])
        )
        if existing is not None:
            if existing != expected:
                raise ValueError("creation receipt is already bound to another rewrite guard")
            return existing
        payload = expected.storage_payload()
        fields = tuple(payload)
        placeholders = ", ".join("?" for _ in fields)
        try:
            connection.execute(
                f"""
                INSERT INTO contextual_restricted_rewrite_guard(
                    {', '.join(fields)}
                ) VALUES({placeholders})
                """,
                [payload[field_name] for field_name in fields],
            )
        except sqlite3.IntegrityError as error:
            existing = self._load_restricted_rewrite_guard_in_transaction(
                connection, int(receipt["id"])
            )
            if existing == expected:
                return existing
            raise ValueError("restricted rewrite guard uniqueness conflict") from error
        stored = self._load_restricted_rewrite_guard_in_transaction(
            connection, int(receipt["id"])
        )
        if stored != expected:
            raise ValueError("restricted rewrite guard did not persist canonically")
        return stored

    def _create_contextual_revisit_runtime_seed_in_transaction(
        self,
        connection,
        *,
        receipt,
        seed: ContextualRevisitRuntimeSeed,
    ) -> ContextualRevisitRuntimeManifestReceipt:
        """Insert one pending seed as part of the cue/edge/receipt transaction."""

        if str(receipt["status"] or "") != "committed_pending_index":
            raise ValueError("runtime manifest seed requires a pending creation receipt")
        existing = self._load_runtime_manifest_in_transaction(connection, int(receipt["id"]))
        if existing is not None:
            if existing.seed != seed:
                raise ValueError("creation receipt is already bound to a different runtime seed")
            return self._runtime_manifest_receipt(
                existing, idempotent=True, promoted=False
            )
        provisional = self._bound_runtime_manifest_from_seed_in_transaction(
            connection,
            receipt=receipt,
            seed=seed,
            manifest_id=1,
        )
        payload = provisional.storage_payload()
        fields = tuple(payload)
        placeholders = ", ".join("?" for _ in fields)
        try:
            cursor = connection.execute(
                f"""
                INSERT INTO contextual_revisit_runtime_manifest(
                    {', '.join(fields)}, created_at
                ) VALUES({placeholders}, ?)
                """,
                [*(payload[field_name] for field_name in fields), provisional.not_before_at],
            )
        except sqlite3.IntegrityError as error:
            existing = self._load_runtime_manifest_in_transaction(
                connection, int(receipt["id"])
            )
            if existing is not None and existing.seed == seed:
                return self._runtime_manifest_receipt(
                    existing, idempotent=True, promoted=False
                )
            raise ValueError("contextual runtime manifest uniqueness conflict") from error
        row = connection.execute(
            self._runtime_manifest_select_sql() + " WHERE manifest.id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        manifest = self._runtime_manifest_from_row(row)
        return self._runtime_manifest_receipt(
            manifest, idempotent=False, promoted=False
        )

    @staticmethod
    def _runtime_manifest_payload_matches(
        actual: ContextualRevisitRuntimeManifest,
        expected: ContextualRevisitRuntimeManifest,
    ) -> bool:
        """Compare every durable field, including current cue/blob bindings."""

        return actual.storage_payload() == expected.storage_payload()

    def _runtime_manifest_contract_for_ready(
        self,
        manifest: ContextualRevisitRuntimeManifest,
        *,
        receipt,
    ) -> ContextualRevisitContract:
        """Build the sole V16 projection from a receipt-bound V17 seed.

        This is intentionally internal: callers cannot submit an arbitrary
        V16 contract during promotion.  The ready receipt supplies the epoch
        and publication fingerprint; all other inputs come from the immutable
        manifest and its seed.
        """

        if manifest.state != "pending_index":
            raise ValueError("only a pending runtime manifest can be promoted")
        if str(receipt["status"] or "") != "ready":
            raise ValueError("runtime manifest promotion requires a ready receipt")
        try:
            ready_index_epoch = int(receipt["ready_index_epoch"])
        except (TypeError, ValueError) as error:
            raise ValueError("ready receipt has no valid index epoch") from error
        if ready_index_epoch <= 0 or not str(receipt["ready_at"] or "").strip():
            raise ValueError("ready receipt has incomplete publication metadata")
        ready_publication_fingerprint = (
            self.contextual_revisit_ready_publication_fingerprint(dict(receipt))
        )
        return ContextualRevisitContract(
            creation_receipt_id=manifest.creation_receipt_id,
            association_id=manifest.association_id,
            context_cue_id=manifest.context_cue_id,
            need_cue_id=manifest.need_cue_id,
            domain=manifest.seed.domain,
            model_id=manifest.model_id,
            embedding_space_id=manifest.embedding_space_id,
            dimension=manifest.dimension,
            dtype=manifest.dtype,
            context_hash=manifest.seed.context_hash,
            slot_need_bindings=manifest.seed.slot_need_bindings,
            requirements_fingerprint=manifest.seed.requirements_fingerprint,
            source_closure_fingerprint=manifest.seed.source_closure_fingerprint,
            retrieval_policy_fingerprint=manifest.seed.retrieval_policy_fingerprint,
            budget_fingerprint=manifest.seed.budget_fingerprint,
            anchor_manifest_fingerprint=manifest.seed.anchor_manifest_fingerprint,
            source_fact_roles_fingerprint=manifest.source_fact_roles_fingerprint,
            source_fact_refs_fingerprint=manifest.seed.source_fact_refs_fingerprint,
            ready_index_epoch=ready_index_epoch,
            ready_publication_fingerprint=ready_publication_fingerprint,
            contract_version=manifest.seed.contract_version,
        )

    def _validate_runtime_manifest_in_transaction(
        self,
        connection,
        manifest: ContextualRevisitRuntimeManifest,
        *,
        require_ready: bool,
    ) -> ContextualRevisitRuntimeManifest:
        """Re-bind a stored row to live authoritative rows or reject it.

        This is the load-time guard that makes direct cue-vector replacement,
        expired/source-revised facts, receipt drift and contract substitution a
        normal-path cache miss rather than a hidden automatic revisit.
        """

        receipt = connection.execute(
            "SELECT * FROM contextual_creation_receipt WHERE id = ?",
            (int(manifest.creation_receipt_id),),
        ).fetchone()
        if receipt is None:
            raise ValueError("runtime manifest creation receipt is missing")
        if require_ready and manifest.state != "ready":
            raise ValueError("runtime manifest is not ready")
        rebound = self._bound_runtime_manifest_from_seed_in_transaction(
            connection,
            receipt=receipt,
            seed=manifest.seed,
            manifest_id=manifest.manifest_id,
            state=manifest.state,
            ready_index_epoch=manifest.ready_index_epoch,
            ready_at=manifest.ready_at,
            ready_publication_fingerprint=manifest.ready_publication_fingerprint,
            contract_id=manifest.contract_id,
            contract_fingerprint=manifest.contract_fingerprint,
            manifest_fingerprint=manifest.manifest_fingerprint,
        )
        if not self._runtime_manifest_payload_matches(manifest, rebound):
            raise ValueError("runtime manifest no longer matches current bindings")
        if manifest.state != "ready":
            return manifest

        contract = manifest.to_revisit_contract()
        if contract.contract_fingerprint != manifest.contract_fingerprint:
            raise ValueError("runtime manifest contract fingerprint mismatch")
        row = connection.execute(
            "SELECT * FROM contextual_revisit_contract WHERE id = ?",
            (int(manifest.contract_id or 0),),
        ).fetchone()
        if row is None:
            raise ValueError("runtime manifest contract is missing")
        stored_contract = self._revisit_contract_from_row(row)
        if stored_contract != contract:
            raise ValueError("runtime manifest contract does not match seed")
        self._canonical_revisit_receipt_in_transaction(connection, stored_contract)
        return manifest

    def _promote_contextual_revisit_runtime_manifest_in_transaction(
        self,
        connection,
        creation_receipt_id: int,
        *,
        restricted_rewrite_ready_manifest_signer: Callable[
            [ContextualRestrictedRewriteGuard, ContextualRevisitRuntimeManifest],
            str,
        ]
        | None = None,
        _allow_restricted_rewrite_ready_transition: bool = False,
    ) -> ContextualRevisitRuntimeManifestReceipt | None:
        """Atomically make a pending seed public with its sole V16 contract."""

        manifest = self._load_runtime_manifest_in_transaction(
            connection, int(creation_receipt_id)
        )
        v22_guard = self._v22_restricted_rewrite_guard_or_legacy_in_transaction(
            connection, int(creation_receipt_id)
        )
        if manifest is None:
            if v22_guard is not None:
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite guard is missing its canonical runtime manifest"
                )
            return None
        if manifest.state == "ready":
            if v22_guard is not None and not v22_guard.ready_manifest_commitment:
                raise RestrictedRewriteReadyBindingError(
                    "ready runtime manifest has a blank restricted rewrite guard"
                )
            self._ready_v22_restricted_rewrite_guard_in_transaction(
                connection, int(creation_receipt_id)
            )
            self._validate_runtime_manifest_in_transaction(
                connection, manifest, require_ready=True
            )
            return self._runtime_manifest_receipt(
                manifest, idempotent=True, promoted=False
            )

        receipt = connection.execute(
            "SELECT * FROM contextual_creation_receipt WHERE id = ?",
            (int(creation_receipt_id),),
        ).fetchone()
        if receipt is None:
            raise ValueError("runtime manifest creation receipt is missing")
        if str(receipt["status"] or "") != "ready":
            # A pending receipt is an expected crash/interleaving state, not a
            # reason to fabricate readiness.  Its index/rebuild owner will
            # call this again only after the canonical receipt is published.
            if str(receipt["status"] or "") == "committed_pending_index":
                return self._runtime_manifest_receipt(
                    manifest, idempotent=True, promoted=False
                )
            raise ValueError("runtime manifest receipt cannot be promoted")

        # A V22 restricted-rewrite guard can receive its final HMAC only in
        # the same trusted transaction that performs the authoritative
        # receipt pending->ready transition.  Never let a later standalone
        # ``promote`` call bless a database-written ready receipt.
        pending_guard = v22_guard
        if pending_guard is not None:
            if pending_guard.ready_manifest_commitment:
                raise RestrictedRewriteReadyBindingError(
                    "pending runtime manifest already has a ready rewrite commitment"
                )
            if not _allow_restricted_rewrite_ready_transition:
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite ready HMAC requires the trusted publication transition"
                )
            if not callable(restricted_rewrite_ready_manifest_signer):
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite ready HMAC requires a process-only signer"
                )

        # Recompute source and cue-blob bindings *before* inserting a V16
        # contract.  If any mutable authority changed, the transaction exits
        # with no contract and the automatic path remains unavailable.
        rebound_pending = self._bound_runtime_manifest_from_seed_in_transaction(
            connection,
            receipt=receipt,
            seed=manifest.seed,
            manifest_id=manifest.manifest_id,
        )
        if not self._runtime_manifest_payload_matches(manifest, rebound_pending):
            raise ValueError("pending runtime manifest no longer matches current bindings")
        if rebound_pending.expires_at <= utc_now():
            raise ValueError("pending runtime manifest is already expired")
        contract = self._runtime_manifest_contract_for_ready(
            manifest, receipt=receipt
        )
        contract_receipt = self._create_contextual_revisit_contract_in_transaction(
            connection, contract
        )
        ready_index_epoch = int(receipt["ready_index_epoch"])
        ready_at = str(receipt["ready_at"])
        ready_publication_fingerprint = (
            self.contextual_revisit_ready_publication_fingerprint(dict(receipt))
        )
        manifest_fingerprint = ContextualRevisitRuntimeManifest.expected_manifest_fingerprint(
            binding_fingerprint=manifest.binding_fingerprint,
            contract_id=contract_receipt.contract_id,
            contract_fingerprint=contract_receipt.contract_fingerprint,
            ready_at=ready_at,
            ready_index_epoch=ready_index_epoch,
            ready_publication_fingerprint=ready_publication_fingerprint,
        )
        ready_manifest = self._bound_runtime_manifest_from_seed_in_transaction(
            connection,
            receipt=receipt,
            seed=manifest.seed,
            manifest_id=manifest.manifest_id,
            state="ready",
            ready_index_epoch=ready_index_epoch,
            ready_at=ready_at,
            ready_publication_fingerprint=ready_publication_fingerprint,
            contract_id=contract_receipt.contract_id,
            contract_fingerprint=contract_receipt.contract_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
        )
        if ready_manifest.to_revisit_contract().contract_fingerprint != (
            contract_receipt.contract_fingerprint
        ):
            raise ValueError("runtime manifest promotion contract projection mismatch")
        ready_guard: ContextualRestrictedRewriteGuard | None = None
        if pending_guard is not None:
            try:
                ready_commitment = restricted_rewrite_ready_manifest_signer(
                    pending_guard, ready_manifest
                )
                ready_guard = pending_guard.with_ready_manifest_commitment(
                    ready_commitment
                )
            except (TypeError, ValueError, UnicodeError) as error:
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite ready signer rejected the canonical ready manifest"
                ) from error
        cursor = connection.execute(
            """
            UPDATE contextual_revisit_runtime_manifest
            SET state = 'ready', ready_index_epoch = ?, ready_at = ?,
                ready_publication_fingerprint = ?, contract_id = ?,
                contract_fingerprint = ?, manifest_fingerprint = ?
            WHERE id = ?
              AND state = 'pending_index'
              AND binding_fingerprint = ?
            """,
            (
                ready_manifest.ready_index_epoch,
                ready_manifest.ready_at,
                ready_manifest.ready_publication_fingerprint,
                ready_manifest.contract_id,
                ready_manifest.contract_fingerprint,
                ready_manifest.manifest_fingerprint,
                ready_manifest.manifest_id,
                ready_manifest.binding_fingerprint,
            ),
        )
        if int(cursor.rowcount) != 1:
            raise ValueError("runtime manifest ready transition lost its pending row")
        stored = self._load_runtime_manifest_in_transaction(
            connection, int(creation_receipt_id)
        )
        if stored is None or stored != ready_manifest:
            raise ValueError("runtime manifest ready transition did not persist canonically")
        if ready_guard is not None:
            cursor = connection.execute(
                """
                UPDATE contextual_restricted_rewrite_guard
                SET ready_manifest_commitment = ?, guard_fingerprint = ?
                WHERE creation_receipt_id = ?
                  AND ready_manifest_commitment = ''
                  AND guard_fingerprint = ?
                """,
                (
                    ready_guard.ready_manifest_commitment,
                    ready_guard.guard_fingerprint,
                    int(ready_guard.creation_receipt_id),
                    pending_guard.guard_fingerprint,
                ),
            )
            if int(cursor.rowcount) != 1:
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite ready guard transition lost its pending row"
                )
            persisted_guard = self._load_restricted_rewrite_guard_in_transaction(
                connection, int(ready_guard.creation_receipt_id)
            )
            if persisted_guard != ready_guard:
                raise RestrictedRewriteReadyBindingError(
                    "restricted rewrite ready guard did not persist canonically"
                )
        self._ready_v22_restricted_rewrite_guard_in_transaction(
            connection, int(creation_receipt_id)
        )
        self._validate_runtime_manifest_in_transaction(
            connection, stored, require_ready=True
        )
        return self._runtime_manifest_receipt(
            stored, idempotent=False, promoted=True
        )

    def promote_contextual_revisit_runtime_manifest(
        self,
        creation_receipt_id: int,
    ) -> ContextualRevisitRuntimeManifestReceipt | None:
        """Promote one ready seed; the caller cannot provide a V16 contract.

        A missing manifest is an ordinary compatibility miss.  A non-ready
        manifest is returned as pending.  Any mismatched source/cue/receipt or
        conflicting durable contract raises and rolls the transaction back.
        """

        if isinstance(creation_receipt_id, bool):
            raise TypeError("creation_receipt_id must be a positive integer")
        try:
            receipt_id = int(creation_receipt_id)
        except (TypeError, ValueError) as error:
            raise TypeError("creation_receipt_id must be a positive integer") from error
        if receipt_id <= 0:
            raise ValueError("creation_receipt_id must be a positive integer")
        with self.db.transaction() as connection:
            return self._promote_contextual_revisit_runtime_manifest_in_transaction(
                connection, receipt_id
            )

    @staticmethod
    def _runtime_manifest_evaluation_as_of(value: str | None) -> str:
        """Normalize an optional public lookup clock without trusting local text."""

        return normalize_evaluation_as_of(value if value is not None else utc_now())

    def load_contextual_revisit_runtime_manifest(
        self,
        creation_receipt_id: int,
        *,
        evaluation_as_of: str | None = None,
    ) -> ContextualRevisitRuntimeManifest | None:
        """Load a ready, unexpired and fully re-bound manifest or fail closed."""

        if isinstance(creation_receipt_id, bool):
            raise TypeError("creation_receipt_id must be a positive integer")
        try:
            receipt_id = int(creation_receipt_id)
        except (TypeError, ValueError) as error:
            raise TypeError("creation_receipt_id must be a positive integer") from error
        if receipt_id <= 0:
            raise ValueError("creation_receipt_id must be a positive integer")
        as_of = self._runtime_manifest_evaluation_as_of(evaluation_as_of)
        with self.db.connection() as connection:
            manifest = self._load_runtime_manifest_in_transaction(connection, receipt_id)
            if manifest is None or manifest.state != "ready":
                return None
            try:
                self._validate_runtime_manifest_in_transaction(
                    connection, manifest, require_ready=True
                )
            except ValueError:
                return None
            if manifest.not_before_at > as_of or manifest.expires_at <= as_of:
                return None
            return manifest

    def find_contextual_revisit_runtime_manifest(
        self,
        lookup: ContextualRevisitRuntimeManifestLookup,
        *,
        evaluation_as_of: str | None = None,
    ) -> ContextualRevisitRuntimeManifest | None:
        """Return one exact public Q2 seed only if its live binding still holds."""

        if not isinstance(lookup, ContextualRevisitRuntimeManifestLookup):
            raise TypeError("runtime manifest lookup must use the V17 typed record")
        as_of = self._runtime_manifest_evaluation_as_of(evaluation_as_of)
        with self.db.connection() as connection:
            rows = connection.execute(
                self._runtime_manifest_select_sql()
                + """
                  WHERE manifest.state = 'ready'
                    AND manifest.domain = ?
                    AND manifest.context_scope_hash = ?
                    AND manifest.context_hash = ?
                    AND manifest.source_request_hash = ?
                    AND manifest.model_id = ?
                    AND manifest.embedding_space_id = ?
                    AND manifest.dimension = ?
                    AND manifest.dtype = ?
                    AND manifest.manifest_version = ?
                  ORDER BY manifest.id
                  LIMIT 2
                """,
                (
                    lookup.domain,
                    lookup.context_scope_hash,
                    lookup.context_hash,
                    lookup.source_request_hash,
                    lookup.model_id,
                    lookup.embedding_space_id,
                    int(lookup.dimension),
                    lookup.dtype,
                    lookup.manifest_version,
                ),
            ).fetchall()
            if len(rows) != 1:
                return None
            try:
                manifest = self._runtime_manifest_from_row(rows[0])
                self._validate_runtime_manifest_in_transaction(
                    connection, manifest, require_ready=True
                )
            except ValueError:
                return None
            if manifest.not_before_at > as_of or manifest.expires_at <= as_of:
                return None
            return manifest

    def load_contextual_restricted_rewrite_guard(
        self,
        creation_receipt_id: int,
        *,
        evaluation_as_of: str | None = None,
    ) -> tuple[
        ContextualRestrictedRewriteGuard,
        ContextualRevisitRuntimeManifest,
    ] | None:
        """Load a sidecar only through its ready, live V17 manifest.

        This is intentionally not a bare table read.  The manifest remains the
        authority for the receipt, cue blobs, source closure, V16 contract,
        index publication and expiry; a stale or tampered sidecar therefore
        becomes an ordinary cache miss.
        """

        if isinstance(creation_receipt_id, bool):
            raise TypeError("creation_receipt_id must be a positive integer")
        try:
            receipt_id = int(creation_receipt_id)
        except (TypeError, ValueError) as error:
            raise TypeError("creation_receipt_id must be a positive integer") from error
        if receipt_id <= 0:
            raise ValueError("creation_receipt_id must be a positive integer")
        as_of = self._runtime_manifest_evaluation_as_of(evaluation_as_of)
        with self.db.connection() as connection:
            try:
                guard = self._load_restricted_rewrite_guard_in_transaction(
                    connection, receipt_id
                )
                manifest = self._load_runtime_manifest_in_transaction(
                    connection, receipt_id
                )
                if guard is None or manifest is None or manifest.state != "ready":
                    return None
                if (
                    not str(guard.ready_manifest_commitment or "")
                    or int(guard.creation_receipt_id) != int(manifest.creation_receipt_id)
                    or int(guard.association_id) != int(manifest.association_id)
                    or str(guard.domain) != str(manifest.seed.domain)
                    or str(guard.context_scope_hash)
                    != str(manifest.seed.context_scope_hash)
                    or str(guard.seed_fingerprint)
                    != str(manifest.seed.seed_fingerprint)
                    or str(guard.created_at) != str(manifest.not_before_at)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(manifest.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(manifest.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(manifest.model_id)
                    or str(guard.embedding_space_id)
                    != str(manifest.embedding_space_id)
                    or int(guard.dimension) != int(manifest.dimension)
                    or str(guard.dtype) != str(manifest.dtype)
                ):
                    return None
                self._validate_runtime_manifest_in_transaction(
                    connection, manifest, require_ready=True
                )
                if manifest.not_before_at > as_of or manifest.expires_at <= as_of:
                    return None
                return guard, manifest
            except (sqlite3.DatabaseError, ValueError, TypeError):
                return None

    def find_contextual_restricted_rewrite_guard(
        self,
        lookup: ContextualRestrictedRewriteGuardLookup,
        *,
        evaluation_as_of: str | None = None,
    ) -> tuple[
        ContextualRestrictedRewriteGuard,
        ContextualRevisitRuntimeManifest,
    ] | None:
        """Find one exact HMAC rewrite authorization or fail closed.

        Association matching may propose the edge ID, but it never proves
        semantic equivalence.  The current request must calculate the same
        scoped HMAC root through the restricted grammar before this lookup can
        return a manifest for the existing T15 verifier.
        """

        if not isinstance(lookup, ContextualRestrictedRewriteGuardLookup):
            raise TypeError("restricted rewrite lookup must use the typed T16 record")
        as_of = self._runtime_manifest_evaluation_as_of(evaluation_as_of)
        association_clause = ""
        parameters: list[object] = []
        if lookup.association_id is not None:
            association_clause = "AND association_id = ?"
            parameters.append(int(lookup.association_id))
        parameters.extend(
            (
                lookup.domain,
                lookup.context_scope_hash,
                lookup.rewrite_commitment,
                lookup.commitment_key_id,
                lookup.grammar_version,
                lookup.signature_version,
            )
        )
        with self.db.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM contextual_restricted_rewrite_guard
                WHERE 1 = 1 {association_clause}
                  AND domain = ?
                  AND context_scope_hash = ?
                  AND rewrite_commitment = ?
                  AND commitment_key_id = ?
                  AND grammar_version = ?
                  AND signature_version = ?
                  ORDER BY id
                  LIMIT 2
                """,
                parameters,
            ).fetchall()
            if len(rows) != 1:
                return None
            try:
                guard = self._restricted_rewrite_guard_from_row(rows[0])
                manifest = self._load_runtime_manifest_in_transaction(
                    connection, guard.creation_receipt_id
                )
                if manifest is None or manifest.state != "ready":
                    return None
                if (
                    not str(guard.ready_manifest_commitment or "")
                    or int(guard.creation_receipt_id) != int(manifest.creation_receipt_id)
                    or int(guard.association_id) != int(manifest.association_id)
                    or str(guard.domain) != str(manifest.seed.domain)
                    or str(guard.context_scope_hash)
                    != str(manifest.seed.context_scope_hash)
                    or str(guard.seed_fingerprint)
                    != str(manifest.seed.seed_fingerprint)
                    or str(guard.created_at) != str(manifest.not_before_at)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(manifest.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(manifest.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(manifest.model_id)
                    or str(guard.embedding_space_id)
                    != str(manifest.embedding_space_id)
                    or int(guard.dimension) != int(manifest.dimension)
                    or str(guard.dtype) != str(manifest.dtype)
                ):
                    return None
                self._validate_runtime_manifest_in_transaction(
                    connection, manifest, require_ready=True
                )
            except (sqlite3.DatabaseError, ValueError, TypeError):
                return None
            if manifest.not_before_at > as_of or manifest.expires_at <= as_of:
                return None
            return guard, manifest

    def _reconcile_contextual_revisit_runtime_manifests_in_transaction(
        self,
        connection,
    ) -> dict[str, int]:
        """Finish only ready pending rows left by an old/interrupted publisher."""

        rows = connection.execute(
            """
            SELECT manifest.creation_receipt_id
            FROM contextual_revisit_runtime_manifest AS manifest
            JOIN contextual_creation_receipt AS receipt
              ON receipt.id = manifest.creation_receipt_id
            WHERE manifest.state = 'pending_index' AND receipt.status = 'ready'
            ORDER BY manifest.id
            """
        ).fetchall()
        promoted = 0
        rejected = 0
        for index, row in enumerate(rows):
            # A bad old row must not leave behind a just-created V16 contract
            # while reconciliation carries on.  Isolate each retry inside a
            # savepoint; outer index publication state remains intact.
            savepoint = f"revisit_manifest_reconcile_{index}"
            connection.execute(f"SAVEPOINT {savepoint}")
            try:
                result = self._promote_contextual_revisit_runtime_manifest_in_transaction(
                    connection, int(row["creation_receipt_id"])
                )
            except (ValueError, sqlite3.IntegrityError):
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                rejected += 1
                continue
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            if result is not None and result.state == "ready":
                promoted += int(result.promoted)
        return {
            "pending_ready": len(rows),
            "promoted": promoted,
            "rejected": rejected,
        }

    def reconcile_contextual_revisit_runtime_manifests(self) -> dict[str, int]:
        """Recovery entry point after restart or an out-of-band index rebuild.

        It never invents a seed for historical V16 rows; it only promotes a
        previously durable pending V17 manifest once the authoritative receipt
        is already ready.  Rows that fail re-binding remain pending and force
        the caller into normal retrieval.
        """

        with self.db.transaction() as connection:
            return self._reconcile_contextual_revisit_runtime_manifests_in_transaction(
                connection
            )

    def _record_legacy_utility(self, observations) -> dict[str, int]:
        """Retain v1 diagnostics without allowing them to promote an edge.

        The old bounded query-hash cache cannot become V15 evidence.  It is
        kept solely for non-V3 compatibility, with the historical lifecycle
        promotion branch deliberately removed.
        """

        counts = {"updated": 0, "successes": 0, "noops": 0, "harms": 0}
        if not observations:
            return counts
        with self.db.transaction() as connection:
            for observation in observations:
                row = connection.execute(
                    """
                    SELECT * FROM association AS edge
                    WHERE edge.id = ?
                      AND edge.association_mode = 'contextual_recall'
                      -- Legacy observations have only bounded query hashes;
                      -- once a strict V3 origin exists they must not mutate
                      -- its association counters or source-request hash.
                      AND NOT EXISTS (
                          SELECT 1
                          FROM contextual_creation_receipt AS receipt
                          WHERE receipt.association_id = edge.id
                            AND receipt.status = 'ready'
                            AND receipt.verification_status IN ('verified', 'source_bound')
                      )
                    """,
                    (int(observation.association_id),),
                ).fetchone()
                if row is None:
                    continue
                query_hash = str(observation.query_hash or "")
                try:
                    seen_query_hashes = json.loads(
                        str(row["utility_query_hashes"] or "[]")
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    seen_query_hashes = []
                if not isinstance(seen_query_hashes, list):
                    seen_query_hashes = []
                if query_hash and query_hash in {
                    str(value) for value in seen_query_hashes
                }:
                    continue
                weight = float(row["utility_weight"])
                outcome = str(observation.outcome)
                if outcome in {"sufficient", "necessary"}:
                    weight = weight + self.contextual_positive_rate * (1.0 - weight)
                    successes = int(row["utility_successes"]) + 1
                    noops = int(row["utility_noops"])
                    harms = int(row["utility_harms"])
                    counts["successes"] += 1
                elif outcome == "harmful":
                    weight *= self.contextual_harm_multiplier
                    successes = int(row["utility_successes"])
                    noops = int(row["utility_noops"])
                    harms = int(row["utility_harms"]) + 1
                    counts["harms"] += 1
                else:
                    weight *= self.contextual_noop_decay
                    successes = int(row["utility_successes"])
                    noops = int(row["utility_noops"]) + 1
                    harms = int(row["utility_harms"])
                    counts["noops"] += 1
                distinct = int(row["distinct_query_count"])
                if query_hash:
                    distinct += 1
                    seen_query_hashes.append(query_hash)
                    seen_query_hashes = list(dict.fromkeys(seen_query_hashes))[-128:]
                connection.execute(
                    """
                    UPDATE association SET utility_weight = ?, utility_successes = ?,
                        utility_noops = ?, utility_harms = ?, distinct_query_count = ?,
                        last_evaluated_at = ?, source_request_hash = ?,
                        utility_query_hashes = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        max(0.0, min(1.0, weight)), successes, noops, harms,
                        distinct, utc_now(), query_hash,
                        json.dumps(seen_query_hashes, ensure_ascii=False),
                        utc_now(), int(row["id"]),
                    ),
                )
                counts["updated"] += 1
        return counts

    def record_utility(self, observations) -> dict[str, int]:
        """Record V15 ledger rows or isolated legacy diagnostics safely.

        Typed V15 rows are append-only and are the only rows visible to the
        future promotion query.  Legacy observations retain their historical
        API shape but can never update lifecycle state or become ledger proof.
        """

        values = tuple(observations or ())
        ledger_values = tuple(
            value
            for value in values
            if isinstance(value, ContextualUtilityLedgerObservation)
        )
        legacy_values = tuple(
            value
            for value in values
            if not isinstance(value, ContextualUtilityLedgerObservation)
        )
        if any(
            not hasattr(value, "association_id")
            or not hasattr(value, "query_hash")
            or not hasattr(value, "outcome")
            for value in legacy_values
        ):
            raise TypeError("record_utility needs legacy observations or V15 ledger records")
        if ledger_values and legacy_values:
            raise TypeError("legacy and V15 utility observations cannot share one batch")
        ledger_receipts = self.record_contextual_utility_ledger(ledger_values)
        legacy_counts = self._record_legacy_utility(legacy_values)
        inserted = sum(not receipt.idempotent for receipt in ledger_receipts)
        idempotent = len(ledger_receipts) - inserted
        counts = dict(legacy_counts)
        counts["updated"] += inserted
        counts["successes"] += sum(
            receipt.promotion_eligible for receipt in ledger_receipts
        )
        counts["harms"] += sum(
            receipt_id.outcome == "harmful"
            for receipt_id in ledger_values
        )
        counts["noops"] += sum(
            value.outcome == "no_op" for value in ledger_values
        )
        counts["ledger_inserted"] = inserted
        counts["ledger_idempotent"] = idempotent
        counts["legacy_updated"] = legacy_counts["updated"]
        return counts

    def retire(self, association_ids, reason: str = "") -> int:
        ids = sorted({int(value) for value in association_ids})
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE association SET lifecycle_state = 'retired', created_reason = CASE WHEN ? = '' THEN created_reason ELSE created_reason || ' | ' || ? END, updated_at = ? WHERE id IN ({placeholders}) AND association_mode = 'contextual_recall'",
                [reason, reason, utc_now(), *ids],
            )
            return int(cursor.rowcount)

    def prune_expired(self, now: str | None = None) -> int:
        timestamp = now or utc_now()
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE association SET lifecycle_state = 'retired', updated_at = ? WHERE association_mode = 'contextual_recall' AND lifecycle_state = 'probation' AND expires_at IS NOT NULL AND expires_at <= ?",
                (utc_now(), timestamp),
            )
            return int(cursor.rowcount)

    def snapshot(self, association_ids: list[int] | None = None) -> dict[int, dict]:
        """Return detached Association rows suitable for experiment deltas.

        SQLite ``Row`` objects are tied to their cursor result.  Stage 5 needs a
        stable before/after representation that can be serialized and later used
        by a read-only overlay, so snapshots are converted to ordinary dicts.
        """
        with self.db.connection() as connection:
            if association_ids is None:
                rows = connection.execute(
                    "SELECT * FROM association ORDER BY id"
                ).fetchall()
            else:
                unique_ids = sorted({int(value) for value in association_ids})
                if not unique_ids:
                    return {}
                placeholders = ",".join("?" for _ in unique_ids)
                rows = connection.execute(
                    f"SELECT * FROM association WHERE id IN ({placeholders}) "
                    "ORDER BY id",
                    unique_ids,
                ).fetchall()
        return {int(row["id"]): dict(row) for row in rows}

    def delete(self, association_id: int) -> bool:
        self._concept_reach_cache.clear()
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM association WHERE id = ?", (association_id,)
            )
            return cursor.rowcount > 0

    def restore_rows(self, rows: dict[int, dict]) -> list[int]:
        """Restore reinforced rows to exact pre-query values.

        Query-time growth may be provisionally committed so the normal graph
        traverser can measure its utility. Rows that did not reach the final
        answer path are restored before the query returns.
        """
        if not rows:
            return []
        mutable_columns = (
            "relation_text",
            "weight",
            "confidence",
            "generation",
            "evidence_count",
            "claim_level",
            "audit_status",
            "evidence_json",
            "audit_json",
            "created_reason",
            "last_used",
            "use_count",
            "association_mode",
            "context_cue_id",
            "need_cue_id",
            "utility_weight",
            "utility_successes",
            "utility_noops",
            "utility_harms",
            "distinct_query_count",
            "lifecycle_state",
            "expires_at",
            "last_evaluated_at",
            "source_request_hash",
            "utility_query_hashes",
            "updated_at",
        )
        restored: list[int] = []
        self._concept_reach_cache.clear()
        with self.db.transaction() as connection:
            for association_id, row in rows.items():
                cursor = connection.execute(
                    f"""
                    UPDATE association
                    SET {', '.join(f'{column} = ?' for column in mutable_columns)}
                    WHERE id = ?
                    """,
                    [
                        *[
                            row[column]
                            if column in row
                            else {
                                "association_mode": "semantic",
                                "utility_weight": 0.0,
                                "utility_successes": 0,
                                "utility_noops": 0,
                                "utility_harms": 0,
                                "distinct_query_count": 0,
                                "lifecycle_state": "active",
                                "source_request_hash": "",
                                "utility_query_hashes": "[]",
                            }.get(column)
                        for column in mutable_columns],
                        int(association_id),
                    ],
                )
                if cursor.rowcount:
                    restored.append(int(association_id))
        return restored

    def mark_used(
        self,
        association_ids: list[int],
        *,
        require_live: Callable[[], None] | None = None,
    ) -> None:
        if not association_ids:
            return
        if require_live is not None:
            require_live()
        with self.db.transaction(before_commit=require_live) as connection:
            self._mark_used_in_transaction(connection, association_ids)

    def list_temporal(self, timeline_scope: str):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT a.*
                FROM association a
                JOIN episode e1 ON a.from_type = 'episode' AND e1.id = a.from_id
                JOIN episode e2 ON a.to_type = 'episode' AND e2.id = a.to_id
                WHERE a.relation_type = 'temporal'
                  AND e1.timeline_scope = ? AND e2.timeline_scope = ?
                ORDER BY a.weight DESC
                """,
                (timeline_scope, timeline_scope),
            ).fetchall()

    def stats(self) -> dict[str, int]:
        with self.db.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS edges,
                       COALESCE(SUM(CASE WHEN polarity < 0 THEN 1 ELSE 0 END), 0) AS negative,
                       COALESCE(SUM(CASE WHEN association_mode = 'contextual_recall' THEN 1 ELSE 0 END), 0) AS contextual,
                       COALESCE(SUM(CASE WHEN association_mode = 'contextual_recall' AND lifecycle_state = 'probation' THEN 1 ELSE 0 END), 0) AS contextual_probation,
                       COALESCE(SUM(CASE WHEN association_mode = 'contextual_recall' AND lifecycle_state = 'active' THEN 1 ELSE 0 END), 0) AS contextual_active,
                       COALESCE(SUM(CASE WHEN association_mode = 'contextual_recall' AND lifecycle_state = 'retired' THEN 1 ELSE 0 END), 0) AS contextual_retired
                FROM association
                """
            ).fetchone()
            return {
                "edges": int(row["edges"]),
                "negative": int(row["negative"]),
                "contextual": int(row["contextual"]),
                "contextual_probation": int(row["contextual_probation"]),
                "contextual_active": int(row["contextual_active"]),
                "contextual_retired": int(row["contextual_retired"]),
            }
