from __future__ import annotations

import json
from datetime import datetime, timezone

from memory_demo.config import WeightConfig
from memory_demo.database import Database, utc_now
from memory_demo.types import AssociationDraft, NodeType


class TemporalCycleError(ValueError):
    """Raised when a positive temporal edge would make chronology cyclic."""

    def __init__(self, earlier_id: int, later_id: int):
        self.earlier_id = int(earlier_id)
        self.later_id = int(later_id)
        super().__init__(
            f"temporal edge {self.earlier_id} -> {self.later_id} would close a cycle: "
            f"{self.later_id} already reaches {self.earlier_id}"
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

    def upsert(self, draft: AssociationDraft) -> int:
        if draft.from_type == draft.to_type and draft.from_id == draft.to_id:
            raise ValueError("self associations are not allowed")
        if isinstance(draft.generation, bool):
            raise ValueError("generation must be a non-negative integer")
        generation = int(draft.generation)
        if generation < 0 or generation != draft.generation:
            raise ValueError("generation must be a non-negative integer")
        # A new topology edge can change both the directly touched Concept and
        # one-hop neighbors. Clearing this small derived cache is safer than
        # attempting partial invalidation.
        self._concept_reach_cache.clear()
        now = utc_now()
        with self.db.transaction() as connection:
            if not self._node_exists(connection, draft.from_type, draft.from_id):
                raise ValueError("from node does not exist")
            if not self._node_exists(connection, draft.to_type, draft.to_id):
                raise ValueError("to node does not exist")
            existing = self._find_equivalent(connection, draft)
            if existing is None:
                endpoints = self._positive_temporal_endpoints(draft)
                if endpoints is not None:
                    earlier_id, later_id = endpoints
                    if self._temporal_reaches(
                        connection, later_id, earlier_id
                    ):
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
                        now,
                        now,
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
                    now,
                    existing["id"],
                ),
            )
            return int(existing["id"])

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

    def get_or_create_cue_prototype(
        self,
        *,
        domain: str,
        cue_kind: str,
        model_id: str,
        dimension: int,
        vector,
        text_hash: str,
        display_text: str = "",
        source_request_hash: str = "",
    ) -> int:
        domain = str(domain or "").strip()
        cue_kind = str(cue_kind or "").strip().casefold()
        model_id = str(model_id or "").strip()
        if not domain or cue_kind not in {"context", "need"} or not model_id:
            raise ValueError("domain, cue_kind and model_id are required")
        if not text_hash:
            raise ValueError("text_hash is required")
        blob = self._cue_blob(vector, int(dimension))
        now = utc_now()
        with self.db.transaction() as connection:
            row = connection.execute(
                """
                SELECT id, dimension, dtype, vector_blob
                FROM association_cue_prototype
                WHERE domain = ? AND cue_kind = ? AND model_id = ? AND text_hash = ?
                """,
                (domain, cue_kind, model_id, text_hash),
            ).fetchone()
            if row is not None:
                if int(row["dimension"]) != int(dimension) or str(row["dtype"]) != "float32":
                    raise ValueError("existing cue prototype has incompatible shape")
                if bytes(row["vector_blob"]) != blob:
                    raise ValueError("cue text hash already has a different vector")
                return int(row["id"])
            cursor = connection.execute(
                """
                INSERT INTO association_cue_prototype(
                    domain, cue_kind, model_id, dimension, dtype, vector_blob,
                    text_hash, display_text, source_request_hash, created_at
                ) VALUES(?, ?, ?, ?, 'float32', ?, ?, ?, ?, ?)
                """,
                (
                    domain,
                    cue_kind,
                    model_id,
                    int(dimension),
                    blob,
                    text_hash,
                    str(display_text or "")[:500],
                    str(source_request_hash or ""),
                    now,
                ),
            )
            return int(cursor.lastrowid)

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

    # Explicit names used by the contextual-association maintenance API.
    insert_cue_prototype = get_or_create_cue_prototype

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
        if candidate.anchor_type not in {"episode", "concept"}:
            raise ValueError("invalid contextual anchor type")
        if int(candidate.target_episode_id) <= 0:
            raise ValueError("contextual target must be an Episode")
        if int(candidate.anchor_id) == int(candidate.target_episode_id) and candidate.anchor_type == "episode":
            raise ValueError("contextual self-edge is not allowed")
        utility_weight = max(0.0, min(1.0, float(utility_weight)))
        expires_at = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + max(0, int(probation_ttl)),
            timezone.utc,
        ).isoformat()
        with self.db.transaction() as connection:
            context = connection.execute(
                "SELECT domain FROM association_cue_prototype WHERE id = ? AND cue_kind = 'context'",
                (int(context_cue_id),),
            ).fetchone()
            need = connection.execute(
                "SELECT domain FROM association_cue_prototype WHERE id = ? AND cue_kind = 'need'",
                (int(need_cue_id),),
            ).fetchone()
            if context is None or need is None or context["domain"] != need["domain"]:
                raise ValueError("context and need cues must exist in the same domain")
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
            if connection.execute(
                "SELECT 1 FROM episode WHERE id = ?", (int(candidate.target_episode_id),)
            ).fetchone() is None:
                raise ValueError("contextual target Episode does not exist")
            target_row = connection.execute(
                "SELECT generation FROM episode WHERE id = ?",
                (int(candidate.target_episode_id),),
            ).fetchone()
            if target_row is None or int(target_row["generation"]) != 0:
                raise ValueError("contextual target must be a generation-0 Episode")
            existing = connection.execute(
                """
                SELECT id FROM association
                WHERE from_type = ? AND from_id = ? AND to_type = 'episode'
                  AND to_id = ? AND relation_type = 'retrieval'
                  AND relation_key = 'contextual_recall' AND polarity = 1
                  AND association_mode = 'contextual_recall'
                  AND context_cue_id = ? AND need_cue_id = ?
                """,
                (
                    candidate.anchor_type,
                    int(candidate.anchor_id),
                    int(candidate.target_episode_id),
                    int(context_cue_id),
                    int(need_cue_id),
                ),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
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
            return int(cursor.lastrowid)

    create_contextual_association = create_contextual

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
        limit: int = 1000,
    ) -> list:
        contexts = sorted({int(value) for value in context_ids})
        needs = sorted({int(value) for value in need_ids})
        if not contexts or not needs:
            return []
        context_ph = ",".join("?" for _ in contexts)
        need_ph = ",".join("?" for _ in needs)
        params: list[object] = [*contexts, *needs]
        domain_clause = ""
        if domain:
            domain_clause = " AND cp.domain = ?"
            params.append(str(domain))
        params.append(max(1, int(limit)))
        with self.db.connection() as connection:
            return connection.execute(
                f"""
                SELECT a.*, cp.domain AS cue_domain
                FROM association a
                JOIN association_cue_prototype cp ON cp.id = a.context_cue_id
                JOIN association_cue_prototype np ON np.id = a.need_cue_id
                WHERE a.association_mode = 'contextual_recall'
                  AND cp.cue_kind = 'context'
                  AND np.cue_kind = 'need'
                  AND a.lifecycle_state IN ('probation', 'active')
                  AND a.context_cue_id IN ({context_ph})
                  AND a.need_cue_id IN ({need_ph})
                  AND cp.domain = np.domain{domain_clause}
                  AND (a.expires_at IS NULL OR a.expires_at > ?)
                ORDER BY a.utility_weight DESC, a.id
                LIMIT ?
                """,
                [*params[:-1], utc_now(), params[-1]],
            ).fetchall()

    def record_utility(self, observations) -> dict[str, int]:
        """Apply a batch of local observations; duplicate query hashes are ignored."""
        counts = {"updated": 0, "successes": 0, "noops": 0, "harms": 0}
        if not observations:
            return counts
        with self.db.transaction() as connection:
            for observation in observations:
                row = connection.execute(
                    "SELECT * FROM association WHERE id = ? AND association_mode = 'contextual_recall'",
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
                    # A repeated query must not self-reinforce the edge.
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
                lifecycle = str(row["lifecycle_state"])
                distinct = int(row["distinct_query_count"])
                if query_hash:
                    distinct += 1
                    seen_query_hashes.append(query_hash)
                    seen_query_hashes = list(dict.fromkeys(seen_query_hashes))[-128:]
                if (
                    self.contextual_promotion_enabled
                    and
                    lifecycle == "probation"
                    and distinct >= self.contextual_min_distinct_successes
                    and successes >= self.contextual_min_distinct_successes
                    and harms == 0
                ):
                    lifecycle = "active"
                connection.execute(
                    """
                    UPDATE association SET utility_weight = ?, utility_successes = ?,
                        utility_noops = ?, utility_harms = ?, distinct_query_count = ?,
                        lifecycle_state = ?, last_evaluated_at = ?, source_request_hash = ?,
                        utility_query_hashes = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        max(0.0, min(1.0, weight)), successes, noops, harms,
                        distinct, lifecycle, utc_now(), query_hash,
                        json.dumps(seen_query_hashes, ensure_ascii=False),
                        utc_now(), int(row["id"])
                    ),
                )
                counts["updated"] += 1
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

    def mark_used(self, association_ids: list[int]) -> None:
        if not association_ids:
            return
        placeholders = ",".join("?" for _ in association_ids)
        with self.db.transaction() as connection:
            connection.execute(
                f"""
                UPDATE association
                SET use_count = use_count + 1, last_used = ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [utc_now(), utc_now(), *association_ids],
            )

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
