from __future__ import annotations

import json
from typing import Iterable

from memory_demo.database import Database, utc_now
from memory_demo.types import EpisodeDraft


class EpisodeRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert(
        self,
        source_id: int,
        source_key: str,
        segment_index: int,
        draft: EpisodeDraft,
        embedding_blob: bytes,
        extraction_run_id: int | None = None,
    ) -> int:
        now = utc_now()
        evidence_quotes_json = json.dumps(draft.evidence_quotes, ensure_ascii=False)
        evidence_spans_json = json.dumps(
            [list(span) for span in draft.evidence_spans],
            ensure_ascii=False,
        )
        evidence_basis = (
            "reasoning_view_nonempty_lines_v1"
            if draft.evidence_quotes and draft.evidence_spans
            else "source_id_v1"
        )
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episode(
                    source_id, source_key, segment_index, text,
                    participants_json, event_type, location_text,
                    story_time_text, timeline_scope, confidence,
                    evidence_origin, epistemic_status, generation, epistemic_note,
                    evidence_quotes_json, evidence_spans_json, evidence_basis,
                    embedding,
                    extraction_run_id, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    source_key,
                    segment_index,
                    draft.text,
                    json.dumps(draft.participants, ensure_ascii=False),
                    draft.event_type,
                    draft.location_text,
                    draft.story_time_text,
                    draft.timeline_scope,
                    draft.confidence,
                    draft.evidence_origin,
                    draft.epistemic_status,
                    draft.generation,
                    draft.epistemic_note,
                    evidence_quotes_json,
                    evidence_spans_json,
                    evidence_basis,
                    embedding_blob,
                    extraction_run_id,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def update_draft(
        self, episode_id: int, draft: EpisodeDraft, embedding_blob: bytes
    ) -> None:
        evidence_quotes_json = json.dumps(draft.evidence_quotes, ensure_ascii=False)
        evidence_spans_json = json.dumps(
            [list(span) for span in draft.evidence_spans],
            ensure_ascii=False,
        )
        evidence_basis = (
            "reasoning_view_nonempty_lines_v1"
            if draft.evidence_quotes and draft.evidence_spans
            else "source_id_v1"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE episode SET
                    text = ?, participants_json = ?, event_type = ?,
                    location_text = ?, story_time_text = ?, timeline_scope = ?,
                    confidence = ?, evidence_origin = ?, epistemic_status = ?,
                    generation = ?, epistemic_note = ?,
                    evidence_quotes_json = ?, evidence_spans_json = ?,
                    evidence_basis = ?, embedding = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    draft.text,
                    json.dumps(draft.participants, ensure_ascii=False),
                    draft.event_type,
                    draft.location_text,
                    draft.story_time_text,
                    draft.timeline_scope,
                    draft.confidence,
                    draft.evidence_origin,
                    draft.epistemic_status,
                    draft.generation,
                    draft.epistemic_note,
                    evidence_quotes_json,
                    evidence_spans_json,
                    evidence_basis,
                    embedding_blob,
                    utc_now(),
                    episode_id,
                ),
            )

    def set_story_order(
        self, episode_id: int, story_order: float | None, timeline_scope: str
    ) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE episode
                SET story_order = ?, timeline_scope = ?, updated_at = ?
                WHERE id = ?
                """,
                (story_order, timeline_scope, utc_now(), episode_id),
            )

    def get(self, episode_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                "SELECT * FROM episode WHERE id = ?", (episode_id,)
            ).fetchone()

    def get_many(self, ids: Iterable[int]):
        id_list = list(dict.fromkeys(int(value) for value in ids))
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        with self.db.connection() as connection:
            return connection.execute(
                f"SELECT * FROM episode WHERE id IN ({placeholders})", id_list
            ).fetchall()

    def count(self) -> int:
        with self.db.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM episode").fetchone()[0])

    def iter_embeddings(self, batch_size: int = 2_000):
        with self.db.connection() as connection:
            cursor = connection.execute("SELECT id, embedding FROM episode ORDER BY id")
            while rows := cursor.fetchmany(batch_size):
                yield rows

    def list_by_source_key(self, source_key: str, extraction_run_id: int | None = None):
        with self.db.connection() as connection:
            if extraction_run_id is not None:
                return connection.execute(
                    """
                    SELECT * FROM episode
                    WHERE source_key = ? AND extraction_run_id = ?
                    ORDER BY segment_index, id
                    """,
                    (source_key, extraction_run_id),
                ).fetchall()
            return connection.execute(
                """
                SELECT * FROM episode
                WHERE source_key = ?
                ORDER BY segment_index, id
                """,
                (source_key,),
            ).fetchall()

    def list_by_source_id(self, source_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT * FROM episode
                WHERE source_id = ? ORDER BY id
                """,
                (source_id,),
            ).fetchall()

    def list_source_ids(self) -> list[int]:
        with self.db.connection() as connection:
            return [
                int(row["source_id"])
                for row in connection.execute(
                    "SELECT DISTINCT source_id FROM episode ORDER BY source_id"
                )
            ]

    def list_ids_by_source_ids(self, source_ids: Iterable[int]):
        source_id_list = list(dict.fromkeys(int(value) for value in source_ids))
        if not source_id_list:
            return []
        placeholders = ",".join("?" for _ in source_id_list)
        with self.db.connection() as connection:
            return connection.execute(
                f"""
                SELECT id, source_id
                FROM episode
                WHERE source_id IN ({placeholders})
                ORDER BY source_id, id
                """,
                source_id_list,
            ).fetchall()

    def list_timeline(self, timeline_scope: str):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT * FROM episode
                WHERE timeline_scope = ?
                ORDER BY story_order IS NULL, story_order, segment_index, id
                """,
                (timeline_scope,),
            ).fetchall()

    def delete(self, episode_id: int) -> bool:
        with self.db.transaction() as connection:
            connection.execute(
                """
                DELETE FROM association
                WHERE (from_type = 'episode' AND from_id = ?)
                   OR (to_type = 'episode' AND to_id = ?)
                """,
                (episode_id, episode_id),
            )
            cursor = connection.execute(
                "DELETE FROM episode WHERE id = ?", (episode_id,)
            )
            return cursor.rowcount > 0
