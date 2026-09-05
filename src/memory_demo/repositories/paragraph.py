from __future__ import annotations

from typing import Iterable

from memory_demo.database import Database, utc_now
from memory_demo.types import ParagraphDraft


class ParagraphRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert_many(
        self,
        source_id: int,
        source_key: str,
        segment_index: int,
        drafts: list[ParagraphDraft],
        embedding_blobs: list[bytes],
    ) -> list[int]:
        if len(drafts) != len(embedding_blobs):
            raise ValueError("paragraph drafts and embeddings must have equal length")
        ids: list[int] = []
        with self.db.transaction() as connection:
            for draft, embedding_blob in zip(drafts, embedding_blobs, strict=True):
                cursor = connection.execute(
                    """
                    INSERT INTO paragraph(
                        source_id, source_key, segment_index, paragraph_index,
                        text, embedding, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        source_key,
                        segment_index,
                        draft.paragraph_index,
                        draft.text,
                        embedding_blob,
                        utc_now(),
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def count(self) -> int:
        with self.db.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM paragraph").fetchone()[0])

    def get_many(self, ids: Iterable[int]):
        id_list = list(dict.fromkeys(int(value) for value in ids))
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        with self.db.connection() as connection:
            return connection.execute(
                f"SELECT * FROM paragraph WHERE id IN ({placeholders})",
                id_list,
            ).fetchall()

    def list_by_source(self, source_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT * FROM paragraph
                WHERE source_id = ? ORDER BY paragraph_index, id
                """,
                (source_id,),
            ).fetchall()

    def iter_embeddings(self, batch_size: int = 2_000):
        with self.db.connection() as connection:
            cursor = connection.execute(
                "SELECT id, embedding FROM paragraph ORDER BY id"
            )
            while rows := cursor.fetchmany(batch_size):
                yield rows

    def sources_without_paragraphs(self):
        """Return importable Sources whose source_key can be recovered from Episode."""
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT
                    s.id,
                    s.raw_text,
                    MIN(e.source_key) AS source_key,
                    MIN(e.segment_index) AS segment_index
                FROM source s
                JOIN episode e ON e.source_id = s.id
                LEFT JOIN paragraph p ON p.source_id = s.id
                WHERE p.id IS NULL
                GROUP BY s.id, s.raw_text
                ORDER BY s.id
                """
            ).fetchall()
