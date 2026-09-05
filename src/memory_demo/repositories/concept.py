from __future__ import annotations

from typing import Iterable
import unicodedata

from memory_demo.database import Database, utc_now
from memory_demo.types import ConceptDraft


def normalize_alias(value: str) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )
    if any(
        "\u3400" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in normalized
    ):
        normalized = normalized.replace(" ", "")
    return normalized


class ConceptRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert(self, draft: ConceptDraft, embedding_blob: bytes) -> int:
        now = utc_now()
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO concept(
                    canonical_name, description, embedding_text, embedding,
                    confidence, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.canonical_name,
                    draft.description,
                    draft.embedding_text,
                    embedding_blob,
                    draft.confidence,
                    now,
                    now,
                ),
            )
            concept_id = int(cursor.lastrowid)
            aliases = [(draft.canonical_name, "canonical"), *draft.aliases]
            for alias, language in aliases:
                if not alias:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO concept_alias(
                        concept_id, alias, language, normalized_alias,
                        confidence, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        concept_id,
                        alias,
                        language,
                        normalize_alias(alias),
                        draft.confidence,
                        now,
                    ),
                )
            return concept_id

    def update(self, concept_id: int, draft: ConceptDraft, embedding_blob: bytes) -> None:
        now = utc_now()
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE concept SET canonical_name = ?, description = ?,
                    embedding_text = ?, embedding = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    draft.canonical_name,
                    draft.description,
                    draft.embedding_text,
                    embedding_blob,
                    draft.confidence,
                    now,
                    concept_id,
                ),
            )
            aliases = [(draft.canonical_name, "canonical"), *draft.aliases]
            for alias, language in aliases:
                if not alias:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO concept_alias(
                        concept_id, alias, language, normalized_alias,
                        confidence, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        concept_id,
                        alias,
                        language,
                        normalize_alias(alias),
                        draft.confidence,
                        now,
                    ),
                )

    def add_aliases(
        self,
        concept_id: int,
        aliases: list[tuple[str, str]],
        confidence: float,
    ) -> int:
        now = utc_now()
        inserted = 0
        with self.db.transaction() as connection:
            for alias, language in aliases:
                if not alias:
                    continue
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO concept_alias(
                        concept_id, alias, language, normalized_alias,
                        confidence, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        concept_id,
                        alias,
                        language,
                        normalize_alias(alias),
                        confidence,
                        now,
                    ),
                )
                inserted += max(0, cursor.rowcount)
        return inserted

    def list_aliases(self, concept_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT alias, language, confidence, created_at
                FROM concept_alias WHERE concept_id = ? ORDER BY id
                """,
                (concept_id,),
            ).fetchall()

    def merge(self, concept_id: int, canonical_concept_id: int) -> None:
        if concept_id == canonical_concept_id:
            raise ValueError("cannot merge a concept into itself")
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE concept SET status = 'merged', canonical_concept_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (canonical_concept_id, utc_now(), concept_id),
            )

    def unmerge(self, concept_id: int) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE concept SET status = 'active', canonical_concept_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), concept_id),
            )

    def find_by_alias(self, alias: str):
        normalized = normalize_alias(alias)
        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT c.*, a.alias, a.language
                FROM concept_alias a
                JOIN concept c ON c.id = a.concept_id
                WHERE a.normalized_alias = ?
                ORDER BY c.status = 'active' DESC, c.confidence DESC
                """,
                (normalized,),
            ).fetchall()

    def resolution_snapshot(self):
        """Load the compact alias ownership state used by one import writer."""

        with self.db.connection() as connection:
            return connection.execute(
                """
                SELECT c.id, c.canonical_name, c.description, c.embedding_text,
                       c.status, c.canonical_concept_id, a.normalized_alias
                FROM concept c
                LEFT JOIN concept_alias a ON a.concept_id = c.id
                ORDER BY c.id, a.id
                """
            ).fetchall()

    def get(self, concept_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                "SELECT * FROM concept WHERE id = ?", (concept_id,)
            ).fetchone()

    def get_many(self, ids: Iterable[int]):
        id_list = list(dict.fromkeys(int(value) for value in ids))
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        with self.db.connection() as connection:
            return connection.execute(
                f"SELECT * FROM concept WHERE id IN ({placeholders})", id_list
            ).fetchall()

    def count(self) -> int:
        with self.db.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM concept").fetchone()[0])

    def iter_embeddings(self, batch_size: int = 2_000):
        with self.db.connection() as connection:
            cursor = connection.execute(
                "SELECT id, embedding FROM concept WHERE status = 'active' ORDER BY id"
            )
            while rows := cursor.fetchmany(batch_size):
                yield rows
