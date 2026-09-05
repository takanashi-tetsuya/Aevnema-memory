from __future__ import annotations

from typing import Iterable

from memory_demo.database import Database


class SourceRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert(self, raw_text: str) -> int:
        if not raw_text.strip():
            raise ValueError("source raw_text cannot be empty")
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO source(raw_text) VALUES(?)", (raw_text,)
            )
            return int(cursor.lastrowid)

    def get(self, source_id: int):
        with self.db.connection() as connection:
            return connection.execute(
                "SELECT id, raw_text FROM source WHERE id = ?", (source_id,)
            ).fetchone()

    def get_many(self, source_ids: Iterable[int]):
        id_list = list(dict.fromkeys(int(value) for value in source_ids))
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        with self.db.connection() as connection:
            return connection.execute(
                f"SELECT id, raw_text FROM source WHERE id IN ({placeholders})",
                id_list,
            ).fetchall()

    def count(self) -> int:
        with self.db.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM source").fetchone()[0])
