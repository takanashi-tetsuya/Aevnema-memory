from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory_demo.database import Database, utc_now
from memory_demo.event_log import redact_for_export


class TerminalRunReplacementError(RuntimeError):
    """A terminal file run cannot be proven safe to replace."""


class ExtractionRepository:
    def __init__(self, db: Database):
        self.db = db

    def start_run(
        self,
        config_snapshot: dict[str, Any],
        prompt_versions: dict[str, str],
        log_path: str | Path,
    ) -> int:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO extraction_run(
                    status, config_snapshot, prompt_versions, log_path, started_at
                ) VALUES('running', ?, ?, ?, ?)
                """,
                (
                    json.dumps(redact_for_export(config_snapshot), ensure_ascii=False),
                    json.dumps(prompt_versions, ensure_ascii=False),
                    str(log_path),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, summary: dict[str, Any]) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE extraction_run
                SET status = ?, finished_at = ?, summary_json = ?
                WHERE id = ?
                """,
                (status, utc_now(), json.dumps(summary, ensure_ascii=False), run_id),
            )

    def start_task(
        self,
        run_id: int,
        source_key: str,
        segment_index: int,
        stage: str,
        model: str,
        prompt_version: str,
        source_id: int | None = None,
    ) -> int:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO extraction_task(
                    run_id, source_id, source_key, segment_index, stage,
                    status, model, prompt_version, started_at
                ) VALUES(?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    source_id,
                    source_key,
                    segment_index,
                    stage,
                    model,
                    prompt_version,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def set_source(self, task_id: int, source_id: int) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE extraction_task SET source_id = ? WHERE id = ?",
                (source_id, task_id),
            )

    def retry(self, task_id: int, error_summary: str) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE extraction_task
                SET retry_count = retry_count + 1, error_summary = ?
                WHERE id = ?
                """,
                (error_summary, task_id),
            )

    def finish_task(self, task_id: int, status: str, error_summary: str = "") -> None:
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE extraction_task
                SET status = ?, error_summary = ?, finished_at = ?
                WHERE id = ?
                """,
                (status, error_summary, utc_now(), task_id),
            )

    def interrupt_running_tasks(self, run_id: int, error_summary: str) -> int:
        """Close tasks left running when an import aborts outside normal handling."""

        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE extraction_task
                SET status = 'interrupted', error_summary = ?, finished_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (error_summary, utc_now(), run_id),
            )
            return int(cursor.rowcount)

    def discard_incomplete_run(self, run_id: int) -> dict[str, int]:
        """Remove only data owned by an interrupted, retryable file run.

        Completed and partial runs are never passed here.  A transport abort
        may have persisted some earlier segments of the same logical file; a
        full retry must first remove exactly those run-owned artifacts so it
        cannot duplicate evidence rows.
        """

        with self.db.transaction() as connection:
            return self._discard_run_artifacts(connection, run_id)

    @staticmethod
    def _run_source_keys(connection, run_id: int) -> set[str]:
        return {
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT source_key FROM extraction_task WHERE run_id = ?
                UNION
                SELECT DISTINCT source_key FROM episode WHERE extraction_run_id = ?
                """,
                (run_id, run_id),
            )
            if str(row[0]).strip()
        }

    @staticmethod
    def _run_source_ids(connection, run_id: int) -> list[int]:
        return sorted(
            {
                int(row[0])
                for row in connection.execute(
                    """
                    SELECT source_id FROM extraction_task
                    WHERE run_id = ? AND source_id IS NOT NULL
                    UNION
                    SELECT source_id FROM episode WHERE extraction_run_id = ?
                    """,
                    (run_id, run_id),
                )
            }
        )

    def _discard_run_artifacts(self, connection, run_id: int) -> dict[str, int]:
        """Delete exactly one run's owned artifacts inside an active transaction."""

        source_ids = self._run_source_ids(connection, run_id)
        episode_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM episode WHERE extraction_run_id = ?", (run_id,)
            )
        ]
        deleted_associations = 0
        if episode_ids:
            marks = ",".join("?" for _ in episode_ids)
            deleted_associations = int(
                connection.execute(
                    f"""
                    DELETE FROM association
                    WHERE (from_type = 'episode' AND from_id IN ({marks}))
                       OR (to_type = 'episode' AND to_id IN ({marks}))
                    """,
                    [*episode_ids, *episode_ids],
                ).rowcount
            )
        deleted_episodes = int(
            connection.execute(
                "DELETE FROM episode WHERE extraction_run_id = ?", (run_id,)
            ).rowcount
        )
        deleted_tasks = int(
            connection.execute(
                "DELETE FROM extraction_task WHERE run_id = ?", (run_id,)
            ).rowcount
        )
        deleted_sources = 0
        if source_ids:
            marks = ",".join("?" for _ in source_ids)
            deleted_sources = int(
                connection.execute(
                    f"""
                    DELETE FROM source
                    WHERE id IN ({marks})
                      AND NOT EXISTS (
                          SELECT 1 FROM episode WHERE episode.source_id = source.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM paragraph WHERE paragraph.source_id = source.id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM extraction_task
                          WHERE extraction_task.source_id = source.id
                      )
                    """,
                    source_ids,
                ).rowcount
            )
        deleted_run = int(
            connection.execute("DELETE FROM extraction_run WHERE id = ?", (run_id,)).rowcount
        )
        foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_issues:
            raise RuntimeError(
                "foreign key check failed after run cleanup: " f"{foreign_key_issues}"
            )
        return {
            "run_id": run_id,
            "deleted_associations": deleted_associations,
            "deleted_episodes": deleted_episodes,
            "deleted_tasks": deleted_tasks,
            "deleted_sources": deleted_sources,
            "deleted_run": deleted_run,
        }

    def replace_terminal_source_run(
        self, run_id: int, source_key: str
    ) -> dict[str, int | str]:
        """Audited opt-in replacement for one terminal, single-file import run.

        Unlike interruption recovery, this can delete a completed or partial
        run only after proving that every task and Episode in it belongs to the
        requested source key and that its Source rows are not shared.  This is
        used for an intentional source-adapter correction, not normal retry.
        """

        with self.db.transaction() as connection:
            run = connection.execute(
                "SELECT status FROM extraction_run WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise TerminalRunReplacementError(f"extraction run {run_id} does not exist")
            status = str(run["status"])
            if status not in {"completed", "partial"}:
                raise TerminalRunReplacementError(
                    f"extraction run {run_id} has non-replaceable status {status!r}"
                )

            source_keys = self._run_source_keys(connection, run_id)
            if source_keys != {source_key}:
                raise TerminalRunReplacementError(
                    "terminal run source-key ownership is ambiguous: "
                    f"expected {source_key!r}, found {sorted(source_keys)!r}"
                )

            source_ids = self._run_source_ids(connection, run_id)
            if source_ids:
                marks = ",".join("?" for _ in source_ids)
                foreign_task_count = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) FROM extraction_task
                        WHERE source_id IN ({marks}) AND run_id != ?
                        """,
                        [*source_ids, run_id],
                    ).fetchone()[0]
                )
                foreign_episode_count = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) FROM episode
                        WHERE source_id IN ({marks}) AND extraction_run_id != ?
                        """,
                        [*source_ids, run_id],
                    ).fetchone()[0]
                )
                paragraph_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM paragraph WHERE source_id IN ({marks})",
                        source_ids,
                    ).fetchone()[0]
                )
                if foreign_task_count or foreign_episode_count or paragraph_count:
                    raise TerminalRunReplacementError(
                        "terminal run has shared or paragraph-owned Sources; "
                        "refusing replacement"
                    )

            deleted = self._discard_run_artifacts(connection, run_id)
        return {"source_key": source_key, "previous_status": status, **deleted}
