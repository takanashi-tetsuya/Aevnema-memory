from __future__ import annotations

"""Recover a file-level import that was stopped before graceful pause existed.

The normal pause path in ``import_corpus.py`` drains in-flight files and never
needs this utility.  This utility is deliberately narrow by default: it only
cleans runs whose ledger records are still ``running``.  ``--include-failed``
is an explicit, auditable opt-in for retrying failed files after a transient
transport outage. Completed and partial files are always left intact.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ledger(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError(f"invalid progress ledger: {path}")
    return value


def save_ledger(path: Path, ledger: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def marks(values: list[int]) -> str:
    return ",".join("?" for _ in values)


def run_source_keys(connection: sqlite3.Connection, run_id: int) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT source_key
            FROM extraction_task
            WHERE run_id = ?
            ORDER BY source_key
            """,
            (run_id,),
        )
    ]


def run_source_ids(connection: sqlite3.Connection, run_id: int) -> list[int]:
    rows = connection.execute(
        """
        SELECT source_id FROM extraction_task
        WHERE run_id = ? AND source_id IS NOT NULL
        UNION
        SELECT source_id FROM episode WHERE extraction_run_id = ?
        """,
        (run_id, run_id),
    ).fetchall()
    return sorted({int(row[0]) for row in rows})


def run_episode_ids(connection: sqlite3.Connection, run_id: int) -> list[int]:
    return [
        int(row[0])
        for row in connection.execute(
            "SELECT id FROM episode WHERE extraction_run_id = ? ORDER BY id",
            (run_id,),
        )
    ]


def describe_run(connection: sqlite3.Connection, run_id: int) -> dict:
    run = connection.execute(
        "SELECT id, status, started_at, finished_at FROM extraction_run WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"extraction run does not exist: {run_id}")
    source_ids = run_source_ids(connection, run_id)
    episode_ids = run_episode_ids(connection, run_id)
    association_count = 0
    if episode_ids:
        placeholders = marks(episode_ids)
        association_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM association
                WHERE (from_type = 'episode' AND from_id IN ({placeholders}))
                   OR (to_type = 'episode' AND to_id IN ({placeholders}))
                """,
                [*episode_ids, *episode_ids],
            ).fetchone()[0]
        )
    return {
        "run": dict(run),
        "source_keys": run_source_keys(connection, run_id),
        "source_ids": source_ids,
        "episode_ids": episode_ids,
        "association_count": association_count,
    }


def delete_run_artifacts(connection: sqlite3.Connection, run_id: int) -> dict:
    """Remove only rows produced by one interrupted extraction run."""
    source_ids = run_source_ids(connection, run_id)
    episode_ids = run_episode_ids(connection, run_id)
    deleted_associations = 0
    if episode_ids:
        placeholders = marks(episode_ids)
        cursor = connection.execute(
            f"""
            DELETE FROM association
            WHERE (from_type = 'episode' AND from_id IN ({placeholders}))
               OR (to_type = 'episode' AND to_id IN ({placeholders}))
            """,
            [*episode_ids, *episode_ids],
        )
        deleted_associations = int(cursor.rowcount)
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
        placeholders = marks(source_ids)
        cursor = connection.execute(
            f"""
            DELETE FROM source
            WHERE id IN ({placeholders})
              AND NOT EXISTS (SELECT 1 FROM episode WHERE episode.source_id = source.id)
              AND NOT EXISTS (SELECT 1 FROM paragraph WHERE paragraph.source_id = source.id)
              AND NOT EXISTS (
                  SELECT 1 FROM extraction_task
                  WHERE extraction_task.source_id = source.id
              )
            """,
            source_ids,
        )
        deleted_sources = int(cursor.rowcount)
    deleted_run = int(
        connection.execute("DELETE FROM extraction_run WHERE id = ?", (run_id,)).rowcount
    )
    return {
        "run_id": run_id,
        "deleted_associations": deleted_associations,
        "deleted_episodes": deleted_episodes,
        "deleted_tasks": deleted_tasks,
        "deleted_sources": deleted_sources,
        "deleted_run": deleted_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or clear artifacts from ledger-running file imports after "
            "an ungraceful stop"
        )
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "also clean ledger/database failed runs so they can be retried; "
            "never includes completed or partial runs"
        ),
    )
    args = parser.parse_args()

    database = args.database.resolve()
    ledger_path = args.ledger.resolve()
    ledger = load_ledger(ledger_path)
    recoverable_statuses = {"running"}
    if args.include_failed:
        recoverable_statuses.add("failed")
    target_keys = sorted(
        key
        for key, record in ledger["files"].items()
        if record.get("status") in recoverable_statuses
    )

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        status_marks = marks(sorted(recoverable_statuses))
        target_runs = [
            int(row[0])
            for row in connection.execute(
                f"SELECT id FROM extraction_run WHERE status IN ({status_marks}) ORDER BY id",
                sorted(recoverable_statuses),
            )
        ]
        reports = [describe_run(connection, run_id) for run_id in target_runs]
        run_keys = sorted(
            {key for report in reports for key in report["source_keys"]}
        )
        unexpected_run_keys = sorted(set(run_keys) - set(target_keys))
        missing_runs = sorted(set(target_keys) - set(run_keys))
        safe_to_apply = not unexpected_run_keys
        report = {
            "database": str(database),
            "ledger": str(ledger_path),
            "recoverable_statuses": sorted(recoverable_statuses),
            "ledger_target_keys": target_keys,
            "target_runs": reports,
            "unexpected_run_keys": unexpected_run_keys,
            "ledger_running_keys_without_run": missing_runs,
            "safe_to_apply": safe_to_apply,
            "applied": False,
        }
        if args.apply:
            if not safe_to_apply:
                raise SystemExit(
                    "refusing cleanup: a target extraction run is not represented "
                    "by a matching ledger record"
                )
            with connection:
                report["deleted"] = [
                    delete_run_artifacts(connection, int(item["run"]["id"]))
                    for item in reports
                ]
            report["deletion_totals"] = {
                field: sum(int(item.get(field, 0)) for item in report["deleted"])
                for field in (
                    "deleted_associations",
                    "deleted_episodes",
                    "deleted_tasks",
                    "deleted_sources",
                    "deleted_run",
                )
            }
            foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_issues:
                raise RuntimeError(f"foreign key check failed after cleanup: {foreign_key_issues}")
            now = utc_now()
            for key in target_keys:
                ledger["files"][key].update(
                    {
                        "status": "interrupted",
                        "finished_at": now,
                        "error": "operator cleanup removed retryable extraction run",
                    }
                )
            ledger["last_process"] = {
                "started_at": now,
                "finished_at": now,
                "status": "paused",
                "reason": "operator cleanup removed retryable extraction runs",
                "interrupted_files": target_keys,
            }
            ledger["last_recovery"] = {
                "at": now,
                "action": "cleaned retryable extraction runs for safe resume",
                "source_keys": target_keys,
                "run_ids": [int(item["run"]["id"]) for item in reports],
                "deletion_totals": report["deletion_totals"],
            }
            save_ledger(ledger_path, ledger)
            report["applied"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
