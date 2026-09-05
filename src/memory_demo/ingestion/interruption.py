from __future__ import annotations

"""Crash-safe ownership and recovery for file-level corpus imports.

The importer writes a ledger entry before it begins each file.  A forced
process exit can therefore leave a file marked ``running`` after its worker
has persisted only part of the file.  This module makes the next startup
reconcile that durable state before it can retry the file.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from memory_demo.database import Database
from memory_demo.repositories.extraction import ExtractionRepository


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImportLeaseConflict(RuntimeError):
    """Another importer still owns the ledger's active-process lease."""


class UnsafeStaleRunRecovery(RuntimeError):
    """Stale database rows cannot be safely matched to ledger files."""


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A process we cannot inspect is safer to treat as live.
        return True
    except OSError:
        # Windows reports an exited PID as WinError 87 rather than
        # ProcessLookupError for os.kill(pid, 0).  Permission failures were
        # handled above and remain fail-closed; other OS errors mean this PID
        # cannot be an active lease owner on the current host.
        return False
    else:
        return True


class ImportProcessLease:
    """An atomic, PID-backed lease that prevents concurrent importers.

    A forced exit leaves the tiny lease file behind.  The next process only
    takes it after confirming that its recorded PID is no longer alive, then
    performs stale-run reconciliation before dispatching work.
    """

    def __init__(self, path: Path, token: str):
        self.path = path
        self.token = token

    @classmethod
    def acquire(cls, ledger_path: Path) -> "ImportProcessLease":
        path = ledger_path.with_suffix(ledger_path.suffix + ".active")
        path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        payload = {
            "version": 1,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "ledger": str(ledger_path),
            "token": token,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for _attempt in range(3):
            try:
                descriptor = os.open(
                    path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                existing_pid = existing.get("pid")
                if isinstance(existing_pid, int) and _pid_is_alive(existing_pid):
                    raise ImportLeaseConflict(
                        "another importer still owns the active lease "
                        f"(pid={existing_pid}, started_at={existing.get('started_at')})"
                    )
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
            else:
                try:
                    os.write(descriptor, encoded)
                finally:
                    os.close(descriptor)
                return cls(path, token)
        raise ImportLeaseConflict("could not acquire the importer active lease")

    def release(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("token") == self.token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _marks(values: list[str]) -> str:
    return ",".join("?" for _ in values)


def _run_source_keys(connection: sqlite3.Connection, run_id: int) -> list[str]:
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


def _run_artifact_counts(connection: sqlite3.Connection, run_id: int) -> dict[str, int]:
    return {
        "tasks": int(
            connection.execute(
                "SELECT COUNT(*) FROM extraction_task WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        ),
        "episodes": int(
            connection.execute(
                "SELECT COUNT(*) FROM episode WHERE extraction_run_id = ?", (run_id,)
            ).fetchone()[0]
        ),
    }


def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    raw_summary = row["summary_json"]
    if not raw_summary:
        return {}
    try:
        value = json.loads(raw_summary)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def recover_stale_file_runs(
    database_path: Path,
    ledger: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Reconcile durable runs left by a process that is no longer alive.

    Completed/partial database runs win over a stale ledger entry, preserving
    their work.  Incomplete runs are deleted only after their source key is
    matched one-to-one with a stale ``running`` ledger entry.  Any ambiguous
    ownership fails closed, leaving the database untouched for inspection.
    """

    files = ledger.get("files")
    if not isinstance(files, dict):
        raise ValueError("ledger files must be an object")
    stale_keys = sorted(
        key for key, record in files.items() if record.get("status") == "running"
    )
    database = Database(database_path)
    by_key: dict[str, sqlite3.Row] = {}
    incomplete_plans: list[tuple[int, str]] = []
    empty_incomplete_run_ids: list[int] = []
    invalid: list[str] = []
    with database.connection() as connection:
        if stale_keys:
            placeholders = _marks(stale_keys)
            rows = connection.execute(
                f"""
                SELECT extraction_run.id, extraction_run.status,
                       extraction_run.finished_at, extraction_run.summary_json,
                       extraction_task.source_key
                FROM extraction_run
                JOIN extraction_task ON extraction_task.run_id = extraction_run.id
                WHERE extraction_task.source_key IN ({placeholders})
                ORDER BY extraction_run.id DESC
                """,
                stale_keys,
            ).fetchall()
            for row in rows:
                key = str(row["source_key"])
                by_key.setdefault(key, row)

        incomplete_rows = connection.execute(
            """
            SELECT id, status FROM extraction_run
            WHERE status IN ('running', 'interrupted')
            ORDER BY id
            """
        ).fetchall()
        seen_keys: set[str] = set()
        for row in incomplete_rows:
            run_id = int(row["id"])
            run_status = str(row["status"])
            source_keys = _run_source_keys(connection, run_id)
            if not source_keys:
                counts = _run_artifact_counts(connection, run_id)
                if counts["tasks"] or counts["episodes"]:
                    if run_status == "running":
                        invalid.append(
                            f"run {run_id} has artifacts but no source-key ownership"
                        )
                    continue
                if run_status == "running" or stale_keys:
                    empty_incomplete_run_ids.append(run_id)
                continue
            if len(source_keys) != 1:
                if run_status == "running":
                    invalid.append(
                        f"run {run_id} belongs to multiple source keys: {source_keys}"
                    )
                continue
            source_key = source_keys[0]
            if source_key not in stale_keys:
                if run_status == "running":
                    invalid.append(
                        f"run {run_id} belongs to non-stale ledger key {source_key}"
                    )
                continue
            if source_key in seen_keys:
                invalid.append(
                    f"multiple running runs belong to stale key {source_key}"
                )
                continue
            seen_keys.add(source_key)
            incomplete_plans.append((run_id, source_key))

    if invalid:
        raise UnsafeStaleRunRecovery("; ".join(invalid))

    repository = ExtractionRepository(database)
    deleted = [
        repository.discard_incomplete_run(run_id)
        for run_id, _source_key in [
            *incomplete_plans,
            *((run_id, "") for run_id in empty_incomplete_run_ids),
        ]
    ]
    now = utc_now()
    reconciled_completed: list[str] = []
    reconciled_partial: list[str] = []
    reconciled_failed: list[str] = []
    interrupted: list[str] = []
    missing_run_keys: list[str] = []
    incomplete_by_key = {
        source_key: run_id for run_id, source_key in incomplete_plans
    }
    for key in stale_keys:
        record = files[key]
        row = by_key.get(key)
        if row is not None and str(row["status"]) in {"completed", "partial", "failed"}:
            status = str(row["status"])
            record.update(
                {
                    "status": status,
                    "finished_at": row["finished_at"] or now,
                    "summary": _summary_from_row(row),
                    "reconciled_at": now,
                    "reconciliation_reason": reason,
                }
            )
            if status == "completed":
                reconciled_completed.append(key)
            elif status == "partial":
                reconciled_partial.append(key)
            else:
                reconciled_failed.append(key)
            continue
        run_id = incomplete_by_key.get(key)
        if run_id is None:
            missing_run_keys.append(key)
        record.update(
            {
                "status": "interrupted",
                "finished_at": now,
                "error": reason,
                "retryable": True,
                "retry_reason": "unexpected_process_exit",
                "recovered_run_id": run_id,
            }
        )
        interrupted.append(key)

    totals = {
        field: sum(int(item.get(field, 0)) for item in deleted)
        for field in (
            "deleted_associations",
            "deleted_episodes",
            "deleted_tasks",
            "deleted_sources",
            "deleted_run",
        )
    }
    report = {
        "at": now,
        "reason": reason,
        "stale_running_keys": stale_keys,
        "interrupted_keys": interrupted,
        "missing_database_run_keys": missing_run_keys,
        "reconciled_completed_keys": reconciled_completed,
        "reconciled_partial_keys": reconciled_partial,
        "reconciled_failed_keys": reconciled_failed,
        "empty_incomplete_run_ids": empty_incomplete_run_ids,
        "deleted": deleted,
        "deletion_totals": totals,
    }
    if stale_keys or empty_incomplete_run_ids:
        ledger["last_recovery"] = report
    return report
