from __future__ import annotations

import argparse
import atexit
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sqlite3
import sys
from threading import Lock
import traceback

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.database import DatabaseBusyError
from memory_demo.ingestion.interruption import (
    ImportProcessLease,
    recover_stale_file_runs,
)
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.llm import ModelClient, ModelTransportUnavailable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "created_at": utc_now(), "files": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError(f"invalid progress ledger: {path}")
    return value


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def select_pending_file_keys(
    ordered_keys: list[str],
    ledger: dict,
    *,
    retry_interrupted: bool,
    retry_failed: bool,
    limit: int | None,
) -> list[str]:
    """Select only explicitly retryable terminal ledger entries."""

    terminal_statuses = {"completed", "partial"}
    if not retry_interrupted:
        terminal_statuses.add("interrupted")
    if not retry_failed:
        terminal_statuses.add("failed")
    selected = [
        key
        for key in ordered_keys
        if ledger["files"].get(key, {}).get("status") not in terminal_statuses
    ]
    if limit is not None:
        return selected[: max(0, limit)]
    return selected


def begin_ledger_file_attempt(
    previous: dict | None,
    *,
    prompt_version: str,
    started_at: str,
) -> dict:
    """Start a ledger attempt without losing a prior failed-run receipt."""

    attempt = {
        "status": "running",
        "started_at": started_at,
        "prompt_version": prompt_version,
    }
    if not previous:
        return attempt
    attempts = deepcopy(previous.get("attempts", []))
    prior_attempt = deepcopy(previous)
    # Prior history is stored once, rather than recursively on every retry.
    prior_attempt.pop("attempts", None)
    attempts.append(prior_attempt)
    attempt["attempts"] = attempts
    attempt["attempt_number"] = len(attempts) + 1
    return attempt


def failed_retry_preflight(
    database_path: Path,
    ledger: dict,
    source_keys: list[str],
) -> list[str]:
    """Fail closed before retrying a failed file with persisted artifacts.

    Failed file runs are preserved as audit history.  A second import is safe
    only when the prior run has no Source, Episode, or Paragraph artifacts.
    """

    failed_keys = [
        key
        for key in source_keys
        if ledger["files"].get(key, {}).get("status") == "failed"
    ]
    if not failed_keys:
        return []
    issues: list[str] = []
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        for key in failed_keys:
            record = ledger["files"][key]
            summary = record.get("summary")
            run_id = summary.get("run_id") if isinstance(summary, dict) else None
            try:
                run_id = int(run_id)
            except (TypeError, ValueError):
                issues.append(f"{key}: ledger failure has no valid extraction run id")
                continue
            run = connection.execute(
                "SELECT status FROM extraction_run WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None or str(run["status"]) != "failed":
                actual = "missing" if run is None else str(run["status"])
                issues.append(
                    f"{key}: extraction run {run_id} is {actual}, not failed"
                )
                continue
            run_keys = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT source_key FROM extraction_task WHERE run_id = ?",
                    (run_id,),
                )
                if str(row[0]).strip()
            }
            if run_keys and run_keys != {key}:
                issues.append(
                    f"{key}: extraction run {run_id} belongs to {sorted(run_keys)}"
                )
                continue
            source_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT source_id FROM extraction_task
                        WHERE run_id = ? AND source_id IS NOT NULL
                        UNION
                        SELECT source_id FROM episode
                        WHERE extraction_run_id = ?
                    )
                    """,
                    (run_id, run_id),
                ).fetchone()[0]
            )
            episode_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM episode WHERE extraction_run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            paragraph_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT paragraph.id)
                    FROM paragraph
                    JOIN extraction_task ON extraction_task.source_id = paragraph.source_id
                    WHERE extraction_task.run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            if source_count or episode_count or paragraph_count:
                issues.append(
                    f"{key}: extraction run {run_id} has persisted artifacts "
                    f"(sources={source_count}, episodes={episode_count}, "
                    f"paragraphs={paragraph_count})"
                )
    finally:
        connection.close()
    return issues


def emit_progress(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def install_graceful_termination_handlers() -> None:
    """Route supported termination signals through the audited shutdown path."""

    def request_shutdown(signum, _frame) -> None:
        raise KeyboardInterrupt(f"received termination signal {signum}")

    for signal_name in ("SIGTERM", "SIGBREAK"):
        candidate = getattr(signal, signal_name, None)
        if candidate is not None:
            signal.signal(candidate, request_shutdown)


def load_selected_keys(path: Path) -> list[str]:
    """Load a frozen corpus selection without silently widening its scope."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_keys = payload.get("selected_files") if isinstance(payload, dict) else None
    if not isinstance(raw_keys, list) or not all(
        isinstance(item, str) and item for item in raw_keys
    ):
        raise ValueError(
            "selection manifest must contain a non-empty selected_files string list"
        )
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("selection manifest contains duplicate source keys")
    return raw_keys


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    install_graceful_termination_handlers()
    parser = argparse.ArgumentParser(
        description="Resumable per-file importer for the Blue Archive corpus"
    )
    parser.add_argument("story_root")
    parser.add_argument("--ledger", default="validation/full-import-progress.json")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="optional frozen JSON manifest whose selected_files list limits import",
    )
    parser.add_argument(
        "--categories", nargs="+", default=["main", "event", "favor"]
    )
    parser.add_argument(
        "--priority",
        nargs="*",
        default=[
            "main/33190.json",
            "main/33200.json",
            "main/33210.json",
            "favor/10005/1000515.json",
        ],
        help="corpus-relative files imported before the remaining sorted files",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="concurrent Source extraction workers (database writes remain serialized)",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="concurrent files; values above 1 require --defer-inference-relations",
    )
    parser.add_argument(
        "--relation-batch-size",
        type=int,
        default=24,
        help="number of Concept/Episode candidate groups per relation LLM call",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="HTTP retry budget for each model request",
    )
    parser.add_argument(
        "--defer-inference-relations",
        action="store_true",
        help=(
            "keep direct Episode-Concept links but skip import-time LLM "
            "Episode-Episode and Concept-Concept inference"
        ),
    )
    parser.add_argument(
        "--retry-interrupted",
        action="store_true",
        help=(
            "retry ledger entries marked interrupted; stale running entries are "
            "reconciled and cleaned safely at startup"
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "retry failed files only after a read-only check proves their prior "
            "runs have no persisted Source, Episode, or Paragraph artifacts"
        ),
    )
    parser.add_argument(
        "--pause-file",
        type=Path,
        help=(
            "control file: when it exists, stop dispatching new files, let "
            "already-started files finish, then exit with a resumable ledger"
        ),
    )
    args = parser.parse_args()
    if args.file_workers > 1 and not args.defer_inference_relations:
        parser.error("--file-workers > 1 requires --defer-inference-relations")

    root = Path(args.story_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"story root does not exist: {root}")
    ledger_path = Path(args.ledger).resolve()
    ledger = load_ledger(ledger_path)
    files = sorted(
        path
        for category in args.categories
        for path in (root / category).rglob("*.json")
        if path.is_file()
    )
    by_key = {path.relative_to(root).as_posix(): path for path in files}
    if args.selection_manifest:
        selected_keys = load_selected_keys(args.selection_manifest.resolve())
        missing = [key for key in selected_keys if key not in by_key]
        if missing:
            raise ValueError(
                "selection manifest references missing corpus files: "
                + ", ".join(missing[:20])
            )
        selected_set = set(selected_keys)
        ordered_keys = [key for key in args.priority if key in selected_set]
        ordered_set = set(ordered_keys)
        for category in args.categories:
            for key in sorted(selected_set - ordered_set):
                if Path(key).parts[0] == category:
                    ordered_keys.append(key)
                    ordered_set.add(key)
        ordered_keys.extend(key for key in selected_keys if key not in ordered_set)
    else:
        ordered_keys = [key for key in args.priority if key in by_key]
        ordered_keys.extend(
            key for key in sorted(by_key) if key not in set(ordered_keys)
        )

    pause_file = args.pause_file.resolve() if args.pause_file else None
    def pending_file_keys() -> list[str]:
        return select_pending_file_keys(
            ordered_keys,
            ledger,
            retry_interrupted=args.retry_interrupted,
            retry_failed=args.retry_failed,
            limit=args.limit,
        )

    if pause_file is not None and pause_file.exists():
        pending_keys = pending_file_keys()
        ledger["last_process"] = {
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "status": "paused",
            "reason": "pause control file already existed before dispatch",
            "pause_file": str(pause_file),
            "pending_files": len(pending_keys),
        }
        save_ledger(ledger_path, ledger)
        emit_progress(
            {
                "event": "corpus_import_paused",
                "reason": "pause control file already existed before dispatch",
                "pause_file": str(pause_file),
                "remaining_files": len(pending_keys),
            }
        )
        return

    config = AppConfig.from_env(args.env_file)
    lease = ImportProcessLease.acquire(ledger_path)
    # Deliberately retain the lease until interpreter shutdown.  This covers
    # every return and uncaught exception without relying on a fragile manual
    # release path; a force-killed process is recovered by the next startup.
    atexit.register(lease.release)
    recovery = recover_stale_file_runs(
        config.database_path,
        ledger,
        reason="previous importer process ended unexpectedly",
    )
    if recovery["stale_running_keys"] or recovery["empty_incomplete_run_ids"]:
        save_ledger(ledger_path, ledger)
        emit_progress(
            {
                "event": "corpus_stale_runs_recovered",
                "recovery": recovery,
            }
        )

    pending_keys = pending_file_keys()
    retry_failed_keys = [
        key for key in pending_keys if ledger["files"].get(key, {}).get("status") == "failed"
    ]
    if retry_failed_keys:
        retry_issues = failed_retry_preflight(
            config.database_path,
            ledger,
            retry_failed_keys,
        )
        if retry_issues:
            raise SystemExit(
                "refusing failed-file retry because prior artifacts are not safe to "
                "replace:\n" + "\n".join(retry_issues)
            )
    config.model.max_retries = max(0, int(args.max_retries))
    app = MemoryApplication(config)
    logger = app.new_logger("full-import")
    model = ModelClient(config.model, logger)
    pipeline = ImportPipeline(
        config,
        app.db,
        model,
        logger,
        app.episode_index,
        app.concept_index,
        prepare_workers=args.workers,
        relation_batch_size=args.relation_batch_size,
        build_inference_relations=not args.defer_inference_relations,
    )
    pipeline.rebuild_indexes()
    started_at = utc_now()
    totals = {
        "eligible_files": len(ordered_keys),
        "selected_files": len(pending_keys),
        "pending_files": len(pending_keys),
        "completed_this_process": 0,
        "partial_this_process": 0,
        "failed_this_process": 0,
        "interrupted_this_process": 0,
        "sources": 0,
        "episodes": 0,
        "failed_tasks": 0,
        "retry_failed_files": len(retry_failed_keys),
    }
    emit_progress(
        {
            "event": "corpus_import_started",
            "started_at": started_at,
            "database": str(config.database_path),
            "prompt_version": config.prompt_version,
            "pending_files": len(pending_keys),
            "ledger": str(ledger_path),
            "selection_manifest": (
                str(args.selection_manifest.resolve())
                if args.selection_manifest
                else None
            ),
            "inference_relations_deferred": args.defer_inference_relations,
            "file_workers": max(1, int(args.file_workers)),
            "retry_interrupted": args.retry_interrupted,
            "retry_failed": args.retry_failed,
            "retry_failed_files": len(retry_failed_keys),
            "pause_file": str(pause_file) if pause_file else None,
        }
    )

    state_lock = Lock()
    transport_state = {
        "requested_at": None,
        "source_key": None,
        "error": None,
        "reason": None,
    }

    def import_one(position: int, key: str) -> None:
        path = by_key[key]
        with state_lock:
            ledger["files"][key] = begin_ledger_file_attempt(
                ledger["files"].get(key),
                prompt_version=config.prompt_version,
                started_at=utc_now(),
            )
            save_ledger(ledger_path, ledger)
            emit_progress(
                {
                    "event": "file_started",
                    "position": position,
                    "pending_files": len(pending_keys),
                    "source_key": key,
                }
            )
        try:
            summary = pipeline.import_path(path, source_root=root)
            status = str(summary.get("status", "partial"))
            with state_lock:
                ledger["files"][key].update(
                    {"status": status, "finished_at": utc_now(), "summary": summary}
                )
                totals[f"{status}_this_process"] += 1
                for field in ("sources", "episodes", "failed_tasks"):
                    totals[field] += int(summary[field])
                save_ledger(ledger_path, ledger)
                emit_progress(
                    {
                        "event": "file_finished",
                        "position": position,
                        "source_key": key,
                        "status": status,
                        "summary": summary,
                    }
                )
        except (ModelTransportUnavailable, DatabaseBusyError) as exc:
            error_traceback = traceback.format_exc()
            retry_reason = (
                "local_transport_circuit"
                if isinstance(exc, ModelTransportUnavailable)
                else "sqlite_writer_busy"
            )
            with state_lock:
                transport_state.update(
                    {
                        "requested_at": utc_now(),
                        "source_key": key,
                        "error": str(exc),
                        "reason": retry_reason,
                    }
                )
                totals["interrupted_this_process"] += 1
                ledger["files"][key].update(
                    {
                        "status": "interrupted",
                        "finished_at": utc_now(),
                        "error": str(exc),
                        "traceback": error_traceback,
                        "retryable": True,
                        "retry_reason": retry_reason,
                    }
                )
                logger.emit(
                    "corpus_file_retryable_interrupted",
                    source_key=key,
                    reason=retry_reason,
                    error=str(exc),
                    traceback=error_traceback,
                )
                save_ledger(ledger_path, ledger)
                emit_progress(
                    {
                        "event": "file_interrupted",
                        "position": position,
                        "source_key": key,
                        "reason": retry_reason,
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            error_traceback = traceback.format_exc()
            with state_lock:
                totals["failed_this_process"] += 1
                ledger["files"][key].update(
                    {
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error": str(exc),
                        "traceback": error_traceback,
                    }
                )
                logger.emit(
                    "corpus_file_failed",
                    source_key=key,
                    error=str(exc),
                    traceback=error_traceback,
                )
                save_ledger(ledger_path, ledger)
                emit_progress(
                    {
                        "event": "file_finished",
                        "position": position,
                        "source_key": key,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    positions = list(enumerate(pending_keys, start=1))
    pause_state = {"requested_at": None, "submitted": 0}

    def pause_requested() -> bool:
        if pause_file is None or not pause_file.exists():
            return False
        if pause_state["requested_at"] is None:
            pause_state["requested_at"] = utc_now()
            emit_progress(
                {
                    "event": "corpus_pause_requested",
                    "pause_file": str(pause_file),
                    "submitted_files": pause_state["submitted"],
                    "remaining_files": len(pending_keys) - pause_state["submitted"],
                }
            )
        return True

    def transport_blocked() -> bool:
        with state_lock:
            if transport_state["requested_at"] is None:
                return False
            if not transport_state.get("emitted"):
                transport_state["emitted"] = True
                emit_progress(
                    {
                        "event": "corpus_import_transport_blocked",
                        "requested_at": transport_state["requested_at"],
                        "source_key": transport_state["source_key"],
                        "reason": transport_state["reason"],
                        "error": transport_state["error"],
                        "submitted_files": pause_state["submitted"],
                        "remaining_files": len(pending_keys) - pause_state["submitted"],
                    }
                )
            return True

    try:
        if args.file_workers <= 1:
            for position, key in positions:
                if pause_requested() or transport_blocked():
                    break
                pause_state["submitted"] += 1
                import_one(position, key)
        else:
            worker_count = max(1, int(args.file_workers))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="file-import",
            ) as executor:
                position_iter = iter(positions)
                futures = {}

                def fill_worker_slots() -> None:
                    while (
                        len(futures) < worker_count
                        and not pause_requested()
                        and not transport_blocked()
                    ):
                        try:
                            position, key = next(position_iter)
                        except StopIteration:
                            return
                        futures[executor.submit(import_one, position, key)] = (
                            position,
                            key,
                        )
                        pause_state["submitted"] += 1

                fill_worker_slots()
                while futures:
                    completed, _ = wait(
                        futures, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        del futures[future]
                        future.result()
                    fill_worker_slots()
    except KeyboardInterrupt as exc:
        with state_lock:
            shutdown_recovery = recover_stale_file_runs(
                config.database_path,
                ledger,
                reason=(
                    "importer received a graceful interruption: "
                    f"{type(exc).__name__}"
                ),
            )
            ledger["last_process"] = {
                "started_at": started_at,
                "finished_at": utc_now(),
                "status": "interrupted",
                "reason": "graceful keyboard or termination interruption",
                "pause_file": str(pause_file) if pause_file else None,
                "submitted_files": pause_state["submitted"],
                "recovery": shutdown_recovery,
                "config": asdict(config),
                "totals": totals,
            }
            save_ledger(ledger_path, ledger)
            emit_progress(
                {
                    "event": "corpus_import_interrupted",
                    "recovery": shutdown_recovery,
                    "totals": totals,
                }
            )
        raise

    process_status = (
        "transport_blocked"
        if transport_state["requested_at"] is not None
        else "paused" if pause_state["requested_at"] else "completed"
    )
    ledger["last_process"] = {
        "started_at": started_at,
        "finished_at": utc_now(),
        "status": process_status,
        "pause_file": str(pause_file) if pause_file else None,
        "pause_requested_at": pause_state["requested_at"],
        "submitted_files": pause_state["submitted"],
        "transport_abort": transport_state,
        "retry_policy": {
            "retry_interrupted": args.retry_interrupted,
            "retry_failed": args.retry_failed,
            "retry_failed_files": len(retry_failed_keys),
        },
        "config": asdict(config),
        "totals": totals,
    }
    save_ledger(ledger_path, ledger)
    if transport_state["requested_at"] is not None:
        emit_progress(
            {
                "event": "corpus_import_transport_blocked",
                "source_key": transport_state["source_key"],
                "reason": transport_state["reason"],
                "error": transport_state["error"],
                "remaining_files": len(pending_keys) - pause_state["submitted"],
                "totals": totals,
            }
        )
    elif pause_state["requested_at"]:
        emit_progress(
            {
                "event": "corpus_import_paused",
                "pause_file": str(pause_file),
                "remaining_files": len(pending_keys) - pause_state["submitted"],
                "totals": totals,
            }
        )
    else:
        emit_progress({"event": "corpus_import_finished", "totals": totals})
    lease.release()


if __name__ == "__main__":
    main()
