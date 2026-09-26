"""Run the unchanged historical five-question diagnostic on isolated full-KB clones.

This is a current-code rerun, not execution of a historical binary. The original
runner and its first-eight-Episode, fifteen-keyword score are reused unchanged.
All model clients share a persisted HTTP-attempt cap; this is not a currency cap.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT, PROJECT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks import run_original_five_question_benchmark as original
from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger, redact_for_answer_evidence_checkpoint
from memory_demo.llm.client import CampaignHttpBudget, ModelClient


SCHEMA = "original-five-current-code-controlled-rerun-v1"
MANIFEST = PROJECT / "benchmarks/manifests/original_five_question_quality_benchmark.json"
DEFAULT_DATABASE = PROJECT / "validation/v3-freeze-20260906T233016Z/testing_kb.sqlite"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(path: Path) -> dict:
    wal = path.with_name(path.name + "-wal")
    return {"sha256": digest(path), "bytes": path.stat().st_size,
            "wal_sha256": digest(wal) if wal.exists() else None}


def code_identity() -> str:
    paths = sorted((PROJECT / "src/memory_demo").rglob("*.py"))
    paths.extend([Path(original.__file__).resolve(), Path(__file__).resolve()])
    result = hashlib.sha256()
    for path in paths:
        result.update(str(path.relative_to(PROJECT)).replace("\\", "/").encode("utf-8"))
        result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def clone_readonly(source: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError("refusing to replace an existing arm database")
    target.parent.mkdir(parents=True, exist_ok=True)
    incoming = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    outgoing = sqlite3.connect(target)
    try:
        incoming.execute("PRAGMA query_only=ON")
        incoming.backup(outgoing)
        outgoing.execute("PRAGMA journal_mode=DELETE")
        tables = {row[0] for row in outgoing.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {table: outgoing.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("source", "episode", "concept", "association", "paragraph") if table in tables}
        if list(outgoing.execute("PRAGMA foreign_key_check")):
            raise ValueError("new arm clone has foreign-key errors")
    finally:
        outgoing.close()
        incoming.close()
    return {"database": str(target), "sha256_after_backup": digest(target), "table_counts": counts}


@contextmanager
def dispatcher_lock(output: Path):
    """The campaign cap supports one dispatcher; prevent concurrent resumes."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".dispatcher.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def clean(value, secrets=()):
    value = redact_for_answer_evidence_checkpoint(value)
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [clean(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: clean(item, secrets) for key, item in value.items()}
    return value


def new_attempt(arm: dict, output: Path, source: Path) -> dict:
    attempt = {"number": len(arm["attempts"]) + 1, "status": "prepared", "prepared_at": now()}
    directory = output / "arms" / arm["id"] / f"attempt_{attempt['number']:03d}"
    attempt.update(clone_readonly(source, directory / "database.sqlite"))
    attempt["report"] = str(directory / "original_report.json")
    arm["attempts"].append(attempt)
    arm["status"] = "prepared"
    return attempt


def summarize(output: Path, plan: dict, budget: CampaignHttpBudget) -> dict:
    summary = {
        "schema": SCHEMA, "run_basis": plan["run_basis"], "updated_at": now(),
        "status": plan["status"], "stop_reason": plan.get("stop_reason"),
        "source_database": plan["source_database"], "source_identity_before": plan["source_identity_before"],
        "source_identity_after": plan.get("source_identity_after"),
        "source_unchanged": plan.get("source_unchanged"),
        "budget": budget.snapshot(),
        "arms": [{"id": arm["id"], "status": arm["status"],
                  "summary": arm.get("summary"), "report": arm["attempts"][-1]["report"]}
                 for arm in plan["arms"]],
        "scoring_scope": "Original 15-keyword coverage within the first eight selected Episodes; not Source-gold recall or answer accuracy.",
    }
    write_json(output / "run_summary.json", summary)
    return summary


def terminal_report(attempt: dict, arm: dict, secrets: tuple) -> bool:
    path = Path(attempt["report"])
    if not path.is_file():
        return False
    result = clean(json.loads(path.read_text(encoding="utf-8")), secrets)
    if (result.get("benchmark") != "original_five_question_evidence_coverage_v1"
            or not isinstance(result.get("rows"), list) or len(result["rows"]) != 5):
        raise ValueError("existing arm report is not a complete five-question attempt")
    write_json(path, result)
    attempt["status"] = "completed"
    attempt["finished_at"] = now()
    attempt["report_sha256"] = digest(path)
    attempt["database_sha256_after"] = digest(Path(attempt["database"]))
    arm["status"] = "completed"
    arm["summary"] = result["summary"]
    arm["all_questions_failed"] = int(result["summary"].get("completed_questions", 0)) == 0
    return True


def run(args, *, runner_main=None, model_factory=None) -> dict:
    output = args.output_dir.resolve()
    with dispatcher_lock(output):
        return _run_locked(args, output, runner_main=runner_main or original.main,
                           model_factory=model_factory or ModelClient)


def _run_locked(args, output: Path, *, runner_main, model_factory) -> dict:
    plan_path = output / "run_manifest.json"
    if args.resume:
        if not plan_path.is_file():
            raise ValueError("--resume requires an existing run_manifest.json")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("schema") != SCHEMA:
            raise ValueError("unsupported run manifest")
        source = Path(plan["source_database"])
        env_file = Path(plan["env_file"])
        checks = ((args.source_database, source, "source database"),
                  (args.env_file, env_file, "environment file"),
                  (args.max_http_attempts, plan["max_http_attempts"], "HTTP cap"),
                  (args.deadline_seconds, plan["deadline_seconds"], "deadline"))
        for supplied, saved, label in checks:
            supplied = supplied.resolve() if isinstance(supplied, Path) else supplied
            if supplied is not None and supplied != saved:
                raise ValueError(f"resume cannot change the saved {label}")
        if args.include_contextual and not plan["include_contextual"]:
            raise ValueError("resume cannot add contextual arms")
        if (source_identity(source) != plan["source_identity_before"]
                or digest(MANIFEST) != plan["question_manifest_sha256"]
                or code_identity() != plan["current_code_sha256"]):
            raise ValueError("source, question manifest, or current code changed since preparation")
    else:
        if plan_path.exists() or (output / "arms").exists():
            raise FileExistsError("output already contains a campaign; use --resume")
        source = (args.source_database or DEFAULT_DATABASE).resolve()
        env_file = (args.env_file or PROJECT / ".env").resolve()
        if not source.is_file() or not env_file.is_file():
            raise FileNotFoundError("source database and environment file must exist")
        original._load_manifest(MANIFEST)
        plan = {
            "schema": SCHEMA, "campaign_id": "original-five-" + uuid4().hex,
            "run_basis": "current_code_rerun_not_historical_binary", "created_at": now(),
            "status": "preparing", "source_database": str(source),
            "source_identity_before": source_identity(source), "env_file": str(env_file),
            "current_code_sha256": code_identity(), "question_manifest": str(MANIFEST),
            "question_manifest_sha256": digest(MANIFEST),
            "max_http_attempts": args.max_http_attempts if args.max_http_attempts is not None else 120,
            "deadline_seconds": args.deadline_seconds if args.deadline_seconds is not None else 120.0,
            "include_contextual": bool(args.include_contextual),
            "provider_policy": "Original configured models/retries/fallbacks unchanged; all real HTTP dispatches share the persisted attempt cap. No currency cap is claimed.",
            "logger_policy": "Operational logger bodies redacted; answer-evidence companion disabled.",
            "resume_policy": "Completed arms, including complete failure reports, are never rerun. Interrupted arms without complete reports get a fresh clone; old attempts and consumed HTTP reservations remain.",
            "arms": [],
        }
        modes = ("baseline", "contextual") if args.include_contextual else ("baseline",)
        for mode in modes:
            for reranker in ("bge_enabled", "bge_disabled"):
                plan["arms"].append({"id": f"{mode}__{reranker}", "mode": mode,
                                     "reranker": reranker, "status": "not_started", "attempts": []})
        write_json(plan_path, plan)
    if plan["max_http_attempts"] < 1 or plan["deadline_seconds"] <= 0:
        raise ValueError("HTTP cap and per-question deadline must be positive")
    budget = CampaignHttpBudget.open(campaign_id=plan["campaign_id"],
                                    max_http_attempts=plan["max_http_attempts"],
                                    receipt_path=output / "http_budget.json")
    for arm in plan["arms"]:
        if not arm["attempts"]:
            new_attempt(arm, output, source)
            write_json(plan_path, plan)
    plan["source_identity_after"] = source_identity(source)
    plan["source_unchanged"] = plan["source_identity_after"] == plan["source_identity_before"]
    if not plan["source_unchanged"]:
        raise ValueError("source database changed during preparation")
    if args.prepare_only:
        plan["status"] = "prepared" if all(arm["status"] == "prepared" for arm in plan["arms"]) else "partially_completed"
        write_json(plan_path, plan)
        return summarize(output, plan, budget)
    config = AppConfig.from_env(env_file)
    secrets = (config.model.api_key,) if config.model.api_key else ()
    plan["status"] = "running"
    plan.pop("stop_reason", None)
    try:
        for arm in plan["arms"]:
            if arm["status"] in ("completed", "failed"):
                continue
            attempt = arm["attempts"][-1]
            if attempt["status"] == "running":
                if terminal_report(attempt, arm, secrets):
                    write_json(plan_path, plan)
                    if arm["all_questions_failed"]:
                        plan["stop_reason"] = "all_questions_failed_in_recovered_arm"
                        break
                    continue
                attempt["status"] = "interrupted"
                attempt = new_attempt(arm, output, source)
            if budget.snapshot()["remaining_http_attempts"] == 0:
                plan["stop_reason"] = "shared_http_budget_exhausted"
                break
            if digest(Path(attempt["database"])) != attempt["sha256_after_backup"]:
                raise ValueError("prepared arm database changed before execution")
            attempt.update({"status": "running", "started_at": now(), "budget_before": budget.snapshot()})
            arm["status"] = "running"
            write_json(plan_path, plan)
            print(json.dumps({"arm": arm["id"], "status": "running", "budget": budget.snapshot()}), flush=True)
            stdout, stderr = io.StringIO(), io.StringIO()

            def controlled_model(*factory_args, **factory_kwargs):
                factory_kwargs["campaign_budget"] = budget
                return model_factory(*factory_args, **factory_kwargs)

            def body_free_logger(path, **kwargs):
                return JsonlEventLogger(path, answer_evidence_enabled=False)

            argv = [str(Path(original.__file__).resolve()), attempt["database"],
                    "--output", attempt["report"], "--manifest", str(MANIFEST),
                    "--env-file", str(env_file), "--mode", arm["mode"],
                    "--reranker", arm["reranker"], "--deadline-seconds", str(plan["deadline_seconds"])]
            started = time.perf_counter()
            try:
                with patch.object(sys, "argv", argv), patch("memory_demo.app.ModelClient", controlled_model), \
                     patch("memory_demo.app.JsonlEventLogger", body_free_logger), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    attempt["exit_code"] = runner_main()
            except Exception as exc:
                attempt["error"] = {"type": type(exc).__name__, "message": clean(str(exc), secrets)}
                attempt["status"] = arm["status"] = "failed"
                plan["stop_reason"] = "runner_failed"
            finally:
                attempt["elapsed_seconds"] = time.perf_counter() - started
                attempt["budget_after"] = budget.snapshot()
                directory = Path(attempt["report"]).parent
                (directory / "stdout.txt").write_text(clean(stdout.getvalue(), secrets), encoding="utf-8")
                (directory / "stderr.txt").write_text(clean(stderr.getvalue(), secrets), encoding="utf-8")
                plan["source_identity_after"] = source_identity(source)
                plan["source_unchanged"] = plan["source_identity_after"] == plan["source_identity_before"]
                write_json(plan_path, plan)
            if arm["status"] != "failed":
                if not terminal_report(attempt, arm, secrets):
                    attempt["status"] = arm["status"] = "failed"
                    plan["stop_reason"] = "runner_returned_without_complete_report"
                elif arm["all_questions_failed"]:
                    plan["stop_reason"] = "all_five_questions_failed_stop_remaining_arms"
            write_json(plan_path, plan)
            summarize(output, plan, budget)
            if not plan["source_unchanged"]:
                raise ValueError("source database changed during execution")
            if plan.get("stop_reason"):
                break
        plan["status"] = "stopped" if plan.get("stop_reason") else "completed"
    except BaseException:
        plan["status"] = "interrupted"
        raise
    finally:
        plan["source_identity_after"] = source_identity(source)
        plan["source_unchanged"] = plan["source_identity_after"] == plan["source_identity_before"]
        plan["updated_at"] = now()
        write_json(plan_path, plan)
        summarize(output, plan, budget)
    return summarize(output, plan, budget)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-http-attempts", type=int, help="shared hard cap, default 120")
    parser.add_argument("--deadline-seconds", type=float, help="per-question deadline, default 120")
    parser.add_argument("--include-contextual", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("prepared", "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
