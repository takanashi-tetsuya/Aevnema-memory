"""Isolated associative-recall trials with fixed runtime questions and budgets.

No evaluator labels, alternative Episode IDs, or scoring functions are loaded.
Terminal attempts never receive extra time on campaign resume.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import time
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]

from benchmarks.run_progressive_five_question_benchmark import (
    _MetadataLogger, _campaign_lock, _digest, _hide_secret, _json_hash, _read, _write,
)
from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.event_log import redact_for_answer_evidence_checkpoint, redact_for_export
from memory_demo.llm import CampaignHttpBudget, ModelClient, ProviderCallAccounting

SCHEMA = "associative_recall_experiment_v1"
TIMEOUT_SECONDS = 360.0


def _load_runtime_questions(path: Path, question_ids=None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not 1 <= len(questions) <= 15:
        raise ValueError("runtime manifest must contain one to fifteen questions")
    if any(not isinstance(q, dict) or set(q) != {"id", "question"}
           or any(not isinstance(q[k], str) or not q[k].strip() for k in ("id", "question"))
           for q in questions):
        raise ValueError("runtime questions must contain only nonempty id and question")
    by_id = {q["id"]: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError("runtime question IDs must be unique")
    if question_ids is not None:
        if not question_ids or len(set(question_ids)) != len(question_ids) or any(i not in by_id for i in question_ids):
            raise ValueError("selected question IDs must be unique IDs from the runtime manifest")
        return [dict(by_id[i]) for i in question_ids]
    return [dict(q) for q in questions]


def _semantic_snapshot(path: Path) -> str:
    """Schema-independent content identity; no evaluator targets are consulted."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = {
            "schema": "historical-source-episode-v1",
            "sources": [[i, digest(raw)] for i, raw in connection.execute("SELECT id,raw_text FROM source ORDER BY id")],
            "episodes": [[i, sid, key, segment, digest(text)] for i, sid, key, segment, text in connection.execute(
                "SELECT id,source_id,source_key,segment_index,text FROM episode ORDER BY id")],
        }
    finally:
        connection.close()
    return _json_hash(payload)


def _clone(source: Path, target: Path, expected_digest: str, semantic_digest: str) -> None:
    # SQLite backup reads any existing WAL through a mode=ro source connection.
    # Unlike immutable replay, this also permits an archived zero-length WAL.
    with target.open("xb"):
        pass
    try:
        original = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        original.execute("PRAGMA query_only=ON")
        destination = sqlite3.connect(target)
        try:
            original.backup(destination)
        finally:
            destination.close()
            original.close()
        if _digest(source) != expected_digest or _semantic_snapshot(target) != semantic_digest:
            raise ValueError("source database changed or clone content differs")
    except BaseException:
        # Preserve any partially prepared directory for inspection; no overwrite.
        raise


def _code_fingerprints() -> dict:
    paths = list((PROJECT / "src/memory_demo").rglob("*.py"))
    paths += list((PROJECT / "config/prompt_config").rglob("*.py"))
    paths += [Path(__file__), PROJECT / "benchmarks/run_progressive_five_question_benchmark.py",
              PROJECT / "benchmarks/run_original_five_question_benchmark.py",
              PROJECT / "benchmarks/support/readonly_database.py"]
    return {p.relative_to(PROJECT).as_posix(): _digest(p) for p in sorted(set(paths))}


def _public_config(config) -> dict:
    value = asdict(config)
    value.pop("database_path", None)
    value.pop("log_dir", None)
    value["model"].pop("api_key", None)
    return json.loads(json.dumps(value, default=str))


def _service_factory(protocol):
    if protocol == 2:
        from memory_demo.retrieval.progressive import ProgressiveRecall
        return ProgressiveRecall
    if protocol == 3:
        from memory_demo.retrieval.progressive_v3 import ProgressiveRecallV3
        return ProgressiveRecallV3
    if protocol == 4:
        from memory_demo.retrieval.progressive_v4 import ProgressiveRecallV4
        return ProgressiveRecallV4
    if protocol == 5:
        from memory_demo.retrieval.progressive_v5 import ProgressiveRecallV5
        return ProgressiveRecallV5
    if protocol == 6:
        from memory_demo.retrieval.progressive_v6 import ProgressiveRecallV6
        return ProgressiveRecallV6
    if protocol == 7:
        from memory_demo.retrieval.progressive_v7 import ProgressiveRecallV7
        return ProgressiveRecallV7
    if protocol == 8:
        from memory_demo.retrieval.progressive_v8 import ProgressiveRecallV8
        return ProgressiveRecallV8
    if protocol == 9:
        from memory_demo.retrieval.progressive_v9 import ProgressiveRecallV9
        return ProgressiveRecallV9
    if protocol == 10:
        from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
        return ProgressiveRecallV10
    if protocol == 11:
        from memory_demo.retrieval.progressive_v11 import ProgressiveRecallV11
        return ProgressiveRecallV11
    if protocol == 12:
        from memory_demo.retrieval.progressive_v12 import ProgressiveRecallV12
        return ProgressiveRecallV12
    raise ValueError("unknown recall protocol")


def _recorded_budget_floor(output: Path) -> int:
    """A stale valid budget receipt cannot erase usage recorded elsewhere."""
    snapshots = []
    report = output / "report.json"
    if report.exists():
        snapshots.append(_read(report).get("campaign_http_budget", {}))
    for path in output.glob("question-*/first/invocations.json"):
        for invocation in _read(path).get("invocations", []):
            snapshots.append(invocation.get("campaign_before", {}))
            snapshots.append(invocation.get("provider", {}).get("campaign_after", {}))
    values = [snapshot.get("reserved_http_attempts", 0) for snapshot in snapshots]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("historical HTTP budget usage is invalid")
    return max(values, default=0)


def _recover_result(state: dict, service, protocol: int) -> dict:
    if state.get("version") != protocol:
        raise ValueError("checkpoint protocol changed; prepare a fresh campaign")
    service._validate_delivered_sources(state)
    needs, facts = state.get("needs", []), state.get("facts", [])
    resolved = set(state.get("resolved_needs", []))
    result = {
        "session_id": state["session_id"], "mode": "deep", "status": state["status"],
        "complete": state["status"] == "complete", "timeout_seconds": TIMEOUT_SECONDS,
        "elapsed_seconds": state.get("elapsed_seconds", 0), "needs": needs,
        "missing_requirements": [n for i, n in enumerate(needs) if i not in resolved] or ([state["question"]] if not needs else []),
        "answer": "\n".join(dict.fromkeys(f["claim"] for f in facts)), "evidence": facts,
        "used_edge_ids": state.get("used_edge_ids", []), "feedback_edge_ids": state.get("feedback_edge_ids", []),
        "metrics": state.get("metrics", {}), "learning": state.get("learning", {"status": "not_attempted"}),
        "error": state.get("error"), "resumable": False,
        "checkpoint_path": str(service.sessions._path(state["session_id"]).resolve()),
        "coverage_basis": f"model_plan_and_independent_source_review_v{protocol}",
        "review_protocol_version": protocol, "need_assessments": state.get("need_assessments", []),
        "pending_evidence_count": len(state.get("pending_facts", [])),
        "answer_status": "complete" if state["status"] == "complete" else "partial" if facts else "unknown",
        "candidate_episode_ids": state.get("candidate_episode_ids") if protocol >= 3 else None,
        "presented_episode_ids": state.get("presented_episode_ids") if protocol >= 3 else None,
        "stage_calls": state.get("stage_calls", []) if protocol >= 3 else None,
        "stage_errors": state.get("stage_errors", []) if protocol >= 3 else None,
    }

    annotate = getattr(service, "annotate_completion", None)
    return annotate(result, state) if callable(annotate) else result


def _committed(directory: Path, identity: dict) -> dict | None:
    marker = directory / "receipt.json"
    if not marker.exists():
        return None
    receipt = _read(marker)
    if receipt.get("identity") != identity or receipt.get("schema") != SCHEMA:
        raise ValueError("attempt receipt belongs to another campaign or question")
    files = receipt.get("files")
    if not isinstance(files, dict) or not {"result.json", "provider.json", "checkpoint.json", "invocations.json"} <= files.keys():
        raise ValueError("attempt receipt is incomplete")
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or _digest(path) != digest:
            raise ValueError("attempt artifact missing or changed")
    result = _read(directory / "result.json")
    if result.get("status") == "running":
        raise ValueError("a committed receipt cannot contain a running result")
    return result


def _attempt(question, directory, database, config, budget, identity, *, protocol, learn,
             application_factory, model_factory, service_factory, wall_clock) -> dict:
    directory.mkdir(exist_ok=True)
    local = deepcopy(config)
    local.database_path, local.log_dir = database, directory / "logs"
    app = application_factory(local)
    service = service_factory(local, app.db, None)
    if service.protocol_version != protocol:
        raise ValueError("service protocol does not match campaign")
    candidates = list((local.log_dir / "recall_sessions").glob("*.json"))
    if len(candidates) > 1:
        raise ValueError("unfinished attempt has multiple checkpoints")
    state = service.sessions.read(candidates[0].stem) if candidates else None
    if state and (state.get("question") != question["question"] or state.get("context") != "" or state.get("version") != protocol):
        raise ValueError("checkpoint request or protocol differs from campaign")
    ledger_path = directory / "invocations.json"
    ledger = _read(ledger_path) if ledger_path.exists() else None
    if state and ledger is None:
        raise ValueError("checkpoint lacks durable total-time budget")
    if ledger is None:
        now = wall_clock()
        ledger = {"identity": identity, "started_at_epoch": now,
                  "deadline_at_epoch": now + TIMEOUT_SECONDS, "invocations": []}
    if ledger.get("identity") != identity or ledger["deadline_at_epoch"] - ledger["started_at_epoch"] != TIMEOUT_SECONDS:
        raise ValueError("attempt time budget identity changed")
    recover_learning = getattr(service, "_recover_committed_learning", None)
    if state and callable(recover_learning) and recover_learning(state, refresh=True):
        # A database commit can precede the terminal JSON write. Recognize its
        # exact receipt locally even when the original wall deadline expired.
        service.sessions.write(state)
    before = budget.snapshot()
    invocation = {"started_at_epoch": wall_clock(), "campaign_before": before,
                  "resume_session": state.get("session_id") if state else None}
    ledger["invocations"].append(invocation)
    _write(ledger_path, ledger)
    accounting = ProviderCallAccounting()
    recovered = bool(state and state.get("status") != "running")
    remaining = min(TIMEOUT_SECONDS - float(state.get("elapsed_seconds", 0) if state else 0),
                    ledger["deadline_at_epoch"] - wall_clock())
    if recovered:
        result = _recover_result(state, service, protocol)
    elif remaining <= 0:
        if state is None:
            state = {"version": protocol, "session_id": uuid4().hex, "question": question["question"],
                     "context": "", "facts": [], "pending_facts": [], "needs": [],
                     "runtime_started": False, "learning": {"status": "not_attempted"},
                     "candidate_episode_ids": [] if protocol == 3 else None,
                     "presented_episode_ids": [] if protocol == 3 else None}
        state["status"] = "time_budget"
        state["elapsed_seconds"] = max(float(state.get("elapsed_seconds", 0)), TIMEOUT_SECONDS)
        service.sessions.write(state)
        result, recovered = _recover_result(state, service, protocol), True
    else:
        if int(before["remaining_http_attempts"]) <= 0:
            raise ValueError("campaign HTTP budget exhausted; no recall dispatched")
        logger = _MetadataLogger(directory / f"provider-{len(ledger['invocations']):03d}.jsonl", config.model.api_key)
        model = model_factory(deepcopy(local.model), logger, accounting=accounting, campaign_budget=budget)
        service.model = model
        result = service.query(question["question"], mode="deep", timeout_seconds=remaining,
                               learn=learn, resume=state["session_id"] if state else None)
    result = dict(result)
    result.setdefault("candidate_episode_ids", None)
    result.setdefault("presented_episode_ids", None)
    result["campaign_resume_extends_terminal_result"] = False
    after = budget.snapshot()
    provider = {
        "current_invocation": redact_for_export(accounting.snapshot()),
        "campaign_before": before, "campaign_after": after,
        "campaign_attempt_reservations": int(after["reserved_http_attempts"]) - int(ledger["invocations"][0]["campaign_before"]["reserved_http_attempts"]),
        "recovered_terminal_checkpoint_without_recall": recovered,
        "prior_interrupted_invocations": len(ledger["invocations"]) - 1,
        "prior_invocation_statistics_may_be_incomplete": len(ledger["invocations"]) > 1,
        "model_roles": {k: getattr(local.model, k) for k in ("embedding_model", "reasoning_model", "fallback_model", "reranker_model")},
    }
    invocation.update(finished_at_epoch=wall_clock(), provider=provider)
    _write(ledger_path, ledger)
    checkpoint_path = Path(str(result.get("checkpoint_path", ""))).resolve()
    if not checkpoint_path.is_relative_to(directory.resolve()) or not checkpoint_path.is_file():
        raise ValueError("runtime must return a checkpoint inside this attempt")
    checkpoint = service.sessions.read(str(result.get("session_id", "")))
    if checkpoint.get("status") == "running" or checkpoint.get("status") != result.get("status"):
        raise ValueError("runtime returned without a matching terminal checkpoint")
    for name, value in {"result.json": result, "provider.json": provider, "checkpoint.json": checkpoint}.items():
        _write(directory / name, _hide_secret(redact_for_answer_evidence_checkpoint(value), config.model.api_key))
    files = {p.relative_to(directory).as_posix(): _digest(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "receipt.json" and not p.name.endswith(".tmp")}
    _write(directory / "receipt.json", {"schema": SCHEMA, "identity": identity, "status": result["status"], "files": files})
    return result


def run(source_database: Path, output: Path, *, manifest: Path, question_ids=None, env_file=".env",
        protocol=3, learn=False, resume=False, max_http_attempts=120, prepare_only=False,
        config=None, application_factory=None, model_factory=None, service_factory=None,
        wall_clock=time) -> dict:
    questions = _load_runtime_questions(Path(manifest), question_ids)
    if type(protocol) is not int or protocol not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        raise ValueError("protocol must be 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 or 12")
    if type(max_http_attempts) is not int or max_http_attempts < 1:
        raise ValueError("max_http_attempts must be positive")
    source_database = Path(source_database).resolve(strict=True)
    output = Path(output).resolve()
    if source_database.is_relative_to(output):
        raise ValueError("output must not contain the archived source database")
    source_hash, semantic_hash = _digest(source_database), _semantic_snapshot(source_database)
    local_config = deepcopy(config) if config is not None else AppConfig.from_env(env_file)
    if local_config.model.reasoning_max_tokens is None:
        local_config.model.reasoning_max_tokens = 8192
    parameters = _public_config(local_config)
    binding = {"schema": SCHEMA, "review_protocol_version": protocol, "source_database": str(source_database),
               "source_sha256": source_hash, "source_semantic_sha256": semantic_hash,
               "runtime_questions": questions, "runtime_manifest_sha256": _json_hash(questions),
               "mode": "deep", "timeout_seconds": TIMEOUT_SECONDS, "learn": bool(learn),
               "max_http_attempts": max_http_attempts, "parameters": parameters,
               "model_roles": parameters["model"], "code_fingerprints": _code_fingerprints()}
    if resume:
        if not (output / "campaign.json").is_file() or not (output / "http-budget.json").is_file():
            raise ValueError("resume requires the existing campaign and durable HTTP budget")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with _campaign_lock(output):
        if resume:
            campaign = _read(output / "campaign.json")
            if campaign.get("binding") != binding:
                raise ValueError("resume source, runtime questions, code fingerprints, model or parameters changed")
        else:
            campaign = {"campaign_id": uuid4().hex, "binding": binding,
                        "created_at": datetime.now(timezone.utc).isoformat()}
            _write(output / "campaign.json", campaign)
        if resume and _read(output / "http-budget.json").get("reserved_http_attempts", -1) < _recorded_budget_floor(output):
            raise ValueError("HTTP budget is below already recorded usage; refusing reset")
        budget = CampaignHttpBudget.open(campaign_id=campaign["campaign_id"], max_http_attempts=max_http_attempts,
                                         receipt_path=output / "http-budget.json")
        stored_budget = _read(output / "http-budget.json")
        if stored_budget["reserved_http_attempts"] != budget.snapshot()["reserved_http_attempts"]:
            raise ValueError("persisted HTTP budget differs from active campaign; refusing reset")
        prepared, rows, stopped = [], [], None
        for ordinal, question in enumerate(questions, 1):
            directory = output / f"question-{ordinal:02d}"
            database, ready = directory / "memory.sqlite", directory / "clone.json"
            identity = {"campaign_id": campaign["campaign_id"], "question_id": question["id"],
                        "source_sha256": source_hash, "protocol": protocol}
            if directory.exists():
                if not ready.is_file() or _read(ready).get("identity") != identity or not database.is_file():
                    raise ValueError("question directory is not an owned complete clone")
            else:
                directory.mkdir()
                _clone(source_database, database, source_hash, semantic_hash)
                _write(ready, {"identity": identity, "initial_clone_sha256": _digest(database),
                               "source_semantic_sha256": semantic_hash})
            if _semantic_snapshot(database) != semantic_hash:
                raise ValueError("working clone Source/Episode content changed")
            prepared.append({"id": question["id"], "question": question["question"], "database": str(database)})
            if prepare_only:
                continue
            attempt = directory / "first"
            result = _committed(attempt, identity)
            if result is None:
                if not attempt.exists() and budget.snapshot()["remaining_http_attempts"] <= 0:
                    stopped = "campaign_http_budget_exhausted"
                    break
                result = _attempt(question, attempt, database, local_config, budget, identity,
                                  protocol=protocol, learn=bool(learn), application_factory=application_factory or MemoryApplication,
                                  model_factory=model_factory or ModelClient, service_factory=service_factory or _service_factory(protocol),
                                  wall_clock=wall_clock)
            rows.append({"id": question["id"], "phase": "first", "database": str(database),
                         "result_path": str(attempt / "result.json"), "status": result["status"],
                         "complete": result.get("complete", False), "elapsed_seconds": result.get("elapsed_seconds")})
            print(json.dumps(_hide_secret({"question_id": question["id"], "status": result["status"],
                  "elapsed_seconds": result.get("elapsed_seconds"),
                  "http_attempts_reserved": budget.snapshot()["reserved_http_attempts"]},
                  local_config.model.api_key), ensure_ascii=False), flush=True)
        if _digest(source_database) != source_hash or _semantic_snapshot(source_database) != semantic_hash:
            raise ValueError("archived source database changed during campaign")
        report = {"schema": SCHEMA, "campaign_id": campaign["campaign_id"], "binding": binding,
                  "source_unchanged": True, "source_sha256_after": source_hash,
                  "prepared_only": bool(prepare_only), "prepared_questions": prepared,
                  "expected_attempts": len(questions), "completed_attempt_receipts": len(rows),
                  "stop_reason": stopped, "campaign_http_budget": budget.snapshot(), "rows": rows,
                  "method": {"runtime_inputs": "id/question only; no evaluator gold or score targets",
                             "learning": bool(learn), "resume": "Terminal results never rerun. Running interrupted attempts retain an absolute original six-minute deadline, including process downtime.",
                             "evaluation": "Separate evaluator; missing v2 candidate/presented diagnostics are null."}}
        _write(output / "report.json", _hide_secret(report, local_config.model.api_key))
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--protocol", type=int, choices=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), default=3)
    parser.add_argument("--max-http-attempts", type=int, default=120)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.source_database, args.output, manifest=args.manifest, question_ids=args.question_id,
                 env_file=args.env_file, protocol=args.protocol, max_http_attempts=args.max_http_attempts,
                 learn=args.learn, resume=args.resume, prepare_only=args.prepare_only)
    print(json.dumps({k: report[k] for k in ("campaign_id", "prepared_only", "completed_attempt_receipts", "expected_attempts", "stop_reason", "campaign_http_budget")}, ensure_ascii=False, indent=2))
    return 0 if args.prepare_only or report["completed_attempt_receipts"] == report["expected_attempts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
