"""Five fixed diagnostic questions with isolated learning and bounded provider use.

Keyword coverage is an evaluator-only historical comparison, not semantic truth.
``--repeat`` is an exact repeat on that question's learned clone, not a paraphrase.
Campaign resume skips committed attempts even when their status is time_budget;
it never automatically extends an already returned six-minute recall.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]

from benchmarks.run_original_five_question_benchmark import _load_manifest, _result_row
from benchmarks.support.readonly_database import ReadOnlyReplayDatabase
from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger, redact_for_answer_evidence_checkpoint, redact_for_export
from memory_demo.llm import CampaignHttpBudget, ModelClient, ProviderCallAccounting
from memory_demo.retrieval.progressive import ProgressiveRecall

DEFAULT_MANIFEST = PROJECT / "benchmarks/manifests/original_five_question_quality_benchmark.json"
SCHEMA = "progressive_five_question_diagnostic_v1"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid benchmark receipt: {path.name}")
    return value


def _hide_secret(value, secret: str):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {k: _hide_secret(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_hide_secret(v, secret) for v in value]
    return value


class _MetadataLogger(JsonlEventLogger):
    def __init__(self, path: Path, secret: str):
        super().__init__(path, answer_evidence_enabled=False)
        self._secret = secret

    def emit(self, event: str, **payload) -> None:
        super().emit(event, **_hide_secret(payload, self._secret))


@contextmanager
def _campaign_lock(output: Path):
    # Kernel locks release on process death, allowing a later --resume. The
    # campaign budget itself assumes exactly one process dispatches requests.
    with (output / ".campaign.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("another process owns this benchmark campaign") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _clone(source: Path, target: Path, expected_digest: str) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite database: {target}")
    # Only the destination uses a write connection. No MemoryApplication or
    # schema initialization ever receives the frozen source database.
    with target.open("xb"):
        pass
    with ReadOnlyReplayDatabase(source).connection() as original:
        destination = sqlite3.connect(target)
        try:
            original.backup(destination)
        finally:
            destination.close()
    if _digest(source) != expected_digest:
        raise ValueError("source database changed during cloning")


def _score_result(question: dict, result: dict, db, elapsed: float, phase: str) -> dict:
    """Evaluator-only lookup of original Episode text in delivered order."""
    episode_ids = list(dict.fromkeys(
        f["episode_id"] for f in result.get("evidence", [])
        if isinstance(f, dict) and type(f.get("episode_id")) is int
    ))[:8]
    evidence = []
    with db.connection() as connection:
        for eid in episode_ids:
            row = connection.execute("SELECT id,source_key,text FROM episode WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise ValueError("delivered Episode disappeared before evaluator scoring")
            evidence.append(dict(row))
        learned = []
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='recall_feedback_change'").fetchone():
            learned = [dict(r) for r in connection.execute(
                "SELECT association_id,before_weight,after_weight,was_created FROM recall_feedback_change WHERE feedback_id=? ORDER BY association_id",
                ("recall:" + str(result.get("session_id", "")),),
            )]
    legacy = _result_row(question, {"evidence_episodes": evidence}, elapsed)
    legacy.pop("contextual_association", None)
    quotes = list(dict.fromkeys(str(f.get("quote", "")) for f in result.get("evidence", []) if isinstance(f, dict)))
    text = "\n".join(quotes).casefold()
    terms = list(question["key_evidence_terms"])
    hits = [term for term in terms if term.casefold() in text]
    return {
        "id": question["id"], "phase": phase,
        "legacy_episode_keywords": legacy,
        "verified_quote_keywords": {"matched_terms": hits, "missing_terms": [t for t in terms if t not in hits],
                                    "coverage": {"matched": len(hits), "total": len(terms)}, "distinct_quotes": len(quotes)},
        "runtime": {"status": result.get("status"), "complete": result.get("complete", False),
                    "needs": result.get("needs", []), "missing_requirements": result.get("missing_requirements", []),
                    "metrics": result.get("metrics", {}), "elapsed_seconds": elapsed,
                    "reported_elapsed_seconds": result.get("elapsed_seconds"),
                    "learning": result.get("learning", {}), "learned_edge_changes": learned,
                    "created_edges": sum(int(row["was_created"]) for row in learned)},
        "semantic_truth": {"evaluated": False, "note": "Keyword presence and model-reported completion are not independent semantic truth judgments."},
    }


def _committed(attempt: Path) -> dict | None:
    marker = attempt / "receipt.json"
    if not marker.exists():
        return None
    receipt = _read(marker)
    files = receipt.get("files")
    if not isinstance(files, dict) or not {"result.json", "provider.json", "score.json"}.issubset(files):
        raise ValueError("incomplete attempt receipt")
    for name, digest in files.items():
        if name not in {"result.json", "provider.json", "score.json", "checkpoint.json"}:
            raise ValueError("unexpected file in attempt receipt")
        path = attempt / name
        if not path.is_file() or _digest(path) != digest:
            raise ValueError("attempt artifact missing or changed; refusing to skip it")
    return _read(attempt / "score.json")


def _recover_result(state: dict, service: ProgressiveRecall) -> dict:
    """Finish persisting an already returned terminal checkpoint without recall."""
    if state.get("version") != 2:
        raise ValueError("checkpoint uses an older review protocol; preserve this campaign and prepare a fresh one")
    service._validate_delivered_sources(state)
    needs, covered = state.get("needs", []), set(state.get("resolved_needs", []))
    return {"session_id": state["session_id"], "mode": "deep", "status": state["status"],
            "complete": state["status"] == "complete", "timeout_seconds": 360.0,
            "elapsed_seconds": state.get("elapsed_seconds", 0), "needs": needs,
            "missing_requirements": [n for i, n in enumerate(needs) if i not in covered],
            "answer": "\n".join(dict.fromkeys(f["claim"] for f in state["facts"])),
            "evidence": state["facts"], "used_edge_ids": state.get("used_edge_ids", []),
            "feedback_edge_ids": state.get("feedback_edge_ids", []), "metrics": state["metrics"],
            "learning": state["learning"], "error": state.get("error"),
            "resumable": state["status"] in {"time_budget", "cancelled", "wave_budget", "technical_error"},
            "checkpoint_path": str(service.sessions._path(state["session_id"])),
            "coverage_basis": "model_plan_and_independent_source_review_v2",
            "review_protocol_version": 2, "need_assessments": state.get("need_assessments", []),
            "pending_evidence_count": len(state.get("pending_facts", [])),
            "answer_status": "complete" if state["status"] == "complete" else "partial" if state["facts"] else "unknown"}


def _attempt(question, directory, database, config, budget, phase, *, application_factory, model_factory) -> dict:
    directory.mkdir(exist_ok=True)
    checkpoint_state = None
    local = deepcopy(config)
    local.database_path, local.log_dir = database, directory / "logs"
    app = application_factory(local)
    service = ProgressiveRecall(local, app.db, None)
    candidates = list((local.log_dir / "recall_sessions").glob("*.json"))
    if len(candidates) > 1:
        raise ValueError("an incomplete attempt has multiple sessions; inspect it before resuming")
    if candidates:
        checkpoint_state = service.sessions.read(candidates[0].stem)
        if checkpoint_state.get("question") != question["question"] or checkpoint_state.get("context") != "":
            raise ValueError("attempt checkpoint belongs to a different runtime request")
    invocation_path = directory / "invocations.json"
    invocations = _read(invocation_path).get("invocations", []) if invocation_path.exists() else []
    before = budget.snapshot()
    invocation = {"started_at": datetime.now(timezone.utc).isoformat(), "campaign_before": before,
                  "resume_session": checkpoint_state.get("session_id") if checkpoint_state else None}
    invocations.append(invocation)
    _write(invocation_path, {"invocations": invocations})
    accounting = ProviderCallAccounting()
    logger = _MetadataLogger(directory / f"provider-{len(invocations):03d}.jsonl", config.model.api_key)
    model = model_factory(deepcopy(local.model), logger, accounting=accounting, campaign_budget=budget)
    started = perf_counter()
    recovered = bool(checkpoint_state and checkpoint_state.get("status") != "running")
    if recovered:
        result = _recover_result(checkpoint_state, service)
    else:
        if int(before["remaining_http_attempts"]) <= 0:
            raise ValueError("campaign HTTP budget exhausted; no new attempt was dispatched")
        remaining = 360.0 - float(checkpoint_state.get("elapsed_seconds", 0) if checkpoint_state else 0)
        if remaining <= 0:
            checkpoint_state["status"] = "time_budget"
            result = _recover_result(checkpoint_state, service)
            recovered = True
        else:
            # The evaluator's labels, keywords and any expected answers NEVER
            # cross this boundary. Repeat deliberately uses this exact string.
            result = app.recall(str(question["question"]), mode="deep", timeout_seconds=remaining,
                                learn=True, model=model, resume=checkpoint_state["session_id"] if checkpoint_state else None)
    elapsed = perf_counter() - started
    provider = {"current_invocation": redact_for_export(accounting.snapshot()),
                "campaign_before": before, "campaign_after": budget.snapshot(),
                "campaign_attempt_reservations": int(budget.snapshot()["reserved_http_attempts"]) - int(invocations[0]["campaign_before"]["reserved_http_attempts"]),
                "recovered_terminal_checkpoint_without_recall": recovered,
                "prior_interrupted_invocations": len(invocations) - 1,
                "prior_invocation_statistics_may_be_incomplete": len(invocations) > 1,
                "model_roles": {name: getattr(local.model, name) for name in ("embedding_model", "reasoning_model", "fallback_model", "reranker_model")}}
    invocation["finished_at"] = datetime.now(timezone.utc).isoformat()
    invocation["provider"] = provider
    _write(invocation_path, {"invocations": invocations})
    score = _score_result(question, result, app.db, elapsed, phase)
    score["provider"] = provider
    score["recovery"] = {"session_id": result.get("session_id"), "checkpoint_path": result.get("checkpoint_path"),
                         "resumable": result.get("resumable", False), "campaign_resume_does_not_extend_terminal_results": True}
    files = {"result.json": result, "provider.json": provider, "score.json": score}
    checkpoint = result.get("checkpoint_path")
    if result.get("session_id"):
        path = Path(str(checkpoint)).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("runtime checkpoint must exist inside its owned attempt directory")
        files["checkpoint.json"] = _read(path)
    for filename, payload in files.items():
        _write(directory / filename, _hide_secret(redact_for_answer_evidence_checkpoint(payload), config.model.api_key))
    _write(directory / "receipt.json", {"schema": SCHEMA, "question_id": question["id"], "phase": phase,
                                         "status": result.get("status"), "files": {name: _digest(directory / name) for name in files}})
    return score


def _summary(rows: list[dict], phase: str) -> dict:
    selected = [r for r in rows if r["phase"] == phase]
    def total(field):
        return {key: sum(r[field]["coverage"][key] for r in selected) for key in ("matched", "total")}
    return {"attempts": len(selected), "model_reported_complete": sum(bool(r["runtime"]["complete"]) for r in selected),
            "legacy_episode_keywords": total("legacy_episode_keywords"), "verified_quote_keywords": total("verified_quote_keywords"),
            "http_attempt_reservations": sum(r["provider"]["campaign_attempt_reservations"] for r in selected),
            "statuses": [r["runtime"]["status"] for r in selected]}


def run(source_database: Path, output: Path, *, manifest: Path = DEFAULT_MANIFEST,
        env_file: str = ".env", repeat: bool = False, resume: bool = False,
        max_http_attempts: int = 120, prepare_only: bool = False, config: AppConfig | None = None,
        application_factory=None, model_factory=None) -> dict:
    questions = _load_manifest(Path(manifest))
    if any(not isinstance(q.get("question"), str) or not q["question"].strip()
           or not isinstance(q.get("id"), str) or not isinstance(q.get("label"), str)
           or any(not isinstance(t, str) or not t.strip() for t in q["key_evidence_terms"]) for q in questions):
        raise ValueError("invalid five-question benchmark manifest")
    if len({q["id"] for q in questions}) != 5:
        raise ValueError("benchmark question IDs must be unique")
    if type(max_http_attempts) is not int or max_http_attempts < 1:
        raise ValueError("max_http_attempts must be a positive integer")
    source_database = Path(source_database).resolve(strict=True)
    with ReadOnlyReplayDatabase(source_database).connection() as connection:
        connection.execute("SELECT COUNT(*) FROM episode").fetchone()
    source_hash = _digest(source_database)
    output = Path(output).resolve()
    if source_database.is_relative_to(output):
        raise ValueError("output must not contain the source database")
    if resume:
        if not (output / "campaign.json").is_file():
            raise ValueError("--resume requires an existing benchmark campaign")
    else:
        output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(config) if config is not None else AppConfig.from_env(env_file)
    if config.model.reasoning_max_tokens is None:
        config.model.reasoning_max_tokens = 8192
    roles = redact_for_export({name: getattr(config.model, name) for name in (
        "base_url", "embedding_model", "embedding_dimension", "reasoning_model", "fallback_model",
        "reranker_model", "reasoning_max_tokens", "reasoning_enable_thinking", "timeout_seconds", "max_retries")})
    binding = {"schema": SCHEMA, "review_protocol_version": 2,
               "source_database": str(source_database), "source_sha256": source_hash,
               "manifest_sha256": _json_hash(questions), "repeat": bool(repeat), "mode": "deep",
               "timeout_seconds": 360.0, "max_http_attempts": max_http_attempts, "model_roles": roles}
    with _campaign_lock(output):
        campaign_file = output / "campaign.json"
        if resume:
            campaign = _read(campaign_file)
            if campaign.get("binding") != binding:
                raise ValueError("resume settings, question manifest, provider roles or source database changed")
        else:
            campaign = {"campaign_id": uuid4().hex, "binding": binding, "created_at": datetime.now(timezone.utc).isoformat()}
            _write(campaign_file, campaign)
        budget = CampaignHttpBudget.open(campaign_id=campaign["campaign_id"], max_http_attempts=max_http_attempts,
                                         receipt_path=output / "http-budget.json")
        rows, stopped = [], None
        for ordinal, question in enumerate(questions, 1):
            directory = output / f"question-{ordinal:02d}"
            database = directory / "memory.sqlite"
            ready = directory / "clone.json"
            identity = {"campaign_id": campaign["campaign_id"], "question_id": question["id"], "source_sha256": source_hash}
            if directory.exists():
                if not ready.is_file() or _read(ready).get("identity") != identity or not database.is_file():
                    raise ValueError("existing question directory is not a complete clone owned by this campaign")
            else:
                directory.mkdir()
                _clone(source_database, database, source_hash)
                _write(ready, {"identity": identity, "initial_clone_sha256": _digest(database)})
            if prepare_only:
                continue
            for phase in ("first", "exact_repeat") if repeat else ("first",):
                attempt = directory / phase
                score = _committed(attempt)
                if score is None:
                    if int(budget.snapshot()["remaining_http_attempts"]) == 0 and not attempt.exists():
                        stopped = "campaign_http_budget_exhausted"
                        break
                    score = _attempt(question, attempt, database, config, budget, phase,
                                     application_factory=application_factory or MemoryApplication,
                                     model_factory=model_factory or ModelClient)
                rows.append(score)
            if stopped:
                break
        after = _digest(source_database)
        if after != source_hash:
            raise ValueError("frozen source database changed during the benchmark")
        report = {"schema": SCHEMA, "campaign_id": campaign["campaign_id"], "binding": binding,
                  "source_sha256_after": after, "source_unchanged": True, "stop_reason": stopped,
                  "prepared_only": bool(prepare_only),
                  "prepared_questions": [{"id": question["id"], "database": str(output / f"question-{ordinal:02d}" / "memory.sqlite")}
                                         for ordinal, question in enumerate(questions, 1)
                                         if (output / f"question-{ordinal:02d}" / "clone.json").is_file()],
                  "expected_attempts": 10 if repeat else 5, "completed_attempt_receipts": len(rows),
                  "method": {"runtime_inputs": "Only original question text, deep mode, six-minute budget and provider; no keywords or expected answers.",
                             "legacy_score": "First eight distinct delivered Episodes in result order, using original episode.text and the unchanged historical substring scorer.",
                             "quote_score": "Declared keywords in all distinct delivered verified Source quotes.",
                             "repeat": "exact_repeat; same question on its own learned clone, not a paraphrase",
                             "isolation": "Each question starts from an independent read-only-source clone.",
                             "truth": "Neither keyword diagnostic nor model-reported completion is independent semantic ground truth.",
                             "resume": "Committed attempts including time_budget are skipped. Terminal checkpoints finish persistence without another recall; interrupted running sessions get one bounded continuation."},
                  "summaries": {phase: _summary(rows, phase) for phase in (("first", "exact_repeat") if repeat else ("first",))},
                  "campaign_http_budget": budget.snapshot(), "rows": rows}
        _write(output / "report.json", _hide_secret(report, config.model.api_key))
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--max-http-attempts", type=int, default=120)
    parser.add_argument("--repeat", action="store_true", help="Run an exact same-question repeat in each question's learned clone")
    parser.add_argument("--resume", action="store_true", help="Continue this campaign without rerunning committed attempts, including time-budget results")
    parser.add_argument("--prepare-only", action="store_true", help="Create the campaign budget and five database clones without constructing a provider or recalling any question")
    args = parser.parse_args(argv)
    report = run(args.source_database, args.output, manifest=args.manifest, env_file=args.env_file,
                 repeat=args.repeat, resume=args.resume, max_http_attempts=args.max_http_attempts,
                 prepare_only=args.prepare_only)
    print(json.dumps({"summaries": report["summaries"], "stop_reason": report["stop_reason"],
                      "campaign_http_budget": report["campaign_http_budget"]}, ensure_ascii=False, indent=2))
    return 0 if args.prepare_only or report["completed_attempt_receipts"] == report["expected_attempts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
