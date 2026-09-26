"""Offline evaluation of the frozen historical 15-question / 57-slot protocol.

No runtime import or model calls. Only Source-verified result.evidence supplies
the primary score. Candidate and presented IDs remain diagnostic layers.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3


class IdentityMismatch(ValueError):
    """Historical Episode IDs cannot safely be scored against this snapshot."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze_historical_manifest(audit: dict, *, audit_sha256: str, identity: dict) -> tuple[list[dict], dict]:
    if audit.get("passed") is not True or audit.get("audit_errors") or audit.get("historical_run_errors"):
        raise ValueError("historical audit must have passed without errors")
    if identity.get("passed") is not True or not identity.get("semantic_snapshot_sha256"):
        raise IdentityMismatch("freeze requires a confirmed semantic snapshot identity")
    questions, bindings = [], {}
    for row in audit["questions"]:
        groups = [{k: deepcopy(v) for k, v in group.items() if k not in ("matches", "passed")}
                  for group in row["layers"]["final"]["groups"]]
        questions.append({"id": row["id"], "question": row["question"], "evidence_groups": groups})
        for binding in row["logged_final_episode_bindings"]:
            eid = binding["episode_id"]
            if eid in bindings and bindings[eid] != binding:
                raise ValueError("conflicting historical Episode text bindings")
            bindings[eid] = deepcopy(binding)
    report = next(item for item in audit["input_files"] if Path(item["path"]).name == "report.json")
    manifest = {
        "schema": "historical-full-recall-evaluation-v1", "historical_run_id": audit["historical_run_id"],
        "historical_audit_sha256": audit_sha256, "historical_report_sha256": report["sha256"],
        "expected_semantic_snapshot_sha256": identity["semantic_snapshot_sha256"],
        "question_count": 15, "required_fact_slots": 57, "verified_episode_limit": 30,
        "questions": questions, "logged_episode_bindings": [bindings[eid] for eid in sorted(bindings)],
        "limits": ["OR within each original alternatives group; all 57 groups are unchanged.",
                   "Historical slot matching is not independent semantic answer grading.",
                   "Some alternatives lack a historical logged text: their identity is covered by the confirmed full semantic snapshot, not a fabricated old text comparison."],
    }
    manifest["manifest_sha256"] = _digest(manifest)
    _validate_manifest(manifest)
    runtime = [{"id": row["id"], "question": row["question"]} for row in questions]
    return runtime, manifest


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != "historical-full-recall-evaluation-v1":
        raise ValueError("unknown historical evaluation schema")
    if manifest.get("manifest_sha256") != _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("evaluation manifest checksum mismatch")
    questions = manifest.get("questions", [])
    if len(questions) != 15 or len({q["id"] for q in questions}) != 15 or sum(len(q["evidence_groups"]) for q in questions) != 57:
        raise ValueError("the original 15 questions and 57 slots must be preserved")
    if manifest.get("question_count") != 15 or manifest.get("required_fact_slots") != 57 or manifest.get("verified_episode_limit") != 30:
        raise ValueError("historical question/slot/verified limits changed")
    for question in questions:
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError("invalid frozen question")
        groups = question["evidence_groups"]
        if len({g["id"] for g in groups}) != len(groups):
            raise ValueError("duplicate fact slot")
        for group in groups:
            if not group.get("alternatives") or any(type(eid) is not int or eid < 0 for eid in group["alternatives"]):
                raise ValueError("invalid historical alternative Episode IDs")


def read_semantic_snapshot(database_path: str | Path) -> dict:
    """Recompute actual content, ignoring schema-only and association changes."""
    path = Path(database_path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        source_rows = connection.execute("SELECT id,raw_text FROM source ORDER BY id").fetchall()
        episode_rows = connection.execute("SELECT id,source_id,source_key,segment_index,text FROM episode ORDER BY id").fetchall()
    sources = {}
    for sid, text in source_rows:
        if type(sid) is not int or not isinstance(text, str):
            raise IdentityMismatch("invalid Source identity fields")
        sources[sid] = _sha(text)
    episodes = {}
    for eid, sid, key, segment, text in episode_rows:
        if type(eid) is not int or sid not in sources or not isinstance(text, str):
            raise IdentityMismatch("invalid Episode identity or missing Source")
        episodes[eid] = {"episode_id": eid, "source_id": sid, "source_key": key,
                         "segment_index": segment, "text_sha256": _sha(text)}
    payload = {"schema": "historical-source-episode-v1",
               "sources": [[sid, digest] for sid, digest in sources.items()],
               "episodes": [[eid, row["source_id"], row["source_key"], row["segment_index"], row["text_sha256"]]
                            for eid, row in episodes.items()]}
    return {"database_path": str(path), "semantic_snapshot_sha256": _digest(payload),
            "source_count": len(sources), "episode_count": len(episodes), "episodes": episodes}


def verify_identity(manifest: dict, identity: dict | None, database_path: str | Path) -> dict:
    _validate_manifest(manifest)
    if not isinstance(identity, dict) or identity.get("passed") is not True or not identity.get("semantic_snapshot_sha256"):
        raise IdentityMismatch("a confirmed historical semantic snapshot identity is required")
    if identity.get("historical_run_id") != manifest["historical_run_id"] or identity.get("input_report", {}).get("sha256") != manifest["historical_report_sha256"]:
        raise IdentityMismatch("identity belongs to a different historical report")
    if not manifest.get("expected_semantic_snapshot_sha256") or identity["semantic_snapshot_sha256"] != manifest["expected_semantic_snapshot_sha256"]:
        raise IdentityMismatch("identity differs from the pre-registered semantic snapshot")
    snapshot = read_semantic_snapshot(database_path)
    if snapshot["semantic_snapshot_sha256"] != identity["semantic_snapshot_sha256"]:
        raise IdentityMismatch("actual SQLite Source/Episode content differs from the confirmed snapshot")
    bindings = manifest.get("logged_episode_bindings", [])
    if not bindings:
        raise IdentityMismatch("historical logged Episode text bindings are missing")
    for binding in bindings:
        row = snapshot["episodes"].get(binding["episode_id"])
        if row is None or any(row[field] != binding[field] for field in ("source_key", "segment_index")) or row["text_sha256"] != binding["logged_episode_text_sha256"]:
            raise IdentityMismatch(f"historical Episode {binding['episode_id']} does not match its logged text/source binding")
    alternatives = {eid for question in manifest["questions"] for group in question["evidence_groups"] for eid in group["alternatives"]}
    if alternatives - snapshot["episodes"].keys():
        raise IdentityMismatch("historical alternative Episode IDs are absent from the actual database")
    return snapshot


def score_groups(item: dict, final_ids: set[int]) -> list[dict]:
    """Independent equivalent of the unchanged original _score_groups."""
    groups = []
    for group in item["evidence_groups"]:
        alternatives = {int(value) for value in group["alternatives"]}
        matches = sorted(final_ids & alternatives)
        groups.append({**group, "matches": matches, "passed": bool(matches)})
    return groups


def _unique_ids(values: object, *, limit: int | None = None) -> list[int]:
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Episode IDs must be a list of nonnegative integers")
    unique = list(dict.fromkeys(values))
    return unique if limit is None else unique[:limit]


def verified_episode_ids(result: dict) -> list[int]:
    evidence = result.get("evidence", [])
    if not isinstance(evidence, list) or any(not isinstance(item, dict) or "episode_id" not in item for item in evidence):
        raise ValueError("runtime evidence must contain structured Episode references")
    return _unique_ids([item["episode_id"] for item in evidence], limit=30)


def _layer(question: dict, ids: list[int] | None) -> dict:
    if ids is None:
        return {"available": False, "episode_ids": None, "episode_count": None,
                "matched_fact_slots": None, "required_fact_slots": len(question["evidence_groups"]),
                "complete": None, "groups": None}
    groups = score_groups(question, set(ids))
    matched = sum(group["passed"] for group in groups)
    return {"available": True, "episode_ids": ids, "episode_count": len(ids), "matched_fact_slots": matched,
            "required_fact_slots": len(groups), "complete": matched == len(groups), "groups": groups}


def evaluate_run(manifest: dict, results_by_id: dict[str, dict], *, identity: dict | None,
                 database_path: str | Path | Mapping[str, str | Path], candidate_episode_ids: dict[str, list[int]] | None = None,
                 presented_episode_ids: dict[str, list[int]] | None = None) -> dict:
    _validate_manifest(manifest)
    if not isinstance(results_by_id, dict):
        raise ValueError("results must be keyed by original question ID")
    expected = {question["id"] for question in manifest["questions"]}
    if results_by_id.keys() - expected:
        raise ValueError("unknown question IDs cannot enter the original suite")
    if isinstance(database_path, Mapping):
        if results_by_id.keys() - database_path.keys():
            raise IdentityMismatch("a database path is required for every result")
        paths = {qid: Path(database_path[qid]).resolve(strict=True) for qid in results_by_id}
        if not paths:
            raise IdentityMismatch("an empty run still needs an explicit reference database path")
    else:
        path = Path(database_path).resolve(strict=True)
        paths = {qid: path for qid in results_by_id}
    snapshots = {str(path): verify_identity(manifest, identity, path)
                 for path in dict.fromkeys(paths.values() if paths else [Path(database_path).resolve(strict=True)])}
    reference = next(iter(snapshots.values()))
    rows, technical_errors, interrupted = [], [], []
    for question in manifest["questions"]:
        qid = question["id"]
        if qid not in results_by_id:
            continue
        result = results_by_id[qid]
        snapshot = snapshots[str(paths[qid])]
        if not isinstance(result, dict):
            raise ValueError("each runtime result must be a JSON object")
        verified = verified_episode_ids(result)
        candidate = (candidate_episode_ids or {}).get(qid, result.get("candidate_episode_ids"))
        presented = (presented_episode_ids or {}).get(qid, result.get("presented_episode_ids"))
        candidate = None if candidate is None else _unique_ids(candidate)
        presented = None if presented is None else _unique_ids(presented)
        if (set(verified) | set(candidate or []) | set(presented or [])) - snapshot["episodes"].keys():
            raise IdentityMismatch(f"{qid}: result contains IDs outside the verified database")
        status = result.get("status", "missing_status")
        if result.get("error") or status not in {"complete", "search_exhausted", "time_budget", "cancelled", "wave_budget", "running"}:
            technical_errors.append({"id": qid, "status": status, "error": result.get("error")})
        if status in {"time_budget", "cancelled", "wave_budget", "running"}:
            interrupted.append({"id": qid, "status": status})
        assessments = result.get("need_assessments", [])
        needs = result.get("needs", [])
        resolved_indices = {item.get("need_index") for item in assessments if isinstance(item, dict)
                            and item.get("status") in {"supported", "refuted"}}
        row = {"id": qid, "question": question["question"], "runtime_status": status,
               "candidate": _layer(question, candidate), "presented": _layer(question, presented),
               "verified": _layer(question, verified),
               "need_resolved": {"runtime_complete": result.get("complete") is True,
                                 "reported_resolved_count": len(resolved_indices), "reported_need_count": len(needs),
                                 "assessments": deepcopy(assessments)},
               "metrics": deepcopy(result.get("metrics", {})),
               "evidence_limit": {"unique_before_limit": len(_unique_ids([item["episode_id"] for item in result.get("evidence", [])])),
                                  "limit": 30, "selection": "first occurrence order, unique Episode IDs"}}
        rows.append(row)
    summaries = {}
    observed_slots = sum(row["verified"]["required_fact_slots"] for row in rows)
    for layer in ("candidate", "presented", "verified"):
        available = [row for row in rows if row[layer]["available"]]
        known_slots = sum(row[layer]["required_fact_slots"] for row in available)
        matched = sum(row[layer]["matched_fact_slots"] for row in available)
        summaries[layer] = {"available_questions": len(available), "unavailable_questions": len(rows) - len(available),
                            "matched_fact_slots": matched if available or layer == "verified" else None,
                            "required_fact_slots": 57, "observed_required_fact_slots": known_slots,
                            "fact_slot_recall": matched / 57 if len(available) == len(rows) else None,
                            "observed_fact_slot_recall": matched / known_slots if known_slots else None,
                            "complete_questions": sum(row[layer]["complete"] for row in available)}
    full_suite = set(results_by_id) == expected
    full_recall = full_suite and summaries["verified"]["matched_fact_slots"] == 57 and not technical_errors and not interrupted
    return {"schema": "historical-full-recall-result-v1", "manifest_sha256": manifest["manifest_sha256"],
            "historical_run_id": manifest["historical_run_id"],
            "actual_semantic_snapshot_sha256": reference["semantic_snapshot_sha256"],
            "database_paths": {qid: str(path) for qid, path in paths.items()},
            "verified_database_count": len(snapshots), "identity_recomputed_readonly": True,
            "full_suite": full_suite, "observed_questions": len(rows), "required_questions": 15,
            "missing_question_ids": [q["id"] for q in manifest["questions"] if q["id"] not in results_by_id],
            "full_recall": bool(full_recall), "technical_errors": technical_errors,
            "interrupted_runs": interrupted, "summary": summaries, "questions": rows,
            "meaning": "full_recall requires verified@30 57/57 over all 15 frozen questions, without technical errors or interrupted runs; candidate/presented/need resolution cannot replace it"}


def evaluate_campaign(manifest: dict, identity: dict, campaign_dir: str | Path, *, phase: str = "first") -> dict:
    """Read terminal receipt-bound results; the runtime runner never scores."""
    directory = Path(campaign_dir).resolve(strict=True)
    report = _read(directory / "report.json")
    if report.get("binding", {}).get("source_semantic_sha256") != manifest["expected_semantic_snapshot_sha256"]:
        raise IdentityMismatch("campaign uses another Source/Episode snapshot")
    frozen = {question["id"]: question["question"] for question in manifest["questions"]}
    for question in report["binding"]["runtime_questions"]:
        if set(question) != {"id", "question"} or frozen.get(question["id"]) != question["question"]:
            raise ValueError("campaign runtime question differs from the frozen original")
    prepared = {item["id"]: item for item in report["prepared_questions"]}
    results, databases = {}, {}
    for row in report["rows"]:
        if row["phase"] != phase:
            continue
        qid = row["id"]
        if qid in results or qid not in prepared:
            raise ValueError("duplicate or unprepared campaign result")
        database = Path(row["database"]).resolve(strict=True)
        result_path = Path(row["result_path"]).resolve(strict=True)
        if result_path.name != "result.json" or not result_path.is_relative_to(directory) or not database.is_relative_to(directory) or database != Path(prepared[qid]["database"]).resolve(strict=True):
            raise ValueError("campaign artifact path is not its prepared local clone")
        receipt = _read(result_path.parent / "receipt.json")
        binding = receipt.get("identity", {})
        if binding.get("campaign_id") != report["campaign_id"] or binding.get("question_id") != qid:
            raise ValueError("result receipt belongs to another campaign/question")
        files = receipt.get("files")
        if not isinstance(files, dict) or not {"result.json", "provider.json", "checkpoint.json", "invocations.json"} <= files.keys():
            raise ValueError("result receipt is incomplete")
        for name, digest in files.items():
            path = (result_path.parent / name).resolve(strict=True)
            if not path.is_relative_to(result_path.parent) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("receipt-bound artifact changed")
        result = _read(result_path)
        if result.get("status") == "running":
            raise ValueError("uncommitted running result cannot be evaluated")
        results[qid], databases[qid] = result, database
    evaluated = evaluate_run(manifest, results, identity=identity,
                             database_path=databases or identity["matched_database"]["path"])
    evaluated["campaign_dir"], evaluated["phase"] = str(directory), phase
    evaluated["result_receipts_verified"] = len(results)
    return evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--audit", type=Path, required=True)
    prepare.add_argument("--identity", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score")
    for name in ("manifest", "identity", "output"):
        score.add_argument("--" + name, type=Path, required=True)
    inputs = score.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--results", type=Path)
    inputs.add_argument("--campaign-dir", type=Path)
    score.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        identity = _read(args.identity)
        runtime, manifest = freeze_historical_manifest(_read(args.audit), audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(), identity=identity)
        verify_identity(manifest, identity, identity["matched_database"]["path"])
        _write(args.output_dir / "runtime-questions.json", runtime)
        _write(args.output_dir / "evaluation-manifest.json", manifest)
        print(json.dumps({"questions": len(runtime), "slots": 57, "manifest_sha256": manifest["manifest_sha256"]}))
    else:
        if args.campaign_dir:
            result = evaluate_campaign(_read(args.manifest), _read(args.identity), args.campaign_dir)
        else:
            if args.database is None:
                parser.error("--database is required with --results")
            result = evaluate_run(_read(args.manifest), _read(args.results), identity=_read(args.identity), database_path=args.database)
        _write(args.output, result)
        print(json.dumps({"full_suite": result["full_suite"], "full_recall": result["full_recall"], "verified": result["summary"]["verified"]}))


if __name__ == "__main__":
    main()
