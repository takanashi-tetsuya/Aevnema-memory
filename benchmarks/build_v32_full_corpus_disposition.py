"""Document the v3.2 full-corpus boundary without rewriting an old ledger.

This is a read-only disposition: it classifies the static inventory and the
four historical non-complete ledger entries, then distinguishes an available
diagnostic snapshot from the not-yet-created source-bound full snapshot.  It
does not retry extraction, alter a ledger, or contact a model provider.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


SCOPE = ("event", "main", "favor")
SCHEMA = "aevnema.v3_2.full_corpus_disposition.v1"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_hashes(source_root: Path, keys: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in keys:
        path = source_root / key
        if not path.is_file():
            rows.append({"source_key": key, "status": "missing"})
            continue
        rows.append(
            {
                "source_key": key,
                "status": "present",
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    return rows


def _task_disposition(database: Path, *, run_id: int, source_key: str) -> list[dict[str, object]]:
    """Return only failure classifications from the preserved task rows.

    Error bodies can contain transport detail that is neither necessary for a
    corpus disposition nor suitable for a shareable summary.  We retain the
    task identity and a small observed category instead.
    """

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, segment_index, stage, status, error_summary
            FROM extraction_task
            WHERE run_id = ? AND source_key = ? AND status != 'completed'
            ORDER BY segment_index, stage, id
            """,
            (run_id, source_key),
        ).fetchall()
    finally:
        connection.close()
    observations: list[dict[str, object]] = []
    for row in rows:
        error_text = str(row["error_summary"] or "")
        parsed: object = None
        try:
            parsed = json.loads(error_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping) and parsed.get("verdict") == "safe_skip":
            category = "audited_safe_skip"
            detail = {
                "verdict": parsed.get("verdict"),
                "source_kind": parsed.get("source_kind"),
                "contract_version": parsed.get("contract_version"),
            }
        elif "HTTP 400" in error_text:
            category = "provider_http_400"
            detail = {"http_status": 400}
        elif "WinError 10013" in error_text:
            category = "historical_windows_socket_permission_error"
            detail = {"network_attempt": "not_retried_by_this_disposition"}
        else:
            category = "recorded_task_failure"
            detail = {"detail": "see preserved ledger/task record"}
        observations.append(
            {
                "task_id": int(row["id"]),
                "segment_index": int(row["segment_index"]),
                "stage": str(row["stage"]),
                "status": str(row["status"]),
                "category": category,
                **detail,
            }
        )
    return observations


def _snapshot_status(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("source", "episode", "association")
        }
        evidence_basis = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT evidence_basis, COUNT(*) FROM episode GROUP BY evidence_basis"
            )
        }
        persisted_quotes = int(
            connection.execute(
                "SELECT COUNT(*) FROM episode WHERE json_array_length(evidence_quotes_json) > 0"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return {
        "path": str(path),
        "sha256": _sha(path),
        "counts": counts,
        "evidence_basis": evidence_basis,
        "episodes_with_persisted_evidence_quotes": persisted_quotes,
        "classification": (
            "K_diag_preserved_legacy_evidence_snapshot"
            if persisted_quotes == 0
            else "snapshot_requires_separate_scope_validation"
        ),
    }


def build_disposition(
    *,
    source_root: Path,
    preflight_path: Path,
    ledger_path: Path,
    historical_database: Path,
    diagnostic_snapshot: Path,
    output_dir: Path,
) -> dict[str, object]:
    source_root = source_root.resolve()
    preflight_path = preflight_path.resolve()
    ledger_path = ledger_path.resolve()
    historical_database = historical_database.resolve()
    diagnostic_snapshot = diagnostic_snapshot.resolve()
    output_dir = output_dir.resolve()
    for path in (preflight_path, ledger_path, historical_database, diagnostic_snapshot):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    preflight = _load_object(preflight_path)
    ledger = _load_object(ledger_path)
    categories = preflight.get("categories")
    categories = categories if isinstance(categories, Mapping) else {}
    scope_categories = {
        category: dict(categories.get(category, {}))
        for category in SCOPE
        if isinstance(categories.get(category), Mapping)
    }
    files = sum(int(item.get("files", 0) or 0) for item in scope_categories.values())
    segments = sum(int(item.get("segments", 0) or 0) for item in scope_categories.values())
    empty_main = [
        str(key)
        for key in preflight.get("empty_file_samples", [])
        if isinstance(key, str) and key.startswith("main/")
    ]

    raw_files = ledger.get("files")
    raw_files = raw_files if isinstance(raw_files, Mapping) else {}
    in_scope_status_counts = Counter(
        str(value.get("status", "not_observed"))
        for key, value in raw_files.items()
        if isinstance(key, str)
        and key.startswith(SCOPE)
        and isinstance(value, Mapping)
    )
    noncomplete: list[dict[str, object]] = []
    for key, value in raw_files.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        if not key.startswith(SCOPE):
            continue
        status = str(value.get("status", "not_observed"))
        if status == "completed":
            continue
        summary = value.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        try:
            run_id = int(summary.get("run_id"))
        except (TypeError, ValueError):
            run_id = -1
        task_observations = (
            _task_disposition(historical_database, run_id=run_id, source_key=key)
            if run_id >= 0
            else []
        )
        categories_seen = {str(item.get("category")) for item in task_observations}
        if categories_seen and categories_seen <= {"audited_safe_skip"}:
            disposition = "semantic_episode_not_required; preserve historical partial ledger"
            next_action = "do_not_retry_for_episode_extraction"
        elif key in empty_main:
            disposition = "metadata_only_empty_source; preserve historical failed ledger"
            next_action = "exclude_from_semantic_episode_denominator; no retry"
        else:
            disposition = "unresolved_historical_task"
            next_action = "retry only in a new source-bound rebuild after paid branch resumes"
        noncomplete.append(
            {
                "source_key": key,
                "ledger_status": status,
                "run_id": run_id if run_id >= 0 else "not_observed",
                "failure_details": summary.get("failure_details", []),
                "task_observations": task_observations,
                "disposition": disposition,
                "next_action": next_action,
            }
        )
    noncomplete.sort(key=lambda item: str(item["source_key"]))
    noncomplete_counts = Counter(str(item["ledger_status"]) for item in noncomplete)
    relevant_hash_keys = sorted({*empty_main, *(str(item["source_key"]) for item in noncomplete)})
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "formal_scoring": "not_run_unapproved_gold",
        "source_root": str(source_root),
        "static_scope": {
            "categories": list(SCOPE),
            "files": files,
            "segments": segments,
            "parse_failures": int(preflight.get("totals", {}).get("parse_failures", 0))
            if isinstance(preflight.get("totals"), Mapping)
            else "not_observed",
            "categories_detail": scope_categories,
            "preflight_artifact": str(preflight_path),
            "preflight_sha256": _sha(preflight_path),
        },
        "historical_ledger": {
            "artifact": str(ledger_path),
            "sha256": _sha(ledger_path),
            "noncomplete_count": len(noncomplete),
            "status_counts": dict(in_scope_status_counts),
            "noncomplete_status_counts": dict(noncomplete_counts),
            "entries": noncomplete,
            "history_mutated": False,
        },
        "static_empty_main_sources": {
            "count": len(empty_main),
            "source_keys": empty_main,
            "disposition": (
                "valid JSON with no usable localized story blocks; retained as a "
                "Source inventory fact but excluded from semantic Episode work"
            ),
        },
        "source_hashes": _source_hashes(source_root, relevant_hash_keys),
        "snapshots": {
            "K_old": "historical snapshots retained separately; not re-scored",
            "K_diag": _snapshot_status(diagnostic_snapshot),
            "K_full": {
                "status": "not_created",
                "reason": (
                    "No static, source-bound current-scope rebuild exists. The paid "
                    "provider branch is paused; no source or Episode rows were patched "
                    "to manufacture persisted evidence quotes."
                ),
            },
        },
        "formal_old_vs_full_comparison": "not_run; K_full and approved gold are both absent",
    }
    _write_json(output_dir / "full_corpus_disposition.full_local.json", payload)
    lines = [
        "# v3.2 full-corpus disposition",
        "",
        "This is a read-only disposition, not a re-import or a formal benchmark.",
        "",
        f"- Static authorized scope: **{files} files / {segments} segments** in event, main, and favor; parse failures: **{payload['static_scope']['parse_failures']}**.",
        f"- Historical ledger is unchanged: **{in_scope_status_counts.get('completed', 0)} completed**, **{in_scope_status_counts.get('partial', 0)} partial**, **{in_scope_status_counts.get('failed', 0)} failed** among its in-scope entries.",
        f"- `{len(empty_main)}` main JSON files are valid metadata containers with no usable localized story blocks. They are retained in inventory but excluded from semantic Episode work.",
        "",
        "## Non-complete historical entries",
        "",
        "| Source | Ledger state | Recorded task finding | Disposition |",
        "| --- | --- | --- | --- |",
    ]
    for item in noncomplete:
        findings = item["task_observations"]
        finding_text = ", ".join(
            f"segment {observation['segment_index']} {observation['stage']}: {observation['category']}"
            for observation in findings
        ) or "not_observed"
        lines.append(
            f"| `{item['source_key']}` | `{item['ledger_status']}` | {finding_text} | {item['disposition']} |"
        )
    diag = payload["snapshots"]["K_diag"]
    assert isinstance(diag, Mapping)
    lines.extend(
        [
            "",
            "## Snapshot boundary",
            "",
            f"The preserved K_diag snapshot has `{diag['counts']['source']}` Sources and `{diag['counts']['episode']}` Episodes, but `{diag['episodes_with_persisted_evidence_quotes']}` Episodes with persisted evidence quotes. It is therefore a legacy-evidence diagnostic snapshot, not K_full.",
            "",
            "K_full has not been created. The current paid-provider pause blocks a new source-bound rebuild; no historical ledger, Source row, or Episode was altered to make that absence look complete. Formal old-vs-full comparison remains not run because K_full and approved independent gold are absent.",
            "",
        ]
    )
    (output_dir / "FULL_CORPUS_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--historical-database", type=Path, required=True)
    parser.add_argument("--diagnostic-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_disposition(
        source_root=args.source_root,
        preflight_path=args.preflight,
        ledger_path=args.ledger,
        historical_database=args.historical_database,
        diagnostic_snapshot=args.diagnostic_snapshot,
        output_dir=args.output_dir,
    )
    print(json.dumps({"schema": result["schema"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
