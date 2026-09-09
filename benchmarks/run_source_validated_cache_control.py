"""Run the v3.2 B3 ordinary source-validated evidence-cache control.

The control consumes a preserved *legal Q1* trace whose Q1 and Q2 have the
same normalized request.  It seeds a benchmark-only cache from the Q1's
already selected EvidenceRef IDs, then re-opens the frozen database and
revalidates those IDs, Source bytes, persisted quotes, and excerpt budget.
It does not call ``QueryEngine`` for a shortcut, read association tables, or
store answer prose.  Existing B0/B1/B4/B4M observations are cited exactly as
historical observations, never rerun or upgraded into a matched live pair.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Mapping

from benchmarks.control_cache import (
    SourceValidatedCacheMiss,
    SourceValidatedEvidenceCache,
)
from benchmarks.run_q1_q2_diagnostic_pilot import _clone_sqlite, _sha256_file, _write_json
from memory_demo.database import Database
from memory_demo.embeddings import normalize_query_text


SCHEMA = "aevnema.v3_2.source_validated_cache_control_run.v1"
FULL_LOCAL_NAME = "source_validated_cache_control.full_local.json"
REPORT_NAME = "CACHE_AND_LATENCY_RESULTS.md"
PAIR_NAME = "paired_latency_results.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _counts(value: object) -> dict[str, int | str]:
    if not isinstance(value, Mapping):
        return {"status": "not_observed"}
    return {
        key: int(value.get(key, 0) or 0)
        for key in ("logical_batches", "http_attempts", "sent", "succeeded", "failed_or_rejected", "fallbacks")
    }


def _q1_evidence_ids(trace: Mapping[str, object]) -> tuple[str, str, str, list[int]]:
    case = trace.get("case")
    q1 = trace.get("q1")
    if not isinstance(case, Mapping) or not isinstance(q1, Mapping):
        raise ValueError("Q1/Q2 trace lacks case or Q1 record")
    q1_text = str(case.get("q1_text", ""))
    q2_text = str(case.get("q2_text", ""))
    if normalize_query_text(q1_text) != normalize_query_text(q2_text):
        raise ValueError("B3 exact control requires equal normalized Q1/Q2 text")
    record = q1.get("record")
    result = record.get("result") if isinstance(record, Mapping) else None
    episodes = result.get("evidence_episodes") if isinstance(result, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("Q1 trace lacks selected evidence episodes")
    ids = [int(item["id"]) for item in episodes if isinstance(item, Mapping) and "id" in item]
    if not ids:
        raise ValueError("Q1 trace lacks a usable EvidenceRef")
    return (
        q2_text,
        str(case.get("domain", "")),
        str(case.get("scope_hash", "")),
        list(dict.fromkeys(ids)),
    )


def _historical_arm(trace: Mapping[str, object], name: str) -> dict[str, object]:
    arms = trace.get("arms")
    arm = arms.get(name) if isinstance(arms, Mapping) else None
    if not isinstance(arm, Mapping):
        return {"status": "not_observed"}
    summary = arm.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    return {
        "status": str(summary.get("run_status", "not_observed")),
        "elapsed_ms": summary.get("elapsed_ms"),
        "delivered_episode_ids": summary.get("delivered_episode_ids", []),
        "delivered_coverage": summary.get("delivered_coverage"),
        "provider_counts": _counts(summary.get("provider_counts")),
        "execution_modules": summary.get("exact_revisit_executed_modules", {}),
        "historical_arm": name,
    }


def _ordinary_baselines(trace: Mapping[str, object]) -> dict[str, object]:
    arms = trace.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    result: dict[str, object] = {}
    for name in ("ordinary_search", "ordinary_query_embedding_cache"):
        arm = arms.get(name)
        arm = arm if isinstance(arm, Mapping) else {}
        summary = arm.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        result[name] = {
            "status": str(summary.get("run_status", arm.get("status", "not_observed"))),
            "evidence_acquisition_ms": summary.get("evidence_acquisition_ms"),
            "delivered_episode_ids": summary.get("delivered_episode_ids", []),
            "provider_counts": _counts(arm.get("provider_counts")),
            "execution_mode": arm.get("execution_mode", "not_observed"),
            "observation_only": True,
        }
    return result


def _source_content_fingerprint(database: Path, episode_ids: list[int]) -> dict[str, object]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = []
        for episode_id in episode_ids:
            row = connection.execute(
                """
                SELECT e.id, e.source_key, e.segment_index, s.raw_text
                FROM episode AS e JOIN source AS s ON s.id = e.source_id
                WHERE e.id = ?
                """,
                (episode_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"episode {episode_id} missing from source snapshot")
            rows.append(
                {
                    "episode_id": int(row[0]),
                    "source_key": str(row[1]),
                    "segment_index": int(row[2]),
                    "source_raw_sha256": "sha256:" + sha256(str(row[3]).encode("utf-8")).hexdigest(),
                }
            )
        return {"episodes": rows}
    finally:
        connection.close()


def _snapshot_sidecars(database: Path) -> dict[str, int]:
    return {
        suffix: int(path.stat().st_size) if path.exists() else 0
        for suffix in ("-wal", "-shm")
        for path in (database.with_name(database.name + suffix),)
    }


def _mutate_only_control_clone(database: Path, *, episode_ids: list[int]) -> None:
    """Invalidate a clone without touching a frozen source or learned snapshot."""

    # This UPDATE fires the Source FTS triggers.  It must use the project
    # connection factory so the clone receives the same SQLite UDF contract
    # as an ordinary application write; a bare sqlite3 connection would fail
    # before the invalidation assertion could be observed.
    clone_database = Database(database)
    with clone_database.transaction() as connection:
        placeholders = ",".join("?" for _ in episode_ids)
        connection.execute(
            "UPDATE source SET raw_text = raw_text || ? "
            "WHERE id IN (SELECT source_id FROM episode WHERE id IN (" + placeholders + "))",
            ("\n[control-cache-source-revision-change]", *episode_ids),
        )


def _render_report(path: Path, payload: Mapping[str, object]) -> None:
    controls = payload.get("controls")
    controls = controls if isinstance(controls, Mapping) else {}
    lines = [
        "# B3 source-validated evidence-cache control",
        "",
        "Diagnostic only: no answer cache, no formal gold, no promotion, and no canary.",
        "",
        "| Arm | Status | Delivered episodes | HTTP | Evidence time (ms) | Interpretation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    order = ("B0_ordinary_search", "B1_request_local_vector_reuse", "B2_historical_plan_cache", "B3_source_validated_evidence_cache", "B4_exact_edge_reuse", "B4M_exact_edge_mask")
    for name in order:
        item = controls.get(name)
        item = item if isinstance(item, Mapping) else {}
        counts = item.get("provider_counts")
        counts = counts if isinstance(counts, Mapping) else {}
        lines.append(
            "| {name} | {status} | {episodes} | {http} | {elapsed} | {note} |".format(
                name=name,
                status=item.get("status", "not_observed"),
                episodes=item.get("delivered_episode_ids", []),
                http=counts.get("http_attempts", 0),
                elapsed=item.get("evidence_acquisition_ms", item.get("elapsed_ms", "not_observed")),
                note=item.get("comparison_status", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "B3 stores only Q1 EvidenceRef identities and exact request/source/budget dependencies. Every B3 read reopens the current SQLite data and revalidates Source bytes, persisted quote mapping, and the bounded original excerpt; it does not read association, receipt, manifest, cue, or answer state.",
            "",
            "The historical B4M arm is an exact-only masked miss and did not run ordinary fallback. It is therefore retained as an entry/mask observation, not presented as a fair B4-vs-B4M end-to-end latency comparison. B0/B1 belong to the preserved ordinary-baseline trace and have a different snapshot identity; matching Source-byte dependencies are recorded, but no cross-trace timing ratio is claimed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_control(
    *,
    learned_snapshot: Path,
    q1_q2_trace: Path,
    ordinary_baseline_trace: Path,
    output_dir: Path,
    source_excerpt_chars: int = 3000,
) -> dict[str, object]:
    learned_snapshot = learned_snapshot.resolve()
    q1_q2_trace = q1_q2_trace.resolve()
    ordinary_baseline_trace = ordinary_baseline_trace.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("control output directory must be new")
    if not all(path.is_file() for path in (learned_snapshot, q1_q2_trace, ordinary_baseline_trace)):
        raise FileNotFoundError("learned snapshot and both preserved traces are required")
    sidecars = _snapshot_sidecars(learned_snapshot)
    # A zero-byte WAL is not state; some preserved historical snapshots keep
    # one alongside the immutable database file.  A nonempty WAL would make
    # the file-only SHA insufficient as a frozen binding, so do not consume
    # it silently.
    if sidecars["-wal"]:
        raise ValueError("learned snapshot has a nonempty WAL sidecar")
    q1_q2 = _read_object(q1_q2_trace)
    ordinary_trace = _read_object(ordinary_baseline_trace)
    question, domain, scope_hash, episode_ids = _q1_evidence_ids(q1_q2)
    ordinary_case = ordinary_trace.get("case")
    if not isinstance(ordinary_case, Mapping):
        raise ValueError("ordinary baseline trace lacks case")
    if normalize_query_text(str(ordinary_case.get("q2_text", ""))) != normalize_query_text(question):
        raise ValueError("ordinary baseline and Q1/Q2 cache control use different Q2 text")
    output_dir.mkdir(parents=True)
    started = _utc_now()
    seed_started = perf_counter()
    cache = SourceValidatedEvidenceCache.seed(
        database=learned_snapshot,
        question=question,
        domain=domain,
        scope_hash=scope_hash,
        episode_ids=episode_ids,
        source_excerpt_chars=source_excerpt_chars,
    )
    seed_ms = round((perf_counter() - seed_started) * 1000.0, 3)
    cache_payload = cache.as_dict()
    _write_json(output_dir / "source_validated_cache_control.json", cache_payload)
    replay_started = perf_counter()
    replay = cache.replay(
        database=learned_snapshot,
        question=question,
        domain=domain,
        scope_hash=scope_hash,
        source_excerpt_chars=source_excerpt_chars,
    )
    replay_ms = round((perf_counter() - replay_started) * 1000.0, 3)
    changed_clone = output_dir / "work" / "source_changed_control.sqlite"
    _clone_sqlite(learned_snapshot, changed_clone)
    _mutate_only_control_clone(changed_clone, episode_ids=episode_ids)
    invalidation_started = perf_counter()
    try:
        cache.replay(
            database=changed_clone,
            question=question,
            domain=domain,
            scope_hash=scope_hash,
            source_excerpt_chars=source_excerpt_chars,
        )
        invalidation: dict[str, object] = {"status": "unexpected_hit"}
    except SourceValidatedCacheMiss as error:
        invalidation = {"status": "rejected", "reason": error.reason}
    invalidation["elapsed_ms"] = round((perf_counter() - invalidation_started) * 1000.0, 3)
    historical_b4 = _historical_arm(q1_q2, "learning_ready_q2")
    historical_b4m = _historical_arm(q1_q2, "edge_masked_q2")
    ordinary = _ordinary_baselines(ordinary_trace)
    b3_refs = replay.get("materialized_source_refs")
    b3_refs = b3_refs if isinstance(b3_refs, list) else []
    controls: dict[str, object] = {
        "B0_ordinary_search": {
            **dict(ordinary["ordinary_search"]),
            "comparison_status": "historical_observation_not_retimed",
        },
        "B1_request_local_vector_reuse": {
            **dict(ordinary["ordinary_query_embedding_cache"]),
            "comparison_status": "request_local_vector_reuse_not_cross_request_evidence_cache",
        },
        "B2_historical_plan_cache": {
            "status": "not_observed_independent_control",
            "comparison_status": "not_run",
        },
        "B3_source_validated_evidence_cache": {
            "status": str(replay.get("cache_status", "not_observed")),
            "evidence_acquisition_ms": replay_ms,
            "cache_seed_source_validation_ms": seed_ms,
            "delivered_episode_ids": [
                item.get("identity", {}).get("episode_id")
                for item in b3_refs if isinstance(item, Mapping) and isinstance(item.get("identity"), Mapping)
            ],
            "provider_counts": _counts({}),
            "executed_modules": replay.get("executed_modules", {}),
            "comparison_status": "actual_local_source_validated_cache_read",
        },
        "B4_exact_edge_reuse": {
            **historical_b4,
            "comparison_status": "historical_exact_edge_observation_not_retimed",
        },
        "B4M_exact_edge_mask": {
            **historical_b4m,
            "comparison_status": "exact_only_mask_miss_no_ordinary_fallback_observed",
        },
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "diagnostic_complete_non_scorable",
        "started_at": started,
        "finished_at": _utc_now(),
        "formal_scoring": {
            "status": "not_scored_unapproved_gold",
            "gold_loaded": False,
            "promotion_prohibited": True,
            "canary_prohibited": True,
        },
        "input": {
            "q1_q2_trace": {"path": str(q1_q2_trace), "sha256": _sha256_file(q1_q2_trace)},
            "ordinary_baseline_trace": {"path": str(ordinary_baseline_trace), "sha256": _sha256_file(ordinary_baseline_trace)},
            "learned_snapshot": {"path": str(learned_snapshot), "sha256": _sha256_file(learned_snapshot)},
            "learned_snapshot_sidecars": sidecars,
            "question": question,
            "domain": domain,
            "scope_hash": scope_hash,
            "q1_evidence_ref_episode_ids": episode_ids,
            "source_excerpt_chars": source_excerpt_chars,
            "source_content_dependencies": _source_content_fingerprint(learned_snapshot, episode_ids),
        },
        "cache_file": {
            "path": str((output_dir / "source_validated_cache_control.json").resolve()),
            "sha256": _sha256_file(output_dir / "source_validated_cache_control.json"),
            "answer_prose_stored": False,
            "association_state_stored": False,
        },
        "controls": controls,
        "source_change_invalidation": invalidation,
        "fairness_boundary": {
            "B3_reads_association_tables": False,
            "B3_uses_answer_cache": False,
            "B3_revalidates_scope_source_and_budget": True,
            "B4M_end_to_end_fallback_observed": False,
            "timing_ratio_claimed": False,
        },
    }
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_json(output_dir / PAIR_NAME, {"schema": SCHEMA, "controls": controls, "fairness_boundary": payload["fairness_boundary"]})
    _render_report(output_dir / REPORT_NAME, payload)
    return {
        "output_dir": str(output_dir),
        "full_local": str(output_dir / FULL_LOCAL_NAME),
        "cache_control": str(output_dir / "source_validated_cache_control.json"),
        "report": str(output_dir / REPORT_NAME),
        "b3_status": controls["B3_source_validated_evidence_cache"]["status"],
        "source_change": invalidation["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--q1-q2-trace", type=Path, required=True)
    parser.add_argument("--ordinary-baseline-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-excerpt-chars", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(run_control(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
