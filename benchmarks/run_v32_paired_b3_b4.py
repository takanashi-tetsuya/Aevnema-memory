"""Measure B3 and B4 locally against the same frozen N11d snapshot.

This is a local retrieval timing control, not an answer benchmark.  B3 uses
the ordinary source-validated cache; B4 uses only the public automatic exact
revisit API.  Setup/index construction is measured separately and excluded
from operation timings for both arms.  Every individual repetition is kept.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Mapping

from benchmarks.control_cache import SourceValidatedEvidenceCache
from benchmarks.run_q1_q2_diagnostic_pilot import ProviderLedger, _clone_sqlite, _open_app, _sha256_file, _trace_context, _write_json
from benchmarks.run_source_validated_cache_control import _q1_evidence_ids, _read_object


SCHEMA = "aevnema.v3_2.paired_b3_b4_local.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min_ms": round(min(values), 3),
        "median_ms": round(median(values), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def run(
    *,
    learned_snapshot: Path,
    q1_q2_trace: Path,
    output_dir: Path,
    env_file: Path,
    repetitions: int = 15,
    source_excerpt_chars: int = 3000,
) -> dict[str, str]:
    learned_snapshot = learned_snapshot.resolve()
    q1_q2_trace = q1_q2_trace.resolve()
    output_dir = output_dir.resolve()
    env_file = env_file.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    if repetitions < 3:
        raise ValueError("at least three paired repetitions are required")
    # A nonempty WAL changes SQLite's logical state and therefore cannot be
    # ignored. A preserved SHM file is merely shared-memory bookkeeping and
    # is not part of the immutable database bytes; the existing B3 control
    # already accepts that harmless sidecar.
    wal = learned_snapshot.with_name(learned_snapshot.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("frozen snapshot has nonempty WAL sidecar")
    trace = _read_object(q1_q2_trace)
    question, domain, scope_hash, episode_ids = _q1_evidence_ids(trace)
    snapshot_hash = _sha256_file(learned_snapshot)
    output_dir.mkdir(parents=True)
    seed_started = perf_counter()
    cache = SourceValidatedEvidenceCache.seed(
        database=learned_snapshot,
        question=question,
        domain=domain,
        scope_hash=scope_hash,
        episode_ids=episode_ids,
        source_excerpt_chars=source_excerpt_chars,
    )
    b3_seed_ms = round((perf_counter() - seed_started) * 1000.0, 3)
    cache_path = output_dir / "b3_seed.source_validated_cache.json"
    _write_json(cache_path, cache.as_dict())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "paid_provider_calls_issued": 0,
        "formal_scoring": "disabled_unapproved_gold",
        "input": {
            "snapshot": {"path": str(learned_snapshot), "sha256": snapshot_hash},
            "q1_q2_trace": {"path": str(q1_q2_trace), "sha256": _sha256_file(q1_q2_trace)},
            "question": question,
            "domain": domain,
            "scope_hash": scope_hash,
            "episode_ids": episode_ids,
            "source_excerpt_chars": source_excerpt_chars,
            "b3_seed_ms_excluded_from_operation_timing": b3_seed_ms,
        },
        "fairness_contract": {
            "same_snapshot": True,
            "same_question_domain_scope_and_excerpt_budget": True,
            "b3_reads_association_state": False,
            "b3_stores_answer_prose": False,
            "b4_entry": "public QueryEngine.try_contextual_revisit",
            "b4_does_not_run_ordinary_fallback": True,
            "setup_and_index_build_excluded": True,
            "all_repetitions_retained": True,
        },
        "repetitions": [],
    }
    for index in range(repetitions):
        order = ("B3", "B4") if index % 2 == 0 else ("B4", "B3")
        pair: dict[str, Any] = {"index": index + 1, "order": list(order)}
        for arm in order:
            if arm == "B3":
                started = perf_counter()
                replay = cache.replay(
                    database=learned_snapshot,
                    question=question,
                    domain=domain,
                    scope_hash=scope_hash,
                    source_excerpt_chars=source_excerpt_chars,
                )
                elapsed = round((perf_counter() - started) * 1000.0, 3)
                refs = replay.get("materialized_source_refs") if isinstance(replay, Mapping) else []
                ids = [
                    item.get("identity", {}).get("episode_id")
                    for item in refs if isinstance(item, Mapping) and isinstance(item.get("identity"), Mapping)
                ]
                pair["B3"] = {
                    "status": str(replay.get("cache_status", "not_observed")) if isinstance(replay, Mapping) else "invalid",
                    "operation_ms": elapsed,
                    "delivered_episode_ids": ids,
                    "provider_http_attempts": 0,
                }
                continue
            database = output_dir / "work" / f"b4_{index + 1:02d}.sqlite"
            setup_started = perf_counter()
            _clone_sqlite(learned_snapshot, database)
            app, config = _open_app(env_file, database, output_dir / "logs" / f"b4_{index + 1:02d}")
            engine = app.query_engine(config=config)
            setup_ms = round((perf_counter() - setup_started) * 1000.0, 3)
            ledger = ProviderLedger()
            started = perf_counter()
            with _trace_context(engine.model, ledger):
                result = engine.try_contextual_revisit(
                    question,
                    contextual_domain=domain,
                    contextual_revisit_scope_hash=scope_hash,
                    deadline_seconds=120.0,
                )
            elapsed = round((perf_counter() - started) * 1000.0, 3)
            provider = ledger.export()
            counts = provider.get("counts") if isinstance(provider, Mapping) else {}
            counts = counts if isinstance(counts, Mapping) else {}
            if int(counts.get("http_attempts", 0) or 0) != 0:
                raise RuntimeError("B4 replay unexpectedly issued provider HTTP")
            if not isinstance(result, Mapping):
                raise RuntimeError("B4 exact edge did not hit the public reuse entry")
            selected = result.get("evidence_episodes")
            selected = selected if isinstance(selected, list) else []
            ids = [item.get("id") for item in selected if isinstance(item, Mapping)]
            pair["B4"] = {
                "status": "public_exact_revisit_hit",
                "operation_ms": elapsed,
                "setup_ms_excluded": setup_ms,
                "delivered_episode_ids": ids,
                "provider_http_attempts": 0,
            }
        payload["repetitions"].append(pair)
        _write_json(output_dir / "paired_b3_b4.full_local.json", payload)
    b3_times = [float(item["B3"]["operation_ms"]) for item in payload["repetitions"]]
    b4_times = [float(item["B4"]["operation_ms"]) for item in payload["repetitions"]]
    payload["status"] = "completed"
    payload["finished_at"] = _utc_now()
    payload["summary"] = {
        "B3_source_validated_cache": _summary(b3_times),
        "B4_public_exact_edge": _summary(b4_times),
        "timing_ratio_claimed": False,
        "interpretation": "Matched local operation distributions are reported; this is not a general latency claim or answer benchmark.",
    }
    payload["snapshot_sha256_after"] = _sha256_file(learned_snapshot)
    payload["snapshot_unchanged"] = payload["snapshot_sha256_after"] == snapshot_hash
    full = output_dir / "paired_b3_b4.full_local.json"
    _write_json(full, payload)
    summary = {key: value for key, value in payload.items() if key != "repetitions"}
    _write_json(output_dir / "paired_b3_b4.json", summary)
    lines = [
        "# Paired local B3/B4 control",
        "",
        "No provider HTTP was issued. Each retained pair used the same frozen N11d snapshot, question, domain, scope and 3,000-character Source budget. B3 cache seeding and B4 clone/index setup are separately measured and excluded from operation time.",
        "",
        "| Arm | n | Median ms | p95 ms | Delivered Episode | HTTP |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
        f"| B3 ordinary source-validated evidence cache | {len(b3_times)} | {payload['summary']['B3_source_validated_cache']['median_ms']} | {payload['summary']['B3_source_validated_cache']['p95_ms']} | {episode_ids} | 0 |",
        f"| B4 public exact-edge reuse | {len(b4_times)} | {payload['summary']['B4_public_exact_edge']['median_ms']} | {payload['summary']['B4_public_exact_edge']['p95_ms']} | {episode_ids} | 0 |",
        "",
        "This paired local result does not calculate or claim a universal speed ratio. B3 and B4 are distinct mechanisms; both retain source validation and neither stores answer prose in this control.",
    ]
    (output_dir / "PAIRED_B3_B4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"full_local": str(full), "report": str(output_dir / "PAIRED_B3_B4.md")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--q1-q2-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--source-excerpt-chars", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
