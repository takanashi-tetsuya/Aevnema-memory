from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.retrieval_stability import (
    aggregate_stability_runs,
    score_fixed_candidate_run,
    validate_frozen_candidate_alignment,
)
from benchmarks.support.stage5 import load_json, sha256_file, write_json
from memory_demo.types import QueryIntent


_THREAD_LOCAL = threading.local()


def _load_candidate_texts(
    database: Path, source_rows: list[dict[str, Any]]
) -> dict[int, str]:
    candidate_ids = sorted(
        {
            int(value)
            for row in source_rows
            for value in (
                row.get("result", {})
                .get("rerank_trace", {})
                .get("candidate_episode_ids", [])
            )
        }
    )
    rows: dict[int, str] = {}
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for start in range(0, len(candidate_ids), 500):
            batch = candidate_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            for episode_id, text in connection.execute(
                f"SELECT id, text FROM episode WHERE id IN ({placeholders})",
                batch,
            ):
                rows[int(episode_id)] = str(text)
    return rows


def _application_for_worker(
    database: Path,
    log_dir: Path,
    review_mode: str,
    max_retries: int,
) -> MemoryApplication:
    app = getattr(_THREAD_LOCAL, "application", None)
    if app is not None:
        return app
    config = AppConfig.from_env()
    config.database_path = database
    config.log_dir = log_dir
    config.model.max_retries = max(0, int(max_retries))
    config.paragraph.enabled = False
    config.retrieval.graph_max_hops = 0
    config.retrieval.growth_max_rounds = 0
    config.retrieval.rerank_enabled = True
    config.retrieval.rerank_backend = "llm"
    config.retrieval.rerank_review_mode = review_mode
    config.retrieval.rerank_coverage_audit_enabled = True
    config.retrieval.rerank_audit_enabled = True
    config.retrieval.answer_episode_limit = 20
    app = MemoryApplication(config)
    _THREAD_LOCAL.application = app
    return app


def _episode_payloads(app: MemoryApplication, candidate_ids: list[int]) -> list[dict]:
    rows_by_id = {
        int(row["id"]): row for row in app.episodes.get_many(candidate_ids)
    }
    missing = [episode_id for episode_id in candidate_ids if episode_id not in rows_by_id]
    if missing:
        raise ValueError(
            "frozen candidate IDs are absent from the selected database: "
            + ", ".join(str(value) for value in missing[:20])
        )
    result: list[dict] = []
    for rank, episode_id in enumerate(candidate_ids, start=1):
        row = rows_by_id[episode_id]
        try:
            participants = json.loads(row["participants_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            participants = []
        result.append(
            {
                "id": episode_id,
                "source_id": int(row["source_id"]),
                "score": 1.0 - (rank - 1) / max(1, len(candidate_ids)),
                "text": str(row["text"]),
                "participants": participants if isinstance(participants, list) else [],
                "source_key": str(row["source_key"]),
                "story_time_text": str(row["story_time_text"] or ""),
                "timeline_scope": str(row["timeline_scope"] or ""),
                "evidence_origin": str(row["evidence_origin"] or "unknown"),
                "epistemic_status": str(row["epistemic_status"] or "unknown"),
                "generation": int(row["generation"] or 0),
                "epistemic_note": str(row["epistemic_note"] or ""),
            }
        )
    return result


def _run_one(
    *,
    source_row: dict[str, Any],
    query_row: dict[str, Any],
    criterion: dict[str, Any],
    repeat_index: int,
    database: Path,
    log_dir: Path,
    review_mode: str,
    max_retries: int,
    floor_policy: str,
) -> dict[str, Any]:
    app = _application_for_worker(database, log_dir, review_mode, max_retries)
    question_id = str(source_row["id"])
    source_result = source_row["result"]
    trace = source_result["rerank_trace"]
    query_result = query_row["result"]
    query_trace = query_result["rerank_trace"]
    candidate_ids = [int(value) for value in trace["candidate_episode_ids"]]
    episodes = _episode_payloads(app, candidate_ids)
    logger = app.new_logger(f"rerank-stability-{question_id}-r{repeat_index:02d}")
    engine = app.query_engine(logger)
    started = time.perf_counter()
    selected_ids, rerank_trace = engine._rerank_answer_episodes(
        str(source_row["question"]),
        QueryIntent.from_dict(query_result["intent"]),
        [str(value) for value in query_trace["atomic_queries"]],
        episodes,
        20,
        (
            [
                int(value)
                for value in dict.fromkeys(
                    [
                        *trace.get("constraint_candidate_ids", []),
                        *source_result.get("atomic_anchor_episode_ids", []),
                    ]
                )
            ]
            if floor_policy == "auto"
            else []
        ),
        required_candidate_ids=(
            [
                int(value)
                for value in trace.get("required_evidence_floor_ids", [])
            ]
            if floor_policy == "auto"
            else []
        ),
        answer_slot_anchor_ids=(
            [int(value) for value in trace.get("answer_slot_anchor_ids", [])]
            if floor_policy == "auto"
            else []
        ),
    )
    elapsed = time.perf_counter() - started
    score = score_fixed_candidate_run(
        selected_episode_ids=selected_ids,
        candidate_episode_ids=candidate_ids,
        required_episode_groups=criterion["required_episode_groups"],
        rerank_trace=rerank_trace,
    )
    return {
        "question_id": question_id,
        "question": str(source_row["question"]),
        "repeat_index": int(repeat_index),
        "elapsed_seconds": round(elapsed, 6),
        "candidate_episode_ids": candidate_ids,
        "selected_episode_ids": selected_ids,
        "score": score,
        "selection_floors": {
            "policy": floor_policy,
            "required_evidence_floor_ids": [
                int(value)
                for value in trace.get("required_evidence_floor_ids", [])
            ],
            "answer_slot_anchor_ids": [
                int(value) for value in trace.get("answer_slot_anchor_ids", [])
            ],
            "constraint_candidate_ids": [
                int(value) for value in trace.get("constraint_candidate_ids", [])
            ],
        },
        "rerank_trace": rerank_trace,
        "log_path": str(logger.path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the current LLM evidence selector over one historical, frozen "
            "Candidate@100 pool and report per-slot stability."
        )
    )
    parser.add_argument("input", type=Path, help="completed base retrieval report")
    parser.add_argument(
        "--query-input",
        type=Path,
        help=(
            "optional report supplying question intent and atomic queries while "
            "the positional input continues to supply Candidate@100"
        ),
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--question-id", action="append")
    parser.add_argument(
        "--review-mode", choices=("strict", "adaptive", "lean"), default="strict"
    )
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--floor-policy",
        choices=("auto", "off"),
        default="auto",
        help="auto replays saved current floors; off isolates candidate/query effects",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.database.exists():
        raise FileNotFoundError(args.database)

    source = load_json(args.input)
    query_source = load_json(args.query_input) if args.query_input else source
    manifest = load_json(args.manifest)
    source_rows = list(source["rows"])
    query_rows_by_id = {
        str(row["id"]): row for row in query_source["rows"]
    }
    if args.question_id:
        requested = set(args.question_id)
        source_rows = [row for row in source_rows if str(row["id"]) in requested]
        missing = requested.difference({str(row["id"]) for row in source_rows})
        if missing:
            raise ValueError(f"unknown question ids: {sorted(missing)}")
    criteria = manifest["questions"]
    for row in source_rows:
        if str(row["id"]) not in criteria:
            raise ValueError(f"manifest has no criterion for {row['id']}")
        query_row = query_rows_by_id.get(str(row["id"]))
        if query_row is None:
            raise ValueError(f"query input has no row for {row['id']}")
        if str(query_row["question"]) != str(row["question"]):
            raise ValueError(f"query input question differs for {row['id']}")

    alignment = validate_frozen_candidate_alignment(
        source_rows,
        _load_candidate_texts(args.database, source_rows),
    )
    if not alignment["passed"]:
        mismatch_ids = [
            int(item["episode_id"])
            for item in alignment["evidence_text_mismatches"][:10]
        ]
        raise ValueError(
            "frozen candidate/database semantic alignment failed; "
            f"missing IDs={alignment['missing_candidate_ids'][:10]}, "
            f"text-mismatched IDs={mismatch_ids}"
        )

    configuration = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "query_input": str((args.query_input or args.input).resolve()),
        "query_input_sha256": sha256_file(args.query_input or args.input),
        "database": str(args.database.resolve()),
        "database_sha256": sha256_file(args.database),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "question_ids": [str(row["id"]) for row in source_rows],
        "repeats": int(args.repeats),
        "workers": int(args.workers),
        "review_mode": args.review_mode,
        "graph_max_hops": 0,
        "growth_max_rounds": 0,
        "paragraph_enabled": False,
        "rerank_backend": "llm",
        "candidate_pool": "frozen_from_input",
        "atomic_queries": "frozen_from_query_input",
        "floor_policy": args.floor_policy,
        "selected_top_k": 20,
        "database_alignment": alignment,
    }
    completed: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        previous = load_json(args.output)
        previous_configuration = previous.get("configuration", {})
        immutable_keys = (
            "input_sha256",
            "database_sha256",
            "manifest_sha256",
            "query_input_sha256",
            "question_ids",
            "repeats",
            "review_mode",
            "floor_policy",
        )
        if any(
            previous_configuration.get(key) != configuration.get(key)
            for key in immutable_keys
        ):
            raise ValueError("resume output uses different frozen inputs")
        completed = list(previous.get("runs", []))

    completed_keys = {
        (str(row["question_id"]), int(row["repeat_index"])) for row in completed
    }
    jobs = [
        (row, query_rows_by_id[str(row["id"])], repeat_index)
        for row in source_rows
        for repeat_index in range(1, args.repeats + 1)
        if (str(row["id"]), repeat_index) not in completed_keys
    ]
    log_dir = args.output.parent / f"{args.output.stem}-logs"

    def payload(checkpoint: bool) -> dict[str, Any]:
        ordered = sorted(
            completed,
            key=lambda item: (str(item["question_id"]), int(item["repeat_index"])),
        )
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "configuration": configuration,
            "summary": aggregate_stability_runs(
                ordered, expected_repeats=args.repeats
            ),
            "runs": ordered,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                source_row=row,
                query_row=query_row,
                criterion=criteria[str(row["id"])],
                repeat_index=repeat_index,
                database=args.database,
                log_dir=log_dir,
                review_mode=args.review_mode,
                max_retries=args.max_retries,
                floor_policy=args.floor_policy,
            ): (str(row["id"]), repeat_index)
            for row, query_row, repeat_index in jobs
        }
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(args.output, payload(checkpoint=True))
            question_id, repeat_index = futures[future]
            print(f"completed {question_id} repeat {repeat_index}", flush=True)

    write_json(args.output, payload(checkpoint=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
