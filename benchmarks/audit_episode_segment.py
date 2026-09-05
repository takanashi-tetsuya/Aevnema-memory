from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.llm import ModelClient
from memory_demo.types import EpisodeDraft


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Re-audit one imported Source segment without modifying its database"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("source_key")
    parser.add_argument("segment_index", type=int)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--independent-extraction",
        action="store_true",
        help="extract from Source without showing the existing candidates",
    )
    parser.add_argument(
        "--factual-audit",
        action="store_true",
        help="run only the focused speaker/identity factual audit",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        help="use the after list from an earlier audit report as candidates",
    )
    parser.add_argument(
        "--candidate-list-key",
        choices=("before", "after"),
        default="after",
        help="candidate list to read from --candidate-report",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AppConfig.from_env(args.env_file)
    if args.model:
        config.model.reasoning_model = args.model
    config.ensure_directories()
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    source = connection.execute(
        """
        SELECT DISTINCT s.id, s.raw_text
        FROM source s JOIN episode e ON e.source_id = s.id
        WHERE e.source_key = ? AND e.segment_index = ?
        ORDER BY s.id DESC LIMIT 1
        """,
        (args.source_key, args.segment_index),
    ).fetchone()
    if source is None:
        raise SystemExit("source segment not found")
    rows = connection.execute(
        """
        SELECT * FROM episode WHERE source_id = ? ORDER BY id
        """,
        (int(source["id"]),),
    ).fetchall()
    connection.close()
    episodes = [
        EpisodeDraft(
            text=str(row["text"]),
            participants=list(json.loads(row["participants_json"])),
            event_type=str(row["event_type"]),
            location_text=str(row["location_text"]),
            story_time_text=str(row["story_time_text"]),
            timeline_scope=str(row["timeline_scope"]),
            confidence=float(row["confidence"]),
            evidence_origin=str(row["evidence_origin"]),
            epistemic_status=str(row["epistemic_status"]),
            generation=int(row["generation"]),
            epistemic_note=str(row["epistemic_note"]),
        )
        for row in rows
    ]
    if args.candidate_report:
        candidate_payload = json.loads(
            args.candidate_report.read_text(encoding="utf-8")
        )
        raw_candidates = candidate_payload.get(args.candidate_list_key, [])
        if not isinstance(raw_candidates, list):
            raise SystemExit(
                f"candidate report {args.candidate_list_key} must be a list"
            )
        episodes = [EpisodeDraft.from_dict(item) for item in raw_candidates]
    logger = JsonlEventLogger(
        config.log_dir / "episode-segment-fidelity-audit.jsonl"
    )
    extractor = MemoryExtractor(
        ModelClient(config.model, logger), logger, episode_audit_mode="off"
    )
    timeline_scope = episodes[0].timeline_scope if episodes else ""
    if args.independent_extraction and args.factual_audit:
        raise SystemExit(
            "--independent-extraction and --factual-audit are mutually exclusive"
        )
    if args.independent_extraction:
        audited, errors = extractor.extract_episodes(
            str(source["raw_text"]), timeline_scope
        )
    elif args.factual_audit:
        reasoning_source = extractor.compact_source_for_reasoning(
            str(source["raw_text"])
        )
        audited, errors = extractor._audit_episode_facts(
            reasoning_source, timeline_scope, episodes
        )
        extractor._remove_unsupported_name_aliases(
            reasoning_source, audited, logger
        )
        extractor._repair_source_name_typos(reasoning_source, audited, logger)
    else:
        audited, errors = extractor._audit_episode_quality(
            str(source["raw_text"]),
            timeline_scope,
            episodes,
            temporal_risk=True,
            granularity_risk=True,
        )
    report = {
        "database": str(args.database.resolve()),
        "source_key": args.source_key,
        "segment_index": args.segment_index,
        "model": config.model.reasoning_model,
        "mode": (
            "independent_extraction"
            if args.independent_extraction
            else "factual_audit"
            if args.factual_audit
            else "candidate_audit"
        ),
        "candidate_report": (
            str(args.candidate_report.resolve())
            if args.candidate_report
            else None
        ),
        "candidate_list_key": (
            args.candidate_list_key if args.candidate_report else None
        ),
        "before": [asdict(episode) for episode in episodes],
        "after": [asdict(episode) for episode in audited],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
