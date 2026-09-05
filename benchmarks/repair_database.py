from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.associations.builder import AssociationBuilder
from memory_demo.config import AppConfig
from memory_demo.database import utc_now
from memory_demo.embeddings import encode_embedding, normalize_embedding
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.ingestion.ordering import infer_timeline_scope
from memory_demo.llm import ModelClient
from memory_demo.types import EpisodeDraft


def row_to_draft(row) -> EpisodeDraft:
    try:
        participants = json.loads(row["participants_json"])
    except (TypeError, json.JSONDecodeError):
        participants = []
    return EpisodeDraft(
        text=str(row["text"]),
        participants=participants if isinstance(participants, list) else [],
        event_type=str(row["event_type"]),
        location_text=str(row["location_text"]),
        story_time_text=str(row["story_time_text"]),
        timeline_scope=str(row["timeline_scope"]),
        confidence=float(row["confidence"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply deterministic evidence and Association integrity guards"
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--database")
    parser.add_argument("--log-dir")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write repairs; without this flag only report proposed changes",
    )
    args = parser.parse_args()

    config = AppConfig.from_env(args.env_file)
    if args.database:
        config.database_path = Path(args.database)
    if args.log_dir:
        config.log_dir = Path(args.log_dir)
    app = MemoryApplication(config)
    logger = app.new_logger("repair")
    with app.db.connection() as connection:
        episode_rows = connection.execute("SELECT * FROM episode ORDER BY id").fetchall()
        association_rows = connection.execute(
            """
            SELECT * FROM association
            WHERE from_type = 'episode' AND to_type = 'episode'
              AND relation_type IN ('identity', 'temporal') AND polarity > 0
            ORDER BY id
            """
        ).fetchall()
    episodes_by_id = {int(row["id"]): row for row in episode_rows}

    episode_repairs: list[tuple[object, EpisodeDraft]] = []
    timeline_scope_repairs: list[tuple[object, str]] = []
    for row in episode_rows:
        draft = row_to_draft(row)
        previous_text = draft.text
        previous_participants = list(draft.participants)
        MemoryExtractor.sanitize_episode_draft(draft)
        if draft.text != previous_text or draft.participants != previous_participants:
            episode_repairs.append((row, draft))
        expected_scope = infer_timeline_scope(str(row["source_key"]))
        if draft.timeline_scope != expected_scope:
            timeline_scope_repairs.append((row, expected_scope))

    association_rejections: list[tuple[object, str]] = []
    for row in association_rows:
        reason = AssociationBuilder.episode_relationship_rejection_reason(
            dict(row),
            episodes_by_id.get(int(row["from_id"])),
            episodes_by_id.get(int(row["to_id"])),
        )
        if reason:
            association_rejections.append((row, reason))

    rejected_ids = {int(row["id"]) for row, _reason in association_rejections}
    temporal_groups: dict[tuple[int, int], list[object]] = {}
    for row in association_rows:
        if row["relation_type"] != "temporal" or int(row["id"]) in rejected_ids:
            continue
        from_id, to_id, relation_key = (
            AssociationBuilder.normalize_episode_relationship(
                int(row["from_id"]), int(row["to_id"]), dict(row)
            )
        )
        if relation_key != "before":
            continue
        temporal_groups.setdefault((from_id, to_id), []).append(row)
    temporal_normalizations = [
        (canonical, rows)
        for canonical, rows in temporal_groups.items()
        if len(rows) > 1
        or int(rows[0]["from_id"]) != canonical[0]
        or int(rows[0]["to_id"]) != canonical[1]
        or str(rows[0]["relation_key"]) != "before"
    ]

    if args.apply and episode_repairs:
        model = ModelClient(config.model, logger)
        matrix = model.embed([draft.text for _row, draft in episode_repairs])
        for (row, draft), raw_vector in zip(episode_repairs, matrix, strict=True):
            vector = normalize_embedding(
                raw_vector, config.model.embedding_dimension
            )
            app.episodes.update_draft(
                int(row["id"]),
                draft,
                encode_embedding(vector, config.model.embedding_dimension),
            )
            logger.emit(
                "episode_unknown_identity_sanitized",
                episode_id=int(row["id"]),
                previous_text=str(row["text"]),
                revised_text=draft.text,
            )

    if args.apply:
        for row, expected_scope in timeline_scope_repairs:
            with app.db.transaction() as connection:
                connection.execute(
                    "UPDATE episode SET timeline_scope = ?, updated_at = ? WHERE id = ?",
                    (expected_scope, utc_now(), int(row["id"])),
                )
            logger.emit(
                "episode_timeline_scope_repaired",
                episode_id=int(row["id"]),
                source_key=str(row["source_key"]),
                previous_scope=str(row["timeline_scope"]),
                revised_scope=expected_scope,
            )

    if args.apply:
        for row, reason in association_rejections:
            deleted = app.associations.delete(int(row["id"]))
            logger.emit(
                "association_rejected_cleanup",
                association_id=int(row["id"]),
                reason=reason,
                relation_text=str(row["relation_text"]),
                deleted=deleted,
            )
        for (from_id, to_id), rows in temporal_normalizations:
            selected = max(
                rows,
                key=lambda row: (float(row["confidence"]), float(row["weight"])),
            )
            row_ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in row_ids)
            with app.db.transaction() as connection:
                connection.execute(
                    f"DELETE FROM association WHERE id IN ({placeholders})", row_ids
                )
                cursor = connection.execute(
                    """
                    INSERT INTO association(
                        from_type, from_id, to_type, to_id,
                        relation_type, relation_key, relation_text, polarity,
                        weight, confidence, generation, evidence_count, created_reason,
                        created_at, updated_at, last_used, use_count
                    ) VALUES('episode', ?, 'episode', ?, 'temporal', 'before',
                             ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        from_id,
                        to_id,
                        str(selected["relation_text"]),
                        max(float(row["weight"]) for row in rows),
                        max(float(row["confidence"]) for row in rows),
                        min(
                            int(row["generation"])
                            if "generation" in row.keys()
                            else 0
                            for row in rows
                        ),
                        sum(int(row["evidence_count"]) for row in rows),
                        "v2.8 temporal canonical repair",
                        min(str(row["created_at"]) for row in rows),
                        utc_now(),
                        max(
                            (str(row["last_used"]) for row in rows if row["last_used"]),
                            default=None,
                        ),
                        sum(int(row["use_count"]) for row in rows),
                    ),
                )
                association_id = int(cursor.lastrowid)
            logger.emit(
                "temporal_associations_canonicalized",
                previous_association_ids=row_ids,
                association_id=association_id,
                from_id=from_id,
                to_id=to_id,
                relation_key="before",
            )

    print(
        json.dumps(
            {
                "database": str(config.database_path),
                "applied": args.apply,
                "episode_repairs": [
                    {
                        "episode_id": int(row["id"]),
                        "previous_text": str(row["text"]),
                        "revised_text": draft.text,
                    }
                    for row, draft in episode_repairs
                ],
                "timeline_scope_repairs": [
                    {
                        "episode_id": int(row["id"]),
                        "source_key": str(row["source_key"]),
                        "previous_scope": str(row["timeline_scope"]),
                        "revised_scope": expected_scope,
                    }
                    for row, expected_scope in timeline_scope_repairs
                ],
                "association_rejections": [
                    {
                        "association_id": int(row["id"]),
                        "reason": reason,
                        "relation_text": str(row["relation_text"]),
                    }
                    for row, reason in association_rejections
                ],
                "temporal_normalizations": [
                    {
                        "previous_association_ids": [int(row["id"]) for row in rows],
                        "from_id": canonical[0],
                        "to_id": canonical[1],
                        "relation_key": "before",
                    }
                    for canonical, rows in temporal_normalizations
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
