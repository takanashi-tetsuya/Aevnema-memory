from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3


INFERENTIAL_TYPE_MARKERS = (
    "推测",
    "推论",
    "考据",
    "社区观点",
    "剧情解读",
)

REFERENCE_META_EVENT_MARKERS = (
    "文档发布",
    "资料发布",
    "文档标题",
    "资料标题",
    "百科标题",
    "文档创建",
    "作品发布",
    "作品标题确认",
)


def is_reference_meta_episode(row: sqlite3.Row) -> bool:
    event_type = str(row["event_type"] or "")
    text = str(row["text"] or "")
    participants = str(row["participants_json"] or "")
    return (
        any(marker in event_type for marker in REFERENCE_META_EVENT_MARKERS)
        or ("文档作者" in participants and "文档" in text)
        or (
            ("文档" in text or "作品标题" in text)
            and any(
                phrase in text
                for phrase in (
                    "被创建",
                    "作为作品标题被正式",
                    "作为官方发布的作品标题",
                )
            )
        )
    )


def concepts_orphaned_by_meta_removal(
    connection: sqlite3.Connection, meta_episode_ids: list[int]
) -> list[int]:
    """Return Concepts whose only Episode evidence is a removed meta Episode."""

    if not meta_episode_ids:
        return []
    placeholders = ",".join("?" for _ in meta_episode_ids)
    candidates = connection.execute(
        f"""
        SELECT DISTINCT CASE
            WHEN from_type = 'concept' THEN from_id ELSE to_id
        END AS concept_id
        FROM association
        WHERE (from_type = 'episode' AND from_id IN ({placeholders})
               AND to_type = 'concept')
           OR (to_type = 'episode' AND to_id IN ({placeholders})
               AND from_type = 'concept')
        """,
        [*meta_episode_ids, *meta_episode_ids],
    ).fetchall()
    orphaned: list[int] = []
    for candidate in candidates:
        concept_id = int(candidate["concept_id"])
        other_episode_link = connection.execute(
            f"""
            SELECT 1 FROM association
            WHERE (
                from_type = 'concept' AND from_id = ? AND to_type = 'episode'
                AND to_id NOT IN ({placeholders})
            ) OR (
                to_type = 'concept' AND to_id = ? AND from_type = 'episode'
                AND from_id NOT IN ({placeholders})
            )
            LIMIT 1
            """,
            [concept_id, *meta_episode_ids, concept_id, *meta_episode_ids],
        ).fetchone()
        canonical_reference = connection.execute(
            "SELECT 1 FROM concept WHERE canonical_concept_id = ? LIMIT 1",
            (concept_id,),
        ).fetchone()
        if other_episode_link is None and canonical_reference is None:
            orphaned.append(concept_id)
    return sorted(orphaned)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inferential_episode_plan(
    connection: sqlite3.Connection,
) -> tuple[list[int], list[int], list[int], dict[int, str], list[int]]:
    source_ids: list[int] = []
    episode_ids: list[int] = []
    ambiguous_source_ids: list[int] = []
    reference_type_by_episode: dict[int, str] = {}
    meta_episode_ids: list[int] = []
    for source_id, raw_text in connection.execute(
        "SELECT id, raw_text FROM source ORDER BY id"
    ):
        reference_types = re.findall(
            r"\[资料类型[：:]\s*([^\]|]+)", str(raw_text)
        )
        if not reference_types:
            continue
        rows = connection.execute(
            """
            SELECT id, event_type, text, participants_json
            FROM episode WHERE source_id = ? ORDER BY id
            """,
            (int(source_id),),
        ).fetchall()
        # A plain title is not a labelled fact record.  Remove only strong
        # metadata-shaped extras until record and Episode counts reconcile.
        # If the remaining cardinality still differs, keep refusing to guess.
        rows = list(rows)
        while len(rows) > len(reference_types):
            meta_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if is_reference_meta_episode(row)
                ),
                None,
            )
            if meta_index is None:
                break
            meta_episode_ids.append(int(rows[meta_index]["id"]))
            rows.pop(meta_index)
        if len(rows) == len(reference_types):
            selected_pairs = [
                (int(row["id"]), reference_type.strip())
                for row, reference_type in zip(rows, reference_types, strict=True)
                if any(
                    marker in reference_type
                    for marker in INFERENTIAL_TYPE_MARKERS
                )
            ]
        elif all(
            any(marker in reference_type for marker in INFERENTIAL_TYPE_MARKERS)
            for reference_type in reference_types
        ):
            selected_pairs = [
                (int(row["id"]), "整段资料均明确标记为推测") for row in rows
            ]
        elif any(
            any(marker in reference_type for marker in INFERENTIAL_TYPE_MARKERS)
            for reference_type in reference_types
        ):
            ambiguous_source_ids.append(int(source_id))
            continue
        else:
            selected_pairs = []
        selected = [episode_id for episode_id, _label in selected_pairs]
        reference_type_by_episode.update(selected_pairs)
        if selected:
            source_ids.append(int(source_id))
            episode_ids.extend(selected)
    return (
        source_ids,
        episode_ids,
        ambiguous_source_ids,
        reference_type_by_episode,
        meta_episode_ids,
    )


def repair_explicit_evidence(
    database: str | Path, *, apply: bool = False
) -> dict:
    database_path = Path(database).resolve()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        (
            source_ids,
            episode_ids,
            ambiguous_source_ids,
            reference_type_by_episode,
            meta_episode_ids,
        ) = inferential_episode_plan(connection)
        orphaned_concept_ids = concepts_orphaned_by_meta_removal(
            connection, meta_episode_ids
        )
        if not episode_ids and not meta_episode_ids:
            return {
                "database": str(database_path),
                "applied": False,
                "source_ids": source_ids,
                "episode_ids": episode_ids,
                "ambiguous_source_ids": ambiguous_source_ids,
                "meta_episode_ids": meta_episode_ids,
                "orphaned_concept_ids": orphaned_concept_ids,
                "association_updates": [],
            }
        placeholders = ",".join("?" for _ in episode_ids)
        episode_rows = (
            connection.execute(
                f"""
                SELECT id, evidence_origin, epistemic_status, generation
                       , source_id, source_key, text
                FROM episode WHERE id IN ({placeholders}) ORDER BY id
                """,
                episode_ids,
            ).fetchall()
            if episode_ids
            else []
        )
        planned_episode_generation = {
            int(row["id"]): max(1, int(row["generation"] or 0))
            for row in episode_rows
        }
        association_updates: list[dict] = []
        if episode_ids:
            episode_placeholders = ",".join("?" for _ in episode_ids)
            association_rows = connection.execute(
                f"""
                SELECT id, from_type, from_id, to_type, to_id, generation,
                       claim_level, created_reason
                FROM association
                WHERE (from_type = 'episode' AND from_id IN ({episode_placeholders}))
                   OR (to_type = 'episode' AND to_id IN ({episode_placeholders}))
                ORDER BY id
                """,
                [*episode_ids, *episode_ids],
            ).fetchall()
            for row in association_rows:
                endpoint_generations = [
                    planned_episode_generation[int(row[f"{prefix}_id"])]
                    for prefix in ("from", "to")
                    if row[f"{prefix}_type"] == "episode"
                    and int(row[f"{prefix}_id"]) in planned_episode_generation
                ]
                minimum = max(endpoint_generations, default=0)
                is_episode_inference = (
                    row["from_type"] == "episode"
                    and row["to_type"] == "episode"
                    and row["claim_level"] == "supported_inference"
                )
                target = max(
                    int(row["generation"] or 0),
                    minimum + int(is_episode_inference),
                )
                if target != int(row["generation"] or 0):
                    association_updates.append(
                        {
                            "id": int(row["id"]),
                            "before": int(row["generation"] or 0),
                            "after": target,
                        }
                    )

        report = {
            "database": str(database_path),
            "applied": bool(apply),
            "source_ids": source_ids,
            "episode_ids": episode_ids,
            "ambiguous_source_ids": ambiguous_source_ids,
            "meta_episode_ids": meta_episode_ids,
            "orphaned_concept_ids": orphaned_concept_ids,
            "episode_repairs": [
                {
                    "id": int(row["id"]),
                    "source_id": int(row["source_id"]),
                    "source_key": str(row["source_key"]),
                    "reference_type": reference_type_by_episode[int(row["id"])],
                    "text": str(row["text"]),
                    "before": {
                        "evidence_origin": row["evidence_origin"],
                        "epistemic_status": row["epistemic_status"],
                        "generation": int(row["generation"] or 0),
                    },
                }
                for row in episode_rows
            ],
            "episode_updates": sum(
                row["evidence_origin"] != "importer"
                or row["epistemic_status"] != "speculative"
                or int(row["generation"] or 0) < 1
                for row in episode_rows
            ),
            "association_updates": association_updates,
        }
        if not apply:
            return report

        now = utc_now()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if episode_ids:
                connection.execute(
                    f"""
                    UPDATE episode
                    SET evidence_origin = 'importer',
                        epistemic_status = 'speculative',
                        generation = MAX(generation, 1),
                        epistemic_note = CASE
                            WHEN trim(epistemic_note) = ''
                            THEN '导入文档明确标记为推测'
                            ELSE epistemic_note
                        END,
                        updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now, *episode_ids],
                )
            connection.executemany(
                "UPDATE association SET generation = ?, updated_at = ? WHERE id = ?",
                [
                    (item["after"], now, item["id"])
                    for item in association_updates
                ],
            )
            if meta_episode_ids:
                meta_placeholders = ",".join("?" for _ in meta_episode_ids)
                connection.execute(
                    f"""
                    DELETE FROM association
                    WHERE (from_type = 'episode' AND from_id IN ({meta_placeholders}))
                       OR (to_type = 'episode' AND to_id IN ({meta_placeholders}))
                    """,
                    [*meta_episode_ids, *meta_episode_ids],
                )
                connection.execute(
                    f"DELETE FROM episode WHERE id IN ({meta_placeholders})",
                    meta_episode_ids,
                )
            if orphaned_concept_ids:
                concept_placeholders = ",".join(
                    "?" for _ in orphaned_concept_ids
                )
                connection.execute(
                    f"""
                    DELETE FROM association
                    WHERE (from_type = 'concept'
                           AND from_id IN ({concept_placeholders}))
                       OR (to_type = 'concept'
                           AND to_id IN ({concept_placeholders}))
                    """,
                    [*orphaned_concept_ids, *orphaned_concept_ids],
                )
                connection.execute(
                    f"DELETE FROM concept_alias WHERE concept_id IN ({concept_placeholders})",
                    orphaned_concept_ids,
                )
                connection.execute(
                    f"DELETE FROM concept WHERE id IN ({concept_placeholders})",
                    orphaned_concept_ids,
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return report
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or repair Episodes from wholly inferential reference Sources"
        )
    )
    parser.add_argument("database")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the repair; omission performs a read-only dry run",
    )
    parser.add_argument("--output", help="optional UTF-8 JSON report path")
    args = parser.parse_args()
    report = repair_explicit_evidence(args.database, apply=args.apply)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
