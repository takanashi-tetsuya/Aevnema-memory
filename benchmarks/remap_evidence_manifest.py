from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def vector(blob: bytes) -> np.ndarray:
    value = np.frombuffer(blob, dtype="<f4")
    if value.size != 1024 or not np.isfinite(value).all():
        raise ValueError("evidence remap requires finite 1024-dimensional float32")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remap fixed evidence groups between independently rebuilt databases."
    )
    parser.add_argument("old_database", type=Path)
    parser.add_argument("new_database", type=Path)
    parser.add_argument("old_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-per-old-episode", type=int, default=3)
    parser.add_argument("--minimum-similarity", type=float, default=0.55)
    args = parser.parse_args()

    old_manifest = load_json(args.old_manifest)
    old_connection = connect_read_only(args.old_database)
    new_connection = connect_read_only(args.new_database)
    old_rows = {
        int(row["id"]): row
        for row in old_connection.execute(
            "SELECT id, source_key, text, embedding FROM episode ORDER BY id"
        )
    }
    new_by_source: dict[str, list[sqlite3.Row]] = {}
    for row in new_connection.execute(
        "SELECT id, source_key, text, embedding FROM episode ORDER BY id"
    ):
        new_by_source.setdefault(str(row["source_key"]), []).append(row)

    output = dict(old_manifest)
    output["version"] = f"{old_manifest.get('version', 'unknown')}-remapped-v2"
    output["database"] = str(args.new_database.resolve())
    output["remapped_from_database"] = str(args.old_database.resolve())
    output_questions: dict[str, Any] = {}
    report_questions: dict[str, Any] = {}
    top_scores: list[float] = []
    missing_old_ids: list[int] = []
    missing_sources: Counter[str] = Counter()
    low_similarity: list[dict[str, Any]] = []

    for question_id, raw_spec in old_manifest.get("questions", {}).items():
        spec = dict(raw_spec)
        mapped_groups: list[list[int]] = []
        group_reports: list[dict[str, Any]] = []
        for group_index, old_group in enumerate(
            raw_spec.get("required_episode_groups", [])
        ):
            mapped_ids: set[int] = set()
            old_reports: list[dict[str, Any]] = []
            for raw_old_id in old_group:
                old_id = int(raw_old_id)
                old_row = old_rows.get(old_id)
                if old_row is None:
                    missing_old_ids.append(old_id)
                    old_reports.append({"old_episode_id": old_id, "error": "missing"})
                    continue
                source_key = str(old_row["source_key"])
                candidates = new_by_source.get(source_key, [])
                if not candidates:
                    missing_sources[source_key] += 1
                    old_reports.append(
                        {
                            "old_episode_id": old_id,
                            "source_key": source_key,
                            "old_text": str(old_row["text"]),
                            "error": "source missing from new database",
                        }
                    )
                    continue
                old_vector = vector(old_row["embedding"])
                scored = sorted(
                    [
                        (
                            float(old_vector @ vector(candidate["embedding"])),
                            candidate,
                        )
                        for candidate in candidates
                    ],
                    key=lambda item: item[0],
                    reverse=True,
                )
                best = scored[0][0]
                top_scores.append(best)
                selected = [
                    (score, candidate)
                    for score, candidate in scored
                    if score >= float(args.minimum_similarity)
                ][: max(1, int(args.top_per_old_episode))]
                if not selected:
                    selected = scored[:1]
                mapped_ids.update(int(candidate["id"]) for _, candidate in selected)
                if best < float(args.minimum_similarity):
                    low_similarity.append(
                        {
                            "question_id": str(question_id),
                            "group_index": group_index,
                            "old_episode_id": old_id,
                            "source_key": source_key,
                            "best_similarity": best,
                        }
                    )
                old_reports.append(
                    {
                        "old_episode_id": old_id,
                        "source_key": source_key,
                        "old_text": str(old_row["text"]),
                        "best_similarity": best,
                        "selected": [
                            {
                                "new_episode_id": int(candidate["id"]),
                                "similarity": score,
                                "text": str(candidate["text"]),
                            }
                            for score, candidate in selected
                        ],
                    }
                )
            mapped_groups.append(sorted(mapped_ids))
            group_reports.append(
                {
                    "group_index": group_index,
                    "old_episode_ids": [int(value) for value in old_group],
                    "new_episode_ids": sorted(mapped_ids),
                    "old_episode_mappings": old_reports,
                }
            )
        spec["required_episode_groups"] = mapped_groups
        output_questions[str(question_id)] = spec
        report_questions[str(question_id)] = {"groups": group_reports}

    output["questions"] = output_questions
    empty_groups = [
        {"question_id": question_id, "group_index": index}
        for question_id, spec in output_questions.items()
        for index, group in enumerate(spec.get("required_episode_groups", []))
        if not group
    ]
    report = {
        "version": "evidence-manifest-remap-v2",
        "old_database": str(args.old_database.resolve()),
        "new_database": str(args.new_database.resolve()),
        "old_manifest": str(args.old_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "configuration": {
            "top_per_old_episode": max(1, int(args.top_per_old_episode)),
            "minimum_similarity": float(args.minimum_similarity),
            "selection": "top-N within identical source_key above minimum similarity",
            "source_key_restricted": True,
            "embedding_dtype": "float32",
        },
        "summary": {
            "questions": len(output_questions),
            "groups": sum(
                len(spec.get("required_episode_groups", []))
                for spec in output_questions.values()
            ),
            "empty_groups": len(empty_groups),
            "missing_old_episode_ids": len(missing_old_ids),
            "missing_new_sources": sum(missing_sources.values()),
            "low_similarity_mappings": len(low_similarity),
            "top_similarity_below_0_70": sum(score < 0.70 for score in top_scores),
            "top_similarity_below_0_75": sum(score < 0.75 for score in top_scores),
            "top_similarity_below_0_80": sum(score < 0.80 for score in top_scores),
            "top_similarity_minimum": min(top_scores, default=None),
            "top_similarity_mean": (
                sum(top_scores) / len(top_scores) if top_scores else None
            ),
            "top_similarity_maximum": max(top_scores, default=None),
        },
        "empty_groups": empty_groups,
        "missing_old_episode_ids": sorted(set(missing_old_ids)),
        "missing_new_sources": dict(missing_sources),
        "low_similarity_mappings": low_similarity,
        "questions": report_questions,
    }
    old_connection.close()
    new_connection.close()
    write_json(args.output_manifest, output)
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "report": str(args.report),
                **report["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if not empty_groups and not missing_old_ids and not missing_sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
