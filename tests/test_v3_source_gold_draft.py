from __future__ import annotations

import gc
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from benchmarks.build_v3_source_gold_draft import DRAFT_STATUS, build_draft
from benchmarks.support.evidence_scorer import CRITERIA


class V3SourceGoldDraftTests(unittest.TestCase):
    @staticmethod
    def _required_episode_ids() -> list[int]:
        return sorted(
            {
                int(episode_id)
                for criterion in CRITERIA.values()
                for alternatives in criterion["required_episode_groups"]
                for episode_id in alternatives
            }
        )

    @staticmethod
    def _source_keys() -> list[str]:
        return list(
            dict.fromkeys(
                source_key
                for criterion in CRITERIA.values()
                for source_key in criterion["required_sources"]
            )
        )

    @staticmethod
    def _allowed_source_keys_by_episode() -> dict[int, list[str]]:
        allowed: dict[int, set[str]] = {}
        for criterion in CRITERIA.values():
            criterion_sources = set(criterion["required_sources"])
            for alternatives in criterion["required_episode_groups"]:
                for raw_episode_id in alternatives:
                    episode_id = int(raw_episode_id)
                    if episode_id in allowed:
                        allowed[episode_id].intersection_update(criterion_sources)
                    else:
                        allowed[episode_id] = set(criterion_sources)
        if any(not source_keys for source_keys in allowed.values()):
            raise AssertionError("shared historical episodes must retain a source closure")
        return {
            episode_id: sorted(source_keys)
            for episode_id, source_keys in allowed.items()
        }

    def _build_snapshot(
        self, root: Path, *, missing_episode_id: int | None = None
    ) -> Path:
        database = root / "synthetic-stage3.db"
        source_keys = self._source_keys()
        allowed_source_keys = self._allowed_source_keys_by_episode()
        self.assertTrue(source_keys)
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE source(
                    id INTEGER PRIMARY KEY,
                    raw_text TEXT NOT NULL
                );
                CREATE TABLE episode(
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER NOT NULL,
                    source_key TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    evidence_spans_json TEXT NOT NULL,
                    evidence_quotes_json TEXT NOT NULL
                );
                """
            )
            source_ids: dict[str, int] = {}
            for source_id, source_key in enumerate(source_keys, start=1):
                source_ids[source_key] = source_id
                connection.execute(
                    "INSERT INTO source(id, raw_text) VALUES(?, ?)",
                    (source_id, f"synthetic raw source for {source_key}"),
                )
            for index, episode_id in enumerate(self._required_episode_ids()):
                if episode_id == missing_episode_id:
                    continue
                source_key = allowed_source_keys[episode_id][0]
                connection.execute(
                    """
                    INSERT INTO episode(
                        id, source_id, source_key, segment_index, text,
                        evidence_spans_json, evidence_quotes_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        source_ids[source_key],
                        source_key,
                        index,
                        f"synthetic episode {episode_id}",
                        "[]",
                        "[]",
                    ),
                )
            connection.commit()
        finally:
            connection.close()
        return database

    @staticmethod
    def _write_inputs(root: Path) -> tuple[Path, Path]:
        questions = root / "questions.json"
        legacy_manifest = root / "legacy-manifest.json"
        questions.write_text(
            json.dumps(
                [
                    {"id": question_id, "question": f"Synthetic {question_id}?"}
                    for question_id in CRITERIA
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        legacy_manifest.write_text(
            json.dumps(
                {
                    "questions": {
                        question_id: {
                            "required_facts": [f"fact for {question_id}"],
                            "anchors": [f"anchor for {question_id}"],
                        }
                        for question_id in CRITERIA
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return questions, legacy_manifest

    def test_build_draft_keeps_historical_slots_pending_and_calibration_only(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._build_snapshot(root)
            questions, legacy_manifest = self._write_inputs(root)

            draft, split, review = build_draft(
                database=database,
                questions_path=questions,
                legacy_manifest_path=legacy_manifest,
            )
            gc.collect()

        expected_family_count = len(CRITERIA)
        expected_group_count = sum(
            len(criterion["required_episode_groups"])
            for criterion in CRITERIA.values()
        )
        self.assertEqual(expected_family_count, 3)
        self.assertEqual(expected_group_count, 9)
        self.assertEqual(len(draft["families"]), expected_family_count)
        self.assertEqual(
            sum(len(family["claim_groups"]) for family in draft["families"]),
            expected_group_count,
        )
        self.assertEqual(draft["status"], DRAFT_STATUS)
        self.assertIn("not an accepted Source-level gold set", review)

        for family in draft["families"]:
            self.assertEqual(family["review_status"], "pending_source_span_review")
            self.assertFalse(family["usable_for_scoring"])
            self.assertEqual(family["question_coverage"]["operator"], "all_of")
            self.assertEqual(
                family["question_coverage"]["members"],
                [group["claim_group_id"] for group in family["claim_groups"]],
            )
            for group in family["claim_groups"]:
                self.assertEqual(group["coverage_clause"]["operator"], "any_of")
                self.assertTrue(group["coverage_clause"]["required"])
                self.assertEqual(group["review_status"], "pending_source_span_review")
                self.assertFalse(group["usable_for_scoring"])
                self.assertTrue(group["evidence_atoms"])
                for atom in group["evidence_atoms"]:
                    self.assertEqual(atom["review_status"], "pending_source_span_review")
                    self.assertFalse(atom["usable_for_scoring"])
                    self.assertIsNone(atom["record_locator"])
                    self.assertIsNone(atom["span_start"])
                    self.assertIsNone(atom["span_end"])
                    self.assertIsNone(atom["raw_span_sha256"])

        self.assertEqual(split["status"], DRAFT_STATUS)
        self.assertEqual(len(split["assignments"]), expected_family_count)
        self.assertTrue(split["unassigned_holdout_reason"])
        self.assertEqual(
            {assignment["split"] for assignment in split["assignments"]},
            {"legacy_calibration"},
        )
        self.assertFalse(
            any(assignment["split"] == "holdout" for assignment in split["assignments"])
        )

    def test_missing_historical_episode_fails_closed(self) -> None:
        missing_episode_id = self._required_episode_ids()[0]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._build_snapshot(root, missing_episode_id=missing_episode_id)
            questions, legacy_manifest = self._write_inputs(root)

            with self.assertRaisesRegex(
                ValueError,
                rf"missing historical Episode IDs: \[{missing_episode_id}\]",
            ):
                build_draft(
                    database=database,
                    questions_path=questions,
                    legacy_manifest_path=legacy_manifest,
                )
            gc.collect()


if __name__ == "__main__":
    unittest.main()
