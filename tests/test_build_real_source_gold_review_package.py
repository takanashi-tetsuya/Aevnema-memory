from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from benchmarks.build_real_source_gold_review_package import (
    DRAFT_STATUS,
    FROZEN_SOURCE_MANIFEST_SCHEMA,
    PENDING_STATUS,
    build_review_package,
    write_review_package,
)


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class RealSourceGoldReviewPackageTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source_root = root / "story"
        source = source_root / "main" / "example.json"
        source.parent.mkdir(parents=True)
        source.write_text(
            json.dumps(
                {
                    "content": [
                        {"TextJp": "前", "TextCn": "前文", "ScriptKr": "0;A;00;前"},
                        {"TextJp": "原文", "TextCn": "直接主张", "ScriptKr": "0;A;00;主张"},
                        {"TextJp": "后", "TextCn": "后文", "ScriptKr": "0;A;00;后"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        legacy_segment = "\n".join(
            [
                "[source_key: main/example.json]",
                "[segment_index: 0]",
                "",
                "[record: 1]",
                "[script_raw: 0;A;00;主张]",
                "zh-CN: 直接主张",
                "ja: 原文",
                "",
            ]
        )
        snapshot = root / "legacy.sqlite"
        connection = sqlite3.connect(snapshot)
        try:
            connection.executescript(
                """
                CREATE TABLE source(id INTEGER PRIMARY KEY, raw_text TEXT NOT NULL);
                CREATE TABLE episode(
                    id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL,
                    source_key TEXT NOT NULL, segment_index INTEGER NOT NULL, text TEXT NOT NULL
                );
                """
            )
            connection.execute("INSERT INTO source(id, raw_text) VALUES(1, ?)", (legacy_segment,))
            connection.execute(
                "INSERT INTO episode(id, source_id, source_key, segment_index, text) VALUES(1, 1, ?, 0, ?)",
                ("main/example.json", "Legacy summary; it is not direct Source gold."),
            )
            connection.commit()
        finally:
            connection.close()
        draft = {
            "schema_version": "aevnema.source-gold.draft.v1",
            "status": DRAFT_STATUS,
            "usable_for_scoring": False,
            "families": [
                {
                    "family_id": "family",
                    "question_id": "q1",
                    "legacy_required_facts": ["narrow direct claim"],
                    "question_coverage": {"operator": "all_of", "members": ["group"]},
                    "claim_groups": [
                        {
                            "claim_group_id": "group",
                            "coverage_clause": {"operator": "any_of", "members": ["candidate-1"]},
                            "candidate_source_keys": ["main/example.json"],
                            "evidence_atoms": [
                                {
                                    "episode_id": 1,
                                    "source_key": "main/example.json",
                                    "segment_index": 0,
                                    "episode_text_sha256": _sha256_text("Legacy summary; it is not direct Source gold."),
                                    "source_segment_sha256": _sha256_text(legacy_segment),
                                    "legacy_episode_evidence_span_count": 0,
                                    "legacy_episode_evidence_quote_count": 0,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        draft_path = root / "draft.json"
        draft_path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
        return draft_path, source_root, snapshot

    def test_package_keeps_every_atom_pending_and_binds_raw_record_text(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft, source_root, snapshot = self._fixture(root)
            package, manifest, markdown = build_review_package(
                draft_path=draft,
                source_root=source_root,
                legacy_snapshot=snapshot,
            )

        self.assertEqual(DRAFT_STATUS, package["status"])
        self.assertFalse(package["scoring_eligible"])
        self.assertTrue(package["promotion_prohibited"])
        self.assertEqual(FROZEN_SOURCE_MANIFEST_SCHEMA, manifest["schema"])
        self.assertEqual(1, package["atom_count"])
        atom = package["atoms"][0]  # type: ignore[index]
        self.assertEqual(PENDING_STATUS, atom["review_status"])
        self.assertFalse(atom["usable_for_scoring"])
        candidate = atom["candidate_records"][0]
        self.assertEqual("/content/1/TextCn", candidate["record_locator"]["pointer"])
        self.assertEqual("直接主张", candidate["full_record_span"]["text"])
        self.assertEqual(_sha256_text("直接主张"), candidate["full_record_span"]["raw_span_sha256"])
        self.assertIn("Legacy summary (non-authoritative)", markdown)
        self.assertIn("Japanese original", markdown)

    def test_rejects_legacy_hash_drift_and_does_not_overwrite_a_review_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft, source_root, snapshot = self._fixture(root)
            payload = json.loads(draft.read_text(encoding="utf-8"))
            payload["families"][0]["claim_groups"][0]["evidence_atoms"][0]["source_segment_sha256"] = "0" * 64
            draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy source segment hash differs"):
                build_review_package(
                    draft_path=draft,
                    source_root=source_root,
                    legacy_snapshot=snapshot,
                )

            draft, source_root, snapshot = self._fixture(root / "second")
            package, manifest, markdown = build_review_package(
                draft_path=draft,
                source_root=source_root,
                legacy_snapshot=snapshot,
            )
            output = root / "package"
            write_review_package(output, package, manifest, markdown)
            with self.assertRaises(FileExistsError):
                write_review_package(output, package, manifest, markdown)


if __name__ == "__main__":
    unittest.main()
