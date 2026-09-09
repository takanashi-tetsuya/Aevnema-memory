from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks.control_cache import (
    SourceValidatedCacheMiss,
    SourceValidatedEvidenceCache,
)


RAW_SOURCE = """[source_key: main/test.json]

[record: 1]
[speaker_raw: Alice]
[script_raw: #na;Alice;支持原组织。]
zh-CN: Alice 支持原组织。
ja: Alice は元の組織を支持する。
"""
PROJECTED_QUOTE = """[record: 1]
[speaker_raw: Alice]
zh-CN: Alice 支持原组织。
ja: Alice は元の組織を支持する。
"""


class SourceValidatedControlCacheTests(unittest.TestCase):
    def _database(self, directory: str) -> Path:
        path = Path(directory) / "source.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE source(id INTEGER PRIMARY KEY, raw_text TEXT NOT NULL);
                CREATE TABLE episode(
                    id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL,
                    source_key TEXT NOT NULL, segment_index INTEGER NOT NULL,
                    text TEXT NOT NULL, participants_json TEXT NOT NULL,
                    evidence_quotes_json TEXT NOT NULL, evidence_origin TEXT NOT NULL,
                    epistemic_status TEXT NOT NULL
                );
                """
            )
            connection.execute("INSERT INTO source VALUES(1, ?)", (RAW_SOURCE,))
            connection.execute(
                "INSERT INTO episode VALUES(18, 1, 'main/test.json', 7, ?, ?, ?, 'source', 'asserted')",
                (
                    "Alice 支持原组织。",
                    json.dumps(["Alice"]),
                    json.dumps([PROJECTED_QUOTE]),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return path

    def _seed(self, database: Path) -> SourceValidatedEvidenceCache:
        return SourceValidatedEvidenceCache.seed(
            database=database,
            question="Alice 支持哪个组织？",
            domain="knowledge",
            scope_hash="scope:test",
            episode_ids=[18],
            source_excerpt_chars=2400,
        )

    def test_replay_revalidates_source_and_never_carries_answer_or_edge_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            cache = self._seed(database)
            result = cache.replay(
                database=database,
                question="Alice 支持哪个组织？",
                domain="knowledge",
                scope_hash="scope:test",
                source_excerpt_chars=2400,
            )
        self.assertEqual("hit_source_validated", result["cache_status"])
        self.assertEqual("source_bound", result["materialized_source_refs"][0]["source_evidence_delivery"])
        self.assertFalse(cache.as_dict()["answer_prose_stored"])
        self.assertFalse(cache.as_dict()["association_state_stored"])
        self.assertFalse(result["executed_modules"]["contextual_matcher"])
        self.assertIn("[script_raw:", result["materialized_source_refs"][0]["source_excerpt"])

    def test_replay_rejects_changed_source_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            cache = self._seed(database)
            with self.assertRaisesRegex(SourceValidatedCacheMiss, "source_excerpt_budget_changed"):
                cache.replay(
                    database=database,
                    question="Alice 支持哪个组织？",
                    domain="knowledge",
                    scope_hash="scope:test",
                    source_excerpt_chars=1200,
                )
            connection = sqlite3.connect(database)
            try:
                connection.execute("UPDATE source SET raw_text = raw_text || '更改'")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(SourceValidatedCacheMiss, "source_or_episode_revision_changed"):
                cache.replay(
                    database=database,
                    question="Alice 支持哪个组织？",
                    domain="knowledge",
                    scope_hash="scope:test",
                    source_excerpt_chars=2400,
                )

    def test_replay_requires_the_exact_question_scope_and_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            cache = self._seed(database)
            with self.assertRaisesRegex(SourceValidatedCacheMiss, "exact_request_scope_or_domain_mismatch"):
                cache.replay(
                    database=database,
                    question="Alice 支持哪个团体？",
                    domain="knowledge",
                    scope_hash="scope:test",
                    source_excerpt_chars=2400,
                )


if __name__ == "__main__":
    unittest.main()
