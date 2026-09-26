from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from memory_demo.associations.feedback import (
    RecallFeedbackService, SourceEvidence, VerifiedRecallLink, source_sha256,
)
from memory_demo.database import Database, transaction_liveness
from memory_demo.repositories import AssociationRepository, EpisodeRepository, SourceRepository
from memory_demo.types import AssociationDraft, EpisodeDraft


class RecallFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Database(Path(self.temp.name) / "memory.db")
        self.db.initialize()
        self.service = RecallFeedbackService(self.db)
        self.associations = AssociationRepository(self.db)
        self.sources = SourceRepository(self.db)
        self.episodes = EpisodeRepository(self.db)
        self.raw = ["甲收到了信。她决定求助。", "乙看完信后帮助了甲。", "丙准备了交通工具。"]
        self.source_ids = [self.sources.insert(text) for text in self.raw]
        self.episode_ids = [
            self.episodes.insert(source_id, f"main/{index}.json", 0,
                                 EpisodeDraft(text), b"\0" * 16)
            for index, (source_id, text) in enumerate(zip(self.source_ids, self.raw))
        ]
        self.edge_ids = [self.associations.upsert(AssociationDraft(
            "episode", self.episode_ids[index], "episode", self.episode_ids[index + 1],
            "semantic", f"test-{index}", "已有联系", weight=0.4, confidence=0.7,
        )) for index in range(2)]

    def evidence(self, index):
        text = self.raw[index]
        return SourceEvidence(self.episode_ids[index], self.source_ids[index],
                              source_sha256(text), 0, len(text), text)

    def link(self, first=0, second=1):
        return VerifiedRecallLink(
            self.episode_ids[first], self.episode_ids[second],
            (self.evidence(first), self.evidence(second)),
            "test-independent-source-review", "Reviewed the letter and response in both sources.",
            verified=True,
        )

    def weight(self, association_id):
        return float(self.associations.get(association_id)["weight"])

    def event_count(self):
        with self.db.connection() as connection:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='recall_feedback_event'"
            ).fetchone()
            return 0 if not present else connection.execute(
                "SELECT COUNT(*) FROM recall_feedback_event"
            ).fetchone()[0]

    def test_user_feedback_only_changes_attributed_weights_without_usage_or_decay(self):
        before = dict(self.associations.get(self.edge_ids[0]))
        result = self.service.apply_user_feedback("positive-1", [self.edge_ids[0]], positive=True)
        self.assertTrue(result.applied)
        self.assertAlmostEqual(self.weight(self.edge_ids[0]), 0.52)
        self.assertAlmostEqual(self.weight(self.edge_ids[1]), 0.4)
        after = dict(self.associations.get(self.edge_ids[0]))
        for field in ("evidence_count", "confidence", "use_count", "utility_weight",
                      "utility_successes", "utility_query_hashes", "last_used", "expires_at"):
            self.assertEqual(before[field], after[field], field)
        self.service.apply_user_feedback("negative-1", [self.edge_ids[0]], positive=False)
        self.assertAlmostEqual(self.weight(self.edge_ids[0]), 0.416)
        self.assertEqual(self.event_count(), 2)

    def test_retry_is_idempotent_and_order_and_duplicate_ids_do_not_matter(self):
        initial = self.service.apply_user_feedback("same", self.edge_ids, positive=True)
        retry = self.service.apply_user_feedback(
            "same", list(reversed(self.edge_ids)) + self.edge_ids, positive=True,
        )
        self.assertFalse(retry.applied)
        self.assertEqual(initial.changes, retry.changes)
        self.assertEqual(retry.association_ids, tuple(self.edge_ids))
        self.assertAlmostEqual(self.weight(self.edge_ids[0]), 0.52)
        self.assertEqual(self.event_count(), 1)
        with self.assertRaisesRegex(ValueError, "different feedback payload"):
            self.service.apply_user_feedback("same", self.edge_ids, positive=False)

    def test_learned_shortcut_is_simple_source_bound_and_not_double_rewarded(self):
        result = self.service.learn_verified("learn-1", [self.link()], association_ids=self.edge_ids[:1])
        created = [change for change in result.changes if change.created]
        self.assertEqual(len(created), 1)
        shortcut = self.associations.get(created[0].association_id)
        self.assertEqual(shortcut["weight"], 0.2)
        self.assertEqual(shortcut["confidence"], 1.0)
        self.assertEqual(shortcut["association_mode"], "simple_recall")
        self.assertEqual(shortcut["claim_level"], "retrieval_only")
        self.assertIsNone(shortcut["cue_embedding"])
        self.assertEqual(shortcut["relation_text"], "")
        self.assertAlmostEqual(self.weight(self.edge_ids[0]), 0.52)
        self.assertAlmostEqual(self.weight(self.edge_ids[1]), 0.4)
        self.assertEqual(self.service.valid_learned_edge_ids(), {int(shortcut["id"])})
        with self.db.connection() as connection:
            row = connection.execute("SELECT * FROM recall_link_evidence").fetchone()
            self.assertIn(source_sha256(self.raw[0]), row["evidence_json"])
            self.assertIn(self.raw[0], row["evidence_json"])
            self.assertEqual(row["feedback_id"], "learn-1")

    def test_learn_retry_and_reverse_pair_reinforcement(self):
        initial = self.service.learn_verified("learn", [self.link()])
        shortcut_id = initial.association_ids[0]
        retry = self.service.learn_verified("learn", [self.link()])
        self.assertFalse(retry.applied)
        self.assertEqual(retry.changes, initial.changes)
        self.assertEqual(self.weight(shortcut_id), 0.2)
        next_result = self.service.learn_verified(
            "learn-again", [self.link(1, 0)], association_ids=[shortcut_id],
        )
        self.assertEqual(next_result.association_ids, (shortcut_id,))
        self.assertAlmostEqual(self.weight(shortcut_id), 0.36)
        self.assertFalse(next_result.changes[0].created)

    def test_changed_source_rejects_learning_and_keeps_all_weights(self):
        reviewed = self.link()
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[1],))
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.service.learn_verified("stale", [reviewed], association_ids=self.edge_ids)
        self.assertEqual(self.event_count(), 0)
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.4, 0.4])

    def test_changed_source_invalidates_existing_shortcut_and_feedback_and_retry(self):
        reviewed = self.link()
        learned = self.service.learn_verified("learn", [reviewed])
        shortcut_id = learned.association_ids[0]
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[0],))
        self.assertEqual(self.service.valid_learned_edge_ids(), set())
        for positive in (True, False):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.service.apply_user_feedback(f"stale-{positive}", [shortcut_id], positive=positive)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.service.learn_verified("learn", [reviewed])
        self.assertEqual(self.weight(shortcut_id), 0.2)
        self.assertEqual(self.event_count(), 1)

    def test_exact_span_is_checked_even_with_a_valid_source_hash(self):
        bad = replace(self.evidence(0), quote="乙" + self.raw[0][1:])
        link = replace(self.link(), evidence=(bad, self.evidence(1)))
        with self.assertRaisesRegex(ValueError, "exact span"):
            self.service.learn_verified("fabricated-quote", [link])
        self.assertEqual(self.event_count(), 0)

    def test_episode_source_rebinding_invalidates_review(self):
        reviewed = self.link()
        with self.db.transaction() as connection:
            connection.execute("UPDATE episode SET source_id=? WHERE id=?", (self.source_ids[2], self.episode_ids[0]))
        with self.assertRaisesRegex(ValueError, "stated episode"):
            self.service.learn_verified("rebound", [reviewed])

    def test_partial_write_failure_rolls_back_weights_and_event(self):
        self.service.ensure_schema()
        with self.db.transaction() as connection:
            connection.execute(f"""
                CREATE TRIGGER feedback_test_failure BEFORE INSERT ON recall_feedback_change
                WHEN NEW.association_id = {self.edge_ids[1]}
                BEGIN SELECT RAISE(ABORT, 'injected feedback failure'); END
            """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected feedback failure"):
            self.service.apply_user_feedback("rollback", self.edge_ids, positive=True)
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.4, 0.4])
        self.assertEqual(self.event_count(), 0)

    def test_learning_write_failure_rolls_back_shortcut_and_source_receipt(self):
        self.service.ensure_schema()
        with self.db.transaction() as connection:
            connection.execute("""
                CREATE TRIGGER feedback_test_failure BEFORE INSERT ON recall_feedback_change
                BEGIN SELECT RAISE(ABORT, 'injected learning failure'); END
            """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected learning failure"):
            self.service.learn_verified("rollback", [self.link()])
        with self.db.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM association").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM recall_link_evidence").fetchone()[0], 0)
        self.assertEqual(self.event_count(), 0)

    def test_commit_liveness_failure_rolls_back_learning(self):
        calls = 0

        def require_live():
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise TimeoutError("request ended before commit")

        with transaction_liveness(require_live), self.assertRaises(TimeoutError):
            self.service.learn_verified("late", [self.link()])
        self.assertEqual(self.event_count(), 0)
        self.assertEqual(self.service.valid_learned_edge_ids(), set())

    def test_validity_read_does_not_create_schema(self):
        with self.db.connection() as connection:
            before = list(connection.execute("SELECT name FROM sqlite_master ORDER BY name"))
        self.assertEqual(self.service.valid_learned_edge_ids(), set())
        with self.db.connection() as connection:
            after = list(connection.execute("SELECT name FROM sqlite_master ORDER BY name"))
        self.assertEqual(before, after)

    def test_empty_unknown_and_non_integer_ids_fail_without_weight_changes(self):
        for ids in ([], [0], [-1], [True], ["1"], [1.0], "1", None, [999999]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.service.apply_user_feedback("invalid", ids, positive=True)
        for feedback_id in ("", " ", None, "x" * 257):
            with self.subTest(feedback_id=feedback_id), self.assertRaises(ValueError):
                self.service.apply_user_feedback(feedback_id, self.edge_ids, positive=True)
        self.assertEqual(self.event_count(), 0)
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.4, 0.4])

    def test_invalid_rate_or_verdict_cannot_create_feedback(self):
        for rate in (float("nan"), float("inf"), 0, -0.1, 1.1, True, "0.2"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.service.apply_user_feedback("bad-rate", self.edge_ids, positive=True, learning_rate=rate)
        with self.assertRaises(ValueError):
            self.service.apply_user_feedback("bad-verdict", self.edge_ids, positive="yes")
        with self.assertRaisesRegex(ValueError, "explicit successful"):
            replace(self.link(), verified=False)
        with self.assertRaisesRegex(ValueError, "both shortcut endpoints"):
            replace(self.link(), evidence=(self.evidence(0),))
        self.assertEqual(self.event_count(), 0)

    def test_empty_or_duplicate_link_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.service.learn_verified("empty", [])
        with self.assertRaisesRegex(ValueError, "only once"):
            self.service.learn_verified("double", [self.link(), self.link(1, 0)])
        self.assertEqual(self.event_count(), 0)

    def test_legacy_neighbors_filters_stale_shortcuts_before_limit_and_keeps_valid_links(self):
        valid_id = self.service.learn_verified("valid", [self.link()], learning_rate=0.8).association_ids[0]
        stale_id = self.service.learn_verified("stale", [self.link(0, 2)], learning_rate=0.9).association_ids[0]
        root = self.episode_ids[0]
        self.assertEqual([int(row["id"]) for row in self.associations.neighbors("episode", root, limit=2)],
                         [stale_id, valid_id])
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[2],))
        self.assertEqual([int(row["id"]) for row in self.associations.neighbors("episode", root, limit=1)], [valid_id])
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[1],))
        rows = self.associations.neighbors("episode", root, limit=1)
        self.assertEqual([int(row["id"]) for row in rows], [self.edge_ids[0]])

    def test_legacy_neighbors_many_filters_before_each_limit_and_refills_from_existing_modes(self):
        valid_id = self.service.learn_verified("valid", [self.link()], learning_rate=0.8).association_ids[0]
        stale_id = self.service.learn_verified("stale", [self.link(0, 2)], learning_rate=0.9).association_ids[0]
        before = self.associations.neighbors_many("episode", self.episode_ids, limit=2)
        self.assertEqual([int(row["id"]) for row in before[self.episode_ids[0]]], [stale_id, valid_id])
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[2],))
        after = self.associations.neighbors_many("episode", self.episode_ids, limit=1)
        self.assertEqual([int(row["id"]) for row in after[self.episode_ids[0]]], [valid_id])
        self.assertEqual([int(row["id"]) for row in after[self.episode_ids[1]]], [valid_id])
        self.assertEqual([int(row["id"]) for row in after[self.episode_ids[2]]], [self.edge_ids[1]])
        self.assertNotIn(stale_id, {int(row["id"]) for rows in after.values() for row in rows})
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '修订' WHERE id=?", (self.source_ids[1],))
        final = self.associations.neighbors_many("episode", self.episode_ids, limit=1)
        self.assertEqual([int(row["id"]) for row in final[self.episode_ids[0]]], [self.edge_ids[0]])
        self.assertTrue(all(int(row["id"]) in self.edge_ids for rows in final.values() for row in rows))


if __name__ == "__main__":
    unittest.main()
