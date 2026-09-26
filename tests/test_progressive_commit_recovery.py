"""Offline recovery tests at the durable learning/checkpoint boundary."""
from __future__ import annotations

import unittest

from memory_demo.retrieval.progressive import ProgressiveRecall
import test_progressive_recall as fixtures


class ProgressiveCommitRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProgressiveRecallIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _rows(self):
        with self.fixture.db.connection() as connection:
            return [tuple(row) for row in connection.execute(
                "SELECT id,weight FROM association ORDER BY id"
            )]

    def _event_count(self, session_id):
        with self.fixture.db.connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM recall_feedback_event WHERE feedback_id=?",
                ("recall:" + session_id,),
            ).fetchone()[0]

    def _capture_precommit_checkpoint(self, *, interrupt=False):
        service = self.fixture.recall
        original = service.feedback.learn_verified
        captured = {}

        def commit_with_failure_point(feedback_id, *args, **kwargs):
            session_id = feedback_id.removeprefix("recall:")
            path = service.sessions._path(session_id)
            captured["path"] = path
            captured["checkpoint"] = path.read_bytes()
            receipt = original(feedback_id, *args, **kwargs)
            if interrupt:
                # The database commit succeeded, but its result never reached
                # the query's learning-state assignment or final checkpoint.
                raise KeyboardInterrupt("interrupt after durable recall learning")
            return receipt

        service.feedback.learn_verified = commit_with_failure_point
        self.addCleanup(setattr, service.feedback, "learn_verified", original)
        return captured

    def test_committed_learning_survives_interrupt_before_receipt_assignment(self):
        self._capture_precommit_checkpoint(interrupt=True)
        fixture = self.fixture

        result = fixture.recall.query(fixture.question, mode="deep")

        self.assertEqual("complete", result["status"], result)
        self.assertTrue(result["complete"])
        self.assertFalse(result["resumable"])
        self.assertEqual("applied", result["learning"]["status"])
        ids = fixture.recall.feedback.valid_learned_edge_ids()
        self.assertEqual(1, len(ids))
        self.assertEqual(ids, set(result["learning"]["association_ids"]))
        self.assertEqual(1, self._event_count(result["session_id"]))
        self.assertEqual([0.8] * 3, [fixture.weight(edge) for edge in fixture.edge_ids])
        self.assertAlmostEqual(0.2, fixture.weight(next(iter(ids))))

    def test_stale_checkpoint_after_hard_stop_recovers_own_commit_without_relearning(self):
        captured = self._capture_precommit_checkpoint()
        fixture = self.fixture
        first = fixture.recall.query(fixture.question, mode="deep")
        self.assertTrue(first["complete"], first)
        before = self._rows()
        model_calls = len(fixture.model.calls)
        embedding_calls = fixture.model.embedding_calls
        # Restore exactly the last checkpoint durable before learning. This
        # models process death after COMMIT without terminating the test runner.
        captured["path"].write_bytes(captured["checkpoint"])
        restored = ProgressiveRecall(fixture.config, fixture.db, fixture.model, clock=fixture.clock)

        resumed = restored.query(fixture.question, mode="deep", resume=first["session_id"])

        self.assertTrue(resumed["complete"], resumed)
        self.assertFalse(resumed["resumable"])
        self.assertEqual("applied", resumed["learning"]["status"])
        self.assertEqual(first["learning"]["association_ids"], resumed["learning"]["association_ids"])
        self.assertEqual(first["answer"], resumed["answer"])
        self.assertEqual(before, self._rows())
        self.assertEqual(1, self._event_count(first["session_id"]))
        self.assertEqual(model_calls, len(fixture.model.calls))
        self.assertEqual(embedding_calls, fixture.model.embedding_calls)
        self.assertEqual(
            restored._snapshot()[-1],
            restored.sessions.read(first["session_id"])["snapshot_hash"],
        )

    def test_own_committed_receipt_does_not_hide_unrelated_graph_changes(self):
        captured = self._capture_precommit_checkpoint()
        fixture = self.fixture
        first = fixture.recall.query(fixture.question, mode="deep")
        self.assertTrue(first["complete"], first)
        captured["path"].write_bytes(captured["checkpoint"])
        with fixture.db.transaction() as connection:
            connection.execute(
                "UPDATE association SET weight=0.81 WHERE id=?",
                (fixture.edge_ids[0],),
            )
        before = self._rows()
        model_calls = len(fixture.model.calls)
        restored = ProgressiveRecall(fixture.config, fixture.db, fixture.model, clock=fixture.clock)

        with self.assertRaisesRegex(ValueError, "changed|snapshot|knowledge"):
            restored.query(fixture.question, mode="deep", resume=first["session_id"])

        self.assertEqual(before, self._rows())
        self.assertEqual(model_calls, len(fixture.model.calls))
        self.assertEqual(1, self._event_count(first["session_id"]))

    def test_created_edge_cannot_be_removed_from_comparison_after_its_confidence_changes(self):
        captured = self._capture_precommit_checkpoint()
        fixture = self.fixture
        first = fixture.recall.query(fixture.question, mode="deep")
        self.assertTrue(first["complete"], first)
        captured["path"].write_bytes(captured["checkpoint"])
        created_id = first["learning"]["association_ids"][0]
        with fixture.db.transaction() as connection:
            connection.execute(
                "UPDATE association SET confidence=0.7 WHERE id=?", (created_id,),
            )
        model_calls = len(fixture.model.calls)
        restored = ProgressiveRecall(fixture.config, fixture.db, fixture.model, clock=fixture.clock)

        with self.assertRaisesRegex(ValueError, "changed|snapshot|knowledge"):
            restored.query(fixture.question, mode="deep", resume=first["session_id"])

        self.assertAlmostEqual(0.7, float(fixture.associations.get(created_id)["confidence"]))
        self.assertAlmostEqual(0.2, fixture.weight(created_id))
        self.assertEqual(model_calls, len(fixture.model.calls))
        self.assertEqual(1, self._event_count(first["session_id"]))


if __name__ == "__main__":
    unittest.main()
