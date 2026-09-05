from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from memory_demo.database import Database
from memory_demo.ingestion.interruption import (
    ImportLeaseConflict,
    ImportProcessLease,
    _pid_is_alive,
    recover_stale_file_runs,
)
from memory_demo.repositories.extraction import ExtractionRepository


class ImportInterruptionRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = Database(self.root / "memory.db")
        self.database.initialize()
        self.extractions = ExtractionRepository(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _start_run(self, source_key: str) -> tuple[int, int]:
        run_id = self.extractions.start_run({}, {}, self.root / "run.jsonl")
        task_id = self.extractions.start_task(
            run_id,
            source_key,
            0,
            "episode_extraction",
            "test-model",
            "test-prompt",
        )
        return run_id, task_id

    def test_recovery_deletes_only_a_matched_incomplete_run(self) -> None:
        run_id, task_id = self._start_run("favor/10001/100011.json")
        ledger = {
            "files": {
                "favor/10001/100011.json": {"status": "running"},
                "favor/10001/100012.json": {"status": "completed"},
            }
        }

        report = recover_stale_file_runs(
            self.database.path,
            ledger,
            reason="test forced process exit",
        )

        self.assertEqual(
            ledger["files"]["favor/10001/100011.json"]["status"], "interrupted"
        )
        self.assertTrue(
            ledger["files"]["favor/10001/100011.json"]["retryable"]
        )
        self.assertEqual(report["deletion_totals"]["deleted_run"], 1)
        self.assertEqual(report["deletion_totals"]["deleted_tasks"], 1)
        with self.database.connection() as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT id FROM extraction_run WHERE id = ?", (run_id,)
                ).fetchone()
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT id FROM extraction_task WHERE id = ?", (task_id,)
                ).fetchone()
            )
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_recovery_preserves_database_completed_run(self) -> None:
        run_id, task_id = self._start_run("favor/10002/100021.json")
        self.extractions.finish_task(task_id, "completed")
        self.extractions.finish_run(
            run_id,
            "completed",
            {"status": "completed", "episodes": 3},
        )
        ledger = {"files": {"favor/10002/100021.json": {"status": "running"}}}

        report = recover_stale_file_runs(
            self.database.path,
            ledger,
            reason="ledger update lost after completed database run",
        )

        record = ledger["files"]["favor/10002/100021.json"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["summary"]["episodes"], 3)
        self.assertEqual(report["reconciled_completed_keys"], ["favor/10002/100021.json"])
        with self.database.connection() as connection:
            self.assertIsNotNone(
                connection.execute(
                    "SELECT id FROM extraction_run WHERE id = ?", (run_id,)
                ).fetchone()
            )

    def test_active_lease_blocks_a_second_importer(self) -> None:
        ledger_path = self.root / "progress.json"
        first = ImportProcessLease.acquire(ledger_path)
        try:
            with self.assertRaises(ImportLeaseConflict):
                ImportProcessLease.acquire(ledger_path)
        finally:
            first.release()
        second = ImportProcessLease.acquire(ledger_path)
        second.release()

    def test_recovery_is_a_noop_without_stale_runs(self) -> None:
        ledger = {"files": {"favor/10003/100031.json": {"status": "completed"}}}

        report = recover_stale_file_runs(
            self.database.path,
            ledger,
            reason="startup consistency check",
        )

        self.assertEqual(report["stale_running_keys"], [])
        self.assertEqual(report["deletion_totals"]["deleted_run"], 0)
        self.assertNotIn("last_recovery", ledger)

    def test_missing_pid_is_not_treated_as_an_active_lease_owner(self) -> None:
        self.assertFalse(_pid_is_alive(999_999_999))


if __name__ == "__main__":
    unittest.main()
