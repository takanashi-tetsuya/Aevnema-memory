from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.import_corpus import (
    begin_ledger_file_attempt,
    failed_retry_preflight,
    select_pending_file_keys,
)
from memory_demo.database import Database
from memory_demo.repositories.extraction import ExtractionRepository
from memory_demo.repositories.source import SourceRepository


class ImportCorpusRetryTests(unittest.TestCase):
    def test_selection_only_retries_requested_terminal_statuses(self) -> None:
        ledger = {
            "files": {
                "completed.json": {"status": "completed"},
                "partial.json": {"status": "partial"},
                "failed.json": {"status": "failed"},
                "interrupted.json": {"status": "interrupted"},
                "new.json": {},
            }
        }
        keys = list(ledger["files"])

        self.assertEqual(
            select_pending_file_keys(
                keys,
                ledger,
                retry_interrupted=False,
                retry_failed=False,
                limit=None,
            ),
            ["new.json"],
        )
        self.assertEqual(
            select_pending_file_keys(
                keys,
                ledger,
                retry_interrupted=True,
                retry_failed=True,
                limit=None,
            ),
            ["failed.json", "interrupted.json", "new.json"],
        )

    def test_failed_attempt_history_is_preserved_without_recursive_nesting(self) -> None:
        previous = {
            "status": "failed",
            "started_at": "old-start",
            "finished_at": "old-finish",
            "summary": {"run_id": 12, "failure_details": [{"stage": "pass1"}]},
            "attempts": [{"status": "older-failed"}],
        }

        attempt = begin_ledger_file_attempt(
            previous,
            prompt_version="current-prompt",
            started_at="new-start",
        )

        self.assertEqual(attempt["status"], "running")
        self.assertEqual(attempt["attempt_number"], 3)
        self.assertEqual(attempt["attempts"][0], {"status": "older-failed"})
        self.assertEqual(attempt["attempts"][1]["summary"]["run_id"], 12)
        self.assertNotIn("attempts", attempt["attempts"][1])
        self.assertEqual(previous["status"], "failed")

    def test_failed_retry_preflight_requires_no_persisted_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root / "retry.db")
            database.initialize()
            extractions = ExtractionRepository(database)
            sources = SourceRepository(database)

            clean_run = extractions.start_run({}, {}, root / "clean.jsonl")
            clean_task = extractions.start_task(
                clean_run, "favor/10001/100011.json", 0, "pass1", "model", "prompt"
            )
            extractions.finish_task(clean_task, "failed", "transport")
            extractions.finish_run(clean_run, "failed", {})

            dirty_run = extractions.start_run({}, {}, root / "dirty.jsonl")
            dirty_task = extractions.start_task(
                dirty_run, "favor/10001/100012.json", 0, "pass1", "model", "prompt"
            )
            extractions.set_source(dirty_task, sources.insert("persisted text"))
            extractions.finish_task(dirty_task, "failed", "transport")
            extractions.finish_run(dirty_run, "failed", {})

            ledger = {
                "files": {
                    "favor/10001/100011.json": {
                        "status": "failed", "summary": {"run_id": clean_run}
                    },
                    "favor/10001/100012.json": {
                        "status": "failed", "summary": {"run_id": dirty_run}
                    },
                }
            }
            issues = failed_retry_preflight(
                database.path,
                ledger,
                list(ledger["files"]),
            )

        self.assertEqual(len(issues), 1)
        self.assertIn("100012.json", issues[0])
        self.assertIn("sources=1", issues[0])


if __name__ == "__main__":
    unittest.main()
