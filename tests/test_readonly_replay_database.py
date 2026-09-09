from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from benchmarks.support.readonly_database import (
    ReadOnlyReplayDatabase,
    ReadOnlyReplayViolation,
)
from memory_demo.repositories.episode import EpisodeRepository


def _sidecar_bytes(path: Path) -> dict[str, bytes | None]:
    return {
        suffix: (
            path.with_name(path.name + suffix).read_bytes()
            if path.with_name(path.name + suffix).exists()
            else None
        )
        for suffix in ("-wal", "-shm", "-journal")
    }


class ReadOnlyReplayDatabaseTests(unittest.TestCase):
    def _seed_database(self, root: Path) -> Path:
        path = root / "frozen-replay.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE episode(id INTEGER PRIMARY KEY, text TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO episode(id, text) VALUES(1, 'frozen')")
            connection.commit()
        finally:
            connection.close()
        return path

    def test_reads_are_row_addressable_and_leave_snapshot_and_sidecars_unchanged(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            path = self._seed_database(Path(temporary))
            before_main = path.read_bytes()
            before_sidecars = _sidecar_bytes(path)
            database = ReadOnlyReplayDatabase(path)

            with database.connection() as connection:
                row = connection.execute(
                    "SELECT id, text FROM episode WHERE id = 1"
                ).fetchone()
                self.assertIsInstance(row, sqlite3.Row)
                self.assertEqual(1, row["id"])
                self.assertEqual("frozen", row["text"])
                self.assertEqual(1, connection.execute("PRAGMA query_only").fetchone()[0])
                self.assertEqual(
                    0, connection.execute("PRAGMA trusted_schema").fetchone()[0]
                )
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("INSERT INTO episode(id, text) VALUES(2, 'nope')")

            self.assertEqual(before_main, path.read_bytes())
            self.assertEqual(before_sidecars, _sidecar_bytes(path))

    def test_read_repository_surface_works_and_writer_surface_fails_immediately(self) -> None:
        with TemporaryDirectory() as temporary:
            path = self._seed_database(Path(temporary))
            database = ReadOnlyReplayDatabase(path)
            repository = EpisodeRepository(database)  # type: ignore[arg-type]

            self.assertEqual(1, repository.count())
            with self.assertRaisesRegex(
                ReadOnlyReplayViolation, "read-only; transaction\\(\\)"
            ):
                database.transaction()

    def test_live_sidecar_is_refused_before_opening_the_snapshot(self) -> None:
        with TemporaryDirectory() as temporary:
            path = self._seed_database(Path(temporary))
            journal = path.with_name(path.name + "-journal")
            journal.write_bytes(b"not a replayable snapshot")
            before_main = path.read_bytes()
            before_sidecars = _sidecar_bytes(path)

            with self.assertRaisesRegex(ReadOnlyReplayViolation, "sidecars"):
                with ReadOnlyReplayDatabase(path).connection():
                    self.fail("a snapshot with a journal sidecar must not open")

            self.assertEqual(before_main, path.read_bytes())
            self.assertEqual(before_sidecars, _sidecar_bytes(path))

    def test_module_has_no_application_model_or_network_dependency(self) -> None:
        import benchmarks.support.readonly_database as module

        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        forbidden_prefixes = (
            "memory_demo",
            "requests",
            "http",
            "urllib",
            "socket",
        )

        self.assertFalse(
            any(
                name == prefix or name.startswith(prefix + ".")
                for name in imported_modules
                for prefix in forbidden_prefixes
            )
        )
        self.assertIn("?mode=ro&immutable=1", source)
        self.assertNotIn("MemoryApplication", source)
        self.assertNotIn("initialize(", source)


if __name__ == "__main__":
    unittest.main()
