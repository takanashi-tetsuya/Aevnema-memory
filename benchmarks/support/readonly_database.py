from __future__ import annotations

"""A deliberately small SQLite adapter for frozen V3 replay reads.

This is not a replacement for the application's ``Database`` class.  It
opens a pre-existing, quiescent SQLite snapshot only, exposes the familiar
``connection()`` context manager for read repositories, and deliberately has
no write or schema-initialisation surface.  In particular it must not be used
for imports, migrations, association learning, or benchmark result storage.

The adapter uses both SQLite's explicit ``mode=ro`` URI and connection-local
``query_only``.  ``immutable=1`` avoids journal/lock side effects; therefore a
snapshot with SQLite sidecars is refused rather than silently replayed without
its WAL contents.  A V3 runner can attest the main-file digest separately.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator, NoReturn


class ReadOnlyReplayViolation(RuntimeError):
    """Raised when code attempts a mutation through a frozen replay adapter."""


class ReadOnlyReplayDatabase:
    """Repository-compatible, read-only connection factory for a frozen DB.

    ``path`` is deliberately kept as a public ``Path`` attribute because the
    read-only repository/query helpers only need ``path``, ``connection()``,
    and (for accidental writer use) ``transaction()``.  It never creates a
    parent directory, enables WAL, registers write-capable hooks, or runs a
    schema migration.
    """

    _SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _snapshot_path(self) -> Path:
        """Return a resolvable regular file with no live SQLite sidecars.

        ``immutable=1`` is appropriate only for a self-contained main DB.  A
        WAL or rollback journal means the visible bytes may not be the frozen
        state to replay, so fail closed before opening SQLite.
        """
        try:
            resolved = self.path.resolve(strict=True)
        except OSError as exc:
            raise ReadOnlyReplayViolation(
                f"frozen replay database is not available: {self.path}"
            ) from exc
        if not resolved.is_file():
            raise ReadOnlyReplayViolation(
                f"frozen replay database must be a regular file: {resolved}"
            )
        sidecars = [
            str(resolved.with_name(resolved.name + suffix))
            for suffix in self._SIDECAR_SUFFIXES
            if resolved.with_name(resolved.name + suffix).exists()
        ]
        if sidecars:
            raise ReadOnlyReplayViolation(
                "frozen replay database has live SQLite sidecars; "
                "supply a quiescent snapshot instead"
            )
        return resolved

    @staticmethod
    def _readonly_uri(path: Path) -> str:
        """Build the only connection URI this adapter permits."""
        return path.as_uri() + "?mode=ro&immutable=1"

    def connect(self) -> sqlite3.Connection:
        """Open one SQLite read connection without creating sidecars."""
        snapshot = self._snapshot_path()
        connection = sqlite3.connect(
            self._readonly_uri(snapshot),
            uri=True,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            # These are connection-local defences.  ``mode=ro`` remains the
            # primary file-open guarantee; query_only catches accidental DML
            # even if a caller retains the native connection object.
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
        except Exception:
            connection.close()
            raise
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived row-addressable SQLite read connection."""
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def transaction(self) -> NoReturn:
        """Reject writer-shaped repository calls before opening SQLite."""
        raise ReadOnlyReplayViolation(
            "frozen V3 replay is read-only; transaction() is not available"
        )
