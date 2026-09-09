from __future__ import annotations

"""Capture a redacted, read-only v3 baseline manifest.

The tool deliberately records hashes and database checks without copying a
database, reading model payloads, or including account credentials.  It is a
*freeze* receipt, so it fails closed when it cannot prove that an input is
stable: a database with a WAL, SHM, or rollback journal is not a stable main
database snapshot, and a workspace outside Git has no reproducible identity.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sqlite3
import stat
import subprocess
import sys
import tempfile
from typing import Any

from memory_demo.config import AppConfig
from memory_demo.event_log import safe_config_snapshot


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload: Any) -> bytes:
    """Encode receipts deterministically without allowing non-JSON floats."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stat_fingerprint(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _require_regular_file(path: Path) -> os.stat_result:
    """Reject links and special files before a baseline receipt can read them."""

    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(file_stat.st_mode):
        raise RuntimeError(f"refusing symbolic-link input: {path.name}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"refusing non-regular input: {path.name}")
    return file_stat


def _resolve_regular_file(path: Path) -> Path:
    """Resolve an explicitly supplied file only after rejecting a final symlink."""

    _require_regular_file(path)
    return path.resolve(strict=True)


def _stable_file_metadata(path: Path) -> tuple[int, str, tuple[int, int, int, int, int]]:
    """Hash one regular file while detecting a concurrent replacement/change.

    Hash through an already-open descriptor so a later path replacement cannot
    redirect the digest.  The before/after lstat checks make a moving target a
    hard failure instead of silently producing an ambiguous receipt.
    """

    before = _require_regular_file(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    digest = sha256()
    try:
        opened = os.fstat(descriptor)
        if _stat_fingerprint(opened) != _stat_fingerprint(before):
            raise RuntimeError(f"input changed while opening: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    finally:
        if descriptor != -1:
            os.close(descriptor)
    after = _require_regular_file(path)
    if _stat_fingerprint(after) != _stat_fingerprint(before):
        raise RuntimeError(f"input changed while hashing: {path.name}")
    return before.st_size, digest.hexdigest(), _stat_fingerprint(before)


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _git_text(project_root: Path, *arguments: str) -> str | None:
    payload = _git_bytes(project_root, *arguments)
    if payload is None:
        return None
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("Git returned a non-UTF-8 baseline value") from error


def _git_bytes(project_root: Path, *arguments: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bytes(result.stdout)


def _git_repository_root(project_root: Path) -> Path:
    raw_root = _git_text(project_root, "rev-parse", "--show-toplevel")
    if not raw_root:
        raise RuntimeError("cannot capture a baseline outside a Git repository")
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise RuntimeError("Git reported an invalid repository root")
    return root


def _safe_untracked_path(project_root: Path, raw_path: str) -> Path:
    """Resolve a Git-reported path without allowing link/traversal escapes."""

    if (
        not raw_path
        or raw_path.startswith(("/", "\\"))
        or any(character == "\x7f" or ord(character) < 32 for character in raw_path)
    ):
        raise RuntimeError("refusing unsafe untracked Git path")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or relative.drive
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("refusing unsafe untracked Git path")
    candidate = project_root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=True).relative_to(project_root.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError("untracked Git path escapes the workspace") from error
    _require_regular_file(candidate)
    return candidate


def _untracked_file_metadata(project_root: Path) -> list[dict[str, Any]]:
    raw_listing = _git_bytes(
        project_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    if raw_listing is None:
        raise RuntimeError("cannot enumerate untracked workspace files")
    raw_entries = [entry for entry in raw_listing.split(b"\0") if entry]
    try:
        paths = sorted(entry.decode("utf-8") for entry in raw_entries)
    except UnicodeDecodeError as error:
        raise RuntimeError("refusing non-UTF-8 untracked Git path") from error
    if len(paths) != len(set(paths)):
        raise RuntimeError("Git returned duplicate untracked workspace paths")

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _safe_untracked_path(project_root, raw_path)
        byte_count, digest, _ = _stable_file_metadata(path)
        records.append(
            {
                "path": raw_path.replace("\\", "/"),
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    return records


def workspace_identity(project_root: Path) -> dict[str, Any]:
    """Return a content-free identity for both tracked and untracked work.

    Staged and unstaged diffs are recorded separately: collapsing them to
    ``git diff HEAD`` would lose an index-only change whose working tree was
    later restored.  Untracked regular files are listed individually with
    path, byte count, and digest; their contents never enter the receipt.
    """

    repository_root = _git_repository_root(project_root)
    head = _git_text(repository_root, "rev-parse", "--verify", "HEAD")
    if not head:
        raise RuntimeError("cannot capture a baseline before the first Git commit")
    staged_diff = _git_bytes(
        repository_root,
        "diff",
        "--cached",
        "--binary",
        "--no-ext-diff",
        "--full-index",
        "HEAD",
        "--",
    )
    worktree_diff = _git_bytes(
        repository_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--full-index",
        "--",
    )
    if staged_diff is None or worktree_diff is None:
        raise RuntimeError("cannot capture tracked workspace changes")
    tracked_diff_receipt = {
        "staged": {
            "bytes": len(staged_diff),
            "sha256": sha256(staged_diff).hexdigest(),
        },
        "unstaged": {
            "bytes": len(worktree_diff),
            "sha256": sha256(worktree_diff).hexdigest(),
        },
    }
    untracked_files = _untracked_file_metadata(repository_root)
    untracked_manifest_sha256 = sha256(_canonical_json_bytes(untracked_files)).hexdigest()
    identity: dict[str, Any] = {
        "schema": "aevnema.v3.workspace_identity.v1",
        "head": head,
        "tracked_diff_present": bool(staged_diff or worktree_diff),
        "tracked_diff": tracked_diff_receipt,
        "tracked_diff_bytes": len(staged_diff) + len(worktree_diff),
        "tracked_diff_sha256": sha256(
            _canonical_json_bytes(tracked_diff_receipt)
        ).hexdigest(),
        "untracked_file_count": len(untracked_files),
        "untracked_files_sha256": untracked_manifest_sha256,
        "untracked_files": untracked_files,
    }
    identity["workspace_identity_sha256"] = sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    return identity


def _table_count(connection: sqlite3.Connection, table: str) -> int | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return None
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def snapshot_record(
    label: str, path: Path, project_root: Path
) -> dict[str, Any]:
    resolved = _resolve_regular_file(path)
    starting_stat = _require_regular_file(resolved)
    _assert_quiescent_sqlite(resolved)
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        payload = {
            "label": label,
            "file": _project_relative(resolved, project_root),
            "quick_check": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
            "foreign_key_errors": len(
                connection.execute("PRAGMA foreign_key_check").fetchall()
            ),
            "counts": {
                table: _table_count(connection, table)
                for table in ("source", "episode", "concept", "association", "paragraph")
            },
        }
    finally:
        connection.close()
    _assert_quiescent_sqlite(resolved)
    byte_count, digest, final_fingerprint = _stable_file_metadata(resolved)
    if final_fingerprint != _stat_fingerprint(starting_stat):
        raise RuntimeError(f"SQLite snapshot changed during validation: {resolved.name}")
    return {**payload, "bytes": byte_count, "sha256": digest}


def _assert_quiescent_sqlite(path: Path) -> None:
    for suffix, label in (("-wal", "WAL"), ("-shm", "SHM"), ("-journal", "journal")):
        sidecar = Path(f"{path}{suffix}")
        if os.path.lexists(sidecar):
            raise RuntimeError(
                f"refusing non-quiescent SQLite database with an active {label}: {path.name}"
            )


def _parse_snapshot(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--snapshot must use LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _render_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Ask POSIX filesystems to persist a replacement when they support it."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_staged_file(directory: Path, name: str, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_write_bundle(output_dir: Path, documents: dict[str, bytes]) -> None:
    """Stage every receipt before atomically replacing any destination file.

    Each destination is replaced only after its complete bytes were written and
    synced in the same directory.  Temporary files are removed on every error,
    so a failed capture cannot leave a truncated JSON receipt behind.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        for name in sorted(documents):
            target = output_dir / name
            if target.parent != output_dir or target.name != name:
                raise ValueError("output document name must be a single filename")
            staged.append((target, _write_staged_file(output_dir, name, documents[name])))
            if os.path.lexists(target):
                _require_regular_file(target)
                backups[target] = _write_staged_file(
                    output_dir, f"{name}.backup", target.read_bytes()
                )
            else:
                backups[target] = None
        for target, temporary in staged:
            os.replace(temporary, target)
            replaced.append(target)
            _fsync_directory(output_dir)
    except BaseException:
        for target in reversed(replaced):
            backup = backups[target]
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
                backups[target] = None
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a read-only, credential-free v3 baseline manifest"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--snapshot",
        type=_parse_snapshot,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="A benchmark manifest to hash; may be repeated.",
    )
    parser.add_argument(
        "--chat-repository",
        type=Path,
        default=None,
        help="Optional independently frozen chat repository; no files are changed.",
    )
    parser.add_argument(
        "--test-command",
        action="append",
        default=[],
        help="A redacted, reproducible test command; may be repeated.",
    )
    parser.add_argument("--tests-collected", type=int)
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-failed", type=int, default=0)
    parser.add_argument("--tests-skipped", type=int, default=0)
    args = parser.parse_args(argv)

    if (args.tests_collected is None) != (args.tests_passed is None):
        raise ValueError(
            "--tests-collected and --tests-passed are both required when recording tests"
        )

    project_root = _git_repository_root(Path.cwd().resolve())
    output_dir = args.output_dir.resolve()
    config = AppConfig.from_env(args.env_file)
    snapshot_inputs = sorted(
        args.snapshot,
        key=lambda item: (item[0], item[1].absolute().as_posix()),
    )
    labels = [label for label, _ in snapshot_inputs]
    if len(labels) != len(set(labels)):
        raise ValueError("snapshot labels must be unique")
    snapshots = [
        snapshot_record(label, path, project_root) for label, path in snapshot_inputs
    ]
    manifests: list[dict[str, Any]] = []
    for path in sorted(
        args.manifest, key=lambda candidate: candidate.absolute().as_posix()
    ):
        resolved = _resolve_regular_file(path)
        byte_count, digest, _ = _stable_file_metadata(resolved)
        manifests.append(
            {
                "file": _project_relative(resolved, project_root),
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    core_identity = workspace_identity(project_root)
    chat_payload: dict[str, Any] = {"status": "not_supplied"}
    if args.chat_repository is not None:
        chat_root = args.chat_repository.resolve()
        if not chat_root.is_dir():
            raise NotADirectoryError(chat_root)
        chat_identity = workspace_identity(chat_root)
        chat_payload = {
            "status": "captured",
            "head": chat_identity["head"],
            "dirty": bool(
                chat_identity["tracked_diff_present"]
                or chat_identity["untracked_file_count"]
            ),
            "workspace_identity": chat_identity,
        }

    captured_at = _utc_now()
    baseline_manifest = {
        "schema": "aevnema.v3.baseline_manifest.v2",
        "captured_at": captured_at,
        "core_repository": {
            "head": core_identity["head"],
            "dirty": bool(
                core_identity["tracked_diff_present"]
                or core_identity["untracked_file_count"]
            ),
            "workspace_identity": core_identity,
        },
        "chat_repository": chat_payload,
        "benchmark_manifests": manifests,
    }
    environment = {
        "schema": "aevnema.v3.environment.redacted.v1",
        "captured_at": captured_at,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "requests", "pytest")
        },
        "config": safe_config_snapshot(config),
    }
    snapshot_manifest = {
        "schema": "aevnema.v3.snapshot_manifest.v1",
        "captured_at": captured_at,
        "snapshots": snapshots,
    }
    documents = {
        "baseline_manifest.json": _render_json(baseline_manifest),
        "environment.redacted.json": _render_json(environment),
        "snapshot_manifest.json": _render_json(snapshot_manifest),
    }
    if args.tests_collected is not None or args.test_command:
        if args.tests_collected is None or args.tests_passed is None:  # pragma: no cover
            raise AssertionError("validated immediately after argument parsing")
        test_baseline = {
            "schema": "aevnema.v3.test_baseline.v1",
            "captured_at": captured_at,
            "runner": "unittest",
            "python_version": platform.python_version(),
            "interpreter": Path(sys.executable).name,
            "collected": args.tests_collected,
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "skipped": args.tests_skipped,
        }
        documents["test-baseline.json"] = _render_json(test_baseline)
        documents["test-commands.txt"] = (
            "\n".join(args.test_command) + ("\n" if args.test_command else "")
        ).encode("utf-8")
    _atomic_write_bundle(output_dir, documents)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "snapshots": len(snapshots),
                "manifests": len(manifests),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
