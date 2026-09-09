from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.freeze_v3_baseline import (
    _atomic_write_bundle,
    main,
    snapshot_record,
    workspace_identity,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


class FreezeV3BaselineTests(unittest.TestCase):
    def _repository(self, root: Path) -> None:
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "baseline-test@example.invalid")
        _git(root, "config", "user.name", "Baseline Test")
        (root / "tracked.txt").write_text("committed baseline\n", encoding="utf-8")
        _git(root, "add", "tracked.txt")
        _git(root, "commit", "-qm", "initial baseline")

    def test_workspace_identity_covers_staged_unstaged_and_untracked_without_content(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            tracked = root / "tracked.txt"
            tracked.write_text("staged replacement\n", encoding="utf-8")
            _git(root, "add", "tracked.txt")
            tracked.write_text("unstaged replacement\n", encoding="utf-8")
            (root / "scratch").mkdir()
            untracked = root / "scratch" / "evidence.bin"
            untracked.write_bytes(b"untracked secret payload")
            other = root / "new.txt"
            other.write_bytes(b"second untracked payload")

            first = workspace_identity(root)
            second = workspace_identity(root)

        self.assertEqual(first, second)
        self.assertTrue(first["tracked_diff_present"])
        self.assertGreater(first["tracked_diff_bytes"], 0)
        self.assertGreater(first["tracked_diff"]["staged"]["bytes"], 0)
        self.assertGreater(first["tracked_diff"]["unstaged"]["bytes"], 0)
        self.assertEqual(2, first["untracked_file_count"])
        self.assertEqual(
            ["new.txt", "scratch/evidence.bin"],
            [entry["path"] for entry in first["untracked_files"]],
        )
        self.assertEqual(
            sha256(b"untracked secret payload").hexdigest(),
            first["untracked_files"][1]["sha256"],
        )
        rendered = json.dumps(first, ensure_ascii=False)
        self.assertNotIn("unstaged replacement", rendered)
        self.assertNotIn("untracked secret payload", rendered)
        self.assertIn("workspace_identity_sha256", first)

    def test_snapshot_rejects_each_sqlite_sidecar_even_when_empty(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "snapshot.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE source(id INTEGER PRIMARY KEY)")
                connection.commit()
            finally:
                connection.close()

            clean = snapshot_record("snapshot", database, root)
            self.assertEqual("snapshot.sqlite", clean["file"])
            self.assertEqual(0, clean["counts"]["source"])
            for suffix in ("-wal", "-shm", "-journal"):
                with self.subTest(suffix=suffix):
                    sidecar = Path(f"{database}{suffix}")
                    sidecar.write_bytes(b"")
                    with self.assertRaisesRegex(RuntimeError, "non-quiescent SQLite"):
                        snapshot_record("snapshot", database, root)
                    sidecar.unlink()

    def test_main_emits_the_content_free_workspace_receipt_after_snapshot_checks(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            database = root / "snapshot.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE source(id INTEGER PRIMARY KEY)")
                connection.commit()
            finally:
                connection.close()
            (root / "untracked.txt").write_text("do not export this body", encoding="utf-8")
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                exit_code = main(
                    [
                        "--output-dir",
                        "receipts",
                        "--snapshot",
                        "core=snapshot.sqlite",
                    ]
                )
            finally:
                os.chdir(previous_directory)

            baseline = json.loads(
                (root / "receipts" / "baseline_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("aevnema.v3.baseline_manifest.v2", baseline["schema"])
        identity = baseline["core_repository"]["workspace_identity"]
        self.assertEqual(
            ["snapshot.sqlite", "untracked.txt"],
            [entry["path"] for entry in identity["untracked_files"]],
        )
        self.assertNotIn("do not export this body", json.dumps(baseline))

    def test_atomic_output_never_overwrites_existing_receipt_when_replace_fails(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "receipts"
            output_dir.mkdir()
            destination = output_dir / "baseline_manifest.json"
            destination.write_bytes(b"previous complete receipt\n")

            with patch(
                "benchmarks.freeze_v3_baseline.os.replace",
                side_effect=OSError("simulated replacement failure"),
            ):
                with self.assertRaises(OSError):
                    _atomic_write_bundle(
                        output_dir,
                        {"baseline_manifest.json": b"new complete receipt\n"},
                    )

            self.assertEqual(b"previous complete receipt\n", destination.read_bytes())
            self.assertEqual([], list(output_dir.glob(".*.tmp")))

            _atomic_write_bundle(
                output_dir,
                {
                    "baseline_manifest.json": b"new complete receipt\n",
                    "snapshot_manifest.json": b"second complete receipt\n",
                },
            )
            self.assertEqual(b"new complete receipt\n", destination.read_bytes())
            self.assertEqual(
                b"second complete receipt\n",
                (output_dir / "snapshot_manifest.json").read_bytes(),
            )

    def test_atomic_bundle_rolls_back_prior_replacements(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "receipts"
            output_dir.mkdir()
            baseline = output_dir / "baseline_manifest.json"
            snapshot = output_dir / "snapshot_manifest.json"
            baseline.write_bytes(b"old baseline\n")
            snapshot.write_bytes(b"old snapshot\n")
            real_replace = os.replace

            def fail_second_replacement(source: str | Path, destination: str | Path) -> None:
                if Path(destination).name == "snapshot_manifest.json":
                    raise OSError("simulated second replacement failure")
                real_replace(source, destination)

            with patch(
                "benchmarks.freeze_v3_baseline.os.replace",
                side_effect=fail_second_replacement,
            ):
                with self.assertRaises(OSError):
                    _atomic_write_bundle(
                        output_dir,
                        {
                            "baseline_manifest.json": b"new baseline\n",
                            "snapshot_manifest.json": b"new snapshot\n",
                        },
                    )

            self.assertEqual(b"old baseline\n", baseline.read_bytes())
            self.assertEqual(b"old snapshot\n", snapshot.read_bytes())
            self.assertEqual([], list(output_dir.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
