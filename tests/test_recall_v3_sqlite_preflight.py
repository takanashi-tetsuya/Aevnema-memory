from __future__ import annotations

import ast
from hashlib import sha256
import inspect
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
import unittest

from benchmarks.support.recall_v3_sqlite_preflight import (
    PREFLIGHT_SCHEMA,
    preflight_sqlite_evidence,
)


_DIMENSION = 4
_VECTOR = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)


def _digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _codes(report: dict[str, object]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}  # type: ignore[index]


class RecallV3SQLitePreflightTests(unittest.TestCase):
    def _seed_snapshot(self, root: Path, *, schema_version: int = 13) -> Path:
        database = root / "frozen.db"
        schema = (
            Path(__file__).resolve().parents[1] / "src" / "memory_demo" / "schema.sql"
        ).read_text(encoding="utf-8")
        connection = sqlite3.connect(database)
        connection.create_function("memory_bigram_tokens", 1, lambda value: "", deterministic=True)
        try:
            connection.executescript(schema)
            connection.execute(
                "INSERT INTO schema_meta(schema_version, created_at) VALUES(?, '2026-09-06T00:00:00+00:00')",
                (schema_version,),
            )
            connection.execute("INSERT INTO source(id, raw_text) VALUES(1, 'synthetic source')")
            connection.execute(
                """
                INSERT INTO paragraph(
                    id, source_id, source_key, segment_index, paragraph_index,
                    text, embedding, created_at
                ) VALUES(1, 1, 'main/example.json', 0, 0, 'synthetic paragraph', ?, 'now')
                """,
                (_VECTOR,),
            )
            for episode_id, segment_index in ((1, 0), (2, 1)):
                connection.execute(
                    """
                    INSERT INTO episode(
                        id, source_id, source_key, segment_index, text, embedding,
                        created_at, updated_at
                    ) VALUES(?, 1, 'main/example.json', ?, 'synthetic episode', ?, 'now', 'now')
                    """,
                    (episode_id, segment_index, _VECTOR),
                )
            for cue_id, cue_kind in ((1, "context"), (2, "need")):
                connection.execute(
                    """
                    INSERT INTO association_cue_prototype(
                        id, domain, cue_kind, model_id, dimension, dtype,
                        vector_blob, text_hash, display_text, source_request_hash, created_at
                    ) VALUES(?, 'story', ?, 'bge-test', ?, 'float32', ?, ?, '', 'request-hash', 'now')
                    """,
                    (cue_id, cue_kind, _DIMENSION, _VECTOR, f"cue-{cue_id}"),
                )
            connection.execute(
                """
                INSERT INTO association(
                    id, from_type, from_id, to_type, to_id, relation_type,
                    relation_key, relation_text, polarity, association_mode,
                    context_cue_id, need_cue_id, lifecycle_state, created_at, updated_at
                ) VALUES(
                    1, 'episode', 1, 'episode', 2, 'retrieval', 'contextual_recall',
                    '', 1, 'contextual_recall', 1, 2, 'active', 'now', 'now'
                )
                """
            )
            connection.commit()
        finally:
            connection.close()
        return database

    @staticmethod
    def _manifests(root: Path) -> tuple[Path, Path]:
        source_manifest = root / "source-manifest.json"
        scope_manifest = root / "scope-manifest.json"
        source_manifest.write_text('{"source":"synthetic"}\n', encoding="utf-8")
        scope_manifest.write_text('{"source_keys":["main/example.json"]}\n', encoding="utf-8")
        return source_manifest, scope_manifest

    def _spec(
        self,
        database: Path,
        source_manifest: Path,
        scope_manifest: Path,
        *,
        schema_version: int = 13,
    ) -> dict[str, object]:
        database_digest = _digest(database)
        source_digest = _digest(source_manifest)
        scope_digest = _digest(scope_manifest)
        return {
            "sqlite_preflight": {
                "database_sha256": database_digest,
                "source_manifest_sha256": source_digest,
                "scope_sha256": scope_digest,
                "schema_version": schema_version,
                "evaluation_as_of": "2026-09-06T00:00:00+00:00",
                "source_keys": [
                    {"source_key": "main/example.json", "require_episode": True}
                ],
                "embedding_space": {
                    "id": "bge-test/normalized-v1",
                    "dimension": _DIMENSION,
                    "model_id": "bge-test",
                    "preprocess_version": "normalized-v1",
                    "normalized": True,
                    "dtype": "float32",
                },
                "expected_counts": {
                    "source": 1,
                    "paragraph": 1,
                    "episode": 2,
                    "concept": 0,
                    "association": 1,
                    "cue_prototype": 2,
                    "contextual_edge": 1,
                },
            },
            "arms": [
                {
                    "arm_id": "contextual",
                    "requires_contextual": True,
                    "candidate_inventory": {"state": "known_empty", "count": 0},
                    "snapshot": {
                        "database_sha256": database_digest,
                        "source_manifest_sha256": source_digest,
                        "scope_sha256": scope_digest,
                        "embedding_space": {
                            "id": "bge-test/normalized-v1",
                            "dimension": _DIMENSION,
                            "model_id": "bge-test",
                            "preprocess_version": "normalized-v1",
                            "normalized": True,
                        },
                    },
                    "contextual": {
                        "cues": [
                            {
                                "cue_id": "context-cue",
                                "database_cue_id": 1,
                                "cue_kind": "context",
                            },
                            {
                                "cue_id": "need-cue",
                                "database_cue_id": 2,
                                "cue_kind": "need",
                            },
                        ],
                        "edges": [
                            {
                                "edge_id": "edge-1",
                                "database_association_id": 1,
                                "target_ref": "episode:2",
                                "cue_ids": ["context-cue", "need-cue"],
                            }
                        ],
                    },
                }
            ],
        }

    def test_complete_database_closure_is_read_only_and_known_empty_is_valid(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            spec = self._spec(database, source_manifest, scope_manifest)
            before = database.read_bytes()

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
            )

            self.assertEqual(before, database.read_bytes())

        self.assertEqual(PREFLIGHT_SCHEMA, report["schema"])
        self.assertEqual("ready", report["status"])
        self.assertTrue(report["can_execute"])
        self.assertEqual(0, report["network_calls"])
        self.assertEqual(0, report["model_calls"])
        self.assertEqual(0, report["database_writes"])
        self.assertEqual("valid_no_hit", report["arms"][0]["execution_state"])
        self.assertEqual(2, report["database"]["counts"]["episode"])
        self.assertEqual([], report["issues"])

    def test_current_v22_schema_contract_is_read_only_and_valid(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root, schema_version=22)
            source_manifest, scope_manifest = self._manifests(root)
            spec = self._spec(
                database,
                source_manifest,
                scope_manifest,
                schema_version=22,
            )
            before = database.read_bytes()

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
            )

            self.assertEqual(before, database.read_bytes())

        self.assertEqual("ready", report["status"])
        self.assertTrue(report["can_execute"])
        self.assertEqual(22, report["database"]["schema_version"])
        self.assertEqual([], report["issues"])

    def test_only_explicit_schema_contracts_are_accepted(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for schema_version in (14, 21, 23):
                with self.subTest(schema_version=schema_version):
                    case_root = root / f"schema-{schema_version}"
                    case_root.mkdir()
                    database = self._seed_snapshot(
                        case_root, schema_version=schema_version
                    )
                    source_manifest, scope_manifest = self._manifests(case_root)
                    spec = self._spec(
                        database,
                        source_manifest,
                        scope_manifest,
                        schema_version=schema_version,
                    )
                    before = database.read_bytes()

                    report = preflight_sqlite_evidence(
                        spec,
                        database,
                        source_manifest=source_manifest,
                        scope_manifest=scope_manifest,
                    )

                    self.assertEqual(before, database.read_bytes())
                    self.assertEqual("invalid_setup", report["status"])
                    self.assertFalse(report["can_execute"])
                    self.assertEqual(schema_version, report["database"]["schema_version"])
                    self.assertEqual({}, report["database"]["counts"])
                    self.assertIn("unsupported_schema_version", _codes(report))
                    self.assertIn("database_schema_version_unsupported", _codes(report))

    def test_spec_and_database_schema_versions_must_match_exactly(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root, schema_version=22)
            source_manifest, scope_manifest = self._manifests(root)
            spec = self._spec(
                database,
                source_manifest,
                scope_manifest,
                schema_version=13,
            )
            before = database.read_bytes()

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
            )

            self.assertEqual(before, database.read_bytes())

        self.assertEqual("invalid_setup", report["status"])
        self.assertFalse(report["can_execute"])
        self.assertEqual(22, report["database"]["schema_version"])
        self.assertEqual({}, report["database"]["counts"])
        self.assertIn("schema_version_mismatch", _codes(report))

    def test_live_sqlite_sidecars_fail_closed_without_opening_snapshot(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in ("-wal", "-shm", "-journal"):
                with self.subTest(suffix=suffix):
                    case_root = root / suffix[1:]
                    case_root.mkdir()
                    database = self._seed_snapshot(case_root)
                    source_manifest, scope_manifest = self._manifests(case_root)
                    spec = self._spec(database, source_manifest, scope_manifest)
                    sidecar = Path(f"{database}{suffix}")
                    sidecar.write_bytes(b"live sidecar")
                    before = database.read_bytes()

                    report = preflight_sqlite_evidence(
                        spec,
                        database,
                        source_manifest=source_manifest,
                        scope_manifest=scope_manifest,
                    )

                    self.assertEqual(before, database.read_bytes())
                    self.assertEqual("invalid_setup", report["status"])
                    self.assertFalse(report["can_execute"])
                    self.assertEqual({}, report["database"]["counts"])
                    self.assertIn(
                        "sqlite_sidecar_snapshot_not_quiescent", _codes(report)
                    )
                    if suffix == "-wal":
                        self.assertIn("wal_snapshot_not_pinned", _codes(report))

    def test_bad_vector_shape_and_missing_source_scope_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            connection = sqlite3.connect(database)
            try:
                connection.execute("UPDATE episode SET embedding = ? WHERE id = 1", (b"bad",))
                connection.commit()
            finally:
                connection.close()
            spec = self._spec(database, source_manifest, scope_manifest)
            spec["sqlite_preflight"]["source_keys"] = ["main/missing.json"]

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertIn("embedding_blob_shape_mismatch", _codes(report))
        self.assertIn("declared_source_key_missing", _codes(report))
        self.assertIn("database_contextual_edge_closure_mismatch", _codes(report))

    def test_missing_external_artifact_or_database_edge_binding_is_not_a_valid_no_hit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            spec = self._spec(database, source_manifest, scope_manifest)
            spec["arms"][0]["contextual"]["edges"][0]["database_association_id"] = 99

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=root / "not-present.json",
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertIn("snapshot_file_missing", _codes(report))
        self.assertIn("database_contextual_edge_not_found", _codes(report))
        self.assertEqual("not_executable", report["arms"][0]["execution_state"])

    def test_pinned_as_of_and_real_anchor_are_required_for_a_declared_edge(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            connection = sqlite3.connect(database)
            try:
                # association has intentionally generic endpoints, so SQLite
                # foreign_key_check cannot catch this logical-anchor defect.
                connection.execute(
                    "UPDATE association SET from_id = 999, expires_at = '2026-09-05T23:59:59+00:00' WHERE id = 1"
                )
                connection.commit()
            finally:
                connection.close()
            spec = self._spec(database, source_manifest, scope_manifest)

            report = preflight_sqlite_evidence(
                spec,
                database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertEqual("2026-09-06T00:00:00+00:00", report["evaluation_as_of"])
        self.assertIn("database_contextual_edge_expired_as_of", _codes(report))
        self.assertIn("database_contextual_edge_anchor_missing", _codes(report))
        self.assertIn("contextual_edge_database_closure_broken", _codes(report))

    def test_module_has_no_application_or_model_import_and_uses_readonly_uri(self) -> None:
        import benchmarks.support.recall_v3_sqlite_preflight as module

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

        self.assertFalse(any(name.startswith("memory_demo") for name in imported_modules))
        self.assertIn("?mode=ro", source)
        self.assertNotIn("MemoryApplication", "\n".join(imported_modules))


if __name__ == "__main__":
    unittest.main()
