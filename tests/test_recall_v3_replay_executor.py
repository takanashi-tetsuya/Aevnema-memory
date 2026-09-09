from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
import unittest

from benchmarks.run_recall_v3 import (
    DATABASE_MANIFEST_SCHEMA,
    FROZEN_INPUT_SCHEMA,
    REQUEST_MANIFEST_SCHEMA,
    REPLAY_RESULT_SCHEMA,
    REQUEST_RECORD_SCHEMA,
    RUN_SPEC_SCHEMA,
    VECTOR_MANIFEST_SCHEMA,
    execute_replay,
)


_DIMENSION = 4
_VECTOR = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)


def _digest_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest_document(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _codes(report: dict[str, object]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}  # type: ignore[index]


def _gold(*, status: str = "approved_for_scoring") -> dict[str, object]:
    return {
        "status": status,
        "usable_for_scoring": True,
        "families": [
            {
                "family_id": "family-a",
                "usable_for_scoring": True,
                "claim_groups": [
                    {
                        "claim_group_id": "claim-a",
                        "usable_for_scoring": True,
                        "evidence_atoms": [
                            {"atom_id": "atom-a", "usable_for_scoring": True}
                        ],
                    }
                ],
            }
        ],
    }


def _split() -> dict[str, object]:
    return {
        "status": "approved_for_scoring",
        "assignments": [{"family_id": "family-a", "split": "holdout"}],
    }


class _FakeV3Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, question: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((question, dict(kwargs)))
        return {
            "query_plan_frozen": False,
            "episode_ids": [2],
            "candidate_episode_ids": [1, 2],
            "association_ids": [1],
            "contextual_association": {
                "enabled": True,
                "backend": "contextual_double_key_contribution_selector_v3",
                "compatibility_projection": "v3_contribution_selector",
                "target_gate": {
                    "stage": "target_checked_before_endpoint_cap_v1",
                    "outcomes": [],
                },
                "contribution_selector": {"selected_episode_ids": [2]},
                "selected_count": 1,
            },
        }


class _MissingMechanismEngine(_FakeV3Engine):
    def query(self, question: str, **kwargs: object) -> dict[str, object]:
        result = super().query(question, **kwargs)
        result["contextual_association"] = {"enabled": True}
        return result


class _WritingEngine(_FakeV3Engine):
    def __init__(self, database: Path) -> None:
        super().__init__()
        self.database = database

    def query(self, question: str, **kwargs: object) -> dict[str, object]:
        result = super().query(question, **kwargs)
        connection = sqlite3.connect(self.database)
        connection.create_function(
            "memory_bigram_tokens", 1, lambda value: "", deterministic=True
        )
        try:
            connection.execute("UPDATE source SET raw_text = 'mutated' WHERE id = 1")
            connection.commit()
        finally:
            connection.close()
        return result


class RecallV3ReplayExecutorTests(unittest.TestCase):
    def _seed_snapshot(self, root: Path) -> Path:
        database = root / "frozen.db"
        schema = (
            Path(__file__).resolve().parents[1] / "src" / "memory_demo" / "schema.sql"
        ).read_text(encoding="utf-8")
        connection = sqlite3.connect(database)
        connection.create_function(
            "memory_bigram_tokens", 1, lambda value: "", deterministic=True
        )
        try:
            connection.executescript(schema)
            connection.execute(
                "INSERT INTO schema_meta(schema_version, created_at) VALUES(13, '2026-09-06T00:00:00+00:00')"
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
        scope_manifest.write_text(
            '{"source_keys":["main/example.json"]}\n', encoding="utf-8"
        )
        return source_manifest, scope_manifest

    def _spec(
        self, database: Path, source_manifest: Path, scope_manifest: Path
    ) -> dict[str, object]:
        database_digest = _digest_file(database)
        source_digest = _digest_file(source_manifest)
        scope_digest = _digest_file(scope_manifest)
        embedding_space = {
            "id": "bge-test/normalized-v1",
            "model_id": "bge-test",
            "preprocess_version": "normalized-v1",
            "dimension": _DIMENSION,
            "normalized": True,
            "dtype": "float32",
        }
        request = {
            "schema": REQUEST_MANIFEST_SCHEMA,
            "question": "synthetic question",
            "intent_override": {"mode": "fact", "search_queries": ["synthetic"]},
            "followup_queries_override": [],
            "contextual_domain": "story",
            "contextual_evaluation_as_of": "2026-09-06T00:00:00+00:00",
        }
        record = {
            "schema": REQUEST_RECORD_SCHEMA,
            "record_id": "request-1",
            "request": request,
            "request_sha256": _digest_document(request),
            "vector_manifest": {
                "schema": VECTOR_MANIFEST_SCHEMA,
                "artifact_id": "vectors-1",
                "artifact_sha256": "sha256:" + "b" * 64,
                "embedding_space_id": embedding_space["id"],
                "dimension": _DIMENSION,
                "normalized": True,
                "binding_ids": ["whole-query", "need-query"],
            },
            "database_manifest": {
                "schema": DATABASE_MANIFEST_SCHEMA,
                "database_sha256": database_digest,
                "source_manifest_sha256": source_digest,
                "scope_sha256": scope_digest,
                "schema_version": 13,
            },
        }
        frozen_input = {
            "schema": FROZEN_INPUT_SCHEMA,
            "artifact_id": "requests-1",
            "records": [record],
        }
        frozen_input["artifact_sha256"] = _digest_document(frozen_input["records"])
        arm = {
            "arm_id": "contextual",
            "primary_score": True,
            "evaluation_role": "treatment",
            "requires_contextual": True,
            "replay_mode": "input_frozen",
            "frozen_input": frozen_input,
            "execution_modules": {"matcher": True, "selector": True},
            "snapshot": {
                "database_sha256": database_digest,
                "source_manifest_sha256": source_digest,
                "scope_sha256": scope_digest,
                "embedding_space": embedding_space,
            },
            "contextual": {
                "matcher": {"enabled": True, "implementation": "v3-matcher"},
                "cues": [
                    {
                        "cue_id": "context-cue",
                        "database_cue_id": 1,
                        "cue_kind": "context",
                        "vector_binding_id": "whole-query",
                    },
                    {
                        "cue_id": "need-cue",
                        "database_cue_id": 2,
                        "cue_kind": "need",
                        "vector_binding_id": "need-query",
                    },
                ],
                "edges": [
                    {
                        "edge_id": "edge-1",
                        "database_association_id": 1,
                        "cue_id": "context-cue",
                        "cue_ids": ["context-cue", "need-cue"],
                        "target_ref": "episode:2",
                    }
                ],
                "query_refs": [
                    {
                        "query_ref_id": "whole-query",
                        "role": "whole",
                        "vector_binding_id": "whole-query",
                    }
                ],
                "vector_bindings": [
                    {
                        "binding_id": "whole-query",
                        "embedding_space_id": embedding_space["id"],
                        "dimension": _DIMENSION,
                        "normalized": True,
                    },
                    {
                        "binding_id": "need-query",
                        "embedding_space_id": embedding_space["id"],
                        "dimension": _DIMENSION,
                        "normalized": True,
                    },
                ],
            },
            "candidate_inventory": {"state": "nonempty", "count": 1},
        }
        return {
            "schema": RUN_SPEC_SCHEMA,
            "safety": {
                "network": "forbidden",
                "model_calls": "forbidden",
                "database_writes": "forbidden",
            },
            "sqlite_preflight": {
                "database_sha256": database_digest,
                "source_manifest_sha256": source_digest,
                "scope_sha256": scope_digest,
                "schema_version": 13,
                "evaluation_as_of": "2026-09-06T00:00:00+00:00",
                "source_keys": [
                    {"source_key": "main/example.json", "require_episode": True}
                ],
                "embedding_space": embedding_space,
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
            "arms": [arm],
        }

    def _execute(
        self,
        engine: object,
        database: Path,
        source_manifest: Path,
        scope_manifest: Path,
        *,
        gold: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return execute_replay(
            self._spec(database, source_manifest, scope_manifest),
            gold or _gold(),
            _split(),
            executor=engine,
            database=database,
            source_manifest=source_manifest,
            scope_manifest=scope_manifest,
            vector_bundle_resolver=lambda record: object(),
        )

    def test_input_frozen_replay_calls_injected_query_engine_and_keeps_database_identical(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            before = database.read_bytes()
            engine = _FakeV3Engine()

            report = self._execute(engine, database, source_manifest, scope_manifest)

            self.assertEqual(before, database.read_bytes())
        self.assertEqual(REPLAY_RESULT_SCHEMA, report["schema"])
        self.assertEqual("executed", report["status"])
        self.assertEqual(1, len(engine.calls))
        question, kwargs = engine.calls[0]
        self.assertEqual("synthetic question", question)
        self.assertFalse(kwargs["generate_answer"])
        self.assertTrue(kwargs["strict_vector_bundle"])
        self.assertNotIn("frozen_plan", kwargs)
        record = report["arms"][0]["records"][0]
        self.assertTrue(record["executed_modules"]["v3_contribution_mechanism"])
        self.assertTrue(record["executed_modules"]["target_gate"])
        self.assertTrue(record["executed_modules"]["contribution_selector"])
        self.assertTrue(report["database_identity"]["immutable"])

    def test_missing_actual_v3_mechanism_rejects_after_executor_call(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            engine = _MissingMechanismEngine()

            report = self._execute(engine, database, source_manifest, scope_manifest)

        self.assertEqual("invalid_setup", report["status"])
        self.assertEqual(1, len(engine.calls))
        self.assertIn("v3_contribution_mechanism_absent", _codes(report))
        self.assertIn("v3_target_gate_absent", _codes(report))
        self.assertIn("v3_selector_absent", _codes(report))

    def test_database_mutation_by_executor_is_detected_as_invalid_setup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            engine = _WritingEngine(database)

            report = self._execute(engine, database, source_manifest, scope_manifest)

        self.assertEqual("invalid_setup", report["status"])
        self.assertIn("database_changed_during_replay", _codes(report))
        self.assertFalse(report["database_identity"]["immutable"])

    def test_unsafe_database_preflight_rejects_before_executor_invocation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            spec = self._spec(database, source_manifest, scope_manifest)
            connection = sqlite3.connect(database)
            connection.create_function(
                "memory_bigram_tokens", 1, lambda value: "", deterministic=True
            )
            try:
                connection.execute("UPDATE source SET raw_text = 'changed before replay' WHERE id = 1")
                connection.commit()
            finally:
                connection.close()
            engine = _FakeV3Engine()

            report = execute_replay(
                spec,
                _gold(),
                _split(),
                executor=engine,
                database=database,
                source_manifest=source_manifest,
                scope_manifest=scope_manifest,
                vector_bundle_resolver=lambda record: object(),
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertEqual([], engine.calls)
        self.assertIn("sqlite_preflight_not_ready", _codes(report))

    def test_draft_gold_skips_executor_and_never_scores(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = self._seed_snapshot(root)
            source_manifest, scope_manifest = self._manifests(root)
            engine = _FakeV3Engine()

            report = self._execute(
                engine,
                database,
                source_manifest,
                scope_manifest,
                gold=_gold(status="draft_pending_source_span_review"),
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertEqual([], engine.calls)
        self.assertFalse(report["scoring"]["performed"])
        self.assertEqual("source_gold_or_split_not_usable", report["scoring"]["reason"])


if __name__ == "__main__":
    unittest.main()
