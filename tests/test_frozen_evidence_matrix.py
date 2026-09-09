from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.run_frozen_evidence_matrix import (
    _prepare_shared_plan,
    _snapshot_edge_receipt,
)
from benchmarks.run_q1_q2_diagnostic_pilot import _call_query


class FrozenEvidenceMatrixTests(unittest.TestCase):
    def _snapshot(self, directory: str, *, ready_manifest: bool) -> Path:
        path = Path(directory) / "snapshot.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE association(
                    id INTEGER PRIMARY KEY, from_type TEXT, from_id INTEGER,
                    to_type TEXT, to_id INTEGER, relation_key TEXT, lifecycle_state TEXT
                );
                CREATE TABLE contextual_creation_receipt(
                    id INTEGER PRIMARY KEY, status TEXT, association_id INTEGER,
                    verification_status TEXT, durable_artifact_hash TEXT
                );
                CREATE TABLE contextual_revisit_runtime_manifest(
                    id INTEGER PRIMARY KEY, state TEXT, association_id INTEGER,
                    creation_receipt_id INTEGER, manifest_fingerprint TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO association VALUES(121, 'episode', 1, 'episode', 2, 'recall', 'probation')"
            )
            connection.execute(
                "INSERT INTO contextual_creation_receipt VALUES(1, 'ready', 121, 'source_bound', 'facts')"
            )
            connection.execute(
                "INSERT INTO contextual_revisit_runtime_manifest VALUES(1, ?, 121, 1, 'manifest')",
                ("ready" if ready_manifest else "pending",),
            )
            connection.commit()
        finally:
            connection.close()
        return path

    def test_requires_one_bound_ready_edge_receipt_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory, ready_manifest=True)
            binding = _snapshot_edge_receipt(snapshot, 121)
        self.assertEqual(121, binding["association"]["id"])
        self.assertEqual("ready", binding["receipt"]["status"])
        self.assertEqual("ready", binding["runtime_manifest"]["state"])

    def test_rejects_snapshot_without_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory, ready_manifest=False)
            with self.assertRaisesRegex(ValueError, "ready runtime manifest"):
                _snapshot_edge_receipt(snapshot, 121)

    def test_public_call_forwards_a_frozen_plan_without_vector_override(self) -> None:
        class Engine:
            model = object()

            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def query(self, question: str, **kwargs: object) -> dict[str, object]:
                self.kwargs = dict(kwargs)
                return {"question": question, "query_plan_frozen": True}

        engine = Engine()
        plan = {
            "plan_schema": "frozen_query_plan_v1",
            "plan_id": "test-plan",
        }
        record = _call_query(
            engine,
            "question",
            domain="knowledge",
            scope_hash="scope",
            deadline_seconds=12.0,
            frozen_plan=plan,
            generate_answer=False,
            stop_after="evidence",
        )
        self.assertEqual("completed", record["status"])
        self.assertIs(plan, engine.kwargs["frozen_plan"])
        self.assertIsNone(engine.kwargs["query_embeddings_override"])
        self.assertEqual("evidence", engine.kwargs["stop_after"])

    def test_preparation_persists_one_edge_independent_plan(self) -> None:
        class Engine:
            model = object()

            def build_frozen_query_input(self, question: str) -> tuple[dict[str, object], object]:
                plan = {
                    "plan_schema": "frozen_query_plan_v1",
                    "plan_id": "shared-plan",
                    "initial_queries": [question],
                    "initial_query_embeddings_float32": [[0.1, 0.2]],
                    "followup_queries": [],
                    "followup_query_embeddings_float32": [],
                    "association_cue_association_ids": [],
                }
                class Bundle:
                    def metadata(self) -> dict[str, object]:
                        return {"physical_vector_count": 1}
                return plan, Bundle()

        class App:
            def query_engine(self, *, config: object) -> Engine:
                return Engine()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.sqlite"
            sqlite3.connect(snapshot).close()
            output = root / "output"
            with patch(
                "benchmarks.run_frozen_evidence_matrix._open_app",
                return_value=(App(), object()),
            ), patch("benchmarks.run_frozen_evidence_matrix._clone_sqlite"):
                result = _prepare_shared_plan(
                    variant_id="same_text",
                    question="question",
                    source_snapshot=snapshot,
                    output_dir=output,
                    env_file=root / ".env",
                    treatment_edge_id=121,
                )
            self.assertEqual("completed", result["status"])
            self.assertEqual("shared-plan", result["plan_id"])
            self.assertEqual(1, result["vector_material"]["initial_embedding_rows"])
            self.assertTrue(Path(str(result["plan_path"])).is_file())


if __name__ == "__main__":
    unittest.main()
