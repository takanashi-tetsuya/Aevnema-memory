from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database, SCHEMA_VERSION, utc_now
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.types import (
    LearningCandidate,
    LearningCandidatePlan,
    RecallLearningEvent,
    SourceFactRef,
)


class ContextualCreationReceiptV14Tests(unittest.TestCase):
    """Focused durable-creation contract tests; no model or network calls."""

    def _application(self, directory: str) -> MemoryApplication:
        return MemoryApplication(
            AppConfig(
                database_path=Path(directory) / "memory.db",
                log_dir=Path(directory) / "logs",
                model=ModelConfig(
                    embedding_model="receipt-test-space",
                    embedding_dimension=3,
                ),
            )
        )

    @staticmethod
    def _fact(
        *,
        source_id: int,
        source_key: str,
        raw_text: str,
        updated_at: str,
        spans: tuple[tuple[int, int], ...] = ((1, 1),),
    ) -> SourceFactRef:
        canonical_spans = tuple(sorted(set(spans)))
        lines = raw_text.splitlines()
        source_revision_material = json.dumps(
            {
                "source_id": source_id,
                "source_key_sha256": hashlib.sha256(
                    source_key.encode("utf-8")
                ).hexdigest(),
                "raw_source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                "revision_marker": datetime.fromisoformat(updated_at).astimezone(
                    timezone.utc
                ).isoformat(),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return SourceFactRef(
            source_revision_id="source-revision:sha256:"
            + hashlib.sha256(source_revision_material).hexdigest(),
            record_span=(
                f"source-row:{source_id}",
                *(f"lines:{start}-{end}" for start, end in canonical_spans),
            ),
            span_hash="sha256:"
            + hashlib.sha256(
                json.dumps(canonical_spans, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            raw_span_hash="sha256:"
            + hashlib.sha256(
                "\x1e".join(
                    "\n".join(lines[start - 1 : end])
                    for start, end in canonical_spans
                ).encode("utf-8")
            ).hexdigest(),
            source_key=source_key,
        )

    def _seed_endpoints(
        self, app: MemoryApplication, *, target_count: int = 1
    ) -> tuple[SourceFactRef, tuple[int, ...]]:
        raw_text = "anchor evidence\ntarget evidence\nthird target evidence"
        source_key = "receipt-source"
        updated_at = "2026-09-06T00:00:00+00:00"
        with app.db.transaction() as connection:
            source_id = int(
                connection.execute("INSERT INTO source(raw_text) VALUES(?)", (raw_text,)).lastrowid
            )
            episode_ids: list[int] = []
            for index in range(target_count + 1):
                cursor = connection.execute(
                    """
                    INSERT INTO episode(
                        source_id, source_key, segment_index, text,
                        evidence_origin, epistemic_status, generation,
                        evidence_quotes_json, evidence_spans_json, evidence_basis,
                        embedding, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, 'source', 'observed', 0, ?, ?,
                             'literal_source_span', ?, ?, ?)
                    """,
                    (
                        source_id,
                        source_key,
                        index,
                        f"episode-{index + 1}",
                        json.dumps(["evidence"]),
                        json.dumps([[1, 1]]),
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                        updated_at,
                        updated_at,
                    ),
                )
                episode_ids.append(int(cursor.lastrowid))
        return (
            self._fact(
                source_id=source_id,
                source_key=source_key,
                raw_text=raw_text,
                updated_at=updated_at,
            ),
            tuple(episode_ids),
        )

    @staticmethod
    def _candidate(
        fact: SourceFactRef,
        *,
        candidate_id: str = "candidate-1",
        request_id: str = "request-1",
        request_hash: str = "request-hash-1",
        anchor_id: int = 1,
        target_id: int = 2,
    ) -> LearningCandidate:
        return LearningCandidate(
            candidate_id=candidate_id,
            anchor_type="episode",
            anchor_id=anchor_id,
            target_episode_id=target_id,
            reason="candidate_missing",
            request_id=request_id,
            request_hash=request_hash,
            source_request_hash="source-request-hash-1",
            context_query_id="context-query-1",
            need_query_id="need-query-1",
            slot_id="slot-1",
            context_vector_ref="context-vector-ref-1",
            need_vector_ref="need-vector-ref-1",
            anchor_vector_ref="anchor-vector-ref-1",
            source_facts=(fact,),
            verification_refs=("verification-ref-1",),
            verification_status="source_bound",
            anchor_contribution_id="contribution-1",
            anchor_provenance_refs=("anchor-provenance-1",),
            target_provenance_refs=("target-provenance-1",),
        )

    @staticmethod
    def _materialization(*, context_hash: str = "context-hash", need_hash: str = "need-hash") -> dict[str, object]:
        return {
            "domain": "knowledge",
            "model_id": "receipt-test-space",
            "dimension": 3,
            "context_vector": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "need_vector": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "context_text_hash": context_hash,
            "need_text_hash": need_hash,
        }

    @staticmethod
    def _counts(app: MemoryApplication) -> tuple[int, int, int]:
        with app.db.connection() as connection:
            return tuple(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "association_cue_prototype",
                    "association",
                    "contextual_creation_receipt",
                )
            )

    def test_app_commits_then_publishes_exact_receipt_and_legacy_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app, target_count=2)
            candidate = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])

            receipt = app.finalize_contextual_association(
                candidate, **self._materialization()
            )

            self.assertEqual("ready", receipt["status"])
            self.assertTrue(receipt["ready_for_revisit"])
            self.assertEqual(1, app.context_cue_index.count)
            self.assertEqual(1, app.need_cue_index.count)
            self.assertGreater(int(receipt["index_epoch"]), 0)
            self.assertEqual(0, app.associations.get(int(receipt["association_id"]))["utility_successes"])
            with app.db.connection() as connection:
                durable_facts = str(
                    connection.execute(
                        "SELECT source_facts_json FROM contextual_creation_receipt WHERE id = ?",
                        (receipt["receipt_id"],),
                    ).fetchone()["source_facts_json"]
                )
            self.assertNotIn("target evidence", durable_facts)
            self.assertNotIn("receipt-source", durable_facts)

            legacy_candidate = self._candidate(
                fact,
                candidate_id="legacy-candidate",
                request_id="legacy-request",
                request_hash="legacy-request-hash",
                anchor_id=ids[0],
                target_id=ids[2],
            ).as_contextual_candidate()
            legacy_edge_id = app.create_contextual_association(
                legacy_candidate,
                **self._materialization(
                    context_hash="legacy-context-hash", need_hash="legacy-need-hash"
                ),
            )
            with app.db.connection() as connection:
                legacy = connection.execute(
                    """
                    SELECT status FROM contextual_creation_receipt
                    WHERE association_id = ?
                    """,
                    (legacy_edge_id,),
                ).fetchone()
            self.assertEqual("legacy_pending_verification", legacy["status"])

    def test_public_finalizer_accepts_reasoning_view_source_fact_locator(self):
        """The finalizer rebuilds the importer's declared coordinate view."""

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            raw_text = "\n".join(
                (
                    "[source_key: receipt-structured]",
                    "",
                    "[record: 1]",
                    "[speaker_raw: A]",
                    "[script_raw: #na;A;anchor proof]",
                    "zh-CN: 锚点证据。",
                    "ja: 根拠です。",
                    "en: Anchor proof.",
                )
            )
            reasoning_lines = MemoryExtractor._single_pass_source_lines(
                MemoryExtractor.compact_source_for_reasoning(raw_text)
            )[0]
            span = (2, len(reasoning_lines))
            span_text = "\n".join(reasoning_lines[span[0] - 1 : span[1]])
            source_key = "receipt-structured"
            updated_at = "2026-09-06T00:00:00+00:00"
            with app.db.transaction() as connection:
                source_id = int(
                    connection.execute(
                        "INSERT INTO source(raw_text) VALUES(?)", (raw_text,)
                    ).lastrowid
                )
                episode_ids = []
                for index in range(2):
                    episode_ids.append(
                        int(
                            connection.execute(
                                """
                                INSERT INTO episode(
                                    source_id, source_key, segment_index, text,
                                    evidence_origin, epistemic_status, generation,
                                    evidence_quotes_json, evidence_spans_json, evidence_basis,
                                    embedding, created_at, updated_at
                                ) VALUES(?, ?, ?, ?, 'source', 'observed', 0, ?, ?,
                                         'reasoning_view_nonempty_lines_v1', ?, ?, ?)
                                """,
                                (
                                    source_id,
                                    source_key,
                                    index,
                                    f"episode-{index + 1}",
                                    json.dumps([span_text]),
                                    json.dumps([list(span)]),
                                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                                    updated_at,
                                    updated_at,
                                ),
                            ).lastrowid
                        )
                    )
            revision_material = json.dumps(
                {
                    "source_id": source_id,
                    "source_key_sha256": hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
                    "raw_source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "revision_marker": updated_at,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fact = SourceFactRef(
                source_revision_id="source-revision:sha256:"
                + hashlib.sha256(revision_material).hexdigest(),
                record_span=(
                    f"source-row:{source_id}",
                    f"reasoning-view-nonempty-lines:{span[0]}-{span[1]}",
                ),
                span_hash="sha256:"
                + hashlib.sha256(
                    json.dumps((span,), separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                raw_span_hash="sha256:"
                + hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                source_key=source_key,
            )

            receipt = app.finalize_contextual_association(
                self._candidate(fact, anchor_id=episode_ids[0], target_id=episode_ids[1]),
                **self._materialization(),
            )

            self.assertEqual("ready", receipt["status"])

    def test_v13_cue_schema_migrates_additively_and_reinitializes_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "v13.db"
            schema_path = Path(__file__).resolve().parents[1] / "src" / "memory_demo" / "schema.sql"
            # Simulate the v13 cue table specifically. initialize() executes
            # the current schema before its version loop, so this also guards
            # that ordering against a missing-column assumption.
            schema = schema_path.read_text(encoding="utf-8").replace(
                "    embedding_space_id TEXT NOT NULL DEFAULT '',\n", "", 1
            )
            connection = sqlite3.connect(database_path)
            connection.create_function(
                "memory_bigram_tokens", 1, lambda value: "", deterministic=True
            )
            try:
                connection.executescript(schema)
                connection.execute(
                    "INSERT INTO schema_meta(schema_version, created_at) VALUES(13, 'legacy')"
                )
                connection.commit()
            finally:
                connection.close()

            migrated = Database(database_path)
            migrated.initialize()
            migrated.initialize()
            with migrated.connection() as connection:
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
                cue_columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(association_cue_prototype)"
                    )
                }
                publication_columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(contextual_index_publication)"
                    )
                }
            self.assertEqual(SCHEMA_VERSION, version)
            self.assertIn("embedding_space_id", cue_columns)
            self.assertIn("embedding_space_id", publication_columns)

    def test_pending_receipt_is_not_query_visible_until_exact_ready_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app)
            candidate = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])
            receipt = app.associations.finalize_contextual_creation(
                candidate, **self._materialization()
            )

            self.assertEqual("committed_pending_index", receipt["status"])
            self.assertFalse(receipt["ready_for_revisit"])
            with self.assertRaises(sqlite3.IntegrityError):
                with app.db.transaction() as connection:
                    connection.execute(
                        """
                        UPDATE contextual_creation_receipt
                        SET source_request_hash = 'mutated'
                        WHERE id = ?
                        """,
                        (receipt["receipt_id"],),
                    )
            hidden = app.associations.get_contextual_for_prototypes(
                [receipt["context_cue_id"]],
                [receipt["need_cue_id"]],
                domain="knowledge",
                anchor_episode_ids=[ids[0]],
                evaluation_as_of=utc_now(),
            )
            self.assertEqual([], hidden)

            app.context_cue_index.upsert(
                int(receipt["context_cue_id"]), [1.0, 0.0, 0.0]
            )
            app.need_cue_index.upsert(int(receipt["need_cue_id"]), [0.0, 1.0, 0.0])
            ready = app.associations.mark_contextual_receipt_ready(
                int(receipt["receipt_id"]),
                context_cue_count=app.context_cue_index.count,
                need_cue_count=app.need_cue_index.count,
                expected_context_cue_id=int(receipt["context_cue_id"]),
                expected_need_cue_id=int(receipt["need_cue_id"]),
            )
            visible = app.associations.get_contextual_for_prototypes(
                [receipt["context_cue_id"]],
                [receipt["need_cue_id"]],
                domain="knowledge",
                anchor_episode_ids=[ids[0]],
                evaluation_as_of=utc_now(),
            )
            self.assertEqual("ready", ready["status"])
            self.assertEqual([receipt["association_id"]], [int(row["id"]) for row in visible])

    def test_transaction_failure_leaves_no_partial_cue_edge_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app)
            candidate = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])

            with patch.object(
                app.associations,
                "_create_contextual_in_transaction",
                side_effect=RuntimeError("injected edge failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected edge failure"):
                    app.associations.finalize_contextual_creation(
                        candidate, **self._materialization()
                    )
            self.assertEqual((0, 0, 0), self._counts(app))

    def test_retry_is_idempotent_and_embedding_space_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app)
            candidate = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])
            first = app.associations.finalize_contextual_creation(
                candidate, **self._materialization()
            )
            retry = app.associations.finalize_contextual_creation(
                candidate, **self._materialization()
            )
            self.assertEqual(first["receipt_id"], retry["receipt_id"])
            self.assertTrue(retry["idempotent"])
            self.assertEqual((2, 1, 1), self._counts(app))

            app.associations.get_or_create_cue_prototype(
                domain="knowledge",
                cue_kind="context",
                model_id="receipt-test-space",
                dimension=3,
                vector=[0.0, 0.0, 1.0],
                text_hash="space-conflict-context",
                embedding_space_id="embedding-space-a",
            )
            mismatch = replace(
                candidate,
                candidate_id="candidate-space-mismatch",
                request_id="request-space-mismatch",
                request_hash="request-hash-space-mismatch",
            )
            with self.assertRaisesRegex(ValueError, "different embedding space"):
                app.associations.finalize_contextual_creation(
                    mismatch,
                    **{
                        **self._materialization(
                            context_hash="space-conflict-context",
                            need_hash="space-conflict-need",
                        ),
                        "embedding_space_id": "embedding-space-b",
                    },
                )
            self.assertEqual((3, 1, 1), self._counts(app))

    def test_altered_source_and_invalid_second_event_candidate_roll_back_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app, target_count=2)
            one = self._candidate(
                fact,
                candidate_id="candidate-event-one",
                request_id="request-event",
                request_hash="request-hash-event",
                anchor_id=ids[0],
                target_id=ids[1],
            )
            stale_fact = replace(fact, raw_span_hash="sha256:stale")
            two = replace(
                self._candidate(
                    fact,
                    candidate_id="candidate-event-two",
                    request_id="request-event",
                    request_hash="request-hash-event",
                    anchor_id=ids[0],
                    target_id=ids[2],
                ),
                source_facts=(stale_fact,),
            )
            event = RecallLearningEvent(
                request_id="request-event",
                request_hash="request-hash-event",
                domain="knowledge",
                target_episode_ids=(ids[1], ids[2]),
                final_selected_episode_ids=(ids[1], ids[2]),
                final_delivered_episode_ids=(ids[1], ids[2]),
                source_request_hash="source-request-hash-1",
                slot_id="slot-1",
                context_query_id="context-query-1",
                need_query_id="need-query-1",
                context_vector_ref="context-vector-ref-1",
                need_vector_ref="need-vector-ref-1",
                source_facts=(fact,),
                verification_refs=("verification-ref-1",),
                verification_status="source_bound",
                target_provenance_refs=("target-provenance-1",),
            )
            plan = LearningCandidatePlan(
                request_id="request-event", candidates=(one, two)
            )
            with self.assertRaisesRegex(ValueError, "span no longer matches"):
                app.associations.finalize_recall_event(
                    event,
                    plan,
                    cue_materializations={
                        one.candidate_id: self._materialization(
                            context_hash="event-context-one", need_hash="event-need-one"
                        ),
                        two.candidate_id: self._materialization(
                            context_hash="event-context-two", need_hash="event-need-two"
                        ),
                    },
                )
            self.assertEqual((0, 0, 0), self._counts(app))

            altered = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])
            with app.db.transaction() as connection:
                connection.execute("UPDATE source SET raw_text = ?", ("changed source",))
            with self.assertRaisesRegex(ValueError, "current source"):
                app.associations.finalize_contextual_creation(
                    altered, **self._materialization()
                )
            self.assertEqual((0, 0, 0), self._counts(app))

    def test_multi_candidate_event_is_atomic_and_idempotent_per_candidate_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            fact, ids = self._seed_endpoints(app, target_count=2)
            candidates = tuple(
                self._candidate(
                    fact,
                    candidate_id=f"candidate-event-{index}",
                    request_id="request-event",
                    request_hash="request-hash-event",
                    anchor_id=ids[0],
                    target_id=target_id,
                )
                for index, target_id in enumerate(ids[1:], start=1)
            )
            event = RecallLearningEvent(
                request_id="request-event",
                request_hash="request-hash-event",
                domain="knowledge",
                target_episode_ids=ids[1:],
                final_selected_episode_ids=ids[1:],
                final_delivered_episode_ids=ids[1:],
                source_request_hash="source-request-hash-1",
                slot_id="slot-1",
                context_query_id="context-query-1",
                need_query_id="need-query-1",
                context_vector_ref="context-vector-ref-1",
                need_vector_ref="need-vector-ref-1",
                source_facts=(fact,),
                verification_refs=("verification-ref-1",),
                verification_status="source_bound",
                target_provenance_refs=("target-provenance-1",),
            )
            plan = LearningCandidatePlan(request_id="request-event", candidates=candidates)
            materializations = {
                candidate.candidate_id: self._materialization(
                    context_hash=f"event-context-{index}",
                    need_hash=f"event-need-{index}",
                )
                for index, candidate in enumerate(candidates, start=1)
            }
            first = app.associations.finalize_recall_event(
                event, plan, cue_materializations=materializations
            )
            retry = app.associations.finalize_recall_event(
                event, plan, cue_materializations=materializations
            )
            self.assertEqual(2, len(first))
            self.assertEqual(
                [item["receipt_id"] for item in first],
                [item["receipt_id"] for item in retry],
            )
            self.assertTrue(all(bool(item["idempotent"]) for item in retry))
            self.assertEqual((4, 2, 2), self._counts(app))

    def test_rebuild_reconciles_pending_receipt_without_model_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            first_app = self._application(directory)
            fact, ids = self._seed_endpoints(first_app)
            candidate = self._candidate(fact, anchor_id=ids[0], target_id=ids[1])
            pending = first_app.associations.finalize_contextual_creation(
                candidate, **self._materialization()
            )
            self.assertEqual("committed_pending_index", pending["status"])

            restarted = self._application(directory)
            with patch("memory_demo.app.ModelClient", side_effect=AssertionError("no model")):
                rebuilt = restarted.rebuild_contextual_indexes()
            restored = restarted.associations.get_contextual_creation_receipt(
                int(pending["receipt_id"])
            )
            self.assertEqual(0, rebuilt["external_calls"])
            self.assertEqual(1, rebuilt["reconciled_receipts"])
            self.assertEqual("ready", restored["status"])
            self.assertTrue(restored["ready_for_revisit"])
            self.assertEqual(1, restarted.context_cue_index.count)
            self.assertEqual(1, restarted.need_cue_index.count)


if __name__ == "__main__":
    unittest.main()
