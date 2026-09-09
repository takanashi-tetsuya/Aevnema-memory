from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from memory_demo.database import Database, SCHEMA_VERSION, utc_now
from memory_demo.repositories.association import AssociationRepository
from memory_demo.types import (
    ContextualRevisitRuntimeManifestLookup,
    ContextualRevisitRuntimeSeed,
    ContextualRevisitSlotNeedBinding,
    LearningCandidate,
    SourceFactRef,
)


def _opaque(namespace: str, value: str) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _bare_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(namespace: str, payload: object) -> str:
    material = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(material).hexdigest()}"


class ContextualRevisitRuntimeManifestV17Tests(unittest.TestCase):
    """Storage-only tests: no model, network, question or answer prose."""

    @staticmethod
    def _database(directory: str) -> Database:
        database = Database(Path(directory) / "runtime-manifest.db")
        database.initialize()
        return database

    @staticmethod
    def _fact(*, source_id: int, source_key: str, raw_text: str) -> SourceFactRef:
        updated_at = "2026-09-06T00:00:00+00:00"
        spans = ((1, 1),)
        source_revision_material = json.dumps(
            {
                "source_id": source_id,
                "source_key_sha256": hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
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
            record_span=(f"source-row:{source_id}", "lines:1-1"),
            span_hash="sha256:"
            + hashlib.sha256(json.dumps(spans, separators=(",", ":")).encode("utf-8")).hexdigest(),
            raw_span_hash="sha256:"
            + hashlib.sha256(raw_text.splitlines()[0].encode("utf-8")).hexdigest(),
            source_key=source_key,
        )

    def _seed_endpoint_rows(
        self, database: Database
    ) -> tuple[SourceFactRef, int, int]:
        raw_text = "anchor and target evidence\nother line"
        source_key = "v17-runtime-manifest-source"
        timestamp = "2026-09-06T00:00:00+00:00"
        with database.transaction() as connection:
            source_id = int(
                connection.execute(
                    "INSERT INTO source(raw_text) VALUES(?)", (raw_text,)
                ).lastrowid
            )
            episode_ids: list[int] = []
            for index in range(2):
                episode_ids.append(
                    int(
                        connection.execute(
                            """
                            INSERT INTO episode(
                                source_id, source_key, segment_index, text,
                                evidence_origin, epistemic_status, generation,
                                evidence_quotes_json, evidence_spans_json,
                                evidence_basis, embedding, created_at, updated_at
                            ) VALUES(?, ?, ?, ?, 'source', 'observed', 0, ?, ?,
                                     'literal_source_span', ?, ?, ?)
                            """,
                            (
                                source_id,
                                source_key,
                                index,
                                f"episode-{index}",
                                json.dumps(["evidence"]),
                                json.dumps([[1, 1]]),
                                np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                                timestamp,
                                timestamp,
                            ),
                        ).lastrowid
                    )
                )
        return self._fact(source_id=source_id, source_key=source_key, raw_text=raw_text), *episode_ids

    @staticmethod
    def _candidate(fact: SourceFactRef, anchor_id: int, target_id: int) -> LearningCandidate:
        return LearningCandidate(
            candidate_id="runtime-candidate-1",
            anchor_type="episode",
            anchor_id=anchor_id,
            target_episode_id=target_id,
            reason="candidate_missing",
            request_id="runtime-request-1",
            request_hash=_opaque("request", "runtime-request-1"),
            source_request_hash=_opaque("source-request", "runtime-request-1"),
            context_query_id="context-query-1",
            need_query_id="need-query-1",
            slot_id="slot-1",
            context_vector_ref="context-vector-1",
            need_vector_ref="need-vector-1",
            anchor_vector_ref="anchor-vector-1",
            source_facts=(fact,),
            verification_refs=("verification-ref-1",),
            verification_status="source_bound",
            anchor_contribution_id="anchor-contribution-1",
            anchor_provenance_refs=("anchor-provenance-1",),
            target_provenance_refs=("target-provenance-1",),
        )

    @staticmethod
    def _source_closure_fingerprint(fact: SourceFactRef) -> str:
        return _digest(
            "revisit-source-closure",
            [
                {
                    "source_revision_id": fact.source_revision_id,
                    "record_span": list(fact.record_span),
                    "span_hash": fact.span_hash,
                    "raw_span_hash": fact.raw_span_hash,
                }
            ],
        )

    @staticmethod
    def _source_fact_refs_fingerprint(fact: SourceFactRef) -> str:
        return _digest("revisit-source-fact-refs", [fact.fact_id])

    def _runtime_seed(
        self,
        fact: SourceFactRef,
        *,
        anchor_id: int,
        target_id: int,
    ) -> ContextualRevisitRuntimeSeed:
        context_hash = _bare_hash("runtime-context-cue")
        need_hash = _bare_hash("runtime-need-cue")
        runtime_slot_ref = _opaque("runtime-slot-ref", "one")
        runtime_query_ref = _opaque("runtime-query-ref", "one")
        runtime_clause_ref = _opaque("runtime-clause-ref", "one")
        runtime_slot_id = "runtime-slot:" + runtime_slot_ref.rsplit(":", 1)[-1]
        runtime_query_id = "runtime-query:" + runtime_query_ref.rsplit(":", 1)[-1]
        binding = ContextualRevisitSlotNeedBinding(
            slot_id=_digest("revisit-slot", runtime_slot_id),
            need_query_id=_digest("revisit-need-query", runtime_query_id),
            need_hash=_digest("revisit-need", {"text_hash": need_hash}),
        )
        endpoint_limit = 1
        episode_limit = 2
        source_fact_limit = 1
        delivery_token_limit = 64
        return ContextualRevisitRuntimeSeed(
            creation_request_id="runtime-candidate-1",
            domain="knowledge",
            context_scope_hash=_opaque("context-scope", "private-scope"),
            context_hash=_opaque("revisit-context", "private-scope-and-question-hash"),
            source_request_hash=_opaque("source-request", "runtime-request-1"),
            context_cue_text_hash=context_hash,
            need_cue_text_hash=need_hash,
            slot_need_bindings=(binding,),
            requirements_fingerprint=_opaque("requirements", "one"),
            source_closure_fingerprint=self._source_closure_fingerprint(fact),
            retrieval_policy_fingerprint=_opaque("revisit-policy", "one"),
            budget_fingerprint=_digest(
                "revisit-budget",
                {
                    "episode_limit": episode_limit,
                    "source_fact_limit": source_fact_limit,
                    "delivery_token_limit": delivery_token_limit,
                },
            ),
            anchor_manifest_fingerprint=_digest(
                "revisit-anchor-manifest",
                [{"episode_id": anchor_id, "activation": 1.0}],
            ),
            source_fact_refs_fingerprint=self._source_fact_refs_fingerprint(fact),
            anchor_episode_id=anchor_id,
            anchor_activation=1.0,
            anchor_source_fact_id=fact.fact_id,
            target_episode_id=target_id,
            target_source_fact_id=fact.fact_id,
            target_mapping_ref=_opaque("target-mapping", "one"),
            runtime_slot_ref=runtime_slot_ref,
            runtime_query_ref=runtime_query_ref,
            runtime_clause_ref=runtime_clause_ref,
            endpoint_limit=endpoint_limit,
            episode_limit=episode_limit,
            source_fact_limit=source_fact_limit,
            delivery_token_limit=delivery_token_limit,
        )

    def _create_pending(
        self, database: Database
    ) -> tuple[
        AssociationRepository,
        dict[str, object],
        ContextualRevisitRuntimeSeed,
        LearningCandidate,
    ]:
        repository = AssociationRepository(database)
        fact, anchor_id, target_id = self._seed_endpoint_rows(database)
        candidate = self._candidate(fact, anchor_id, target_id)
        seed = self._runtime_seed(fact, anchor_id=anchor_id, target_id=target_id)
        receipt = repository.finalize_contextual_creation(
            candidate,
            domain="knowledge",
            model_id="runtime-manifest-test",
            dimension=3,
            context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            context_text_hash=seed.context_cue_text_hash,
            need_text_hash=seed.need_cue_text_hash,
            embedding_space_id="runtime-manifest-space",
            revisit_runtime_seed=seed,
        )
        return repository, receipt, seed, candidate

    @staticmethod
    def _lookup(
        seed: ContextualRevisitRuntimeSeed,
    ) -> ContextualRevisitRuntimeManifestLookup:
        return ContextualRevisitRuntimeManifestLookup(
            domain=seed.domain,
            context_scope_hash=seed.context_scope_hash,
            context_hash=seed.context_hash,
            source_request_hash=seed.source_request_hash,
            model_id="runtime-manifest-test",
            embedding_space_id="runtime-manifest-space",
            dimension=3,
        )

    def test_q1_seed_is_atomic_redacted_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, candidate = self._create_pending(database)
            retry = repository.finalize_contextual_creation(
                candidate,
                domain="knowledge",
                model_id="runtime-manifest-test",
                dimension=3,
                context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                context_text_hash=seed.context_cue_text_hash,
                need_text_hash=seed.need_cue_text_hash,
                embedding_space_id="runtime-manifest-space",
                revisit_runtime_seed=seed,
            )
            self.assertTrue(retry["idempotent"])
            self.assertEqual("committed_pending_index", retry["status"])
            with database.connection() as connection:
                row = connection.execute(
                    "SELECT * FROM contextual_revisit_runtime_manifest"
                ).fetchone()
                assert row is not None
                columns = {
                    item["name"]
                    for item in connection.execute(
                        "PRAGMA table_info(contextual_revisit_runtime_manifest)"
                    )
                }
            self.assertEqual(22, SCHEMA_VERSION)
            self.assertEqual(seed.seed_fingerprint, row["seed_fingerprint"])
            self.assertIn("context_cue_vector_fingerprint", columns)
            self.assertIn("need_cue_vector_fingerprint", columns)
            self.assertNotIn("vector_blob", columns)
            self.assertNotIn("question", " ".join(columns).casefold())
            self.assertNotIn("answer", " ".join(columns).casefold())
            self.assertNotIn("source_text", columns)

    def test_ready_promotion_is_contract_bound_and_exact_lookup_is_live(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, _candidate = self._create_pending(database)
            ready = repository.mark_contextual_receipt_ready(
                int(receipt["receipt_id"]), context_cue_count=1, need_cue_count=1
            )
            promoted = repository.promote_contextual_revisit_runtime_manifest(
                int(receipt["receipt_id"])
            )
            restored = repository.find_contextual_revisit_runtime_manifest(
                self._lookup(seed), evaluation_as_of=str(ready["ready_at"])
            )
            self.assertEqual("ready", ready["status"])
            assert promoted is not None
            self.assertEqual("ready", promoted.state)
            self.assertTrue(promoted.idempotent)
            assert restored is not None
            self.assertEqual(seed.runtime_slot_id, restored.seed.runtime_slot_id)
            self.assertEqual(seed.runtime_query_id, restored.seed.runtime_query_id)
            self.assertEqual(
                restored.to_revisit_contract().contract_fingerprint,
                promoted.contract_fingerprint,
            )
            self.assertIsNone(
                repository.find_contextual_revisit_runtime_manifest(
                    self._lookup(seed), evaluation_as_of="2099-01-01T00:00:00+00:00"
                )
            )

    def test_vector_mutation_or_old_ready_pending_state_fails_closed_or_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, _candidate = self._create_pending(database)
            # Simulate an older publisher dying after it made the canonical
            # receipt ready but before V17 promotion.  Reconciliation creates
            # the contract and manifest together from the durable seed.
            ready_at = utc_now()
            with database.transaction() as connection:
                connection.execute(
                    """
                    UPDATE contextual_index_publication
                    SET index_epoch = 1, embedding_space_id = ?,
                        context_cue_count = 1, need_cue_count = 1, published_at = ?
                    WHERE singleton = 1
                    """,
                    ("runtime-manifest-space", ready_at),
                )
                connection.execute(
                    """
                    UPDATE contextual_creation_receipt
                    SET status = 'ready', ready_index_epoch = 1, ready_at = ?
                    WHERE id = ?
                    """,
                    (ready_at, int(receipt["receipt_id"])),
                )
            recovery = repository.reconcile_contextual_revisit_runtime_manifests()
            self.assertEqual(1, recovery["promoted"], recovery)
            self.assertIsNotNone(
                repository.find_contextual_revisit_runtime_manifest(
                    self._lookup(seed), evaluation_as_of=ready_at
                )
            )
            with database.transaction() as connection:
                connection.execute(
                    "UPDATE association_cue_prototype SET vector_blob = ? WHERE id = ?",
                    (b"\x00" * 12, int(receipt["context_cue_id"])),
                )
            self.assertIsNone(
                repository.load_contextual_revisit_runtime_manifest(
                    int(receipt["receipt_id"]), evaluation_as_of=ready_at
                )
            )

    def test_source_revision_drift_is_a_normal_path_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, _candidate = self._create_pending(database)
            ready = repository.mark_contextual_receipt_ready(
                int(receipt["receipt_id"]), context_cue_count=1, need_cue_count=1
            )
            self.assertIsNotNone(
                repository.find_contextual_revisit_runtime_manifest(
                    self._lookup(seed), evaluation_as_of=str(ready["ready_at"])
                )
            )

    def test_stale_seed_cannot_rollback_receipt_publication(self) -> None:
        """A broken optional V17 seed must not block ordinary cue recovery."""

        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, _candidate = self._create_pending(database)
            with database.transaction() as connection:
                connection.execute("UPDATE source SET raw_text = 'revised before ready'")

            # The one-receipt RAM publication path keeps the authoritative
            # contextual receipt usable, but does not leave a stale V16
            # contract or ready automatic manifest behind.
            ready = repository.mark_contextual_receipt_ready(
                int(receipt["receipt_id"]), context_cue_count=1, need_cue_count=1
            )
            self.assertEqual("ready", ready["status"])
            self.assertIsNone(
                repository.load_contextual_revisit_runtime_manifest(
                    int(receipt["receipt_id"]), evaluation_as_of=str(ready["ready_at"])
                )
            )
            with database.connection() as connection:
                self.assertEqual(
                    0,
                    int(
                        connection.execute(
                            "SELECT COUNT(*) FROM contextual_revisit_contract"
                        ).fetchone()[0]
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository, receipt, seed, _candidate = self._create_pending(database)
            with database.transaction() as connection:
                connection.execute("UPDATE source SET raw_text = 'revised before rebuild'")
            rebuilt = repository.reconcile_contextual_receipts_after_index_rebuild(
                context_cue_ids=(int(receipt["context_cue_id"]),),
                need_cue_ids=(int(receipt["need_cue_id"]),),
                context_cue_count=1,
                need_cue_count=1,
                embedding_space_id="runtime-manifest-space",
            )
            self.assertEqual(1, rebuilt["reconciled"])
            self.assertEqual(1, rebuilt["runtime_manifest_rejected"])
            ready = repository.get_contextual_creation_receipt(
                int(receipt["receipt_id"])
            )
            assert ready is not None
            self.assertEqual("ready", ready["status"])
            self.assertIsNone(
                repository.find_contextual_revisit_runtime_manifest(
                    self._lookup(seed), evaluation_as_of=str(ready["ready_at"])
                )
            )
            with database.transaction() as connection:
                connection.execute("UPDATE source SET raw_text = 'revised source'")
            self.assertIsNone(
                repository.find_contextual_revisit_runtime_manifest(
                    self._lookup(seed), evaluation_as_of=str(ready["ready_at"])
                )
            )

    def test_runtime_source_closure_is_the_seed_bound_anchor_target_subset(self) -> None:
        """Extra ordinary-Q1 facts must not make an optional V17 seed fail."""

        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            fact, anchor_id, target_id = self._seed_endpoint_rows(database)
            source_id = int(fact.record_span[0].split(":", 1)[1])
            extra = SourceFactRef(
                source_revision_id=fact.source_revision_id,
                record_span=(f"source-row:{source_id}", "lines:2-2"),
                span_hash="sha256:"
                + hashlib.sha256(b"[[2,2]]").hexdigest(),
                raw_span_hash="sha256:"
                + hashlib.sha256(b"other line").hexdigest(),
                source_key=fact.source_key,
            )
            candidate = replace(
                self._candidate(fact, anchor_id, target_id),
                source_facts=(fact, extra),
            )
            seed = self._runtime_seed(fact, anchor_id=anchor_id, target_id=target_id)
            receipt = repository.finalize_contextual_creation(
                candidate,
                domain="knowledge",
                model_id="runtime-manifest-test",
                dimension=3,
                context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                context_text_hash=seed.context_cue_text_hash,
                need_text_hash=seed.need_cue_text_hash,
                embedding_space_id="runtime-manifest-space",
                revisit_runtime_seed=seed,
            )
            self.assertEqual("committed_pending_index", receipt["status"])
            with database.connection() as connection:
                source_facts = json.loads(
                    connection.execute(
                        "SELECT source_facts_json FROM contextual_creation_receipt WHERE id = ?",
                        (int(receipt["receipt_id"]),),
                    ).fetchone()["source_facts_json"]
                )
            self.assertEqual(2, len(source_facts))

    def test_v16_migration_creates_no_inferred_runtime_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "v16-runtime-manifest.db"
            schema_path = (
                Path(__file__).resolve().parents[1]
                / "src"
                / "memory_demo"
                / "schema.sql"
            )
            schema = schema_path.read_text(encoding="utf-8")
            v17_start = schema.index("-- V17 adds a recovery seed")
            fts_start = schema.index("-- FTS is a retrieval index", v17_start)
            schema = schema[:v17_start] + schema[fts_start:]
            connection = sqlite3.connect(database_path)
            connection.create_function(
                "memory_bigram_tokens", 1, lambda value: "", deterministic=True
            )
            try:
                connection.executescript(schema)
                connection.execute(
                    "INSERT INTO schema_meta(schema_version, created_at) VALUES(16, 'legacy')"
                )
                connection.commit()
            finally:
                connection.close()

            database = Database(database_path)
            database.initialize()
            database.initialize()
            with database.connection() as connection:
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
                runtime_rows = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_runtime_manifest"
                    ).fetchone()[0]
                )
            self.assertEqual(SCHEMA_VERSION, version)
            self.assertEqual(0, runtime_rows)
