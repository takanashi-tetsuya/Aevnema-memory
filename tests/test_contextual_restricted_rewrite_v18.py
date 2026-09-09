from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from memory_demo.database import Database, SCHEMA_VERSION, utc_now
from memory_demo.repositories.association import (
    AssociationRepository,
    RestrictedRewriteReadyBindingError,
)
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.retrieval.revisit import (
    RestrictedRewriteCommitmentKey,
    exact_revisit_context_hash,
    exact_revisit_request_hash,
    parse_restricted_rewrite_question,
    restricted_rewrite_guard_draft,
    restricted_rewrite_manifest_binding_signer,
    restricted_rewrite_ready_manifest_signer,
)
from memory_demo.types import (
    ContextualRestrictedRewriteGuardLookup,
    ContextualRevisitRuntimeManifest,
)
from memory_demo.types import EvidenceSlot
from tests import test_contextual_revisit_runtime_manifest_v17 as _v17


class ContextualRestrictedRewriteV18Tests(unittest.TestCase):
    """T16 storage/grammar tests use only local SQLite and fixed vectors."""

    question = "What is the title of Alice?"
    rewrite = "For Alice, what is the title?"

    @staticmethod
    def _cue_fingerprint(vector: np.ndarray) -> str:
        return "cue-vector:sha256:" + hashlib.sha256(vector.tobytes()).hexdigest()

    @staticmethod
    def _key() -> RestrictedRewriteCommitmentKey:
        return RestrictedRewriteCommitmentKey(
            key_id="test-key-v1",
            secret=b"local-test-restricted-rewrite-key",
        )

    @classmethod
    def _manifest_binding_signer(cls):
        return restricted_rewrite_manifest_binding_signer(cls._key())

    @classmethod
    def _ready_manifest_signer(cls):
        return restricted_rewrite_ready_manifest_signer(cls._key())

    @classmethod
    def _draft(cls, seed):
        key = cls._key()
        ir = parse_restricted_rewrite_question(cls.question)
        assert ir is not None
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(
                EvidenceSlot(
                    slot_id="restricted-rewrite-slot",
                    question=cls.question,
                    required=True,
                    query_id="restricted-rewrite-query",
                    support_mode="alternative",
                    clause_ids=("restricted-rewrite-clause",),
                ),
            ),
            planner_origin="test",
        )
        return restricted_rewrite_guard_draft(
            question=cls.question,
            requirements=requirements,
            creation_request_id=seed.creation_request_id,
            domain=seed.domain,
            context_scope_hash=seed.context_scope_hash,
            runtime_seed=seed,
            context_cue_vector_fingerprint=cls._cue_fingerprint(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            ),
            need_cue_vector_fingerprint=cls._cue_fingerprint(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            ),
            model_id="runtime-manifest-test",
            embedding_space_id="runtime-manifest-space",
            dimension=3,
            dtype="float32",
            commitment_key=key,
        )

    @staticmethod
    def _database(directory: str) -> Database:
        database = Database(Path(directory) / "restricted-rewrite.db")
        database.initialize()
        return database

    def _pending_with_guard(self, directory: str, *, manifest_binding_signer=None):
        database = self._database(directory)
        fixture = _v17.ContextualRevisitRuntimeManifestV17Tests()
        fact, anchor_id, target_id = fixture._seed_endpoint_rows(database)
        candidate = fixture._candidate(fact, anchor_id, target_id)
        seed = fixture._runtime_seed(fact, anchor_id=anchor_id, target_id=target_id)
        # This storage fixture predates the T16 construction boundary and
        # intentionally uses opaque placeholder hashes.  A real restricted
        # rewrite must bind its seed to the exact originating question.
        seed = replace(
            seed,
            context_hash=exact_revisit_context_hash(
                self.question,
                context_scope_hash=seed.context_scope_hash,
            ),
            source_request_hash=exact_revisit_request_hash(self.question),
        )
        candidate = replace(candidate, source_request_hash=seed.source_request_hash)
        guard = self._draft(seed)
        assert guard is not None
        repository = AssociationRepository(database)
        receipt = repository.finalize_contextual_creation(
            candidate,
            domain=seed.domain,
            model_id="runtime-manifest-test",
            dimension=3,
            context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            context_text_hash=seed.context_cue_text_hash,
            need_text_hash=seed.need_cue_text_hash,
            embedding_space_id="runtime-manifest-space",
            revisit_runtime_seed=seed,
            restricted_rewrite_guard=guard,
            restricted_rewrite_manifest_binding_signer=(
                self._manifest_binding_signer()
                if manifest_binding_signer is None
                else manifest_binding_signer
            ),
        )
        return database, repository, receipt, seed, candidate, guard

    def _mark_ready(
        self,
        repository: AssociationRepository,
        receipt: dict[str, object],
        *,
        ready_manifest_signer=None,
    ) -> dict[str, object]:
        return repository.mark_contextual_receipt_ready(
            int(receipt["receipt_id"]),
            context_cue_count=1,
            need_cue_count=1,
            restricted_rewrite_ready_manifest_signer=(
                self._ready_manifest_signer()
                if ready_manifest_signer is None
                else ready_manifest_signer
            ),
        )

    def test_controlled_forms_share_an_hmac_but_semantic_extensions_miss(self) -> None:
        key = self._key()
        origin = parse_restricted_rewrite_question(self.question)
        rewrite = parse_restricted_rewrite_question(self.rewrite)
        self.assertIsNotNone(origin)
        self.assertEqual(origin, rewrite)
        assert origin is not None
        self.assertEqual(
            key.commitment(
                context_scope_hash="scope:sha256:" + "0" * 64,
                ir=origin,
            ),
            key.commitment(
                context_scope_hash="scope:sha256:" + "0" * 64,
                ir=rewrite,
            ),
        )
        for unsafe in (
            "What is the title of Alice in 2024?",
            "What is not the title of Alice?",
            "What is the title of Alice and Bob?",
            "What is the capital of Republic of Congo?",
            "For Congo, what is the capital of Republic?",
            "What is the title of Bob?",
            "Why is the title of Alice?",
            "What is the title of Alice versus Bob?",
            "What is the title of this person?",
            "What is the only title of Alice?",
            "What is the second title of Alice?",
            "What is the title of Alice unless Bob agrees?",
        ):
            parsed = parse_restricted_rewrite_question(unsafe)
            if unsafe == "What is the title of Bob?":
                self.assertIsNotNone(parsed)
                assert parsed is not None
                self.assertNotEqual(
                    key.commitment(
                        context_scope_hash="scope:sha256:" + "0" * 64,
                        ir=origin,
                    ),
                    key.commitment(
                        context_scope_hash="scope:sha256:" + "0" * 64,
                        ir=parsed,
                    ),
                )
            else:
                self.assertIsNone(parsed, unsafe)

    def test_q1_guard_is_atomic_redacted_and_ready_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, seed, candidate, guard = (
                self._pending_with_guard(directory)
            )
            with database.connection() as connection:
                row = connection.execute(
                    "SELECT * FROM contextual_restricted_rewrite_guard"
                ).fetchone()
                assert row is not None
                columns = {
                    item["name"]
                    for item in connection.execute(
                        "PRAGMA table_info(contextual_restricted_rewrite_guard)"
                    )
                }
            self.assertEqual(22, SCHEMA_VERSION)
            self.assertEqual(guard.rewrite_commitment, row["rewrite_commitment"])
            self.assertEqual(guard.binding_commitment, row["binding_commitment"])
            self.assertEqual(guard.commitment_key_id, row["commitment_key_id"])
            self.assertIn("manifest_binding_commitment", columns)
            self.assertIn("ready_manifest_commitment", columns)
            self.assertTrue(
                str(row["manifest_binding_commitment"]).startswith(
                    "restricted-rewrite-manifest-binding:sha256:"
                )
            )
            self.assertEqual("", str(row["ready_manifest_commitment"]))
            self.assertNotIn("question", " ".join(columns).casefold())
            self.assertNotIn("answer", " ".join(columns).casefold())
            self.assertNotIn("vector_blob", " ".join(columns).casefold())
            self.assertIsNone(
                repository.find_contextual_restricted_rewrite_guard(
                    ContextualRestrictedRewriteGuardLookup(
                        association_id=None,
                        domain=seed.domain,
                        context_scope_hash=seed.context_scope_hash,
                        rewrite_commitment=guard.rewrite_commitment,
                        commitment_key_id=guard.commitment_key_id,
                        grammar_version=guard.grammar_version,
                    )
                )
            )
            # Q1 is idempotent while the guard is still in its immutable
            # pending form.  Once publication fills the third HMAC, the old
            # draft must not overwrite or re-authorize that completed guard.
            retry = repository.finalize_contextual_creation(
                candidate,
                domain=seed.domain,
                model_id="runtime-manifest-test",
                dimension=3,
                context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                context_text_hash=seed.context_cue_text_hash,
                need_text_hash=seed.need_cue_text_hash,
                embedding_space_id="runtime-manifest-space",
                revisit_runtime_seed=seed,
                restricted_rewrite_guard=guard,
                restricted_rewrite_manifest_binding_signer=(
                    self._manifest_binding_signer()
                ),
            )
            self.assertTrue(retry["idempotent"])
            ready = self._mark_ready(repository, receipt)
            restored = repository.find_contextual_restricted_rewrite_guard(
                ContextualRestrictedRewriteGuardLookup(
                    association_id=None,
                    domain=seed.domain,
                    context_scope_hash=seed.context_scope_hash,
                    rewrite_commitment=guard.rewrite_commitment,
                    commitment_key_id=guard.commitment_key_id,
                    grammar_version=guard.grammar_version,
                ),
                evaluation_as_of=str(ready["ready_at"]),
            )
            self.assertIsNotNone(restored)
            assert restored is not None
            restored_guard, manifest = restored
            self.assertEqual(guard.rewrite_commitment, restored_guard.rewrite_commitment)
            self.assertTrue(
                restored_guard.ready_manifest_commitment.startswith(
                    "restricted-rewrite-ready-manifest:sha256:"
                )
            )
            self.assertEqual(seed.seed_fingerprint, manifest.seed.seed_fingerprint)
            with database.connection() as connection:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_restricted_rewrite_guard"
                    ).fetchone()[0]
                )
            self.assertEqual(1, count)

    def test_malformed_v5_guard_cannot_create_or_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, seed, _candidate, guard = (
                self._pending_with_guard(directory)
            )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        "DELETE FROM contextual_restricted_rewrite_guard WHERE creation_receipt_id = ?",
                        (int(receipt["receipt_id"]),),
                    )
            with database.connection() as connection:
                # Simulate an offline database attacker that has already
                # bypassed SQLite's normal immutable trigger.  The typed
                # guard fingerprint must still make the row unreadable.
                connection.execute(
                    "DROP TRIGGER contextual_restricted_rewrite_guard_no_update"
                )
                connection.execute(
                    "UPDATE contextual_restricted_rewrite_guard SET rewrite_commitment = ? WHERE creation_receipt_id = ?",
                    ("tampered:sha256:" + "0" * 64, int(receipt["receipt_id"])),
                )
                connection.commit()
            with self.assertRaisesRegex(
                RestrictedRewriteReadyBindingError,
                "invalid V22/future signature",
            ):
                self._mark_ready(repository, receipt)
            with database.connection() as connection:
                receipt_row = connection.execute(
                    "SELECT status, ready_index_epoch, ready_at "
                    "FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                publication = connection.execute(
                    "SELECT index_epoch FROM contextual_index_publication "
                    "WHERE singleton = 1"
                ).fetchone()
                manifest = connection.execute(
                    "SELECT state FROM contextual_revisit_runtime_manifest "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
            assert receipt_row is not None
            assert publication is not None
            assert manifest is not None
            self.assertEqual("committed_pending_index", receipt_row["status"])
            self.assertIsNone(receipt_row["ready_index_epoch"])
            self.assertIsNone(receipt_row["ready_at"])
            self.assertEqual(0, int(publication["index_epoch"]))
            self.assertEqual("pending_index", manifest["state"])
            self.assertIsNone(
                repository.find_contextual_restricted_rewrite_guard(
                    ContextualRestrictedRewriteGuardLookup(
                        association_id=None,
                        domain=seed.domain,
                        context_scope_hash=seed.context_scope_hash,
                        rewrite_commitment=guard.rewrite_commitment,
                        commitment_key_id=guard.commitment_key_id,
                        grammar_version=guard.grammar_version,
                    ),
                )
            )

    def test_raw_deleted_pending_manifest_cannot_publish_a_v5_guard(self) -> None:
        """A supplied ready signer cannot bless a DB-deleted V22 seed."""

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, _seed, _candidate, _guard = (
                self._pending_with_guard(directory)
            )
            with database.connection() as connection:
                # Simulate a raw local database writer bypassing the normal
                # immutable trigger before the RAM publication step.
                connection.execute(
                    "DROP TRIGGER contextual_revisit_runtime_manifest_no_delete"
                )
                connection.execute(
                    "DELETE FROM contextual_revisit_runtime_manifest "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                )
                connection.commit()

            with self.assertRaises(RestrictedRewriteReadyBindingError):
                self._mark_ready(repository, receipt)

            with database.connection() as connection:
                receipt_row = connection.execute(
                    "SELECT status, ready_index_epoch, ready_at "
                    "FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                publication = connection.execute(
                    "SELECT index_epoch FROM contextual_index_publication "
                    "WHERE singleton = 1"
                ).fetchone()
                manifest_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_runtime_manifest "
                        "WHERE creation_receipt_id = ?",
                        (int(receipt["receipt_id"]),),
                    ).fetchone()[0]
                )
                guard = connection.execute(
                    "SELECT ready_manifest_commitment "
                    "FROM contextual_restricted_rewrite_guard "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
            assert receipt_row is not None
            assert publication is not None
            assert guard is not None
            self.assertEqual("committed_pending_index", receipt_row["status"])
            self.assertIsNone(receipt_row["ready_index_epoch"])
            self.assertIsNone(receipt_row["ready_at"])
            self.assertEqual(0, int(publication["index_epoch"]))
            self.assertEqual(0, manifest_count)
            self.assertEqual("", str(guard["ready_manifest_commitment"]))

    def test_tampered_pending_v5_cue_cannot_publish_or_leave_blank_guard(self) -> None:
        """A normal promotion mismatch is fatal when its V22 guard is present."""

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, _seed, _candidate, _guard = (
                self._pending_with_guard(directory)
            )
            replacement_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            with database.transaction() as connection:
                # The raw replacement remains a valid float32 unit vector, so
                # the failure is the V17 pending binding mismatch rather than
                # a malformed SQLite payload.
                connection.execute(
                    "UPDATE association_cue_prototype SET vector_blob = ? WHERE id = ?",
                    (
                        replacement_vector.tobytes(),
                        int(receipt["context_cue_id"]),
                    ),
                )

            with self.assertRaisesRegex(
                RestrictedRewriteReadyBindingError,
                "could not atomically bind its ready manifest",
            ):
                self._mark_ready(repository, receipt)

            with database.connection() as connection:
                receipt_row = connection.execute(
                    "SELECT status, ready_index_epoch, ready_at "
                    "FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                publication = connection.execute(
                    "SELECT index_epoch FROM contextual_index_publication "
                    "WHERE singleton = 1"
                ).fetchone()
                manifest = connection.execute(
                    "SELECT state, contract_id, manifest_fingerprint "
                    "FROM contextual_revisit_runtime_manifest "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                guard = connection.execute(
                    "SELECT ready_manifest_commitment "
                    "FROM contextual_restricted_rewrite_guard "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                contract_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_contract"
                    ).fetchone()[0]
                )
            assert receipt_row is not None
            assert publication is not None
            assert manifest is not None
            assert guard is not None
            self.assertEqual("committed_pending_index", receipt_row["status"])
            self.assertIsNone(receipt_row["ready_index_epoch"])
            self.assertIsNone(receipt_row["ready_at"])
            self.assertEqual(0, int(publication["index_epoch"]))
            self.assertEqual("pending_index", manifest["state"])
            self.assertIsNone(manifest["contract_id"])
            self.assertIsNone(manifest["manifest_fingerprint"])
            self.assertEqual("", str(guard["ready_manifest_commitment"]))
            self.assertEqual(0, contract_count)

    def test_raw_ready_v5_guard_with_blank_third_mac_is_not_idempotent(self) -> None:
        """A forged ready state cannot turn the one-shot V22 signer into a retry."""

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, _seed, _candidate, _guard = (
                self._pending_with_guard(directory)
            )
            with database.transaction() as connection:
                pending_manifest = repository._load_runtime_manifest_in_transaction(
                    connection, int(receipt["receipt_id"])
                )
                assert pending_manifest is not None
                ready_at = utc_now()
                connection.execute(
                    "DROP TRIGGER contextual_creation_receipt_immutable"
                )
                connection.execute(
                    "DROP TRIGGER contextual_revisit_runtime_manifest_ready_transition"
                )
                connection.execute(
                    """
                    UPDATE contextual_index_publication
                    SET index_epoch = 1, embedding_space_id = ?,
                        context_cue_count = 1, need_cue_count = 1, published_at = ?
                    WHERE singleton = 1
                    """,
                    (pending_manifest.embedding_space_id, ready_at),
                )
                connection.execute(
                    """
                    UPDATE contextual_creation_receipt
                    SET status = 'ready', ready_index_epoch = 1, ready_at = ?
                    WHERE id = ?
                    """,
                    (ready_at, int(receipt["receipt_id"])),
                )
                ready_receipt = connection.execute(
                    "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert ready_receipt is not None
                contract = repository._runtime_manifest_contract_for_ready(
                    pending_manifest, receipt=ready_receipt
                )
                contract_receipt = (
                    repository._create_contextual_revisit_contract_in_transaction(
                        connection, contract
                    )
                )
                ready_publication = (
                    repository.contextual_revisit_ready_publication_fingerprint(
                        dict(ready_receipt)
                    )
                )
                ready_manifest_fingerprint = (
                    ContextualRevisitRuntimeManifest.expected_manifest_fingerprint(
                        binding_fingerprint=pending_manifest.binding_fingerprint,
                        contract_id=contract_receipt.contract_id,
                        contract_fingerprint=contract_receipt.contract_fingerprint,
                        ready_at=ready_at,
                        ready_index_epoch=1,
                        ready_publication_fingerprint=ready_publication,
                    )
                )
                raw_ready_manifest = (
                    repository._bound_runtime_manifest_from_seed_in_transaction(
                        connection,
                        receipt=ready_receipt,
                        seed=pending_manifest.seed,
                        manifest_id=pending_manifest.manifest_id,
                        state="ready",
                        ready_index_epoch=1,
                        ready_at=ready_at,
                        ready_publication_fingerprint=ready_publication,
                        contract_id=contract_receipt.contract_id,
                        contract_fingerprint=contract_receipt.contract_fingerprint,
                        manifest_fingerprint=ready_manifest_fingerprint,
                    )
                )
                connection.execute(
                    """
                    UPDATE contextual_revisit_runtime_manifest
                    SET state = 'ready', ready_index_epoch = ?, ready_at = ?,
                        ready_publication_fingerprint = ?, contract_id = ?,
                        contract_fingerprint = ?, manifest_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        raw_ready_manifest.ready_index_epoch,
                        raw_ready_manifest.ready_at,
                        raw_ready_manifest.ready_publication_fingerprint,
                        raw_ready_manifest.contract_id,
                        raw_ready_manifest.contract_fingerprint,
                        raw_ready_manifest.manifest_fingerprint,
                        int(receipt["receipt_id"]),
                    ),
                )

            signer_calls: list[object] = []
            base_signer = self._ready_manifest_signer()

            def supplied_signer(*args):
                signer_calls.append(args)
                assert base_signer is not None
                return base_signer(*args)

            with self.assertRaisesRegex(
                RestrictedRewriteReadyBindingError,
                "missing its ready HMAC",
            ):
                self._mark_ready(
                    repository,
                    receipt,
                    ready_manifest_signer=supplied_signer,
                )
            self.assertEqual([], signer_calls)

            with database.connection() as connection:
                raw_receipt = connection.execute(
                    "SELECT status FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                raw_manifest = connection.execute(
                    "SELECT state FROM contextual_revisit_runtime_manifest "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                raw_guard = connection.execute(
                    "SELECT ready_manifest_commitment "
                    "FROM contextual_restricted_rewrite_guard "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
            assert raw_receipt is not None
            assert raw_manifest is not None
            assert raw_guard is not None
            self.assertEqual("ready", raw_receipt["status"])
            self.assertEqual("ready", raw_manifest["state"])
            self.assertEqual("", str(raw_guard["ready_manifest_commitment"]))

    def test_q1_signer_failure_rolls_back_every_contextual_artifact(self) -> None:
        """The second-MAC signer runs inside Q1's all-or-nothing transaction."""

        def rejecting_signer(*_args):
            raise ValueError("test signer rejection")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "manifest signer rejected the canonical pending binding"
            ):
                self._pending_with_guard(
                    directory,
                    manifest_binding_signer=rejecting_signer,
                )
            database = self._database(directory)
            with database.connection() as connection:
                counts = {
                    table: int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                    for table in (
                        "association_cue_prototype",
                        "association",
                        "contextual_creation_receipt",
                        "contextual_revisit_runtime_manifest",
                        "contextual_restricted_rewrite_guard",
                    )
                }
            self.assertEqual(
                {
                    "association_cue_prototype": 0,
                    "association": 0,
                    "contextual_creation_receipt": 0,
                    "contextual_revisit_runtime_manifest": 0,
                    "contextual_restricted_rewrite_guard": 0,
                },
                counts,
            )

    def test_ready_signer_failure_rolls_back_the_publication_transaction(self) -> None:
        """The V22 signer cannot leave a ready receipt or contract behind."""

        def rejecting_signer(*_args):
            raise ValueError("test ready signer rejection")

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, _seed, _candidate, _guard = (
                self._pending_with_guard(directory)
            )
            with self.assertRaisesRegex(
                RestrictedRewriteReadyBindingError,
                "ready signer rejected the canonical ready manifest",
            ):
                self._mark_ready(
                    repository,
                    receipt,
                    ready_manifest_signer=rejecting_signer,
                )

            with database.connection() as connection:
                receipt_row = connection.execute(
                    "SELECT status, ready_index_epoch, ready_at "
                    "FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                publication = connection.execute(
                    "SELECT index_epoch FROM contextual_index_publication "
                    "WHERE singleton = 1"
                ).fetchone()
                manifest = connection.execute(
                    "SELECT state, contract_id, manifest_fingerprint "
                    "FROM contextual_revisit_runtime_manifest "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                guard = connection.execute(
                    "SELECT ready_manifest_commitment "
                    "FROM contextual_restricted_rewrite_guard "
                    "WHERE creation_receipt_id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                contract_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_contract"
                    ).fetchone()[0]
                )
            assert receipt_row is not None
            assert publication is not None
            assert manifest is not None
            assert guard is not None
            self.assertEqual("committed_pending_index", receipt_row["status"])
            self.assertIsNone(receipt_row["ready_index_epoch"])
            self.assertIsNone(receipt_row["ready_at"])
            self.assertEqual(0, int(publication["index_epoch"]))
            self.assertEqual("pending_index", manifest["state"])
            self.assertIsNone(manifest["contract_id"])
            self.assertIsNone(manifest["manifest_fingerprint"])
            self.assertEqual("", str(guard["ready_manifest_commitment"]))
            self.assertEqual(0, contract_count)

            retried = self._mark_ready(repository, receipt)
            self.assertEqual("ready", retried["status"])

    def test_q1_guard_builder_refuses_a_seed_for_another_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _database, _repository, _receipt, seed, _candidate, _guard = (
                self._pending_with_guard(directory)
            )
            mismatched_seed = replace(
                seed,
                context_hash=(
                    "revisit-context:sha256:" + "f" * 64
                ),
            )
            self.assertIsNone(self._draft(mismatched_seed))

    def test_v17_to_v22_migration_never_backfills_a_rewrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            fixture = _v17.ContextualRevisitRuntimeManifestV17Tests()
            fact, anchor_id, target_id = fixture._seed_endpoint_rows(database)
            candidate = fixture._candidate(fact, anchor_id, target_id)
            seed = fixture._runtime_seed(fact, anchor_id=anchor_id, target_id=target_id)
            repository = AssociationRepository(database)
            repository.finalize_contextual_creation(
                candidate,
                domain=seed.domain,
                model_id="runtime-manifest-test",
                dimension=3,
                context_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                need_vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                context_text_hash=seed.context_cue_text_hash,
                need_text_hash=seed.need_cue_text_hash,
                embedding_space_id="runtime-manifest-space",
                revisit_runtime_seed=seed,
            )
            with database.connection() as connection:
                for trigger_name in (
                    "contextual_restricted_rewrite_guard_no_replace",
                    "contextual_restricted_rewrite_guard_redaction_guard",
                    "contextual_restricted_rewrite_guard_source_guard",
                    "contextual_restricted_rewrite_guard_no_update",
                    "contextual_restricted_rewrite_guard_no_delete",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx"
                )
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx"
                )
                connection.execute("DROP TABLE contextual_restricted_rewrite_guard")
                connection.execute("UPDATE schema_meta SET schema_version = 17")
                connection.commit()
            database.initialize()
            with database.connection() as connection:
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
                guard_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_restricted_rewrite_guard"
                    ).fetchone()[0]
                )
                manifest_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_runtime_manifest"
                    ).fetchone()[0]
                )
            self.assertEqual(22, version)
            self.assertEqual(0, guard_count)
            self.assertEqual(1, manifest_count)

    def test_v20_guard_migration_keeps_empty_second_and_third_macs_fail_closed(self) -> None:
        """A V20 sidecar is retained for audit, never silently re-signed."""

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, seed, _candidate, guard = (
                self._pending_with_guard(directory)
            )
            ready = self._mark_ready(repository, receipt)
            lookup = ContextualRestrictedRewriteGuardLookup(
                association_id=None,
                domain=seed.domain,
                context_scope_hash=seed.context_scope_hash,
                rewrite_commitment=guard.rewrite_commitment,
                commitment_key_id=guard.commitment_key_id,
                grammar_version=guard.grammar_version,
            )
            self.assertIsNotNone(
                repository.find_contextual_restricted_rewrite_guard(
                    lookup, evaluation_as_of=str(ready["ready_at"])
                )
            )
            with database.connection() as connection:
                for trigger_name in (
                    "contextual_restricted_rewrite_guard_no_replace",
                    "contextual_restricted_rewrite_guard_redaction_guard",
                    "contextual_restricted_rewrite_guard_source_guard",
                    "contextual_restricted_rewrite_guard_no_update",
                    "contextual_restricted_rewrite_guard_no_delete",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx"
                )
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx"
                )
                connection.executescript(
                    """
                    ALTER TABLE contextual_restricted_rewrite_guard
                        RENAME TO contextual_restricted_rewrite_guard_v21_source;
                    CREATE TABLE contextual_restricted_rewrite_guard(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        creation_receipt_id INTEGER NOT NULL UNIQUE,
                        association_id INTEGER NOT NULL,
                        domain TEXT NOT NULL,
                        context_scope_hash TEXT NOT NULL,
                        rewrite_commitment TEXT NOT NULL,
                        binding_commitment TEXT NOT NULL,
                        context_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                        need_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                        model_id TEXT NOT NULL DEFAULT '',
                        embedding_space_id TEXT NOT NULL DEFAULT '',
                        dimension INTEGER NOT NULL DEFAULT 0,
                        dtype TEXT NOT NULL DEFAULT '',
                        commitment_key_id TEXT NOT NULL,
                        grammar_version TEXT NOT NULL,
                        seed_fingerprint TEXT NOT NULL,
                        signature_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        guard_fingerprint TEXT NOT NULL UNIQUE,
                        FOREIGN KEY(creation_receipt_id)
                            REFERENCES contextual_creation_receipt(id),
                        FOREIGN KEY(association_id) REFERENCES association(id)
                    );
                    INSERT INTO contextual_restricted_rewrite_guard(
                        id, creation_receipt_id, association_id, domain,
                        context_scope_hash, rewrite_commitment, binding_commitment,
                        context_cue_vector_fingerprint,
                        need_cue_vector_fingerprint, model_id,
                        embedding_space_id, dimension, dtype,
                        commitment_key_id, grammar_version, seed_fingerprint,
                        signature_version, created_at, guard_fingerprint
                    )
                    SELECT id, creation_receipt_id, association_id, domain,
                           context_scope_hash, rewrite_commitment, binding_commitment,
                           context_cue_vector_fingerprint,
                           need_cue_vector_fingerprint, model_id,
                           embedding_space_id, dimension, dtype,
                           commitment_key_id, grammar_version, seed_fingerprint,
                           signature_version, created_at, guard_fingerprint
                    FROM contextual_restricted_rewrite_guard_v21_source;
                    DROP TABLE contextual_restricted_rewrite_guard_v21_source;
                    """
                )
                connection.execute("UPDATE schema_meta SET schema_version = 20")
                connection.commit()

            database.initialize()
            with database.connection() as connection:
                row = connection.execute(
                    """
                    SELECT manifest_binding_commitment,
                           ready_manifest_commitment,
                           context_cue_vector_fingerprint,
                           need_cue_vector_fingerprint, model_id,
                           embedding_space_id, dimension, dtype
                    FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert row is not None
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
            self.assertEqual(22, version)
            self.assertEqual(
                (
                    "",
                    "",
                    guard.context_cue_vector_fingerprint,
                    guard.need_cue_vector_fingerprint,
                    guard.model_id,
                    guard.embedding_space_id,
                    guard.dimension,
                    guard.dtype,
                ),
                tuple(row),
            )
            self.assertIsNone(
                repository.find_contextual_restricted_rewrite_guard(
                    lookup, evaluation_as_of=str(ready["ready_at"])
                )
            )

    def test_v21_guard_migration_keeps_empty_third_mac_fail_closed(self) -> None:
        """A ready V21 record cannot be retroactively authorized for V22."""

        with tempfile.TemporaryDirectory() as directory:
            database, repository, receipt, seed, _candidate, guard = (
                self._pending_with_guard(directory)
            )
            ready = self._mark_ready(repository, receipt)
            lookup = ContextualRestrictedRewriteGuardLookup(
                association_id=None,
                domain=seed.domain,
                context_scope_hash=seed.context_scope_hash,
                rewrite_commitment=guard.rewrite_commitment,
                commitment_key_id=guard.commitment_key_id,
                grammar_version=guard.grammar_version,
            )
            fresh = repository.find_contextual_restricted_rewrite_guard(
                lookup, evaluation_as_of=str(ready["ready_at"])
            )
            self.assertIsNotNone(fresh)
            assert fresh is not None
            fresh_guard, _manifest = fresh
            self.assertTrue(fresh_guard.manifest_binding_commitment)
            self.assertTrue(fresh_guard.ready_manifest_commitment)

            with database.connection() as connection:
                for trigger_name in (
                    "contextual_restricted_rewrite_guard_no_replace",
                    "contextual_restricted_rewrite_guard_redaction_guard",
                    "contextual_restricted_rewrite_guard_source_guard",
                    "contextual_restricted_rewrite_guard_no_update",
                    "contextual_restricted_rewrite_guard_no_delete",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx"
                )
                connection.execute(
                    "DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx"
                )
                connection.executescript(
                    """
                    ALTER TABLE contextual_restricted_rewrite_guard
                        RENAME TO contextual_restricted_rewrite_guard_v22_source;
                    CREATE TABLE contextual_restricted_rewrite_guard(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        creation_receipt_id INTEGER NOT NULL UNIQUE,
                        association_id INTEGER NOT NULL,
                        domain TEXT NOT NULL,
                        context_scope_hash TEXT NOT NULL,
                        rewrite_commitment TEXT NOT NULL,
                        binding_commitment TEXT NOT NULL,
                        manifest_binding_commitment TEXT NOT NULL DEFAULT '',
                        context_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                        need_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                        model_id TEXT NOT NULL DEFAULT '',
                        embedding_space_id TEXT NOT NULL DEFAULT '',
                        dimension INTEGER NOT NULL DEFAULT 0,
                        dtype TEXT NOT NULL DEFAULT '',
                        commitment_key_id TEXT NOT NULL,
                        grammar_version TEXT NOT NULL,
                        seed_fingerprint TEXT NOT NULL,
                        signature_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        guard_fingerprint TEXT NOT NULL UNIQUE,
                        FOREIGN KEY(creation_receipt_id)
                            REFERENCES contextual_creation_receipt(id),
                        FOREIGN KEY(association_id) REFERENCES association(id)
                    );
                    INSERT INTO contextual_restricted_rewrite_guard(
                        id, creation_receipt_id, association_id, domain,
                        context_scope_hash, rewrite_commitment, binding_commitment,
                        manifest_binding_commitment,
                        context_cue_vector_fingerprint,
                        need_cue_vector_fingerprint, model_id,
                        embedding_space_id, dimension, dtype,
                        commitment_key_id, grammar_version, seed_fingerprint,
                        signature_version, created_at, guard_fingerprint
                    )
                    SELECT id, creation_receipt_id, association_id, domain,
                           context_scope_hash, rewrite_commitment, binding_commitment,
                           manifest_binding_commitment,
                           context_cue_vector_fingerprint,
                           need_cue_vector_fingerprint, model_id,
                           embedding_space_id, dimension, dtype,
                           commitment_key_id, grammar_version, seed_fingerprint,
                           signature_version, created_at, guard_fingerprint
                    FROM contextual_restricted_rewrite_guard_v22_source;
                    DROP TABLE contextual_restricted_rewrite_guard_v22_source;
                    """
                )
                connection.execute("UPDATE schema_meta SET schema_version = 21")
                connection.commit()

            database.initialize()
            with database.connection() as connection:
                row = connection.execute(
                    """
                    SELECT manifest_binding_commitment, ready_manifest_commitment
                    FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert row is not None
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
            self.assertEqual(22, version)
            self.assertEqual(
                fresh_guard.manifest_binding_commitment,
                str(row["manifest_binding_commitment"]),
            )
            self.assertEqual("", str(row["ready_manifest_commitment"]))
            self.assertIsNone(
                repository.find_contextual_restricted_rewrite_guard(
                    lookup, evaluation_as_of=str(ready["ready_at"])
                )
            )
