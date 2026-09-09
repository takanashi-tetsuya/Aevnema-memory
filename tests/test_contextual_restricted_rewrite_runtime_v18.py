from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.embeddings import EmbeddingCoordinator
from memory_demo.retrieval.revisit import (
    RestrictedRewriteCommitmentKey,
    restricted_rewrite_guard_draft,
    restricted_rewrite_manifest_binding_signer,
    restricted_rewrite_ready_manifest_signer,
)
from memory_demo.types import (
    ContextualRestrictedRewriteGuard,
    ContextualRevisitRuntimeManifest,
)
from tests import test_contextual_auto_revisit_v17 as _v17


class _DiagnosticsModel(_v17._NoProviderModel):
    """Small local model used only to seed ordinary-request diagnostics."""

    def embed(self, texts):
        self.calls.append("embedding")
        return np.asarray(
            [[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32
        )

    def chat_json(self, system, *_args, **_kwargs):
        self.calls.append("chat_json")
        if "查询解析器" in system:
            return {
                "language": "en",
                "target_entities": [],
                "search_queries": [],
                "requested_relation": "",
                "temporal_constraint": "",
                "causal_constraint": "",
                "answer_shape": "complete_evidence_answer",
                "uncertainty_required": True,
            }
        raise AssertionError(f"unexpected ordinary-query provider purpose: {system}")


class ContextualRestrictedRewriteRuntimeV18Tests(unittest.TestCase):
    """T16 Q2 uses a fake local model and never contacts a provider."""

    origin_question = "What is the title of Alice?"
    rewrite_question = "For Alice, what is the title?"
    hmac_environment = {
        "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY_ID": "test-key-v1",
        "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY": (
            "local-test-restricted-rewrite-key"
        ),
    }

    @staticmethod
    def _key() -> RestrictedRewriteCommitmentKey:
        return RestrictedRewriteCommitmentKey(
            key_id="test-key-v1",
            secret=b"local-test-restricted-rewrite-key",
        )

    @staticmethod
    def _fixture_provenance() -> dict[str, object]:
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        fingerprint = "cue-vector:sha256:" + hashlib.sha256(
            vector.tobytes()
        ).hexdigest()
        model_id = "v17-auto-local-embedding"
        return {
            "context_cue_vector_fingerprint": fingerprint,
            "need_cue_vector_fingerprint": fingerprint,
            "model_id": model_id,
            "embedding_space_id": EmbeddingCoordinator(
                _v17._NoProviderModel(),
                model_id=model_id,
                dimension=3,
            ).embedding_space().canonical_id,
            "dimension": 3,
            "dtype": "float32",
        }

    def _fixture(self, directory: str):
        fixture = _v17.ContextualAutomaticRevisitV17Tests()
        fixture.question = self.origin_question

        def config_for(path: str):
            config = _v17.ContextualAutomaticRevisitV17Tests._config(path)
            config.retrieval.contextual_restricted_rewrite_enabled = True
            return config

        def guard_for_seed(*, question, requirements, seed):
            return restricted_rewrite_guard_draft(
                question=question,
                requirements=requirements,
                creation_request_id=seed.creation_request_id,
                domain=seed.domain,
                context_scope_hash=seed.context_scope_hash,
                runtime_seed=seed,
                **self._fixture_provenance(),
                commitment_key=self._key(),
            )

        fixture._config = config_for
        fixture._restricted_rewrite_guard_for_seed = guard_for_seed
        fixture._restricted_rewrite_manifest_binding_signer = lambda **_kwargs: (
            restricted_rewrite_manifest_binding_signer(self._key())
        )
        fixture._restricted_rewrite_ready_manifest_signer = lambda **_kwargs: (
            restricted_rewrite_ready_manifest_signer(self._key())
        )
        return fixture._fixture(directory)

    @staticmethod
    def _counts(app) -> tuple[int, ...]:
        with app.db.connection() as connection:
            return tuple(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "association",
                    "association_cue_prototype",
                    "contextual_creation_receipt",
                    "contextual_revisit_contract",
                    "contextual_revisit_runtime_manifest",
                    "contextual_restricted_rewrite_guard",
                )
            )

    def test_controlled_rewrite_hits_without_planner_embedding_or_reranker(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, _receipt, _anchor, target_id = self._fixture(
                directory
            )
            before = self._counts(app)
            domain = _v17.ContextualAutomaticRevisitV17Tests.domain

            result = engine.query(
                self.rewrite_question,
                generate_answer=False,
                contextual_domain=domain,
                contextual_revisit_scope_hash=scope_hash,
            )

            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)
            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertEqual("restricted_rewrite", result["exact_revisit"]["mode"])
            self.assertTrue(result["exact_revisit"]["planner_skipped"])
            self.assertTrue(result["exact_revisit"]["embedding_skipped"])
            self.assertTrue(result["exact_revisit"]["reranker_skipped"])
            self.assertIn(target_id, result["episode_ids"])
            self.assertNotIn("guard_fingerprint", result["exact_revisit"])
            self.assertNotIn("rewrite_commitment", result["exact_revisit"])

    def test_t16_automatic_hit_clears_prior_request_diagnostics(self) -> None:
        """A private automatic hit cannot expose diagnostics from a prior request."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, _fixture_model, _fixture_engine, scope_hash, _receipt, _anchor, _target = (
                self._fixture(directory)
            )
            app.config.retrieval.followup_planning_mode = "off"
            app.config.retrieval.rerank_enabled = False
            model = _DiagnosticsModel()
            engine = _v17.ContextualAutomaticRevisitV17Tests._engine(app, model)
            domain = _v17.ContextualAutomaticRevisitV17Tests.domain

            engine.query(
                "Which local record describes the diagnostic?",
                generate_answer=False,
                contextual_domain=domain,
                contextual_revisit_scope_hash=scope_hash,
            )
            self.assertTrue(engine.last_query_embeddings)
            self.assertTrue(
                engine.last_query_embedding_cache_trace["hits"]
                or engine.last_query_embedding_cache_trace["misses"]
            )
            self.assertIn("embedding", model.calls)

            model.calls.clear()
            result = engine.query(
                self.rewrite_question,
                generate_answer=False,
                contextual_domain=domain,
                contextual_revisit_scope_hash=scope_hash,
            )

            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertEqual("restricted_rewrite", result["exact_revisit"]["mode"])
            self.assertEqual({}, engine.last_query_embeddings)
            self.assertEqual(
                {"hits": [], "misses": []},
                engine.last_query_embedding_cache_trace,
            )
            self.assertEqual([], model.calls)

    def test_new_time_number_negation_or_object_falls_through(self) -> None:
        unsafe_questions = (
            "What is the title of Alice in 2024?",
            "What is not the title of Alice?",
            "What is the title of Bob?",
            "What is the title of Alice and Bob?",
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            _app, model, engine, scope_hash, _receipt, _anchor, _target = self._fixture(
                directory
            )
            for question in unsafe_questions:
                with self.subTest(question=question):
                    sentinel = {"fallback": question}
                    with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                        result = engine.query(
                            question,
                            generate_answer=False,
                            contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                            contextual_revisit_scope_hash=scope_hash,
                        )
                    self.assertIs(sentinel, result)
                    fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_missing_key_or_sidecar_change_falls_through_before_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            sentinel = {"fallback": True}
            with patch.dict(
                os.environ,
                {
                    "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY": "",
                },
                clear=False,
            ), patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()

            with app.db.transaction() as connection:
                connection.execute(
                    "DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update"
                )
                connection.execute(
                    "UPDATE contextual_restricted_rewrite_guard SET guard_fingerprint = ? WHERE creation_receipt_id = ?",
                    ("tampered:sha256:" + "0" * 64, int(receipt["receipt_id"])),
                )
            sentinel = {"fallback": "tampered"}
            with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_recomputed_public_guard_fingerprint_cannot_transplant_root(self) -> None:
        """A copied root cannot authorize a different V17 seed without its HMAC."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            with app.db.transaction() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert row is not None
                transplanted_binding = "restricted-rewrite-binding:sha256:" + "a" * 64
                recomputed_public_fingerprint = (
                    ContextualRestrictedRewriteGuard.expected_fingerprint(
                        creation_receipt_id=int(row["creation_receipt_id"]),
                        association_id=int(row["association_id"]),
                        domain=str(row["domain"]),
                        context_scope_hash=str(row["context_scope_hash"]),
                        rewrite_commitment=str(row["rewrite_commitment"]),
                        binding_commitment=transplanted_binding,
                        manifest_binding_commitment=str(
                            row["manifest_binding_commitment"]
                        ),
                        ready_manifest_commitment=str(
                            row["ready_manifest_commitment"]
                        ),
                        context_cue_vector_fingerprint=str(
                            row["context_cue_vector_fingerprint"]
                        ),
                        need_cue_vector_fingerprint=str(
                            row["need_cue_vector_fingerprint"]
                        ),
                        model_id=str(row["model_id"]),
                        embedding_space_id=str(row["embedding_space_id"]),
                        dimension=int(row["dimension"]),
                        dtype=str(row["dtype"]),
                        commitment_key_id=str(row["commitment_key_id"]),
                        grammar_version=str(row["grammar_version"]),
                        seed_fingerprint=str(row["seed_fingerprint"]),
                        signature_version=str(row["signature_version"]),
                        created_at=str(row["created_at"]),
                    )
                )
                # Simulate a database-only attacker that bypasses immutable
                # triggers and can recompute public SHA-256 metadata.
                connection.execute(
                    "DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update"
                )
                connection.execute(
                    """
                    UPDATE contextual_restricted_rewrite_guard
                    SET binding_commitment = ?, guard_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        transplanted_binding,
                        recomputed_public_fingerprint,
                        int(receipt["receipt_id"]),
                    ),
                )
            sentinel = {"fallback": "binding-hmac-rejected"}
            with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_recomputed_public_vector_provenance_cannot_reuse_hmac(self) -> None:
        """Changing blobs and every public manifest hash still loses the v3 HMAC."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            manifest = app.associations.load_contextual_revisit_runtime_manifest(
                int(receipt["receipt_id"])
            )
            assert manifest is not None
            replacement = np.asarray([0.9992, 0.04, 0.0], dtype=np.float32)
            replacement /= np.linalg.norm(replacement)
            replacement_blob = replacement.tobytes()
            replacement_fingerprint = "cue-vector:sha256:" + hashlib.sha256(
                replacement_blob
            ).hexdigest()
            public_binding = (
                ContextualRevisitRuntimeManifest.expected_binding_fingerprint(
                    creation_receipt_id=manifest.creation_receipt_id,
                    association_id=manifest.association_id,
                    context_cue_id=manifest.context_cue_id,
                    need_cue_id=manifest.need_cue_id,
                    model_id=manifest.model_id,
                    embedding_space_id=manifest.embedding_space_id,
                    dimension=manifest.dimension,
                    dtype=manifest.dtype,
                    seed=manifest.seed,
                    context_cue_vector_fingerprint=replacement_fingerprint,
                    need_cue_vector_fingerprint=replacement_fingerprint,
                    source_fact_roles_fingerprint=(
                        manifest.source_fact_roles_fingerprint
                    ),
                    not_before_at=manifest.not_before_at,
                    expires_at=manifest.expires_at,
                )
            )
            public_manifest = (
                ContextualRevisitRuntimeManifest.expected_manifest_fingerprint(
                    binding_fingerprint=public_binding,
                    contract_id=int(manifest.contract_id or 0),
                    contract_fingerprint=manifest.contract_fingerprint,
                    ready_at=manifest.ready_at,
                    ready_index_epoch=int(manifest.ready_index_epoch or 0),
                    ready_publication_fingerprint=(
                        manifest.ready_publication_fingerprint
                    ),
                )
            )
            with app.db.transaction() as connection:
                guard = connection.execute(
                    """
                    SELECT * FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert guard is not None
                public_guard = ContextualRestrictedRewriteGuard.expected_fingerprint(
                    creation_receipt_id=int(guard["creation_receipt_id"]),
                    association_id=int(guard["association_id"]),
                    domain=str(guard["domain"]),
                    context_scope_hash=str(guard["context_scope_hash"]),
                    rewrite_commitment=str(guard["rewrite_commitment"]),
                    binding_commitment=str(guard["binding_commitment"]),
                    manifest_binding_commitment=str(
                        guard["manifest_binding_commitment"]
                    ),
                    ready_manifest_commitment=str(
                        guard["ready_manifest_commitment"]
                    ),
                    context_cue_vector_fingerprint=replacement_fingerprint,
                    need_cue_vector_fingerprint=replacement_fingerprint,
                    model_id=str(guard["model_id"]),
                    embedding_space_id=str(guard["embedding_space_id"]),
                    dimension=int(guard["dimension"]),
                    dtype=str(guard["dtype"]),
                    commitment_key_id=str(guard["commitment_key_id"]),
                    grammar_version=str(guard["grammar_version"]),
                    seed_fingerprint=str(guard["seed_fingerprint"]),
                    signature_version=str(guard["signature_version"]),
                    created_at=str(guard["created_at"]),
                )
                # Simulate a database-only attacker who can replace raw rows
                # and recalculate all public SHA-256 fields, but has no HMAC
                # key.  The immutable trigger is deliberately removed only
                # inside this adversarial local test.
                for trigger_name in (
                    "contextual_restricted_rewrite_guard_no_update",
                    "contextual_revisit_runtime_manifest_ready_transition",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "UPDATE association_cue_prototype SET vector_blob = ? WHERE id IN (?, ?)",
                    (
                        replacement_blob,
                        int(receipt["context_cue_id"]),
                        int(receipt["need_cue_id"]),
                    ),
                )
                connection.execute(
                    """
                    UPDATE contextual_revisit_runtime_manifest
                    SET context_cue_vector_fingerprint = ?,
                        need_cue_vector_fingerprint = ?,
                        binding_fingerprint = ?, manifest_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        replacement_fingerprint,
                        replacement_fingerprint,
                        public_binding,
                        public_manifest,
                        int(receipt["receipt_id"]),
                    ),
                )
                connection.execute(
                    """
                    UPDATE contextual_restricted_rewrite_guard
                    SET context_cue_vector_fingerprint = ?,
                        need_cue_vector_fingerprint = ?, guard_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        replacement_fingerprint,
                        replacement_fingerprint,
                        public_guard,
                        int(receipt["receipt_id"]),
                    ),
                )
            app.context_cue_index.upsert(int(receipt["context_cue_id"]), replacement)
            app.need_cue_index.upsert(int(receipt["need_cue_id"]), replacement)
            sentinel = {"fallback": "vector-provenance-hmac-rejected"}
            with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_recomputed_public_lifecycle_cannot_reuse_second_mac(self) -> None:
        """An offline expiry extension cannot replay the old post-manifest MAC."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            manifest = app.associations.load_contextual_revisit_runtime_manifest(
                int(receipt["receipt_id"])
            )
            assert manifest is not None
            extended_expiry = "2099-01-01T00:00:00+00:00"
            public_binding = (
                ContextualRevisitRuntimeManifest.expected_binding_fingerprint(
                    creation_receipt_id=manifest.creation_receipt_id,
                    association_id=manifest.association_id,
                    context_cue_id=manifest.context_cue_id,
                    need_cue_id=manifest.need_cue_id,
                    model_id=manifest.model_id,
                    embedding_space_id=manifest.embedding_space_id,
                    dimension=manifest.dimension,
                    dtype=manifest.dtype,
                    seed=manifest.seed,
                    context_cue_vector_fingerprint=(
                        manifest.context_cue_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=(
                        manifest.need_cue_vector_fingerprint
                    ),
                    source_fact_roles_fingerprint=(
                        manifest.source_fact_roles_fingerprint
                    ),
                    not_before_at=manifest.not_before_at,
                    expires_at=extended_expiry,
                )
            )
            public_manifest = (
                ContextualRevisitRuntimeManifest.expected_manifest_fingerprint(
                    binding_fingerprint=public_binding,
                    contract_id=int(manifest.contract_id or 0),
                    contract_fingerprint=manifest.contract_fingerprint,
                    ready_at=manifest.ready_at,
                    ready_index_epoch=int(manifest.ready_index_epoch or 0),
                    ready_publication_fingerprint=(
                        manifest.ready_publication_fingerprint
                    ),
                )
            )
            with app.db.transaction() as connection:
                guard = connection.execute(
                    """
                    SELECT * FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert guard is not None
                old_second_mac = str(guard["manifest_binding_commitment"])
                old_third_mac = str(guard["ready_manifest_commitment"])
                public_guard = ContextualRestrictedRewriteGuard.expected_fingerprint(
                    creation_receipt_id=int(guard["creation_receipt_id"]),
                    association_id=int(guard["association_id"]),
                    domain=str(guard["domain"]),
                    context_scope_hash=str(guard["context_scope_hash"]),
                    rewrite_commitment=str(guard["rewrite_commitment"]),
                    binding_commitment=str(guard["binding_commitment"]),
                    manifest_binding_commitment=old_second_mac,
                    ready_manifest_commitment=old_third_mac,
                    context_cue_vector_fingerprint=str(
                        guard["context_cue_vector_fingerprint"]
                    ),
                    need_cue_vector_fingerprint=str(
                        guard["need_cue_vector_fingerprint"]
                    ),
                    model_id=str(guard["model_id"]),
                    embedding_space_id=str(guard["embedding_space_id"]),
                    dimension=int(guard["dimension"]),
                    dtype=str(guard["dtype"]),
                    commitment_key_id=str(guard["commitment_key_id"]),
                    grammar_version=str(guard["grammar_version"]),
                    seed_fingerprint=str(guard["seed_fingerprint"]),
                    signature_version=str(guard["signature_version"]),
                    created_at=str(guard["created_at"]),
                )
                # A database-only attacker can alter lifecycle/public SHA-256
                # metadata after bypassing immutable triggers, but cannot
                # derive a replacement process-keyed second MAC.
                for trigger_name in (
                    "contextual_restricted_rewrite_guard_no_update",
                    "contextual_revisit_runtime_manifest_ready_transition",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "UPDATE association SET expires_at = ? WHERE id = ?",
                    (extended_expiry, int(manifest.association_id)),
                )
                connection.execute(
                    """
                    UPDATE contextual_revisit_runtime_manifest
                    SET expires_at = ?, binding_fingerprint = ?, manifest_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        extended_expiry,
                        public_binding,
                        public_manifest,
                        int(receipt["receipt_id"]),
                    ),
                )
                connection.execute(
                    """
                    UPDATE contextual_restricted_rewrite_guard
                    SET guard_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (public_guard, int(receipt["receipt_id"])),
                )

            # The database's ordinary redacted shape is internally coherent;
            # the private second-MAC check in the runtime lane is what stops
            # this public-field replay.
            rebound = app.associations.load_contextual_restricted_rewrite_guard(
                int(receipt["receipt_id"])
            )
            self.assertIsNotNone(rebound)
            assert rebound is not None
            self.assertEqual(old_second_mac, rebound[0].manifest_binding_commitment)
            self.assertEqual(old_third_mac, rebound[0].ready_manifest_commitment)
            self.assertEqual(public_binding, rebound[1].binding_fingerprint)

            sentinel = {"fallback": "lifecycle-second-mac-rejected"}
            with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_recomputed_ready_publication_cannot_reuse_third_mac(self) -> None:
        """A DB-only ready manifest rewrite loses the final publication HMAC."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            manifest = app.associations.load_contextual_revisit_runtime_manifest(
                int(receipt["receipt_id"])
            )
            assert manifest is not None
            with app.db.transaction() as connection:
                receipt_row = connection.execute(
                    "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                guard = connection.execute(
                    """
                    SELECT * FROM contextual_restricted_rewrite_guard
                    WHERE creation_receipt_id = ?
                    """,
                    (int(receipt["receipt_id"]),),
                ).fetchone()
                assert receipt_row is not None
                assert guard is not None
                tampered_epoch = int(receipt_row["ready_index_epoch"]) + 1
                tampered_receipt = dict(receipt_row)
                tampered_receipt["ready_index_epoch"] = tampered_epoch
                tampered_publication = (
                    app.associations.contextual_revisit_ready_publication_fingerprint(
                        tampered_receipt
                    )
                )
                tampered_contract = replace(
                    manifest.to_revisit_contract(),
                    ready_index_epoch=tampered_epoch,
                    ready_publication_fingerprint=tampered_publication,
                )
                tampered_manifest_fingerprint = (
                    ContextualRevisitRuntimeManifest.expected_manifest_fingerprint(
                        binding_fingerprint=manifest.binding_fingerprint,
                        contract_id=int(manifest.contract_id or 0),
                        contract_fingerprint=tampered_contract.contract_fingerprint,
                        ready_at=manifest.ready_at,
                        ready_index_epoch=tampered_epoch,
                        ready_publication_fingerprint=tampered_publication,
                    )
                )
                old_third_mac = str(guard["ready_manifest_commitment"])
                public_guard = ContextualRestrictedRewriteGuard.expected_fingerprint(
                    creation_receipt_id=int(guard["creation_receipt_id"]),
                    association_id=int(guard["association_id"]),
                    domain=str(guard["domain"]),
                    context_scope_hash=str(guard["context_scope_hash"]),
                    rewrite_commitment=str(guard["rewrite_commitment"]),
                    binding_commitment=str(guard["binding_commitment"]),
                    manifest_binding_commitment=str(
                        guard["manifest_binding_commitment"]
                    ),
                    ready_manifest_commitment=old_third_mac,
                    context_cue_vector_fingerprint=str(
                        guard["context_cue_vector_fingerprint"]
                    ),
                    need_cue_vector_fingerprint=str(
                        guard["need_cue_vector_fingerprint"]
                    ),
                    model_id=str(guard["model_id"]),
                    embedding_space_id=str(guard["embedding_space_id"]),
                    dimension=int(guard["dimension"]),
                    dtype=str(guard["dtype"]),
                    commitment_key_id=str(guard["commitment_key_id"]),
                    grammar_version=str(guard["grammar_version"]),
                    seed_fingerprint=str(guard["seed_fingerprint"]),
                    signature_version=str(guard["signature_version"]),
                    created_at=str(guard["created_at"]),
                )

                # This is deliberately a raw database attacker: every public
                # receipt, publication, contract, manifest and guard digest is
                # rebuilt coherently, while the existing process-only third
                # HMAC is replayed unchanged.
                for trigger_name in (
                    "contextual_creation_receipt_immutable",
                    "contextual_revisit_contract_no_update",
                    "contextual_revisit_runtime_manifest_ready_transition",
                    "contextual_restricted_rewrite_guard_no_update",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                connection.execute(
                    "UPDATE contextual_index_publication SET index_epoch = ? "
                    "WHERE singleton = 1",
                    (tampered_epoch,),
                )
                connection.execute(
                    "UPDATE contextual_creation_receipt SET ready_index_epoch = ? "
                    "WHERE id = ?",
                    (tampered_epoch, int(receipt["receipt_id"])),
                )
                connection.execute(
                    """
                    UPDATE contextual_revisit_contract
                    SET ready_index_epoch = ?, ready_publication_fingerprint = ?,
                        contract_fingerprint = ?
                    WHERE id = ?
                    """,
                    (
                        tampered_epoch,
                        tampered_publication,
                        tampered_contract.contract_fingerprint,
                        int(manifest.contract_id or 0),
                    ),
                )
                connection.execute(
                    """
                    UPDATE contextual_revisit_runtime_manifest
                    SET ready_index_epoch = ?, ready_publication_fingerprint = ?,
                        contract_fingerprint = ?, manifest_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (
                        tampered_epoch,
                        tampered_publication,
                        tampered_contract.contract_fingerprint,
                        tampered_manifest_fingerprint,
                        int(receipt["receipt_id"]),
                    ),
                )
                connection.execute(
                    """
                    UPDATE contextual_restricted_rewrite_guard
                    SET guard_fingerprint = ?
                    WHERE creation_receipt_id = ?
                    """,
                    (public_guard, int(receipt["receipt_id"])),
                )

            # The public data shape remains structurally coherent.  Only the
            # final HMAC, which covers the ready manifest fingerprint, can
            # distinguish this from the trusted V22 publication transaction.
            rebound = app.associations.load_contextual_restricted_rewrite_guard(
                int(receipt["receipt_id"])
            )
            self.assertIsNotNone(rebound)
            assert rebound is not None
            self.assertEqual(old_third_mac, rebound[0].ready_manifest_commitment)
            self.assertEqual(
                tampered_manifest_fingerprint, rebound[1].manifest_fingerprint
            )

            sentinel = {"fallback": "ready-manifest-third-mac-rejected"}
            with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertEqual([], model.calls)

    def test_sidecar_change_between_full_passes_cannot_reach_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            original_guard = engine._exact_revisit_delivery_guard
            guard_calls = 0

            def guard_then_tamper(*args, **kwargs):
                nonlocal guard_calls
                value = original_guard(*args, **kwargs)
                guard_calls += 1
                if guard_calls == 1:
                    with app.db.transaction() as connection:
                        connection.execute(
                            "DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update"
                        )
                        connection.execute(
                            "UPDATE contextual_restricted_rewrite_guard SET guard_fingerprint = ? WHERE creation_receipt_id = ?",
                            (
                                "tampered:sha256:" + "1" * 64,
                                int(receipt["receipt_id"]),
                            ),
                        )
                return value

            sentinel = {"fallback": True}
            with patch.object(
                engine,
                "_exact_revisit_delivery_guard",
                side_effect=guard_then_tamper,
            ), patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=False,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            self.assertGreaterEqual(guard_calls, 1)
            self.assertEqual([], model.calls)

    def test_change_after_second_guard_cannot_start_answer_generation(self) -> None:
        """The final pre-answer snapshot check closes the second-pass gap."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.hmac_environment, clear=False
        ):
            app, model, engine, scope_hash, receipt, _anchor, _target = self._fixture(
                directory
            )
            original_guard = engine._exact_revisit_delivery_guard
            guard_calls = 0

            def second_guard_then_tamper(*args, **kwargs):
                nonlocal guard_calls
                value = original_guard(*args, **kwargs)
                guard_calls += 1
                if guard_calls == 2:
                    with app.db.transaction() as connection:
                        connection.execute(
                            "DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update"
                        )
                        connection.execute(
                            """
                            UPDATE contextual_restricted_rewrite_guard
                            SET guard_fingerprint = ?
                            WHERE creation_receipt_id = ?
                            """,
                            (
                                "tampered:sha256:" + "2" * 64,
                                int(receipt["receipt_id"]),
                            ),
                        )
                return value

            sentinel = {"fallback": "final-answer-guard"}
            with patch.object(
                engine,
                "_exact_revisit_delivery_guard",
                side_effect=second_guard_then_tamper,
            ), patch.object(
                engine,
                "_generate_audited_answer",
                return_value=("must-not-run", [], 0),
            ) as generate, patch.object(
                engine,
                "_query_impl",
                return_value=sentinel,
            ) as fallback:
                result = engine.query(
                    self.rewrite_question,
                    generate_answer=True,
                    contextual_domain=_v17.ContextualAutomaticRevisitV17Tests.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            fallback.assert_called_once()
            generate.assert_not_called()
            self.assertGreaterEqual(guard_calls, 3)
            self.assertEqual([], model.calls)
