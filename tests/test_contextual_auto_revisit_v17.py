from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingCoordinator
from memory_demo.event_log import JsonlEventLogger
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.retrieval.revisit import (
    exact_revisit_aggregate_mapping_ref,
    exact_revisit_anchor_manifest_fingerprint,
    exact_revisit_budget_fingerprint,
    exact_revisit_context_hash,
    exact_revisit_policy_fingerprint,
    exact_revisit_request_hash,
    exact_revisit_requirements_fingerprint,
    exact_revisit_slot_need_bindings,
    exact_revisit_source_closure_fingerprint,
    exact_revisit_source_fact_refs_fingerprint,
)
from memory_demo.types import (
    ContextualRevisitRuntimeSeed,
    EvidenceSelectionBudget,
    EvidenceSlot,
    LearningCandidate,
    QueryVectorRequest,
)


def _opaque(namespace: str, value: str) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class _NoProviderModel:
    """A local model fixture: a V17 hit must never invoke planning/embedding."""

    embedding_provider = "v17-auto-local"
    embedding_revision = "v1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, _texts):
        self.calls.append("embedding")
        raise AssertionError("automatic revisit must not embed")

    def chat_json(self, *_args, **_kwargs):
        self.calls.append("chat_json")
        raise AssertionError("automatic revisit must not plan or rerank")

    def chat_text(self, *_args, **_kwargs):
        self.calls.append("chat_text")
        raise AssertionError("generate_answer=False must not invoke an answer model")


class ContextualAutomaticRevisitV17Tests(unittest.TestCase):
    """Public Q2 tests using only local SQLite, indexes and fake models."""

    question = "What does the local target record establish?"
    domain = "knowledge"

    @staticmethod
    def _scope(value: str = "private-conversation") -> str:
        return _opaque("v17-scope", value)

    @staticmethod
    def _config(directory: str) -> AppConfig:
        config = AppConfig(
            database_path=Path(directory) / "v17-auto-revisit.db",
            log_dir=Path(directory) / "logs",
            model=ModelConfig(
                embedding_model="v17-auto-local-embedding",
                embedding_dimension=3,
            ),
        )
        config.retrieval.contextual_association_enabled = True
        config.retrieval.contextual_association_shadow = False
        config.retrieval.contextual_context_threshold = 0.10
        config.retrieval.contextual_need_threshold = 0.10
        config.retrieval.answer_episode_limit = 2
        config.retrieval.sparse_enabled = False
        config.retrieval.source_key_cohort_enabled = False
        config.retrieval.graph_max_hops = 0
        config.retrieval.growth_max_rounds = 0
        return config

    @staticmethod
    def _counts(app: MemoryApplication) -> tuple[int, ...]:
        with app.db.connection() as connection:
            return tuple(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "source",
                    "episode",
                    "association_cue_prototype",
                    "association",
                    "contextual_creation_receipt",
                    "contextual_revisit_contract",
                    "contextual_revisit_runtime_manifest",
                )
            )

    @staticmethod
    def _restricted_rewrite_guard_for_seed(**_kwargs):
        """T16/V17 fixture hook; V17 itself never emits a rewrite guard."""

        return None

    @staticmethod
    def _restricted_rewrite_manifest_binding_signer(**_kwargs):
        """T16 fixture hook for the process-only post-manifest HMAC signer."""

        return None

    @staticmethod
    def _restricted_rewrite_ready_manifest_signer(**_kwargs):
        """V22 fixture hook for the process-only ready-manifest HMAC signer."""

        return None

    @staticmethod
    def _seed_episodes(app: MemoryApplication) -> tuple[int, int]:
        marker = "2026-09-06T00:00:00+00:00"
        ids: list[int] = []
        with app.db.transaction() as connection:
            for label, body in (
                ("anchor", "Anchor record is independently source-bound."),
                ("target", "Target record establishes the requested local fact."),
            ):
                raw_source = f"[record: {label}]\n{body}"
                source_id = int(
                    connection.execute(
                        "INSERT INTO source(raw_text) VALUES(?)", (raw_source,)
                    ).lastrowid
                )
                ids.append(
                    int(
                        connection.execute(
                            """
                            INSERT INTO episode(
                                source_id, source_key, segment_index, text,
                                evidence_origin, epistemic_status, generation,
                                evidence_quotes_json, evidence_spans_json, evidence_basis,
                                embedding, created_at, updated_at
                            ) VALUES(?, ?, 0, ?, 'source', 'observed', 0, ?, ?,
                                     'source_id', ?, ?, ?)
                            """,
                            (
                                source_id,
                                f"v17/{label}.txt",
                                body,
                                json.dumps([raw_source]),
                                json.dumps([[1, 2]]),
                                np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                                marker,
                                marker,
                            ),
                        ).lastrowid
                    )
                )
        app.rebuild_indexes()
        return tuple(ids)  # type: ignore[return-value]

    @staticmethod
    def _engine(app: MemoryApplication, model: _NoProviderModel) -> QueryEngine:
        coordinator = EmbeddingCoordinator(
            model,
            model_id=app.config.model.embedding_model,
            dimension=app.config.model.embedding_dimension,
        )
        matcher = ContextualAssociationMatcher(
            app.context_cue_index,
            app.need_cue_index,
            app.associations,
            context_threshold=app.config.retrieval.contextual_context_threshold,
            need_threshold=app.config.retrieval.contextual_need_threshold,
            edge_top_k=1,
            endpoint_limit=1,
            embedding_space_id=coordinator.embedding_space().canonical_id,
        )
        return QueryEngine(
            app.config,
            model,
            app.episode_index,
            app.concept_index,
            app.episodes,
            app.concepts,
            app.sources,
            app.associations,
            contextual_matcher=matcher,
        )

    def _fixture(self, directory: str):
        app = MemoryApplication(self._config(directory))
        anchor_id, target_id = self._seed_episodes(app)
        model = _NoProviderModel()
        engine = self._engine(app, model)
        facts, reasons = engine._v3_source_fact_closure((anchor_id, target_id))
        self.assertEqual(
            {anchor_id: "source_bound", target_id: "source_bound"}, reasons
        )

        scope_hash = self._scope()
        normalized_question = self.question
        question_hash = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
        source_request_hash = exact_revisit_request_hash(normalized_question)
        context_hash = exact_revisit_context_hash(
            normalized_question, context_scope_hash=scope_hash
        )
        runtime_slot_ref = _opaque("runtime-slot-ref", "v17-auto")
        runtime_query_ref = _opaque("runtime-query-ref", "v17-auto")
        runtime_clause_ref = _opaque("runtime-clause-ref", "v17-auto")
        runtime_slot_id = "runtime-slot:" + runtime_slot_ref.rsplit(":", 1)[-1]
        runtime_query_id = "runtime-query:" + runtime_query_ref.rsplit(":", 1)[-1]
        runtime_clause_id = "runtime-clause:" + runtime_clause_ref.rsplit(":", 1)[-1]
        whole_query_id = "runtime-whole:" + runtime_query_ref.rsplit(":", 1)[-1]
        slot = EvidenceSlot(
            slot_id=runtime_slot_id,
            question=normalized_question,
            required=True,
            query_id=runtime_query_id,
            query_refs=(runtime_query_id,),
            origin="reused_template",
            support_mode="alternative",
            clause_ids=(runtime_clause_id,),
        )
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(slot,),
            planner_origin="reused_template",
        )
        coordinator = EmbeddingCoordinator(
            model,
            model_id=app.config.model.embedding_model,
            dimension=app.config.model.embedding_dimension,
        )
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        bundle = coordinator.bundle_from_precomputed_vectors(
            (
                QueryVectorRequest(
                    role="whole",
                    text=normalized_question,
                    query_id=whole_query_id,
                ),
                QueryVectorRequest(
                    role="atomic",
                    text=normalized_question,
                    query_id=runtime_query_id,
                    slot_id=runtime_slot_id,
                ),
            ),
            {normalized_question: vector},
            source_request_hash=source_request_hash,
        )
        verification_ref = _opaque("verification", "v17-auto")
        mapping_ref = exact_revisit_aggregate_mapping_ref(
            target_episode_id=target_id,
            slot_id=runtime_slot_id,
            clause_ids=(runtime_clause_id,),
            source_fact_id=facts[target_id].fact_id,
            verification_refs=(verification_ref,),
        )
        budget = EvidenceSelectionBudget(episode_limit=2)
        seed = ContextualRevisitRuntimeSeed(
            creation_request_id="v17-auto-candidate",
            domain=self.domain,
            context_scope_hash=scope_hash,
            context_hash=context_hash,
            source_request_hash=source_request_hash,
            context_cue_text_hash=question_hash,
            need_cue_text_hash=question_hash,
            slot_need_bindings=exact_revisit_slot_need_bindings(requirements, bundle),
            requirements_fingerprint=exact_revisit_requirements_fingerprint(requirements),
            source_closure_fingerprint=exact_revisit_source_closure_fingerprint(
                (facts[anchor_id], facts[target_id])
            ),
            retrieval_policy_fingerprint=exact_revisit_policy_fingerprint(
                app.config, engine.contextual_matcher, endpoint_limit=1
            ),
            budget_fingerprint=exact_revisit_budget_fingerprint(budget),
            anchor_manifest_fingerprint=exact_revisit_anchor_manifest_fingerprint(
                {anchor_id: 1.0}
            ),
            source_fact_refs_fingerprint=exact_revisit_source_fact_refs_fingerprint(
                (facts[anchor_id], facts[target_id])
            ),
            anchor_episode_id=anchor_id,
            anchor_activation=1.0,
            anchor_source_fact_id=facts[anchor_id].fact_id,
            target_episode_id=target_id,
            target_source_fact_id=facts[target_id].fact_id,
            target_mapping_ref=mapping_ref,
            runtime_slot_ref=runtime_slot_ref,
            runtime_query_ref=runtime_query_ref,
            runtime_clause_ref=runtime_clause_ref,
            endpoint_limit=1,
            episode_limit=2,
            source_fact_limit=None,
            delivery_token_limit=None,
        )
        candidate = LearningCandidate(
            candidate_id=seed.creation_request_id,
            anchor_type="episode",
            anchor_id=anchor_id,
            target_episode_id=target_id,
            reason="candidate_missing",
            request_id="v17-auto-request",
            request_hash=_opaque("request", "v17-auto"),
            source_request_hash=source_request_hash,
            context_query_id=whole_query_id,
            need_query_id=runtime_query_id,
            slot_id=runtime_slot_id,
            context_vector_ref=bundle.whole_physical_id,
            need_vector_ref=bundle.bindings_for_slot(runtime_slot_id)[0].physical_id,
            anchor_vector_ref=bundle.whole_physical_id,
            source_facts=(facts[anchor_id], facts[target_id]),
            verification_refs=(verification_ref,),
            verification_status="source_bound",
            anchor_contribution_id=_opaque("contribution", "v17-auto-anchor"),
            anchor_provenance_refs=(_opaque("anchor", "v17-auto"),),
            target_provenance_refs=(_opaque("target", "v17-auto"),),
        )
        receipt = app.associations.finalize_contextual_creation(
            candidate,
            domain=self.domain,
            model_id=app.config.model.embedding_model,
            dimension=app.config.model.embedding_dimension,
            context_vector=vector,
            need_vector=vector,
            context_text_hash=question_hash,
            need_text_hash=question_hash,
            embedding_space_id=bundle.embedding_space_id,
            creation_request_id=seed.creation_request_id,
            creation_request_hash=candidate.request_hash,
            revisit_runtime_seed=seed,
            restricted_rewrite_guard=self._restricted_rewrite_guard_for_seed(
                question=normalized_question,
                requirements=requirements,
                seed=seed,
            ),
            restricted_rewrite_manifest_binding_signer=(
                self._restricted_rewrite_manifest_binding_signer(
                    question=normalized_question,
                    requirements=requirements,
                    seed=seed,
                )
            ),
        )
        self.assertEqual("committed_pending_index", receipt["status"])
        app.context_cue_index.upsert(int(receipt["context_cue_id"]), vector)
        app.need_cue_index.upsert(int(receipt["need_cue_id"]), vector)
        ready = app.associations.mark_contextual_receipt_ready(
            int(receipt["receipt_id"]),
            context_cue_count=app.context_cue_index.count,
            need_cue_count=app.need_cue_index.count,
            expected_context_cue_id=int(receipt["context_cue_id"]),
            expected_need_cue_id=int(receipt["need_cue_id"]),
            restricted_rewrite_ready_manifest_signer=(
                self._restricted_rewrite_ready_manifest_signer(
                    question=normalized_question,
                    requirements=requirements,
                    seed=seed,
                )
            ),
        )
        promoted = app.associations.promote_contextual_revisit_runtime_manifest(
            int(receipt["receipt_id"])
        )
        self.assertEqual("ready", ready["status"])
        self.assertIsNotNone(promoted)
        manifest = app.associations.load_contextual_revisit_runtime_manifest(
            int(receipt["receipt_id"])
        )
        self.assertIsNotNone(manifest)
        return app, model, engine, scope_hash, receipt, anchor_id, target_id

    def _fallback(self, engine: QueryEngine, **kwargs):
        sentinel = {"fallback": True}
        with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
            result = engine.query(self.question, generate_answer=False, **kwargs)
        self.assertIs(sentinel, result)
        fallback.assert_called_once()

    def test_automatic_hit_reconstructs_from_ready_manifest_without_provider_work(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, scope_hash, _receipt, _anchor, target_id = self._fixture(directory)
            before = self._counts(app)

            result = engine.query(
                self.question,
                generate_answer=False,
                contextual_domain=self.domain,
                contextual_revisit_scope_hash=scope_hash,
            )

            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)
            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertTrue(result["exact_revisit"]["planner_skipped"])
            self.assertTrue(result["exact_revisit"]["embedding_skipped"])
            self.assertIn(target_id, result["episode_ids"])
            # The output deliberately remains the existing exact-result shape:
            # no durable manifest identity or reconstructed ticket is leaked.
            self.assertNotIn("manifest_id", result["exact_revisit"])
            self.assertNotIn("runtime_manifest", result["exact_revisit"])

    def test_public_preflight_returns_a_local_exact_hit_without_query_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, scope_hash, _receipt, _anchor, target_id = self._fixture(directory)
            before = self._counts(app)

            with patch.object(
                engine,
                "_query_impl",
                side_effect=AssertionError("preflight must not run ordinary retrieval"),
            ) as ordinary_query:
                result = engine.try_contextual_revisit(
                    self.question,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )

            ordinary_query.assert_not_called()
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)
            self.assertIsNotNone(result)
            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertTrue(result["exact_revisit"]["planner_skipped"])
            self.assertTrue(result["exact_revisit"]["embedding_skipped"])
            self.assertIn(target_id, result["episode_ids"])

    def test_public_preflight_miss_returns_none_without_query_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            _app, model, engine, _scope_hash, _receipt, _anchor, _target = self._fixture(directory)

            with patch.object(
                engine,
                "_query_impl",
                side_effect=AssertionError("preflight must not run ordinary retrieval"),
            ) as ordinary_query:
                result = engine.try_contextual_revisit(
                    self.question,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=self._scope("other-conversation"),
                )

            ordinary_query.assert_not_called()
            self.assertIsNone(result)
            self.assertEqual([], model.calls)

    def test_public_preflight_never_appends_contextual_event_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            _app, model, engine, scope_hash, _receipt, _anchor, _target = self._fixture(directory)
            log_path = Path(directory) / "preflight-events.jsonl"
            log_path.write_text("existing-event\n", encoding="utf-8")
            engine.logger = JsonlEventLogger(log_path)
            before = log_path.read_bytes()

            with patch.object(
                engine,
                "_query_impl",
                side_effect=AssertionError("preflight must not run ordinary retrieval"),
            ) as ordinary_query:
                hit = engine.try_contextual_revisit(
                    self.question,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
                self.assertIsNotNone(hit)
                self.assertEqual(before, log_path.read_bytes())

                miss = engine.try_contextual_revisit(
                    self.question,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=self._scope("other-conversation"),
                )

            ordinary_query.assert_not_called()
            self.assertIsNone(miss)
            self.assertEqual(before, log_path.read_bytes())
            self.assertEqual([], model.calls)

    def test_restart_rebuild_keeps_the_same_auto_path_local(self):
        with tempfile.TemporaryDirectory() as directory:
            app, _model, _engine, scope_hash, _receipt, _anchor, target_id = self._fixture(directory)
            restarted = MemoryApplication(app.config)
            restarted.rebuild_indexes()
            rebuild = restarted.rebuild_contextual_indexes()
            self.assertGreaterEqual(int(rebuild["context_prototypes"]), 1)
            self.assertGreaterEqual(int(rebuild["need_prototypes"]), 1)
            model = _NoProviderModel()
            engine = self._engine(restarted, model)

            result = engine.query(
                self.question,
                generate_answer=False,
                contextual_domain=self.domain,
                contextual_revisit_scope_hash=scope_hash,
            )

            self.assertEqual([], model.calls)
            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertIn(target_id, result["episode_ids"])

    def test_missing_or_wrong_scope_is_an_ordinary_path_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            _app, model, engine, _scope_hash, _receipt, _anchor, _target = self._fixture(directory)
            self._fallback(engine, contextual_domain=self.domain)
            self.assertEqual([], model.calls)
            self._fallback(
                engine,
                contextual_domain=self.domain,
                contextual_revisit_scope_hash=self._scope("other-conversation"),
            )
            self.assertEqual([], model.calls)

    def test_tampered_cue_manifest_or_source_is_an_ordinary_path_miss(self):
        for change in ("cue", "manifest", "source"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                app, model, engine, scope_hash, receipt, _anchor, target = self._fixture(directory)
                with app.db.transaction() as connection:
                    if change == "cue":
                        connection.execute(
                            "UPDATE association_cue_prototype SET vector_blob = ? WHERE id = ?",
                            (
                                np.asarray([0.0, 1.0, 0.0], dtype=np.float32).tobytes(),
                                int(receipt["context_cue_id"]),
                            ),
                        )
                    elif change == "manifest":
                        # The production trigger makes a ready manifest
                        # immutable.  Disable it only in this isolated temp
                        # database to simulate an out-of-band corruption.
                        connection.execute(
                            "DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_ready_transition"
                        )
                        connection.execute(
                            "UPDATE contextual_revisit_runtime_manifest SET manifest_fingerprint = ?",
                            (_opaque("manifest", "tampered"),),
                        )
                    else:
                        source_id = int(
                            connection.execute(
                                "SELECT source_id FROM episode WHERE id = ?", (target,)
                            ).fetchone()[0]
                        )
                        connection.execute(
                            "UPDATE source SET raw_text = ? WHERE id = ?",
                            ("[record: target]\nrevised source invalidates old evidence", source_id),
                        )
                self._fallback(
                    engine,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
                self.assertEqual([], model.calls)

    def test_manifest_view_change_between_exact_passes_falls_through(self):
        """The private manifest identity is part of the delivery recheck."""

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, scope_hash, _receipt, _anchor, _target = self._fixture(directory)
            original_guard = engine._exact_revisit_delivery_guard
            original_load = app.associations.load_contextual_revisit_runtime_manifest
            changed = False
            guard_calls = 0

            def load_manifest(*args, **kwargs):
                if changed:
                    return None
                return original_load(*args, **kwargs)

            def guard_then_change(*args, **kwargs):
                nonlocal changed, guard_calls
                self.assertIsNotNone(kwargs.get("expected_runtime_manifest_id"))
                self.assertTrue(kwargs.get("expected_runtime_manifest_fingerprint"))
                guard = original_guard(*args, **kwargs)
                guard_calls += 1
                if guard_calls == 1:
                    changed = True
                return guard

            sentinel = {"fallback": True}
            with patch.object(
                app.associations,
                "load_contextual_revisit_runtime_manifest",
                side_effect=load_manifest,
            ), patch.object(
                engine,
                "_exact_revisit_delivery_guard",
                side_effect=guard_then_change,
            ), patch.object(engine, "_query_impl", return_value=sentinel):
                result = engine.query(
                    self.question,
                    generate_answer=False,
                    contextual_domain=self.domain,
                    contextual_revisit_scope_hash=scope_hash,
                )
            self.assertIs(sentinel, result)
            self.assertGreaterEqual(guard_calls, 1)
            self.assertEqual([], model.calls)

    def test_automatic_probe_obeys_the_shared_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, scope_hash, _receipt, _anchor, _target = self._fixture(directory)
            original_find = app.associations.find_contextual_revisit_runtime_manifest

            def delayed_find(*args, **kwargs):
                time.sleep(0.04)
                return original_find(*args, **kwargs)

            with patch.object(
                app.associations,
                "find_contextual_revisit_runtime_manifest",
                side_effect=delayed_find,
            ):
                with self.assertRaisesRegex(
                    TimeoutError, r"^query deadline exceeded before automatic_revisit",
                ):
                    engine.query(
                        self.question,
                        generate_answer=False,
                        contextual_domain=self.domain,
                        contextual_revisit_scope_hash=scope_hash,
                        deadline_seconds=0.005,
                    )
            self.assertEqual([], model.calls)
