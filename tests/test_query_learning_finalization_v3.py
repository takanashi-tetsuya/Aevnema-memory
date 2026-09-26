from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.llm.prompts import (
    ANSWER_AUDIT_SYSTEM,
    ANSWER_SYSTEM,
    EVIDENCE_RERANK_SYSTEM,
)
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.types import (
    ContextualRestrictedRewriteGuardDraft,
    ContextualRevisitRuntimeSeed,
    EvidenceSlot,
    QueryIntent,
    QueryVector,
    QueryVectorBundle,
)


class _DirectBaseFixtureModel:
    """Small local provider fixture for a full, no-edge V3 query."""

    answer = "The local record confirms the observed fact."

    def __init__(self, requirement: str, episode_ids: tuple[int, int]) -> None:
        self.requirement = requirement
        self.episode_ids = episode_ids
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append("embedding")
        return np.asarray(
            [[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32
        )

    def chat_json(self, system: str, _prompt: str, **_kwargs: object) -> dict:
        if system == EVIDENCE_RERANK_SYSTEM:
            self.calls.append("rerank")
            return {
                "selected_episode_ids": [self.episode_ids[0]],
                # Both direct retrieval results have a pre-existing, frozen
                # source-mapping observation for this one required slot.  The
                # selector may choose only one under its answer budget, while
                # the other remains a separately proven direct anchor.
                "coverage": [
                    {
                        "query": self.requirement,
                        "episode_ids": list(self.episode_ids),
                    }
                ],
            }
        if system == ANSWER_AUDIT_SYSTEM:
            self.calls.append("answer_audit")
            return {
                "reviews": [
                    {
                        "claim": self.answer,
                        "verdict": "supported_fact",
                    }
                ]
            }
        raise AssertionError(f"unexpected local fixture JSON system: {system!r}")

    def chat_text(self, system: str, _prompt: str, **_kwargs: object) -> str:
        if system != ANSWER_SYSTEM:
            raise AssertionError(f"unexpected local fixture text system: {system!r}")
        self.calls.append("answer")
        return self.answer


class QueryLearningFinalizationV3Tests(unittest.TestCase):
    """T13b local-only public-query learning boundary tests.

    Most cases inject a controlled, already source-bound V3 selector capture
    to isolate the post-answer finalization boundary.  The direct-base case
    below additionally runs the full local request/retrieval/selector path.
    In every case source closure, candidate derivation, repository transaction,
    and RAM publication remain real local components.
    """

    restricted_rewrite_hmac_environment = {
        "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY_ID": "test-key-v1",
        "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY": (
            "local-test-restricted-rewrite-key"
        ),
    }

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _application(self, directory: str) -> MemoryApplication:
        config = AppConfig(
            database_path=Path(directory) / "memory.db",
            log_dir=Path(directory) / "logs",
            model=ModelConfig(
                embedding_model="t13b-local-embedding",
                embedding_dimension=3,
            ),
        )
        config.retrieval.contextual_association_enabled = True
        config.retrieval.contextual_association_shadow = False
        # The test replaces only _query_impl, so avoid staging unrelated graph
        # growth around the public query boundary.
        config.retrieval.growth_max_rounds = 0
        return MemoryApplication(config)

    def _seed_source_bound_episodes(
        self,
        app: MemoryApplication,
        *,
        extra_targets: int = 0,
        invalid_target_provenance: bool = False,
    ) -> tuple[int, ...]:
        rows = [
            ("[record: anchor]\nanchor source proof", "t13b/anchor.json"),
            ("[record: target]\ntarget source proof", "t13b/target.json"),
            *[
                (
                    f"[record: target-{index}]\ntarget-{index} source proof",
                    f"t13b/target-{index}.json",
                )
                for index in range(2, max(2, int(extra_targets) + 2))
            ],
        ]
        episode_ids: list[int] = []
        marker = "2026-09-06T00:00:00+00:00"
        with app.db.transaction() as connection:
            for index, (raw_text, source_key) in enumerate(rows):
                source_id = int(
                    connection.execute(
                        "INSERT INTO source(raw_text) VALUES(?)", (raw_text,)
                    ).lastrowid
                )
                cursor = connection.execute(
                    """
                    INSERT INTO episode(
                        source_id, source_key, segment_index, text,
                        evidence_origin, epistemic_status, generation,
                        evidence_quotes_json, evidence_spans_json, evidence_basis,
                        embedding, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, 'source', 'observed', 0, ?, ?,
                             'source_id', ?, ?, ?)
                    """,
                    (
                        source_id,
                        source_key,
                        index,
                        f"episode-{index + 1}",
                        json.dumps(
                            [
                                "quote absent from the persisted source span"
                                if invalid_target_provenance and index == 1
                                else raw_text
                            ]
                        ),
                        json.dumps([[1, 2]]),
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                        marker,
                        marker,
                    ),
                )
                episode_ids.append(int(cursor.lastrowid))
        return tuple(episode_ids)  # type: ignore[return-value]

    def _engine_and_capture(
        self,
        app: MemoryApplication,
        *,
        question: str,
        shadow: bool = False,
        answer: str = "local audited answer",
        answer_generation_skipped: object = False,
        answer_terminal_state: dict[str, object] | None = None,
        bundle_source_request_hash: str | None = None,
        independent_base: bool = True,
        target_gate_completed: bool = True,
        direct_base_validation_completed: bool = False,
        invalid_target_provenance: bool = False,
        extra_targets: int = 0,
        runtime_eligible: bool = False,
    ) -> tuple[QueryEngine, dict[str, object]]:
        episode_ids = self._seed_source_bound_episodes(
            app,
            extra_targets=extra_targets,
            invalid_target_provenance=invalid_target_provenance,
        )
        anchor_id, *target_ids = episode_ids
        engine = QueryEngine(
            app.config,
            model=object(),
            episode_index=app.episode_index,
            concept_index=app.concept_index,
            episodes=app.episodes,
            concepts=app.concepts,
            sources=app.sources,
            associations=app.associations,
            contextual_matcher=object(),
            contextual_learning_finalizer=app.finalize_recall_event,
        )
        slot_question = (
            question if runtime_eligible else "which source-bound evidence is needed"
        )
        slot = EvidenceSlot(
            slot_id="need-slot",
            question=slot_question,
            query_id="need-query",
            clause_ids=("need-clause",),
        )
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(slot,),
            planner_origin="controlled_local_fixture",
        )
        expected_source_request_hash = "sha256:" + engine._request_vector_hash(
            question
        )
        bundle = QueryVectorBundle(
            model_id=app.config.model.embedding_model,
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            source_request_hash=(
                expected_source_request_hash
                if bundle_source_request_hash is None
                else bundle_source_request_hash
            ),
            queries=(
                QueryVector(
                    query_id="whole-query",
                    text_hash=self._hash(question),
                    role="whole",
                    vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    text=question,
                ),
                QueryVector(
                    query_id="need-query",
                    text_hash=self._hash(slot.question),
                    role="atomic",
                    vector=np.asarray(
                        [1.0, 0.0, 0.0]
                        if runtime_eligible
                        else [0.0, 1.0, 0.0],
                        dtype=np.float32,
                    ),
                    slot_id=slot.slot_id,
                    text=slot.question,
                ),
            ),
        )
        contributions, _reasons, _trace = engine._v3_build_candidate_contributions(
            episodes=tuple(
                {
                    "id": episode_id,
                    "score": 1.0 - (index * 0.1),
                    "source_key": "local-only",
                }
                for index, episode_id in enumerate(episode_ids, start=1)
            ),
            slots=(slot,),
            slot_support={episode_id: {slot.slot_id} for episode_id in episode_ids},
            base_episode_ids=episode_ids,
        )

        def controlled_query_impl(*_args: object, **_kwargs: object) -> dict[str, object]:
            # The target was absent from the earlier independent candidate
            # snapshot but is a current source-bound base contribution in the
            # final V3 selector universe.  This is the valid
            # candidate-missing learning case; it is not a contextual target.
            engine._capture_v3_learning_selection(
                slots=(slot,),
                contributions=contributions,
                bundle=bundle,
                initial_candidate_episode_ids=(anchor_id,),
                independent_base_episode_ids=(anchor_id,) if independent_base else (),
                initial_delivered_episode_ids=(anchor_id,),
                final_selected_episode_ids=episode_ids,
                contextual_expansion_episode_ids=(),
                cue_endpoint_episode_ids=(),
                missing_required_clauses=(),
                delivery_loss=False,
                shadow=shadow,
                target_gate_completed=target_gate_completed,
                direct_base_validation_completed=direct_base_validation_completed,
                requirements=requirements,
                endpoint_limit=1,
            )
            return {
                "answer": answer,
                "answer_generation_skipped": answer_generation_skipped,
                "answer_terminal_state": (
                    {"terminal_state": "completed_verified"}
                    if answer_terminal_state is None
                    else answer_terminal_state
                ),
                "episode_ids": list(episode_ids),
                "answer_audits": [{"valid": True}],
                "rerank_trace": {},
            }

        return engine, {
            "query_impl": controlled_query_impl,
            "target_ids": tuple(target_ids),
        }

    @staticmethod
    def _creation_counts(app: MemoryApplication) -> tuple[int, int, int]:
        with app.db.connection() as connection:
            return tuple(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "association_cue_prototype",
                    "association",
                    "contextual_creation_receipt",
                )
            )

    def test_whole_question_runtime_projection_aligns_seed_and_v16_draft(self):
        """Q1 may persist a V17 seed only for its strict reconstructable shape."""

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "whole question runtime projection"
            engine, state = self._engine_and_capture(
                app,
                question=question,
                runtime_eligible=True,
            )
            observed: dict[str, dict[str, object]] = {}

            def observe_materializations(event, plan, materializations):
                observed.update(
                    {
                        str(candidate_id): dict(materialization)
                        for candidate_id, materialization in materializations.items()
                    }
                )
                return app.finalize_recall_event(event, plan, materializations)

            engine.contextual_learning_finalizer = observe_materializations
            scope_hash = "test-scope:sha256:" + self._hash("runtime-projection-scope")
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                result = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-runtime-projection",
                    contextual_domain="knowledge",
                    contextual_revisit_scope_hash=scope_hash,
                )

            self.assertEqual("ready", result["contextual_learning"]["status"])
            self.assertEqual(1, len(observed))
            materialization = next(iter(observed.values()))
            seed = materialization.get("revisit_runtime_seed")
            draft = materialization.get("revisit_contract_draft")
            self.assertIsInstance(seed, ContextualRevisitRuntimeSeed)
            self.assertIsNotNone(draft)
            assert isinstance(seed, ContextualRevisitRuntimeSeed)
            self.assertEqual(seed.slot_need_bindings, draft.slot_need_bindings)
            self.assertEqual(
                seed.requirements_fingerprint, draft.requirements_fingerprint
            )
            self.assertEqual(seed.runtime_slot_id, draft.target_mapping.slot_id)
            self.assertEqual(
                (seed.runtime_clause_id,), draft.target_mapping.clause_ids
            )
            self.assertEqual(seed.target_mapping_ref, draft.target_mapping.mapping_ref)

            receipt_id = result["contextual_learning"]["receipt_ids"][0]
            manifest = app.associations.load_contextual_revisit_runtime_manifest(
                receipt_id
            )
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual("ready", manifest.state)
            self.assertEqual(seed, manifest.seed)
            self.assertEqual(
                draft.source_fact_roles_fingerprint(manifest.association_id),
                manifest.source_fact_roles_fingerprint,
            )
            self.assertNotIn(question, json.dumps(seed.canonical_payload()))

    def test_restricted_rewrite_learning_callback_never_receives_q1_signer(self):
        """The app creates the Q1 signer only after the generic callback boundary."""

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, self.restricted_rewrite_hmac_environment, clear=False
        ):
            app = self._application(directory)
            app.config.retrieval.contextual_restricted_rewrite_enabled = True
            question = "What is the title of Alice?"
            engine, state = self._engine_and_capture(
                app,
                question=question,
                runtime_eligible=True,
            )
            observed: dict[str, dict[str, object]] = {}

            def observe_materializations(event, plan, materializations):
                observed.update(
                    {
                        str(candidate_id): dict(materialization)
                        for candidate_id, materialization in materializations.items()
                    }
                )
                return app.finalize_recall_event(event, plan, materializations)

            engine.contextual_learning_finalizer = observe_materializations
            scope_hash = "test-scope:sha256:" + self._hash(
                "restricted-rewrite-generic-finalizer"
            )
            with patch.object(
                app,
                "_restricted_rewrite_manifest_binding_signer",
                wraps=app._restricted_rewrite_manifest_binding_signer,
            ) as trusted_signer, patch.object(
                engine, "_query_impl", side_effect=state["query_impl"]
            ):
                result = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-restricted-rewrite-finalizer",
                    contextual_domain="knowledge",
                    contextual_revisit_scope_hash=scope_hash,
                )

            self.assertEqual("ready", result["contextual_learning"]["status"])
            trusted_signer.assert_called_once_with()
            self.assertEqual(1, len(observed))
            materialization = next(iter(observed.values()))
            self.assertIsInstance(
                materialization.get("restricted_rewrite_guard"),
                ContextualRestrictedRewriteGuardDraft,
            )
            self.assertNotIn(
                "restricted_rewrite_manifest_binding_signer", materialization
            )

            receipt_id = result["contextual_learning"]["receipt_ids"][0]
            ready_guard = app.associations.load_contextual_restricted_rewrite_guard(
                receipt_id
            )
            self.assertIsNotNone(ready_guard)
            assert ready_guard is not None
            self.assertTrue(ready_guard[0].manifest_binding_commitment)
            self.assertTrue(ready_guard[0].ready_manifest_commitment)

    def test_nonreconstructable_q1_candidate_keeps_ordinary_v16_draft_without_seed(self):
        """A different need text remains ordinary V16 Q1 material, not V17."""

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "ordinary v16 fallback"
            engine, state = self._engine_and_capture(app, question=question)
            observed: dict[str, dict[str, object]] = {}

            def observe_materializations(event, plan, materializations):
                observed.update(
                    {
                        str(candidate_id): dict(materialization)
                        for candidate_id, materialization in materializations.items()
                    }
                )
                return app.finalize_recall_event(event, plan, materializations)

            engine.contextual_learning_finalizer = observe_materializations
            scope_hash = "test-scope:sha256:" + self._hash("ordinary-v16-scope")
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                result = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-ordinary-v16",
                    contextual_domain="knowledge",
                    contextual_revisit_scope_hash=scope_hash,
                )

            self.assertEqual("ready", result["contextual_learning"]["status"])
            self.assertEqual(1, len(observed))
            materialization = next(iter(observed.values()))
            self.assertIn("revisit_contract_draft", materialization)
            self.assertNotIn("revisit_runtime_seed", materialization)
            receipt_id = result["contextual_learning"]["receipt_ids"][0]
            self.assertIsNone(
                app.associations.load_contextual_revisit_runtime_manifest(receipt_id)
            )

    def test_public_opt_in_creates_ready_source_bound_edge_and_retries_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "private local learning question"
            engine, state = self._engine_and_capture(app, question=question)

            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                first = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-t13b-1",
                    contextual_domain="knowledge",
                )
                second = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-t13b-1",
                    contextual_domain="knowledge",
                )

            learning = first["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(1, len(learning["receipt_ids"]))
            self.assertEqual(learning["receipt_ids"], second["contextual_learning"]["receipt_ids"])
            # Q1 may still create its verified edge without a scope, but it
            # must not mint a cross-conversation exact-revisit contract from a
            # bare question hash.
            self.assertEqual([], learning["revisit_contracts"])
            self.assertEqual((2, 1, 1), self._creation_counts(app))
            with app.db.connection() as connection:
                self.assertEqual(
                    0,
                    int(
                        connection.execute(
                            "SELECT COUNT(*) FROM contextual_revisit_contract"
                        ).fetchone()[0]
                    ),
                )
            self.assertEqual(1, app.context_cue_index.count)
            self.assertEqual(1, app.need_cue_index.count)
            rendered = json.dumps(learning, ensure_ascii=False)
            self.assertNotIn(question, rendered)
            self.assertNotIn("target source proof", rendered)
            self.assertNotIn("t13b/target.json", rendered)

    def test_real_direct_base_selection_can_create_without_a_contextual_target_gate(self):
        """A complete no-edge query may learn only from two direct base routes.

        This exercises the public query path end-to-end: request vectors,
        ordinary rerank, V3 source closure and selector, answer audit, and the
        application finalizer all run locally.  It does not patch
        ``_query_impl`` or manufacture a contextual target-gate result.
        """

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            app.config.retrieval.sparse_enabled = False
            app.config.retrieval.source_key_cohort_enabled = False
            app.config.retrieval.graph_max_hops = 0
            app.config.retrieval.followup_planning_mode = "off"
            app.config.retrieval.rerank_review_mode = "lean"
            app.config.retrieval.rerank_coverage_audit_enabled = False
            app.config.retrieval.rerank_audit_enabled = False
            app.config.retrieval.answer_episode_limit = 1
            episode_ids = self._seed_source_bound_episodes(app)
            self.assertEqual(2, len(episode_ids))
            app.rebuild_indexes()

            requirement = "Which locally recorded source establishes the fact?"
            model = _DirectBaseFixtureModel(
                requirement, (episode_ids[0], episode_ids[1])
            )
            engine = QueryEngine(
                app.config,
                model=model,
                episode_index=app.episode_index,
                concept_index=app.concept_index,
                episodes=app.episodes,
                concepts=app.concepts,
                sources=app.sources,
                associations=app.associations,
                contextual_matcher=ContextualAssociationMatcher(
                    app.context_cue_index,
                    app.need_cue_index,
                    app.associations,
                ),
                contextual_learning_finalizer=app.finalize_recall_event,
            )
            scope_hash = "test-scope:sha256:" + self._hash(
                "real-direct-base-contract-scope"
            )
            query_kwargs = {
                "intent_override": QueryIntent(search_queries=[requirement]),
                "contextual_learning": True,
                "learning_request_id": "real-direct-base-first-creation",
                "contextual_domain": "knowledge",
                "contextual_revisit_scope_hash": scope_hash,
            }
            result = engine.query(
                requirement,
                **query_kwargs,
            )
            replay = engine.query(requirement, **query_kwargs)

            selector = result["evidence_slot_trace"]["slot_selector_v3"]
            self.assertEqual("no_unresolved_slots", selector["reason"])
            self.assertEqual([], result["contextual_association"]["attached_edges"])
            self.assertEqual(
                [], result["contextual_association"]["attached_episode_ids"]
            )
            self.assertEqual(0, result["contextual_association"]["candidate_count"])
            self.assertEqual(_DirectBaseFixtureModel.answer, result["answer"])
            self.assertIs(False, result["answer_generation_skipped"])
            self.assertEqual(
                [
                    "embedding", "rerank", "answer", "answer_audit",
                    "embedding", "rerank", "answer", "answer_audit",
                ],
                model.calls,
            )
            self.assertEqual("ready", result["contextual_learning"]["status"])
            self.assertEqual(1, len(result["contextual_learning"]["receipt_ids"]))
            self.assertEqual(
                ["created"],
                [
                    item["status"]
                    for item in result["contextual_learning"]["revisit_contracts"]
                ],
            )
            self.assertEqual(
                ["idempotent"],
                [
                    item["status"]
                    for item in replay["contextual_learning"]["revisit_contracts"]
                ],
            )
            self.assertEqual(
                result["contextual_learning"]["receipt_ids"],
                replay["contextual_learning"]["receipt_ids"],
            )
            receipt_id = result["contextual_learning"]["receipt_ids"][0]
            self.assertIsNotNone(
                app.associations.load_contextual_revisit_contract(receipt_id)
            )
            with app.db.connection() as connection:
                self.assertEqual(
                    1,
                    int(
                        connection.execute(
                            "SELECT COUNT(*) FROM contextual_revisit_contract"
                        ).fetchone()[0]
                    ),
                )
            rendered = json.dumps(result["contextual_learning"], ensure_ascii=False)
            self.assertNotIn(requirement, rendered)
            self.assertNotIn("anchor source proof", rendered)
            self.assertEqual((2, 1, 1), self._creation_counts(app))

    def test_default_and_shadow_requests_cannot_create_a_learning_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            engine, state = self._engine_and_capture(app, question="default no-write")
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                engine.query("default no-write", contextual_domain="knowledge")
                no_answer = engine.query(
                    "default no-write",
                    generate_answer=False,
                    contextual_learning=True,
                    learning_request_id="request-no-answer",
                    contextual_domain="knowledge",
                )
                frozen = engine.query(
                    "default no-write",
                    contextual_learning=True,
                    learning_request_id="request-frozen",
                    contextual_domain="knowledge",
                    frozen_plan={"legacy": "controlled-test-only"},
                )
            self.assertEqual(
                "answer_generation_required", no_answer["contextual_learning"]["reason"]
            )
            self.assertEqual(
                "frozen_or_strict_replay", frozen["contextual_learning"]["reason"]
            )
            self.assertEqual((0, 0, 0), self._creation_counts(app))

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            engine, state = self._engine_and_capture(
                app, question="shadow no-write", shadow=True
            )
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                result = engine.query(
                    "shadow no-write",
                    contextual_learning=True,
                    learning_request_id="request-shadow",
                    contextual_domain="knowledge",
                )
            self.assertEqual("skipped", result["contextual_learning"]["status"])
            self.assertEqual("shadow_query", result["contextual_learning"]["reason"])
            self.assertEqual((0, 0, 0), self._creation_counts(app))

    def test_missing_answer_request_hash_or_direct_anchor_provenance_is_zero_write(self):
        scenarios = (
            ({"answer": ""}, "answer_delivery_not_confirmed"),
            ({"answer_generation_skipped": None}, "answer_delivery_not_confirmed"),
            (
                {"answer_terminal_state": {"terminal_state": "completed_limited"}},
                "answer_terminal_not_verified",
            ),
            ({"answer_terminal_state": {}}, "answer_terminal_not_verified"),
            ({"bundle_source_request_hash": ""}, "bundle_request_hash_mismatch"),
            ({"independent_base": False}, "no_eligible_source_bound_candidate"),
        )
        for kwargs, expected_reason in scenarios:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                app = self._application(directory)
                engine, state = self._engine_and_capture(
                    app,
                    question="fail closed local boundary",
                    **kwargs,
                )
                with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                    result = engine.query(
                        "fail closed local boundary",
                        contextual_learning=True,
                        learning_request_id="request-fail-closed",
                        contextual_domain="knowledge",
                    )
                self.assertEqual(
                    expected_reason, result["contextual_learning"]["reason"]
                )
                self.assertEqual((0, 0, 0), self._creation_counts(app))

    def test_unvalidated_contextual_or_malformed_source_provenance_is_zero_write(self):
        """The direct-base exception cannot relax either ordinary fail-closed gate."""

        scenarios = (
            (
                {
                    "target_gate_completed": False,
                    "direct_base_validation_completed": False,
                },
                "target_or_direct_base_gate_not_completed",
            ),
            (
                {"invalid_target_provenance": True},
                "no_eligible_source_bound_candidate",
            ),
        )
        for kwargs, expected_reason in scenarios:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                app = self._application(directory)
                engine, state = self._engine_and_capture(
                    app,
                    question="strict source closure boundary",
                    **kwargs,
                )
                with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                    result = engine.query(
                        "strict source closure boundary",
                        contextual_learning=True,
                        learning_request_id="request-gate-or-source-closed",
                        contextual_domain="knowledge",
                    )
                self.assertEqual(
                    expected_reason, result["contextual_learning"]["reason"]
                )
                self.assertEqual((0, 0, 0), self._creation_counts(app))

    def test_multiple_options_use_one_transaction_and_report_deferred_options(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            engine, state = self._engine_and_capture(
                app,
                question="one event transaction boundary",
                extra_targets=1,
            )
            callback_calls: list[object] = []

            def finalize_once(event, plan, materializations):
                callback_calls.append((event, plan, materializations))
                if len(callback_calls) > 1:
                    raise AssertionError("a second per-target transaction was attempted")
                return app.finalize_recall_event(event, plan, materializations)

            engine.contextual_learning_finalizer = finalize_once
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                result = engine.query(
                    "one event transaction boundary",
                    contextual_learning=True,
                    learning_request_id="request-one-event",
                    contextual_domain="knowledge",
                )

            learning = result["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(1, len(callback_calls))
            self.assertEqual(1, len(callback_calls[0][1].candidates))
            self.assertEqual(1, learning["deferred_candidate_options"])
            self.assertEqual((2, 1, 1), self._creation_counts(app))

    def test_publication_failure_stays_pending_and_offline_rebuild_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "pending after local index failure"
            engine, state = self._engine_and_capture(app, question=question)
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]), patch.object(
                app.context_cue_index,
                "upsert",
                side_effect=RuntimeError("injected local index failure"),
            ):
                result = engine.query(
                    question,
                    contextual_learning=True,
                    learning_request_id="request-pending",
                    contextual_domain="knowledge",
                    contextual_revisit_scope_hash=(
                        "test-scope:sha256:" + self._hash("pending-contract-scope")
                    ),
                )
            learning = result["contextual_learning"]
            self.assertEqual("committed_pending_index", learning["status"])
            self.assertEqual(
                ["deferred_not_ready"],
                [item["status"] for item in learning["revisit_contracts"]],
            )
            receipt_id = learning["receipt_ids"][0]
            receipt = app.associations.get_contextual_creation_receipt(receipt_id)
            self.assertEqual("committed_pending_index", receipt["status"])

            restarted = self._application(directory)
            rebuilt = restarted.rebuild_contextual_indexes()
            restored = restarted.associations.get_contextual_creation_receipt(receipt_id)
            self.assertEqual(0, rebuilt["external_calls"])
            self.assertEqual("ready", restored["status"])
            self.assertTrue(restored["ready_for_revisit"])
            self.assertIsNone(
                restarted.associations.load_contextual_revisit_contract(receipt_id)
            )

    def test_different_creation_request_cannot_replace_ready_contract(self):
        """A later Q1 origin may retain its edge receipt, never overwrite v16."""

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "different creation origin"
            engine, state = self._engine_and_capture(app, question=question)
            scope_hash = "test-scope:sha256:" + self._hash("origin-contract-scope")

            def run(request_id: str) -> dict[str, object]:
                with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                    return engine.query(
                        question,
                        contextual_learning=True,
                        learning_request_id=request_id,
                        contextual_domain="knowledge",
                        contextual_revisit_scope_hash=scope_hash,
                    )

            first = run("request-origin-one")
            second = run("request-origin-two")
            first_learning = first["contextual_learning"]
            second_learning = second["contextual_learning"]
            self.assertEqual(["created"], [
                item["status"] for item in first_learning["revisit_contracts"]
            ])
            self.assertEqual("ready", second_learning["status"])
            self.assertEqual(["failed"], [
                item["status"] for item in second_learning["revisit_contracts"]
            ])
            first_receipt_id = first_learning["receipt_ids"][0]
            second_receipt_id = second_learning["receipt_ids"][0]
            self.assertNotEqual(first_receipt_id, second_receipt_id)
            self.assertIsNotNone(
                app.associations.load_contextual_revisit_contract(first_receipt_id)
            )
            self.assertIsNone(
                app.associations.load_contextual_revisit_contract(second_receipt_id)
            )
            with app.db.connection() as connection:
                self.assertEqual(
                    1,
                    int(
                        connection.execute(
                            "SELECT COUNT(*) FROM contextual_revisit_contract"
                        ).fetchone()[0]
                    ),
                )

    def test_contract_database_error_leaves_ready_q1_edge_and_retries_later(self):
        """A post-commit v16 write error cannot relabel or roll back Q1."""

        with tempfile.TemporaryDirectory() as directory:
            app = self._application(directory)
            question = "contract database isolation"
            engine, state = self._engine_and_capture(app, question=question)
            kwargs = {
                "contextual_learning": True,
                "learning_request_id": "request-contract-database-error",
                "contextual_domain": "knowledge",
                "contextual_revisit_scope_hash": (
                    "test-scope:sha256:" + self._hash("contract-db-error-scope")
                ),
            }
            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]), patch.object(
                app.associations,
                "create_contextual_revisit_contract",
                side_effect=sqlite3.IntegrityError("injected contract database failure"),
            ):
                failed = engine.query(question, **kwargs)

            learning = failed["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(
                ["failed"],
                [item["status"] for item in learning["revisit_contracts"]],
            )
            self.assertEqual((2, 1, 1), self._creation_counts(app))
            receipt_id = learning["receipt_ids"][0]
            self.assertEqual(
                "ready",
                app.associations.get_contextual_creation_receipt(receipt_id)["status"],
            )
            self.assertIsNone(app.associations.load_contextual_revisit_contract(receipt_id))

            with patch.object(engine, "_query_impl", side_effect=state["query_impl"]):
                retried = engine.query(question, **kwargs)
            self.assertEqual("ready", retried["contextual_learning"]["status"])
            self.assertEqual(
                ["created"],
                [
                    item["status"]
                    for item in retried["contextual_learning"]["revisit_contracts"]
                ],
            )
            self.assertIsNotNone(
                app.associations.load_contextual_revisit_contract(receipt_id)
            )


if __name__ == "__main__":
    unittest.main()
