from __future__ import annotations

from dataclasses import replace
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
from memory_demo.embeddings.coordinator import EmbeddingCoordinator
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.retrieval.revisit import (
    ExactRevisitInput,
    ExactRevisitSourceMappingProof,
    exact_revisit_context_hash,
    exact_revisit_mapping_contribution_id,
    exact_revisit_mapping_roles_fingerprint,
    exact_revisit_policy_fingerprint,
    exact_revisit_request_hash,
    exact_revisit_source_closure_fingerprint,
    exact_revisit_source_fact_refs_fingerprint,
)
from memory_demo.types import (
    ContextualRevisitContract,
    EvidenceSelectionBudget,
    EvidenceSlot,
    LearningCandidate,
    QueryIntent,
    QueryVectorRequest,
)


def _opaque(namespace: str, value: str) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class _NoProviderModel:
    """A local test model: an exact hit must never invoke it."""

    embedding_provider = "t15-local"
    embedding_revision = "v1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, _texts):
        self.calls.append("embedding")
        raise AssertionError("exact revisit must not embed")

    def chat_json(self, *_args, **_kwargs):
        self.calls.append("chat_json")
        raise AssertionError("exact revisit must not plan or rerank")

    def chat_text(self, *_args, **_kwargs):
        self.calls.append("chat_text")
        raise AssertionError("generate_answer=False must not invoke an answer model")


class ContextualExactRevisitT15Tests(unittest.TestCase):
    """One real local v16 contract and public exact-revisit query path."""

    question = "What does the local target record establish?"
    evaluation_as_of = "2026-09-06T00:00:00+00:00"

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
                )
            )

    @staticmethod
    def _requirements(slot_id: str = "slot-t15") -> RequirementResolution:
        slot = EvidenceSlot(
            slot_id=slot_id,
            question="What does the target source establish?",
            required=True,
            query_id="need-query-t15",
            origin="reused_template",
            support_mode="alternative",
            clause_ids=("clause-t15",),
        )
        return RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=(slot,),
            planner_origin="reused_template",
        )

    @staticmethod
    def _bundle(
        model: _NoProviderModel,
        config: AppConfig,
        requirements: RequirementResolution,
    ):
        slot = requirements.requirements[0]
        coordinator = EmbeddingCoordinator(
            model,
            model_id=config.model.embedding_model,
            dimension=config.model.embedding_dimension,
        )
        return coordinator.bundle_from_precomputed_vectors(
            (
                QueryVectorRequest(
                    role="whole",
                    text=ContextualExactRevisitT15Tests.question,
                    query_id="whole-query-t15",
                ),
                QueryVectorRequest(
                    role="atomic",
                    text=slot.question,
                    query_id=slot.query_id,
                    slot_id=slot.slot_id,
                ),
            ),
            {
                ContextualExactRevisitT15Tests.question: np.asarray(
                    [1.0, 0.0, 0.0], dtype=np.float32
                ),
                slot.question: np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            },
            source_request_hash=exact_revisit_request_hash(
                ContextualExactRevisitT15Tests.question
            ),
        )

    @staticmethod
    def _seed_episodes(
        app: MemoryApplication,
        *,
        marker: str | None = None,
    ) -> tuple[int, int]:
        marker = marker or ContextualExactRevisitT15Tests.evaluation_as_of
        rows = (
            ("anchor", "Anchor record is independently source-bound.", [1.0, 0.0, 0.0]),
            ("target", "Target record establishes the requested local fact.", [0.0, 1.0, 0.0]),
        )
        ids: list[int] = []
        with app.db.transaction() as connection:
            for label, body, vector in rows:
                raw_source = f"[record: {label}]\n{body}"
                source_id = int(
                    connection.execute(
                        "INSERT INTO source(raw_text) VALUES(?)", (raw_source,)
                    ).lastrowid
                )
                episode_id = int(
                    connection.execute(
                        """
                        INSERT INTO episode(
                            source_id, source_key, segment_index, text,
                            evidence_origin, epistemic_status, generation,
                            evidence_quotes_json, evidence_spans_json, evidence_basis,
                            embedding, created_at, updated_at
                        ) VALUES(?, ?, 0, ?, 'source', 'observed', 0, ?, ?,
                                 'literal_source_span', ?, ?, ?)
                        """,
                        (
                            source_id,
                            f"t15/{label}.txt",
                            body,
                            json.dumps([raw_source]),
                            json.dumps([[1, 2]]),
                            np.asarray(vector, dtype=np.float32).tobytes(),
                            marker,
                            marker,
                        ),
                    ).lastrowid
                )
                ids.append(episode_id)
        app.rebuild_indexes()
        return tuple(ids)  # type: ignore[return-value]

    def _fixture(
        self,
        directory: str,
        *,
        create_contract: bool = True,
        evaluation_as_of: str | None = None,
        episode_marker: str | None = None,
    ):
        ticket_evaluation_as_of = evaluation_as_of or self.evaluation_as_of
        config = AppConfig(
            database_path=Path(directory) / "t15-revisit.db",
            log_dir=Path(directory) / "logs",
            model=ModelConfig(
                embedding_model="t15-local-fixture",
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
        app = MemoryApplication(config)
        anchor_id, target_id = self._seed_episodes(
            app,
            marker=episode_marker or ticket_evaluation_as_of,
        )
        model = _NoProviderModel()
        requirements = self._requirements()
        bundle = self._bundle(model, config, requirements)
        matcher = ContextualAssociationMatcher(
            app.context_cue_index,
            app.need_cue_index,
            app.associations,
            context_threshold=config.retrieval.contextual_context_threshold,
            need_threshold=config.retrieval.contextual_need_threshold,
            edge_top_k=1,
            endpoint_limit=1,
            embedding_space_id=bundle.embedding_space_id,
        )
        engine = QueryEngine(
            config,
            model,
            app.episode_index,
            app.concept_index,
            app.episodes,
            app.concepts,
            app.sources,
            app.associations,
            contextual_matcher=matcher,
        )
        facts, reasons = engine._v3_source_fact_closure((anchor_id, target_id))
        self.assertEqual({anchor_id: "source_bound", target_id: "source_bound"}, reasons)
        self.assertEqual({anchor_id, target_id}, set(facts))
        whole = next(item for item in bundle.logical_bindings if item.role == "whole")
        atomic = bundle.bindings_for_slot(requirements.requirements[0].slot_id)[0]
        candidate = LearningCandidate(
            candidate_id=_opaque("candidate", "t15"),
            anchor_type="episode",
            anchor_id=anchor_id,
            target_episode_id=target_id,
            reason="candidate_missing",
            request_id="t15-local-creation",
            request_hash=_opaque("request", "t15-creation"),
            source_request_hash=exact_revisit_request_hash(self.question),
            context_query_id=whole.query_id,
            need_query_id=atomic.query_id,
            slot_id=requirements.requirements[0].slot_id,
            context_vector_ref=whole.physical_id,
            need_vector_ref=atomic.physical_id,
            anchor_vector_ref=whole.physical_id,
            source_facts=(facts[target_id],),
            verification_refs=(_opaque("verification", "t15"),),
            verification_status="source_bound",
            anchor_contribution_id=_opaque("contribution", "t15-anchor"),
            anchor_provenance_refs=(_opaque("anchor", "t15"),),
            target_provenance_refs=(_opaque("target", "t15"),),
        )
        receipt = app.finalize_contextual_association(
            candidate,
            domain="knowledge",
            model_id=config.model.embedding_model,
            dimension=config.model.embedding_dimension,
            context_vector=bundle.vector_for(whole),
            need_vector=bundle.vector_for(atomic),
            context_text_hash=whole.text_hash,
            need_text_hash=atomic.text_hash,
            embedding_space_id=bundle.embedding_space_id,
            creation_request_id="t15-local-creation",
            creation_request_hash=_opaque("request", "t15-creation"),
        )
        self.assertEqual("ready", receipt["status"])
        proof = ExactRevisitSourceMappingProof(
            association_id=int(receipt["association_id"]),
            target_episode_id=target_id,
            slot_id=requirements.requirements[0].slot_id,
            clause_ids=requirements.requirements[0].clause_ids,
            source_fact_id=facts[target_id].fact_id,
            mapping_ref=_opaque("mapping", "t15-target-source-bound"),
        )
        budget = EvidenceSelectionBudget(episode_limit=2)
        ticket = ExactRevisitInput(
            domain="knowledge",
            context_hash=exact_revisit_context_hash(self.question),
            requirements=requirements,
            intent=QueryIntent(search_queries=[requirements.requirements[0].question]),
            query_vector_bundle=bundle,
            budget=budget,
            retrieval_policy_fingerprint=exact_revisit_policy_fingerprint(
                config, matcher, endpoint_limit=1
            ),
            source_closure_fingerprint=exact_revisit_source_closure_fingerprint(
                tuple(facts.values())
            ),
            source_fact_roles_fingerprint=exact_revisit_mapping_roles_fingerprint(
                facts,
                base_slot_support={},
                target_source_mapping_proofs=(proof,),
            ),
            source_fact_refs_fingerprint=exact_revisit_source_fact_refs_fingerprint(
                tuple(facts.values())
            ),
            evaluation_as_of=ticket_evaluation_as_of,
            endpoint_limit=1,
            anchor_activations={anchor_id: 1.0},
            base_episode_ids=(anchor_id,),
            target_source_mapping_proofs=(proof,),
        )
        receipt_row = app.associations.get_contextual_creation_receipt(
            int(receipt["receipt_id"])
        )
        self.assertIsNotNone(receipt_row)
        contract = ContextualRevisitContract(
            creation_receipt_id=int(receipt["receipt_id"]),
            association_id=int(receipt["association_id"]),
            context_cue_id=int(receipt["context_cue_id"]),
            need_cue_id=int(receipt["need_cue_id"]),
            domain="knowledge",
            model_id=config.model.embedding_model,
            embedding_space_id=bundle.embedding_space_id,
            dimension=3,
            dtype="float32",
            context_hash=ticket.context_hash,
            slot_need_bindings=ticket.slot_need_bindings,
            requirements_fingerprint=ticket.requirements_fingerprint,
            source_closure_fingerprint=ticket.source_closure_fingerprint,
            retrieval_policy_fingerprint=ticket.retrieval_policy_fingerprint,
            budget_fingerprint=ticket.budget_fingerprint,
            anchor_manifest_fingerprint=ticket.anchor_manifest_fingerprint,
            source_fact_roles_fingerprint=ticket.source_fact_roles_fingerprint,
            source_fact_refs_fingerprint=ticket.source_fact_refs_fingerprint,
            ready_index_epoch=int(receipt["ready_index_epoch"]),
            ready_publication_fingerprint=(
                app.associations.contextual_revisit_ready_publication_fingerprint(
                    receipt_row
                )
            ),
        )
        if create_contract:
            app.associations.create_contextual_revisit_contract(contract)
        return app, model, engine, ticket, anchor_id, target_id, int(receipt["association_id"])

    def test_public_exact_revisit_reruns_gate_closure_and_selector_without_provider_work(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, anchor_id, target_id, association_id = self._fixture(directory)
            before = self._counts(app)

            result = engine.query(
                self.question,
                generate_answer=False,
                contextual_domain="knowledge",
                exact_revisit_input=ticket,
            )

            self.assertEqual([], model.calls)
            self.assertEqual(before, self._counts(app))
            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertTrue(result["exact_revisit"]["planner_skipped"])
            self.assertTrue(result["exact_revisit"]["reranker_skipped"])
            self.assertTrue(result["exact_revisit"]["embedding_skipped"])
            self.assertFalse(result["exact_revisit"]["answer_cache_used"])
            self.assertEqual(
                {
                    "planner": False,
                    "embedding": False,
                    "reranker": False,
                    "contextual_matcher": True,
                    "target_gate": True,
                    "source_closure": True,
                    "contribution_selector": True,
                    "answer_generation": False,
                },
                result["exact_revisit"]["executed_modules"],
            )
            self.assertEqual("", result["answer"])
            self.assertTrue(result["answer_generation_skipped"])
            self.assertIn(target_id, result["episode_ids"])
            selector = result["evidence_slot_trace"]["slot_selector_v3"]
            self.assertEqual(
                result["episode_ids"], selector["treatment_selector"]["selected_episode_ids"]
            )
            self.assertEqual(
                "target_checked_before_endpoint_cap_v1",
                selector["target_gate"]["stage"],
            )
            self.assertGreater(
                selector["target_gate"]["accepted_proposal_count"], 0
            )
            self.assertEqual([association_id], result["association_ids"])
            merged = result["contextual_association"]["merged_candidates"]
            lanes = {row["lane"] for row in merged}
            self.assertIn("contextual", lanes)
            self.assertIn("contextual_source_mapping", lanes)
            self.assertGreater(result["contextual_association"]["context_prototype_hits"], 0)
            self.assertGreater(result["contextual_association"]["need_prototype_hits"], 0)
            ranking_only = set(
                result["contextual_association"]["ranking_only_contextual_contribution_ids"]
            )
            selected = set(
                result["contextual_association"]["selected_contextual_contribution_ids"]
            )
            self.assertTrue(ranking_only)
            self.assertTrue(selected.difference(ranking_only))
            self.assertIn(anchor_id, ticket.base_episode_ids)

    def _assert_public_falls_through(self, engine: QueryEngine, ticket: ExactRevisitInput, **kwargs) -> None:
        sentinel = {"fallback": True}
        with patch.object(engine, "_query_impl", return_value=sentinel) as fallback:
            result = engine.query(
                self.question,
                generate_answer=False,
                exact_revisit_input=ticket,
                **kwargs,
            )
        self.assertIs(sentinel, result)
        fallback.assert_called_once()

    def _assert_delivery_mutation_falls_through(
        self,
        app: MemoryApplication,
        model: _NoProviderModel,
        engine: QueryEngine,
        ticket: ExactRevisitInput,
        mutate_after_first_guard,
    ) -> None:
        """Change durable state only between the two exact local passes."""

        before = self._counts(app)
        original_guard = engine._exact_revisit_delivery_guard
        guard_calls = 0

        def guard_then_mutate(*args, **kwargs):
            nonlocal guard_calls
            guard = original_guard(*args, **kwargs)
            guard_calls += 1
            if guard_calls == 1:
                self.assertIsNotNone(guard)
                mutate_after_first_guard()
            return guard

        with patch.object(
            engine,
            "_exact_revisit_delivery_guard",
            side_effect=guard_then_mutate,
        ):
            self._assert_public_falls_through(engine, ticket)
        self.assertGreaterEqual(guard_calls, 1)
        self.assertEqual(before, self._counts(app))
        self.assertEqual([], model.calls)

    def test_exact_revisit_delivery_revalidation_falls_through_on_source_edge_or_publication_change(self):
        """No mixed SQLite/RAM view may reach exact delivery after a mutation."""

        with self.subTest(change="source"):
            with tempfile.TemporaryDirectory() as directory:
                app, model, engine, ticket, _anchor_id, target_id, _association_id = (
                    self._fixture(directory)
                )

                def mutate_source():
                    with app.db.transaction() as connection:
                        source_id = int(
                            connection.execute(
                                "SELECT source_id FROM episode WHERE id = ?",
                                (target_id,),
                            ).fetchone()[0]
                        )
                        connection.execute(
                            "UPDATE source SET raw_text = ? WHERE id = ?",
                            (
                                "[record: target]\nA revised source invalidates the old proof.",
                                source_id,
                            ),
                        )

                self._assert_delivery_mutation_falls_through(
                    app, model, engine, ticket, mutate_source
                )

        with self.subTest(change="edge"):
            with tempfile.TemporaryDirectory() as directory:
                app, model, engine, ticket, _anchor_id, _target_id, association_id = (
                    self._fixture(directory)
                )
                self._assert_delivery_mutation_falls_through(
                    app,
                    model,
                    engine,
                    ticket,
                    lambda: app.associations.retire([association_id], reason="t15-test"),
                )

        with self.subTest(change="publication"):
            with tempfile.TemporaryDirectory() as directory:
                app, model, engine, ticket, _anchor_id, _target_id, _association_id = (
                    self._fixture(directory)
                )

                def mutate_publication():
                    app.associations.reconcile_contextual_receipts_after_index_rebuild(
                        context_cue_ids=app.context_cue_index.id_to_index,
                        need_cue_ids=app.need_cue_index.id_to_index,
                        context_cue_count=app.context_cue_index.count,
                        need_cue_count=app.need_cue_index.count,
                        embedding_space_id=ticket.query_vector_bundle.embedding_space_id,
                    )

                self._assert_delivery_mutation_falls_through(
                    app, model, engine, ticket, mutate_publication
                )

    def test_exact_revisit_delivery_revalidation_falls_through_on_contract_view_change(self):
        """A changed contract returned on the final pass cannot be delivered."""

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, _target_id, _association_id = (
                self._fixture(directory)
            )
            original_find = app.associations.find_contextual_revisit_contract
            use_changed_contract = False

            def changed_contract(lookup):
                contract = original_find(lookup)
                if use_changed_contract and contract is not None:
                    return replace(
                        contract,
                        ready_publication_fingerprint=_opaque(
                            "publication", "changed-contract-view"
                        ),
                    )
                return contract

            def enable_changed_contract():
                nonlocal use_changed_contract
                use_changed_contract = True

            with patch.object(
                app.associations,
                "find_contextual_revisit_contract",
                side_effect=changed_contract,
            ):
                self._assert_delivery_mutation_falls_through(
                    app,
                    model,
                    engine,
                    ticket,
                    enable_changed_contract,
                )

    def test_missing_contract_or_scope_mismatch_falls_through_without_writes(self):
        """A complete transient ticket never bypasses lookup/scope guards."""

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, _target_id, _association_id = self._fixture(
                directory,
                create_contract=False,
            )
            before = self._counts(app)
            self._assert_public_falls_through(engine, ticket)
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, _target_id, _association_id = self._fixture(directory)
            before = self._counts(app)
            self._assert_public_falls_through(
                engine,
                ticket,
                contextual_revisit_scope_hash=_opaque("scope", "other-conversation"),
            )
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)

    def test_exact_revisit_metadata_and_mask_mismatches_fall_through_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, target_id, association_id = self._fixture(directory)
            before = self._counts(app)
            alternate_requirements = self._requirements("slot-t15-other")
            alternate_bundle = self._bundle(model, engine.config, alternate_requirements)
            alternate_proof = replace(
                ticket.target_source_mapping_proofs[0],
                slot_id="slot-t15-other",
            )
            alternate_slot_ticket = replace(
                ticket,
                requirements=alternate_requirements,
                query_vector_bundle=alternate_bundle,
                target_source_mapping_proofs=(alternate_proof,),
            )
            alternate_space_model = _NoProviderModel()
            alternate_space_model.embedding_revision = "v2"
            alternate_space_bundle = self._bundle(
                alternate_space_model, engine.config, ticket.requirements
            )
            cases = (
                replace(ticket, domain="other-domain"),
                replace(ticket, context_hash=_opaque("revisit-context", "wrong")),
                alternate_slot_ticket,
                replace(ticket, query_vector_bundle=alternate_space_bundle),
                replace(ticket, source_closure_fingerprint=_opaque("source", "stale")),
                replace(ticket, masked_edge_ids=(association_id,)),
                replace(
                    ticket,
                    masked_contribution_ids=(
                        exact_revisit_mapping_contribution_id(
                            ticket.target_source_mapping_proofs[0]
                        ),
                    ),
                ),
            )
            for mismatched in cases:
                with self.subTest(ticket=mismatched.domain, mask=mismatched.masked_edge_ids):
                    self._assert_public_falls_through(engine, mismatched)
                    self.assertEqual(before, self._counts(app))
            # This is a current-closure miss rather than merely a caller-side
            # fingerprint typo: the stored quote/span is now invalid for the
            # target's revised full source, so the local source check must
            # prevent the exact route from delivering the old fact mapping.
            with app.db.transaction() as connection:
                source_id = int(
                    connection.execute(
                        "SELECT source_id FROM episode WHERE id = ?", (target_id,)
                    ).fetchone()[0]
                )
                connection.execute(
                    "UPDATE source SET raw_text = ? WHERE id = ?",
                    ("[record: target]\nA revised target source no longer matches.", source_id),
                )
            self._assert_public_falls_through(engine, ticket)
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)

    def test_old_ticket_time_cannot_reactivate_expired_edge_or_enable_replay_shortcut(self):
        """Exact revisit rechecks lifecycle/source state at the live request time."""

        with tempfile.TemporaryDirectory() as directory:
            ticket_time = "2020-01-01T00:00:00+00:00"
            app, model, engine, ticket, _anchor_id, _target_id, association_id = (
                self._fixture(
                    directory,
                    evaluation_as_of=ticket_time,
                    episode_marker="2019-12-31T00:00:00+00:00",
                )
            )
            before = self._counts(app)

            # Supplying a public as-of value is an explicit replay request.
            # It remains available to the normal pipeline, but must never
            # persuade exact revisit to use a historical database view.
            self._assert_public_falls_through(
                engine,
                ticket,
                contextual_evaluation_as_of=ticket.evaluation_as_of,
            )
            self.assertEqual(before, self._counts(app))

            # At the ticket's old instant this edge was valid.  This makes the
            # test discriminate against the former behavior, which passed the
            # ticket timestamp directly into the matcher/target gate.
            historical = engine.contextual_recall(
                ticket.query_vector_bundle,
                domain=ticket.domain,
                endpoint_limit=ticket.endpoint_limit,
                active_anchor_ids=ticket.anchor_activation_map,
                unresolved_slot_ids=[
                    ticket.requirements.requirements[0].slot_id
                ],
                evaluation_as_of=ticket.evaluation_as_of,
            )
            self.assertGreater(len(historical["pre_target_proposals"]), 0)

            # The real clock is long after this expiry.  The durable contract
            # still exists, so only a live lifecycle recheck can reject it.
            with app.db.transaction() as connection:
                connection.execute(
                    "UPDATE association SET expires_at = ? WHERE id = ?",
                    ("2020-01-02T00:00:00+00:00", association_id),
                )

            self._assert_public_falls_through(engine, ticket)
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)

    def test_exact_revisit_respects_deadline_and_reports_it_without_using_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, _target_id, _association_id = (
                self._fixture(directory)
            )
            before = self._counts(app)

            success = engine.query(
                self.question,
                generate_answer=False,
                contextual_domain="knowledge",
                exact_revisit_input=ticket,
                deadline_seconds=5.0,
            )
            self.assertEqual("hit", success["exact_revisit"]["status"])
            self.assertEqual(5.0, success["timings"]["deadline_seconds"])

            # This keeps the real contract/matcher/target-gate route intact,
            # but makes one critical local stage exceed a deliberately tiny
            # request budget.  Exact revisit must raise the same TimeoutError
            # shape as normal retrieval rather than return a stale hit or
            # restart the timer in the fallback path.
            original_match = engine.contextual_recall

            def delayed_match(*args, **kwargs):
                time.sleep(0.04)
                return original_match(*args, **kwargs)

            with patch.object(engine, "contextual_recall", side_effect=delayed_match):
                with self.assertRaisesRegex(
                    TimeoutError,
                    r"^query deadline exceeded before exact_revisit",
                ):
                    engine.query(
                        self.question,
                        generate_answer=False,
                        contextual_domain="knowledge",
                        exact_revisit_input=ticket,
                        deadline_seconds=0.005,
                    )

            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)

    def test_public_exact_revisit_threads_the_existing_deadline_to_answer_generation(self):
        """A successful exact hit must not lose or restart its request deadline."""

        with tempfile.TemporaryDirectory() as directory:
            _app, model, engine, ticket, _anchor_id, target_id, _association_id = (
                self._fixture(directory)
            )
            with patch.object(
                engine,
                "_generate_audited_answer",
                return_value=("local generated answer", [], 0),
            ) as generate:
                result = engine.query(
                    self.question,
                    generate_answer=True,
                    contextual_domain="knowledge",
                    exact_revisit_input=ticket,
                    deadline_seconds=5.0,
                )

            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertEqual("local generated answer", result["answer"])
            self.assertIn(target_id, result["episode_ids"])
            generate.assert_called_once()
            deadline_at = generate.call_args.kwargs.get("deadline_at")
            self.assertIsInstance(deadline_at, float)
            self.assertGreater(deadline_at, 0.0)
            self.assertEqual([], model.calls)

    def test_public_evidence_only_exact_revisit_returns_materialized_source_without_writes(self):
        """The public evidence boundary exposes actual Source, never an answer."""

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine, ticket, _anchor_id, target_id, _association_id = (
                self._fixture(directory)
            )
            before = self._counts(app)
            result = engine.query(
                self.question,
                contextual_domain="knowledge",
                exact_revisit_input=ticket,
                stop_after="evidence",
                deadline_seconds=5.0,
            )

            self.assertEqual("hit", result["exact_revisit"]["status"])
            self.assertEqual("live_evidence", result["execution_profile"])
            self.assertEqual("", result["answer"])
            self.assertFalse(result["association_usage_recorded"])
            evidence = result["evidence_result"]
            self.assertEqual("complete", evidence["evidence_state"])
            self.assertTrue(evidence["delivery_states"]["evidence_materialized"])
            self.assertFalse(evidence["delivery_states"]["answer_input_sent"])
            self.assertFalse(evidence["learning_eligible"])
            self.assertEqual("contributed", evidence["participation"]["edge_state"])
            excerpts = evidence["materialized_source_refs"]
            self.assertTrue(
                any(
                    item["episode_id"] == target_id
                    and "Target record establishes the requested local fact."
                    in item["source_excerpt"]
                    for item in excerpts
                )
            )
            self.assertEqual(before, self._counts(app))
            self.assertEqual([], model.calls)
