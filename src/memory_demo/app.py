from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np

from memory_demo.config import AppConfig
from memory_demo.database import Database
from memory_demo.embeddings import (
    EmbeddingIndex,
    decode_embedding,
    encode_embedding,
)
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.llm import ModelClient
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine, SQLiteSparseIndex
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.revisit import (
    ContextualRevisitContractDraft,
    RestrictedRewriteCommitmentKey,
    restricted_rewrite_manifest_binding_signer,
    restricted_rewrite_ready_manifest_signer,
)
from memory_demo.types import (
    ContextualRecallCandidate,
    ContextualRevisitContract,
    ContextualRevisitRuntimeSeed,
    ContextualRestrictedRewriteGuardDraft,
    LearningCandidatePlan,
    RecallLearningEvent,
)
from memory_demo.retrieval.cue_index import association_cue_text


class MemoryApplication:
    def __init__(self, config: AppConfig):
        self.config = config
        config.ensure_directories()
        self.db = Database(config.database_path)
        self.db.initialize()
        self.sources = SourceRepository(self.db)
        self.episodes = EpisodeRepository(self.db)
        self.concepts = ConceptRepository(self.db)
        self.paragraphs = ParagraphRepository(self.db)
        self.associations = AssociationRepository(
            self.db,
            config.weights,
            contextual_noop_decay=config.retrieval.contextual_noop_decay,
            contextual_harm_multiplier=config.retrieval.contextual_harm_multiplier,
            contextual_min_distinct_successes=(
                config.retrieval.contextual_min_distinct_successes
            ),
            contextual_promotion_enabled=(
                config.retrieval.contextual_promotion_enabled
            ),
        )
        self.episode_index = EmbeddingIndex(config.model.embedding_dimension)
        self.concept_index = EmbeddingIndex(config.model.embedding_dimension)
        self.paragraph_index = EmbeddingIndex(config.model.embedding_dimension)
        self.association_index = EmbeddingIndex(config.model.embedding_dimension)
        self.context_cue_index = EmbeddingIndex(config.model.embedding_dimension)
        self.need_cue_index = EmbeddingIndex(config.model.embedding_dimension)
        self.episode_sparse_index = SQLiteSparseIndex(self.db, "episode")
        self.source_sparse_index = SQLiteSparseIndex(self.db, "source")

    def new_logger(self, operation: str) -> JsonlEventLogger:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        unique = uuid4().hex[:8]
        return JsonlEventLogger(
            self.config.log_dir / f"{operation}-{timestamp}-{unique}.jsonl",
            answer_evidence_enabled=bool(
                self.config.retrieval.answer_evidence_checkpoint_enabled
            ),
        )

    def _restricted_rewrite_ready_manifest_signer(self):
        """Return a process-only V22 publication authority when configured.

        The closure is intentionally created at the trusted application
        publication boundary, handed straight to the repository transaction,
        and never retained in a receipt, index, trace, or result.
        """

        if not bool(
            getattr(
                self.config.retrieval,
                "contextual_restricted_rewrite_enabled",
                False,
            )
        ):
            return None
        return restricted_rewrite_ready_manifest_signer(
            RestrictedRewriteCommitmentKey.from_environment()
        )

    def _restricted_rewrite_manifest_binding_signer(self):
        """Create the Q1 signer only inside the trusted application finalizer.

        QueryEngine deliberately passes the redacted draft, never a closure
        over the commitment key, through its generic finalizer callback.  This
        application-owned boundary derives the short-lived signer immediately
        before handing it to the repository's Q1 transaction.
        """

        if not bool(
            getattr(
                self.config.retrieval,
                "contextual_restricted_rewrite_enabled",
                False,
            )
        ):
            return None
        return restricted_rewrite_manifest_binding_signer(
            RestrictedRewriteCommitmentKey.from_environment()
        )

    def rebuild_indexes(self) -> None:
        self.episode_index.rebuild(
            self.episodes.count(), self.episodes.iter_embeddings()
        )
        self.concept_index.rebuild(
            self.concepts.count(), self.concepts.iter_embeddings()
        )
        self.paragraph_index.rebuild(
            self.paragraphs.count(), self.paragraphs.iter_embeddings()
        )
        self.rebuild_contextual_indexes()

    def rebuild_contextual_indexes(self) -> dict[str, int | str]:
        """Restore persisted context/need prototypes without network access."""
        prototypes = self.associations.list_cue_prototypes()
        dimension = self.config.model.embedding_dimension
        model_id = str(self.config.model.embedding_model)
        default_space = self.associations._default_embedding_space_id(
            model_id, dimension
        )
        publication = self.associations.get_contextual_index_publication()
        published_space = str(publication["embedding_space_id"] or "").strip()
        compatible_rows: list[tuple[object, str]] = []
        skipped = 0
        for row in prototypes:
            if (
                str(row["model_id"]) != model_id
                or int(row["dimension"]) != dimension
                or str(row["dtype"]) != "float32"
            ):
                skipped += 1
                continue
            space_id = str(row["embedding_space_id"] or "").strip() or default_space
            compatible_rows.append((row, space_id))
        known_spaces = {space_id for _row, space_id in compatible_rows}
        if published_space:
            selected_space = published_space
        elif default_space in known_spaces:
            selected_space = default_space
        elif len(known_spaces) == 1:
            selected_space = next(iter(known_spaces))
        else:
            # One flat RAM index cannot safely blend two full vector spaces.
            # Leave ambiguous receipts pending instead of guessing.
            selected_space = ""
        selected_rows = [
            row for row, space_id in compatible_rows if space_id == selected_space
        ]
        skipped += len(compatible_rows) - len(selected_rows)
        context_rows = [
            row for row in selected_rows if str(row["cue_kind"]) == "context"
        ]
        need_rows = [
            row for row in selected_rows if str(row["cue_kind"]) == "need"
        ]
        self.context_cue_index = EmbeddingIndex(
            dimension,
            initial_capacity=max(16, len(context_rows) + 1),
        )
        self.need_cue_index = EmbeddingIndex(
            dimension,
            initial_capacity=max(16, len(need_rows) + 1),
        )
        for row in selected_rows:
            try:
                vector = decode_embedding(row["vector_blob"], dimension)
                target = (
                    self.context_cue_index
                    if str(row["cue_kind"]) == "context"
                    else self.need_cue_index
                )
                target.add(int(row["id"]), vector)
            except (TypeError, ValueError):
                skipped += 1
        ready_manifest_signer = self._restricted_rewrite_ready_manifest_signer()
        reconciliation = self.associations.reconcile_contextual_receipts_after_index_rebuild(
            context_cue_ids=self.context_cue_index.id_to_index,
            need_cue_ids=self.need_cue_index.id_to_index,
            context_cue_count=self.context_cue_index.count,
            need_cue_count=self.need_cue_index.count,
            embedding_space_id=selected_space,
            restricted_rewrite_ready_manifest_signer=ready_manifest_signer,
        )
        # A receipt can already be ready when a prior publisher was interrupted
        # after its index transition but before the V17 manifest promotion.  The
        # repository reconstructs and verifies that immutable state itself; the
        # application deliberately supplies no contract-like object here.
        runtime_reconciliation = (
            self.associations.reconcile_contextual_revisit_runtime_manifests()
        )
        result = {
            "context_prototypes": self.context_cue_index.count,
            "need_prototypes": self.need_cue_index.count,
            "skipped": skipped,
            "external_calls": 0,
            "embedding_space_id": selected_space,
            "reconciled_receipts": int(reconciliation["reconciled"]),
            "reconciled_runtime_manifests": int(
                reconciliation.get("runtime_manifest_promoted", 0)
            )
            + int(runtime_reconciliation["promoted"]),
            "rejected_runtime_manifests": int(
                reconciliation.get("runtime_manifest_rejected", 0)
            )
            + int(runtime_reconciliation["rejected"]),
        }
        return result

    def finalize_contextual_association(
        self,
        candidate: ContextualRecallCandidate,
        *,
        domain: str,
        model_id: str,
        dimension: int | None = None,
        context_vector,
        need_vector,
        context_text_hash: str,
        need_text_hash: str,
        context_display_text: str = "",
        need_display_text: str = "",
        embedding_space_id: str = "",
        creation_request_id: str = "",
        creation_request_hash: str = "",
    ) -> dict[str, object]:
        """Atomically commit one creation, then locally publish its RAM cues."""

        if str(model_id) != str(self.config.model.embedding_model):
            raise ValueError("contextual cue model does not match application model")
        if (
            dimension is not None
            and int(dimension) != int(self.config.model.embedding_dimension)
        ):
            raise ValueError("contextual cue dimension does not match application model")
        receipt = self.associations.finalize_contextual_creation(
            candidate,
            domain=domain,
            model_id=model_id,
            dimension=self.config.model.embedding_dimension,
            context_vector=context_vector,
            need_vector=need_vector,
            context_text_hash=context_text_hash,
            need_text_hash=need_text_hash,
            context_display_text=context_display_text,
            need_display_text=need_display_text,
            embedding_space_id=embedding_space_id,
            creation_request_id=creation_request_id,
            creation_request_hash=creation_request_hash,
            utility_weight=0.20,
            probation_ttl=self.config.retrieval.contextual_probation_ttl,
        )
        if str(receipt["status"]) == "legacy_pending_verification":
            # The legacy candidate shape has no source/verification closure.
            # It remains durable for diagnosis but is never indexed/revisit-ready.
            return receipt
        publication = self.associations.get_contextual_index_publication()
        published_space = str(publication["embedding_space_id"] or "").strip()
        receipt_space = str(receipt["embedding_space_id"] or "").strip()
        if published_space and published_space != receipt_space:
            raise ValueError("contextual index is published for another embedding space")
        # Database is authoritative. A crash after either local upsert leaves
        # this receipt pending; offline rebuild can restore both vectors and
        # perform the exact guarded readiness transition without a model call.
        self.context_cue_index.upsert(int(receipt["context_cue_id"]), context_vector)
        self.need_cue_index.upsert(int(receipt["need_cue_id"]), need_vector)
        if str(receipt["status"]) == "committed_pending_index":
            return self.associations.mark_contextual_receipt_ready(
                int(receipt["receipt_id"]),
                context_cue_count=self.context_cue_index.count,
                need_cue_count=self.need_cue_index.count,
                expected_context_cue_id=int(receipt["context_cue_id"]),
                expected_need_cue_id=int(receipt["need_cue_id"]),
            )
        return receipt

    def create_contextual_association(
        self,
        candidate: ContextualRecallCandidate,
        **kwargs,
    ) -> int:
        """Compatibility entry point returning the historical edge ID."""

        return int(self.finalize_contextual_association(candidate, **kwargs)["association_id"])

    @staticmethod
    def _revisit_contract_status(
        draft: ContextualRevisitContractDraft,
        receipt_id: int | None,
        status: str,
    ) -> dict[str, object]:
        """Return a deliberately redacted post-publication contract outcome."""

        return {
            "candidate_id": str(draft.creation_request_id),
            "receipt_id": int(receipt_id) if receipt_id and receipt_id > 0 else None,
            "status": str(status),
        }

    def _finalize_revisit_contract_after_publication(
        self,
        receipt: dict[str, object],
        materialization: dict[str, object],
    ) -> dict[str, object] | None:
        """Write one v16 contract only for a fresh canonical ready receipt.

        This occurs after the Q1 transaction and RAM cue publication.  The
        typed draft carries all selector/source semantics from QueryEngine;
        this facade only verifies it binds exactly to the immutable receipt
        and association before it calls the existing repository contract API.
        A failure is isolated from the already durable edge/receipt.
        """

        draft = materialization.get("revisit_contract_draft")
        if draft is None:
            return None
        if not isinstance(draft, ContextualRevisitContractDraft):
            return {
                "candidate_id": "",
                "receipt_id": None,
                "status": "skipped_invalid_draft",
            }
        try:
            receipt_id = int(receipt.get("receipt_id", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return self._revisit_contract_status(draft, None, "skipped_invalid_receipt")
        if receipt_id <= 0:
            return self._revisit_contract_status(draft, None, "skipped_invalid_receipt")
        if str(receipt.get("status", "") or "") != "ready":
            return self._revisit_contract_status(draft, receipt_id, "deferred_not_ready")
        try:
            # Do not trust the mutable in-memory publication loop value.  The
            # repository row is the canonical source of every receipt-bound
            # field used below.
            canonical = self.associations.get_contextual_creation_receipt(receipt_id)
            if canonical is None:
                return self._revisit_contract_status(
                    draft, receipt_id, "skipped_missing_receipt"
                )
            if (
                str(canonical.get("status", "") or "") != "ready"
                or str(canonical.get("verification_status", "") or "")
                not in {"verified", "source_bound"}
                or str(canonical.get("creation_request_id", "") or "")
                != str(draft.creation_request_id)
                or str(canonical.get("source_request_hash", "") or "")
                != str(draft.source_request_hash)
                or str(canonical.get("domain", "") or "") != str(draft.domain)
            ):
                return self._revisit_contract_status(
                    draft, receipt_id, "skipped_receipt_mismatch"
                )
            association_id = int(canonical.get("association_id", 0) or 0)
            association = self.associations.get(association_id)
            if association is None or (
                str(association["from_type"] or "") != "episode"
                or int(association["from_id"] or 0) != int(draft.anchor_episode_id)
                or str(association["to_type"] or "") != "episode"
                or int(association["to_id"] or 0)
                != int(draft.target_mapping.target_episode_id)
                or str(association["association_mode"] or "")
                != "contextual_recall"
            ):
                return self._revisit_contract_status(
                    draft, receipt_id, "skipped_association_mismatch"
                )
            contract = ContextualRevisitContract(
                creation_receipt_id=receipt_id,
                association_id=association_id,
                context_cue_id=int(canonical["context_cue_id"]),
                need_cue_id=int(canonical["need_cue_id"]),
                domain=str(canonical["domain"]),
                model_id=str(canonical["model_id"]),
                embedding_space_id=str(canonical["embedding_space_id"]),
                dimension=int(canonical["dimension"]),
                dtype=str(canonical["dtype"]),
                context_hash=draft.context_hash,
                slot_need_bindings=draft.slot_need_bindings,
                requirements_fingerprint=draft.requirements_fingerprint,
                source_closure_fingerprint=draft.source_closure_fingerprint,
                retrieval_policy_fingerprint=draft.retrieval_policy_fingerprint,
                budget_fingerprint=draft.budget_fingerprint,
                anchor_manifest_fingerprint=draft.anchor_manifest_fingerprint,
                source_fact_roles_fingerprint=draft.source_fact_roles_fingerprint(
                    association_id
                ),
                source_fact_refs_fingerprint=draft.source_fact_refs_fingerprint,
                ready_index_epoch=int(canonical["ready_index_epoch"]),
                ready_publication_fingerprint=(
                    self.associations.contextual_revisit_ready_publication_fingerprint(
                        canonical
                    )
                ),
            )
            contract_receipt = self.associations.create_contextual_revisit_contract(
                contract
            )
            return self._revisit_contract_status(
                draft,
                receipt_id,
                "idempotent" if contract_receipt.idempotent else "created",
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            # Edge/cue/receipt persistence has already completed.  The next
            # identical Q1 finalization may safely retry only this immutable
            # contract write; never roll the successful edge back here.
            return self._revisit_contract_status(draft, receipt_id, "failed")

    @staticmethod
    def _runtime_manifest_status(
        seed: ContextualRevisitRuntimeSeed,
        receipt_id: int | None,
        status: str,
    ) -> dict[str, object]:
        """Return a redacted V17 publication outcome.

        Runtime seeds deliberately contain only opaque IDs and fingerprints.
        Keeping this public status shape aligned with the older contract
        outcome avoids surfacing any recovery material in an application
        response.
        """

        return {
            "candidate_id": str(seed.creation_request_id),
            "receipt_id": int(receipt_id) if receipt_id and receipt_id > 0 else None,
            "status": str(status),
        }

    def _finalize_runtime_manifest_after_publication(
        self,
        receipt: dict[str, object],
        materialization: dict[str, object],
        *,
        newly_published: bool,
    ) -> dict[str, object] | None:
        """Promote a V17 seed through the receipt-only repository boundary.

        The seed was persisted atomically with the cue/edge/receipt by the
        repository.  Once RAM publication marks that receipt ready, this call
        is an idempotent final verification only.  In particular, no caller
        may pass a V16 contract or re-create one from mutable materialization
        data.
        """

        seed = materialization.get("revisit_runtime_seed")
        if seed is None:
            return None
        if not isinstance(seed, ContextualRevisitRuntimeSeed):
            return {
                "candidate_id": "",
                "receipt_id": None,
                "status": "skipped_invalid_runtime_seed",
            }
        try:
            receipt_id = int(receipt.get("receipt_id", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return self._runtime_manifest_status(
                seed, None, "skipped_invalid_receipt"
            )
        if receipt_id <= 0:
            return self._runtime_manifest_status(
                seed, None, "skipped_invalid_receipt"
            )
        if str(receipt.get("status", "") or "") != "ready":
            return self._runtime_manifest_status(
                seed, receipt_id, "deferred_not_ready"
            )
        try:
            promoted = self.associations.promote_contextual_revisit_runtime_manifest(
                receipt_id
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            # The receipt stays durable and ready.  A later rebuild or the
            # same finalization may safely retry the repository-owned check.
            return self._runtime_manifest_status(seed, receipt_id, "failed")
        if promoted is None:
            return self._runtime_manifest_status(
                seed, receipt_id, "skipped_missing_runtime_seed"
            )
        if promoted.state != "ready":
            return self._runtime_manifest_status(
                seed, receipt_id, "deferred_not_ready"
            )
        # `mark_contextual_receipt_ready` promotes a fresh V17 seed in the
        # same database transaction.  The second receipt-only call above is
        # normally idempotent; report the original first publication as a
        # creation so the existing result contract remains useful to callers.
        if newly_published:
            status = "created"
        else:
            status = "idempotent" if promoted.idempotent else "created"
        return self._runtime_manifest_status(seed, receipt_id, status)

    def finalize_recall_event(
        self,
        event: RecallLearningEvent,
        plan: LearningCandidatePlan,
        cue_materializations: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        """Finalize one bounded V3 learning plan, then publish its RAM cues.

        The repository owns the all-or-nothing cue/edge/receipt transaction.
        Publication deliberately happens only afterwards: a local index error
        leaves the durable receipt ``committed_pending_index`` so a no-network
        ``rebuild_contextual_indexes`` can recover it on the next process.
        """

        candidates = tuple(plan.candidates)
        prepared: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            candidate_id = str(candidate.candidate_id).strip()
            materialization = cue_materializations.get(candidate_id)
            if not isinstance(materialization, dict):
                raise ValueError("candidate cue materialization is required")
            if str(materialization.get("model_id", "")) != str(
                self.config.model.embedding_model
            ):
                raise ValueError("contextual cue model does not match application model")
            try:
                dimension = int(materialization.get("dimension", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("contextual cue dimension is invalid") from exc
            if dimension != int(self.config.model.embedding_dimension):
                raise ValueError("contextual cue dimension does not match application model")
            for vector_key in ("context_vector", "need_vector"):
                vector = np.asarray(materialization.get(vector_key))
                if (
                    vector.dtype != np.float32
                    or vector.shape != (dimension,)
                    or not np.all(np.isfinite(vector))
                ):
                    raise ValueError("contextual cue vector is invalid")
            prepared_materialization = dict(materialization)
            # A generic QueryEngine finalizer callback is not a capability
            # transport.  Discard any supplied closure and derive a fresh Q1
            # signer only here, immediately before the trusted repository
            # transaction.  The draft itself remains redacted and carries no
            # secret material.
            prepared_materialization.pop(
                "restricted_rewrite_manifest_binding_signer", None
            )
            restricted_rewrite_guard = prepared_materialization.get(
                "restricted_rewrite_guard"
            )
            if restricted_rewrite_guard is not None:
                if not isinstance(
                    restricted_rewrite_guard, ContextualRestrictedRewriteGuardDraft
                ):
                    raise ValueError("restricted rewrite guard is invalid")
                manifest_binding_signer = (
                    self._restricted_rewrite_manifest_binding_signer()
                )
                if not callable(manifest_binding_signer):
                    raise ValueError(
                        "restricted rewrite guard requires a process-only manifest signer"
                    )
                prepared_materialization[
                    "restricted_rewrite_manifest_binding_signer"
                ] = manifest_binding_signer
            prepared[candidate_id] = prepared_materialization

        # The repository validates source closure again inside its one DB
        # transaction.  Do not turn these source-fact objects into text here.
        committed = self.associations.finalize_recall_event(
            event,
            plan,
            cue_materializations=prepared,
            utility_weight=0.20,
            probation_ttl=self.config.retrieval.contextual_probation_ttl,
        )
        published: list[dict[str, object]] = []
        publication_failures = 0
        newly_published_receipt_ids: set[int] = set()
        for receipt in committed:
            current = dict(receipt)
            candidate_id = str(current.get("creation_request_id", "") or "")
            materialization = prepared.get(candidate_id)
            if materialization is None:
                # This should be impossible after repository validation.  It
                # is still safer to leave the committed receipt pending than
                # to infer vectors from a database cue record.
                publication_failures += 1
                published.append(current)
                continue
            try:
                publication = self.associations.get_contextual_index_publication()
                published_space = str(
                    publication.get("embedding_space_id", "") or ""
                ).strip()
                receipt_space = str(
                    current.get("embedding_space_id", "") or ""
                ).strip()
                if published_space and published_space != receipt_space:
                    raise ValueError(
                        "contextual index is published for another embedding space"
                    )
                self.context_cue_index.upsert(
                    int(current["context_cue_id"]),
                    materialization["context_vector"],
                )
                self.need_cue_index.upsert(
                    int(current["need_cue_id"]),
                    materialization["need_vector"],
                )
                if str(current.get("status", "")) == "committed_pending_index":
                    current = self.associations.mark_contextual_receipt_ready(
                        int(current["receipt_id"]),
                        context_cue_count=self.context_cue_index.count,
                        need_cue_count=self.need_cue_index.count,
                        expected_context_cue_id=int(current["context_cue_id"]),
                        expected_need_cue_id=int(current["need_cue_id"]),
                        restricted_rewrite_ready_manifest_signer=(
                            self._restricted_rewrite_ready_manifest_signer()
                        ),
                    )
                    if str(current.get("status", "") or "") == "ready":
                        newly_published_receipt_ids.add(int(current["receipt_id"]))
            except Exception:
                # Commit has already happened.  Keep the authoritative durable
                # status (normally committed_pending_index); do not make an
                # optimistic readiness claim or retry a model operation.
                publication_failures += 1
                durable = self.associations.get_contextual_creation_receipt(
                    int(current.get("receipt_id", 0) or 0)
                )
                if durable is not None:
                    current = durable
            published.append(dict(current))
        revisit_contracts: list[dict[str, object]] = []
        revisit_projections: list[dict[str, object]] = []
        for receipt in published:
            candidate_id = str(receipt.get("creation_request_id", "") or "")
            materialization = prepared.get(candidate_id)
            if materialization is None:
                continue
            projection_marker = str(
                materialization.get("revisit_projection", "") or ""
            )
            projection_detail = str(
                materialization.get("revisit_projection_reason", "") or ""
            )
            if projection_marker not in {
                "projected",
                "rejected_not_reconstructable",
            }:
                projection_marker = ""
            runtime_seed = materialization.get("revisit_runtime_seed")
            if runtime_seed is not None:
                outcome = self._finalize_runtime_manifest_after_publication(
                    receipt,
                    materialization,
                    newly_published=(
                        int(receipt.get("receipt_id", 0) or 0)
                        in newly_published_receipt_ids
                    ),
                )
            else:
                outcome = self._finalize_revisit_contract_after_publication(
                    receipt,
                    materialization,
                )
            if outcome is not None:
                revisit_contracts.append(outcome)
            if not projection_marker:
                continue
            try:
                receipt_id = int(receipt.get("receipt_id", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                receipt_id = 0
            if projection_marker == "rejected_not_reconstructable":
                allowed_rejection_reasons = {
                    "ordinary_contract_input_or_scope_invalid",
                    "ordinary_contract_single_required_slot_required",
                    "ordinary_contract_required_clause_missing",
                    "ordinary_contract_selected_anchor_unavailable",
                    "ordinary_contract_selected_anchor_invalid",
                    "ordinary_contract_source_closure_unavailable",
                    "ordinary_contract_source_mapping_support_unavailable",
                    "ordinary_contract_source_mapping_clause_mismatch",
                    "ordinary_contract_source_mapping_verification_missing",
                    "ordinary_contract_endpoint_limit_invalid",
                    "ordinary_contract_delivery_budget_invalid",
                    "ordinary_contract_draft_construction_failed_AttributeError",
                    "ordinary_contract_draft_construction_failed_KeyError",
                    "ordinary_contract_draft_construction_failed_TypeError",
                    "ordinary_contract_draft_construction_failed_ValueError",
                    "ordinary_contract_draft_construction_failed_FloatingPointError",
                    "ordinary_contract_runtime_projection_unavailable",
                }
                revisit_projections.append(
                    {
                        "candidate_id": candidate_id,
                        "receipt_id": receipt_id if receipt_id > 0 else None,
                        "status": "rejected_not_reconstructable",
                        "reason": (
                            projection_detail
                            if projection_detail in allowed_rejection_reasons
                            else "runtime_projection_not_reconstructable"
                        ),
                    }
                )
                continue
            outcome_status = (
                str(outcome.get("status", "") or "")
                if isinstance(outcome, dict)
                else "failed"
            )
            if outcome_status in {"created", "idempotent"}:
                projection_status = "ready"
                projection_reason = "runtime_manifest_ready"
            elif outcome_status == "deferred_not_ready":
                projection_status = "deferred_not_ready"
                projection_reason = "runtime_manifest_deferred"
            else:
                projection_status = "failed"
                projection_reason = "runtime_manifest_failed"
            revisit_projections.append(
                {
                    "candidate_id": candidate_id,
                    "receipt_id": receipt_id if receipt_id > 0 else None,
                    "status": projection_status,
                    "reason": projection_reason,
                }
            )
        return {
            "receipts": published,
            "publication_failures": publication_failures,
            "revisit_contracts": revisit_contracts,
            "revisit_projections": revisit_projections,
        }

    def rebuild_association_cue_index(
        self,
        *,
        batch_size: int = 64,
        logger: JsonlEventLogger | None = None,
    ) -> dict[str, int | str]:
        """Load persisted cue vectors and embed only new/changed relations."""
        active_logger = logger or self.new_logger("association-cue-index")
        rows = list(self.associations.list_cue_candidates())
        self.association_index = EmbeddingIndex(
            self.config.model.embedding_dimension,
            initial_capacity=max(16, len(rows) + 1),
        )
        size = max(1, int(batch_size))
        dimension = self.config.model.embedding_dimension
        missing: list[tuple[object, str]] = []
        reused = 0
        for row in rows:
            blob = row["cue_embedding"]
            relation_text = str(row["relation_text"])
            cue_text = association_cue_text(row)
            if blob is not None and str(row["cue_embedding_text"]) == cue_text:
                try:
                    vector = decode_embedding(blob, dimension)
                except (TypeError, ValueError):
                    missing.append((row, cue_text))
                else:
                    self.association_index.upsert(int(row["id"]), vector)
                    reused += 1
            else:
                missing.append((row, cue_text))
        model = ModelClient(self.config.model, active_logger) if missing else None
        embedded = 0
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            assert model is not None
            matrix = model.embed([cue_text for _row, cue_text in batch])
            for (row, cue_text), vector in zip(batch, matrix, strict=True):
                self.association_index.upsert(int(row["id"]), vector)
                self.associations.store_cue_embedding(
                    int(row["id"]),
                    str(row["relation_text"]),
                    cue_text,
                    encode_embedding(vector, dimension),
                )
                embedded += 1
        result = {
            "dtype": "float32",
            "dimension": dimension,
            "eligible_rows": len(rows),
            "indexed_rows": self.association_index.count,
            "reused_persisted_rows": reused,
            "embedded_rows": embedded,
            "memory_bytes": self.association_index.memory_bytes,
        }
        active_logger.emit("association_cue_index_rebuilt", result=result)
        return result

    def import_path(
        self,
        path: str | Path,
        source_root: str | Path | None = None,
        *,
        rebuild_indexes_before_import: bool = True,
    ) -> dict:
        logger = self.new_logger("import")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        # Standalone callers need to load persisted vectors. Long-lived
        # services already own current indexes and update them in place after
        # every insert, so rebuilding before every file is pure repeated work.
        if rebuild_indexes_before_import:
            pipeline.rebuild_indexes()
        return pipeline.import_path(path, source_root=source_root)

    def backfill_paragraphs(self) -> dict[str, int]:
        logger = self.new_logger("paragraph-backfill")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        pipeline.rebuild_indexes()
        return pipeline.backfill_paragraphs()

    def augment_concepts(self) -> dict[str, int | str]:
        logger = self.new_logger("concept-augmentation")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        pipeline.rebuild_indexes()
        return pipeline.augment_concepts()

    def query_engine(
        self,
        logger: JsonlEventLogger | None = None,
        *,
        config: AppConfig | None = None,
    ) -> QueryEngine:
        """Create a query-local engine over the application's shared indexes.

        ``config`` may be a request-scoped snapshot.  Repositories and RAM
        indexes remain shared, while QueryEngine's mutable traversal/growth
        helpers and ModelClient are isolated per request.
        """

        active_logger = logger or self.new_logger("query")
        active_config = config or self.config
        model = ModelClient(active_config.model, active_logger)
        return QueryEngine(
            active_config,
            model,
            self.episode_index,
            self.concept_index,
            self.episodes,
            self.concepts,
            self.sources,
            self.associations,
            active_logger,
            association_index=self.association_index,
            paragraph_index=self.paragraph_index,
            paragraphs=self.paragraphs,
            episode_sparse_index=self.episode_sparse_index,
            source_sparse_index=self.source_sparse_index,
            contextual_matcher=(
                ContextualAssociationMatcher(
                    self.context_cue_index,
                    self.need_cue_index,
                    self.associations,
                    context_threshold=active_config.retrieval.contextual_context_threshold,
                    need_threshold=active_config.retrieval.contextual_need_threshold,
                    combine_mode=active_config.retrieval.contextual_combine_mode,
                    context_top_k=active_config.retrieval.contextual_context_top_k,
                    need_top_k=active_config.retrieval.contextual_need_top_k,
                    edge_top_k=active_config.retrieval.contextual_edge_top_k,
                )
                if active_config.retrieval.contextual_association_enabled
                else None
            ),
            # The engine receives only this narrow post-answer callback.  It
            # cannot reach back into MemoryApplication or write directly.
            contextual_learning_finalizer=self.finalize_recall_event,
        )

    def stats(self) -> dict:
        return {
            "database": str(self.config.database_path),
            "sources": self.sources.count(),
            "episodes": self.episodes.count(),
            "concepts": self.concepts.count(),
            "paragraphs": self.paragraphs.count(),
            "associations": self.associations.stats(),
            "episode_index_count": self.episode_index.count,
            "concept_index_count": self.concept_index.count,
            "paragraph_index_count": self.paragraph_index.count,
            "association_index_count": self.association_index.count,
            "episode_index_memory_bytes": self.episode_index.memory_bytes,
            "concept_index_memory_bytes": self.concept_index.memory_bytes,
            "paragraph_index_memory_bytes": self.paragraph_index.memory_bytes,
            "association_index_memory_bytes": self.association_index.memory_bytes,
            "context_cue_index_count": self.context_cue_index.count,
            "need_cue_index_count": self.need_cue_index.count,
            "contextual_association_enabled": (
                self.config.retrieval.contextual_association_enabled
            ),
            "association_cue_enabled": (
                self.config.retrieval.association_cue_enabled
            ),
            "paragraph_retrieval_enabled": self.config.paragraph.enabled,
            "sparse_retrieval_enabled": self.config.retrieval.sparse_enabled,
            "episode_sparse_index_count": self.episode_sparse_index.count,
            "source_sparse_index_count": self.source_sparse_index.count,
            "concept_extraction_profile": self.config.concept_extraction.profile,
            "optimization_profile": self.config.optimization_profile,
            "episode_audit_mode": self.config.ingestion.episode_audit_mode,
            "episode_audit_always": self.config.ingestion.episode_audit_always,
            "episode_factual_audit_mode": (
                self.config.ingestion.episode_factual_audit_mode
            ),
            "episode_factual_audit_model": (
                self.config.ingestion.episode_factual_audit_model
                or self.config.model.reasoning_model
            ),
            "rerank_review_mode": self.config.retrieval.rerank_review_mode,
            "rerank_backend": self.config.retrieval.rerank_backend,
            "reranker_model": self.config.model.reranker_model,
            "embedding_dtype": "float32",
            "embedding_dimension": self.config.model.embedding_dimension,
        }
