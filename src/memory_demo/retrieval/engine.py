from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import json
import math
import re
import sqlite3
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Sequence
import unicodedata

import numpy as np

from memory_demo.association_overlay import (
    AssociationDelta,
    AssociationOverlay,
    StagedAssociationOverlay,
)
from memory_demo.associations.growth import AssociationGrowthEngine
from memory_demo.associations.plasticity import plan_learning_candidates
from memory_demo.associations.traversal import GraphTraverser, TraversedNode
from memory_demo.chronology import ChronologyService
from memory_demo.config import AppConfig
from memory_demo.database import Database, transaction_liveness, utc_now
from memory_demo.embeddings import (
    EmbeddingCoordinator,
    decode_embedding,
    normalize_embedding,
    normalize_query_text,
)
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.llm.client import TracePersistenceError
from memory_demo.llm.prompts import (
    ANSWER_AUDIT_SYSTEM,
    ANSWER_SYSTEM,
    ASSOCIATION_CUE_GATE_SYSTEM,
    EVENT_CONTINUITY_AUDIT_SYSTEM,
    EVIDENCE_COVERAGE_AUDIT_SYSTEM,
    EVIDENCE_RERANK_AUDIT_SYSTEM,
    EVIDENCE_RERANK_SYSTEM,
    HOP_QUERY_SYSTEM,
    QUERY_SYSTEM,
    answer_audit_prompt,
    answer_correction_prompt,
    answer_prompt,
    event_continuity_audit_prompt,
    evidence_coverage_audit_prompt,
    evidence_rerank_audit_prompt,
    evidence_rerank_prompt,
    hop_query_prompt,
    query_intent_prompt,
)
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.repositories.association import normalize_evaluation_as_of
from memory_demo.retrieval.context import (
    source_excerpt,
    verified_evidence_view_outcomes,
    verified_evidence_views,
)
from memory_demo.retrieval.cue_index import LexicalAssociationCueIndex
from memory_demo.retrieval.contextual_association import (
    ContextualAssociationMatcher,
    ContextualPreTargetProposal,
)
from memory_demo.retrieval.coverage import (
    aggregate_contributions,
    contribution_coverage_state,
    contribution_contextual_attribution,
    mask_contribution_aggregates,
    select_evidence,
    strict_contextual_attribution,
    treatment_masked_delta,
)
from memory_demo.retrieval.revisit import (
    ContextualRevisitContractDraft,
    ExactRevisitInput,
    ExactRevisitSourceMappingProof,
    ExactRevisitTargetMappingDraft,
    exact_revisit_aggregate_mapping_ref,
    exact_revisit_anchor_manifest_fingerprint,
    exact_revisit_budget_fingerprint,
    exact_revisit_context_hash,
    exact_revisit_mapping_contribution_id,
    exact_revisit_mapping_roles_fingerprint,
    exact_revisit_policy_fingerprint,
    exact_revisit_requirements_fingerprint,
    exact_revisit_request_hash,
    exact_revisit_source_closure_fingerprint,
    exact_revisit_source_fact_refs_fingerprint,
    exact_revisit_slot_need_bindings,
    RestrictedRewriteCommitmentKey,
    parse_restricted_rewrite_question,
    restricted_rewrite_guard_draft,
)
from memory_demo.retrieval.query_planning import (
    ResidualRepairPlan,
    ResidualSourceMappingReceipt,
    RequirementResolution,
    expand_rerank_atomic_queries,
    infer_request_mode,
    limit_rerank_atomic_queries,
    plan_local_residual_repair,
    requirement_support_from_records,
    requires_entity_resolved_followup,
    resolve_authoritative_requirements,
    structural_queries,
)
from memory_demo.types import (
    CandidateContribution,
    ClauseSupport,
    ContextualRestrictedRewriteGuardDraft,
    ContextualRestrictedRewriteGuard,
    ContextualRestrictedRewriteGuardLookup,
    ContextualRevisitRuntimeManifest,
    ContextualRevisitRuntimeManifestLookup,
    ContextualRevisitRuntimeSeed,
    EvidenceSelectionBudget,
    EvidenceSlot,
    LearningAnchor,
    LearningCandidate,
    LearningCandidatePlan,
    ContextualSlotHit,
    QueryIntent,
    PhysicalQueryVector,
    QueryVector,
    QueryVectorBundle,
    QueryVectorRequest,
    RecallLearningEvent,
    SearchHit,
    SlotCandidate,
    SourceFactRef,
)

if TYPE_CHECKING:
    from memory_demo.trace import (
        QueryTraceBridge,
        QueryTraceObservationCheckpoint,
    )


_PATH_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_PATH_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")


@dataclass(frozen=True, slots=True)
class ContextualTargetCheckOutcome:
    """One proposal's target validation and post-validation gate decision.

    The contextual matcher intentionally knows only local cue/anchor scores.
    This record keeps the later, independent target validation explicit: a
    current cosine relevance estimate can influence examination order, but it
    is never represented as factual verification or allowed to override an
    invalid source/evidence version.
    """

    proposal_key: str
    association_id: int
    target_episode_id: int
    matched_slot_id: str
    rank_before_endpoint_cap: int
    status: Literal["accepted", "rejected_target", "rejected_endpoint_cap"]
    reason: str
    evaluation_as_of: str
    source_id: int | None = None
    source_key: str = ""
    source_version: str = ""
    source_version_basis: str = ""
    source_fingerprint: str = ""
    evidence_status: str = "unknown"
    relevance_estimate: float | None = None
    relevance_status: str = "not_checked"

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def as_trace_payload(self) -> dict[str, object]:
        """Return structured diagnostic data without source text or vectors."""

        source_key_hash = (
            "sha256:" + hashlib.sha256(self.source_key.encode("utf-8")).hexdigest()
            if self.source_key
            else ""
        )
        source_version_hash = (
            "sha256:" + hashlib.sha256(
                self.source_version.encode("utf-8")
            ).hexdigest()
            if self.source_version
            else ""
        )

        result = {
            "proposal_key": self.proposal_key,
            "association_id": int(self.association_id),
            "target_episode_id": int(self.target_episode_id),
            "matched_slot_id": self.matched_slot_id,
            "rank_before_endpoint_cap": int(self.rank_before_endpoint_cap),
            "status": self.status,
            "reason": self.reason,
            "evaluation_as_of": self.evaluation_as_of,
            "source_id": self.source_id,
            "source_key_sha256": source_key_hash,
            "source_version_sha256": source_version_hash,
            "source_version_basis": self.source_version_basis,
            "source_fingerprint": self.source_fingerprint,
            "evidence_status": self.evidence_status,
            "relevance_estimate": self.relevance_estimate,
            "relevance_status": self.relevance_status,
        }
        return result


@dataclass(frozen=True, slots=True)
class ContextualTargetGateResult:
    """Typed output of target checking before the distinct-endpoint cap.

    ``hits`` is deliberately only a compatibility projection for the existing
    slot selector.  ``outcomes`` retains every proposal (including competing
    contributions and rejections) for T10's contribution-level selector.
    """

    hits: tuple[ContextualSlotHit, ...]
    outcomes: tuple[ContextualTargetCheckOutcome, ...]
    selected_endpoint_ids: tuple[int, ...]
    endpoint_limit: int

    def as_trace_payload(self) -> dict[str, object]:
        rejected: dict[str, int] = {}
        for outcome in self.outcomes:
            if outcome.accepted:
                continue
            rejected[outcome.reason] = rejected.get(outcome.reason, 0) + 1
        return {
            "stage": "target_checked_before_endpoint_cap_v1",
            "endpoint_limit": int(self.endpoint_limit),
            "selected_endpoint_ids": [int(value) for value in self.selected_endpoint_ids],
            "accepted_proposal_count": sum(
                1 for outcome in self.outcomes if outcome.accepted
            ),
            "rejected_proposal_count": sum(
                1 for outcome in self.outcomes if not outcome.accepted
            ),
            "rejected_reasons": dict(sorted(rejected.items())),
            "outcomes": [item.as_trace_payload() for item in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class _V3LearningCapture:
    """Non-serialised V3 selector state used only by an opted-in finalizer.

    The public query result deliberately contains a redacted selector trace.
    It has neither source-fact objects nor query vectors, so it is not a safe
    input to durable learning.  This tiny request-local capture keeps the
    already-validated runtime objects until the outer query boundary can make
    an explicit, post-answer finalization decision.
    """

    slots: tuple[EvidenceSlot, ...]
    contributions: tuple[CandidateContribution, ...]
    bundle: QueryVectorBundle | None
    initial_candidate_episode_ids: tuple[int, ...]
    # This is deliberately narrower than ``initial_candidate_episode_ids``.
    # It is populated only from direct retrieval seeds, before graph/cue
    # expansion.  T13 may use it for an anchor; it must never infer this
    # property from a later materialized Episode row.
    independent_base_episode_ids: tuple[int, ...]
    initial_delivered_episode_ids: tuple[int, ...]
    final_selected_episode_ids: tuple[int, ...]
    contextual_expansion_episode_ids: tuple[int, ...]
    cue_endpoint_episode_ids: tuple[int, ...]
    missing_required_clauses: tuple[str, ...]
    delivery_loss: bool
    shadow: bool
    target_gate_completed: bool
    # This is not a contextual target-gate result.  It records the distinct,
    # no-unresolved-slot case where the V3 contribution selector itself
    # completed a direct-base coverage check.  T13 may only use it for a
    # direct base target; contextual targets still require the real target
    # gate above.
    direct_base_validation_completed: bool
    # This request-frozen denominator exists only while the public query is
    # still live.  A compatibility capture may omit it, but Q1 contract
    # creation must then fail closed rather than rebuild it from trace data.
    requirements: RequirementResolution | None = None
    # ``None`` means the matcher default was in force.  The finalizer resolves
    # that live default before fingerprinting a possible revisit contract.
    endpoint_limit: int | None = None


@dataclass(frozen=True, slots=True)
class _V17AutomaticExactRevisit:
    """One private, fully reconstructed automatic exact-revisit attempt.

    This wrapper is intentionally process-local.  It carries only the
    transient input needed by the existing T15 validator plus the immutable
    manifest identity that must still be current at delivery time.  Neither
    object is copied into a public query result.
    """

    revisit_input: ExactRevisitInput
    manifest_id: int
    manifest_fingerprint: str


@dataclass(frozen=True, slots=True)
class _T16RestrictedRewriteRevisit:
    """A private HMAC capability for the narrow restricted-rewrite lane.

    It is never accepted from a public query caller.  The engine rebuilds it
    only after a current question is fully parsed by the controlled grammar
    and its scope-derived HMAC root identifies exactly one ready sidecar.
    """

    revisit_input: ExactRevisitInput
    manifest_id: int
    manifest_fingerprint: str
    creation_receipt_id: int
    guard_fingerprint: str
    rewrite_commitment: str
    binding_commitment: str
    manifest_binding_commitment: str
    ready_manifest_commitment: str
    context_cue_vector_fingerprint: str
    need_cue_vector_fingerprint: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    commitment_key_id: str
    grammar_version: str


class QueryEngine:
    def __init__(
        self,
        config: AppConfig,
        model,
        episode_index,
        concept_index,
        episodes: EpisodeRepository,
        concepts: ConceptRepository,
        sources: SourceRepository,
        associations: AssociationRepository,
        logger: JsonlEventLogger | None = None,
        *,
        association_index=None,
        paragraph_index=None,
        paragraphs: ParagraphRepository | None = None,
        episode_sparse_index=None,
        source_sparse_index=None,
        contextual_matcher: ContextualAssociationMatcher | None = None,
        contextual_learning_finalizer: Callable[..., object] | None = None,
    ):
        self.config = config
        self.model = model
        self.episode_index = episode_index
        self.concept_index = concept_index
        self.paragraph_index = paragraph_index
        self.association_index = association_index
        self.episodes = episodes
        self.concepts = concepts
        self.sources = sources
        self.paragraphs = paragraphs
        self.episode_sparse_index = episode_sparse_index
        self.source_sparse_index = source_sparse_index
        self.contextual_matcher = contextual_matcher
        # This callback is supplied only by MemoryApplication.  QueryEngine
        # never imports or reaches back into the application layer, which
        # keeps direct/test engine construction read-only by default.
        self.contextual_learning_finalizer = contextual_learning_finalizer
        self._v3_learning_capture: _V3LearningCapture | None = None
        self.associations = associations
        self._concept_reach_cache: dict[int, int] = {}
        self.last_query_embeddings: dict[str, np.ndarray] = {}
        self.last_query_embedding_cache_trace: dict[str, list[str]] = {
            "hits": [],
            "misses": [],
        }
        self.logger = logger
        if hasattr(self.model, "logger"):
            self.model.logger = logger
        self.traverser = GraphTraverser(associations)
        self.growth = AssociationGrowthEngine(
            model, associations, episodes, config.weights, logger
        )
        self.chronology = ChronologyService(episodes, associations, logger)
        self._lexical_association_cues: LexicalAssociationCueIndex | None = None

    def _provider_purpose_scope(self, role: str, purpose: str):
        """Attach one finite, observed query role when the client supports it.

        The ordinary test/replay models intentionally do not need to implement
        provider tracing.  In that case no synthetic trace event is produced;
        a strict trace with ``ModelClient`` receives the literal scope at the
        call boundary instead.
        """

        bind_purpose = getattr(self.model, "provider_purpose", None)
        if not callable(bind_purpose):
            return nullcontext()
        return bind_purpose(role, purpose)

    def _provider_deadline_scope(self, deadline_at: float | None):
        """Bind the public query's absolute deadline to capable providers.

        Lightweight/local test models intentionally have no transport budget
        API, so they retain the engine's existing stage-boundary behavior.
        ``ModelClient`` supplies this scope to stop queueing/retrying and to
        discard a late synchronous HTTP result before it can reach learning.
        """

        bind_deadline = getattr(self.model, "deadline_budget", None)
        if deadline_at is None or not callable(bind_deadline):
            return nullcontext()
        return bind_deadline(deadline_at)

    @staticmethod
    def _resolve_contextual_evaluation_as_of(value: str | None) -> str:
        """Pin one canonical UTC instant for all contextual reads in a request."""

        return normalize_evaluation_as_of(value if value is not None else utc_now())

    def contextual_recall(
        self,
        bundle,
        *,
        domain: str | None = None,
        endpoint_limit: int | None = None,
        active_anchor_ids=None,
        unresolved_slot_ids=None,
        target_support_scores=None,
        target_support_floor: float = 0.0,
        evaluation_as_of: str | None = None,
        emit_event: bool = True,
        scoring_mode: str = "cn",
    ) -> dict:
        """Run the optional local double-key lane for a prepared query bundle.

        This method is intentionally additive: the normal ``query`` path is
        unchanged while the experiment flag is disabled.  Callers can attach
        returned target Episode IDs to their candidate pool and feed them
        through the existing evidence contract.
        """
        if self.contextual_matcher is None or not self.config.retrieval.contextual_association_enabled:
            return {
                "enabled": False,
                "hits": [],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
            }
        as_of = self._resolve_contextual_evaluation_as_of(evaluation_as_of)
        result = self.contextual_matcher.match_bundle(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=active_anchor_ids,
            unresolved_slot_ids=unresolved_slot_ids,
            target_support_scores=target_support_scores,
            target_support_floor=target_support_floor,
            evaluation_as_of=as_of,
            scoring_mode=scoring_mode,
        )
        result["enabled"] = True
        result["backend"] = "contextual_double_key"
        result["scoring_mode"] = str(scoring_mode or "cn").strip().casefold()
        result["external_calls"] = 0
        result.setdefault("evaluation_as_of", as_of)
        result["attached_episode_ids"] = [
            int(hit.target_episode_id) for hit in result.get("hits", [])
        ]
        result["attached_edges"] = [
            int(hit.association_id) for hit in result.get("hits", [])
        ]
        # A public exact-revisit preflight is intentionally observationally
        # read-only.  Its normal route shares this local matcher with ordinary
        # retrieval, so logging must be an explicit opt-out rather than an
        # accidental write inherited from the ordinary query path.
        if emit_event and self.logger:
            self.logger.emit(
                "contextual_association_retrieval",
                enabled=True,
                backend="contextual_double_key",
                context_hits=len(result.get("context_hits", [])),
                need_hits=sum(len(item) for item in result.get("need_hits", [])),
                attached_edges=result["attached_edges"],
                attached_episode_ids=result["attached_episode_ids"],
                external_calls=0,
            )
        return result

    def _new_query_embedding_coordinator(self) -> EmbeddingCoordinator:
        return EmbeddingCoordinator(
            self.model,
            model_id=getattr(self.config.model, "embedding_model", ""),
            dimension=self.config.model.embedding_dimension,
        )

    @staticmethod
    def _request_vector_hash(text: str) -> str:
        return hashlib.sha256(
            normalize_query_text(text).encode("utf-8")
        ).hexdigest()

    def embed_query_bundle(self, texts):
        """Build a request-level vector bundle through one provider batch.

        This public compatibility entry point is also the ordinary contextual
        helper used by focused tests. Production ``query`` creates the same
        bundle earlier, before vector retrieval, with an operation-specific
        purpose label.
        """

        coordinator = self._new_query_embedding_coordinator()
        with self._provider_purpose_scope(
            "embedding", "contextual:query_vector_bundle"
        ):
            return coordinator.embed_request_bundle_sync(texts)

    def _embed_request_vector_bundle(
        self,
        coordinator: EmbeddingCoordinator,
        requests: Sequence[QueryVectorRequest],
        *,
        provider_purpose: str,
        source_request_hash: str,
    ) -> QueryVectorBundle:
        """Make one observed provider batch for the still-missing texts."""

        with self._provider_purpose_scope("embedding", provider_purpose):
            return coordinator.embed_request_bundle_sync(
                requests,
                source_request_hash=source_request_hash,
            )

    @staticmethod
    def _request_vector_specs(
        queries: Sequence[str],
        *,
        role: str,
        authoritative_requirements: RequirementResolution | None = None,
    ) -> list[QueryVectorRequest]:
        """Give every logical query an explicit role and evidence-slot link."""

        requirements_by_text: dict[str, list[EvidenceSlot]] = {}
        for requirement in (
            authoritative_requirements.requirements
            if authoritative_requirements is not None
            else ()
        ):
            normalized_requirement = normalize_query_text(requirement.question)
            if normalized_requirement:
                requirements_by_text.setdefault(
                    normalized_requirement, []
                ).append(requirement)
        result: list[QueryVectorRequest] = []
        for raw_query in queries:
            text = normalize_query_text(str(raw_query))
            if not text:
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if role == "whole":
                query_id = f"whole:{text_hash[:24]}"
                slot_id = ""
            else:
                # An atomic planner query may satisfy more than one frozen
                # obligation. Keep one *logical* binding per slot even though
                # the coordinator will deduplicate their shared text into one
                # physical provider vector. Follow-ups remain discovery-only
                # and deliberately never acquire a requirement slot here.
                requirements = (
                    requirements_by_text.get(text, [])
                    if role == "atomic"
                    else []
                )
                if requirements:
                    for requirement in requirements:
                        query_id = str(
                            requirement.query_id
                            or (
                                requirement.query_refs[0]
                                if requirement.query_refs
                                else ""
                            )
                            or f"requirement:{text_hash[:24]}"
                        )
                        result.append(
                            QueryVectorRequest(
                                role="atomic",
                                text=text,
                                query_id=query_id,
                                slot_id=str(requirement.slot_id),
                            )
                        )
                    continue
                query_id = f"{role}:{text_hash[:24]}"
                slot_id = ""
            result.append(
                QueryVectorRequest(
                    role=(role if role in {"whole", "atomic", "followup"} else "atomic"),  # type: ignore[arg-type]
                    text=text,
                    query_id=query_id,
                    slot_id=slot_id,
                )
            )
        return result

    @staticmethod
    def _bundle_vectors_for_specs(
        bundles: Sequence[QueryVectorBundle],
        requests: Sequence[QueryVectorRequest],
    ) -> tuple[dict[str, np.ndarray], list[QueryVectorRequest], QueryVectorBundle | None]:
        """Resolve supplied logical vectors without collapsing roles by text.

        The returned map is keyed by normalized text only because dense
        retrieval needs one aligned row per query text; the rebuilt bundle
        below still retains all logical bindings and slot IDs separately.
        """

        by_query_id: dict[str, np.ndarray] = {}
        by_slot_id: dict[str, np.ndarray] = {}
        by_text_hash: dict[str, np.ndarray] = {}
        selected_bundle: QueryVectorBundle | None = None
        for bundle in bundles:
            if int(bundle.dimension) <= 0:
                raise ValueError("query vector bundle dimension is invalid")
            for item in bundle.queries:
                try:
                    vector = np.asarray(bundle.vector_for(item), dtype=np.float32)
                except KeyError as exc:
                    raise ValueError("query vector bundle has a broken physical reference") from exc
                if vector.shape != (int(bundle.dimension),):
                    raise ValueError("query vector bundle contains an invalid vector shape")
                by_query_id.setdefault(str(item.query_id), vector)
                if str(item.slot_id).strip():
                    by_slot_id.setdefault(str(item.slot_id), vector)
                by_text_hash.setdefault(str(item.text_hash), vector)
        available: dict[str, np.ndarray] = {}
        missing: list[QueryVectorRequest] = []
        for request in requests:
            text = normalize_query_text(request.text)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            vector = by_query_id.get(str(request.query_id))
            if vector is None and str(request.slot_id).strip():
                vector = by_slot_id.get(str(request.slot_id))
            if vector is None:
                vector = by_text_hash.get(text_hash)
            if vector is None:
                missing.append(request)
                continue
            available.setdefault(text, vector)
            for bundle in bundles:
                if any(
                    str(item.query_id) == str(request.query_id)
                    or (
                        str(request.slot_id).strip()
                        and str(item.slot_id) == str(request.slot_id)
                    )
                    or str(item.text_hash) == text_hash
                    for item in bundle.queries
                ):
                    if selected_bundle is None:
                        selected_bundle = bundle
                    elif (
                        selected_bundle.embedding_space_id
                        != bundle.embedding_space_id
                    ):
                        raise ValueError(
                            "query vector bundles use incompatible embedding spaces"
                        )
                    break
        deduplicated_missing: list[QueryVectorRequest] = []
        seen_missing_texts: set[str] = set()
        for request in missing:
            text = normalize_query_text(request.text)
            if text not in seen_missing_texts:
                seen_missing_texts.add(text)
                deduplicated_missing.append(request)
        return available, deduplicated_missing, selected_bundle

    def _prepare_request_vector_bundle(
        self,
        coordinator: EmbeddingCoordinator,
        requests: Sequence[QueryVectorRequest],
        *,
        source_bundles: Sequence[QueryVectorBundle] = (),
        legacy_overrides: dict[str, np.ndarray] | None = None,
        provider_purpose: str,
        source_request_hash: str,
        strict_vector_bundle: bool = False,
    ) -> QueryVectorBundle:
        """Complete one logical bundle without bypassing the coordinator.

        Strict mode is intended for replay: it accepts only a complete,
        versioned vector bundle and rejects the historical free-form override
        map or an unplanned dynamic follow-up before any embedding call.
        """

        if strict_vector_bundle and legacy_overrides:
            raise ValueError(
                "strict vector replay does not allow legacy query_embeddings_override"
            )
        current_space = coordinator.embedding_space()
        for supplied in source_bundles:
            if int(supplied.dimension) != self.config.model.embedding_dimension:
                raise ValueError("query vector bundle dimension does not match engine")
            if supplied.embedding_space_id != current_space.canonical_id:
                raise ValueError(
                    "supplied query vector bundle uses an incompatible embedding space"
                )
        available, missing, supplied_bundle = self._bundle_vectors_for_specs(
            source_bundles,
            requests,
        )
        for raw_text, vector in (legacy_overrides or {}).items():
            normalized = normalize_query_text(raw_text)
            if normalized:
                available.setdefault(normalized, np.asarray(vector, dtype=np.float32))
        missing = [
            request
            for request in missing
            if normalize_query_text(request.text) not in available
        ]
        selected_space = (
            supplied_bundle.embedding_space
            if supplied_bundle is not None
            else current_space
        )
        if missing:
            if strict_vector_bundle:
                opaque_missing = ",".join(
                    self._request_vector_hash(item.text)[:16] for item in missing
                )
                raise ValueError(
                    "strict vector bundle is missing precomputed query vectors: "
                    + opaque_missing
                )
            embedded = self._embed_request_vector_bundle(
                coordinator,
                missing,
                provider_purpose=provider_purpose,
                source_request_hash=source_request_hash,
            )
            available.update(
                {
                    normalize_query_text(item.text): np.asarray(
                        embedded.vector_for(item), dtype=np.float32
                    )
                    for item in embedded.queries
                }
            )
            selected_space = embedded.embedding_space
        if int(selected_space.dimension) != self.config.model.embedding_dimension:
            raise ValueError("query vector bundle dimension does not match engine")
        return coordinator.bundle_from_precomputed_vectors(
            requests,
            available,
            embedding_space=selected_space,
            source_request_hash=source_request_hash,
        )

    @staticmethod
    def _bundle_overrides_for_queries(
        bundle: QueryVectorBundle,
        queries: Sequence[str],
    ) -> dict[str, np.ndarray]:
        by_hash: dict[str, np.ndarray] = {}
        for item in bundle.queries:
            by_hash.setdefault(
                str(item.text_hash),
                np.asarray(bundle.vector_for(item), dtype=np.float32),
            )
        result: dict[str, np.ndarray] = {}
        for raw_query in queries:
            text = normalize_query_text(raw_query)
            vector = by_hash.get(hashlib.sha256(text.encode("utf-8")).hexdigest())
            if vector is None:
                raise ValueError("request vector bundle lacks an aligned query vector")
            result[str(raw_query)] = vector
        return result

    @staticmethod
    def _query_vector_trace_metadata(bundle: QueryVectorBundle) -> dict:
        """Return trace-safe structural vector metadata without raw query text."""

        whole_query = next(
            (item for item in bundle.queries if item.role == "whole"),
            None,
        )
        return {
            "context_query_id": (
                str(whole_query.query_id) if whole_query is not None else ""
            ),
            "query_vectors": [
                {
                    "query_id": str(item.query_id),
                    "text_hash": str(item.text_hash),
                    "role": str(item.role),
                    "slot_id": str(item.slot_id),
                    "physical_id": str(item.physical_id),
                    "embedding_space_id": str(item.embedding_space_id),
                }
                for item in bundle.queries
            ],
            "query_vector_bundle": bundle.metadata(),
        }

    @staticmethod
    def _trace_unique_artifacts(references: Sequence[object]) -> list[object]:
        """Keep a trace event's content-addressed artifact list canonical.

        The trace writer deliberately rejects duplicate references within one
        JSONL record.  Query text can occur in several logical roles, so the
        engine deduplicates only the *artifact references* here; it never
        collapses the corresponding logical query bindings in the payload.
        """

        result: list[object] = []
        seen_ids: set[str] = set()
        for reference in references:
            artifact_id = str(getattr(reference, "artifact_id", "")).strip()
            if not artifact_id or artifact_id in seen_ids:
                continue
            seen_ids.add(artifact_id)
            result.append(reference)
        return result

    @classmethod
    def _emit_trace_requirements_resolved(
        cls,
        trace_bridge: QueryTraceBridge,
        resolution: RequirementResolution,
    ) -> str:
        """Persist the one requirements stage at the point it is observed.

        The payload is owned by ``RequirementResolution`` and contains only
        opaque references.  Its natural-language requirement questions remain
        in full-local, content-addressed artifacts so JSONL never becomes a
        second prompt/source-text channel.
        """

        artifacts = cls._trace_unique_artifacts(
            [
                trace_bridge.session.writer.artifacts.put_text(
                    str(requirement.question), visibility="full_local"
                )
                for requirement in resolution.requirements
                if str(requirement.question).strip()
            ]
        )
        checkpoint = trace_bridge.checkpoint()
        parents = (
            checkpoint.request_event_id,
            *checkpoint.provider_event_ids,
        )
        return trace_bridge.emit_observed_stage(
            "requirements_resolved",
            stage="requirements",
            payload=resolution.as_trace_payload(),
            artifact_refs=artifacts,
            parent_event_ids=parents,
            observation_checkpoint=checkpoint,
        )

    @classmethod
    def _emit_trace_vector_bundle_ready(
        cls,
        trace_bridge: QueryTraceBridge,
        bundle: QueryVectorBundle,
        *,
        causal_parent_event_id: str,
        before_vector_observations: QueryTraceObservationCheckpoint,
        physical_origins: dict[str, str],
        strict_single_batch: bool,
        stage_index: int,
        extra_provider_parent_event_ids: Sequence[str] = (),
    ) -> str:
        """Persist one complete, artifact-backed request vector bundle.

        This is intentionally a narrow observation hook: it can run only
        immediately after a coordinator has returned a complete bundle.  It
        does not infer a later retrieval, selector, contextual, or delivery
        stage from the final query result.
        """

        provider_delta = trace_bridge.observations_since(before_vector_observations)
        default_origin = (
            "provider"
            if int(provider_delta.embedding_logical_batches) > 0
            else "local_encoder"
        )
        text_by_binding = {
            (str(item.query_id), str(item.physical_id)): str(item.text)
            for item in bundle.queries
        }
        artifacts: list[object] = []
        query_refs: list[dict[str, object]] = []
        whole_query_ref = ""
        valid_hash = re.compile(r"[0-9a-f]{64}\Z")
        for binding in bundle.logical_bindings:
            query_id = str(binding.query_id).strip()
            physical_id = str(binding.physical_id).strip()
            text_hash = str(binding.text_hash).strip()
            text = str(
                binding.text
                or text_by_binding.get((query_id, physical_id), "")
            )
            normalized_text = normalize_query_text(text)
            if (
                not query_id
                or not physical_id
                or not normalized_text
                or not valid_hash.fullmatch(text_hash)
                or cls._request_vector_hash(normalized_text) != text_hash
            ):
                # Without the original query text we cannot truthfully satisfy
                # p_vectors.text_artifact_id.  A strict receipt must stop,
                # rather than fabricate a query artifact from a hash.
                raise TracePersistenceError(
                    "strict trace cannot persist a complete query vector bundle"
                )
            text_artifact = trace_bridge.session.writer.artifacts.put_text(
                normalized_text, visibility="full_local"
            )
            artifacts.append(text_artifact)
            role = str(binding.role)
            if role == "whole":
                if whole_query_ref and whole_query_ref != query_id:
                    raise TracePersistenceError(
                        "strict trace query vector bundle has multiple whole queries"
                    )
                whole_query_ref = query_id
            query_refs.append(
                {
                    "query_id": query_id,
                    "vector_id": physical_id,
                    "text_sha256": text_hash,
                    "role": role,
                    "slot_ids": (
                        [str(binding.slot_id)] if str(binding.slot_id) else []
                    ),
                    "text_artifact_id": text_artifact.artifact_id,
                }
            )

        physical_vectors: list[dict[str, object]] = []
        for physical in bundle.physical_vectors:
            physical_id = str(physical.physical_id).strip()
            vector = np.asarray(physical.vector)
            if (
                not physical_id
                or vector.dtype != np.float32
                or vector.shape != (int(bundle.dimension),)
                or not np.all(np.isfinite(vector))
                or not np.isclose(
                    float(np.linalg.norm(vector)),
                    1.0,
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise TracePersistenceError(
                    "strict trace query vector bundle is not normalized float32"
                )
            vector_artifact = trace_bridge.session.writer.artifacts.put_numpy(
                vector, visibility="full_local"
            )
            artifacts.append(vector_artifact)
            origin = str(physical_origins.get(physical_id, default_origin))
            if origin not in {
                "provider",
                "persisted",
                "request_cache",
                "local_encoder",
            }:
                raise TracePersistenceError("strict trace query vector origin is invalid")
            physical_vectors.append(
                {
                    "vector_id": physical_id,
                    "artifact_id": vector_artifact.artifact_id,
                    "array_key": physical_id,
                    "sha256": vector_artifact.sha256,
                    "dtype": "float32",
                    "dimension": int(bundle.dimension),
                    "embedding_space_id": str(bundle.embedding_space_id),
                    "normalization": "l2",
                    "origin": origin,
                }
            )

        if not whole_query_ref:
            raise TracePersistenceError(
                "strict trace query vector bundle lacks a whole-query binding"
            )
        physical_ids = {str(item["vector_id"]) for item in physical_vectors}
        if any(str(item["vector_id"]) not in physical_ids for item in query_refs):
            raise TracePersistenceError(
                "strict trace query vector bundle has a broken physical reference"
            )
        checkpoint = trace_bridge.checkpoint()
        parents = tuple(
            dict.fromkeys(
                (
                    causal_parent_event_id,
                    *extra_provider_parent_event_ids,
                    *provider_delta.provider_event_ids,
                )
            )
        )
        return trace_bridge.emit_observed_stage(
            "vector_bundle_ready",
            stage="vectors",
            payload={
                "query_refs": query_refs,
                "physical_vectors": physical_vectors,
                "whole_query_ref": whole_query_ref,
                "embedding_space_id": str(bundle.embedding_space_id),
                "logical_batches": int(provider_delta.embedding_logical_batches),
                "http_attempts": int(provider_delta.cloud_http_attempts),
                "stage_index": int(stage_index),
                "strict_single_batch": bool(strict_single_batch),
            },
            artifact_refs=cls._trace_unique_artifacts(artifacts),
            parent_event_ids=parents,
            observation_checkpoint=checkpoint,
        )

    @staticmethod
    def _slot_query_vector(
        bundle: QueryVectorBundle | None,
        slot: EvidenceSlot,
    ):
        """Resolve a slot to an already-prepared request vector, if any."""
        if bundle is None:
            return None
        slot_id = str(slot.slot_id).strip()
        query_id = str(slot.query_id).strip()
        # Exact logical bindings win. This matters when two independent
        # evidence slots intentionally share one text/physical vector: a text
        # fallback must never silently attach slot A to slot B's binding.
        if slot_id:
            matched = next(
                (item for item in bundle.queries if str(item.slot_id) == slot_id),
                None,
            )
            if matched is not None:
                return matched
        if query_id:
            matched = next(
                (item for item in bundle.queries if str(item.query_id) == query_id),
                None,
            )
            if matched is not None:
                return matched
        for query_ref in slot.query_refs:
            matched = next(
                (
                    item
                    for item in bundle.queries
                    if str(item.query_id) == str(query_ref)
                ),
                None,
            )
            if matched is not None:
                return matched
        question = str(slot.question).strip()
        if not question:
            return None
        for item in bundle.queries:
            if str(getattr(item, "text", "")).strip() == question:
                return item
        return None

    @staticmethod
    def _request_evidence_slots(
        rerank_trace: dict,
        bundle: QueryVectorBundle | None = None,
        authoritative_requirements: Sequence[EvidenceSlot] | None = None,
    ) -> tuple[list[EvidenceSlot], dict[int, set[str]]]:
        """Translate existing coverage/floor traces into local slot support.

        The conversion intentionally uses only an already-produced *valid*
        reranker coverage mapping.  It never asks a model to label candidates
        just for contextual recall.  Deterministic floors remain useful for
        candidate admission and ranking, but are not a semantic mapping: a
        lexical or vector floor cannot by itself satisfy a current request
        requirement merely because the selected Episode has a valid Source
        span.

        When a caller supplies ``authoritative_requirements``, those request
        requirements are the complete slot denominator.  Candidate records
        may attach support to them but cannot create, discard, or rewrite a
        requirement.  The legacy candidate-derived branch remains only for
        older direct callers until every path uses the v3 resolver.
        """
        if authoritative_requirements is not None:
            slots = list(authoritative_requirements)
            observations: list[dict] = []
            if QueryEngine._rerank_slot_mapping_status(rerank_trace) == "accepted":
                merged = rerank_trace.get("merged_coverage", {})
                coverage = (
                    merged.get("coverage", []) if isinstance(merged, dict) else []
                )
                for item in coverage if isinstance(coverage, list) else []:
                    if not isinstance(item, dict):
                        continue
                    observations.append(
                        {
                            "query": item.get("query"),
                            "episode_ids": item.get("episode_ids", []),
                            "clause_ids": item.get("clause_ids", []),
                        }
                    )
            return slots, requirement_support_from_records(slots, observations)

        slots: list[EvidenceSlot] = []
        support: dict[int, set[str]] = {}
        seen_queries: set[str] = set()
        used_slot_ids: set[str] = set()

        def normalized(value: object) -> str:
            return " ".join(str(value or "").casefold().split())

        def add_slot(query: object, episode_ids: object, prefix: str) -> None:
            if not isinstance(episode_ids, list):
                return
            question = str(query or "").strip()
            key = normalized(question)
            if not question or key in seen_queries:
                return
            ids = [
                int(value)
                for value in episode_ids
                if isinstance(value, (int, float, str)) and str(value).strip().lstrip("-").isdigit()
            ]
            if not ids:
                return
            query_vector = next(
                (
                    item
                    for item in (bundle.queries if bundle is not None else ())
                    if normalized(getattr(item, "text", "")) == key
                ),
                None,
            )
            slot_id = (
                str(query_vector.slot_id or query_vector.query_id).strip()
                if query_vector is not None
                else f"{prefix}:{len(slots)}"
            )
            if not slot_id or slot_id in used_slot_ids:
                slot_id = f"{prefix}:{len(slots)}"
            slots.append(
                EvidenceSlot(
                    slot_id=slot_id,
                    question=question,
                    query_id=(
                        str(query_vector.query_id)
                        if query_vector is not None
                        else question
                    ),
                    required=True,
                )
            )
            seen_queries.add(key)
            used_slot_ids.add(slot_id)
            for episode_id in ids:
                support.setdefault(episode_id, set()).add(slot_id)

        merged = rerank_trace.get("merged_coverage", {})
        coverage = merged.get("coverage", []) if isinstance(merged, dict) else []
        if QueryEngine._rerank_slot_mapping_status(rerank_trace) == "accepted":
            for item in coverage if isinstance(coverage, list) else []:
                if isinstance(item, dict):
                    add_slot(item.get("query"), item.get("episode_ids"), "coverage")
        return slots, support

    @staticmethod
    def _rerank_slot_mapping_status(rerank_trace: Mapping[str, object]) -> str:
        """Classify whether a saved rerank coverage mapping may cover a slot.

        Source closure verifies that a delivered quote still belongs to its
        Source.  It cannot repair a failed or stale semantic rerank response.
        This small request-local classifier keeps those contracts separate;
        it neither calls a model nor interprets the source text.
        """

        if not isinstance(rerank_trace, Mapping):
            return "not_observed"
        if str(rerank_trace.get("error", "")).strip():
            return "rerank_error"
        reuse = rerank_trace.get("candidate_input_reuse")
        if isinstance(reuse, Mapping):
            reuse_status = str(reuse.get("status", "")).strip()
            if reuse_status in {
                "frozen_input_fingerprint_missing_or_mismatched",
                "legacy_frozen_input_fingerprint_missing",
            }:
                return reuse_status
        merged = rerank_trace.get("merged_coverage")
        coverage = merged.get("coverage") if isinstance(merged, Mapping) else None
        if not isinstance(coverage, list) or not coverage:
            return "coverage_not_observed"
        return "accepted"

    @staticmethod
    def _frozen_rerank_reuse_decision(
        expected_fingerprint: object,
        observed_fingerprint: object,
    ) -> str:
        """Return the only permitted frozen rerank reuse decision.

        This deliberately takes no candidate-origin flag: provenance explains
        an edge's effect, whereas replay validity is solely a contract between
        the saved reranker input and the input about to be reused.
        """

        expected = str(expected_fingerprint or "")
        observed = str(observed_fingerprint or "")
        if not expected:
            return "legacy_frozen_input_fingerprint_missing"
        if hmac.compare_digest(expected, observed):
            return "frozen_input_fingerprint_matched"
        return "frozen_input_fingerprint_missing_or_mismatched"

    def _residual_repair_slot_bindings(
        self,
        bundle: QueryVectorBundle | None,
        plan: ResidualRepairPlan,
    ) -> tuple[
        list[tuple[EvidenceSlot, object, np.ndarray, str]],
        list[str],
    ]:
        """Resolve only exact, already-batched bindings for a repair plan.

        There is deliberately no query-id/text fallback here.  A residual
        repair must be able to prove that its local dense/sparse lookup used
        the missing slot's request-scoped physical vector, not a whole-query
        or another slot's coincidentally identical text.
        """

        if bundle is None or int(bundle.dimension) != int(
            self.config.model.embedding_dimension
        ):
            return [], list(plan.slot_ids)
        resolved: list[tuple[EvidenceSlot, object, np.ndarray, str]] = []
        skipped: list[str] = []
        for slot in plan.slots:
            bindings = bundle.bindings_for_slot(slot.slot_id)
            # One logical binding is the unambiguous contract.  Multiple
            # bindings (even if they happen to share a vector) need an
            # explicit future policy rather than a silent arbitrary choice.
            if len(bindings) != 1:
                skipped.append(slot.slot_id)
                continue
            binding = bindings[0]
            text = normalize_query_text(str(getattr(binding, "text", "")))
            if (
                str(getattr(binding, "slot_id", "")) != str(slot.slot_id)
                or not str(slot.query_id).strip()
                or str(getattr(binding, "query_id", ""))
                != str(slot.query_id)
                or str(getattr(binding, "role", "")) != "atomic"
                or not text
                or text != normalize_query_text(slot.question)
            ):
                skipped.append(slot.slot_id)
                continue
            try:
                vector = np.asarray(bundle.vector_for(binding), dtype=np.float32)
            except (AttributeError, KeyError, TypeError, ValueError):
                skipped.append(slot.slot_id)
                continue
            if (
                vector.shape != (int(bundle.dimension),)
                or not np.all(np.isfinite(vector))
                or not np.isclose(
                    float(np.linalg.norm(vector)),
                    1.0,
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                skipped.append(slot.slot_id)
                continue
            resolved.append((slot, binding, vector.copy(), text))
        return resolved, skipped

    @staticmethod
    def _residual_candidate_missing_episode_ids(
        rerank_trace: Mapping[str, object],
        *,
        limit: int,
    ) -> list[int]:
        """Read a bounded prior-candidate hint without treating it as proof.

        The JSON trace is useful only to recover an Episode ID that Candidate@N
        dropped.  It contains no replayable source-fact/clause mapping, so it
        can never establish slot support.  A separate typed receipt and a
        current source-closure check are required later for delivery.
        """

        deterministic = rerank_trace.get("deterministic_evidence_floor", {})
        if not isinstance(deterministic, Mapping) or limit <= 0:
            return []
        result: list[int] = []
        for lane in ("atomic_slots", "constraint_slots"):
            items = deterministic.get(lane, ())
            if not isinstance(items, (list, tuple)):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                raw_ids = item.get("candidate_missing_episode_ids", ())
                if not isinstance(raw_ids, (list, tuple)):
                    continue
                for raw_id in raw_ids:
                    if isinstance(raw_id, bool):
                        continue
                    try:
                        episode_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if episode_id > 0 and episode_id not in result:
                        result.append(episode_id)
                    if len(result) >= limit:
                        return result
        return result

    def _residual_mapping_receipt_diagnostics(
        self,
        *,
        mapping_receipts: Sequence[object],
        slots: Sequence[EvidenceSlot],
    ) -> tuple[dict[int, str], list[dict[str, object]]]:
        """Check typed receipt shape/current fact identity for diagnostics.

        A receipt is not a semantic-verifier capability.  The checks below
        are deliberately useful for trace/debugging only: this function never
        creates a ``ClauseSupport`` or a candidate contribution, and callers
        must not treat a ``current_source_fact`` status as factual coverage.
        """

        slot_by_id = {str(slot.slot_id): slot for slot in slots}
        typed_receipts = [
            receipt
            for receipt in mapping_receipts
            if isinstance(receipt, ResidualSourceMappingReceipt)
        ]
        facts, source_reasons = self._v3_source_fact_closure(
            [receipt.episode_id for receipt in typed_receipts]
        )
        receipt_trace: list[dict[str, object]] = []
        for raw_receipt in mapping_receipts:
            if not isinstance(raw_receipt, ResidualSourceMappingReceipt):
                receipt_trace.append({"status": "invalid_typed_receipt"})
                continue
            receipt = raw_receipt
            slot = slot_by_id.get(receipt.slot_id)
            if slot is None:
                receipt_trace.append(receipt.as_trace_payload(status="unknown_slot"))
                continue
            expected_query_id = str(slot.query_id).strip()
            expected_clause_ids = tuple(
                sorted(slot.clause_ids or (str(slot.slot_id),))
            )
            if not expected_query_id or receipt.query_id != expected_query_id:
                receipt_trace.append(receipt.as_trace_payload(status="query_id_mismatch"))
                continue
            if tuple(receipt.clause_ids) != expected_clause_ids:
                receipt_trace.append(receipt.as_trace_payload(status="clause_ids_mismatch"))
                continue
            fact = facts.get(int(receipt.episode_id))
            if fact is None:
                receipt_trace.append(
                    receipt.as_trace_payload(
                        status=source_reasons.get(
                            int(receipt.episode_id),
                            "source_closure_missing",
                        )
                    )
                )
                continue
            if fact.fact_id != receipt.source_fact_id:
                receipt_trace.append(
                    receipt.as_trace_payload(status="source_fact_identity_mismatch")
                )
                continue
            receipt_trace.append(
                receipt.as_trace_payload(
                    status="current_source_fact_identity_diagnostic"
                )
            )
        return source_reasons, receipt_trace

    @staticmethod
    def _residual_ranked_episode_ids(
        rankings: Mapping[str, object],
        query_index: int,
        *,
        limit: int,
    ) -> tuple[list[int], dict[int, float], set[int]]:
        """Extract one bounded local shortlist from an existing index pass."""

        result: list[int] = []
        scores: dict[int, float] = {}
        source_recovered: set[int] = set()

        def add_rows(key: str, *, source_lane: bool = False) -> None:
            groups = rankings.get(key, ())
            if not isinstance(groups, (list, tuple)) or query_index >= len(groups):
                return
            rows = groups[query_index]
            if not isinstance(rows, (list, tuple)):
                return
            for row in rows:
                if len(result) >= limit:
                    return
                if not isinstance(row, Mapping):
                    continue
                raw_id = row.get("episode_id", row.get("id"))
                try:
                    episode_id = int(raw_id)
                    score = float(
                        row.get(
                            "episode_cosine",
                            row.get("score", 0.0),
                        )
                    )
                except (TypeError, ValueError):
                    continue
                if episode_id <= 0 or not math.isfinite(score):
                    continue
                scores[episode_id] = max(scores.get(episode_id, -math.inf), score)
                if episode_id not in result:
                    result.append(episode_id)
                if source_lane:
                    source_recovered.add(episode_id)

        # ``atomic_episode`` interleaves dense, sparse, and source expansion
        # channels per slot; explicit source expansion is also retained so the
        # trace can distinguish a candidate-missing Source recovery.
        add_rows("atomic_episode")
        add_rows("sparse_source_episode_expansion", source_lane=True)
        add_rows("fused_episode")
        return result[:limit], scores, source_recovered

    def _repair_residual_evidence_slots(
        self,
        *,
        selected_episodes: Sequence[dict],
        candidate_episodes: Sequence[dict],
        rerank_trace: Mapping[str, object],
        authoritative_requirements: RequirementResolution,
        query_vector_bundle: QueryVectorBundle | None,
        mapping_receipts: Sequence[ResidualSourceMappingReceipt] = (),
        dynamic_missing_slot_ids: Sequence[str] = (),
    ) -> tuple[list[dict], dict[str, object]]:
        """Diagnose one bounded residual gap without altering delivered proof.

        This component has no trusted semantic-verifier capability. Existing
        vectors, local rankings, candidate-missing hints, and typed mapping
        receipts are diagnostic data only: they cannot establish slot support
        or append an Episode. ``dynamic_missing_slot_ids`` is the sole source
        of the plan; an empty input never expands to all requirements.
        """

        # Candidate rows cannot be materialized without a verifier-gated
        # integration. Keep this seam, but deliberately do not consume it.
        del candidate_episodes
        delivered = [dict(item) for item in selected_episodes if isinstance(item, dict)]
        delivered_ids: set[int] = set()
        for item in delivered:
            try:
                episode_id = int(item.get("id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if episode_id > 0:
                delivered_ids.add(episode_id)
        trace: dict[str, object] = {
            "version": "residual-local-repair-v1",
            "mode": "local_diagnostic_only",
            "round_limit": 1,
            "local_index_rounds": 0,
            "provider_calls": 0,
            "external_calls": 0,
            "whole_query_fallback": False,
            "full_rerank_skipped": True,
            "semantic_verifier_available": False,
            "preserved_selected_episode_ids": sorted(delivered_ids),
            "added_episode_ids": [],
            "candidate_missing_delivery_disabled": True,
            "ranking_only_candidate_episode_ids": [],
            "candidate_missing_candidate_episode_ids": [],
            "source_sparse_candidate_episode_ids": [],
            "mapping_receipt_count": len(mapping_receipts),
            "mapping_receipts": [],
        }
        if not bool(self.config.retrieval.contextual_association_enabled):
            trace["reason"] = "contextual_residual_repair_disabled"
            trace["early_stopped"] = True
            return delivered, trace

        slots = list(authoritative_requirements.requirements)
        # Receipt shape and current source-fact identity are only diagnostics.
        # They are not a semantic-verifier capability and never cover a slot.
        _receipt_source_reasons, receipt_trace = (
            self._residual_mapping_receipt_diagnostics(
                mapping_receipts=mapping_receipts,
                slots=slots,
            )
        )
        trace["mapping_receipts"] = receipt_trace
        plan = plan_local_residual_repair(
            authoritative_requirements,
            dynamic_missing_slot_ids,
        )
        trace.update(plan.as_trace_payload())
        trace["initial_missing_slot_ids"] = list(plan.slot_ids)
        trace["final_missing_slot_ids"] = list(plan.slot_ids)
        if not plan.enabled:
            trace["early_stopped"] = True
            return delivered, trace

        bindings, skipped_slot_ids = self._residual_repair_slot_bindings(
            query_vector_bundle,
            plan,
        )
        trace["bound_slot_ids"] = [slot.slot_id for slot, *_rest in bindings]
        trace["skipped_slot_ids"] = sorted(skipped_slot_ids)
        trace["query_vector_physical_ids"] = [
            str(getattr(binding, "physical_id", ""))
            for _slot, binding, _vector, _text in bindings
        ]
        if not bindings:
            trace["reason"] = "missing_exact_slot_vector_binding"
            trace["early_stopped"] = True
            return delivered, trace

        # One bounded local pass.  It operates directly on the exact physical
        # vectors selected above; it never calls ``model.embed`` or the
        # follow-up planner/reranker path.
        per_slot_limit = max(
            1,
            min(
                max(1, int(self.config.retrieval.candidate_limit)),
                2,
            ),
        )
        queries = [text for _slot, _binding, _vector, text in bindings]
        matrix = np.asarray(
            [vector for _slot, _binding, vector, _text in bindings],
            dtype=np.float32,
        )
        local_scores: dict[int, float] = {}
        local_ids_by_slot: dict[str, list[int]] = {}
        source_sparse_candidate_ids: set[int] = set()
        try:
            _hits, rankings = self._vector_seed_hits_from_matrix(
                queries,
                matrix,
                episode_anchor_ids=[],
                first_query_anchor_limit=None,
            )
            trace["local_index_rounds"] = 1
        except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError):
            trace["reason"] = "local_index_recovery_failed"
            trace["early_stopped"] = True
            return delivered, trace
        for index, (slot, _binding, _vector, _text) in enumerate(bindings):
            ranked_ids, ranked_scores, source_ids = self._residual_ranked_episode_ids(
                rankings,
                index,
                limit=per_slot_limit,
            )
            local_ids_by_slot[slot.slot_id] = ranked_ids
            source_sparse_candidate_ids.update(source_ids)
            for episode_id, score in ranked_scores.items():
                local_scores[episode_id] = max(
                    local_scores.get(episode_id, -math.inf), score
                )
        trace["local_candidate_counts"] = {
            slot_id: len(candidate_ids)
            for slot_id, candidate_ids in sorted(local_ids_by_slot.items())
        }

        total_candidate_limit = max(
            1,
            min(
                max(1, int(self.config.retrieval.candidate_limit)),
                per_slot_limit * len(plan.slots),
            ),
        )
        candidate_missing_ids = self._residual_candidate_missing_episode_ids(
            rerank_trace,
            limit=total_candidate_limit,
        )
        trace["candidate_missing_candidate_episode_ids"] = candidate_missing_ids
        candidate_ids: list[int] = []
        # Preserve deterministic plan/slot order.  A candidate-missing ID is
        # merely a retrieval hint; the typed receipt below decides whether it
        # can be source-bound.  It is deliberately not associated with a slot
        # through rerank JSON.
        for episode_id in candidate_missing_ids:
            if episode_id not in delivered_ids and episode_id not in candidate_ids:
                candidate_ids.append(episode_id)
            if len(candidate_ids) >= total_candidate_limit:
                break
        for slot in plan.slots:
            slot_candidates = local_ids_by_slot.get(slot.slot_id, ())
            admitted = 0
            for episode_id in slot_candidates:
                if episode_id in delivered_ids or episode_id in candidate_ids:
                    continue
                if admitted >= per_slot_limit or len(candidate_ids) >= total_candidate_limit:
                    break
                candidate_ids.append(episode_id)
                admitted += 1
        # No ranking, candidate-missing hint, or typed receipt is a trusted
        # semantic verifier.  Keep every candidate explicitly diagnostic and
        # stop before any materialization, coverage calculation, or delivery.
        trace["ranking_only_candidate_episode_ids"] = list(candidate_ids)
        trace["source_sparse_candidate_episode_ids"] = sorted(
            source_sparse_candidate_ids.intersection(candidate_ids)
        )
        trace["reason"] = "semantic_verifier_unavailable"
        trace["early_stopped"] = True
        return delivered, trace

    @staticmethod
    def _record_has_field(record: object | None, field_name: str) -> bool:
        """Support sqlite rows, mappings, and small test doubles uniformly."""

        if record is None:
            return False
        keys = getattr(record, "keys", None)
        if callable(keys):
            try:
                return field_name in keys()
            except (AttributeError, TypeError):
                return False
        try:
            record[field_name]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return hasattr(record, field_name)
        return True

    @classmethod
    def _record_value(
        cls,
        record: object | None,
        field_name: str,
        default: object | None = None,
    ) -> object | None:
        if record is None:
            return default
        try:
            return record[field_name]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return getattr(record, field_name, default)

    @staticmethod
    def _text_value(value: object | None) -> str:
        return str(value or "").strip()

    @classmethod
    def _explicit_target_version(
        cls,
        record: object | None,
        *,
        label: str,
    ) -> tuple[str, str, str | None]:
        """Find an explicit source revision without silently treating blank as old."""

        for field_name in (
            "source_version",
            "source_revision",
            "source_hash",
            "source_content_sha256",
            "content_hash",
            "revision",
            "version",
        ):
            if not cls._record_has_field(record, field_name):
                continue
            value = cls._text_value(cls._record_value(record, field_name))
            if not value:
                return "", f"{label}.{field_name}", "target_source_version_missing"
            return value, f"{label}.{field_name}", None
        return "", "", None

    @classmethod
    def _evidence_payload_present(cls, record: object, field_name: str) -> bool:
        """Require a parseable nonempty source-evidence payload, not a cosine."""

        value = cls._record_value(record, field_name)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
        return isinstance(value, (list, tuple)) and bool(value)

    @classmethod
    def _target_metadata_check(
        cls,
        proposal: ContextualPreTargetProposal,
        *,
        episode_rows: dict[int, object],
        source_rows: dict[int, object],
        evaluation_as_of: str,
    ) -> tuple[str | None, dict[str, object]]:
        """Validate current source-backed target state before relevance ranking.

        There is no separate source-version column in the historical schema.
        For that schema, the target's canonical ``updated_at`` is the minimal
        durable revision marker.  Newer callers may provide an explicit source
        version/hash, which is checked when both endpoint and source expose it.
        A blank or malformed declared version always fails closed.
        """

        target_id = int(proposal.target_episode_id)
        episode = episode_rows.get(target_id)
        metadata: dict[str, object] = {
            "source_id": None,
            "source_key": "",
            "source_version": "",
            "source_version_basis": "",
            "source_fingerprint": "",
            "evidence_status": "unknown",
        }
        if episode is None:
            return "target_episode_missing", metadata

        try:
            source_id = int(cls._record_value(episode, "source_id"))
        except (TypeError, ValueError):
            return "target_source_id_invalid", metadata
        if source_id <= 0:
            return "target_source_id_invalid", metadata
        metadata["source_id"] = source_id
        source_key = cls._text_value(cls._record_value(episode, "source_key"))
        metadata["source_key"] = source_key
        if not source_key:
            return "target_source_key_missing", metadata

        source = source_rows.get(source_id)
        if source is None:
            return "target_source_missing", metadata
        try:
            resolved_source_id = int(cls._record_value(source, "id"))
        except (TypeError, ValueError):
            return "target_source_id_invalid", metadata
        if resolved_source_id != source_id:
            return "target_source_mismatch", metadata
        source_text = cls._text_value(cls._record_value(source, "raw_text"))
        if not source_text:
            return "target_source_empty", metadata
        metadata["source_fingerprint"] = "sha256:" + hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest()

        episode_version, episode_basis, episode_version_error = (
            cls._explicit_target_version(episode, label="episode")
        )
        source_version, source_basis, source_version_error = (
            cls._explicit_target_version(source, label="source")
        )
        if episode_version_error is not None:
            metadata["source_version_basis"] = episode_basis
            return episode_version_error, metadata
        if source_version_error is not None:
            metadata["source_version_basis"] = source_basis
            return source_version_error, metadata
        if episode_version and source_version:
            # Values from different named fields may be complementary (for
            # example a source revision and an endpoint content hash); compare
            # only like-for-like declarations.
            if (
                episode_basis.rsplit(".", 1)[-1]
                == source_basis.rsplit(".", 1)[-1]
                and episode_version != source_version
            ):
                metadata["source_version"] = episode_version
                metadata["source_version_basis"] = episode_basis
                return "target_source_version_conflict", metadata
        if episode_version:
            metadata["source_version"] = episode_version
            metadata["source_version_basis"] = episode_basis
        elif source_version:
            metadata["source_version"] = source_version
            metadata["source_version_basis"] = source_basis

        updated_at = cls._text_value(cls._record_value(episode, "updated_at"))
        if not updated_at:
            return "target_source_version_missing", metadata
        try:
            canonical_updated_at = normalize_evaluation_as_of(updated_at)
        except (TypeError, ValueError):
            return "target_source_version_invalid", metadata
        if canonical_updated_at > evaluation_as_of:
            return "target_version_after_evaluation_as_of", metadata
        if not metadata["source_version"]:
            metadata["source_version"] = canonical_updated_at
            metadata["source_version_basis"] = "episode.updated_at"

        origin = cls._text_value(cls._record_value(episode, "evidence_origin")).casefold()
        if origin not in {"source", "mixed"}:
            metadata["evidence_status"] = "not_source_derived"
            return "target_evidence_origin_invalid", metadata
        try:
            generation = int(cls._record_value(episode, "generation"))
        except (TypeError, ValueError):
            metadata["evidence_status"] = "generation_unknown"
            return "target_evidence_generation_invalid", metadata
        if generation != 0:
            metadata["evidence_status"] = "non_source_generation"
            return "target_evidence_generation_invalid", metadata
        epistemic_status = cls._text_value(
            cls._record_value(episode, "epistemic_status")
        ).casefold()
        if epistemic_status not in {"observed", "asserted", "reported", "mixed"}:
            metadata["evidence_status"] = "epistemic_status_invalid"
            return "target_evidence_status_invalid", metadata
        evidence_basis = cls._text_value(
            cls._record_value(episode, "evidence_basis")
        ).casefold()
        if (
            not evidence_basis
            or evidence_basis in {"legacy_unavailable", "unknown"}
            or evidence_basis.startswith("legacy_")
        ):
            metadata["evidence_status"] = "evidence_basis_unavailable"
            return "target_evidence_basis_invalid", metadata
        if not cls._evidence_payload_present(episode, "evidence_quotes_json"):
            metadata["evidence_status"] = "evidence_quotes_missing"
            return "target_evidence_quotes_missing", metadata
        if not cls._evidence_payload_present(episode, "evidence_spans_json"):
            metadata["evidence_status"] = "evidence_spans_missing"
            return "target_evidence_spans_missing", metadata
        metadata["evidence_status"] = "current_source_evidence"
        return None, metadata

    @staticmethod
    def _rows_by_id(rows: Sequence[object]) -> dict[int, object]:
        result: dict[int, object] = {}
        for row in rows:
            try:
                row_id = int(QueryEngine._record_value(row, "id"))
            except (TypeError, ValueError):
                continue
            if row_id > 0:
                result[row_id] = row
        return result

    def _current_target_rows(
        self,
        target_episode_ids: Sequence[int],
    ) -> tuple[dict[int, object], dict[int, object]]:
        """Read the complete target/source closure once, before endpoint capping."""

        wanted = tuple(sorted({int(value) for value in target_episode_ids if int(value) > 0}))
        if not wanted:
            return {}, {}
        episode_rows = self._rows_by_id(self.episodes.get_many(wanted))
        source_ids: list[int] = []
        for episode in episode_rows.values():
            try:
                source_id = int(self._record_value(episode, "source_id"))
            except (TypeError, ValueError):
                continue
            if source_id > 0:
                source_ids.append(source_id)
        source_rows = self._rows_by_id(
            self.sources.get_many(tuple(sorted(set(source_ids))))
        )
        return episode_rows, source_rows

    def _contextual_target_relevance_scores(
        self,
        bundle: QueryVectorBundle | None,
        slots: Sequence[EvidenceSlot],
        target_episode_ids: Sequence[int],
    ) -> dict[tuple[str, int], float]:
        """Estimate current target relevance from request-local vectors only.

        This is a retrieval ordering signal.  It is deliberately named
        ``relevance`` rather than support/verification: source/evidence checks
        remain a separate, fail-closed condition in ``_target_metadata_check``.
        """

        if bundle is None or not target_episode_ids:
            return {}
        vectors = self.episode_index.get_many(target_episode_ids)
        if not isinstance(vectors, dict) or not vectors:
            return {}
        scores: dict[tuple[str, int], float] = {}
        seen_slots: set[str] = set()
        for slot in slots:
            slot_id = str(slot.slot_id)
            if not slot_id or slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            query = self._slot_query_vector(bundle, slot)
            if query is None:
                continue
            try:
                query_vector = normalize_embedding(
                    query.vector,
                    self.config.model.embedding_dimension,
                )
            except (TypeError, ValueError, FloatingPointError):
                continue
            for episode_id, target_vector in vectors.items():
                try:
                    target = normalize_embedding(
                        target_vector,
                        self.config.model.embedding_dimension,
                    )
                    value = float(np.dot(query_vector, target))
                except (TypeError, ValueError, FloatingPointError):
                    continue
                if math.isfinite(value):
                    scores[(slot_id, int(episode_id))] = value
        return scores

    def _gate_contextual_targets(
        self,
        *,
        proposals: Sequence[ContextualPreTargetProposal],
        bundle: QueryVectorBundle | None,
        unresolved_slots: Sequence[EvidenceSlot],
        endpoint_limit: int | None,
        evaluation_as_of: str,
    ) -> ContextualTargetGateResult:
        """Validate every proposed endpoint, then cap distinct Episodes only.

        No compatibility matcher hit is consulted here.  Every pre-target
        proposal gets a target-check outcome first, so a high-ranked invalid
        target cannot consume the endpoint budget and hide a later valid one.
        """

        as_of = normalize_evaluation_as_of(evaluation_as_of)
        proposal_list = list(proposals)
        default_limit = (
            int(self.contextual_matcher.endpoint_limit)
            if self.contextual_matcher is not None
            else 1
        )
        distinct_endpoint_limit = max(
            1,
            default_limit if endpoint_limit is None else int(endpoint_limit),
        )
        target_ids = [int(item.target_episode_id) for item in proposal_list]
        episode_rows, source_rows = self._current_target_rows(target_ids)
        relevance_scores = self._contextual_target_relevance_scores(
            bundle,
            unresolved_slots,
            target_ids,
        )

        outcomes: list[ContextualTargetCheckOutcome] = []
        accepted_candidates: list[
            tuple[int, ContextualPreTargetProposal, ContextualSlotHit]
        ] = []
        for index, proposal in enumerate(proposal_list):
            reason, metadata = self._target_metadata_check(
                proposal,
                episode_rows=episode_rows,
                source_rows=source_rows,
                evaluation_as_of=as_of,
            )
            relevance = relevance_scores.get(
                (str(proposal.matched_slot_id), int(proposal.target_episode_id))
            )
            relevance_status = (
                "estimated"
                if relevance is not None and math.isfinite(float(relevance))
                else "missing_or_invalid"
            )
            common = {
                "proposal_key": str(proposal.proposal_key),
                "association_id": int(proposal.association_id),
                "target_episode_id": int(proposal.target_episode_id),
                "matched_slot_id": str(proposal.matched_slot_id),
                "rank_before_endpoint_cap": int(proposal.rank_before_endpoint_cap),
                "evaluation_as_of": as_of,
                "source_id": metadata["source_id"],
                "source_key": str(metadata["source_key"]),
                "source_version": str(metadata["source_version"]),
                "source_version_basis": str(metadata["source_version_basis"]),
                "source_fingerprint": str(metadata["source_fingerprint"]),
                "evidence_status": str(metadata["evidence_status"]),
                "relevance_estimate": relevance,
                "relevance_status": relevance_status,
            }
            if reason is not None:
                outcomes.append(
                    ContextualTargetCheckOutcome(
                        status="rejected_target",
                        reason=reason,
                        **common,
                    )
                )
                continue
            if not math.isfinite(float(proposal.pre_target_score)) or float(
                proposal.pre_target_score
            ) <= 0:
                outcomes.append(
                    ContextualTargetCheckOutcome(
                        status="rejected_target",
                        reason="target_pre_score_invalid",
                        **common,
                    )
                )
                continue
            if relevance is None or not math.isfinite(float(relevance)):
                outcomes.append(
                    ContextualTargetCheckOutcome(
                        status="rejected_target",
                        reason="target_relevance_missing",
                        **common,
                    )
                )
                continue
            if float(relevance) <= 0:
                outcomes.append(
                    ContextualTargetCheckOutcome(
                        status="rejected_target",
                        reason="target_relevance_nonpositive",
                        **common,
                    )
                )
                continue
            total_score = float(proposal.pre_target_score) * float(relevance)
            if not math.isfinite(total_score) or total_score <= 0:
                outcomes.append(
                    ContextualTargetCheckOutcome(
                        status="rejected_target",
                        reason="target_combined_score_invalid",
                        **common,
                    )
                )
                continue
            outcome = ContextualTargetCheckOutcome(
                status="accepted",
                reason="target_current_and_relevant",
                **common,
            )
            outcomes.append(outcome)
            accepted_candidates.append(
                (
                    index,
                    proposal,
                    ContextualSlotHit(
                        association_id=int(proposal.association_id),
                        anchor_episode_id=int(proposal.anchor_episode_id),
                        target_episode_id=int(proposal.target_episode_id),
                        matched_slot_id=str(proposal.matched_slot_id),
                        matched_query_id=str(proposal.matched_query_id),
                        context_similarity=float(proposal.context_similarity),
                        need_similarity=float(proposal.need_similarity),
                        anchor_activation=float(proposal.anchor_activation),
                        utility_weight=float(proposal.utility_weight),
                        # Compatibility field only.  T09 records the separate
                        # ``relevance_estimate`` above; it is not a factual
                        # support/verification claim.
                        target_support_score=float(relevance),
                        lifecycle_state=str(proposal.lifecycle_state),
                        total_score=total_score,
                    ),
                )
            )

        accepted_candidates.sort(
            key=lambda item: (
                -float(item[2].total_score),
                int(item[1].rank_before_endpoint_cap),
                int(item[1].association_id),
                int(item[1].target_episode_id),
                str(item[1].matched_slot_id),
                str(item[1].proposal_key),
            )
        )
        selected_endpoint_ids: list[int] = []
        selected_endpoint_set: set[int] = set()
        hits: list[ContextualSlotHit] = []
        for outcome_index, proposal, hit in accepted_candidates:
            target_id = int(hit.target_episode_id)
            if target_id not in selected_endpoint_set:
                if len(selected_endpoint_set) >= distinct_endpoint_limit:
                    outcomes[outcome_index] = replace(
                        outcomes[outcome_index],
                        status="rejected_endpoint_cap",
                        reason="endpoint_budget_truncated",
                    )
                    continue
                selected_endpoint_set.add(target_id)
                selected_endpoint_ids.append(target_id)
            # A selected Episode may retain every independently matched slot;
            # it occupies one endpoint budget position, not one per slot.
            hits.append(hit)

        return ContextualTargetGateResult(
            hits=tuple(hits),
            outcomes=tuple(outcomes),
            selected_endpoint_ids=tuple(selected_endpoint_ids),
            endpoint_limit=distinct_endpoint_limit,
        )

    @staticmethod
    def _base_anchor_activations(seeds: list[SearchHit]) -> dict[int, float]:
        """Keep only independently retrieved Episode anchors for v2 recall."""
        raw = {
            int(item.node_id): max(0.0, float(item.score))
            for item in seeds
            if item.node_type == "episode" and float(item.score) > 0
        }
        maximum = max(raw.values(), default=0.0)
        return (
            {episode_id: score / maximum for episode_id, score in raw.items()}
            if maximum > 0
            else {}
        )

    @staticmethod
    def _association_cue_episode_ids(entries: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
        """Return cue-derived Episode endpoints that cannot seed T13 anchors.

        ``_vector_seed_hits_with_cues`` intentionally merges dense and cue
        hits for ordinary retrieval.  A learning anchor needs the stricter
        origin guarantee, so this helper preserves only the endpoint IDs to
        exclude; it does not expose cue text or alter normal selection.
        """

        result: set[int] = set()
        for entry in entries:
            endpoints = entry.get("endpoints", ())
            if not isinstance(endpoints, (list, tuple)):
                continue
            for endpoint in endpoints:
                if not isinstance(endpoint, (list, tuple)) or len(endpoint) != 2:
                    continue
                if str(endpoint[0]).strip() != "episode":
                    continue
                try:
                    endpoint_id = int(endpoint[1])
                except (TypeError, ValueError):
                    continue
                if endpoint_id > 0:
                    result.add(endpoint_id)
        return tuple(sorted(result))

    @staticmethod
    def _prepared_early_candidate_pool_hits(
        accepted_candidates: object,
        direct_base_episode_ids: set[int],
    ) -> tuple[list[SearchHit], tuple[int, ...]]:
        """Turn verified, non-base proposals into ordinary traversal seeds.

        This is intentionally a one-pass filter, not a second ranker. The
        caller supplies only targets that passed the existing source/target
        gate; a direct base target stays independent and is not reintroduced
        under contextual provenance.
        """

        if not isinstance(accepted_candidates, list):
            return [], ()
        injected_hits: list[SearchHit] = []
        eligible_ids: list[int] = []
        seen_eligible: set[int] = set()
        for candidate in accepted_candidates:
            if not isinstance(candidate, Mapping):
                continue
            try:
                target_episode_id = int(candidate.get("target_episode_id", 0))
                score = float(candidate.get("combined_score", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                target_episode_id <= 0
                or target_episode_id in direct_base_episode_ids
                or target_episode_id in seen_eligible
                or not math.isfinite(score)
                or score < 0.0
            ):
                continue
            seen_eligible.add(target_episode_id)
            eligible_ids.append(target_episode_id)
            injected_hits.append(
                SearchHit("episode", target_episode_id, max(0.0, score))
            )
        return injected_hits, tuple(eligible_ids)

    def _prepared_early_contextual_proposal(
        self,
        *,
        bundle: QueryVectorBundle | None,
        domain: str | None,
        endpoint_limit: int | None,
        anchor_activations: Mapping[int, float],
        authoritative_requirements: RequirementResolution,
        evaluation_as_of: str | None,
        cue_endpoint_episode_ids: Sequence[int] = (),
    ) -> dict[str, object]:
        """Observe a non-exact contextual proposal before broad retrieval.

        This is W06's deliberately narrow P2 diagnostic lane.  It reuses the
        normal matcher and target gate, but only after the public request has
        frozen its requirements and prepared its shared vector bundle.  The
        first implementation is shadow-only: a proposal cannot enter the
        candidate pool, prove a requirement, suppress an ordinary module, or
        create a learning/use record.  That keeps the trace useful without
        treating an association relevance signal as delivered evidence.
        """

        excluded_anchor_ids = {
            int(value) for value in cue_endpoint_episode_ids if int(value) > 0
        }
        independent_anchors: dict[int, float] = {}
        for raw_id, raw_score in anchor_activations.items():
            try:
                episode_id = int(raw_id)
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if (
                episode_id > 0
                and episode_id not in excluded_anchor_ids
                and math.isfinite(score)
                and score > 0
            ):
                independent_anchors[episode_id] = score
        required_slots = [
            item
            for item in authoritative_requirements.requirements
            if bool(item.required) and str(item.slot_id).strip()
        ]
        trace: dict[str, object] = {
            "stage": "prepared_early_before_initial_graph_expansion_v1",
            "enabled": bool(
                self.config.retrieval.contextual_association_enabled
                and self.config.retrieval.contextual_prepared_early_enabled
            ),
            "shadow": bool(self.config.retrieval.contextual_prepared_early_shadow),
            "request_requirement_status": authoritative_requirements.status,
            "current_requirement_slot_ids": [
                str(item.slot_id) for item in required_slots
            ],
            # The matcher parameter retains its historical name
            # ``unresolved_slot_ids``.  At this early position these are the
            # complete, request-frozen requirements, not a late residual.
            "need_origin": "current_request_requirements_before_base_selection",
            "vector_origin": "request_bound_query_vector_bundle",
            "key_scoring_mode": str(
                self.config.retrieval.contextual_prepared_early_scoring_mode
            ),
            "evaluation_as_of": evaluation_as_of,
            "independent_anchor_episode_ids": sorted(independent_anchors),
            "cue_excluded_anchor_episode_ids": sorted(excluded_anchor_ids),
            "proposal_generated": False,
            # Backwards-compatible name for the metadata/current-version
            # target gate below. It is deliberately not a quote/span binding
            # claim; candidate treatment performs that stricter closure only
            # before an accepted proposal may alter graph seeds.
            "source_validated": False,
            "source_metadata_validated": False,
            "source_binding_validated": False,
            "source_binding_status": "not_checked",
            "current_requirement_supported": False,
            "current_requirement_support_status": (
                "not_evaluated_before_base_selection"
            ),
            "early_stop_allowed": False,
            "candidate_pool_mutated": False,
            "candidate_pool_requested": bool(
                self.config.retrieval.contextual_prepared_early_candidate_pool_enabled
            ),
            "candidate_pool_validated_episode_ids": [],
            "candidate_pool_eligible_episode_ids": [],
            "candidate_pool_injected_episode_ids": [],
            "candidate_pool_injection_reason": "not_requested",
            "ordinary_retrieval_continues": True,
            "pre_target_candidate_order": [],
            "accepted_candidate_order": [],
            "target_gate": {
                "stage": "not_entered",
                "outcomes": [],
            },
            "executed_modules": {
                "prepared_early_matcher": False,
                "prepared_early_target_gate": False,
                "broad_retrieval": "continues_after_this_stage",
            },
            "external_calls": 0,
        }
        if bundle is not None:
            trace.update(self._query_vector_trace_metadata(bundle))
            context_physical_id = str(bundle.whole_physical_id)
            need_physical_ids = sorted(
                {
                    str(binding.physical_id)
                    for slot in required_slots
                    for binding in bundle.bindings_for_slot(str(slot.slot_id))
                    if str(binding.physical_id)
                }
            )
            trace["context_physical_id"] = context_physical_id
            trace["need_physical_ids"] = need_physical_ids
            trace["context_need_vector_degenerate"] = bool(
                need_physical_ids
                and context_physical_id
                and all(item == context_physical_id for item in need_physical_ids)
            )
            trace["context_need_vector_relationship"] = (
                "same_physical_vector"
                if trace["context_need_vector_degenerate"]
                else "distinct_physical_vectors"
                if need_physical_ids and context_physical_id
                else "not_observed"
            )
        if not trace["enabled"]:
            trace["retention_reason"] = "candidate_only"
            trace["reason"] = "prepared_early_disabled"
            return trace
        if self.contextual_matcher is None:
            trace["retention_reason"] = "incomplete_input"
            trace["reason"] = "contextual_matcher_unavailable"
            return trace
        if bundle is None or evaluation_as_of is None or not required_slots:
            trace["retention_reason"] = "incomplete_input"
            trace["reason"] = "missing_bound_vectors_evaluation_or_requirements"
            return trace
        if not independent_anchors:
            trace["retention_reason"] = "no_independent_anchor"
            trace["reason"] = "no_independent_anchor"
            return trace

        # ``contextual_recall`` is the same local public-query matcher used by
        # the late selector.  Suppressing its generic event avoids a second,
        # ambiguous request-level log record; this structured branch trace is
        # the authoritative observation for the prepared-early stage.
        preliminary = self.contextual_recall(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=independent_anchors,
            unresolved_slot_ids=[item.slot_id for item in required_slots],
            evaluation_as_of=evaluation_as_of,
            emit_event=False,
            scoring_mode=(
                self.config.retrieval.contextual_prepared_early_scoring_mode
            ),
        )
        trace["executed_modules"] = {
            "prepared_early_matcher": True,
            "prepared_early_target_gate": False,
            "broad_retrieval": "continues_after_this_stage",
        }
        trace["matcher_backend"] = str(preliminary.get("backend", ""))
        trace["key_scoring_mode"] = str(preliminary.get("scoring_mode", "cn"))
        trace["matcher_reason"] = str(preliminary.get("reason", ""))
        raw_proposals = preliminary.get("pre_target_proposals", [])
        if not isinstance(raw_proposals, (list, tuple)) or not all(
            isinstance(item, ContextualPreTargetProposal) for item in raw_proposals
        ):
            trace["retention_reason"] = "incomplete_input"
            trace["reason"] = "invalid_pre_target_proposals"
            return trace
        trace["proposal_generated"] = bool(raw_proposals)
        trace["pre_target_candidate_order"] = [
            {
                "rank_before_endpoint_cap": int(item.rank_before_endpoint_cap),
                "proposal_key": str(item.proposal_key),
                "association_id": int(item.association_id),
                "anchor_episode_id": int(item.anchor_episode_id),
                "target_episode_id": int(item.target_episode_id),
                "matched_slot_id": str(item.matched_slot_id),
                "context_similarity": float(item.context_similarity),
                "need_similarity": float(item.need_similarity),
                "context_gate_score": float(item.context_gate_score),
                "need_gate_score": float(item.need_gate_score),
                "pre_target_score": float(item.pre_target_score),
            }
            for item in raw_proposals
        ]
        if not raw_proposals:
            matcher_reason = str(preliminary.get("reason", ""))
            trace["retention_reason"] = (
                "no_independent_anchor"
                if matcher_reason == "no_active_anchor"
                else "no_anchor_conditioned_edges"
                if matcher_reason == "no_anchor_conditioned_edges"
                else "cue_below_threshold"
            )
            trace["reason"] = matcher_reason or "no_pre_target_candidates"
            return trace

        gate_result = self._gate_contextual_targets(
            proposals=raw_proposals,
            bundle=bundle,
            unresolved_slots=required_slots,
            endpoint_limit=endpoint_limit,
            evaluation_as_of=evaluation_as_of,
        )
        trace["executed_modules"] = {
            "prepared_early_matcher": True,
            "prepared_early_target_gate": True,
            "broad_retrieval": "continues_after_this_stage",
        }
        trace["target_gate"] = gate_result.as_trace_payload()
        trace["accepted_candidate_order"] = [
            {
                "association_id": int(item.association_id),
                "anchor_episode_id": int(item.anchor_episode_id),
                "target_episode_id": int(item.target_episode_id),
                "matched_slot_id": str(item.matched_slot_id),
                "context_similarity": float(item.context_similarity),
                "need_similarity": float(item.need_similarity),
                "target_relevance_score": float(item.target_support_score),
                "combined_score": float(item.total_score),
            }
            for item in gate_result.hits
        ]
        trace["source_validated"] = bool(gate_result.hits)
        trace["source_metadata_validated"] = bool(gate_result.hits)
        trace["candidate_pool_validated_episode_ids"] = [
            int(item.target_episode_id) for item in gate_result.hits
        ]
        if gate_result.hits:
            # The exact same target/source gate that the late selector uses
            # has passed.  This says only that the target is currently
            # source-valid and relevant enough to inspect; it still does not
            # convert edge relevance into proof of the complete requirement.
            trace["retention_reason"] = "candidate_only"
            trace["reason"] = "shadow_candidate_source_validated"
            return trace

        rejected_reasons = {
            str(item.reason) for item in gate_result.outcomes if not item.accepted
        }
        if any(reason.startswith("target_source") or reason.startswith("target_evidence") for reason in rejected_reasons):
            trace["retention_reason"] = "source_invalid"
        elif "endpoint_budget_truncated" in rejected_reasons:
            trace["retention_reason"] = "budget_exhausted"
        else:
            trace["retention_reason"] = "candidate_only"
        trace["reason"] = "no_source_validated_prepared_candidate"
        return trace

    def _slot_candidates(
        self,
        episodes: list[dict],
        slot_support: dict[int, set[str]],
        reranked_episode_ids: list[int],
        contextual_hits=(),
        target_relevance_scores: dict[tuple[str, int], float] | None = None,
    ) -> list[SlotCandidate]:
        rank = {
            int(episode_id): index
            for index, episode_id in enumerate(reranked_episode_ids, start=1)
        }
        span = max(1, len(rank))
        contextual_by_episode: dict[int, list] = {}
        for hit in contextual_hits:
            contextual_by_episode.setdefault(int(hit.target_episode_id), []).append(hit)
        result: list[SlotCandidate] = []
        for episode in episodes:
            episode_id = int(episode["id"])
            hits = contextual_by_episode.get(episode_id, [])
            contextual_slots = {str(hit.matched_slot_id) for hit in hits if hit.matched_slot_id}
            contextual_score = max((float(hit.total_score) for hit in hits), default=0.0)
            edge_id = (
                max(hits, key=lambda item: (item.total_score, -item.association_id)).association_id
                if hits
                else None
            )
            direct_rank = (span - rank[episode_id] + 1) / span if episode_id in rank else 0.0
            # This is a retrieval relevance estimate, not a factual support
            # verdict. Target provenance/evidence was independently checked by
            # T09 before a contextual hit entered this compatibility selector.
            local_relevance = max(
                (
                    float(value)
                    for (slot_id, target_id), value in (target_relevance_scores or {}).items()
                    if target_id == episode_id and slot_id in contextual_slots
                ),
                default=0.0,
            )
            result.append(
                SlotCandidate(
                    episode_id=episode_id,
                    slot_ids=frozenset(slot_support.get(episode_id, set()).union(contextual_slots)),
                    source_lane="contextual" if hits else "base",
                    direct_score=max(float(episode.get("score", 0.0)), direct_rank, local_relevance),
                    specificity_score=local_relevance,
                    redundancy_group=str(episode.get("source_key", "")),
                    contextual_edge_id=int(edge_id) if edge_id is not None else None,
                    contextual_score=contextual_score,
                    source_quality=(
                        0.10
                        if int(episode.get("generation", 0) or 0) == 0
                        and str(episode.get("evidence_origin", "source")) in {"source", "mixed"}
                        else 0.0
                    ),
                )
            )
        return result

    # ------------------------------------------------------------------
    # V3 contribution selector
    # ------------------------------------------------------------------
    #
    # The historical SlotCandidate path above remains available for callers
    # which have not supplied the request-frozen requirement contract.  V3
    # calls take the branch below instead.  Its small conversion layer is
    # intentionally local to QueryEngine: the pure aggregation and selection
    # rules live in retrieval.coverage, while repository reads needed to prove
    # a source span stay at this boundary.

    @staticmethod
    def _v3_opaque_ref(*parts: object) -> str:
        """Return an opaque deterministic id without leaking source labels."""

        payload = json.dumps(
            [str(part) for part in parts],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _v17_revisit_runtime_ref(kind: str, *parts: object) -> str:
        """Build one seed-safe opaque runtime reference.

        ``ContextualRevisitRuntimeSeed`` derives its public-in-process IDs
        from the final SHA-256 component.  Keeping the ref itself opaque makes
        that derivation deterministic without carrying a planner's original
        slot, query, or clause label into the durable runtime manifest.
        """

        normalized_kind = str(kind or "").strip()
        if normalized_kind not in {"slot", "query", "clause"}:
            raise ValueError("runtime revisit reference kind is invalid")
        return (
            f"revisit-runtime-{normalized_kind}:"
            + QueryEngine._v3_opaque_ref(
                "v17-runtime-projection-v1", normalized_kind, *parts
            )
        )

    @staticmethod
    def _v3_finite_score(value: object, default: float = 0.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default
        return score if math.isfinite(score) else default

    @staticmethod
    def _v3_json_list(value: object) -> list[object] | None:
        """Decode persisted evidence lists without allowing query failures."""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        return list(value) if isinstance(value, (list, tuple)) else None

    @classmethod
    def _v3_source_fact_for_closure(
        cls,
        *,
        episode_id: int,
        episode: object | None,
        source: object | None,
    ) -> tuple[SourceFactRef | None, str]:
        """Build one fact locator only from a full current source closure.

        ``_materialize_nodes`` deliberately exposes a short excerpt for answer
        generation.  That excerpt, an Episode summary, or a source key cannot
        prove a factual clause.  This helper therefore starts again from the
        current Episode and full Source repository rows, validates the
        persisted 1-based line spans against raw text, and otherwise returns a
        relevance-only reason.  It is deliberately fail-closed and does not
        raise for imperfect historical imports.
        """

        if episode is None or source is None:
            return None, "source_closure_missing"
        try:
            source_id = int(cls._record_value(episode, "source_id"))
            resolved_source_id = int(cls._record_value(source, "id"))
        except (TypeError, ValueError):
            return None, "source_closure_id_invalid"
        if source_id <= 0 or resolved_source_id != source_id:
            return None, "source_closure_id_mismatch"
        # Preserve the persisted Source string for the revision closure.  The
        # evidence-basis contract below selects its own deterministic line
        # view; normalizing the raw source here would otherwise silently
        # change either raw-source or reasoning-view locators.
        raw_source = str(cls._record_value(source, "raw_text") or "")
        if not raw_source.strip():
            return None, "source_raw_text_missing"

        origin = cls._text_value(
            cls._record_value(episode, "evidence_origin")
        ).casefold()
        if origin not in {"source", "mixed"}:
            return None, "source_evidence_origin_invalid"
        try:
            generation = int(cls._record_value(episode, "generation"))
        except (TypeError, ValueError):
            return None, "source_evidence_generation_invalid"
        if generation != 0:
            return None, "source_evidence_generation_invalid"
        epistemic_status = cls._text_value(
            cls._record_value(episode, "epistemic_status")
        ).casefold()
        if epistemic_status not in {"observed", "asserted", "reported", "mixed"}:
            return None, "source_evidence_status_invalid"
        evidence_basis = cls._text_value(
            cls._record_value(episode, "evidence_basis")
        ).casefold()
        if (
            not evidence_basis
            or evidence_basis in {"unknown", "legacy_unavailable"}
            or evidence_basis.startswith("legacy_")
            or evidence_basis not in {"source_id", "reasoning_view_nonempty_lines_v1"}
        ):
            return None, "source_evidence_basis_invalid"

        raw_spans = cls._v3_json_list(
            cls._record_value(episode, "evidence_spans_json")
        )
        raw_quotes = cls._v3_json_list(
            cls._record_value(episode, "evidence_quotes_json")
        )
        if not raw_spans or not raw_quotes:
            return None, "source_evidence_payload_missing"
        spans: list[tuple[int, int]] = []
        for raw_span in raw_spans:
            if (
                not isinstance(raw_span, (list, tuple))
                or len(raw_span) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_span
                )
            ):
                return None, "source_evidence_span_invalid"
            start, end = int(raw_span[0]), int(raw_span[1])
            # Importer evidence spans are inclusive 1-based line numbers.
            if start < 1 or end < start:
                return None, "source_evidence_span_out_of_range"
            if (start, end) not in spans:
                spans.append((start, end))
        if not spans:
            return None, "source_evidence_span_missing"
        quotes = [
            str(value).strip()
            for value in raw_quotes
            if isinstance(value, str) and str(value).strip()
        ]
        if not quotes:
            return None, "source_evidence_quote_missing"
        # ``reasoning_view_nonempty_lines_v1`` is the exact view supplied to
        # the extractor: compact the structured Source, then number only
        # nonblank lines. Earlier V3 code incorrectly treated those indices
        # as raw-Source line numbers, rejecting valid persisted evidence.
        # A small number of historical rows were labelled with this basis but
        # persisted raw-line coordinates. Preserve them only when their quote
        # still exactly reconstructs from that raw view; do not relax the
        # source-proof check or infer a new span.
        if evidence_basis == "reasoning_view_nonempty_lines_v1":
            coordinate_views = (
                (
                    MemoryExtractor._single_pass_source_lines(
                        MemoryExtractor.compact_source_for_reasoning(raw_source)
                    )[0],
                    "reasoning-view-nonempty-lines",
                ),
                (raw_source.splitlines(), "lines"),
            )
        else:
            coordinate_views = ((raw_source.splitlines(), "lines"),)
        lines: list[str] | None = None
        span_locator_prefix: str | None = None
        saw_in_range_view = False
        for candidate_lines, candidate_prefix in coordinate_views:
            if not candidate_lines:
                continue
            if any(end > len(candidate_lines) for _, end in spans):
                continue
            saw_in_range_view = True
            candidate_span_texts = [
                "\n".join(candidate_lines[start - 1 : end])
                for start, end in spans
            ]
            if all(
                any(
                    quote in span_text
                    for span_text in candidate_span_texts
                    if span_text
                )
                for quote in quotes
            ):
                lines = candidate_lines
                span_locator_prefix = candidate_prefix
                break
        if lines is None or span_locator_prefix is None:
            if not saw_in_range_view:
                return None, "source_evidence_span_out_of_range"
            return None, "source_evidence_quote_span_mismatch"

        episode_version, episode_basis, episode_error = cls._explicit_target_version(
            episode, label="episode"
        )
        source_version, source_basis, source_error = cls._explicit_target_version(
            source, label="source"
        )
        if episode_error is not None or source_error is not None:
            return None, "source_revision_invalid"
        if (
            episode_version
            and source_version
            and episode_basis.rsplit(".", 1)[-1]
            == source_basis.rsplit(".", 1)[-1]
            and episode_version != source_version
        ):
            return None, "source_revision_conflict"
        revision_marker = source_version or episode_version
        if not revision_marker:
            updated_at = cls._text_value(cls._record_value(episode, "updated_at"))
            try:
                revision_marker = normalize_evaluation_as_of(updated_at)
            except (TypeError, ValueError):
                return None, "source_revision_missing"

        source_key = cls._text_value(cls._record_value(episode, "source_key"))
        raw_source_hash = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()
        source_revision_material = json.dumps(
            {
                "source_id": source_id,
                "source_key_sha256": hashlib.sha256(
                    source_key.encode("utf-8")
                ).hexdigest(),
                "raw_source_sha256": raw_source_hash,
                "revision_marker": revision_marker,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        source_revision_id = "source-revision:sha256:" + hashlib.sha256(
            source_revision_material
        ).hexdigest()
        canonical_spans = tuple(sorted(spans))
        canonical_span_texts = [
            "\n".join(lines[start - 1 : end])
            for start, end in canonical_spans
        ]
        span_hash = "sha256:" + hashlib.sha256(
            json.dumps(canonical_spans, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        raw_span_hash = "sha256:" + hashlib.sha256(
            "\x1e".join(canonical_span_texts).encode("utf-8")
        ).hexdigest()
        try:
            return (
                SourceFactRef(
                    source_revision_id=source_revision_id,
                    record_span=(
                        f"source-row:{source_id}",
                        *(
                            f"{span_locator_prefix}:{start}-{end}"
                            for start, end in canonical_spans
                        ),
                    ),
                    span_hash=span_hash,
                    raw_span_hash=raw_span_hash,
                    # This local-only diagnostic is intentionally omitted from
                    # every contribution trace payload below.
                    source_key=source_key,
                ),
                "source_bound",
            )
        except (TypeError, ValueError):
            return None, "source_fact_construction_invalid"

    def _v3_source_fact_closure(
        self,
        episode_ids: Sequence[int],
        *,
        cache: dict[int, tuple[SourceFactRef | None, str]] | None = None,
    ) -> tuple[dict[int, SourceFactRef], dict[int, str]]:
        """Fetch source truth once for all request-local candidate Episodes.

        ``cache`` is intentionally request-local and retains the strict
        ``SourceFactRef`` revision identity alongside the decision. It avoids
        rescanning the same raw Source when a prepared candidate later reaches
        the contribution selector, without upgrading a metadata-only gate to
        a source-binding proof.
        """

        normalized_ids: set[int] = set()
        for value in episode_ids:
            try:
                episode_id = int(value)
            except (TypeError, ValueError):
                continue
            if episode_id > 0:
                normalized_ids.add(episode_id)
        ids = tuple(sorted(normalized_ids))
        if not ids:
            return {}, {}
        cached_ids = set(cache or {})
        missing_ids = tuple(episode_id for episode_id in ids if episode_id not in cached_ids)
        facts: dict[int, SourceFactRef] = {}
        reasons: dict[int, str] = {}
        if cache is not None:
            for episode_id in ids:
                cached = cache.get(episode_id)
                if cached is None:
                    continue
                fact, reason = cached
                reasons[episode_id] = reason
                if fact is not None:
                    facts[episode_id] = fact
        if not missing_ids:
            return facts, reasons
        try:
            episode_rows, source_rows = self._current_target_rows(missing_ids)
        except (AttributeError, TypeError, ValueError, KeyError):
            unavailable = {
                episode_id: "source_closure_unavailable" for episode_id in missing_ids
            }
            reasons.update(unavailable)
            if cache is not None:
                for episode_id, reason in unavailable.items():
                    cache[episode_id] = (None, reason)
            return facts, reasons
        for episode_id in missing_ids:
            episode = episode_rows.get(episode_id)
            try:
                source_id = int(self._record_value(episode, "source_id"))
            except (TypeError, ValueError):
                source_id = -1
            fact, reason = self._v3_source_fact_for_closure(
                episode_id=episode_id,
                episode=episode,
                source=source_rows.get(source_id),
            )
            reasons[episode_id] = reason
            if fact is not None:
                facts[episode_id] = fact
            if cache is not None:
                cache[episode_id] = (fact, reason)
        return facts, reasons

    @classmethod
    def _v3_contribution_trace_payload(
        cls,
        contribution: CandidateContribution,
        source_reason: str,
    ) -> dict[str, object]:
        """Serialize candidate provenance without source text, path, or key."""

        return {
            "contribution_id": contribution.contribution_id,
            "episode_id": int(contribution.episode_id),
            "slot_id": contribution.slot_id,
            "lane": contribution.lane,
            "edge_id": contribution.edge_id,
            "parent_contribution_ids": list(contribution.parent_contribution_ids),
            "query_ref": contribution.query_ref,
            "rank_features": dict(contribution.rank_features),
            "source_fact_ids": [fact.fact_id for fact in contribution.source_facts],
            "source_fact_status": source_reason,
            "source_span_refs": list(contribution.source_span_refs),
            "evidence_ref": contribution.evidence_ref,
            "clause_supports": [
                {
                    "slot_id": support.slot_id,
                    "clause_id": support.clause_id,
                    "support_mode": support.support_mode,
                    "verification_status": support.verification_status,
                    "source_fact_id": (
                        support.source_fact.fact_id
                        if support.source_fact is not None
                        else ""
                    ),
                }
                for support in contribution.clause_supports
            ],
            "ranking_only": not any(
                support.is_required_coverage
                for support in contribution.clause_supports
            ),
        }

    def _v3_build_candidate_contributions(
        self,
        *,
        episodes: Sequence[dict],
        slots: Sequence[EvidenceSlot],
        slot_support: dict[int, set[str]],
        contextual_hits: Sequence[ContextualSlotHit] = (),
        target_relevance_scores: dict[tuple[str, int], float] | None = None,
        base_episode_ids: Sequence[int] | None = None,
        source_fact_cache: dict[int, tuple[SourceFactRef | None, str]] | None = None,
    ) -> tuple[list[CandidateContribution], dict[int, str], list[dict[str, object]]]:
        """Preserve all base and accepted contextual routes before merging.

        The only path allowed to make ``source_bound`` support is an already
        existing base ``slot_support`` mapping plus a current full-source fact
        closure.  Contextual matching itself remains ``relevance_only`` even
        when the target's source provenance is valid.  ``base_episode_ids`` is
        a frozen pre-context endpoint manifest: a target materialized solely
        because an edge pointed to it may receive a contextual contribution,
        but it must not acquire a synthetic base rerank/mapping route that
        would survive a later edge mask.
        """

        slot_by_id = {str(slot.slot_id): slot for slot in slots}
        episode_rows = [
            item for item in episodes if isinstance(item, dict) and int(item.get("id", 0) or 0) > 0
        ]
        materialized_episode_ids = {int(item["id"]) for item in episode_rows}
        if base_episode_ids is None:
            normalized_base_episode_ids = set(materialized_episode_ids)
        else:
            normalized_base_episode_ids: set[int] = set()
            for raw_episode_id in base_episode_ids:
                try:
                    episode_id = int(raw_episode_id)
                except (TypeError, ValueError):
                    continue
                if episode_id > 0 and episode_id in materialized_episode_ids:
                    normalized_base_episode_ids.add(episode_id)
        facts, source_reasons = self._v3_source_fact_closure(
            [int(item["id"]) for item in episode_rows],
            cache=source_fact_cache,
        )
        contributions: list[CandidateContribution] = []
        trace_rows: list[dict[str, object]] = []

        def append(
            item: CandidateContribution,
            episode_id: int,
            *,
            endpoint_provenance: Literal["initial_base", "contextual_edge"],
        ) -> None:
            contributions.append(item)
            trace_payload = self._v3_contribution_trace_payload(
                item,
                source_reasons.get(episode_id, "source_closure_missing"),
            )
            trace_payload["endpoint_provenance"] = endpoint_provenance
            trace_rows.append(trace_payload)

        for occurrence, episode in enumerate(episode_rows, start=1):
            episode_id = int(episode["id"])
            if episode_id not in normalized_base_episode_ids:
                continue
            fact = facts.get(episode_id)
            source_refs = (fact,) if fact is not None else ()
            evidence_ref = fact.fact_id if fact is not None else ""
            score = self._v3_finite_score(episode.get("score", 0.0))
            rank_contribution = CandidateContribution(
                contribution_id=self._v3_opaque_ref(
                    "v3-base-rank", episode_id, occurrence
                ),
                episode_id=episode_id,
                slot_id="__base_rank__",
                lane="base_rerank",
                query_ref=self._v3_opaque_ref("base-rerank", occurrence),
                rank_features={"fusion_rank_score": score},
                source_facts=source_refs,
                source_span_refs=(evidence_ref,) if evidence_ref else (),
                evidence_ref=evidence_ref,
            )
            append(
                rank_contribution,
                episode_id,
                endpoint_provenance="initial_base",
            )

            # Do not infer a requirement mapping from a score.  These are the
            # exact pre-existing mapping rows from the frozen rerank/floor
            # contract.  A malformed source closure leaves them visible but
            # relevance-only, rather than turning them into facts.
            for slot_id in sorted(
                str(value) for value in slot_support.get(episode_id, set()) if str(value)
            ):
                slot = slot_by_id.get(slot_id)
                supports: tuple[ClauseSupport, ...] = ()
                if slot is not None:
                    status = "source_bound" if fact is not None else "relevance_only"
                    supports = tuple(
                        ClauseSupport(
                            slot_id=slot.slot_id,
                            clause_id=clause_id,
                            source_fact=fact,
                            support_mode=slot.support_mode,
                            verification_status=status,
                            mapping_ref=self._v3_opaque_ref(
                                "existing-source-mapping",
                                episode_id,
                                slot.slot_id,
                                clause_id,
                                fact.fact_id if fact is not None else "missing",
                            ),
                        )
                        for clause_id in slot.clause_ids
                    )
                mapped = CandidateContribution(
                    contribution_id=self._v3_opaque_ref(
                        "v3-base-mapping", episode_id, occurrence, slot_id
                    ),
                    episode_id=episode_id,
                    slot_id=slot_id,
                    lane="base_source_mapping",
                    query_ref=self._v3_opaque_ref("base-mapping", slot_id),
                    rank_features={"fusion_rank_score": score},
                    source_facts=source_refs,
                    clause_supports=supports,
                    source_span_refs=(evidence_ref,) if evidence_ref else (),
                    evidence_ref=evidence_ref,
                )
                append(
                    mapped,
                    episode_id,
                    endpoint_provenance="initial_base",
                )

        # Source identity, rather than an edge/query/cue occurrence, defines
        # an independent local activation.  Read current anchor provenance in
        # one batch; missing Source closure leaves a conservative legacy max.
        anchor_source_ids: dict[int, int] = {}
        try:
            anchor_rows, anchor_sources = self._current_target_rows(
                [int(hit.anchor_episode_id) for hit in contextual_hits]
            )
        except (AttributeError, TypeError, ValueError, KeyError):
            anchor_rows, anchor_sources = {}, {}
        for anchor_id, anchor_row in anchor_rows.items():
            try:
                source_id = int(self._record_value(anchor_row, "source_id"))
            except (TypeError, ValueError):
                continue
            if 0 < source_id <= (2**53 - 1) and source_id in anchor_sources:
                anchor_source_ids[anchor_id] = source_id

        relevance_scores = target_relevance_scores or {}
        for occurrence, hit in enumerate(contextual_hits, start=1):
            episode_id = int(hit.target_episode_id)
            if episode_id not in materialized_episode_ids:
                # A target can be source-valid yet unavailable to answer
                # delivery if materialization failed.  Record it explicitly;
                # do not select an Episode for which no answer context exists.
                trace_rows.append(
                    {
                        "contribution_id": self._v3_opaque_ref(
                            "v3-contextual-undeliverable",
                            hit.association_id,
                            episode_id,
                            hit.matched_slot_id,
                            occurrence,
                        ),
                        "episode_id": episode_id,
                        "slot_id": str(hit.matched_slot_id),
                        "lane": "contextual",
                        "edge_id": int(hit.association_id),
                        "ranking_only": True,
                        "delivery_status": "target_not_materialized",
                        "endpoint_provenance": "contextual_edge",
                    }
                )
                continue
            slot_id = str(hit.matched_slot_id)
            slot = slot_by_id.get(slot_id)
            fact = facts.get(episode_id)
            source_refs = (fact,) if fact is not None else ()
            evidence_ref = fact.fact_id if fact is not None else ""
            relevance = self._v3_finite_score(
                relevance_scores.get((slot_id, episode_id), hit.target_support_score)
            )
            contextual_score = self._v3_finite_score(hit.total_score)
            # The target gate already produced a request-local, normalised
            # target relevance estimate.  It may change inspection order only;
            # the contribution below remains relevance-only and cannot cover a
            # factual requirement.  Keeping it as a distinct feature lets an
            # edge mask remove this path while preserving an overlapping base
            # retrieval contribution for the same Episode.
            contextual_priority = min(1.0, max(0.0, relevance))
            supports: tuple[ClauseSupport, ...] = ()
            if slot is not None:
                # The double-key match says "inspect this target for this
                # requirement", not "this target proves the requirement".
                # Keep that valuable signal visible as a relevance-only
                # mapping even when the source fact itself is well formed.
                supports = tuple(
                    ClauseSupport(
                        slot_id=slot.slot_id,
                        clause_id=clause_id,
                        source_fact=fact,
                        support_mode=slot.support_mode,
                        verification_status="relevance_only",
                        mapping_ref=self._v3_opaque_ref(
                            "contextual-ranking-only",
                            hit.association_id,
                            episode_id,
                            slot.slot_id,
                            clause_id,
                        ),
                    )
                    for clause_id in slot.clause_ids
                )
            contextual = CandidateContribution(
                contribution_id=self._v3_opaque_ref(
                    "v3-contextual",
                    hit.association_id,
                    episode_id,
                    slot_id,
                    hit.matched_query_id,
                    occurrence,
                ),
                episode_id=episode_id,
                slot_id=slot_id or "__contextual_rank__",
                lane="contextual",
                edge_id=int(hit.association_id),
                query_ref=self._v3_opaque_ref(
                    "contextual-query", hit.matched_query_id, slot_id
                ),
                rank_features={
                    "current_relevance_estimate": relevance,
                    "contextual_combined_score": contextual_score,
                    "contextual_priority_score": contextual_priority,
                    "contextual_anchor_episode_id": int(hit.anchor_episode_id),
                    "contextual_anchor_source_id": anchor_source_ids.get(
                        int(hit.anchor_episode_id)
                    ),
                },
                source_facts=source_refs,
                clause_supports=supports,
                source_span_refs=(evidence_ref,) if evidence_ref else (),
                evidence_ref=evidence_ref,
            )
            append(
                contextual,
                episode_id,
                endpoint_provenance="contextual_edge",
            )
        return contributions, source_reasons, trace_rows

    @staticmethod
    def _v3_selection_trace_payload(selection) -> dict[str, object]:
        """Trace the pure selector result with ids and numeric values only."""

        return {
            "selection_strategy": selection.selection_strategy,
            "budget": {
                "episode_limit": int(selection.budget.episode_limit),
                "source_fact_limit": selection.budget.source_fact_limit,
                "delivery_token_limit": selection.budget.delivery_token_limit,
            },
            "selected_episode_ids": list(selection.selected_episode_ids),
            "selected_contribution_ids": list(selection.selected_contribution_ids),
            "covered_required_clauses": sorted(selection.covered_required_clauses),
            "missing_required_clauses": sorted(selection.missing_required_clauses),
            "covered_optional_clauses": sorted(selection.covered_optional_clauses),
            "incomplete_joint_clauses": sorted(selection.incomplete_joint_clauses),
            "budget_exhausted": bool(selection.budget_exhausted),
            "episode_budget_exhausted": bool(selection.episode_budget_exhausted),
            "source_fact_budget_exhausted": bool(selection.source_fact_budget_exhausted),
            "delivery_token_budget_exhausted": bool(
                selection.delivery_token_budget_exhausted
            ),
            "delivery_loss": bool(selection.delivery_loss),
            "delivery_loss_clauses": sorted(selection.delivery_loss_clauses),
            "actual_delivery_episode_count": int(
                selection.actual_delivery_episode_count
            ),
            "actual_delivery_source_fact_count": int(
                selection.actual_delivery_source_fact_count
            ),
            "actual_delivery_token_cost": int(
                selection.actual_delivery_token_cost
            ),
            "stop_reason": selection.stop_reason,
            "decisions": [
                {
                    "iteration": int(decision.iteration),
                    "chosen_episode_id": decision.chosen_episode_id,
                    "chosen_contribution_ids": list(
                        decision.chosen_contribution_ids
                    ),
                    "covered_required_before": list(
                        decision.covered_required_before
                    ),
                    "newly_covered_required_clauses": list(
                        decision.newly_covered_required_clauses
                    ),
                    "joint_incomplete_clauses": list(
                        decision.joint_incomplete_clauses
                    ),
                    "remaining_episode_budget": int(
                        decision.remaining_episode_budget
                    ),
                    "remaining_source_fact_budget": (
                        decision.remaining_source_fact_budget
                    ),
                    "remaining_delivery_token_budget": (
                        decision.remaining_delivery_token_budget
                    ),
                    "score_components": dict(decision.score_components),
                    "stop_reason": decision.stop_reason,
                }
                for decision in selection.decisions
            ],
        }

    @staticmethod
    def _v3_selection_delta(masked, treatment) -> dict[str, object]:
        """Compare V3 selector states without reviving edge-level masking."""

        masked_covered = set(masked.covered_clauses)
        treatment_covered = set(treatment.covered_clauses)
        new_slots = treatment_covered - masked_covered
        lost_slots = masked_covered - treatment_covered
        return {
            "treatment_episode_ids": list(treatment.selected_episode_ids),
            "masked_episode_ids": list(masked.selected_episode_ids),
            "new_slots": sorted(new_slots),
            "new_slot_count": len(new_slots),
            "lost_slots": sorted(lost_slots),
            "harm": bool(lost_slots),
        }

    @classmethod
    def _v3_counterfactual_trace_payload(cls, attribution) -> dict[str, object]:
        """Serialize T11 masks without source paths, text, or cue internals.

        The pure contribution selector already preserves all base and
        contextual routes.  This adapter exposes only opaque fingerprints,
        contribution/edge identifiers, selection receipts, and clause deltas
        so a strict trace can prove that every treatment/mask replay used the
        same frozen universe and budget without copying source material.
        """

        edge_rows: list[dict[str, object]] = []
        for edge in attribution.edges:
            edge_rows.append(
                {
                    "edge_id": int(edge.edge_id),
                    "contribution_ids": list(edge.contribution_ids),
                    "selected_in_treatment": bool(edge.selected_in_treatment),
                    "sufficient": bool(edge.sufficient),
                    "necessary": bool(edge.necessary),
                    "harmful": bool(edge.harmful),
                    "classification": edge.classification,
                    "single_edge": edge.single_edge.as_dict(),
                    "leave_one_out": edge.leave_one_out.as_dict(),
                }
            )
        return {
            "candidate_universe_fingerprint": attribution.candidate_universe_fingerprint,
            "requirements_fingerprint": attribution.requirements_fingerprint,
            "budget_fingerprint": attribution.budget_fingerprint,
            "input_fingerprint": attribution.input_fingerprint,
            "treatment": cls._v3_selection_trace_payload(attribution.treatment),
            "masked": cls._v3_selection_trace_payload(attribution.masked),
            "mask_evaluation": attribution.treatment_vs_masked.as_dict(),
            "edges": edge_rows,
        }

    @classmethod
    def _v3_record_counterfactual_trace(cls, trace: dict, attribution) -> None:
        """Attach one canonical T11 receipt and retain legacy V3 keys."""

        masked = attribution.masked
        treatment = attribution.treatment
        trace["contribution_counterfactual"] = cls._v3_counterfactual_trace_payload(
            attribution
        )
        trace["masked_selector"] = cls._v3_selection_trace_payload(masked)
        trace["treatment_selector"] = cls._v3_selection_trace_payload(treatment)
        trace["masked_selected_episode_ids"] = list(masked.selected_episode_ids)
        trace["masked_missing_slots"] = sorted(masked.missing_required_clauses)
        trace["treatment_selected_episode_ids"] = list(
            treatment.selected_episode_ids
        )
        trace["selected_count"] = len(treatment.selected_episode_ids)
        # Historical readers consume new/lost slot keys.  Preserve their
        # combined required/optional semantics while the richer T11 delta is
        # available under ``contribution_counterfactual.mask_evaluation``.
        trace.update(cls._v3_selection_delta(masked, treatment))

    @staticmethod
    def _v3_selected_episode_rows(
        selection,
        episodes: Sequence[dict],
    ) -> list[dict]:
        """Materialize exactly the rows the selector recorded as delivered."""

        by_id = {
            int(item["id"]): item
            for item in episodes
            if isinstance(item, dict) and int(item.get("id", 0) or 0) > 0
        }
        selected = [
            by_id[episode_id]
            for episode_id in selection.selected_episode_ids
            if episode_id in by_id
        ]
        # Do not silently substitute the pre-V3 baseline here: that would
        # make delivered answer evidence diverge from the selector receipt.
        # A non-materialized selected id is already prevented when contextual
        # contributions are built; this list is therefore the exact delivery
        # set represented by ``selected_episode_ids``.
        return selected

    def _capture_v3_learning_selection(
        self,
        *,
        slots: Sequence[EvidenceSlot],
        contributions: Sequence[CandidateContribution],
        bundle: QueryVectorBundle | None,
        initial_candidate_episode_ids: Sequence[int],
        independent_base_episode_ids: Sequence[int],
        initial_delivered_episode_ids: Sequence[int],
        final_selected_episode_ids: Sequence[int],
        contextual_expansion_episode_ids: Sequence[int],
        cue_endpoint_episode_ids: Sequence[int],
        missing_required_clauses: Sequence[str],
        delivery_loss: bool,
        shadow: bool,
        target_gate_completed: bool,
        direct_base_validation_completed: bool,
        requirements: RequirementResolution | None = None,
        endpoint_limit: int | None = None,
    ) -> None:
        """Keep runtime-only selector provenance for a possible T13 action."""

        def normalized_ids(values: Sequence[int]) -> tuple[int, ...]:
            return tuple(
                sorted(
                    {
                        int(value)
                        for value in values
                        if int(value) > 0
                    }
                )
            )

        try:
            normalized_endpoint_limit = (
                int(endpoint_limit) if endpoint_limit is not None else None
            )
        except (TypeError, ValueError):
            normalized_endpoint_limit = None
        if normalized_endpoint_limit is not None and normalized_endpoint_limit <= 0:
            normalized_endpoint_limit = None

        self._v3_learning_capture = _V3LearningCapture(
            slots=tuple(slots),
            contributions=tuple(contributions),
            bundle=bundle,
            initial_candidate_episode_ids=normalized_ids(
                initial_candidate_episode_ids
            ),
            independent_base_episode_ids=normalized_ids(
                independent_base_episode_ids
            ),
            initial_delivered_episode_ids=normalized_ids(
                initial_delivered_episode_ids
            ),
            final_selected_episode_ids=normalized_ids(final_selected_episode_ids),
            contextual_expansion_episode_ids=normalized_ids(
                contextual_expansion_episode_ids
            ),
            cue_endpoint_episode_ids=normalized_ids(cue_endpoint_episode_ids),
            missing_required_clauses=tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in missing_required_clauses
                        if str(value).strip()
                    }
                )
            ),
            delivery_loss=bool(delivery_loss),
            shadow=bool(shadow),
            target_gate_completed=bool(target_gate_completed),
            direct_base_validation_completed=bool(
                direct_base_validation_completed
            ),
            requirements=(
                requirements
                if isinstance(requirements, RequirementResolution)
                else None
            ),
            endpoint_limit=normalized_endpoint_limit,
        )

    def _select_contextual_slots_v3(
        self,
        *,
        episodes: list[dict],
        slots: list[EvidenceSlot],
        slot_support: dict[int, set[str]],
        bundle: QueryVectorBundle | None,
        domain: str | None,
        endpoint_limit: int | None,
        anchor_activations: dict[int, float],
        authoritative_requirements: RequirementResolution,
        evaluation_as_of: str | None,
        cue_endpoint_episode_ids: Sequence[int] = (),
        learning_initial_candidate_episode_ids: Sequence[int] | None = None,
        learning_independent_base_episode_ids: Sequence[int] | None = None,
        learning_initial_delivered_episode_ids: Sequence[int] | None = None,
        prepared_early_contextual_episode_ids: Sequence[int] = (),
        prepared_early_contextual_derived_episode_ids: Sequence[int] = (),
        source_fact_cache: dict[int, tuple[SourceFactRef | None, str]] | None = None,
    ) -> tuple[list[dict], dict]:
        """Run the V3 request-local contribution treatment/masked comparison."""

        budget = EvidenceSelectionBudget(
            # This is deliberately the configured request budget, never the
            # pre-treatment pool size.  Contextual recovery may legitimately
            # add deliverable Episodes up to this limit.
            episode_limit=max(0, int(self.config.retrieval.answer_episode_limit)),
        )
        # ``base_route_episode_ids`` is the complete pre-context candidate
        # universe used by the selector.  It must stay distinct from the
        # earlier query snapshot used to explain why an eventual target was
        # missing or dropped: graph expansion can legitimately add a new,
        # independently source-bound base route after that earlier snapshot.
        prepared_early_contextual_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in prepared_early_contextual_episode_ids
                    if int(value) > 0
                }
            )
        )
        prepared_early_contextual_derived_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in prepared_early_contextual_derived_episode_ids
                    if int(value) > 0
                }
            )
        )
        prepared_early_contextual_id_set = set(
            prepared_early_contextual_ids
        ).union(prepared_early_contextual_derived_ids)
        base_route_episode_ids = tuple(
            sorted(
                int(item["id"])
                for item in episodes
                if (
                    isinstance(item, dict)
                    and int(item.get("id", 0) or 0) > 0
                    and int(item["id"]) not in prepared_early_contextual_id_set
                )
            )
        )
        candidate_snapshot_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in (
                        learning_initial_candidate_episode_ids
                        if learning_initial_candidate_episode_ids is not None
                        else base_route_episode_ids
                    )
                    if int(value) > 0
                }
            )
        )
        candidate_snapshot_set = set(candidate_snapshot_ids)
        # A materialized pre-context row may have arrived through graph
        # traversal or an association cue.  It is useful evidence, but it is
        # not evidence that the row was independently retrieved.  Do not
        # infer direct-base provenance from the candidate universe.
        independent_base_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in (learning_independent_base_episode_ids or ())
                    if int(value) > 0
                }
            )
        )
        delivered_snapshot_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in (
                        learning_initial_delivered_episode_ids
                        if learning_initial_delivered_episode_ids is not None
                        else ()
                    )
                    if int(value) > 0 and int(value) in candidate_snapshot_set
                }
            )
        )
        base_contributions, source_reasons, base_trace_rows = (
            self._v3_build_candidate_contributions(
                episodes=episodes,
                slots=slots,
                slot_support=slot_support,
                base_episode_ids=base_route_episode_ids,
                source_fact_cache=source_fact_cache,
            )
        )
        masked_aggregates = aggregate_contributions(base_contributions)
        # Even before a contextual match is attempted, create the T11 receipt
        # through the same contribution universe/selector.  With no edge it
        # is a deterministic treatment==masked no-op, rather than a special
        # legacy baseline path.
        base_attribution = contribution_contextual_attribution(
            base_contributions,
            slots,
            budget,
        )
        initial_masked = base_attribution.masked
        initial_masked_episodes = self._v3_selected_episode_rows(
            initial_masked,
            episodes,
        )
        trace: dict[str, object] = {
            "enabled": bool(self.config.retrieval.contextual_association_enabled),
            "backend": "contextual_double_key_contribution_selector_v3",
            "slots": [
                {
                    "slot_id": item.slot_id,
                    "question": item.question,
                    "required": item.required,
                    "query_refs": list(item.query_refs),
                    "origin": item.origin,
                    "support_mode": item.support_mode,
                    "clause_ids": list(item.clause_ids),
                }
                for item in slots
            ],
            "requirements_status": authoritative_requirements.status,
            "authoritative_requirements": authoritative_requirements.as_trace_payload(),
            # This is the frozen pre-context endpoint manifest.  It is not a
            # source locator and lets a later mask receipt distinguish a
            # genuine base route from an endpoint introduced by an edge.
            "base_endpoint_manifest": list(base_route_episode_ids),
            "prepared_early_contextual_endpoint_manifest": list(
                prepared_early_contextual_ids
            ),
            "prepared_early_contextual_derived_endpoint_manifest": list(
                prepared_early_contextual_derived_ids
            ),
            "independent_base_endpoint_manifest": list(independent_base_ids),
            "masked_selected_episode_ids": list(initial_masked.selected_episode_ids),
            "masked_missing_slots": sorted(initial_masked.missing_required_clauses),
            "merged_candidates": base_trace_rows,
            "merged_aggregates": [
                {
                    "episode_id": aggregate.episode_id,
                    "contribution_ids": list(aggregate.contribution_ids),
                    "source_fact_ids": [
                        fact.fact_id for fact in aggregate.source_facts
                    ],
                }
                for aggregate in masked_aggregates
            ],
            "masked_selector": self._v3_selection_trace_payload(initial_masked),
            "source_provenance_status": [
                {"episode_id": episode_id, "status": reason}
                for episode_id, reason in sorted(source_reasons.items())
            ],
            "hits": [],
            "context_hits": [],
            "need_hits": [],
            "attached_edges": [],
            "attached_episode_ids": [],
            "external_calls": 0,
        }
        if bundle is not None:
            trace.update(self._query_vector_trace_metadata(bundle))
        if (
            not self.config.retrieval.contextual_association_enabled
            or self.contextual_matcher is None
            or bundle is None
            or not initial_masked.missing_required_clauses
        ):
            trace["reason"] = (
                "no_unresolved_slots"
                if not initial_masked.missing_required_clauses
                else "contextual_disabled_or_no_bundle"
            )
            self._v3_record_counterfactual_trace(trace, base_attribution)
            trace["selected_contextual_contribution_ids"] = []
            trace["compatibility_projection"] = "v3_contribution_selector"
            self._capture_v3_learning_selection(
                slots=slots,
                contributions=base_contributions,
                bundle=bundle,
                initial_candidate_episode_ids=candidate_snapshot_ids,
                independent_base_episode_ids=independent_base_ids,
                initial_delivered_episode_ids=(
                    delivered_snapshot_ids
                    if learning_initial_delivered_episode_ids is not None
                    else initial_masked.selected_episode_ids
                ),
                final_selected_episode_ids=initial_masked.selected_episode_ids,
                contextual_expansion_episode_ids=(),
                cue_endpoint_episode_ids=cue_endpoint_episode_ids,
                missing_required_clauses=initial_masked.missing_required_clauses,
                delivery_loss=bool(initial_masked.delivery_loss),
                shadow=False,
                # No target gate ran on this short-circuit path.  A future
                # learning action must not treat it as a V3 repair result.
                target_gate_completed=False,
                # A fully covered V3 base selection has a separate direct
                # validation meaning.  It never validates a contextual
                # target and is accepted later only for direct-base targets.
                direct_base_validation_completed=bool(
                    not initial_masked.missing_required_clauses
                    and not initial_masked.delivery_loss
                ),
                requirements=authoritative_requirements,
                endpoint_limit=endpoint_limit,
            )
            return initial_masked_episodes, trace

        unresolved_slots = [
            item
            for item in slots
            if item.slot_id in initial_masked.missing_required_clauses
        ]
        request_evaluation_as_of = self._resolve_contextual_evaluation_as_of(
            evaluation_as_of
        )
        preliminary = self.contextual_recall(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=anchor_activations,
            unresolved_slot_ids=[item.slot_id for item in unresolved_slots],
            evaluation_as_of=request_evaluation_as_of,
        )
        # Keep matcher metadata but rebuild selection-owned fields below.
        trace.update(preliminary)
        # ``contextual_recall`` reports the matcher backend for compatibility;
        # the request receipt must identify the outer contribution selector so
        # downstream readers do not mistake a V3 selection for old strict-edge
        # attribution.
        trace["matcher_backend"] = str(preliminary.get("backend", ""))
        trace["backend"] = "contextual_double_key_contribution_selector_v3"
        trace["enabled"] = bool(self.config.retrieval.contextual_association_enabled)
        raw_proposals = trace.get("pre_target_proposals", [])
        if not isinstance(raw_proposals, (list, tuple)) or not all(
            isinstance(item, ContextualPreTargetProposal)
            for item in raw_proposals
        ):
            trace["hits"] = []
            trace["attached_edges"] = []
            trace["attached_episode_ids"] = []
            trace["reason"] = "invalid_pre_target_proposals"
            trace["target_gate"] = {
                "stage": "target_checked_before_endpoint_cap_v1",
                "reason": "invalid_pre_target_proposals",
                "outcomes": [],
            }
            self._v3_record_counterfactual_trace(trace, base_attribution)
            trace["selected_contextual_contribution_ids"] = []
            trace["compatibility_projection"] = "v3_contribution_selector"
            self._capture_v3_learning_selection(
                slots=slots,
                contributions=base_contributions,
                bundle=bundle,
                initial_candidate_episode_ids=candidate_snapshot_ids,
                independent_base_episode_ids=independent_base_ids,
                initial_delivered_episode_ids=(
                    delivered_snapshot_ids
                    if learning_initial_delivered_episode_ids is not None
                    else initial_masked.selected_episode_ids
                ),
                final_selected_episode_ids=initial_masked.selected_episode_ids,
                contextual_expansion_episode_ids=(),
                cue_endpoint_episode_ids=cue_endpoint_episode_ids,
                missing_required_clauses=initial_masked.missing_required_clauses,
                delivery_loss=bool(initial_masked.delivery_loss),
                shadow=False,
                target_gate_completed=False,
                direct_base_validation_completed=False,
                requirements=authoritative_requirements,
                endpoint_limit=endpoint_limit,
            )
            return initial_masked_episodes, trace

        gate_result = self._gate_contextual_targets(
            proposals=raw_proposals,
            bundle=bundle,
            unresolved_slots=unresolved_slots,
            endpoint_limit=endpoint_limit,
            evaluation_as_of=request_evaluation_as_of,
        )
        target_relevance = {
            (item.matched_slot_id, int(item.target_episode_id)): float(
                item.relevance_estimate
            )
            for item in gate_result.outcomes
            if item.accepted
            and item.relevance_estimate is not None
            and math.isfinite(float(item.relevance_estimate))
        }
        trace["hits"] = list(gate_result.hits)
        trace["attached_episode_ids"] = [
            int(item.target_episode_id) for item in gate_result.hits
        ]
        trace["attached_edges"] = [
            int(item.association_id) for item in gate_result.hits
        ]
        trace["target_gate"] = gate_result.as_trace_payload()
        trace["target_checks"] = [
            item.as_trace_payload() for item in gate_result.outcomes
        ]
        trace["evaluation_as_of"] = request_evaluation_as_of
        existing = {int(item["id"]) for item in episodes}
        new_nodes = [
            TraversedNode(
                "episode", int(hit.target_episode_id), float(hit.target_support_score)
            )
            for hit in gate_result.hits
            if int(hit.target_episode_id) not in existing
        ]
        if new_nodes:
            recovered, _ = self._materialize_nodes(new_nodes, include_sources=True)
            episodes = [*episodes, *recovered]

        treatment_contributions, treatment_reasons, treatment_trace_rows = (
            self._v3_build_candidate_contributions(
                episodes=episodes,
                slots=slots,
                slot_support=slot_support,
                contextual_hits=gate_result.hits,
                target_relevance_scores=target_relevance,
                base_episode_ids=base_route_episode_ids,
                source_fact_cache=source_fact_cache,
            )
        )
        treatment_aggregates = aggregate_contributions(treatment_contributions)
        # Treatment, masked, every single-edge, and every leave-one-out replay
        # now start from this exact post-materialization contribution universe.
        # In particular, removing an edge rebuilds only its contribution and
        # never deletes an independent base route to the same Episode.
        attribution = contribution_contextual_attribution(
            treatment_contributions,
            slots,
            budget,
        )
        treatment = attribution.treatment
        masked = attribution.masked
        masked_episodes = self._v3_selected_episode_rows(
            masked,
            episodes,
        )
        treatment_episodes = self._v3_selected_episode_rows(
            treatment,
            episodes,
        )
        self._v3_record_counterfactual_trace(trace, attribution)
        trace["merged_candidates"] = treatment_trace_rows
        trace["merged_aggregates"] = [
            {
                "episode_id": aggregate.episode_id,
                "contribution_ids": list(aggregate.contribution_ids),
                "source_fact_ids": [fact.fact_id for fact in aggregate.source_facts],
            }
            for aggregate in treatment_aggregates
        ]
        trace["source_provenance_status"] = [
            {"episode_id": episode_id, "status": reason}
            for episode_id, reason in sorted(treatment_reasons.items())
        ]
        trace["selected_contextual_contribution_ids"] = [
            contribution.contribution_id
            for aggregate in treatment.selected
            for contribution in aggregate.contributions
            if contribution.is_contextual
        ]
        trace["ranking_only_contextual_contribution_ids"] = [
            contribution.contribution_id
            for aggregate in treatment_aggregates
            for contribution in aggregate.contributions
            if contribution.is_contextual
            and not any(
                support.is_required_coverage
                for support in contribution.clause_supports
            )
        ]
        # T11 will add per-contribution single-edge and leave-one-out masks.
        # Do not recreate the old one-edge-per-Episode attribution here.
        trace["strict_attribution"] = []
        trace["compatibility_projection"] = "v3_contribution_selector"
        self._capture_v3_learning_selection(
            slots=slots,
            contributions=treatment_contributions,
            bundle=bundle,
            initial_candidate_episode_ids=candidate_snapshot_ids,
            independent_base_episode_ids=independent_base_ids,
            initial_delivered_episode_ids=(
                delivered_snapshot_ids
                if learning_initial_delivered_episode_ids is not None
                else initial_masked.selected_episode_ids
            ),
            final_selected_episode_ids=treatment.selected_episode_ids,
            contextual_expansion_episode_ids=tuple(
                int(item.target_episode_id) for item in gate_result.hits
            ),
            cue_endpoint_episode_ids=cue_endpoint_episode_ids,
            missing_required_clauses=treatment.missing_required_clauses,
            delivery_loss=bool(treatment.delivery_loss),
            shadow=bool(self.config.retrieval.contextual_association_shadow),
            target_gate_completed=True,
            direct_base_validation_completed=False,
            requirements=authoritative_requirements,
            endpoint_limit=endpoint_limit,
        )
        if self.config.retrieval.contextual_association_shadow:
            trace["shadow"] = True
            return masked_episodes, trace
        trace["shadow"] = False
        return treatment_episodes, trace

    def _select_contextual_slots(
        self,
        *,
        episodes: list[dict],
        baseline_selected: list[dict],
        rerank_trace: dict,
        reranked_episode_ids: list[int],
        bundle: QueryVectorBundle | None,
        domain: str | None,
        endpoint_limit: int | None,
        anchor_activations: dict[int, float],
        authoritative_requirements: RequirementResolution | None = None,
        evaluation_as_of: str | None = None,
        cue_endpoint_episode_ids: Sequence[int] = (),
        learning_initial_candidate_episode_ids: Sequence[int] | None = None,
        learning_independent_base_episode_ids: Sequence[int] | None = None,
        learning_initial_delivered_episode_ids: Sequence[int] | None = None,
        prepared_early_contextual_episode_ids: Sequence[int] = (),
        prepared_early_contextual_derived_episode_ids: Sequence[int] = (),
        source_fact_cache: dict[int, tuple[SourceFactRef | None, str]] | None = None,
    ) -> tuple[list[dict], dict]:
        """Run masked/treatment set-cover without changing factual evidence rules."""
        base_support_mapping_status = self._rerank_slot_mapping_status(
            rerank_trace
        )
        slots, slot_support = self._request_evidence_slots(
            rerank_trace,
            bundle,
            (
                authoritative_requirements.requirements
                if authoritative_requirements is not None
                else None
            ),
        )
        if not slots:
            empty_trace = {
                "enabled": False,
                "reason": (
                    f"requirements_{authoritative_requirements.status}"
                    if authoritative_requirements is not None
                    else "no_evidence_slots"
                ),
                "requirements_status": (
                    authoritative_requirements.status
                    if authoritative_requirements is not None
                    else "legacy_unresolved"
                ),
                "authoritative_requirements": (
                    authoritative_requirements.as_trace_payload()
                    if authoritative_requirements is not None
                    else None
                ),
                "base_support_mapping_status": base_support_mapping_status,
                "hits": [],
                "external_calls": 0,
            }
            if bundle is not None:
                empty_trace.update(self._query_vector_trace_metadata(bundle))
            return baseline_selected, empty_trace
        if authoritative_requirements is not None:
            # V3 callers provide a request-frozen requirement denominator.
            # Route them through contribution aggregation rather than the
            # legacy flat SlotCandidate/strict-edge attribution selector.
            selected, trace = self._select_contextual_slots_v3(
                episodes=episodes,
                slots=slots,
                slot_support=slot_support,
                bundle=bundle,
                domain=domain,
                endpoint_limit=endpoint_limit,
                anchor_activations=anchor_activations,
                authoritative_requirements=authoritative_requirements,
                evaluation_as_of=evaluation_as_of,
                cue_endpoint_episode_ids=cue_endpoint_episode_ids,
                learning_initial_candidate_episode_ids=(
                    learning_initial_candidate_episode_ids
                ),
                learning_independent_base_episode_ids=(
                    learning_independent_base_episode_ids
                ),
                learning_initial_delivered_episode_ids=(
                    learning_initial_delivered_episode_ids
                ),
                prepared_early_contextual_episode_ids=(
                    prepared_early_contextual_episode_ids
                ),
                prepared_early_contextual_derived_episode_ids=(
                    prepared_early_contextual_derived_episode_ids
                ),
                source_fact_cache=source_fact_cache,
            )
            trace["base_support_mapping_status"] = base_support_mapping_status
            return selected, trace
        base_candidates = self._slot_candidates(
            episodes, slot_support, reranked_episode_ids
        )
        budget = min(self.config.retrieval.answer_episode_limit, len(episodes))
        masked = select_evidence(base_candidates, slots, budget)
        masked_by_id = {int(item["id"]): item for item in episodes}
        masked_episodes = [
            masked_by_id[item.episode_id]
            for item in masked.selected
            if item.episode_id in masked_by_id
        ] or baseline_selected
        trace: dict = {
            "enabled": bool(self.config.retrieval.contextual_association_enabled),
            "backend": "contextual_double_key_slot_selector_v2",
            "slots": [
                {
                    "slot_id": item.slot_id,
                    "question": item.question,
                    "required": item.required,
                    "query_refs": list(item.query_refs),
                    "origin": item.origin,
                    "support_mode": item.support_mode,
                    "clause_ids": list(item.clause_ids),
                }
                for item in slots
            ],
            "requirements_status": (
                authoritative_requirements.status
                if authoritative_requirements is not None
                else "legacy_unresolved"
            ),
            # This payload contains hashes/opaque ids only.  It may be copied
            # to a strict trace by a future observed-stage integration without
            # accidentally carrying the raw request or source text.
            "authoritative_requirements": (
                authoritative_requirements.as_trace_payload()
                if authoritative_requirements is not None
                else None
            ),
            "masked_selected_episode_ids": [int(item["id"]) for item in masked_episodes],
            "masked_missing_slots": sorted(masked.missing_required),
            "hits": [],
            "context_hits": [],
            "need_hits": [],
            "attached_edges": [],
            "attached_episode_ids": [],
            "external_calls": 0,
        }
        if bundle is not None:
            trace.update(self._query_vector_trace_metadata(bundle))
        if (
            not self.config.retrieval.contextual_association_enabled
            or self.contextual_matcher is None
            or bundle is None
            or not masked.missing_required
        ):
            trace["reason"] = (
                "no_unresolved_slots" if not masked.missing_required else "contextual_disabled_or_no_bundle"
            )
            return masked_episodes, trace
        unresolved_slots = [
            item for item in slots if item.slot_id in masked.missing_required
        ]
        # T09 deliberately consumes the matcher's uncapped proposals exactly
        # once.  The target/source/evidence gate below runs before any distinct
        # Episode cap, so a high-scoring stale target cannot hide a later valid
        # candidate.  Do not substitute compatibility ``hits`` here.
        request_evaluation_as_of = self._resolve_contextual_evaluation_as_of(
            evaluation_as_of
        )
        preliminary = self.contextual_recall(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=anchor_activations,
            unresolved_slot_ids=[item.slot_id for item in unresolved_slots],
            evaluation_as_of=request_evaluation_as_of,
        )
        trace = preliminary
        raw_proposals = trace.get("pre_target_proposals", [])
        if not isinstance(raw_proposals, (list, tuple)) or not all(
            isinstance(item, ContextualPreTargetProposal)
            for item in raw_proposals
        ):
            trace["hits"] = []
            trace["attached_edges"] = []
            trace["attached_episode_ids"] = []
            trace["reason"] = "invalid_pre_target_proposals"
            trace["target_gate"] = {
                "stage": "target_checked_before_endpoint_cap_v1",
                "reason": "invalid_pre_target_proposals",
                "outcomes": [],
            }
            return masked_episodes, trace
        gate_result = self._gate_contextual_targets(
            proposals=raw_proposals,
            bundle=bundle,
            unresolved_slots=unresolved_slots,
            endpoint_limit=endpoint_limit,
            evaluation_as_of=request_evaluation_as_of,
        )
        target_relevance = {
            (item.matched_slot_id, int(item.target_episode_id)): float(
                item.relevance_estimate
            )
            for item in gate_result.outcomes
            if item.relevance_estimate is not None
            and math.isfinite(float(item.relevance_estimate))
        }
        trace["hits"] = list(gate_result.hits)
        trace["attached_episode_ids"] = [
            int(item.target_episode_id) for item in gate_result.hits
        ]
        trace["attached_edges"] = [
            int(item.association_id) for item in gate_result.hits
        ]
        trace["target_gate"] = gate_result.as_trace_payload()
        # T10 will replace this flat hit list with contribution-aware selection.
        # Until then, retain every typed target outcome next to the old selector
        # projection so no rejected/competing edge is silently lost.
        trace["target_checks"] = [
            item.as_trace_payload() for item in gate_result.outcomes
        ]
        trace["compatibility_projection"] = "contextual_slot_hits_pending_t10_contribution_selector"
        trace["slots"] = [
            {"slot_id": item.slot_id, "question": item.question, "required": item.required}
            for item in slots
        ]
        trace["masked_selected_episode_ids"] = [int(item["id"]) for item in masked_episodes]
        trace["masked_missing_slots"] = sorted(masked.missing_required)
        trace.update(self._query_vector_trace_metadata(bundle))
        if not trace.get("hits"):
            trace.setdefault("reason", "no_contextual_slot_hit")
            return masked_episodes, trace
        existing = {int(item["id"]) for item in episodes}
        new_nodes = [
            TraversedNode("episode", int(hit.target_episode_id), float(hit.target_support_score))
            for hit in trace["hits"]
            if int(hit.target_episode_id) not in existing
        ]
        if new_nodes:
            recovered, _ = self._materialize_nodes(new_nodes, include_sources=True)
            episodes = [*episodes, *recovered]
        treatment_candidates = self._slot_candidates(
            episodes,
            slot_support,
            reranked_episode_ids,
            trace["hits"],
            target_relevance,
        )
        attribution = strict_contextual_attribution(
            treatment_candidates, slots, budget
        )
        treatment = attribution["treatment"]
        by_id = {int(item["id"]): item for item in episodes}
        treatment_episodes = [
            by_id[item.episode_id]
            for item in treatment.selected
            if item.episode_id in by_id
        ] or masked_episodes
        delta = treatment_masked_delta(treatment, masked)
        trace.update(delta)
        trace["strict_attribution"] = attribution["edges"]
        trace["treatment_selected_episode_ids"] = [int(item["id"]) for item in treatment_episodes]
        trace["selected_count"] = len(treatment_episodes)
        trace["new_slot_count"] = int(delta["new_slot_count"])
        trace["harm_count"] = int(bool(delta["harm"]))
        if self.config.retrieval.contextual_association_shadow:
            trace["shadow"] = True
            return masked_episodes, trace
        trace["shadow"] = False
        return treatment_episodes, trace

    @staticmethod
    def _record_phase(
        phase_seconds: dict[str, float],
        name: str,
        started_at: float,
    ) -> None:
        phase_seconds[name] = round(
            phase_seconds.get(name, 0.0) + perf_counter() - started_at,
            6,
        )

    @property
    def paragraph_retrieval_enabled(self) -> bool:
        return bool(
            self.config.paragraph.enabled
            and getattr(self, "paragraph_index", None) is not None
            and getattr(self, "paragraphs", None) is not None
            and self.paragraph_index.count > 0
            and self.config.retrieval.paragraph_top_k > 0
            and self.config.retrieval.paragraph_rrf_weight > 0.0
        )

    @property
    def sparse_retrieval_enabled(self) -> bool:
        return bool(
            self.config.retrieval.sparse_enabled
            and getattr(self, "episode_sparse_index", None) is not None
            and getattr(self, "source_sparse_index", None) is not None
            and self.episode_sparse_index.count > 0
            and self.source_sparse_index.count > 0
            and (
                self.config.retrieval.sparse_episode_top_k > 0
                or self.config.retrieval.sparse_source_top_k > 0
            )
        )

    @property
    def association_cue_retrieval_enabled(self) -> bool:
        return bool(
            self.config.retrieval.association_cue_enabled
            and getattr(self, "association_index", None) is not None
            and self.association_index.count > 0
            and self.config.retrieval.association_cue_top_k > 0
        )

    @staticmethod
    def _association_is_cue_eligible(row) -> bool:
        return bool(
            row is not None
            and str(row["audit_status"]) == "dual_accepted"
            and str(row["relation_key"]) != "involves"
            and str(row["relation_text"]).strip()
            and "查询中自主增长：" in str(row["created_reason"])
        )

    @classmethod
    def _association_has_fast_path_confirmation(cls, row) -> bool:
        """Keep strong factual edges conservative without stalling cache edges.

        ``evidence_count`` counts independent insert/reinforcement events; it
        does not count the two endpoint Episodes or the two audit passes.  A
        bounded evidence bridge is explicitly a retrieval hint rather than a
        world-model fact, so one dual-audited observation is sufficient for
        immediate reuse.  Stronger semantic/causal edges still require a
        second independent reinforcement before they can bypass reranking.
        """

        relation_key = str(
            cls._association_row_value(row, "relation_key", "") or ""
        ).casefold()
        evidence_count = int(
            cls._association_row_value(row, "evidence_count", 0) or 0
        )
        minimum = 1 if relation_key == "evidence_bridge" else 2
        return evidence_count >= minimum

    def _association_cue_entries_from_matrix(
        self,
        queries: list[str],
        matrix: np.ndarray,
    ) -> list[dict]:
        if not self.association_cue_retrieval_enabled or not queries:
            return []
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (
            len(queries),
            self.config.model.embedding_dimension,
        ):
            raise ValueError("association cue matrix shape does not match queries")
        best_by_id: dict[int, dict] = {}
        for query_index, vector in enumerate(matrix):
            ranked = self.association_index.search(
                vector,
                self.config.retrieval.association_cue_top_k,
            )
            for rank, (association_id, cosine) in enumerate(ranked, start=1):
                if cosine < self.config.retrieval.association_cue_min_similarity:
                    continue
                row = self.associations.get(int(association_id))
                if not self._association_is_cue_eligible(row):
                    continue
                endpoint_score = (
                    self.config.retrieval.association_cue_rrf_weight
                    / (60 + rank)
                )
                item = {
                    "association_id": int(association_id),
                    "query_index": query_index,
                    "query": queries[query_index],
                    "rank": rank,
                    "cosine": float(cosine),
                    "endpoint_score": float(endpoint_score),
                    "endpoints": [
                        [str(row["from_type"]), int(row["from_id"])],
                        [str(row["to_type"]), int(row["to_id"])],
                    ],
                }
                previous = best_by_id.get(int(association_id))
                if previous is None or (
                    float(item["cosine"]), -int(item["rank"])
                ) > (
                    float(previous["cosine"]), -int(previous["rank"])
                ):
                    best_by_id[int(association_id)] = item
        entries = sorted(
            best_by_id.values(),
            key=lambda item: (
                float(item["cosine"]),
                -int(item["rank"]),
                -int(item["association_id"]),
            ),
            reverse=True,
        )
        if self.logger and entries:
            self.logger.emit("association_cue_retrieval", entries=entries)
        return entries

    def _association_cue_entries_from_text(self, queries: list[str]) -> list[dict]:
        """Find an unambiguous audited edge without a cloud embedding call."""

        retrieval = self.config.retrieval
        if (
            not self.association_cue_retrieval_enabled
            or not retrieval.association_cue_fast_path_enabled
            or not retrieval.association_cue_fast_path_local_enabled
            or not queries
        ):
            return []
        if self._lexical_association_cues is None:
            self._lexical_association_cues = LexicalAssociationCueIndex(
                self.associations.list_cue_candidates()
            )
        best_by_id: dict[int, dict] = {}
        for query_index, query in enumerate(queries):
            ranked = self._lexical_association_cues.search(
                query,
                max(2, int(retrieval.association_cue_top_k)),
            )
            if not ranked:
                continue
            top_score = float(ranked[0][1])
            runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            if top_score < float(
                retrieval.association_cue_fast_path_local_min_coverage
            ) or top_score - runner_up < float(
                retrieval.association_cue_fast_path_local_min_margin
            ):
                continue
            row = ranked[0][0]
            if not self._association_is_cue_eligible(row):
                continue
            association_id = int(row["id"])
            item = {
                "association_id": association_id,
                "query_index": query_index,
                "query": query,
                "rank": 1,
                "cosine": top_score,
                "local_coverage": top_score,
                "local_margin": top_score - runner_up,
                "score_kind": "local_field_idf",
                "endpoint_score": float(
                    retrieval.association_cue_rrf_weight / 61
                ),
                "endpoints": [
                    [str(row["from_type"]), int(row["from_id"])],
                    [str(row["to_type"]), int(row["to_id"])],
                ],
            }
            previous = best_by_id.get(association_id)
            if previous is None or top_score > float(previous["local_coverage"]):
                best_by_id[association_id] = item
        return sorted(
            best_by_id.values(),
            key=lambda item: (
                float(item["local_coverage"]),
                float(item["local_margin"]),
            ),
            reverse=True,
        )

    def match_association_cue_text(self, text: str) -> dict | None:
        """Expose the same fail-closed local cue used by the retrieval lane."""

        entries = self._association_cue_entries_from_text([text])
        if not self._association_cue_fast_endpoint_keys(entries):
            return None
        entry = entries[0]
        return {
            "association_id": int(entry["association_id"]),
            "coverage": float(entry["local_coverage"]),
            "margin": float(entry["local_margin"]),
            "score_kind": str(entry["score_kind"]),
        }

    def _association_cue_endpoint_summary(
        self, node_type: str, node_id: int
    ) -> dict:
        if node_type == "episode":
            row = self.episodes.get(node_id)
            if row is None:
                return {"type": node_type, "id": node_id, "missing": True}
            return {
                "type": node_type,
                "id": node_id,
                "source_key": str(row["source_key"]),
                "text": str(row["text"])[:600],
            }
        if node_type == "concept":
            row = self.concepts.get(node_id)
            if row is None:
                return {"type": node_type, "id": node_id, "missing": True}
            return {
                "type": node_type,
                "id": node_id,
                "text": (
                    f"{row['canonical_name']}：{row['description']}"
                )[:600],
            }
        return {"type": node_type, "id": node_id, "missing": True}

    def _association_cue_fast_endpoint_keys(
        self, entries: list[dict]
    ) -> set[tuple[str, int]]:
        """Return only strong cue endpoints that must survive pool truncation."""

        retrieval = self.config.retrieval
        if not retrieval.association_cue_fast_path_enabled:
            return set()
        keys: set[tuple[str, int]] = set()
        accepted_edges = 0
        ordered_entries = sorted(
            entries,
            key=lambda item: float(item.get("cosine", 0.0)),
            reverse=True,
        )
        if len(ordered_entries) >= 2 and (
            float(ordered_entries[0].get("cosine", 0.0))
            - float(ordered_entries[1].get("cosine", 0.0))
            < float(retrieval.association_cue_fast_path_min_margin)
        ):
            return set()
        for entry in ordered_entries:
            if accepted_edges >= max(
                1, int(retrieval.association_cue_fast_path_max_edges)
            ):
                break
            if str(entry.get("score_kind", "")).startswith("local_"):
                if float(entry.get("local_coverage", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_coverage
                ) or float(entry.get("local_margin", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_margin
                ):
                    continue
            elif float(entry.get("cosine", 0.0)) < float(
                retrieval.association_cue_fast_path_min_similarity
            ):
                continue
            row = self.associations.get(int(entry.get("association_id", -1)))
            if not self._association_is_cue_eligible(row):
                continue
            if float(
                self._association_row_value(row, "confidence", 0.0) or 0.0
            ) < float(retrieval.association_cue_fast_path_min_confidence):
                continue
            if not self._association_has_fast_path_confirmation(row):
                continue
            endpoints = [
                (str(node_type), int(node_id))
                for node_type, node_id in entry.get("endpoints", [])
            ]
            if len(endpoints) != 2 or any(
                node_type != "episode" for node_type, _ in endpoints
            ):
                continue
            keys.update(endpoints)
            accepted_edges += 1
        return keys

    @staticmethod
    def _truncate_traversed_nodes(
        nodes: list[TraversedNode],
        limit: int,
        protected: set[tuple[str, int]] | None = None,
    ) -> list[TraversedNode]:
        """Apply a hard pool budget without dropping a protected recall lane."""

        if limit <= 0:
            return []
        if len(nodes) <= limit or not protected:
            return nodes[:limit]
        protected_nodes = [
            item
            for item in nodes
            if (item.node_type, int(item.node_id)) in protected
        ][:limit]
        protected_keys = {
            (item.node_type, int(item.node_id)) for item in protected_nodes
        }
        ordinary = [
            item
            for item in nodes
            if (item.node_type, int(item.node_id)) not in protected_keys
        ][: max(0, limit - len(protected_nodes))]
        return [*ordinary, *protected_nodes]

    def _semantic_gate_association_cues(
        self,
        question: str,
        entries: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Fail closed unless both relation endpoints fit the query scope."""
        if (
            not self.config.retrieval.association_cue_semantic_gate_enabled
            or not entries
        ):
            return entries, []
        candidates: list[dict] = []
        for entry in entries:
            association_id = int(entry["association_id"])
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            endpoints = [
                self._association_cue_endpoint_summary(
                    str(row["from_type"]), int(row["from_id"])
                ),
                self._association_cue_endpoint_summary(
                    str(row["to_type"]), int(row["to_id"])
                ),
            ]
            candidates.append(
                {
                    "association_id": association_id,
                    "cosine": round(float(entry.get("cosine", 0.0)), 6),
                    "relation_key": str(row["relation_key"]),
                    "relation_text": str(row["relation_text"]),
                    "association_mode": (
                        str(row["association_mode"])
                        if "association_mode" in row.keys()
                        else "semantic"
                    ),
                    "generation": int(row["generation"]),
                    "claim_level": str(row["claim_level"]),
                    "endpoints": endpoints,
                }
            )
        with self._provider_purpose_scope(
            "audit", "association_cue_semantic_gate"
        ):
            payload = self.model.chat_json(
                ASSOCIATION_CUE_GATE_SYSTEM,
                json.dumps(
                    {"question": question, "candidates": candidates},
                    ensure_ascii=False,
                ),
            )
        raw_decisions = (
            payload.get("decisions", []) if isinstance(payload, dict) else []
        )
        allowed_ids = {int(item["association_id"]) for item in candidates}
        decisions_by_id: dict[int, dict] = {}
        if isinstance(raw_decisions, list):
            for raw in raw_decisions:
                if not isinstance(raw, dict):
                    continue
                try:
                    association_id = int(raw["association_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if association_id not in allowed_ids:
                    continue
                decisions_by_id[association_id] = {
                    "association_id": association_id,
                    "accept": raw.get("accept") is True,
                    "reason": str(raw.get("reason", ""))[:500],
                }
        decisions = [
            decisions_by_id.get(
                int(candidate["association_id"]),
                {
                    "association_id": int(candidate["association_id"]),
                    "accept": False,
                    "reason": "模型未返回有效决定，按 fail-closed 拒绝。",
                },
            )
            for candidate in candidates
        ]
        accepted_ids = {
            int(item["association_id"])
            for item in decisions
            if item["accept"]
        }
        maximum = max(
            0,
            int(
                self.config.retrieval.association_cue_semantic_gate_max_selected
            ),
        )
        accepted = [
            dict(entry)
            for entry in entries
            if int(entry["association_id"]) in accepted_ids
        ][:maximum]
        for entry in accepted:
            decision = decisions_by_id[int(entry["association_id"])]
            entry["semantic_gate_accept"] = True
            entry["semantic_gate_reason"] = decision["reason"]
        if self.logger:
            self.logger.emit(
                "association_cue_semantic_gate",
                question=question,
                decisions=decisions,
                accepted_association_ids=[
                    int(item["association_id"]) for item in accepted
                ],
            )
        return accepted, decisions

    def _active_association_cue_hits(
        self,
        entries: list[dict],
    ) -> tuple[list[SearchHit], list[int], list[dict]]:
        hits: list[SearchHit] = []
        association_ids: list[int] = []
        active_entries: list[dict] = []
        for raw in entries:
            association_id = int(raw["association_id"])
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            association_ids.append(association_id)
            item = dict(raw)
            # The durable row is authoritative if an overlay restored it.
            item["endpoints"] = [
                [str(row["from_type"]), int(row["from_id"])],
                [str(row["to_type"]), int(row["to_id"])],
            ]
            active_entries.append(item)
            score = float(item.get("endpoint_score", 0.0))
            for node_type, node_id in item["endpoints"]:
                hits.append(SearchHit(str(node_type), int(node_id), score))
        return self._merge_hits(hits), list(dict.fromkeys(association_ids)), active_entries

    def _source_episode_expansions(
        self,
        query: np.ndarray,
        source_ranked: list[tuple[int, float]],
    ) -> list[dict[str, float | int]]:
        """Map Source lexical hits to the locally closest persisted Episodes."""
        if not source_ranked:
            return []
        episode_rows = self.episodes.list_ids_by_source_ids(
            source_id for source_id, _score in source_ranked
        )
        ids_by_source: dict[int, list[int]] = {}
        for row in episode_rows:
            ids_by_source.setdefault(int(row["source_id"]), []).append(int(row["id"]))
        vectors = self.episode_index.get_many(
            episode_id
            for episode_ids in ids_by_source.values()
            for episode_id in episode_ids
        )
        limit = max(
            1, self.config.retrieval.sparse_source_episode_expansion_limit
        )
        expansions: list[dict[str, float | int]] = []
        for source_rank, (source_id, sparse_score) in enumerate(
            source_ranked, start=1
        ):
            scored = sorted(
                (
                    (episode_id, float(vectors[episode_id] @ query))
                    for episode_id in ids_by_source.get(int(source_id), [])
                    if episode_id in vectors
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            expansions.extend(
                {
                    "source_id": int(source_id),
                    "source_rank": source_rank,
                    "source_sparse_score": float(sparse_score),
                    "episode_local_rank": local_rank,
                    "episode_id": episode_id,
                    "episode_cosine": episode_cosine,
                }
                for local_rank, (episode_id, episode_cosine) in enumerate(
                    scored, start=1
                )
            )
        # Do not let all local Episodes from Source rank 1 consume the atomic
        # anchor budget before Source rank 2 is inspected.  Round-robin by
        # local rank preserves the sparse Source ranking while representing
        # several independently matched Source rows early.
        expansions.sort(
            key=lambda item: (
                int(item["episode_local_rank"]),
                int(item["source_rank"]),
                int(item["episode_id"]),
            )
        )
        return expansions

    def _source_key_cohort_hits(
        self,
        episode_anchor_ids: list[int],
        traversed: list[TraversedNode],
    ) -> tuple[list[SearchHit], dict]:
        """Recover bounded Episodes split across one logical input file.

        Source is intentionally a short evidence fragment, while source_key is
        the stable logical file identity.  Two independent Episode anchors in
        the same source_key are enough evidence that the file is locally
        relevant; all of its Episodes may then enter the rerank candidate pool
        when the cohort is small.  The added rows remain ordinary direct
        Episode evidence and do not become Associations.
        """
        retrieval = self.config.retrieval
        trace = {
            "enabled": bool(retrieval.source_key_cohort_enabled),
            "supported_source_keys": [],
            "added_episode_ids": [],
            "boosted_episode_ids": [],
            "skipped_source_keys": [],
        }
        if not retrieval.source_key_cohort_enabled or not episode_anchor_ids:
            return [], trace

        ordered_anchor_ids = list(dict.fromkeys(int(value) for value in episode_anchor_ids))
        anchor_rows = {
            int(row["id"]): row
            for row in self.episodes.get_many(ordered_anchor_ids)
        }
        ids_by_key: dict[str, list[int]] = {}
        first_rank: dict[str, int] = {}
        for rank, episode_id in enumerate(ordered_anchor_ids):
            row = anchor_rows.get(episode_id)
            if row is None:
                continue
            source_key = str(row["source_key"] or "")
            if not source_key:
                continue
            ids_by_key.setdefault(source_key, []).append(episode_id)
            first_rank.setdefault(source_key, rank)

        minimum_hits = max(1, int(retrieval.source_key_cohort_min_anchor_hits))
        supported_keys = sorted(
            (
                source_key
                for source_key, ids in ids_by_key.items()
                if len(set(ids)) >= minimum_hits
            ),
            key=lambda source_key: (
                first_rank[source_key],
                -len(set(ids_by_key[source_key])),
                source_key,
            ),
        )
        trace["supported_source_keys"] = supported_keys
        if not supported_keys:
            return [], trace

        existing_ids = {
            int(item.node_id)
            for item in traversed
            if item.node_type == "episode"
        }
        traversed_scores = {
            int(item.node_id): float(item.score)
            for item in traversed
            if item.node_type == "episode"
        }
        per_key_limit = max(
            minimum_hits,
            int(retrieval.source_key_cohort_max_episodes_per_key),
        )
        remaining = max(0, int(retrieval.source_key_cohort_total_limit))
        score_ratio = max(0.0, min(1.0, float(retrieval.source_key_cohort_score_ratio)))
        hits: list[SearchHit] = []
        maximum_keys = max(0, int(retrieval.source_key_cohort_max_keys))
        for source_key in supported_keys[:maximum_keys]:
            rows = list(self.episodes.list_by_source_key(source_key))
            if len(rows) > per_key_limit:
                trace["skipped_source_keys"].append(
                    {
                        "source_key": source_key,
                        "episode_count": len(rows),
                        "reason": "cohort_exceeds_per_key_limit",
                    }
                )
                continue
            if len(rows) > remaining:
                trace["skipped_source_keys"].append(
                    {
                        "source_key": source_key,
                        "episode_count": len(rows),
                        "reason": "cohort_exceeds_total_limit",
                    }
                )
                continue
            anchor_score = max(
                (
                    traversed_scores.get(episode_id, 0.0)
                    for episode_id in ids_by_key[source_key]
                ),
                default=0.0,
            )
            cohort_score = max(0.01, anchor_score * score_ratio)
            for row in rows:
                episode_id = int(row["id"])
                hits.append(SearchHit("episode", episode_id, cohort_score))
                if episode_id in existing_ids:
                    if traversed_scores.get(episode_id, 0.0) < cohort_score:
                        trace["boosted_episode_ids"].append(episode_id)
                else:
                    existing_ids.add(episode_id)
                    trace["added_episode_ids"].append(episode_id)
            remaining -= len(rows)
            if remaining <= 0:
                break
        return hits, trace

    def _paragraph_episode_expansions(
        self,
        query: np.ndarray,
        paragraph_ranked: list[tuple[int, float]],
    ) -> list[dict[str, float | int]]:
        """Map a raw-text hit to the most query-relevant Episodes in its Source."""
        if not paragraph_ranked or self.paragraphs is None:
            return []
        rows = {
            int(row["id"]): row
            for row in self.paragraphs.get_many(
                paragraph_id for paragraph_id, _score in paragraph_ranked
            )
        }
        best_by_source: dict[int, tuple[int, int, float]] = {}
        for rank, (paragraph_id, score) in enumerate(paragraph_ranked, start=1):
            row = rows.get(int(paragraph_id))
            if row is None:
                continue
            source_id = int(row["source_id"])
            best_by_source.setdefault(
                source_id, (rank, int(paragraph_id), float(score))
            )
        episode_rows = self.episodes.list_ids_by_source_ids(best_by_source)
        ids_by_source: dict[int, list[int]] = {}
        for row in episode_rows:
            ids_by_source.setdefault(int(row["source_id"]), []).append(int(row["id"]))
        all_episode_ids = [
            episode_id
            for episode_ids in ids_by_source.values()
            for episode_id in episode_ids
        ]
        vectors = self.episode_index.get_many(all_episode_ids)
        limit = max(1, self.config.retrieval.paragraph_episode_expansion_limit)
        expansions: list[dict[str, float | int]] = []
        for source_id, (rank, paragraph_id, paragraph_cosine) in best_by_source.items():
            scored = sorted(
                (
                    (episode_id, float(vectors[episode_id] @ query))
                    for episode_id in ids_by_source.get(source_id, [])
                    if episode_id in vectors
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            expansions.extend(
                {
                    "paragraph_id": paragraph_id,
                    "paragraph_rank": rank,
                    "paragraph_cosine": paragraph_cosine,
                    "source_id": source_id,
                    "episode_id": episode_id,
                    "episode_cosine": episode_cosine,
                }
                for episode_id, episode_cosine in scored
            )
        return expansions

    def _paragraph_context_by_source(
        self,
        paragraph_rankings: list[list[dict]],
    ) -> dict[int, list[dict]]:
        """Keep query-matched raw Paragraphs as Source-level rerank evidence.

        The snippets deliberately do not become Episode facts.  Every candidate
        from the same Source receives the same labelled raw context so the LLM
        can use omitted dialogue while preserving the provenance boundary.
        """
        retrieval = self.config.retrieval
        if (
            not self.paragraph_retrieval_enabled
            or not retrieval.paragraph_rerank_context_enabled
            or self.paragraphs is None
        ):
            return {}
        best: dict[int, dict] = {}
        for query_index, ranking in enumerate(paragraph_rankings):
            for rank, item in enumerate(ranking, start=1):
                paragraph_id = int(item["id"])
                score = float(item.get("score", 0.0))
                current = best.get(paragraph_id)
                candidate = {
                    "paragraph_id": paragraph_id,
                    "query_index": query_index,
                    "rank": rank,
                    "score": score,
                }
                if current is None or (score, -rank, -query_index) > (
                    float(current["score"]),
                    -int(current["rank"]),
                    -int(current["query_index"]),
                ):
                    best[paragraph_id] = candidate
        rows = {
            int(row["id"]): row
            for row in self.paragraphs.get_many(best)
        }
        by_source: dict[int, list[dict]] = {}
        maximum_chars = max(1, retrieval.paragraph_rerank_context_chars)
        for paragraph_id, match in best.items():
            row = rows.get(paragraph_id)
            if row is None:
                continue
            source_id = int(row["source_id"])
            by_source.setdefault(source_id, []).append(
                {
                    **match,
                    "source_id": source_id,
                    "text": str(row["text"])[:maximum_chars],
                    "provenance": "source_level_raw_paragraph",
                }
            )
        per_source = max(1, retrieval.paragraph_rerank_context_per_source)
        for source_id, items in by_source.items():
            items.sort(
                key=lambda item: (
                    float(item["score"]),
                    -int(item["rank"]),
                    -int(item["paragraph_id"]),
                ),
                reverse=True,
            )
            by_source[source_id] = items[:per_source]
        return by_source

    def _parse_intent(self, question: str) -> QueryIntent:
        with self._provider_purpose_scope("planner", "query_intent"):
            payload = self.model.chat_json(
                QUERY_SYSTEM, query_intent_prompt(question)
            )
        if not isinstance(payload, dict):
            raise ValueError("query intent response must be an object")
        intent = QueryIntent.from_dict(payload)
        intent.search_queries = list(
            dict.fromkeys(
                [
                    *intent.search_queries,
                    *structural_queries(question, intent),
                ]
            )
        )
        return intent

    def _vector_seed_hits(
        self,
        queries: list[str],
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
        *,
        provider_purpose: str = "base_vector_retrieval",
    ) -> list[SearchHit]:
        if not queries:
            return []
        with self._provider_purpose_scope("embedding", provider_purpose):
            matrix = np.asarray(self.model.embed(queries), dtype=np.float32)
        hits, _ = self._vector_seed_hits_from_matrix(
            queries,
            matrix,
            episode_anchor_ids,
            first_query_anchor_limit,
        )
        return hits

    def _vector_seed_hits_with_cues(
        self,
        queries: list[str],
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
        cue_scope_question: str | None = None,
        *,
        phase_seconds: dict[str, float] | None = None,
        timing_prefix: str = "",
        query_embeddings_override: dict[str, np.ndarray] | None = None,
        provider_purpose: str = "base_vector_retrieval",
    ) -> tuple[list[SearchHit], list[int], list[dict], dict[str, list]]:
        if not queries:
            return [], [], [], {
                "episode": [],
                "concept": [],
                "paragraph": [],
                "paragraph_episode_expansion": [],
            }
        # A strongly separated local match against an already dual-audited
        # learned edge is a true cache hit.  Its two direct Episode endpoints
        # are sufficient seed evidence, so neither query embedding nor broad
        # vector/sparse recall is needed on this lane.
        local_entries = self._association_cue_entries_from_text(queries)
        local_cue_hits, local_cue_ids, local_active_entries = (
            self._active_association_cue_hits(local_entries)
        )
        if self._association_cue_fast_endpoint_keys(local_active_entries):
            rankings = {
                "episode": [],
                "concept": [],
                "paragraph": [],
                "paragraph_episode_expansion": [],
                "association_capsule_fast_lane": True,
                "association_capsule_lookup": "local_field_idf",
            }
            if phase_seconds is not None and timing_prefix:
                phase_seconds[f"{timing_prefix}_embedding"] = 0.0
                phase_seconds[f"{timing_prefix}_retrieval"] = 0.0
            return (
                local_cue_hits,
                local_cue_ids,
                local_active_entries,
                rankings,
            )
        embed_started = perf_counter()
        overrides = query_embeddings_override or {}
        missing_queries = [query for query in queries if query not in overrides]
        embedded_by_query: dict[str, np.ndarray] = {}
        if missing_queries:
            with self._provider_purpose_scope("embedding", provider_purpose):
                missing_matrix = np.asarray(
                    self.model.embed(missing_queries), dtype=np.float32
                )
            if missing_matrix.shape != (
                len(missing_queries),
                self.config.model.embedding_dimension,
            ):
                raise ValueError("query embedding matrix shape does not match queries")
            embedded_by_query = {
                query: vector
                for query, vector in zip(
                    missing_queries, missing_matrix, strict=True
                )
            }
        normalized_by_query = {
            query: normalize_embedding(
                np.asarray(
                    overrides[query]
                    if query in overrides
                    else embedded_by_query[query],
                    dtype=np.float32,
                ),
                self.config.model.embedding_dimension,
            ).copy()
            for query in queries
        }
        matrix = np.asarray(
            [normalized_by_query[query] for query in queries],
            dtype=np.float32,
        )
        self.last_query_embeddings.update(
            {str(query): vector.copy() for query, vector in normalized_by_query.items()}
        )
        self.last_query_embedding_cache_trace["hits"].extend(
            query for query in queries if query in overrides
        )
        self.last_query_embedding_cache_trace["misses"].extend(missing_queries)
        if phase_seconds is not None and timing_prefix:
            self._record_phase(
                phase_seconds,
                f"{timing_prefix}_embedding",
                embed_started,
            )
        retrieval_started = perf_counter()
        entries = self._association_cue_entries_from_matrix(queries, matrix)
        entries, _decisions = self._semantic_gate_association_cues(
            cue_scope_question or queries[0], entries
        )
        cue_hits, cue_ids, active_entries = self._active_association_cue_hits(entries)
        fast_lane = bool(self._association_cue_fast_endpoint_keys(active_entries))
        vector_hits, rankings = self._vector_seed_hits_from_matrix(
            queries,
            matrix,
            episode_anchor_ids,
            first_query_anchor_limit,
            dense_only=fast_lane,
        )
        rankings["association_capsule_fast_lane"] = fast_lane
        if phase_seconds is not None and timing_prefix:
            self._record_phase(
                phase_seconds,
                f"{timing_prefix}_retrieval",
                retrieval_started,
            )
        return (
            self._merge_hits(vector_hits, cue_hits),
            cue_ids,
            active_entries,
            rankings,
        )

    def embed_query_text(self, text: str) -> np.ndarray:
        """Return one normalized query vector for request-cache matching.

        This uses the embedding endpoint only.  Embeddings never fall back to
        a reasoning model, matching the retrieval and import contract.
        """

        with self._provider_purpose_scope("embedding", "ad_hoc_query_embedding"):
            matrix = np.asarray(self.model.embed([text]), dtype=np.float32)
        if matrix.shape != (1, self.config.model.embedding_dimension):
            raise ValueError("query embedding shape does not match configured dimension")
        return normalize_embedding(
            matrix[0], self.config.model.embedding_dimension
        ).copy()

    def _vector_seed_hits_from_matrix(
        self,
        queries: list[str],
        matrix: np.ndarray,
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
        *,
        dense_only: bool = False,
    ) -> tuple[list[SearchHit], dict[str, list[list[dict[str, float | int]]]]]:
        """Rank fixed float32 query vectors and expose rankings for replay."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (
            len(queries),
            self.config.model.embedding_dimension,
        ):
            raise ValueError(
                "query embedding matrix shape does not match queries and dimension"
            )
        fused_scores: dict[tuple[str, int], float] = {}
        best_cosines: dict[tuple[str, int], float] = {}
        episode_rankings: list[list[tuple[int, float]]] = []
        concept_rankings: list[list[tuple[int, float]]] = []
        concept_raw_rankings: list[list[tuple[int, float]]] = []
        concept_gate_audits: list[list[dict[str, int | float | bool]]] = []
        paragraph_rankings: list[list[tuple[int, float]]] = []
        paragraph_expansion_rankings: list[list[dict[str, float | int]]] = []
        sparse_episode_rankings: list[list[tuple[int, float]]] = []
        sparse_source_rankings: list[list[tuple[int, float]]] = []
        sparse_source_expansion_rankings: list[
            list[dict[str, float | int]]
        ] = []
        fused_episode_rankings: list[list[tuple[int, float]]] = []
        atomic_episode_rankings: list[list[tuple[int, float]]] = []
        paragraph_only_scores: dict[tuple[str, int], float] = {}
        paragraph_only_cosines: dict[tuple[str, int], float] = {}
        for query_index, vector in enumerate(matrix):
            normalized = normalize_embedding(
                vector, self.config.model.embedding_dimension
            )
            episode_ranked = self.episode_index.search(
                normalized, self.config.retrieval.episode_top_k
            )
            concept_ranked_raw = self.concept_index.search(
                normalized, self.config.retrieval.concept_top_k
            )
            minimum_concept_reach = max(
                1,
                self.config.retrieval.concept_seed_min_reachable_episodes,
            )
            if minimum_concept_reach <= 1:
                concept_degrees = {
                    int(node_id): 1 for node_id, _score in concept_ranked_raw
                }
                concept_ranked = list(concept_ranked_raw)
            else:
                raw_ids = [int(node_id) for node_id, _score in concept_ranked_raw]
                missing_ids = [
                    node_id
                    for node_id in raw_ids
                    if node_id not in self._concept_reach_cache
                ]
                if missing_ids:
                    self._concept_reach_cache.update(
                        self.associations.concept_reachable_episode_counts(
                            missing_ids
                        )
                    )
                concept_degrees = {
                    node_id: self._concept_reach_cache.get(node_id, 0)
                    for node_id in raw_ids
                }
                concept_ranked = [
                    (node_id, score)
                    for node_id, score in concept_ranked_raw
                    if concept_degrees.get(int(node_id), 0)
                    >= minimum_concept_reach
                ]
            paragraph_ranked = (
                self.paragraph_index.search(
                    normalized, self.config.retrieval.paragraph_top_k
                )
                if self.paragraph_retrieval_enabled and not dense_only
                else []
            )
            episode_rankings.append(episode_ranked)
            concept_raw_rankings.append(concept_ranked_raw)
            concept_rankings.append(concept_ranked)
            admitted_concept_ids = {int(node_id) for node_id, _ in concept_ranked}
            concept_gate_audits.append(
                [
                    {
                        "id": int(node_id),
                        "score": float(score),
                        "reachable_episode_count": int(
                            concept_degrees.get(int(node_id), 0)
                        ),
                        "admitted": int(node_id) in admitted_concept_ids,
                    }
                    for node_id, score in concept_ranked_raw
                ]
            )
            paragraph_rankings.append(paragraph_ranked)
            sparse_episode_ranked = (
                self.episode_sparse_index.search(
                    queries[query_index],
                    self.config.retrieval.sparse_episode_top_k,
                )
                if self.sparse_retrieval_enabled and not dense_only
                else []
            )
            sparse_source_ranked = (
                self.source_sparse_index.search(
                    queries[query_index],
                    self.config.retrieval.sparse_source_top_k,
                )
                if self.sparse_retrieval_enabled and not dense_only
                else []
            )
            sparse_episode_rankings.append(sparse_episode_ranked)
            sparse_source_rankings.append(sparse_source_ranked)
            query_episode_scores: dict[int, float] = {}
            ranked_groups = (
                (
                    "episode",
                    episode_ranked,
                ),
                (
                    "concept",
                    concept_ranked,
                ),
            )
            for node_type, ranked in ranked_groups:
                for rank, (node_id, cosine) in enumerate(ranked, start=1):
                    key = (node_type, node_id)
                    fused_scores[key] = (
                        fused_scores.get(key, 0.0) + 1.0 / (60 + rank)
                    )
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0), cosine
                    )
                    if node_type == "episode":
                        query_episode_scores[node_id] = (
                            query_episode_scores.get(node_id, 0.0)
                            + 1.0 / (60 + rank)
                        )
            sparse_vectors = (
                self.episode_index.get_many(
                    node_id for node_id, _score in sparse_episode_ranked
                )
                if sparse_episode_ranked
                else {}
            )
            for rank, (node_id, _sparse_score) in enumerate(
                sparse_episode_ranked, start=1
            ):
                key = ("episode", node_id)
                contribution = (
                    self.config.retrieval.sparse_episode_rrf_weight
                    / (60 + rank)
                )
                fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                query_episode_scores[node_id] = (
                    query_episode_scores.get(node_id, 0.0) + contribution
                )
                vector_row = sparse_vectors.get(node_id)
                if vector_row is not None:
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0),
                        float(vector_row @ normalized),
                    )
            source_expansions = self._source_episode_expansions(
                normalized, sparse_source_ranked
            )
            sparse_source_expansion_rankings.append(source_expansions)
            for expansion in source_expansions:
                node_id = int(expansion["episode_id"])
                key = ("episode", node_id)
                combined_rank = (
                    int(expansion["source_rank"])
                    + int(expansion["episode_local_rank"])
                    - 1
                )
                contribution = (
                    self.config.retrieval.sparse_source_rrf_weight
                    / (60 + combined_rank)
                )
                fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                query_episode_scores[node_id] = (
                    query_episode_scores.get(node_id, 0.0) + contribution
                )
                best_cosines[key] = max(
                    best_cosines.get(key, -1.0),
                    float(expansion["episode_cosine"]),
                )
            paragraph_expansions = self._paragraph_episode_expansions(
                normalized, paragraph_ranked
            )
            paragraph_expansion_rankings.append(paragraph_expansions)
            for expansion in paragraph_expansions:
                if not self.config.retrieval.paragraph_seed_enabled:
                    continue
                key = ("episode", int(expansion["episode_id"]))
                contribution = (
                    self.config.retrieval.paragraph_rrf_weight
                    / (60 + int(expansion["paragraph_rank"]))
                )
                if self.config.retrieval.paragraph_recall_only:
                    paragraph_only_scores[key] = (
                        paragraph_only_scores.get(key, 0.0) + contribution
                    )
                    paragraph_only_cosines[key] = max(
                        paragraph_only_cosines.get(key, -1.0),
                        float(expansion["episode_cosine"]),
                    )
                else:
                    fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0),
                        float(expansion["episode_cosine"]),
                    )
                    query_episode_scores[int(expansion["episode_id"])] = (
                        query_episode_scores.get(int(expansion["episode_id"]), 0.0)
                        + contribution
                    )
            fused_episode_ranking = sorted(
                query_episode_scores.items(),
                key=lambda item: (
                    item[1],
                    best_cosines.get(("episode", item[0]), -1.0),
                ),
                reverse=True,
            )
            fused_episode_rankings.append(fused_episode_ranking)
            source_episode_ranking = [
                (
                    int(expansion["episode_id"]),
                    float(expansion["episode_cosine"]),
                )
                for expansion in source_expansions
            ]
            paragraph_episode_ranking = [
                (
                    int(expansion["episode_id"]),
                    float(expansion["episode_cosine"]),
                )
                for expansion in paragraph_expansions
            ]
            atomic_ranking: list[tuple[int, float]] = []
            atomic_seen: set[int] = set()
            channels = [
                episode_ranked,
                sparse_episode_ranked,
                source_episode_ranking,
                fused_episode_ranking,
            ]
            if (
                self.config.retrieval.paragraph_seed_enabled
                and not self.config.retrieval.paragraph_recall_only
            ):
                channels.insert(3, paragraph_episode_ranking)
            maximum_channel_length = max((len(channel) for channel in channels), default=0)
            for rank_index in range(maximum_channel_length):
                for channel in channels:
                    if rank_index >= len(channel):
                        continue
                    node_id, score = channel[rank_index]
                    if node_id not in atomic_seen:
                        atomic_seen.add(node_id)
                        atomic_ranking.append((node_id, score))
            atomic_episode_rankings.append(atomic_ranking)
        if episode_anchor_ids is not None and episode_rankings:
            remaining_rankings = atomic_episode_rankings
            if first_query_anchor_limit is not None:
                for node_id, _ in atomic_episode_rankings[0][:first_query_anchor_limit]:
                    if node_id not in episode_anchor_ids:
                        episode_anchor_ids.append(node_id)
                remaining_rankings = atomic_episode_rankings[1:]
            # Round-robin keeps every atomic evidence slot represented before a
            # single broad subquery can consume the whole answer budget.
            for rank_index in range(
                self.config.retrieval.answer_anchor_episodes_per_query
            ):
                for ranking in remaining_rankings:
                    if rank_index >= len(ranking):
                        continue
                    node_id = ranking[rank_index][0]
                    if node_id not in episode_anchor_ids:
                        episode_anchor_ids.append(node_id)
        if (
            self.config.retrieval.paragraph_seed_enabled
            and self.config.retrieval.paragraph_recall_only
        ):
            # Add only genuinely missing Episode seeds.  Existing baseline
            # scores and atomic anchor order remain byte-for-byte unaffected.
            for key, score in paragraph_only_scores.items():
                if key in fused_scores:
                    continue
                fused_scores[key] = score
                best_cosines[key] = paragraph_only_cosines[key]
        maximum = max(fused_scores.values(), default=1.0)
        hits = [
            SearchHit(
                node_type,
                node_id,
                0.99 * (score / maximum)
                + 0.01
                * max(0.0, min(1.0, (best_cosines[key] + 1.0) / 2.0)),
            )
            for key, score in fused_scores.items()
            for node_type, node_id in (key,)
        ]
        rankings = {
            "episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in episode_rankings
            ],
            "concept": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in concept_rankings
            ],
            "concept_raw": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in concept_raw_rankings
            ],
            "concept_gate": concept_gate_audits,
            "paragraph": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in paragraph_rankings
            ],
            "paragraph_episode_expansion": paragraph_expansion_rankings,
            "sparse_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in sparse_episode_rankings
            ],
            "sparse_source": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in sparse_source_rankings
            ],
            "sparse_source_episode_expansion": sparse_source_expansion_rankings,
            "fused_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in fused_episode_rankings
            ],
            "atomic_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in atomic_episode_rankings
            ],
        }
        if getattr(self, "logger", None) and any(paragraph_rankings):
            self.logger.emit(
                "paragraph_retrieval",
                queries=queries,
                paragraph_rankings=rankings["paragraph"],
                episode_expansions=paragraph_expansion_rankings,
            )
        if getattr(self, "logger", None) and any(sparse_episode_rankings):
            self.logger.emit(
                "sparse_retrieval",
                queries=queries,
                episode_rankings=rankings["sparse_episode"],
                source_rankings=rankings["sparse_source"],
                source_episode_expansions=sparse_source_expansion_rankings,
                fused_episode_rankings=rankings["fused_episode"],
            )
        return hits, rankings

    @staticmethod
    def _merge_hits(*groups: list[SearchHit]) -> list[SearchHit]:
        best: dict[tuple[str, int], SearchHit] = {}
        for hit in (item for group in groups for item in group):
            key = (hit.node_type, hit.node_id)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return sorted(best.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _normalize_warm_episode_activations(
        activations: dict[int, float] | None,
    ) -> dict[int, float] | None:
        """Bound request-local search hints before they can seed traversal.

        These scores carry attention from an earlier retrieval round.  They
        are not evidence of a factual claim or independent current-query
        anchors, and must never be persisted as graph weights.
        """

        if activations is None:
            return None
        if not isinstance(activations, dict) or len(activations) > 32:
            raise ValueError("warm_episode_activations must be a dict of at most 32 episodes")
        normalized: dict[int, float] = {}
        for episode_id, score in activations.items():
            if type(episode_id) is not int or not 0 < episode_id <= 2**53 - 1:
                raise ValueError("warm episode IDs must be positive safe integers")
            if type(score) not in (int, float):
                raise ValueError("warm episode activation scores must be numbers")
            value = float(score)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("warm episode activation scores must be finite and in (0, 1]")
            normalized[episode_id] = value
        return normalized

    @staticmethod
    def _frozen_search_hits(value: object) -> list[SearchHit]:
        """Decode a saved seed stage without inferring an omitted stage.

        Frozen plans are external inputs.  A malformed row is ignored rather
        than being coerced into a seed, and callers must separately classify a
        plan that never recorded a stage at all.
        """

        if not isinstance(value, list):
            return []
        hits: list[SearchHit] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            try:
                node_type = str(item["node_type"])
                node_id = int(item["node_id"])
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if node_id <= 0 or not math.isfinite(score):
                continue
            hits.append(SearchHit(node_type, node_id, score))
        return QueryEngine._merge_hits(hits)

    @staticmethod
    def _seed_hit_source_summary(
        hits: Sequence[SearchHit],
        rankings: Mapping[str, object],
    ) -> list[dict[str, object]]:
        """Record the observed local retrieval lanes for a saved seed stage."""

        episode_lanes: dict[int, set[str]] = {}

        def add_ranked(lane: str, rows: object, *, id_key: str = "id") -> None:
            if not isinstance(rows, list):
                return
            for ranking in rows:
                if not isinstance(ranking, list):
                    continue
                for row in ranking:
                    if not isinstance(row, Mapping):
                        continue
                    try:
                        episode_id = int(row.get(id_key, 0))
                    except (TypeError, ValueError):
                        continue
                    if episode_id > 0:
                        episode_lanes.setdefault(episode_id, set()).add(lane)

        add_ranked("dense_episode", rankings.get("episode"))
        add_ranked("sparse_episode", rankings.get("sparse_episode"))
        add_ranked("fused_episode", rankings.get("fused_episode"))
        add_ranked(
            "sparse_source_episode_expansion",
            rankings.get("sparse_source_episode_expansion"),
            id_key="episode_id",
        )
        add_ranked(
            "paragraph_episode_expansion",
            rankings.get("paragraph_episode_expansion"),
            id_key="episode_id",
        )
        result: list[dict[str, object]] = []
        for hit in hits:
            lanes = (
                sorted(episode_lanes.get(int(hit.node_id), set()))
                if hit.node_type == "episode"
                else []
            )
            result.append(
                {
                    "node_type": hit.node_type,
                    "node_id": int(hit.node_id),
                    "retrieval_lanes": lanes or ["not_observed_in_rankings"],
                }
            )
        return result

    @staticmethod
    def _interleave_anchor_ids(*groups: list[int]) -> list[int]:
        """Interleave retrieval phases so later-hop anchors are not tail-dropped."""
        merged: list[int] = []
        seen: set[int] = set()
        maximum = max((len(group) for group in groups), default=0)
        for index in range(maximum):
            for group in groups:
                if index >= len(group):
                    continue
                node_id = int(group[index])
                if node_id not in seen:
                    seen.add(node_id)
                    merged.append(node_id)
        return merged

    def _seed_hits(
        self,
        question: str,
        intent: QueryIntent,
        episode_anchor_ids: list[int] | None = None,
    ) -> list[SearchHit]:
        queries = list(dict.fromkeys([question, *intent.search_queries]))
        hits, _cue_ids, _cue_entries, _rankings = self._vector_seed_hits_with_cues(
            queries,
            episode_anchor_ids,
            self.config.retrieval.answer_whole_question_anchor_episodes,
            question,
        )
        for entity in intent.target_entities:
            for row in self.concepts.find_by_alias(entity):
                concept_id = int(row["canonical_concept_id"] or row["id"])
                hits.append(SearchHit("concept", concept_id, 1.0))
        return self._merge_hits(hits)

    def _replay_configuration(self) -> dict[str, int | float | bool]:
        retrieval = self.config.retrieval
        configuration: dict[str, int | float | bool] = {
            "embedding_dimension": self.config.model.embedding_dimension,
            "episode_top_k": retrieval.episode_top_k,
            "concept_top_k": retrieval.concept_top_k,
            "graph_beam_width": retrieval.graph_beam_width,
            "graph_max_hops": retrieval.graph_max_hops,
            "candidate_limit": retrieval.candidate_limit,
            "answer_episode_limit": retrieval.answer_episode_limit,
            "answer_concept_limit": retrieval.answer_concept_limit,
            "answer_path_limit": retrieval.answer_path_limit,
            "growth_persist_only_used": retrieval.growth_persist_only_used,
            "growth_counterfactual_utility_enabled": (
                retrieval.growth_counterfactual_utility_enabled
            ),
            "growth_staging_enabled": retrieval.growth_staging_enabled,
            "source_key_cohort_enabled": retrieval.source_key_cohort_enabled,
            "source_key_cohort_min_anchor_hits": (
                retrieval.source_key_cohort_min_anchor_hits
            ),
            "source_key_cohort_max_keys": retrieval.source_key_cohort_max_keys,
            "source_key_cohort_max_episodes_per_key": (
                retrieval.source_key_cohort_max_episodes_per_key
            ),
            "source_key_cohort_total_limit": (
                retrieval.source_key_cohort_total_limit
            ),
            "source_key_cohort_score_ratio": (
                retrieval.source_key_cohort_score_ratio
            ),
            "learned_bridge_slots": retrieval.learned_bridge_slots,
            "learned_bridge_min_query_relevance": (
                retrieval.learned_bridge_min_query_relevance
            ),
            "learned_bridge_duplicate_threshold": (
                retrieval.learned_bridge_duplicate_threshold
            ),
            "answer_whole_question_anchor_episodes": (
                retrieval.answer_whole_question_anchor_episodes
            ),
            "answer_anchor_episodes_per_query": (
                retrieval.answer_anchor_episodes_per_query
            ),
            "rerank_atomic_query_limit": retrieval.rerank_atomic_query_limit,
            "rerank_atomic_floor_enabled": retrieval.rerank_atomic_floor_enabled,
            "rerank_atomic_floor_query_limit": (
                retrieval.rerank_atomic_floor_query_limit
            ),
            "rerank_atomic_floor_per_query": (
                retrieval.rerank_atomic_floor_per_query
            ),
            "rerank_atomic_floor_total_limit": (
                retrieval.rerank_atomic_floor_total_limit
            ),
            "rerank_constraint_floor_per_query": (
                retrieval.rerank_constraint_floor_per_query
            ),
            "rerank_constraint_floor_total_limit": (
                retrieval.rerank_constraint_floor_total_limit
            ),
            "rerank_answer_slot_neighbor_radius": (
                retrieval.rerank_answer_slot_neighbor_radius
            ),
            "rerank_answer_slot_neighbor_total_limit": (
                retrieval.rerank_answer_slot_neighbor_total_limit
            ),
            "rerank_constraint_candidate_per_query": (
                retrieval.rerank_constraint_candidate_per_query
            ),
            "rerank_constraint_candidate_total_limit": (
                retrieval.rerank_constraint_candidate_total_limit
            ),
            "rerank_question_sparse_floor_limit": (
                retrieval.rerank_question_sparse_floor_limit
            ),
            "rerank_combined_floor_limit": (
                retrieval.rerank_combined_floor_limit
            ),
        }
        if self.paragraph_retrieval_enabled:
            configuration.update(
                {
                    "paragraph_enabled": True,
                    "paragraph_top_k": retrieval.paragraph_top_k,
                    "paragraph_episode_expansion_limit": (
                        retrieval.paragraph_episode_expansion_limit
                    ),
                    "paragraph_rrf_weight": retrieval.paragraph_rrf_weight,
                    "paragraph_seed_enabled": retrieval.paragraph_seed_enabled,
                    "paragraph_rerank_context_enabled": (
                        retrieval.paragraph_rerank_context_enabled
                    ),
                    "paragraph_rerank_context_per_source": (
                        retrieval.paragraph_rerank_context_per_source
                    ),
                    "paragraph_rerank_context_chars": (
                        retrieval.paragraph_rerank_context_chars
                    ),
                }
            )
        if self.sparse_retrieval_enabled:
            configuration.update(
                {
                    "sparse_enabled": True,
                    "sparse_episode_top_k": retrieval.sparse_episode_top_k,
                    "sparse_source_top_k": retrieval.sparse_source_top_k,
                    "sparse_source_episode_expansion_limit": (
                        retrieval.sparse_source_episode_expansion_limit
                    ),
                    "sparse_episode_rrf_weight": retrieval.sparse_episode_rrf_weight,
                    "sparse_source_rrf_weight": retrieval.sparse_source_rrf_weight,
                }
            )
        if retrieval.rerank_enabled:
            configuration.update(
                {
                    "rerank_enabled": True,
                    "rerank_candidate_limit": retrieval.rerank_candidate_limit,
                    "rerank_precompression_limit": (
                        retrieval.rerank_precompression_limit
                    ),
                    "rerank_shortlist_limit": retrieval.rerank_shortlist_limit,
                    "rerank_coverage_audit_enabled": (
                        retrieval.rerank_coverage_audit_enabled
                    ),
                    "rerank_audit_enabled": retrieval.rerank_audit_enabled,
                }
            )
        if retrieval.association_cue_enabled:
            configuration.update(
                {
                    "association_cue_enabled": True,
                    "association_cue_top_k": retrieval.association_cue_top_k,
                    "association_cue_min_similarity": (
                        retrieval.association_cue_min_similarity
                    ),
                    "association_cue_rrf_weight": (
                        retrieval.association_cue_rrf_weight
                    ),
                }
            )
            if retrieval.association_cue_semantic_gate_enabled:
                configuration.update(
                    {
                        "association_cue_semantic_gate_enabled": True,
                        "association_cue_semantic_gate_max_selected": (
                            retrieval.association_cue_semantic_gate_max_selected
                        ),
                    }
                )
            if retrieval.association_cue_fast_path_enabled:
                configuration.update(
                    {
                        "association_cue_fast_path_enabled": True,
                        "association_cue_fast_path_min_similarity": (
                            retrieval.association_cue_fast_path_min_similarity
                        ),
                        "association_cue_fast_path_min_confidence": (
                            retrieval.association_cue_fast_path_min_confidence
                        ),
                        "association_cue_fast_path_min_margin": (
                            retrieval.association_cue_fast_path_min_margin
                        ),
                        "association_cue_fast_path_max_edges": (
                            retrieval.association_cue_fast_path_max_edges
                        ),
                    }
                )
        if retrieval.contextual_association_enabled:
            configuration.update(
                {
                    "contextual_association_enabled": True,
                    "contextual_association_shadow": retrieval.contextual_association_shadow,
                    "contextual_promotion_enabled": retrieval.contextual_promotion_enabled,
                    "contextual_context_top_k": retrieval.contextual_context_top_k,
                    "contextual_need_top_k": retrieval.contextual_need_top_k,
                    "contextual_edge_top_k": retrieval.contextual_edge_top_k,
                    "contextual_context_threshold": retrieval.contextual_context_threshold,
                    "contextual_need_threshold": retrieval.contextual_need_threshold,
                    "contextual_combine_mode": retrieval.contextual_combine_mode,
                    "contextual_endpoint_limit_light": retrieval.contextual_endpoint_limit_light,
                    "contextual_endpoint_limit_standard": retrieval.contextual_endpoint_limit_standard,
                    "contextual_endpoint_limit_deep": retrieval.contextual_endpoint_limit_deep,
                }
            )
        return configuration

    @staticmethod
    def _serialize_hits(hits: list[SearchHit]) -> list[dict]:
        return [
            {
                "node_type": hit.node_type,
                "node_id": int(hit.node_id),
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def attach_association_cues(self, bundle: dict) -> dict:
        """Attach deterministic cue matches to an already frozen base plan.

        The reranker plan stays the no-cue baseline. Replay can then add or
        remove only relation-derived seed endpoints without another model call.
        """
        if not self.association_cue_retrieval_enabled:
            raise ValueError("association cue retrieval is not active")
        result = deepcopy(bundle)
        initial_queries = [str(value) for value in result["initial_queries"]]
        initial_matrix = np.asarray(
            result["initial_query_embeddings_float32"], dtype=np.float32
        )
        entries = self._association_cue_entries_from_matrix(
            initial_queries, initial_matrix
        )
        followup_queries = [
            str(value) for value in result.get("followup_queries", [])
        ]
        if followup_queries:
            followup_matrix = np.asarray(
                result["followup_query_embeddings_float32"], dtype=np.float32
            )
            entries.extend(
                self._association_cue_entries_from_matrix(
                    followup_queries, followup_matrix
                )
            )
        best_by_id: dict[int, dict] = {}
        for entry in entries:
            association_id = int(entry["association_id"])
            previous = best_by_id.get(association_id)
            if previous is None or float(entry["cosine"]) > float(
                previous["cosine"]
            ):
                best_by_id[association_id] = dict(entry)
        entries = sorted(
            best_by_id.values(),
            key=lambda item: float(item["cosine"]),
            reverse=True,
        )
        entries, gate_decisions = self._semantic_gate_association_cues(
            str(result["question"]), entries
        )
        base_seed_payload = list(
            result.get("base_final_seed_hits", result["final_seed_hits"])
        )
        base_hits = [
            SearchHit(
                str(item["node_type"]),
                int(item["node_id"]),
                float(item["score"]),
            )
            for item in base_seed_payload
        ]
        cue_hits, cue_ids, active_entries = self._active_association_cue_hits(
            entries
        )
        result["version"] = 4
        result["base_final_seed_hits"] = base_seed_payload
        result["association_cue_entries"] = active_entries
        result["association_cue_gate_decisions"] = gate_decisions
        result["association_cue_association_ids"] = cue_ids
        result["final_seed_hits"] = self._serialize_hits(
            self._merge_hits(base_hits, cue_hits)
        )
        result["configuration"] = self._replay_configuration()
        return result

    def build_replay_bundle(self, question: str) -> dict:
        """Plan Q2 once and freeze every stochastic/vector input for replay."""
        intent = self._parse_intent(question)
        initial_queries = list(dict.fromkeys([question, *intent.search_queries]))
        with self._provider_purpose_scope(
            "embedding", "replay_initial_vector_retrieval"
        ):
            initial_matrix = np.asarray(
                self.model.embed(initial_queries), dtype=np.float32
            )
        initial_anchor_ids: list[int] = []
        initial_hits, initial_rankings = self._vector_seed_hits_from_matrix(
            initial_queries,
            initial_matrix,
            initial_anchor_ids,
            self.config.retrieval.answer_whole_question_anchor_episodes,
        )
        alias_hits: list[SearchHit] = []
        for entity in intent.target_entities:
            for row in self.concepts.find_by_alias(entity):
                alias_hits.append(
                    SearchHit(
                        "concept",
                        int(row["canonical_concept_id"] or row["id"]),
                        1.0,
                    )
                )
        seeds = self._merge_hits(initial_hits, alias_hits)
        traversed, _ = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        traversed = traversed[: self.config.retrieval.candidate_limit]
        should_plan_followup, _reason = self._followup_planning_decision(
            question,
            intent,
        )
        followup_queries = (
            self._plan_followup_queries(question, intent, traversed)
            if should_plan_followup
            else []
        )
        followup_matrix = np.empty(
            (0, self.config.model.embedding_dimension), dtype=np.float32
        )
        followup_rankings: dict[str, list] = {"episode": [], "concept": []}
        followup_hits: list[SearchHit] = []
        source_cohort_hits: list[SearchHit] = []
        followup_anchor_ids: list[int] = []
        if followup_queries:
            with self._provider_purpose_scope(
                "embedding", "replay_followup_vector_retrieval"
            ):
                followup_matrix = np.asarray(
                    self.model.embed(followup_queries), dtype=np.float32
                )
            followup_hits, followup_rankings = self._vector_seed_hits_from_matrix(
                followup_queries,
                followup_matrix,
                followup_anchor_ids,
            )
            seeds = self._merge_hits(seeds, followup_hits)
        episode_anchor_ids = self._interleave_anchor_ids(
            followup_anchor_ids,
            initial_anchor_ids,
        ) if followup_queries else list(initial_anchor_ids)
        final_traversed, _final_paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        final_traversed = final_traversed[: self.config.retrieval.candidate_limit]
        source_cohort_hits, source_key_cohort_trace = self._source_key_cohort_hits(
            initial_anchor_ids,
            final_traversed,
        )
        if source_cohort_hits:
            seeds = self._merge_hits(seeds, source_cohort_hits)
            final_traversed, _final_paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            final_traversed = final_traversed[
                : self.config.retrieval.candidate_limit
            ]
        candidate_episodes, _candidate_concepts = self._materialize_nodes(
            final_traversed
        )
        paragraph_context_by_source = self._paragraph_context_by_source(
            [
                *initial_rankings.get("paragraph", []),
                *followup_rankings.get("paragraph", []),
            ]
        )
        evidence_floor_trace = self._hybrid_evidence_floor_trace(
            question,
            initial_rankings,
            followup_rankings,
            initial_queries,
            followup_queries,
        )
        constraint_candidate_ids = self._constraint_candidate_ids(
            question,
            initial_queries,
            initial_rankings,
            followup_queries,
            followup_rankings,
        )
        reranked_episode_ids, rerank_trace = self._rerank_answer_episodes(
            question,
            intent,
            list(dict.fromkeys([*initial_queries, *followup_queries])),
            candidate_episodes,
            min(self.config.retrieval.answer_episode_limit, len(candidate_episodes)),
            [*constraint_candidate_ids, *episode_anchor_ids],
            paragraph_context_by_source,
            required_candidate_ids=evidence_floor_trace["selected_episode_ids"],
        )
        evidence_floor_trace = self._admit_evidence_floor_trace(
            evidence_floor_trace,
            rerank_trace.get("required_evidence_floor_ids", []),
        )
        rerank_trace["deterministic_evidence_floor"] = evidence_floor_trace
        rerank_trace["constraint_candidate_ids"] = constraint_candidate_ids
        rerank_trace["candidate_input_fingerprint"] = (
            self._rerank_candidate_input_fingerprint(
                candidate_episodes,
                paragraph_context_by_source,
            )
        )
        bundle = {
            "version": (
                3
                if self.sparse_retrieval_enabled
                else (2 if self.paragraph_retrieval_enabled else 1)
            ),
            "question": question,
            "intent": asdict(intent),
            "initial_queries": initial_queries,
            "followup_queries": followup_queries,
            "initial_query_embeddings_float32": initial_matrix.tolist(),
            "followup_query_embeddings_float32": followup_matrix.tolist(),
            "initial_rankings": initial_rankings,
            "followup_rankings": followup_rankings,
            "initial_seed_hits": self._serialize_hits(
                self._merge_hits(initial_hits, alias_hits)
            ),
            "initial_seed_hit_sources": self._seed_hit_source_summary(
                self._merge_hits(initial_hits, alias_hits), initial_rankings
            ),
            "followup_seed_hits": self._serialize_hits(followup_hits),
            "source_cohort_seed_hits": self._serialize_hits(source_cohort_hits),
            "final_seed_hits": self._serialize_hits(seeds),
            "initial_episode_anchor_ids": initial_anchor_ids,
            "followup_episode_anchor_ids": followup_anchor_ids,
            "episode_anchor_ids": episode_anchor_ids,
            "source_key_cohort": source_key_cohort_trace,
            "reranked_episode_ids": reranked_episode_ids,
            "rerank_trace": rerank_trace,
            "configuration": self._replay_configuration(),
        }
        if self.association_cue_retrieval_enabled:
            return self.attach_association_cues(bundle)
        return bundle

    def build_query_plan(self, question: str) -> dict:
        """Build a serializable plan shared by all comparable query arms.

        The historical replay bundle already freezes intent parsing, follow-up
        planning, embeddings, dense/sparse rankings and the LLM rerank.  A
        query plan gives that artifact a stable identity so the normal answer
        and growth pipeline can consume it, rather than limiting it to offline
        retrieval probes.
        """
        plan = self.build_replay_bundle(question)
        plan["plan_schema"] = "frozen_query_plan_v1"
        canonical = json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        plan["plan_id"] = hashlib.sha256(canonical).hexdigest()
        return plan

    def restore_frozen_query_vectors(
        self,
        question: str,
        frozen_plan: Mapping[str, object],
    ) -> QueryVectorBundle:
        """Rebuild a checked request-vector bundle from one frozen public plan.

        ``build_query_plan`` stores the observed float32 matrices alongside its
        planner/retrieval decisions.  A comparable edge-on/edge-masked Q2
        needs the logical slot bindings as well as those matrices: without the
        bindings the public contextual matcher correctly rejects the request
        as having no vector for its resolved requirement.  This public helper
        supplies that missing bridge without embedding, reranking, matching,
        or mutating any association.

        It is deliberately narrower than a generic vector loader.  The plan
        must pass the ordinary frozen-plan validation for this engine and the
        supplied text must be the plan's text; a caller cannot attach a vector
        bundle to a different question or configuration.
        """

        plan = dict(frozen_plan)
        self._validate_query_plan(question, plan)
        intent = QueryIntent.from_dict(dict(plan["intent"]))
        rerank_atomic_limit = int(self.config.retrieval.rerank_atomic_query_limit)
        requirement_queries = (
            [question]
            if rerank_atomic_limit == 1
            else limit_rerank_atomic_queries(
                list(intent.search_queries), rerank_atomic_limit
            )
        )
        requirements = resolve_authoritative_requirements(
            question,
            replace(intent, search_queries=requirement_queries),
            request_mode=infer_request_mode(question, intent),
            planner_origin="persisted",
        )
        atomic_request_texts = (
            [question, *intent.search_queries]
            if rerank_atomic_limit == 1
            else list(intent.search_queries)
        )
        initial_requests = [
            *self._request_vector_specs(
                [question], role="whole", authoritative_requirements=requirements
            ),
            *self._request_vector_specs(
                atomic_request_texts,
                role="atomic",
                authoritative_requirements=requirements,
            ),
        ]
        followup_queries = [str(value) for value in plan.get("followup_queries", [])]
        followup_requests = self._request_vector_specs(
            followup_queries,
            role="followup",
            authoritative_requirements=requirements,
        )
        initial_rows = np.asarray(
            plan.get("initial_query_embeddings_float32", []), dtype=np.float32
        )
        frozen_initial_queries = [
            normalize_query_text(str(value))
            for value in plan.get("initial_queries", [])
        ]
        frozen_followup_queries = [
            normalize_query_text(str(value)) for value in followup_queries
        ]
        raw_followup_rows = plan.get("followup_query_embeddings_float32", [])
        followup_rows = (
            np.empty((0, int(self.config.model.embedding_dimension)), dtype=np.float32)
            if not frozen_followup_queries and raw_followup_rows == []
            else np.asarray(raw_followup_rows, dtype=np.float32)
        )
        dimension = int(self.config.model.embedding_dimension)
        if initial_rows.shape != (len(frozen_initial_queries), dimension):
            raise ValueError("frozen plan initial vector matrix is invalid")
        if followup_rows.shape != (len(frozen_followup_queries), dimension):
            raise ValueError("frozen plan follow-up vector matrix is invalid")
        vectors_by_text: dict[str, np.ndarray] = {}
        for text, vector in zip(frozen_initial_queries, initial_rows, strict=True):
            if not text:
                raise ValueError("frozen plan has an empty initial query")
            vectors_by_text.setdefault(text, vector)
        for text, vector in zip(frozen_followup_queries, followup_rows, strict=True):
            if not text:
                raise ValueError("frozen plan has an empty follow-up query")
            previous = vectors_by_text.setdefault(text, vector)
            if previous is not vector and not np.array_equal(previous, vector):
                raise ValueError("frozen plan maps one query text to conflicting vectors")
        coordinator = self._new_query_embedding_coordinator()
        return coordinator.bundle_from_precomputed_vectors(
            [*initial_requests, *followup_requests],
            vectors_by_text,
            source_request_hash="sha256:" + self._request_vector_hash(question),
        )

    def build_frozen_query_input(self, question: str) -> tuple[dict, QueryVectorBundle]:
        """Public one-shot preparation for paired frozen-input diagnostics.

        The returned plan and bundle are generated once from the normal public
        planner/retrieval path.  Reusing them across edge conditions performs
        no per-edge embedding or model call; ``query`` still owns all source,
        scope, and contextual-edge validation when each condition executes.
        """

        plan = self.build_query_plan(question)
        return plan, self.restore_frozen_query_vectors(question, plan)

    def _validate_query_plan(self, question: str, plan: dict) -> None:
        if plan.get("plan_schema") != "frozen_query_plan_v1":
            raise ValueError("unsupported frozen query plan schema")
        if str(plan.get("question", "")) != question:
            raise ValueError("frozen query plan question does not match query")
        expected_id = str(plan.get("plan_id", ""))
        payload = deepcopy(plan)
        payload.pop("plan_id", None)
        actual_id = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if not expected_id or expected_id != actual_id:
            raise ValueError("frozen query plan fingerprint is invalid")
        if plan.get("configuration") != self._replay_configuration():
            raise ValueError("frozen query plan configuration does not match engine")

    def replay_retrieval(self, bundle: dict) -> dict:
        """Replay graph expansion/evidence selection without model calls or writes."""
        if int(bundle.get("version", 0)) not in {1, 2, 3, 4}:
            raise ValueError("unsupported replay bundle version")
        if bundle.get("configuration") != self._replay_configuration():
            raise ValueError("replay bundle configuration does not match engine")
        question = str(bundle["question"])
        base_seed_payload = bundle.get(
            "base_final_seed_hits", bundle["final_seed_hits"]
        )
        seeds = [
            SearchHit(
                str(item["node_type"]),
                int(item["node_id"]),
                float(item["score"]),
            )
            for item in base_seed_payload
        ]
        active_cue_ids: list[int] = []
        active_cue_entries: list[dict] = []
        if self.config.retrieval.association_cue_enabled:
            cue_hits, active_cue_ids, active_cue_entries = (
                self._active_association_cue_hits(
                    list(bundle.get("association_cue_entries", []))
                )
            )
            seeds = self._merge_hits(seeds, cue_hits)
        traversed, paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        cue_scores = {
            int(item["association_id"]): float(item.get("cosine", 0.0))
            for item in active_cue_entries
        }
        paths.extend(
            self._explicit_association_paths(active_cue_ids, cue_scores)
        )
        traversed = self._truncate_traversed_nodes(
            traversed,
            self.config.retrieval.candidate_limit,
            self._association_cue_fast_endpoint_keys(active_cue_entries),
        )
        preferred_episode_ids = [
            int(value)
            for value in (
                bundle.get("reranked_episode_ids")
                or bundle.get("episode_anchor_ids", [])
            )
        ]
        late_nodes, late_paths = self._late_learned_bridge_closure(
            question,
            preferred_episode_ids,
            traversed,
        )
        if late_nodes:
            traversed = [*traversed, *late_nodes]
            paths.extend(late_paths)
        candidate_episode_ids = [
            item.node_id for item in traversed if item.node_type == "episode"
        ]
        episodes, concepts = self._materialize_nodes(traversed, include_sources=True)
        episode_limit = min(self.config.retrieval.answer_episode_limit, len(episodes))
        selected_episodes, answer_paths = self._select_answer_evidence(
            episodes,
            paths,
            episode_limit,
            self.config.retrieval.answer_path_limit,
            set(active_cue_ids),
            question=question,
            preferred_episode_ids=[
                *preferred_episode_ids
            ],
            learned_bridge_slots=self.config.retrieval.learned_bridge_slots,
            learned_bridge_min_query_relevance=(
                self.config.retrieval.learned_bridge_min_query_relevance
            ),
            learned_bridge_duplicate_threshold=(
                self.config.retrieval.learned_bridge_duplicate_threshold
            ),
            coverage_groups=(
                bundle.get("rerank_trace", {})
                .get("merged_coverage", {})
                .get("coverage", [])
            ),
            preferred_association_scores=cue_scores,
        )
        chronology = self.chronology.order(
            [int(item["id"]) for item in selected_episodes]
        )
        episode_map = {int(item["id"]): item for item in selected_episodes}
        ordered_episodes = [
            episode_map[node_id]
            for node_id in chronology.ordered_ids
            if node_id in episode_map
        ]
        seed_episode_ids = {
            int(item["node_id"])
            for item in base_seed_payload
            if item["node_type"] == "episode"
        }
        cue_episode_ids = {
            int(endpoint[1])
            for entry in active_cue_entries
            for endpoint in entry.get("endpoints", [])
            if isinstance(endpoint, (list, tuple))
            and len(endpoint) == 2
            and endpoint[0] == "episode"
        }
        return {
            "question": question,
            "intent": dict(bundle["intent"]),
            "followup_search_queries": list(bundle.get("followup_queries", [])),
            "atomic_anchor_episode_ids": [
                int(value) for value in bundle.get("episode_anchor_ids", [])
            ],
            "reranked_episode_ids": [
                int(value) for value in bundle.get("reranked_episode_ids", [])
            ],
            "evidence_slot_trace": self._final_evidence_slot_trace(
                dict(bundle.get("rerank_trace", {})),
                [int(item["id"]) for item in ordered_episodes],
            ),
            "seed_episode_ids": sorted(seed_episode_ids),
            "candidate_episode_ids": candidate_episode_ids,
            "graph_added_episode_ids": [
                value for value in candidate_episode_ids if value not in seed_episode_ids
            ],
            "association_cue_ids": active_cue_ids,
            "association_cue_entries": active_cue_entries,
            "association_cue_added_episode_ids": sorted(
                cue_episode_ids.difference(seed_episode_ids)
            ),
            "episode_ids": [int(item["id"]) for item in ordered_episodes],
            "concept_ids": [
                int(item["id"])
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_ids": list(
                dict.fromkeys(
                    int(item["association_id"])
                    for item in answer_paths
                    if "association_id" in item
                )
            ),
            "chronology_notes": chronology.notes,
            "evidence_episodes": [
                {
                    key: item[key]
                    for key in (
                        "id",
                        "score",
                        "text",
                        "participants",
                        "source_key",
                        "segment_index",
                        "story_time_text",
                        "story_order",
                        "timeline_scope",
                        "evidence_origin",
                        "epistemic_status",
                        "generation",
                        "epistemic_note",
                    )
                }
                for item in ordered_episodes
            ],
            "evidence_concepts": [
                {
                    key: item[key]
                    for key in ("id", "score", "canonical_name", "description")
                }
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_paths": answer_paths,
            "paragraph_retrieval_enabled": self.paragraph_retrieval_enabled,
            "sparse_retrieval_enabled": self.sparse_retrieval_enabled,
        }

    def _plan_followup_queries(
        self,
        question: str,
        intent: QueryIntent,
        traversed,
    ) -> list[str]:
        episodes, concepts = self._materialize_nodes(traversed)
        with self._provider_purpose_scope("planner", "followup_query_planning"):
            payload = self.model.chat_json(
                HOP_QUERY_SYSTEM,
                hop_query_prompt(
                    question,
                    asdict(intent),
                    episodes[:30],
                    concepts[:20],
                ),
            )
        # The natural-language planner may return the requested questions as
        # a plain list. Normalize that shape locally instead of requiring the
        # model to reproduce a machine wrapper around readable text.
        raw_queries = (
            payload.get("followup_queries", [])
            if isinstance(payload, dict) else payload
        )
        if not isinstance(raw_queries, list):
            return []
        existing = {question, *intent.search_queries}
        followups: list[str] = []
        for raw_query in raw_queries:
            query = str(raw_query).strip()
            if query and query not in existing and query not in followups:
                followups.append(query)
        return followups[:8]

    def _followup_planning_decision(
        self,
        question: str,
        intent: QueryIntent,
    ) -> tuple[bool, str]:
        mode = str(
            self.config.retrieval.followup_planning_mode
        ).strip().casefold()
        if mode == "always":
            return True, "configured_always"
        if mode == "off":
            return False, "configured_off"
        if mode == "missing_slots":
            # This mode is resolved after initial retrieval.  It deliberately
            # avoids a speculative planner call before a local candidate pass
            # has established that a slot is actually absent.
            return False, "deferred_until_initial_slot_check"
        if mode != "entity_resolved":
            raise ValueError(f"unknown followup planning mode: {mode}")
        if len(intent.search_queries) >= 12:
            return True, "high_slot_count_safety_net"
        if requires_entity_resolved_followup(question, intent):
            return True, "answer_dependent_reference_detected"
        return False, "intent_queries_already_name_each_relation"

    @staticmethod
    def _initial_missing_followup_slots(
        initial_queries: list[str],
        initial_rankings: Mapping[str, object],
    ) -> list[str]:
        """Return initial evidence slots that have no local candidate.

        This is intentionally a narrow, pre-LLM signal: a slot is missing
        only when every recorded initial episode ranking lane is empty.  A
        low-ranked or otherwise imperfect candidate remains a candidate and
        must not trigger speculative follow-up planning.
        """

        slots = list(initial_queries[1:] or initial_queries[:1])
        ranking_lanes = (
            initial_rankings.get("fused_episode", []),
            initial_rankings.get("atomic_episode", []),
            initial_rankings.get("sparse_episode", []),
        )
        missing: list[str] = []
        for index, query in enumerate(slots):
            # Initial rankings include the whole question at index zero, then
            # the intent/constraint slots in the same order.
            ranking_index = index + 1 if len(initial_queries) > 1 else index
            has_candidate = any(
                isinstance(lane, list)
                and ranking_index < len(lane)
                and isinstance(lane[ranking_index], list)
                and bool(lane[ranking_index])
                for lane in ranking_lanes
            )
            if not has_candidate:
                missing.append(str(query))
        return missing

    @staticmethod
    def _validated_rerank_ids(
        payload: dict | None,
        field: str,
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        if not isinstance(payload, dict):
            return []
        raw_ids = payload.get(field, [])
        if not isinstance(raw_ids, list):
            return []
        result: list[int] = []
        for raw_id in raw_ids:
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id in allowed_ids and node_id not in result:
                result.append(node_id)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _coverage_rerank_ids(
        payload: dict | None,
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        groups = QueryEngine._coverage_groups(payload, allowed_ids)
        result: list[int] = []
        maximum = max((len(group) for group in groups), default=0)
        for rank_index in range(maximum):
            for group in groups:
                if rank_index < len(group) and group[rank_index] not in result:
                    result.append(group[rank_index])
                    if len(result) >= limit:
                        return result
        return result

    @staticmethod
    def _coverage_groups(
        payload: dict | None,
        allowed_ids: set[int],
    ) -> list[list[int]]:
        if not isinstance(payload, dict):
            return []
        coverage = payload.get("coverage", [])
        if not isinstance(coverage, list):
            return []
        groups: list[list[int]] = []
        for item in coverage:
            if not isinstance(item, dict) or not isinstance(
                item.get("episode_ids"), list
            ):
                continue
            group: list[int] = []
            for raw_id in item["episode_ids"]:
                try:
                    node_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if node_id in allowed_ids and node_id not in group:
                    group.append(node_id)
            if group:
                if str(item.get("mode", "alternatives")).lower() == "joint":
                    groups.extend([[node_id] for node_id in group[:5]])
                else:
                    groups.append(group[:5])
        return groups

    @staticmethod
    def _merge_coverage_payloads(
        primary: dict | None,
        supplemental: dict | None,
    ) -> dict:
        """Interleave two independent passes and union evidence for equal slots."""
        entries: dict[str, dict] = {}
        positions: dict[str, int] = {}
        for pass_index, payload in enumerate((primary, supplemental)):
            if not isinstance(payload, dict) or not isinstance(
                payload.get("coverage"), list
            ):
                continue
            for item_index, item in enumerate(payload["coverage"]):
                if not isinstance(item, dict):
                    continue
                raw_ids = item.get("episode_ids", [])
                if not isinstance(raw_ids, list):
                    continue
                ids: list[int] = []
                for raw_id in raw_ids:
                    try:
                        node_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if node_id not in ids:
                        ids.append(node_id)
                normalized = {
                    "query": str(item.get("query", "")).strip(),
                    "mode": (
                        "joint"
                        if str(item.get("mode", "alternatives")).lower()
                        == "joint"
                        else "alternatives"
                    ),
                    "episode_ids": ids,
                    "reason": str(item.get("reason", "")),
                }
                key = re.sub(
                    r"[（(][^）)]*[）)]",
                    "",
                    normalized["query"],
                )
                key = re.sub(r"\s+", "", key).casefold()
                position = item_index * 2 + pass_index
                if key not in entries:
                    entries[key] = normalized
                    positions[key] = position
                    continue
                existing = entries[key]
                if (
                    existing["mode"] != "joint"
                    or normalized["mode"] != "joint"
                ):
                    existing["mode"] = "alternatives"
                for node_id in normalized["episode_ids"]:
                    if node_id not in existing["episode_ids"]:
                        existing["episode_ids"].append(node_id)
                if (
                    normalized["reason"]
                    and normalized["reason"] not in existing["reason"]
                ):
                    existing["reason"] = (
                        f"{existing['reason']} | 独立侦察：{normalized['reason']}"
                    ).strip(" |")
                positions[key] = min(positions[key], position)
        coverage = [
            entries[key]
            for key in sorted(entries, key=lambda item: positions[item])
        ]
        missing: list[str] = []
        for payload in (primary, supplemental):
            if not isinstance(payload, dict) or not isinstance(
                payload.get("missing_aspects"), list
            ):
                continue
            for value in payload["missing_aspects"]:
                item = str(value).strip()
                if item and item not in missing:
                    missing.append(item)
        return {"coverage": coverage, "missing_aspects": missing}

    @staticmethod
    def _enforce_coverage_selection(
        payload: dict | None,
        selected_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]]]:
        """Keep one candidate for each declared evidence slot when possible."""
        if limit <= 0:
            return [], []
        result = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in selected_ids)
            if node_id in allowed_ids
        ][:limit]
        groups = QueryEngine._coverage_groups(payload, allowed_ids)

        # A Top-K list cannot represent more than K mutually disjoint slots.
        # Coverage order follows the user's question, so the first K slots win.
        protected_groups = groups[:limit]
        changes: list[dict[str, int]] = []
        for group_index, group in enumerate(protected_groups):
            if any(node_id in result for node_id in group):
                continue
            add_id = group[0]
            if len(result) < limit:
                result.append(add_id)
                changes.append({"remove_id": -1, "add_id": add_id})
                continue
            previous_groups = protected_groups[:group_index]
            replacement_index = None
            for index in range(len(result) - 1, -1, -1):
                trial = result[:index] + result[index + 1 :] + [add_id]
                if all(any(node_id in trial for node_id in prior) for prior in previous_groups):
                    replacement_index = index
                    break
            if replacement_index is None:
                continue
            remove_id = result[replacement_index]
            result[replacement_index] = add_id
            changes.append({"remove_id": remove_id, "add_id": add_id})
        return list(dict.fromkeys(result))[:limit], changes

    @staticmethod
    def _enforce_final_selection_constraints(
        selected_ids: list[int],
        required_ids: list[int],
        coverage_payload: dict | None,
        allowed_ids: set[int],
        coverage_allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]], list[dict[str, int]]]:
        """Apply cheap recall floors before the stronger coverage contract.

        Atomic floors deliberately trade precision for recall.  They must not
        get the final opportunity to evict a unique Episode chosen by the
        independent evidence coverage passes.
        """
        with_floor, floor_changes = QueryEngine._enforce_selection_floor(
            selected_ids,
            required_ids,
            allowed_ids,
            limit,
        )
        final_ids, coverage_changes = QueryEngine._enforce_coverage_selection(
            coverage_payload,
            with_floor,
            coverage_allowed_ids,
            limit,
        )
        return final_ids, floor_changes, coverage_changes

    @staticmethod
    def _apply_rerank_replacements(
        payload: dict | None,
        selected_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        """Execute valid replacement directives even if the model's final list drifts."""
        result = list(selected_ids)
        if not isinstance(payload, dict):
            return result
        replacements = payload.get("replacements", [])
        if not isinstance(replacements, list):
            return result
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            try:
                add_id = int(replacement.get("add_id"))
            except (TypeError, ValueError):
                continue
            if add_id not in allowed_ids or add_id in result:
                continue
            try:
                remove_id = int(replacement.get("remove_id"))
            except (TypeError, ValueError):
                remove_id = -1
            if remove_id in result:
                result[result.index(remove_id)] = add_id
            elif len(result) < limit:
                result.append(add_id)
            elif result:
                result[-1] = add_id
        return list(dict.fromkeys(result))[:limit]

    def _rerank_review_level(
        self,
        atomic_queries: list[str],
        initial: dict | None,
        allowed_ids: set[int],
        question: str = "",
    ) -> tuple[str, list[str]]:
        """Choose the cheapest evidence review path that fits visible risk."""
        mode = str(self.config.retrieval.rerank_review_mode).casefold()
        if mode == "strict":
            return "strict", ["configured_strict"]
        if mode == "lean":
            return "none", ["configured_lean"]
        if mode not in {"adaptive", "fast_adaptive"}:
            raise ValueError(f"unknown rerank review mode: {mode}")
        reasons: list[str] = []
        if not isinstance(initial, dict):
            return "strict", ["invalid_initial_payload"]
        # The reply path needs a bounded tail.  It trusts a complete initial
        # coverage map, but still spends one compressor call when the first
        # pass reports a visible gap, has too many atomic slots, or is too
        # thin.  Background/deep retrieval keeps the stronger independent
        # audit policy below.
        if mode == "fast_adaptive":
            missing = initial.get("missing_aspects", [])
            reported_missing = isinstance(missing, list) and any(
                str(item).strip() for item in missing
            )
            if reported_missing:
                reasons.append("initial_reported_missing_aspects")
            if len(atomic_queries) >= max(
                1,
                int(
                    self.config.retrieval.rerank_strict_atomic_query_threshold
                ),
            ):
                reasons.append("many_atomic_queries")
            coverage_group_count = len(
                self._coverage_groups(initial, allowed_ids)
            )
            if coverage_group_count <= max(
                0,
                int(
                    self.config.retrieval.rerank_compress_coverage_group_threshold
                ),
            ):
                reasons.append("thin_initial_coverage")
            # A compressor receives the same evidence shortlist.  A declared
            # source gap alone cannot be repaired by asking that model to
            # compress the list again; preserve it as answer-time uncertainty.
            # Compression is still useful for a structurally thin first pass
            # or an unusually large set of genuinely independent slots.
            if any(
                reason in {"many_atomic_queries", "thin_initial_coverage"}
                for reason in reasons
            ):
                return "compress", reasons
            if reported_missing:
                return "none", [
                    *reasons,
                    "reported_gap_preserved_as_uncertainty",
                ]
            return "none", ["initial_coverage_sufficient"]
        if self._requires_strict_evidence_review(question):
            reasons.append("high_risk_contrast_causality_or_projection")
        missing = initial.get("missing_aspects", [])
        if isinstance(missing, list) and any(str(item).strip() for item in missing):
            reasons.append("initial_reported_missing_aspects")
        if len(atomic_queries) >= max(
            1, int(self.config.retrieval.rerank_strict_atomic_query_threshold)
        ):
            reasons.append("many_atomic_queries")
        if reasons:
            return "strict", reasons
        coverage_group_count = len(self._coverage_groups(initial, allowed_ids))
        if coverage_group_count <= max(
            0,
            int(
                self.config.retrieval.rerank_compress_coverage_group_threshold
            ),
        ):
            return "compress", ["thin_initial_coverage"]
        return "none", ["initial_coverage_sufficient"]

    @staticmethod
    def _requires_strict_evidence_review(question: str) -> bool:
        explicit_risk = bool(
            re.search(
                r"为什么|为何|原因|因果|身份|属于|关系|"
                r"比较|对比|先后|第一次|最后|是否|有没有|"
                r"推断|证明|反驳|否定",
                question,
            )
        )
        # Two or more independently requested facts require coverage-aware
        # selection even when none of the old high-risk keywords is present.
        # This changes evidence protection, not workload routing.
        claim_markers = re.findall(
            r"谁|哪(?:位|个|些|一)|什么|为何|为什么|怎么|如何|是否|有没有",
            question,
        )
        return explicit_risk or len(claim_markers) >= 2

    def _atomic_evidence_floor_ids(
        self,
        question: str,
        atomic_rankings: list[list[dict]],
    ) -> list[int]:
        retrieval = self.config.retrieval
        if (
            not retrieval.rerank_atomic_floor_enabled
            or not self._requires_strict_evidence_review(question)
        ):
            return []
        query_limit = max(0, int(retrieval.rerank_atomic_floor_query_limit))
        per_query = max(0, int(retrieval.rerank_atomic_floor_per_query))
        total_limit = max(0, int(retrieval.rerank_atomic_floor_total_limit))
        rankings = atomic_rankings[:query_limit]
        selected: list[int] = []
        for rank_index in range(per_query):
            for ranking in rankings:
                if rank_index >= len(ranking):
                    continue
                episode_id = int(ranking[rank_index]["id"])
                if episode_id not in selected:
                    selected.append(episode_id)
                if len(selected) >= total_limit:
                    return selected
        return selected

    @staticmethod
    def _balanced_atomic_floor_rankings(
        initial_rankings: list[list[dict]],
        followup_rankings: list[list[dict]],
        query_limit: int,
    ) -> list[list[dict]]:
        """Reserve part of the floor for entity-resolved follow-up hops."""
        limit = max(0, int(query_limit))
        if not limit:
            return []
        initial = list(initial_rankings)
        followup = list(followup_rankings)
        if not initial or not followup:
            return (initial or followup)[:limit]

        followup_budget = min(len(followup), max(1, limit // 3))
        initial_budget = min(len(initial), limit - followup_budget)
        remaining = limit - initial_budget - followup_budget
        if remaining:
            extra_initial = min(len(initial) - initial_budget, remaining)
            initial_budget += extra_initial
            remaining -= extra_initial
        if remaining:
            followup_budget += min(
                len(followup) - followup_budget,
                remaining,
            )
        return [
            *initial[:initial_budget],
            *followup[:followup_budget],
        ]

    def _hybrid_evidence_floor_ids(
        self,
        question: str,
        initial_rankings: dict[str, list],
        followup_rankings: dict[str, list],
        initial_queries: list[str] | None = None,
        followup_queries: list[str] | None = None,
    ) -> list[int]:
        return self._hybrid_evidence_floor_trace(
            question,
            initial_rankings,
            followup_rankings,
            initial_queries,
            followup_queries,
        )["selected_episode_ids"]

    @staticmethod
    def _is_constraint_candidate_query(query: str) -> bool:
        normalized = str(query).strip()
        return normalized.startswith(
            ("__constraint_slot__ ", "__answer_slot__ ")
        )

    def _constraint_candidate_ids(
        self,
        question: str,
        initial_queries: list[str],
        initial_rankings: dict[str, list],
        followup_queries: list[str],
        followup_rankings: dict[str, list],
    ) -> list[int]:
        """Reserve rerank admission for late causal/identity subproblems.

        These IDs are candidate-only: the evidence reranker must still decide
        whether they support the question. Follow-up queries are inspected
        first because they are produced after first-hop evidence has made the
        missing relation more specific.
        """

        has_explicit_slot = any(
            str(query).startswith(("__constraint_slot__ ", "__answer_slot__ "))
            for query in [
                *(initial_queries or []),
                *(followup_queries or []),
            ]
        )
        if (
            not self._requires_strict_evidence_review(question)
            and not has_explicit_slot
        ):
            return []
        per_query = max(
            0,
            int(self.config.retrieval.rerank_constraint_candidate_per_query),
        )
        total_limit = max(
            0,
            int(self.config.retrieval.rerank_constraint_candidate_total_limit),
        )
        if not per_query or not total_limit:
            return []

        selected: list[int] = []
        groups = (
            (followup_queries, followup_rankings),
            (initial_queries, initial_rankings),
        )
        for queries, rankings in groups:
            fused = rankings.get("fused_episode", [])
            atomic = rankings.get("atomic_episode", [])
            for index, query in enumerate(queries):
                if not self._is_constraint_candidate_query(query):
                    continue
                rows = (
                    fused[index]
                    if index < len(fused)
                    else atomic[index]
                    if index < len(atomic)
                    else []
                )
                for item in rows[:per_query]:
                    episode_id = int(item["id"])
                    if episode_id not in selected:
                        selected.append(episode_id)
                    if len(selected) >= total_limit:
                        return selected
        return selected

    def _hybrid_evidence_floor_trace(
        self,
        question: str,
        initial_rankings: dict[str, list],
        followup_rankings: dict[str, list],
        initial_queries: list[str] | None = None,
        followup_queries: list[str] | None = None,
    ) -> dict:
        retrieval = self.config.retrieval
        has_explicit_slot = any(
            str(query).startswith(("__constraint_slot__ ", "__answer_slot__ "))
            for query in [
                *(initial_queries or []),
                *(followup_queries or []),
            ]
        )
        if (
            not self._requires_strict_evidence_review(question)
            and not has_explicit_slot
        ):
            return {
                "enabled": False,
                "constraint_slots": [],
                "atomic_slots": [],
                "whole_question_sparse_ids": [],
                "atomic_floor_ids": [],
                "selected_episode_ids": [],
            }
        sparse_limit = max(
            0, int(retrieval.rerank_question_sparse_floor_limit)
        )
        sparse_rankings = initial_rankings.get("sparse_episode", [])
        whole_question_sparse = sparse_rankings[0] if sparse_rankings else []
        sparse_ids = [
            int(item["id"])
            for item in whole_question_sparse[:sparse_limit]
        ]
        atomic_floor_rankings = self._balanced_atomic_floor_rankings(
            initial_rankings.get("atomic_episode", []),
            followup_rankings.get("atomic_episode", []),
            int(retrieval.rerank_atomic_floor_query_limit),
        )
        atomic_ids = self._atomic_evidence_floor_ids(
            question,
            atomic_floor_rankings,
        )
        query_limit = max(0, int(retrieval.rerank_atomic_floor_query_limit))
        per_atomic_query = max(
            0, int(retrieval.rerank_atomic_floor_per_query)
        )
        initial_pairs = list(
            zip(
                list(initial_queries or []),
                initial_rankings.get("atomic_episode", []),
            )
        )
        followup_pairs = list(
            zip(
                list(followup_queries or []),
                followup_rankings.get("atomic_episode", []),
            )
        )
        if initial_pairs and followup_pairs and query_limit:
            followup_budget = min(
                len(followup_pairs), max(1, query_limit // 3)
            )
            initial_budget = min(
                len(initial_pairs), query_limit - followup_budget
            )
            remaining = query_limit - initial_budget - followup_budget
            if remaining:
                extra_initial = min(
                    len(initial_pairs) - initial_budget, remaining
                )
                initial_budget += extra_initial
                remaining -= extra_initial
            if remaining:
                followup_budget += min(
                    len(followup_pairs) - followup_budget, remaining
                )
            atomic_pairs = [
                *initial_pairs[:initial_budget],
                *followup_pairs[:followup_budget],
            ]
        else:
            atomic_pairs = (initial_pairs or followup_pairs)[:query_limit]
        atomic_id_set = set(atomic_ids)
        atomic_slots = [
            {
                "query": str(query),
                "candidate_episode_ids": [
                    int(item["id"])
                    for item in ranking[:per_atomic_query]
                ],
                "floor_episode_ids": [
                    int(item["id"])
                    for item in ranking[:per_atomic_query]
                    if int(item["id"]) in atomic_id_set
                ],
            }
            for query, ranking in atomic_pairs
        ]
        constraint_slots: list[dict] = []
        constraint_ids: list[int] = []
        per_query = max(
            0, int(retrieval.rerank_constraint_floor_per_query)
        )
        constraint_total = max(
            0, int(retrieval.rerank_constraint_floor_total_limit)
        )
        query_ranking_groups = (
            (
                list(initial_queries or []),
                initial_rankings,
            ),
            (
                list(followup_queries or []),
                followup_rankings,
            ),
        )
        active_groups = (
            query_ranking_groups if per_query and constraint_total else ()
        )
        slot_candidates: list[tuple[int, int, str, list[dict]]] = []
        for group_index, (queries, rankings) in enumerate(active_groups):
            fused = rankings.get("fused_episode", [])
            atomic = rankings.get("atomic_episode", [])
            sparse = rankings.get("sparse_episode", [])
            for index, query in enumerate(queries):
                if (
                    not query.startswith(
                        ("__constraint_slot__ ", "__answer_slot__ ")
                    )
                ):
                    continue
                rows = (
                    fused[index]
                    if index < len(fused) and fused[index]
                    else atomic[index]
                    if index < len(atomic) and atomic[index]
                    else sparse[index]
                    if index < len(sparse)
                    else []
                )
                slot_candidates.append(
                    (
                        0 if query.startswith("__answer_slot__ ") else 1,
                        group_index,
                        query,
                        rows,
                    )
                )
        for _priority, _group_index, query, rows in sorted(
            slot_candidates, key=lambda item: (item[0], item[1])
        ):
                candidates = [int(item["id"]) for item in rows[:per_query]]
                admitted: list[int] = []
                for node_id in candidates:
                    if node_id not in constraint_ids:
                        constraint_ids.append(node_id)
                        admitted.append(node_id)
                    if len(constraint_ids) >= constraint_total:
                        break
                constraint_slots.append(
                    {
                        "query": query,
                        "candidate_episode_ids": candidates,
                        "floor_episode_ids": admitted,
                    }
                )
                if len(constraint_ids) >= constraint_total:
                    break
        combined_limit = max(0, int(retrieval.rerank_combined_floor_limit))
        # Whole-question sparse results are high-recall candidates, not proof
        # that every top lexical match belongs in the final answer.  Hard-floor
        # only the bounded typed/atomic slots and leave the remaining budget to
        # the evidence selector.
        selected = list(
            dict.fromkeys([*constraint_ids, *atomic_ids])
        )[:combined_limit]
        return {
            "enabled": True,
            "constraint_slots": constraint_slots,
            "atomic_slots": atomic_slots,
            "whole_question_sparse_ids": sparse_ids,
            "atomic_floor_ids": atomic_ids,
            "selected_episode_ids": selected,
        }

    @staticmethod
    def _final_evidence_slot_trace(
        rerank_trace: dict,
        final_episode_ids: list[int],
    ) -> dict:
        """Explain which deterministic/LLM evidence slots reached the answer."""
        final_ids = {int(value) for value in final_episode_ids}
        deterministic = deepcopy(
            rerank_trace.get("deterministic_evidence_floor", {})
        )
        for slot in deterministic.get("constraint_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["final_matched_episode_ids"] = [
                value for value in floor_ids if value in final_ids
            ]
            slot["satisfied"] = bool(slot["final_matched_episode_ids"])
        for slot in deterministic.get("atomic_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["final_matched_episode_ids"] = [
                value for value in floor_ids if value in final_ids
            ]
            slot["satisfied"] = bool(slot["final_matched_episode_ids"])
        selected_floor_ids = [
            int(value)
            for value in deterministic.get("selected_episode_ids", [])
        ]
        deterministic["final_matched_episode_ids"] = [
            value for value in selected_floor_ids if value in final_ids
        ]
        deterministic["missing_floor_episode_ids"] = [
            value for value in selected_floor_ids if value not in final_ids
        ]
        coverage_slots: list[dict] = []
        coverage = (
            rerank_trace.get("merged_coverage", {}).get("coverage", [])
        )
        coverage_items = coverage if isinstance(coverage, list) else []
        for item in coverage_items:
            if not isinstance(item, dict):
                continue
            candidate_ids = [
                int(value) for value in item.get("episode_ids", [])
            ]
            matched = [value for value in candidate_ids if value in final_ids]
            coverage_slots.append(
                {
                    "query": str(item.get("query", "")),
                    "mode": str(item.get("mode", "alternatives")),
                    "candidate_episode_ids": candidate_ids,
                    "final_matched_episode_ids": matched,
                    "satisfied": bool(matched),
                }
            )
        return {
            "final_episode_ids": sorted(final_ids),
            "deterministic": deterministic,
            "coverage_slots": coverage_slots,
        }

    @staticmethod
    def _admit_evidence_floor_trace(
        floor_trace: dict,
        admitted_episode_ids: list[int],
    ) -> dict:
        """Separate planned floor hits from rows admitted to rerank candidates."""
        result = deepcopy(floor_trace)
        admitted = {int(value) for value in admitted_episode_ids}
        planned = [
            int(value) for value in result.get("selected_episode_ids", [])
        ]
        result["planned_episode_ids"] = planned
        result["selected_episode_ids"] = [
            value for value in planned if value in admitted
        ]
        result["candidate_missing_episode_ids"] = [
            value for value in planned if value not in admitted
        ]
        for slot in result.get("constraint_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["planned_floor_episode_ids"] = floor_ids
            slot["floor_episode_ids"] = [
                value for value in floor_ids if value in admitted
            ]
            slot["candidate_missing_episode_ids"] = [
                value for value in floor_ids if value not in admitted
            ]
        for slot in result.get("atomic_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["planned_floor_episode_ids"] = floor_ids
            slot["floor_episode_ids"] = [
                value for value in floor_ids if value in admitted
            ]
            slot["candidate_missing_episode_ids"] = [
                value for value in floor_ids if value not in admitted
            ]
        return result

    @staticmethod
    def _enforce_selection_floor(
        selected_ids: list[int],
        required_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]]]:
        required = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in required_ids)
            if node_id in allowed_ids
        ][:limit]
        required_set = set(required)
        result = list(dict.fromkeys(int(value) for value in selected_ids))[:limit]
        changes: list[dict[str, int]] = []
        for add_id in required:
            if add_id in result:
                continue
            if len(result) < limit:
                result.append(add_id)
                changes.append({"add_id": add_id, "remove_id": -1})
                continue
            remove_id = next(
                (
                    node_id
                    for node_id in reversed(result)
                    if node_id not in required_set
                ),
                None,
            )
            if remove_id is None:
                break
            result[result.index(remove_id)] = add_id
            changes.append({"add_id": add_id, "remove_id": remove_id})
        return list(dict.fromkeys(result))[:limit], changes

    @staticmethod
    def _answer_slot_neighbor_floor_ids(
        anchor_ids: list[int],
        episodes: list[dict],
        radius: int,
        total_limit: int,
    ) -> list[int]:
        """Protect bounded narrative neighbors around explicit answer slots."""
        if radius <= 0 or total_limit <= 0 or not anchor_ids:
            return []
        episode_by_id = {int(item["id"]): item for item in episodes}
        by_source: dict[str, list[int]] = {}
        for item in episodes:
            source_key = str(item.get("source_key", "")).strip()
            if not source_key:
                continue
            by_source.setdefault(source_key, []).append(int(item["id"]))
        for ids in by_source.values():
            ids.sort()

        selected: list[int] = []
        for anchor_id in dict.fromkeys(int(value) for value in anchor_ids):
            anchor = episode_by_id.get(anchor_id)
            if anchor is None:
                continue
            siblings = by_source.get(str(anchor.get("source_key", "")).strip(), [])
            try:
                position = siblings.index(anchor_id)
            except ValueError:
                continue
            for distance in range(1, radius + 1):
                for neighbor_index in (position - distance, position + distance):
                    if not 0 <= neighbor_index < len(siblings):
                        continue
                    neighbor_id = siblings[neighbor_index]
                    if neighbor_id not in selected and neighbor_id not in anchor_ids:
                        selected.append(neighbor_id)
                    if len(selected) >= total_limit:
                        return selected
        return selected

    @staticmethod
    def _precompress_rerank_candidate_ids(
        candidate_pool_ids: list[int],
        preferred_ids: list[int],
        required_ids: list[int],
        limit: int,
    ) -> tuple[list[int], list[dict]]:
        """Deterministically reduce Candidate@N without erasing answer slots.

        ``candidate_pool_ids`` is already a one-global/two-atomic interleave,
        while ``preferred_ids`` is round-robin across the atomic retrieval
        queries.  Required floor rows are admitted first; the remainder keeps
        the established pool order.  Consequently compression adds no model
        judgement and is exactly replayable from the trace.
        """
        pool = list(dict.fromkeys(int(value) for value in candidate_pool_ids))
        maximum = min(len(pool), max(0, int(limit)))
        preferred = set(int(value) for value in preferred_ids)
        required = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in required_ids)
            if node_id in pool
        ]
        if maximum >= len(pool):
            selected = list(pool)
        else:
            selected = required[:maximum]
            for node_id in pool:
                if len(selected) >= maximum:
                    break
                if node_id not in selected:
                    selected.append(node_id)

        selected_set = set(selected)
        required_set = set(required)
        decisions = []
        for pool_rank, node_id in enumerate(pool, start=1):
            if node_id not in selected_set:
                reason = "dropped_budget"
            elif node_id in required_set:
                reason = "required_evidence_floor"
            elif node_id in preferred:
                reason = "atomic_anchor"
            else:
                reason = "global_rank"
            decisions.append(
                {
                    "episode_id": node_id,
                    "pool_rank": pool_rank,
                    "kept": node_id in selected_set,
                    "reason": reason,
                }
            )
        return selected, decisions

    @staticmethod
    def _association_row_value(row, key: str, default=None):
        """Read sqlite3.Row and dict fixtures through one narrow boundary."""

        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return default

    def _association_cue_fast_rerank(
        self,
        *,
        episodes: list[dict],
        limit: int,
        required_ids: list[int],
        preferred_ids: list[int],
        association_cue_entries: list[dict],
    ) -> tuple[list[int], dict] | None:
        """Reuse a learned relation as a local, evidence-bound rerank cache.

        Relation text is only the lookup key.  It is never promoted to source
        evidence: both endpoints must be direct/source-grounded Episodes, and
        both are passed to the answer model.  Any failed guard falls back to
        the configured cross-encoder or LLM reranker.
        """

        retrieval = self.config.retrieval
        if (
            not retrieval.association_cue_fast_path_enabled
            or not association_cue_entries
            or limit < 2
        ):
            return None
        episode_by_id = {int(item["id"]): item for item in episodes}
        accepted: list[dict] = []
        endpoint_ids: list[int] = []
        ordered_entries = sorted(
            association_cue_entries,
            key=lambda item: float(item.get("cosine", 0.0)),
            reverse=True,
        )
        if len(ordered_entries) >= 2 and (
            float(ordered_entries[0].get("cosine", 0.0))
            - float(ordered_entries[1].get("cosine", 0.0))
            < float(retrieval.association_cue_fast_path_min_margin)
        ):
            return None
        for entry in ordered_entries:
            if len(accepted) >= max(
                1, int(retrieval.association_cue_fast_path_max_edges)
            ):
                break
            cosine = float(entry.get("cosine", 0.0))
            if str(entry.get("score_kind", "")).startswith("local_"):
                if float(entry.get("local_coverage", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_coverage
                ) or float(entry.get("local_margin", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_margin
                ):
                    continue
            elif cosine < float(
                retrieval.association_cue_fast_path_min_similarity
            ):
                continue
            association_id = int(entry.get("association_id", -1))
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            if float(
                self._association_row_value(row, "confidence", 0.0) or 0.0
            ) < float(retrieval.association_cue_fast_path_min_confidence):
                continue
            if not self._association_has_fast_path_confirmation(row):
                continue
            endpoints = [
                (
                    str(self._association_row_value(row, "from_type", "")),
                    int(self._association_row_value(row, "from_id", -1)),
                ),
                (
                    str(self._association_row_value(row, "to_type", "")),
                    int(self._association_row_value(row, "to_id", -1)),
                ),
            ]
            if any(node_type != "episode" for node_type, _ in endpoints):
                continue
            candidate_endpoint_ids = [node_id for _, node_id in endpoints]
            endpoint_rows = [
                episode_by_id.get(node_id) for node_id in candidate_endpoint_ids
            ]
            if any(item is None for item in endpoint_rows):
                continue
            if any(
                int(item.get("generation", 0) or 0) != 0
                or str(item.get("evidence_origin", "")).casefold()
                not in {"source", "direct", "imported"}
                for item in endpoint_rows
                if item is not None
            ):
                continue
            accepted.append(
                {
                    "association_id": association_id,
                    "cosine": cosine,
                    "score_kind": str(entry.get("score_kind", "embedding")),
                    "relation_type": str(
                        self._association_row_value(row, "relation_type", "")
                    ),
                    "relation_key": str(
                        self._association_row_value(row, "relation_key", "")
                    ),
                    "relation_text": str(
                        self._association_row_value(row, "relation_text", "")
                    )[:900],
                    "endpoint_episode_ids": candidate_endpoint_ids,
                    "generation": int(
                        self._association_row_value(row, "generation", 0) or 0
                    ),
                    "confidence": float(
                        self._association_row_value(row, "confidence", 0.0)
                        or 0.0
                    ),
                }
            )
            endpoint_ids.extend(candidate_endpoint_ids)
        endpoint_ids = list(dict.fromkeys(endpoint_ids))
        if len(endpoint_ids) < 2:
            return None
        protected_ids = list(dict.fromkeys([*endpoint_ids, *required_ids]))
        if len(protected_ids) > limit:
            return None
        global_ids = [int(item["id"]) for item in episodes]
        selected_ids = list(
            dict.fromkeys(
                [*endpoint_ids, *required_ids, *preferred_ids, *global_ids]
            )
        )[:limit]
        trace = {
            "enabled": True,
            "backend": "association_capsule",
            "model": "local_audited_edge_selector",
            "review_mode": "local",
            "review_level": "association_cache",
            "coverage_audit_performed": False,
            "compressor_performed": False,
            "cache_hit": True,
            "cache_kind": "audited_association_cue",
            "cloud_requests_avoided": 1,
            "candidate_episode_ids": global_ids,
            "rerank_input_episode_ids": selected_ids,
            "required_evidence_floor_ids": list(required_ids),
            "atomic_evidence_floor_ids": list(required_ids),
            "required_evidence_floor_replacements": [],
            "atomic_evidence_floor_replacements": [],
            "association_capsules": accepted,
            "final_episode_ids": selected_ids,
        }
        if self.logger:
            self.logger.emit(
                "association_cue_fast_rerank",
                trace=trace,
            )
        return selected_ids, trace

    @staticmethod
    def _rerank_candidate_input_fingerprint(
        episodes: Sequence[Mapping[str, object]],
        paragraph_context_by_source: Mapping[int, Sequence[Mapping[str, object]]]
        | None = None,
    ) -> str:
        """Bind reusable rerank output to its exact local candidate input.

        The digest covers IDs, order, text and request-visible paragraph
        contexts. It intentionally records no text itself. A frozen rerank
        result is reusable only when this fingerprint matches; a candidate
        treatment otherwise stays a local candidate diagnostic instead of
        inheriting an LLM judgement made for a different candidate list.
        """

        normalized_episodes: list[dict[str, object]] = []
        for position, episode in enumerate(episodes, start=1):
            try:
                episode_id = int(episode.get("id", 0))
            except (TypeError, ValueError):
                continue
            if episode_id <= 0:
                continue
            text = str(episode.get("text", ""))
            normalized_episodes.append(
                {
                    "position": position,
                    "episode_id": episode_id,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "source_id": int(episode.get("source_id", 0) or 0),
                    "source_key_sha256": hashlib.sha256(
                        str(episode.get("source_key", "")).encode("utf-8")
                    ).hexdigest(),
                    "participants": [str(item) for item in episode.get("participants", [])],
                    "story_time_text": str(episode.get("story_time_text", "")),
                }
            )
        contexts: list[dict[str, object]] = []
        for source_id, rows in sorted((paragraph_context_by_source or {}).items()):
            for position, row in enumerate(rows, start=1):
                contexts.append(
                    {
                        "source_id": int(source_id),
                        "position": position,
                        "paragraph_id": int(row.get("paragraph_id", 0) or 0),
                        "text_sha256": hashlib.sha256(
                            str(row.get("text", "")).encode("utf-8")
                        ).hexdigest(),
                    }
                )
        payload = json.dumps(
            {"version": 1, "episodes": normalized_episodes, "contexts": contexts},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def _rerank_answer_episodes(
        self,
        question: str,
        intent: QueryIntent,
        atomic_queries: list[str],
        episodes: list[dict],
        limit: int,
        preferred_candidate_ids: list[int] | None = None,
        paragraph_context_by_source: dict[int, list[dict]] | None = None,
        result_cache: dict[str, tuple[list[int], dict]] | None = None,
        required_candidate_ids: list[int] | None = None,
        answer_slot_anchor_ids: list[int] | None = None,
        association_cue_entries: list[dict] | None = None,
    ) -> tuple[list[int], dict]:
        """Use an evidence-bound LLM pass to compress Candidate@100 to Top-20."""
        if (
            not self.config.retrieval.rerank_enabled
            or self.model is None
            or limit <= 0
        ):
            return [], {"enabled": False}
        expanded_atomic_queries = expand_rerank_atomic_queries(
            atomic_queries,
            intent,
            question,
        )
        atomic_queries = limit_rerank_atomic_queries(
            expanded_atomic_queries,
            int(self.config.retrieval.rerank_atomic_query_limit),
        )
        candidate_limit = max(limit, self.config.retrieval.rerank_candidate_limit)
        episode_by_id = {int(item["id"]): item for item in episodes}
        global_ids = [int(item["id"]) for item in episodes]
        required_ids = [
            int(value)
            for value in dict.fromkeys(required_candidate_ids or [])
            if int(value) in episode_by_id
        ]
        answer_slot_neighbor_ids = self._answer_slot_neighbor_floor_ids(
            list(answer_slot_anchor_ids or []),
            episodes,
            max(
                0,
                int(
                    self.config.retrieval.rerank_answer_slot_neighbor_radius
                ),
            ),
            max(
                0,
                int(
                    self.config.retrieval.rerank_answer_slot_neighbor_total_limit
                ),
            ),
        )
        required_ids = list(
            dict.fromkeys([*required_ids, *answer_slot_neighbor_ids])
        )
        preferred_ids = [
            int(value)
            for value in dict.fromkeys(
                [*required_ids, *(preferred_candidate_ids or [])]
            )
            if int(value) in episode_by_id
        ]
        cue_fast_result = self._association_cue_fast_rerank(
            episodes=episodes,
            limit=limit,
            required_ids=required_ids,
            preferred_ids=preferred_ids,
            association_cue_entries=association_cue_entries or [],
        )
        if cue_fast_result is not None:
            return cue_fast_result
        # Atomic slots are the scarce resource in multi-hop questions. One
        # global candidate followed by two atomic-channel candidates preserves
        # the broad lane while representing more independent subproblems.
        ordered_candidate_ids: list[int] = []
        global_index = 0
        preferred_index = 0
        while len(ordered_candidate_ids) < candidate_limit:
            added = False
            for _ in range(1):
                while (
                    global_index < len(global_ids)
                    and global_ids[global_index] in ordered_candidate_ids
                ):
                    global_index += 1
                if global_index < len(global_ids):
                    ordered_candidate_ids.append(global_ids[global_index])
                    global_index += 1
                    added = True
            for _ in range(2):
                while (
                    preferred_index < len(preferred_ids)
                    and preferred_ids[preferred_index] in ordered_candidate_ids
                ):
                    preferred_index += 1
                if preferred_index < len(preferred_ids):
                    ordered_candidate_ids.append(preferred_ids[preferred_index])
                    preferred_index += 1
                    added = True
            if not added:
                break
        candidate_pool_ids = ordered_candidate_ids[:candidate_limit]
        rerank_backend = str(
            self.config.retrieval.rerank_backend
        ).strip().casefold()
        if rerank_backend not in {"llm", "cross_encoder"}:
            raise ValueError(f"unknown rerank backend: {rerank_backend}")
        rerank_input_limit = min(
            len(candidate_pool_ids),
            max(
                limit,
                int(self.config.retrieval.rerank_precompression_limit),
            ),
        )
        precompression_pool_ids = list(candidate_pool_ids)
        bge_prefilter: dict = {"performed": False}
        reranker_model = str(
            getattr(self.config.model, "reranker_model", "") or ""
        ).strip()
        if (
            rerank_backend == "llm"
            and reranker_model
            and len(candidate_pool_ids) > rerank_input_limit
        ):
            try:
                prefilter_documents = [
                    "\n".join(
                        value
                        for value in (
                            str(episode_by_id[node_id].get("text", "")),
                            "人物：" + ", ".join(
                                episode_by_id[node_id].get("participants", [])
                            )
                            if episode_by_id[node_id].get("participants")
                            else "",
                            "故事时间：" + str(
                                episode_by_id[node_id].get("story_time_text", "")
                            )
                            if episode_by_id[node_id].get("story_time_text")
                            else "",
                        )
                        if value
                    )
                    for node_id in candidate_pool_ids
                ]
                with self._provider_purpose_scope(
                    "reranker", "rerank_precompression"
                ):
                    ranked = self.model.rerank(
                        question,
                        prefilter_documents,
                        top_n=len(prefilter_documents),
                    )
                ranked_pool_ids = [
                    candidate_pool_ids[int(item["index"])] for item in ranked
                ]
                if ranked_pool_ids:
                    precompression_pool_ids = list(
                        dict.fromkeys(ranked_pool_ids)
                    )
                bge_prefilter = {
                    "performed": True,
                    "model": reranker_model,
                    "input_count": len(candidate_pool_ids),
                    "ranked_episode_ids": precompression_pool_ids,
                    "scores": [
                        round(float(item["relevance_score"]), 8)
                        for item in ranked
                    ],
                }
            except TracePersistenceError:
                raise
            except Exception as exc:
                bge_prefilter = {
                    "performed": True,
                    "model": reranker_model,
                    "input_count": len(candidate_pool_ids),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        ordered_candidate_ids, precompression_decisions = (
            self._precompress_rerank_candidate_ids(
                precompression_pool_ids,
                preferred_ids,
                required_ids,
                rerank_input_limit,
            )
        )
        candidate_rows = [episode_by_id[node_id] for node_id in ordered_candidate_ids]
        preferred_rank = {
            int(node_id): rank
            for rank, node_id in enumerate(
                dict.fromkeys(preferred_candidate_ids or []), start=1
            )
        }
        paragraph_context_catalog = sorted(
            (
                context
                for contexts in (paragraph_context_by_source or {}).values()
                for context in contexts
            ),
            key=lambda item: int(item["paragraph_id"]),
        )
        candidates = [
            {
                "id": int(item["id"]),
                "score": round(float(item.get("score", 0.0)), 6),
                "text": str(item.get("text", ""))[:700],
                "participants": item.get("participants", []),
                "source_key": str(item.get("source_key", "")),
                "story_time_text": str(item.get("story_time_text", "")),
                "timeline_scope": str(item.get("timeline_scope", "")),
                "evidence_origin": str(item.get("evidence_origin", "unknown")),
                "epistemic_status": str(item.get("epistemic_status", "unknown")),
                "generation": int(item.get("generation", 0) or 0),
                "epistemic_note": str(item.get("epistemic_note", "")),
                "atomic_anchor_rank": preferred_rank.get(int(item["id"])),
                "source_context_refs": [
                    int(context["paragraph_id"])
                    for context in (paragraph_context_by_source or {}).get(
                        int(item.get("source_id", -1)), []
                    )
                ],
            }
            for item in candidate_rows
        ]
        if rerank_backend == "llm":
            initial_prompt = evidence_rerank_prompt(
                question,
                asdict(intent),
                atomic_queries,
                candidates,
                limit,
                paragraph_context_catalog,
            )
            coverage_prompt = evidence_coverage_audit_prompt(
                question,
                atomic_queries,
                candidates,
                paragraph_context_catalog,
            )
        else:
            initial_prompt = ""
            coverage_prompt = ""
        cache_payload = {
            "version": 1,
            "model": (
                self.config.model.reranker_model
                if rerank_backend == "cross_encoder"
                else self.config.model.reasoning_model
            ),
            "rerank_backend": rerank_backend,
            "prompt_version": self.config.prompt_version,
            "question": question,
            "intent": asdict(intent),
            "atomic_queries": atomic_queries,
            "candidates": candidates,
            "candidate_pool_ids": candidate_pool_ids,
            "bge_prefilter": bge_prefilter,
            "rerank_precompression_limit": rerank_input_limit,
            "source_contexts": paragraph_context_catalog,
            "required_candidate_ids": required_ids,
            "answer_slot_anchor_ids": list(answer_slot_anchor_ids or []),
            "answer_slot_neighbor_ids": answer_slot_neighbor_ids,
            "limit": limit,
            "shortlist_limit": self.config.retrieval.rerank_shortlist_limit,
            "coverage_audit_enabled": (
                self.config.retrieval.rerank_coverage_audit_enabled
            ),
            "rerank_audit_enabled": self.config.retrieval.rerank_audit_enabled,
            "rerank_review_mode": self.config.retrieval.rerank_review_mode,
            "rerank_atomic_query_limit": (
                self.config.retrieval.rerank_atomic_query_limit
            ),
            "rerank_strict_atomic_query_threshold": (
                self.config.retrieval.rerank_strict_atomic_query_threshold
            ),
            "rerank_compress_coverage_group_threshold": (
                self.config.retrieval.rerank_compress_coverage_group_threshold
            ),
            "rerank_answer_slot_neighbor_radius": (
                self.config.retrieval.rerank_answer_slot_neighbor_radius
            ),
            "rerank_answer_slot_neighbor_total_limit": (
                self.config.retrieval.rerank_answer_slot_neighbor_total_limit
            ),
            "systems": (
                []
                if rerank_backend == "cross_encoder"
                else [
                    EVIDENCE_RERANK_SYSTEM,
                    EVIDENCE_COVERAGE_AUDIT_SYSTEM,
                    EVIDENCE_RERANK_AUDIT_SYSTEM,
                ]
            ),
            "initial_prompt": initial_prompt,
            "coverage_prompt": coverage_prompt,
        }
        input_hash = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        allowed_ids = {int(item["id"]) for item in candidates}
        trace: dict = {
            "enabled": True,
            "backend": rerank_backend,
            "model": (
                self.config.model.reranker_model
                if rerank_backend == "cross_encoder"
                else self.config.model.reasoning_model
            ),
            # Preserve Candidate@100 semantics for recall measurement even
            # when fewer rows are sent to the model.
            "candidate_episode_ids": candidate_pool_ids,
            "rerank_input_episode_ids": [
                int(item["id"]) for item in candidates
            ],
            "precompression": {
                "enabled": len(candidate_pool_ids) > len(candidates),
                "pool_limit": candidate_limit,
                "configured_limit": int(
                    self.config.retrieval.rerank_precompression_limit
                ),
                "effective_limit": rerank_input_limit,
                "pool_count": len(candidate_pool_ids),
                "rerank_input_count": len(candidates),
                "dropped_count": len(candidate_pool_ids) - len(candidates),
                "decisions": precompression_decisions,
            },
            "bge_prefilter": bge_prefilter,
            "atomic_queries": atomic_queries,
            "expanded_atomic_query_count": len(expanded_atomic_queries),
            "atomic_queries_truncated": (
                len(atomic_queries) < len(expanded_atomic_queries)
            ),
            "input_hash": input_hash,
            "cache_hit": False,
            "paragraph_context_episode_count": sum(
                bool(item["source_context_refs"]) for item in candidates
            ),
            "paragraph_context_ids": sorted(
                {
                    int(paragraph_id)
                    for item in candidates
                    for paragraph_id in item["source_context_refs"]
                }
            ),
            "required_evidence_floor_ids": required_ids,
            "atomic_evidence_floor_ids": required_ids,
            "answer_slot_anchor_ids": [
                int(value) for value in (answer_slot_anchor_ids or [])
            ],
            "answer_slot_neighbor_floor_ids": answer_slot_neighbor_ids,
            "rerank_candidate_text_chars": sum(
                len(str(item.get("text", ""))) for item in candidates
            ),
            "initial_prompt_chars": len(initial_prompt),
            "coverage_prompt_chars": len(coverage_prompt),
        }
        if result_cache is not None and input_hash in result_cache:
            cached_ids, cached_trace = result_cache[input_hash]
            reused_trace = deepcopy(cached_trace)
            reused_trace["cache_hit"] = True
            reused_trace["cache_source_input_hash"] = input_hash
            if self.logger:
                self.logger.emit(
                    "evidence_rerank_reused",
                    question=question,
                    input_hash=input_hash,
                )
            return list(cached_ids), reused_trace
        if rerank_backend == "cross_encoder":
            try:
                documents = [
                    "\n".join(
                        value
                        for value in (
                            str(item.get("text", "")),
                            (
                                "人物：" + ", ".join(item["participants"])
                                if item.get("participants")
                                else ""
                            ),
                            (
                                "故事时间：" + str(item["story_time_text"])
                                if item.get("story_time_text")
                                else ""
                            ),
                            (
                                "时间线：" + str(item["timeline_scope"])
                                if item.get("timeline_scope")
                                else ""
                            ),
                        )
                        if value
                    )
                    for item in candidates
                ]
                with self._provider_purpose_scope("reranker", "rerank_final"):
                    ranked = self.model.rerank(
                        question,
                        documents,
                        top_n=len(documents),
                    )
                ranked_ids = [
                    int(candidates[int(item["index"])]["id"])
                    for item in ranked
                ]
                selected_ids = list(dict.fromkeys(ranked_ids))[:limit]
                selected_ids, floor_changes = self._enforce_selection_floor(
                    selected_ids,
                    required_ids,
                    allowed_ids,
                    limit,
                )
                for item in candidate_rows:
                    node_id = int(item["id"])
                    if len(selected_ids) >= limit:
                        break
                    if node_id not in selected_ids:
                        selected_ids.append(node_id)
                trace.update(
                    {
                        "review_mode": "cross_encoder",
                        "review_level": "cross_encoder",
                        "coverage_audit_performed": False,
                        "compressor_performed": False,
                        "cross_encoder_ranked_episode_ids": ranked_ids,
                        "cross_encoder_scores": [
                            round(float(item["relevance_score"]), 8)
                            for item in ranked
                        ],
                        "required_evidence_floor_replacements": floor_changes,
                        "atomic_evidence_floor_replacements": floor_changes,
                        "final_episode_ids": selected_ids,
                    }
                )
                if result_cache is not None:
                    result_cache[input_hash] = (
                        list(selected_ids),
                        deepcopy(trace),
                    )
                if self.logger:
                    self.logger.emit(
                        "evidence_reranked", question=question, trace=trace
                    )
                return selected_ids, trace
            except TracePersistenceError:
                raise
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"
                if self.logger:
                    self.logger.emit(
                        "evidence_rerank_failed",
                        question=question,
                        error=trace["error"],
                    )
                return [], trace
        try:
            with self._provider_purpose_scope(
                "reranker", "evidence_rerank_selection"
            ):
                initial = self.model.chat_json(
                    EVIDENCE_RERANK_SYSTEM,
                    initial_prompt,
                    allow_fallback=False,
                    max_retries=0,
                )
            review_level, review_reasons = self._rerank_review_level(
                atomic_queries,
                initial,
                allowed_ids,
                question,
            )
            run_coverage_audit = bool(
                review_level == "strict"
                and self.config.retrieval.rerank_coverage_audit_enabled
            )
            run_compressor = bool(
                review_level in {"strict", "compress"}
                and self.config.retrieval.rerank_audit_enabled
            )
            trace["review_mode"] = self.config.retrieval.rerank_review_mode
            trace["review_level"] = review_level
            trace["review_reasons"] = review_reasons
            trace["coverage_audit_performed"] = run_coverage_audit
            trace["compressor_performed"] = run_compressor
            coverage_audit = None
            coverage_payload = initial
            if run_coverage_audit:
                try:
                    with self._provider_purpose_scope(
                        "audit", "evidence_coverage_audit"
                    ):
                        coverage_audit = self.model.chat_json(
                            EVIDENCE_COVERAGE_AUDIT_SYSTEM,
                            coverage_prompt,
                            allow_fallback=False,
                            max_retries=0,
                        )
                    coverage_payload = self._merge_coverage_payloads(
                        initial,
                        coverage_audit,
                    )
                except TracePersistenceError:
                    raise
                except Exception as exc:
                    trace["coverage_audit_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            shortlist_limit = max(
                limit, self.config.retrieval.rerank_shortlist_limit
            )
            initial_ids = self._coverage_rerank_ids(
                coverage_payload, allowed_ids, shortlist_limit
            )
            if not initial_ids:
                initial_ids = self._validated_rerank_ids(
                    initial, "selected_episode_ids", allowed_ids, shortlist_limit
                )
            shortlist_ids = list(initial_ids)
            for item in candidate_rows:
                node_id = int(item["id"])
                if len(shortlist_ids) >= shortlist_limit:
                    break
                if node_id not in shortlist_ids:
                    shortlist_ids.append(node_id)
            candidate_by_id = {int(item["id"]): item for item in candidates}
            shortlist_candidates = [
                candidate_by_id[node_id]
                for node_id in shortlist_ids
                if node_id in candidate_by_id
            ]
            trace["initial"] = initial
            trace["coverage_audit"] = coverage_audit
            trace["independent_coverage"] = coverage_audit
            trace["merged_coverage"] = coverage_payload
            trace["shortlist_episode_ids"] = shortlist_ids
            selected_ids = initial_ids[:limit]
            trace["audit"] = None
            if run_compressor and shortlist_ids:
                with self._provider_purpose_scope(
                    "audit", "evidence_rerank_audit"
                ):
                    audit = self.model.chat_json(
                        EVIDENCE_RERANK_AUDIT_SYSTEM,
                        evidence_rerank_audit_prompt(
                            question,
                            atomic_queries,
                            shortlist_candidates,
                            coverage_payload,
                            limit,
                            paragraph_context_catalog,
                        ),
                        allow_fallback=False,
                        max_retries=0,
                    )
                audited_ids = self._validated_rerank_ids(
                    audit, "final_episode_ids", allowed_ids, limit
                )
                if audited_ids:
                    selected_ids = self._apply_rerank_replacements(
                        audit,
                        audited_ids,
                        set(shortlist_ids),
                        limit,
                    )
                trace["audit"] = audit
            selected_ids, floor_changes, coverage_changes = (
                self._enforce_final_selection_constraints(
                    selected_ids,
                    required_ids,
                    coverage_payload,
                    allowed_ids,
                    set(shortlist_ids),
                    limit,
                )
            )
            trace["required_evidence_floor_replacements"] = floor_changes
            trace["atomic_evidence_floor_replacements"] = floor_changes
            trace["coverage_enforced_replacements"] = coverage_changes
            # A valid coverage/audit selection is the prompt-facing evidence
            # decision.  Padding it with every broad candidate turns a
            # deliberately concise result into a very large answer prompt.
            # Retain the ordinary candidate fallback only when no reviewed
            # evidence survived validation at all.
            if not selected_ids:
                for item in candidate_rows:
                    node_id = int(item["id"])
                    if len(selected_ids) >= limit:
                        break
                    if node_id not in selected_ids:
                        selected_ids.append(node_id)
            trace["final_episode_ids"] = selected_ids
            if result_cache is not None and not any(
                trace.get(key)
                for key in (
                    "error",
                    "coverage_audit_error",
                    "independent_coverage_error",
                )
            ):
                result_cache[input_hash] = (
                    list(selected_ids),
                    deepcopy(trace),
                )
            if self.logger:
                self.logger.emit("evidence_reranked", question=question, trace=trace)
            return selected_ids, trace
        except TracePersistenceError:
            raise
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.emit(
                    "evidence_rerank_failed",
                    question=question,
                    error=trace["error"],
                )
            return [], trace

    @staticmethod
    def _diagnostic_content_hash(value: object) -> str:
        """Hash diagnostic input without placing its text in the event log."""

        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _answer_budget_policy(self) -> dict[str, float | int]:
        """Read the narrow V3.1 correction-admission envelope.

        Lightweight fixtures construct ``QueryEngine`` with ``object.__new__``
        and intentionally omit configuration.  Retain their existing unlimited
        local behavior by supplying the production defaults here; public
        queries always have ``self.config.retrieval``.
        """

        retrieval = getattr(getattr(self, "config", None), "retrieval", None)
        return {
            "max_revisions": max(
                0, min(1, int(getattr(retrieval, "answer_correction_max_revisions", 1)))
            ),
            "correction_attempt_seconds": max(
                0.0,
                float(getattr(
                    retrieval, "answer_correction_attempt_envelope_seconds", 25.0
                )),
            ),
            "fallback_seconds": max(
                0.0,
                float(getattr(
                    retrieval, "answer_correction_fallback_envelope_seconds", 25.0
                )),
            ),
            "reaudit_seconds": max(
                0.0,
                float(getattr(retrieval, "answer_reaudit_envelope_seconds", 25.0)),
            ),
            "finalization_reserve_seconds": max(
                0.0,
                float(getattr(
                    retrieval, "answer_finalization_reserve_seconds", 5.0
                )),
            ),
        }

    def _answer_correction_admission(
        self, deadline_at: float | None
    ) -> dict[str, object]:
        """Decide before correction whether its mandatory tail still fits."""

        policy = self._answer_budget_policy()
        required = round(
            float(policy["correction_attempt_seconds"])
            + float(policy["fallback_seconds"])
            + float(policy["reaudit_seconds"])
            + float(policy["finalization_reserve_seconds"]),
            6,
        )
        if deadline_at is None:
            return {
                "admitted": True,
                "reason": "no_public_deadline",
                "remaining_seconds": None,
                "required_seconds": required,
                "policy": policy,
            }
        remaining = max(0.0, float(deadline_at) - monotonic())
        return {
            "admitted": remaining >= required,
            "reason": (
                "full_chain_budget_available"
                if remaining >= required
                else "insufficient_budget_for_correction_reaudit_and_finalization"
            ),
            "remaining_seconds": round(remaining, 6),
            "required_seconds": required,
            "policy": policy,
        }

    def _emit_answer_checkpoint(self, event: str, **payload: object) -> None:
        """Write a redacted diagnostic checkpoint without business mutation."""

        if self.logger:
            self.logger.emit(event, **payload)

    @classmethod
    def _evidence_only_result_payload(
        cls,
        result: Mapping[str, object],
        selected_evidence: object,
    ) -> dict[str, object]:
        """Return the public, read-only evidence contract for a live query.

        This describes delivery, not semantic truth: ``source_bound`` means
        the selected Source excerpt passed the existing provenance contract;
        it never means the answer's claim was independently entailed.
        """

        rows = (
            [dict(item) for item in selected_evidence if isinstance(item, Mapping)]
            if isinstance(selected_evidence, Sequence)
            and not isinstance(selected_evidence, (str, bytes))
            else []
        )
        raw_requirements = result.get("authoritative_requirements")
        requirements = (
            list(raw_requirements.get("requirements", []))
            if isinstance(raw_requirements, Mapping)
            and isinstance(raw_requirements.get("requirements"), list)
            else []
        )
        required_slot_ids = {
            str(item.get("slot_id", ""))
            for item in requirements
            if isinstance(item, Mapping)
            and bool(item.get("required", False))
            and str(item.get("slot_id", ""))
        }
        clause_to_slot = {
            str(clause_id): str(item.get("slot_id"))
            for item in requirements
            if isinstance(item, Mapping) and str(item.get("slot_id", ""))
            for clause_id in item.get("clause_ids", [])
            if str(clause_id)
        }

        # Coverage is an end-state selector property, not a recursive union of
        # every intermediate initial/masked/joint diagnostic.  The latter made
        # a source-complete output appear unresolved merely because an earlier
        # stage had not yet selected its evidence.
        evidence_slot_trace = result.get("evidence_slot_trace")
        evidence_slot_trace = (
            evidence_slot_trace
            if isinstance(evidence_slot_trace, Mapping)
            else {}
        )
        selector_trace = evidence_slot_trace.get("slot_selector_v3")
        if not isinstance(selector_trace, Mapping):
            selector_trace = evidence_slot_trace.get("slot_selector_v2")
        selector_trace = (
            selector_trace if isinstance(selector_trace, Mapping) else {}
        )
        selector_key = (
            "masked_selector"
            if bool(selector_trace.get("shadow", False))
            else "treatment_selector"
        )
        terminal_selector = selector_trace.get(selector_key)
        if not isinstance(terminal_selector, Mapping):
            terminal_selector = selector_trace.get("masked_selector")
        terminal_selector = (
            terminal_selector
            if isinstance(terminal_selector, Mapping)
            else {}
        )
        missing_markers = {
            str(item)
            for item in terminal_selector.get("missing_required_clauses", [])
            if str(item)
        }
        coverage_observed = bool(terminal_selector)
        unresolved_requirements: list[dict[str, str]] = []
        for marker in sorted(missing_markers):
            if marker in required_slot_ids:
                unresolved_requirements.append(
                    {"reference_type": "slot", "reference": marker,
                     "status": "selector_reported_missing"}
                )
            elif marker in clause_to_slot:
                unresolved_requirements.append(
                    {"reference_type": "clause", "reference": marker,
                     "slot_id": clause_to_slot[marker],
                     "status": "selector_reported_missing"}
                )

        selected_refs: list[dict[str, object]] = []
        materialized_refs: list[dict[str, object]] = []
        for episode in rows:
            reference = {
                "episode_id": episode.get("id"),
                "source_id": episode.get("source_id"),
                "source_key": episode.get("source_key"),
                "segment_index": episode.get("segment_index"),
                "source_evidence_delivery": episode.get("source_evidence_delivery"),
                "source_evidence_delivery_reason": episode.get(
                    "source_evidence_delivery_reason"
                ),
                "source_evidence_quote_count": episode.get(
                    "source_evidence_quote_count"
                ),
            }
            selected_refs.append(reference)
            source_excerpt = str(episode.get("source_text", ""))
            if source_excerpt:
                materialized_refs.append(
                    {
                        **reference,
                        "source_excerpt": source_excerpt,
                        "source_excerpt_hash": cls._diagnostic_content_hash(
                            source_excerpt
                        ),
                    }
                )

        all_source_bound = bool(rows) and all(
            str(item.get("source_evidence_delivery", "")) == "source_bound"
            for item in rows
        )
        source_delivery_state = (
            "failed" if not rows or not materialized_refs else
            "complete" if all_source_bound else "partial"
        )
        contextual = result.get("contextual_association")
        contextual = contextual if isinstance(contextual, Mapping) else {}
        prepared_early = contextual.get("prepared_early")
        prepared_early = (
            prepared_early if isinstance(prepared_early, Mapping) else {}
        )
        exact = result.get("exact_revisit")
        exact = exact if isinstance(exact, Mapping) else {}
        association_ids = result.get("association_ids", [])
        selected_contributions = contextual.get(
            "selected_contextual_contribution_ids", []
        )
        if str(exact.get("status", "")) == "hit":
            edge_state = "contributed" if association_ids else "evaluated_no_increment"
        elif not bool(contextual.get("enabled", False)):
            edge_state = "not_entered"
        elif isinstance(selected_contributions, (list, tuple)) and selected_contributions:
            edge_state = "contributed"
        else:
            edge_state = "evaluated_no_increment"
        timing = result.get("timings")
        timing = timing if isinstance(timing, Mapping) else {}
        phase_seconds = timing.get("phases_seconds", {})
        phase_seconds = phase_seconds if isinstance(phase_seconds, Mapping) else {}
        rerank = result.get("rerank_trace")
        rerank = rerank if isinstance(rerank, Mapping) else {}
        frozen_input = bool(result.get("query_plan_frozen", False))
        rerank_reuse = rerank.get("candidate_input_reuse")
        rerank_reuse = rerank_reuse if isinstance(rerank_reuse, Mapping) else {}
        rerank_reuse_status = str(rerank_reuse.get("status", ""))
        rerank_result_reused = bool(
            (frozen_input and rerank_reuse_status == "frozen_input_fingerprint_matched")
            or (not frozen_input and bool(rerank.get("cache_hit", False)))
        )
        executed_modules = {
            "planner": "intent_parse" in phase_seconds and not exact,
            "embedding": bool(
                isinstance(result.get("query_embedding_cache"), Mapping)
                and int(result["query_embedding_cache"].get("miss_count", 0) or 0)
                > 0
            ),
            "retrieval": "initial_retrieval" in phase_seconds,
            "reranker": bool(
                rerank.get("enabled", False) and not rerank_result_reused
            ),
            "contextual_matcher": (
                edge_state != "not_entered"
                or bool(
                    isinstance(prepared_early.get("executed_modules"), Mapping)
                    and prepared_early["executed_modules"].get(
                        "prepared_early_matcher", False
                    )
                )
            ),
            "prepared_early_contextual": bool(
                isinstance(prepared_early.get("executed_modules"), Mapping)
                and prepared_early["executed_modules"].get(
                    "prepared_early_matcher", False
                )
            ),
            "source_materialization": bool(materialized_refs),
            "answer_generation": False,
            "answer_audit": False,
            "association_growth": False,
            "association_use_write": False,
            "learning_finalizer": False,
        }
        return {
            "request_id": None,
            "request_id_status": "not_observed_by_query_api",
            "execution_profile": "input_frozen" if frozen_input else "live_evidence",
            # Compatibility: this historical field described excerpt delivery,
            # not semantic support.  The explicit fields below remove that
            # ambiguity for new consumers.
            "evidence_state": source_delivery_state,
            "evidence_state_compatibility_note": "source_delivery_state",
            "source_delivery_state": source_delivery_state,
            "requirement_coverage_state": (
                "not_observed"
                if not coverage_observed
                else "complete"
                if not missing_markers
                else "unresolved"
            ),
            "terminal_selector": selector_key if coverage_observed else "not_observed",
            "replay_states": {
                "prepared_inputs_replayed": frozen_input,
                "rerank_result_reused": rerank_result_reused,
                "rerank_reuse_status": rerank_reuse_status or "not_observed",
            },
            "requirements": requirements,
            "selected_refs": selected_refs,
            "materialized_source_refs": materialized_refs,
            "unresolved_requirements": unresolved_requirements,
            "runtime_coverage_status": (
                "selector_missing_markers_observed"
                if coverage_observed else "not_observed"
            ),
            "participation": {
                "edge_state": edge_state,
                "entry_reason": str(contextual.get("reason", "not_observed")),
                "masked_missing_slots": list(
                    contextual.get("masked_missing_slots", [])
                ),
                "association_ids": list(association_ids)
                if isinstance(association_ids, (list, tuple)) else [],
                "selected_contextual_contribution_ids": list(selected_contributions)
                if isinstance(selected_contributions, (list, tuple)) else [],
            },
            "delivery_states": {
                "evidence_materialized": bool(materialized_refs),
                "answer_input_sent": False,
                "public_response_returned": True,
                "adapter_sent": "not_observed_by_query_api",
            },
            "executed_modules": executed_modules,
            "actual_skipped_modules": sorted(
                name for name, value in executed_modules.items() if not value
            ),
            "timing_and_call_refs": {
                "phase_seconds": dict(phase_seconds),
                "provider_call_refs": "not_observed_by_query_api",
            },
            "learning_eligible": False,
        }

    def _answer_checkpoint_version_binding(
        self,
        *,
        requirements_hash: str,
        evidence_set_hash: str,
        answer_prompt_hash: str,
    ) -> dict[str, object]:
        """Return the non-secret identity binding for an answer evidence receipt."""

        return {
            "prompt_version": getattr(getattr(self, "config", None), "prompt_version", None),
            "answer_system_hash": self._diagnostic_content_hash(ANSWER_SYSTEM),
            "answer_audit_system_hash": self._diagnostic_content_hash(
                ANSWER_AUDIT_SYSTEM
            ),
            "answer_prompt_hash": answer_prompt_hash,
            "requirements_hash": requirements_hash,
            "evidence_set_hash": evidence_set_hash,
        }

    @staticmethod
    def _answer_selected_evidence_binding(episodes: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        """Keep only actual answer-boundary evidence, never the full prompt."""

        bindings: list[dict[str, object]] = []
        for episode in episodes:
            source_excerpt = str(episode.get("source_text", ""))
            bindings.append(
                {
                    "episode_id": episode.get("id"),
                    "source_id": episode.get("source_id"),
                    "source_key": episode.get("source_key"),
                    "segment_index": episode.get("segment_index"),
                    "episode_text": episode.get("text"),
                    "source_evidence_delivery": episode.get(
                        "source_evidence_delivery"
                    ),
                    "source_evidence_delivery_reason": episode.get(
                        "source_evidence_delivery_reason"
                    ),
                    "source_evidence_quote_count": episode.get(
                        "source_evidence_quote_count"
                    ),
                    "source_excerpt": source_excerpt,
                    "source_excerpt_hash": QueryEngine._diagnostic_content_hash(
                        source_excerpt
                    ),
                }
            )
        return bindings

    @staticmethod
    def _answer_prompt_evidence_assertions(
        prompt: str, episodes: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        """Record whether the actual composed prompt retained each Source view.

        This is deliberately an assertion over the final string, rather than a
        second prompt payload channel.  The companion receipt already contains
        the local Source excerpts; these booleans show that the answer/audit
        formatter did not drop them after evidence selection.
        """

        rows: list[dict[str, object]] = []
        for episode in episodes:
            source_excerpt = str(episode.get("source_text", ""))
            speaker_marker_present = "[speaker_raw:" not in source_excerpt or (
                "[speaker_raw:" in prompt
            )
            rows.append(
                {
                    "episode_id": episode.get("id"),
                    "source_excerpt_present": (
                        not source_excerpt
                        or source_excerpt in prompt
                        or json.dumps(source_excerpt, ensure_ascii=False) in prompt
                    ),
                    "speaker_marker_present": speaker_marker_present,
                }
            )
        return {
            "all_source_excerpts_present": all(
                bool(row["source_excerpt_present"]) for row in rows
            ),
            "all_speaker_markers_present": all(
                bool(row["speaker_marker_present"]) for row in rows
            ),
            "episodes": rows,
        }

    def _emit_answer_evidence_checkpoint(self, event: str, **payload: object) -> None:
        """Persist the narrowly allow-listed local evidence companion when supported."""

        if not bool(
            getattr(
                getattr(getattr(self, "config", None), "retrieval", None),
                "answer_evidence_checkpoint_enabled",
                False,
            )
        ):
            return
        emitter = getattr(self.logger, "emit_answer_evidence_checkpoint", None)
        if callable(emitter):
            emitter(event, **payload)

    def _generate_audited_answer(
        self,
        question: str,
        intent: QueryIntent,
        episodes: list[dict],
        concepts: list[dict],
        paths: list[dict],
        chronology_notes: list[str],
        *,
        deadline_at: float | None = None,
    ) -> tuple[str, list[dict], int]:
        """Generate, verify, and at most once revise an evidence-bound answer."""
        intent_dict = asdict(intent)
        base_prompt = answer_prompt(
            question,
            intent_dict,
            episodes,
            concepts,
            paths,
            chronology_notes,
        )
        requirements_hash = self._diagnostic_content_hash(intent_dict)
        evidence_set_hash = self._diagnostic_content_hash(episodes)
        answer_prompt_hash = self._diagnostic_content_hash(base_prompt)
        version_binding = self._answer_checkpoint_version_binding(
            requirements_hash=requirements_hash,
            evidence_set_hash=evidence_set_hash,
            answer_prompt_hash=answer_prompt_hash,
        )
        selected_evidence_binding = self._answer_selected_evidence_binding(episodes)
        answer_prompt_evidence_assertions = self._answer_prompt_evidence_assertions(
            base_prompt, episodes
        )
        self._emit_answer_checkpoint(
            "answer_input_checkpoint",
            question=question,
            requirements_hash=requirements_hash,
            evidence_set_hash=evidence_set_hash,
            answer_prompt_hash=answer_prompt_hash,
            selected_episode_ids=[int(item["id"]) for item in episodes if "id" in item],
            selected_evidence=episodes,
            remaining_budget_seconds=(
                None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
            ),
        )
        self._emit_answer_evidence_checkpoint(
            "answer_input_checkpoint",
            stage="answer_input",
            question=question,
            requirements=intent_dict,
            version_binding=version_binding,
            selected_evidence=selected_evidence_binding,
            prompt_evidence_assertions={
                "answer_input": answer_prompt_evidence_assertions,
            },
            remaining_budget_seconds=(
                None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
            ),
        )
        try:
            with self._provider_purpose_scope("answer", "answer_generation"):
                answer = self.model.chat_text(ANSWER_SYSTEM, base_prompt)
        except BaseException as exc:
            self._last_answer_execution_state = {
                "terminal_state": "technical_failure",
                "reason": "answer_generation_exception",
                "error_type": type(exc).__name__,
            }
            self._emit_answer_checkpoint(
                "answer_revision_checkpoint",
                stage="answer_generation_exception",
                requirements_hash=requirements_hash,
                evidence_set_hash=evidence_set_hash,
                error_type=type(exc).__name__,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
            self._emit_answer_evidence_checkpoint(
                "answer_revision_checkpoint",
                stage="answer_generation_exception",
                question=question,
                version_binding=version_binding,
                selected_evidence=selected_evidence_binding,
                error_type=type(exc).__name__,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
            raise
        audits: list[dict] = []
        revision_count = 0
        self._last_answer_execution_state = {
            "terminal_state": "in_progress",
            "reason": "initial_answer_generated",
        }
        self._emit_answer_checkpoint(
            "answer_revision_checkpoint",
            stage="initial_answer",
            answer_revision_hash=self._diagnostic_content_hash(answer),
            requirements_hash=requirements_hash,
            evidence_set_hash=evidence_set_hash,
            revision_count=revision_count,
            remaining_budget_seconds=(
                None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
            ),
        )
        self._emit_answer_evidence_checkpoint(
            "answer_revision_checkpoint",
            stage="initial_answer",
            question=question,
            version_binding=version_binding,
            selected_evidence=selected_evidence_binding,
            answer=answer,
            answer_revision_hash=self._diagnostic_content_hash(answer),
            revision_count=revision_count,
            remaining_budget_seconds=(
                None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
            ),
        )
        audit_event_continuity = self._should_audit_event_continuity(
            question, intent, episodes
        )
        max_revisions = int(self._answer_budget_policy()["max_revisions"])
        for audit_index in range(max_revisions + 1):
            try:
                audit_prompt = answer_audit_prompt(
                    question, intent_dict, answer, episodes
                )
                audit_prompt_evidence_assertions = (
                    self._answer_prompt_evidence_assertions(audit_prompt, episodes)
                )
                with self._provider_purpose_scope(
                    "audit", "answer_evidence_audit"
                ):
                    payload = self.model.chat_json(
                        ANSWER_AUDIT_SYSTEM,
                        audit_prompt,
                    )
                if audit_event_continuity:
                    with self._provider_purpose_scope(
                        "audit", "event_continuity_audit"
                    ):
                        continuity_payload = self.model.chat_json(
                            EVENT_CONTINUITY_AUDIT_SYSTEM,
                            event_continuity_audit_prompt(
                                question, answer, episodes
                            ),
                        )
                    if isinstance(payload, dict) and isinstance(
                        continuity_payload, dict
                    ):
                        continuity_reviews = continuity_payload.get("reviews", [])
                        if not isinstance(continuity_reviews, list):
                            continuity_reviews = [
                                {
                                    "claim": "<跨事件连续性审计响应>",
                                    "verdict": "unsupported",
                                    "requires_revision": True,
                                    "reason": "reviews 必须是数组",
                                }
                            ]
                        continuity_reviews = [
                            {
                                **review,
                                "audit_scope": "event_continuity",
                            }
                            if isinstance(review, dict)
                            else review
                            for review in continuity_reviews
                        ]
                        payload_reviews = payload.get("reviews")
                        if isinstance(payload_reviews, list):
                            payload["reviews"] = [
                                *payload_reviews,
                                *continuity_reviews,
                            ]
                        elif continuity_reviews:
                            payload["reviews"] = continuity_reviews
                        payload["event_continuity_reviews"] = continuity_reviews
                        corrections = continuity_payload.get(
                            "correction_instructions", []
                        )
                        if isinstance(corrections, list):
                            existing_corrections = payload.get(
                                "correction_instructions", []
                            )
                            if not isinstance(existing_corrections, list):
                                existing_corrections = []
                            payload["correction_instructions"] = [
                                *existing_corrections,
                                *corrections,
                            ]
            except TracePersistenceError:
                raise
            except Exception as exc:
                audit = {
                    "valid": None,
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
                audits.append(audit)
                if self.logger:
                    self.logger.emit(
                        "answer_audit_failed",
                        question=question,
                        audit_index=audit_index,
                        error=audit["audit_error"],
                    )
                self._last_answer_execution_state = {
                    "terminal_state": "technical_failure",
                    "reason": "audit_exception",
                    "audit_index": audit_index,
                    "error_type": type(exc).__name__,
                }
                self._emit_answer_checkpoint(
                    "answer_revision_checkpoint",
                    stage="audit_exception",
                    answer_revision_hash=self._diagnostic_content_hash(answer),
                    requirements_hash=requirements_hash,
                    evidence_set_hash=evidence_set_hash,
                    audit_index=audit_index,
                    error_type=type(exc).__name__,
                    remaining_budget_seconds=(
                        None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                    ),
                )
                self._emit_answer_evidence_checkpoint(
                    "answer_revision_checkpoint",
                    stage="audit_exception",
                    question=question,
                    version_binding=version_binding,
                    selected_evidence=selected_evidence_binding,
                    prompt_evidence_assertions={
                        "answer_input": answer_prompt_evidence_assertions,
                        "audit_input": locals().get(
                            "audit_prompt_evidence_assertions", "not_observed"
                        ),
                    },
                    answer=answer,
                    answer_revision_hash=self._diagnostic_content_hash(answer),
                    audit_index=audit_index,
                    error_type=type(exc).__name__,
                    terminal_reason="audit_exception",
                    remaining_budget_seconds=(
                        None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                    ),
                )
                break
            if not isinstance(payload, dict):
                payload = {
                    "valid": False,
                    "issues": [
                        {
                            "claim": "<audit response>",
                            "reason": "answer audit did not return a JSON object",
                        }
                    ],
                    "correction_instructions": ["重新核对所有直接结论"],
                }
            audit = self._normalize_answer_audit(payload, answer)
            audits.append(audit)
            audit_issues = audit.get("issues", [])
            if not isinstance(audit_issues, list):
                audit_issues = []
            if self.logger:
                self.logger.emit(
                    "answer_evidence_audit",
                    question=question,
                    audit_index=audit_index,
                    audit=audit,
                )
            self._emit_answer_checkpoint(
                "answer_revision_checkpoint",
                stage="audit_result",
                answer_revision_hash=self._diagnostic_content_hash(answer),
                requirements_hash=requirements_hash,
                evidence_set_hash=evidence_set_hash,
                audit_index=audit_index,
                audit_valid=audit.get("valid"),
                audit_requires_revision=any(
                    isinstance(item, Mapping) and item.get("requires_revision") is True
                    for item in audit_issues
                ),
                revision_count=revision_count,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
            self._emit_answer_evidence_checkpoint(
                "answer_revision_checkpoint",
                stage="audit_result",
                question=question,
                version_binding=version_binding,
                selected_evidence=selected_evidence_binding,
                prompt_evidence_assertions={
                    "answer_input": answer_prompt_evidence_assertions,
                    "audit_input": audit_prompt_evidence_assertions,
                },
                answer=answer,
                answer_revision_hash=self._diagnostic_content_hash(answer),
                audit=audit,
                audit_index=audit_index,
                revision_count=revision_count,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
            if audit.get("valid") is True:
                self._last_answer_execution_state = {
                    "terminal_state": "completed_verified",
                    "reason": "audit_valid",
                    "audit_index": audit_index,
                    "revision_count": revision_count,
                }
                break
            if revision_count >= max_revisions:
                self._last_answer_execution_state = {
                    "terminal_state": "completed_limited",
                    "reason": "correction_revision_limit_reached",
                    "audit_index": audit_index,
                    "revision_count": revision_count,
                }
                break
            admission = self._answer_correction_admission(deadline_at)
            self._emit_answer_checkpoint(
                "answer_correction_admission",
                audit_index=audit_index,
                answer_revision_hash=self._diagnostic_content_hash(answer),
                requirements_hash=requirements_hash,
                evidence_set_hash=evidence_set_hash,
                **admission,
            )
            if not bool(admission["admitted"]):
                self._last_answer_execution_state = {
                    "terminal_state": "completed_limited",
                    "reason": str(admission["reason"]),
                    "audit_index": audit_index,
                    "revision_count": revision_count,
                }
                self._emit_answer_evidence_checkpoint(
                    "answer_revision_checkpoint",
                    stage="correction_not_started",
                    question=question,
                    version_binding=version_binding,
                    selected_evidence=selected_evidence_binding,
                    answer=answer,
                    answer_revision_hash=self._diagnostic_content_hash(answer),
                    audit=audit,
                    audit_index=audit_index,
                    revision_count=revision_count,
                    terminal_reason=str(admission["reason"]),
                    remaining_budget_seconds=admission["remaining_seconds"],
                )
                break
            try:
                with self._provider_purpose_scope("answer", "answer_correction"):
                    answer = self.model.chat_text(
                        ANSWER_SYSTEM,
                        base_prompt + answer_correction_prompt(answer, audit),
                    )
            except BaseException as exc:
                self._last_answer_execution_state = {
                    "terminal_state": "technical_failure",
                    "reason": "correction_exception",
                    "audit_index": audit_index,
                    "error_type": type(exc).__name__,
                }
                self._emit_answer_checkpoint(
                    "answer_revision_checkpoint",
                    stage="correction_exception",
                    answer_revision_hash=self._diagnostic_content_hash(answer),
                    requirements_hash=requirements_hash,
                    evidence_set_hash=evidence_set_hash,
                    audit_index=audit_index,
                    error_type=type(exc).__name__,
                )
                raise
            revision_count += 1
            self._emit_answer_checkpoint(
                "answer_revision_checkpoint",
                stage="corrected_answer",
                answer_revision_hash=self._diagnostic_content_hash(answer),
                requirements_hash=requirements_hash,
                evidence_set_hash=evidence_set_hash,
                audit_index=audit_index,
                revision_count=revision_count,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
            self._emit_answer_evidence_checkpoint(
                "answer_revision_checkpoint",
                stage="corrected_answer",
                question=question,
                version_binding=version_binding,
                selected_evidence=selected_evidence_binding,
                answer=answer,
                answer_revision_hash=self._diagnostic_content_hash(answer),
                audit_index=audit_index,
                revision_count=revision_count,
                remaining_budget_seconds=(
                    None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
                ),
            )
        return answer, audits, revision_count

    @staticmethod
    def _should_audit_event_continuity(
        question: str,
        intent: QueryIntent,
        episodes: list[dict],
    ) -> bool:
        source_keys = {
            str(episode.get("source_key", ""))
            for episode in episodes
            if episode.get("source_key")
        }
        if len(source_keys) < 2:
            return False
        causal_text = f"{question} {intent.causal_constraint}".casefold()
        return any(
            marker in causal_text
            for marker in (
                "为什么",
                "为何",
                "因为",
                "导致",
                "因果",
                "原因",
                "trigger",
                "cause",
                "enable",
            )
        )

    @staticmethod
    def _normalize_answer_audit(payload: dict, answer: str = "") -> dict:
        """Derive validity from per-claim verdicts instead of free-form prose."""
        audit = dict(payload)
        reviews = audit.get("reviews")
        if not isinstance(reviews, list):
            return audit
        normalized_reviews: list[dict] = []
        issues: list[dict] = []
        allowed = {
            "supported_fact",
            "supported_inference",
            "unsupported",
            "contradicted",
        }
        for raw_review in reviews:
            if not isinstance(raw_review, dict):
                review = {
                    "claim": "<malformed review>",
                    "verdict": "unsupported",
                    "requires_revision": True,
                    "reason": "answer audit review must be an object",
                }
            else:
                review = dict(raw_review)
                verdict = str(review.get("verdict", "")).strip()
                if verdict not in allowed:
                    verdict = "unsupported"
                    review["verdict"] = verdict
                    review["reason"] = (
                        str(review.get("reason", ""))
                        + " [invalid or missing verdict]"
                    ).strip()
                requires_revision = verdict in {"unsupported", "contradicted"}
                review["requires_revision"] = requires_revision
            normalized_reviews.append(review)
            if review.get("requires_revision") is True:
                issues.append(review)
        if not normalized_reviews:
            issues.append(
                {
                    "claim": "<missing reviews>",
                    "verdict": "unsupported",
                    "requires_revision": True,
                    "reason": "answer audit returned no claim reviews",
                }
            )
        audit["reviews"] = normalized_reviews
        audit["issues"] = issues
        audit["valid"] = not issues
        return audit

    def _explicit_association_paths(
        self,
        association_ids: list[int],
        cue_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """Make newly changed edges eligible even when neighbor fan-out is full."""
        cue_scores = cue_scores or {}
        paths: list[dict] = []
        for association_id in dict.fromkeys(association_ids):
            row = self.associations.get(int(association_id))
            if row is None:
                continue
            paths.append(
                {
                    "association_id": int(row["id"]),
                    "from": [str(row["from_type"]), int(row["from_id"])],
                    "to": [str(row["to_type"]), int(row["to_id"])],
                    "relation_type": str(row["relation_type"]),
                    "relation_key": str(row["relation_key"]),
                    "relation_text": str(row["relation_text"]),
                    "association_mode": (
                        str(row["association_mode"])
                        if "association_mode" in row.keys()
                        else "semantic"
                    ),
                    "polarity": int(row["polarity"]),
                    "weight": float(row["weight"]),
                    "confidence": float(row["confidence"]),
                    "generation": (
                        int(row["generation"])
                        if "generation" in row.keys()
                        else 0
                    ),
                    "claim_level": (
                        str(row["claim_level"])
                        if "claim_level" in row.keys()
                        else "direct_fact"
                    ),
                    "audit_status": (
                        str(row["audit_status"])
                        if "audit_status" in row.keys()
                        else "not_required"
                    ),
                    "created_reason": (
                        str(row["created_reason"])
                        if "created_reason" in row.keys()
                        else ""
                    ),
                    "path_score": float(row["weight"])
                    * float(row["confidence"]),
                    "association_cue_similarity": cue_scores.get(
                        int(row["id"])
                    ),
                }
            )
        return paths

    def _late_learned_bridge_closure(
        self,
        question: str,
        preferred_episode_ids: list[int],
        traversed: list[TraversedNode],
    ) -> tuple[list[TraversedNode], list[dict]]:
        """Admit one-hop endpoints of strong learned edges after reranking."""

        remaining = max(0, int(self.config.retrieval.learned_bridge_slots))
        if remaining <= 0:
            return [], []
        existing = {
            (str(item.node_type), int(item.node_id)) for item in traversed
        }
        score_by_episode = {
            int(item.node_id): float(item.score)
            for item in traversed
            if item.node_type == "episode"
        }
        preferred = list(
            dict.fromkeys(
                [
                    *(int(value) for value in preferred_episode_ids),
                    *(
                        int(item.node_id)
                        for item in traversed
                        if item.node_type == "episode"
                    ),
                ]
            )
        )
        additions: list[TraversedNode] = []
        paths: list[dict] = []
        used_edges: set[int] = set()
        neighbors_many = getattr(self.associations, "neighbors_many", None)
        if callable(neighbors_many):
            neighbor_map = neighbors_many("episode", preferred, limit=24)
        else:
            neighbor_map = {
                node_id: self.associations.neighbors(
                    "episode", node_id, limit=24
                )
                for node_id in preferred
            }
        for rank, anchor_id in enumerate(preferred, start=1):
            if remaining <= 0:
                break
            anchor_score = score_by_episode.get(
                anchor_id,
                0.55 / (1.0 + (rank - 1) / 20.0),
            )
            for row in neighbor_map.get(anchor_id, []):
                association_id = int(row["id"])
                if association_id in used_edges:
                    continue
                if (
                    str(row["audit_status"]) != "dual_accepted"
                    or int(row["generation"] or 0) > 1
                    or float(row["confidence"]) < 0.75
                    or int(row["polarity"]) < 0
                    or str(row["relation_key"]) == "involves"
                    or str(row["from_type"]) != "episode"
                    or str(row["to_type"]) != "episode"
                ):
                    continue
                explicit_paths = self._explicit_association_paths(
                    [association_id]
                )
                if not explicit_paths:
                    continue
                path = explicit_paths[0]
                relevance = self._path_query_relevance(question, path)
                if relevance < float(
                    self.config.retrieval.learned_bridge_min_query_relevance
                ):
                    continue
                other_id = (
                    int(row["to_id"])
                    if int(row["from_id"]) == anchor_id
                    else int(row["from_id"])
                )
                path_score = (
                    anchor_score
                    * float(row["weight"])
                    * float(row["confidence"])
                    * 0.85
                )
                if ("episode", other_id) not in existing:
                    additions.append(
                        TraversedNode(
                            "episode",
                            other_id,
                            path_score,
                            [association_id],
                        )
                    )
                path.update(
                    {
                        "from": ["episode", anchor_id],
                        "to": ["episode", other_id],
                        "path_score": path_score,
                        "late_bridge_closure": True,
                    }
                )
                paths.append(path)
                existing.add(("episode", other_id))
                used_edges.add(association_id)
                remaining -= 1
                if remaining <= 0:
                    break
        return additions, paths

    def _materialize_nodes(
        self, traversed, include_sources: bool = False
    ) -> tuple[list[dict], list[dict]]:
        episode_scores = {
            item.node_id: item.score for item in traversed if item.node_type == "episode"
        }
        concept_scores = {
            item.node_id: item.score for item in traversed if item.node_type == "concept"
        }
        episode_rows = self.episodes.get_many(episode_scores)
        concept_rows = self.concepts.get_many(concept_scores)
        source_map = {}
        if include_sources:
            source_map = {
                int(row["id"]): row
                for row in self.sources.get_many(
                    int(episode["source_id"]) for episode in episode_rows
                )
            }
        episodes: list[dict] = []
        for row in episode_rows:
            try:
                participants = json.loads(row["participants_json"])
            except (TypeError, json.JSONDecodeError):
                participants = []
            source = source_map.get(int(row["source_id"]))
            try:
                raw_evidence_quotes = json.loads(
                    str(row["evidence_quotes_json"] or "[]")
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                raw_evidence_quotes = []
            evidence_quotes = (
                [
                    item.strip()
                    for item in raw_evidence_quotes
                    if isinstance(item, str) and item.strip()
                ]
                if isinstance(raw_evidence_quotes, list)
                else []
            )
            raw_source_text = str(source["raw_text"]) if source else ""
            evidence_view_outcomes = verified_evidence_view_outcomes(
                raw_source_text, evidence_quotes
            )
            verified_evidence = [
                str(outcome["view"])
                for outcome in evidence_view_outcomes
                if outcome["view"] is not None
            ]
            source_text = (
                source_excerpt(
                    raw_source_text,
                    str(row["text"]),
                    participants,
                    self.config.retrieval.source_excerpt_chars,
                    evidence_quotes=evidence_quotes,
                )
                if source
                else ""
            )
            delivered_quotes = [
                quote for quote in verified_evidence if quote in source_text
            ]
            if evidence_quotes:
                if len(verified_evidence) != len(evidence_quotes):
                    source_evidence_delivery = (
                        "source_evidence_not_delivered_within_excerpt_budget"
                    )
                    source_evidence_delivery_reason = "quote_verification_failed"
                    source_evidence_verification_reasons = [
                        str(outcome["reason"])
                        for outcome in evidence_view_outcomes
                        if outcome["view"] is None
                    ]
                elif len(delivered_quotes) != len(verified_evidence):
                    source_evidence_delivery = (
                        "source_evidence_not_delivered_within_excerpt_budget"
                    )
                    source_evidence_delivery_reason = "verified_quote_exceeds_excerpt_budget"
                    source_evidence_verification_reasons = [
                        str(outcome["reason"])
                        for outcome in evidence_view_outcomes
                    ]
                else:
                    source_evidence_delivery = "source_bound"
                    source_evidence_delivery_reason = "all_quotes_verified_and_delivered"
                    source_evidence_verification_reasons = [
                        str(outcome["reason"])
                        for outcome in evidence_view_outcomes
                    ]
            else:
                source_evidence_delivery = "legacy_heuristic_excerpt"
                source_evidence_delivery_reason = "no_persisted_evidence_quote"
                source_evidence_verification_reasons = []
            episodes.append(
                {
                    "type": "episode",
                    "id": int(row["id"]),
                    "source_id": int(row["source_id"]),
                    "score": episode_scores[int(row["id"])],
                    "text": row["text"],
                    "participants": participants,
                    "source_key": row["source_key"],
                    "segment_index": int(row["segment_index"]),
                    "story_time_text": row["story_time_text"],
                    "story_order": row["story_order"],
                    "timeline_scope": row["timeline_scope"],
                    "evidence_origin": row["evidence_origin"],
                    "epistemic_status": row["epistemic_status"],
                    "generation": int(row["generation"] or 0),
                    "epistemic_note": row["epistemic_note"],
                    "source_evidence_delivery": source_evidence_delivery,
                    "source_evidence_delivery_reason": source_evidence_delivery_reason,
                    "source_evidence_verification_reasons": source_evidence_verification_reasons,
                    "source_evidence_quote_count": len(evidence_quotes),
                    "source_text": source_text,
                }
            )
        concepts = [
            {
                "type": "concept",
                "id": int(row["id"]),
                "score": concept_scores[int(row["id"])],
                "canonical_name": row["canonical_name"],
                "description": row["description"],
            }
            for row in concept_rows
        ]
        episodes.sort(key=lambda item: item["score"], reverse=True)
        concepts.sort(key=lambda item: item["score"], reverse=True)
        return episodes, concepts

    @staticmethod
    def _path_text_features(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        features = set(_PATH_WORD_RE.findall(normalized.replace("_", " ")))
        for run in _PATH_CJK_RUN_RE.findall(normalized):
            for size in (2, 3):
                features.update(
                    run[index : index + size]
                    for index in range(max(0, len(run) - size + 1))
                )
        return features

    @classmethod
    def _path_query_relevance(cls, question: str, path: dict) -> float:
        if not question:
            return 0.0
        question_features = cls._path_text_features(question)
        relation_features = cls._path_text_features(
            f"{path.get('relation_key', '')} {path.get('relation_text', '')}"
        )
        if not question_features or not relation_features:
            return 0.0
        overlap = len(question_features.intersection(relation_features))
        return overlap / math.sqrt(len(question_features) * len(relation_features))

    @classmethod
    def _rank_answer_paths(
        cls,
        paths: list[dict],
        question: str,
        path_limit: int,
        preferred_association_ids: set[int] | None = None,
        preferred_association_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """Deduplicate and diversify paths before they consume answer budget.

        Raw graph score remains useful inside a relation class, but high-confidence
        Episode→Concept ``involves`` edges must not crowd every cross-Episode bridge
        out of the answer. Query text overlap is deliberately lightweight and only
        reranks already traversed, evidence-backed Association rows.
        """
        preferred = preferred_association_ids or set()
        preferred_scores = preferred_association_scores or {}
        best_by_association: dict[int, dict] = {}
        for raw_path in paths:
            association_id = int(raw_path.get("association_id", -1))
            previous = best_by_association.get(association_id)
            use_raw = previous is None or float(
                raw_path.get("path_score", 0.0)
            ) > float((previous or {}).get("path_score", 0.0))
            winner = dict(raw_path if use_raw else previous)
            if raw_path.get("late_bridge_closure") or (
                previous and previous.get("late_bridge_closure")
            ):
                # The same edge can arrive through ordinary graph traversal
                # and the stricter late-closure lane. Keep the best score but
                # do not discard the latter's admission contract.
                winner["late_bridge_closure"] = True
            best_by_association[association_id] = winner

        candidates: list[dict] = []
        for raw_path in best_by_association.values():
            # Contextual double-key edges are retrieval hints, not claims.
            # Their Episode endpoints may still be selected as evidence, but
            # the edge itself must never be presented as answer provenance or
            # fed into growth as if it were a factual relation.
            if (
                str(raw_path.get("association_mode", "")) == "contextual_recall"
                or str(raw_path.get("claim_level", "")) == "retrieval_only"
                or str(raw_path.get("relation_key", "")) == "contextual_recall"
            ):
                continue
            item = dict(raw_path)
            association_id = int(item.get("association_id", -1))
            if association_id in preferred_scores:
                item["association_cue_similarity"] = float(
                    preferred_scores[association_id]
                )
            item["query_relevance"] = round(
                cls._path_query_relevance(question, item), 6
            )
            item["path_class"] = (
                "structural_involves"
                if str(item.get("relation_key", "")) == "involves"
                else "informative"
            )
            candidates.append(item)

        def score_key(item: dict) -> tuple[float, float]:
            return (
                float(item.get("query_relevance", 0.0)),
                float(item.get("path_score", 0.0)),
            )

        def preferred_score_key(item: dict) -> tuple[int, float, float, float]:
            cue_similarity = item.get("association_cue_similarity")
            return (
                1 if cue_similarity is not None else 0,
                float(cue_similarity or 0.0),
                float(item.get("query_relevance", 0.0)),
                float(item.get("path_score", 0.0)),
            )

        preferred_paths = sorted(
            [
                item
                for item in candidates
                if int(item.get("association_id", -1)) in preferred
            ],
            key=preferred_score_key,
            reverse=True,
        )
        remaining = [
            item
            for item in candidates
            if int(item.get("association_id", -1)) not in preferred
        ]
        informative = sorted(
            [item for item in remaining if item["path_class"] == "informative"],
            key=score_key,
            reverse=True,
        )
        structural = sorted(
            [
                item
                for item in remaining
                if item["path_class"] == "structural_involves"
            ],
            key=score_key,
            reverse=True,
        )
        informative_quota = max(1, math.ceil(max(1, path_limit) * 0.75))
        structural_quota = max(0, max(1, path_limit) - informative_quota)
        ranked = [
            *preferred_paths,
            *informative[:informative_quota],
            *structural[:structural_quota],
        ]
        used = {int(item["association_id"]) for item in ranked}
        ranked.extend(
            sorted(
                [
                    item
                    for item in remaining
                    if int(item["association_id"]) not in used
                ],
                key=score_key,
                reverse=True,
            )
        )
        return ranked

    @classmethod
    def _select_diverse_episode_ids(
        cls,
        episodes: list[dict],
        quota: int,
        question: str,
        preferred_episode_ids: list[int] | None,
    ) -> set[int]:
        """Compress a broad candidate set without selecting paraphrase clusters.

        Dense/Sparse retrieval is deliberately high-recall.  This selector uses
        lightweight lexical coverage and redundancy penalties only after recall,
        so no evidence is invented and no corpus-specific rule is required.
        """
        if quota <= 0:
            return set()
        episode_by_id = {int(item["id"]): item for item in episodes}
        preferred = [
            int(value)
            for value in dict.fromkeys(preferred_episode_ids or [])
            if int(value) in episode_by_id
        ]
        # Preserve the explicit contract for small atomic-query anchor sets.
        if len(preferred) <= quota:
            selected = set(preferred)
        else:
            selected = set()

        question_features = cls._path_text_features(question)
        feature_map = {
            node_id: cls._path_text_features(
                " ".join(
                    (
                        str(item.get("text", "")),
                        " ".join(str(value) for value in item.get("participants", [])),
                        str(item.get("source_key", "")),
                        str(item.get("story_time_text", "")),
                    )
                )
            )
            for node_id, item in episode_by_id.items()
        }
        preferred_rank = {
            node_id: rank for rank, node_id in enumerate(preferred, start=1)
        }
        maximum_score = max(
            (float(item.get("score", 0.0)) for item in episodes), default=1.0
        ) or 1.0
        covered: set[str] = set()
        source_counts: dict[str, int] = {}
        for node_id in selected:
            covered.update(feature_map[node_id].intersection(question_features))
            source_key = str(episode_by_id[node_id].get("source_key", ""))
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

        pool = list(episode_by_id)
        while len(selected) < min(quota, len(pool)):
            best_id: int | None = None
            best_score = -math.inf
            for node_id in pool:
                if node_id in selected:
                    continue
                item = episode_by_id[node_id]
                features = feature_map[node_id]
                overlap = features.intersection(question_features)
                relevance = (
                    len(overlap)
                    / math.sqrt(len(features) * len(question_features))
                    if features and question_features
                    else 0.0
                )
                novel = overlap.difference(covered)
                novelty = (
                    len(novel) / math.sqrt(len(features) * len(question_features))
                    if features and question_features
                    else 0.0
                )
                redundancy = 0.0
                for selected_id in selected:
                    other = feature_map[selected_id]
                    union = features.union(other)
                    if union:
                        redundancy = max(
                            redundancy,
                            len(features.intersection(other)) / len(union),
                        )
                rank = preferred_rank.get(node_id)
                preferred_bonus = (
                    0.12 / (1.0 + (rank - 1) / 20.0)
                    if rank is not None
                    else 0.0
                )
                source_key = str(item.get("source_key", ""))
                source_count = source_counts.get(source_key, 0)
                score = (
                    0.42 * relevance
                    + 0.30 * novelty
                    + 0.20 * (float(item.get("score", 0.0)) / maximum_score)
                    + preferred_bonus
                    + (0.04 if source_count == 0 else 0.0)
                    - 0.25 * redundancy
                    - 0.025 * source_count
                )
                if score > best_score:
                    best_id = node_id
                    best_score = score
            if best_id is None:
                break
            selected.add(best_id)
            covered.update(feature_map[best_id].intersection(question_features))
            source_key = str(episode_by_id[best_id].get("source_key", ""))
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        return selected

    @classmethod
    def _select_answer_evidence(
        cls,
        episodes: list[dict],
        paths: list[dict],
        episode_limit: int,
        path_limit: int,
        preferred_association_ids: set[int] | None = None,
        question: str = "",
        preferred_episode_ids: list[int] | None = None,
        learned_bridge_slots: int = 2,
        learned_bridge_min_query_relevance: float = 0.03,
        learned_bridge_duplicate_threshold: float = 0.05,
        coverage_groups: list[dict] | None = None,
        preferred_association_scores: dict[int, float] | None = None,
        protected_episode_ids: set[int] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Keep the base Top-K and admit only a few audited learned bridges.

        Legacy graph density must not automatically evict a quarter of the
        reranked evidence. A durable query-grown edge may replace a tail item
        only when it is relevant to this question and bridges from an Episode
        already selected. Edges created by the current query may retain both
        endpoints so their provenance stays immediately auditable.
        """
        if episode_limit <= 0:
            return [], []
        episode_by_id = {int(item["id"]): item for item in episodes}
        preferred = [
            int(value)
            for value in dict.fromkeys(preferred_episode_ids or [])
            if int(value) in episode_by_id
        ]
        preferred_rank = {
            node_id: rank for rank, node_id in enumerate(preferred, start=1)
        }
        ranked_paths = cls._rank_answer_paths(
            paths,
            question,
            path_limit,
            preferred_association_ids,
            preferred_association_scores,
        )
        # Once the reranker has produced an ordered shortlist, generic graph
        # nodes must not enter a smaller budget merely because they expand the
        # lexical pool. Learned bridges get a separate, explicit lane below.
        base_pool = episodes
        if preferred:
            preferred_set = set(preferred)
            preferred_pool = [
                item for item in episodes if int(item["id"]) in preferred_set
            ]
            if preferred_pool:
                base_pool = preferred_pool
        selected_ids = cls._select_diverse_episode_ids(
            base_pool,
            min(len(episodes), episode_limit),
            question,
            preferred,
        )
        externally_protected = {
            int(value)
            for value in (protected_episode_ids or set())
            if int(value) in episode_by_id
        }
        for node_id in externally_protected:
            if node_id in selected_ids:
                continue
            if len(selected_ids) >= episode_limit:
                removable = [
                    value
                    for value in selected_ids
                    if value not in externally_protected
                ]
                if not removable:
                    break
                selected_ids.remove(
                    max(
                        removable,
                        key=lambda value: preferred_rank.get(
                            value, len(preferred) + 1
                        ),
                    )
                )
            selected_ids.add(node_id)

        current_growth_ids = preferred_association_ids or set()
        retrieval_cue_ids = set(preferred_association_scores or {})
        bridge_paths_used: set[int] = set()
        bridge_added_ids: set[int] = set()
        remaining_bridge_slots = max(0, int(learned_bridge_slots))

        def coverage_protected_ids() -> set[int]:
            protected_ids: set[int] = set()
            for group in coverage_groups or []:
                raw_ids = group.get("episode_ids", [])
                if not isinstance(raw_ids, list):
                    continue
                present = {
                    int(value) for value in raw_ids
                    if int(value) in selected_ids
                }
                if str(group.get("mode", "alternatives")) == "joint":
                    protected_ids.update(present)
                elif len(present) == 1:
                    protected_ids.update(present)
            return protected_ids

        feature_map = {
            node_id: cls._path_text_features(
                " ".join(
                    (
                        str(item.get("text", "")),
                        " ".join(
                            str(value) for value in item.get("participants", [])
                        ),
                        str(item.get("source_key", "")),
                    )
                )
            )
            for node_id, item in episode_by_id.items()
        }
        coverage_membership: dict[int, int] = {}
        for group in coverage_groups or []:
            raw_ids = group.get("episode_ids", [])
            if not isinstance(raw_ids, list):
                continue
            for value in raw_ids:
                node_id = int(value)
                coverage_membership[node_id] = (
                    coverage_membership.get(node_id, 0) + 1
                )
        for path in ranked_paths:
            if remaining_bridge_slots <= 0:
                break
            association_id = int(path.get("association_id", -1))
            is_current_growth = association_id in current_growth_ids
            is_retrieval_cue = association_id in retrieval_cue_ids
            created_reason = str(path.get("created_reason", ""))
            is_oracle = created_reason.startswith("Stage 8 oracle probe")
            is_durable_learned = (
                "查询中自主增长：" in created_reason
                and str(path.get("audit_status", "")) == "dual_accepted"
            )
            if not (is_current_growth or is_oracle or is_durable_learned):
                continue
            if int(path.get("polarity", 1)) < 0:
                continue
            if (
                not is_current_growth
                and not is_retrieval_cue
                and float(path.get("query_relevance", 0.0))
                < float(learned_bridge_min_query_relevance)
            ):
                continue
            endpoint_ids = [
                int(endpoint[1])
                for endpoint in (path.get("from"), path.get("to"))
                if isinstance(endpoint, (list, tuple))
                and len(endpoint) == 2
                and endpoint[0] == "episode"
                and int(endpoint[1]) in episode_by_id
            ]
            missing = [node_id for node_id in endpoint_ids if node_id not in selected_ids]
            if not is_current_growth and missing and not any(
                node_id in selected_ids or node_id in preferred_rank
                for node_id in endpoint_ids
            ):
                continue
            additions = missing[:remaining_bridge_slots]
            if not additions:
                continue
            protected = set(endpoint_ids).union(externally_protected)
            for node_id in additions:
                candidate_features = feature_map.get(node_id, set())
                maximum_redundancy = 0.0
                for selected_id in selected_ids:
                    selected_features = feature_map.get(selected_id, set())
                    union = candidate_features.union(selected_features)
                    if union:
                        maximum_redundancy = max(
                            maximum_redundancy,
                            len(candidate_features.intersection(selected_features))
                            / len(union),
                        )
                adds_uncovered_slot = any(
                    node_id in {
                        int(value) for value in group.get("episode_ids", [])
                    }
                    and not selected_ids.intersection(
                        int(value) for value in group.get("episode_ids", [])
                    )
                    for group in coverage_groups or []
                    if isinstance(group.get("episode_ids", []), list)
                )
                # A late closure is already restricted to positive,
                # dual-audited generation-1 Episode bridges. Its endpoints are
                # expected to share entity vocabulary, so the generic 0.05
                # lexical duplicate threshold is too aggressive here. Exact
                # or near-exact duplicate Episodes still fail the 0.20 guard.
                effective_duplicate_threshold = float(
                    learned_bridge_duplicate_threshold
                )
                if path.get("late_bridge_closure"):
                    effective_duplicate_threshold = max(
                        effective_duplicate_threshold,
                        0.20,
                    )
                if (
                    maximum_redundancy
                    >= effective_duplicate_threshold
                    and not adds_uncovered_slot
                    and not is_retrieval_cue
                ):
                    continue
                if len(selected_ids) >= episode_limit:
                    slot_protected = coverage_protected_ids()
                    removable = [
                        selected_id
                        for selected_id in selected_ids
                        if selected_id not in protected
                        and selected_id not in bridge_added_ids
                        and selected_id not in slot_protected
                    ]
                    if not removable:
                        # Preserve the entire protected base Top-K. A late
                        # bridge has its own bounded slot, so it may extend the
                        # answer set instead of deleting required evidence.
                        if not path.get("late_bridge_closure"):
                            break
                    else:
                        def replacement_key(
                            selected_id: int,
                        ) -> tuple[int, float, int, float]:
                            selected_features = feature_map.get(selected_id, set())
                            union = candidate_features.union(selected_features)
                            redundancy = (
                                len(candidate_features.intersection(selected_features))
                                / len(union)
                                if union
                                else 0.0
                            )
                            return (
                                -coverage_membership.get(selected_id, 0),
                                redundancy,
                                preferred_rank.get(
                                    selected_id, len(preferred) + 1
                                ),
                                -float(
                                    episode_by_id[selected_id].get("score", 0.0)
                                ),
                            )

                        victim = max(
                            removable,
                            key=replacement_key,
                        )
                        selected_ids.remove(victim)
                selected_ids.add(node_id)
                bridge_added_ids.add(node_id)
                remaining_bridge_slots -= 1
                bridge_paths_used.add(association_id)
                if remaining_bridge_slots <= 0:
                    break
        for item in episodes:
            if len(selected_ids) >= episode_limit:
                break
            selected_ids.add(int(item["id"]))
        selected_episodes = [
            item for item in episodes if int(item["id"]) in selected_ids
        ]
        # Membership is selected with sets for efficient diversity/bridge
        # decisions, but presentation order must retain the rerank contract.
        # Otherwise repository/graph score order can move the strongest direct
        # evidence behind a large source cohort before the answer model sees it.
        selected_episodes.sort(
            key=lambda item: (
                preferred_rank.get(int(item["id"]), len(preferred) + 1),
                -float(item.get("score", 0.0)),
            )
        )
        auditable_paths: list[dict] = []
        for path in ranked_paths:
            if int(path.get("association_id", -1)) in bridge_paths_used:
                path["learned_bridge_slot_used"] = True
            endpoints_are_available = all(
                not (
                    isinstance(endpoint, (list, tuple))
                    and len(endpoint) == 2
                    and endpoint[0] == "episode"
                )
                or int(endpoint[1]) in selected_ids
                for endpoint in (path.get("from"), path.get("to"))
            )
            if endpoints_are_available:
                auditable_paths.append(path)
            if len(auditable_paths) >= path_limit:
                break
        return selected_episodes, auditable_paths

    @staticmethod
    def _growth_premise_ids(row: dict) -> set[int]:
        """Return Association premises recorded in a durable growth row."""
        try:
            evidence = json.loads(str(row.get("evidence_json", "[]")))
        except (TypeError, json.JSONDecodeError):
            return set()
        if not isinstance(evidence, list):
            return set()
        premise_ids: set[int] = set()
        for item in evidence:
            if not isinstance(item, dict) or item.get("type") != "association":
                continue
            try:
                premise_ids.add(int(item["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return premise_ids

    @staticmethod
    def _counterfactual_utility_from_results(
        treatment: dict,
        masked: dict,
        changed_ids: set[int],
    ) -> dict:
        """Measure query-batch utility from a shared deterministic plan.

        No evidence manifest is consulted here. A durable growth batch must be
        visible in the treatment answer paths *and* expose at least one direct
        Episode that disappears when every changed row is masked.
        """
        treatment_episode_ids = [
            int(value) for value in treatment.get("episode_ids", [])
        ]
        masked_episode_ids = [int(value) for value in masked.get("episode_ids", [])]
        treatment_set = set(treatment_episode_ids)
        masked_set = set(masked_episode_ids)
        changed_path_ids = sorted(
            {
                int(path["association_id"])
                for path in treatment.get("association_paths", [])
                if int(path.get("association_id", -1)) in changed_ids
            }
        )
        additional_episode_ids = sorted(treatment_set - masked_set)
        displaced_episode_ids = sorted(masked_set - treatment_set)
        # With a shared plan the only treatment/mask difference is the changed
        # Association batch. A newly selected Episode is therefore causal even
        # when the bounded answer-path list omits the traversed bridge itself.
        utility_observed = bool(additional_episode_ids)
        treatment_sources = {
            str(item.get("source_key", ""))
            for item in treatment.get("evidence_episodes", [])
            if str(item.get("source_key", ""))
        }
        masked_sources = {
            str(item.get("source_key", ""))
            for item in masked.get("evidence_episodes", [])
            if str(item.get("source_key", ""))
        }
        return {
            "enabled": True,
            "scope": "query_batch",
            "changed_association_ids": sorted(changed_ids),
            "changed_path_ids": changed_path_ids,
            "treatment_episode_ids": treatment_episode_ids,
            "masked_episode_ids": masked_episode_ids,
            "additional_episode_ids": additional_episode_ids,
            "displaced_episode_ids": displaced_episode_ids,
            "additional_source_keys": sorted(treatment_sources - masked_sources),
            "displaced_source_keys": sorted(masked_sources - treatment_sources),
            "evidence_gain_count": len(additional_episode_ids),
            "causal_utility_observed": utility_observed,
            "persistable_changed_ids": (
                (changed_path_ids or sorted(changed_ids))
                if utility_observed
                else []
            ),
        }

    def _counterfactual_growth_utility(
        self,
        replay_plan: dict,
        created_ids: list[int],
        reinforced_ids: list[int],
        reinforced_before: dict[int, dict],
    ) -> dict:
        changed = set(
            int(value) for value in [*created_ids, *reinforced_ids]
        )
        if not changed:
            return {
                "enabled": bool(
                    self.config.retrieval.growth_counterfactual_utility_enabled
                ),
                "scope": "query_batch",
                "changed_association_ids": [],
                "changed_path_ids": [],
                "additional_episode_ids": [],
                "displaced_episode_ids": [],
                "additional_source_keys": [],
                "displaced_source_keys": [],
                "evidence_gain_count": 0,
                "causal_utility_observed": False,
                "persistable_changed_ids": [],
            }
        if not self.config.retrieval.growth_counterfactual_utility_enabled:
            return {
                "enabled": False,
                "scope": "query_batch",
                "changed_association_ids": sorted(changed),
                "persistable_changed_ids": sorted(changed),
            }
        after = self.associations.snapshot(list(changed))
        delta = AssociationDelta(
            created=[
                {"id": value, "before": None, "after": after[value]}
                for value in dict.fromkeys(int(item) for item in created_ids)
                if value in after
            ],
            reinforced=[
                {
                    "id": value,
                    "before": reinforced_before[value],
                    "after": after[value],
                }
                for value in dict.fromkeys(int(item) for item in reinforced_ids)
                if value not in set(int(item) for item in created_ids)
                and value in reinforced_before
                and value in after
            ],
        )
        treatment = self.replay_retrieval(replay_plan)
        overlay = AssociationOverlay.from_delta(self.associations, delta)
        masked_engine = QueryEngine(
            self.config,
            self.model,
            self.episode_index,
            self.concept_index,
            self.episodes,
            self.concepts,
            self.sources,
            overlay,
            self.logger,
            association_index=self.association_index,
            paragraph_index=self.paragraph_index,
            paragraphs=self.paragraphs,
            episode_sparse_index=self.episode_sparse_index,
            source_sparse_index=self.source_sparse_index,
        )
        masked = masked_engine.replay_retrieval(replay_plan)
        result = self._counterfactual_utility_from_results(
            treatment, masked, changed
        )
        if self.logger:
            self.logger.emit("growth_counterfactual_utility", **result)
        return result

    def _prune_unused_growth(
        self,
        created_ids: list[int],
        reinforced_ids: list[int],
        reinforced_before: dict[int, dict],
        used_edge_ids: list[int],
    ) -> dict:
        """Keep only query growth that is traceable from final answer paths.

        The existing repository remains the provisional execution substrate,
        so no second graph implementation is needed. Before returning, unused
        creations are deleted and unused reinforcements are restored. Premise
        edges required by a retained higher-generation edge are retained too.
        """
        created = list(dict.fromkeys(int(value) for value in created_ids))
        reinforced = list(dict.fromkeys(int(value) for value in reinforced_ids))
        changed = set([*created, *reinforced])
        retained = changed.intersection(int(value) for value in used_edge_ids)
        rows = self.associations.snapshot(list(changed)) if changed else {}
        pending = list(retained)
        while pending:
            association_id = pending.pop()
            row = rows.get(association_id)
            if row is None:
                continue
            for premise_id in self._growth_premise_ids(row):
                if premise_id in changed and premise_id not in retained:
                    retained.add(premise_id)
                    pending.append(premise_id)

        removed_created: list[int] = []
        for association_id in reversed(created):
            if association_id in retained:
                continue
            if self.associations.delete(association_id):
                removed_created.append(association_id)

        restore_payload = {
            association_id: reinforced_before[association_id]
            for association_id in reinforced
            if association_id not in set(created)
            and association_id not in retained
            and association_id in reinforced_before
        }
        restored_reinforced = self.associations.restore_rows(restore_payload)
        retained_created = [value for value in created if value in retained]
        retained_reinforced = [
            value for value in reinforced
            if value in retained and value not in set(retained_created)
        ]
        result = {
            "enabled": True,
            "retained_created_ids": retained_created,
            "retained_reinforced_ids": retained_reinforced,
            "removed_created_ids": sorted(removed_created),
            "restored_reinforced_ids": sorted(restored_reinforced),
            "premise_closure_ids": sorted(
                retained.difference(int(value) for value in used_edge_ids)
            ),
        }
        if self.logger:
            self.logger.emit("growth_utility_gate", **result)
        return result

    @staticmethod
    def _remap_staged_result(result: dict, mapping: dict[int, int]) -> dict:
        if not mapping:
            result["growth_staging"] = {
                "enabled": True,
                "committed": False,
                "temporary_to_durable_ids": {},
            }
            return result

        def remap_list(values) -> list[int]:
            return [mapping.get(int(value), int(value)) for value in values]

        for key in (
            "association_ids",
            "new_association_ids",
            "reinforced_association_ids",
        ):
            result[key] = remap_list(result.get(key, []))
        for path in result.get("association_paths", []):
            if "association_id" in path:
                value = int(path["association_id"])
                path["association_id"] = mapping.get(value, value)
        gate = result.get("growth_utility_gate", {})
        for key in (
            "retained_created_ids",
            "retained_reinforced_ids",
            "premise_closure_ids",
        ):
            gate[key] = remap_list(gate.get(key, []))
        utility = result.get("growth_counterfactual_utility", {})
        for key in (
            "changed_association_ids",
            "changed_path_ids",
            "persistable_changed_ids",
        ):
            utility[key] = remap_list(utility.get(key, []))
        answer = str(result.get("answer", ""))
        chronology_notes = [
            str(value) for value in result.get("chronology_notes", [])
        ]
        for temporary_id, durable_id in mapping.items():
            answer = answer.replace(f"#{temporary_id}", f"#{durable_id}")
            answer = answer.replace(
                f"Association {temporary_id}", f"Association {durable_id}"
            )
            chronology_notes = [
                value.replace(f"#{temporary_id}", f"#{durable_id}")
                for value in chronology_notes
            ]
        result["answer"] = answer
        result["chronology_notes"] = chronology_notes
        result["growth_staging"] = {
            "enabled": True,
            "committed": True,
            "temporary_to_durable_ids": {
                str(key): value for key, value in mapping.items() if key < 0
            },
        }
        return result

    def _execute_strict_trace_read_only(self, execute_impl) -> dict:
        """Run a traced query without allowing association-side mutations.

        A strict receipt is persisted separately from SQLite.  Until learning
        has a durable cross-store receipt, committing a staged association (or
        even incrementing ``use_count``) before ``request_completed`` would
        create an unverifiable half-success if trace persistence failed.  A
        traced request therefore uses the normal retrieval view through an
        :class:`AssociationOverlay`, whose mutation hooks are no-ops, and
        disables query-time growth.

        This is deliberately narrower than the future v3 replay runner: the
        ordinary query still performs its real retrieval work and may make the
        configured, observed model calls.  It simply cannot learn or mutate
        association usage while its strict receipt is being assembled.
        """

        durable_associations = self.associations
        if isinstance(durable_associations, StagedAssociationOverlay):
            # An outer caller could otherwise commit this mutable overlay after
            # the trace boundary, which makes the receipt's learning state
            # ambiguous.  Refuse rather than silently treating it as read-only.
            raise RuntimeError(
                "strict trace cannot run inside a mutable staged association overlay"
            )

        if isinstance(durable_associations, AssociationOverlay):
            result = execute_impl(allow_association_learning=False)
        elif isinstance(durable_associations, AssociationRepository):
            read_only_associations = AssociationOverlay(durable_associations)
            durable_traverser = self.traverser
            durable_growth = self.growth
            durable_chronology = self.chronology
            self.associations = read_only_associations
            self.traverser = GraphTraverser(read_only_associations)
            self.growth = AssociationGrowthEngine(
                self.model,
                read_only_associations,
                self.episodes,
                self.config.weights,
                self.logger,
            )
            self.chronology = ChronologyService(
                self.episodes, read_only_associations, self.logger
            )
            try:
                result = execute_impl(allow_association_learning=False)
            finally:
                self.associations = durable_associations
                self.traverser = durable_traverser
                self.growth = durable_growth
                self.chronology = durable_chronology
        elif hasattr(durable_associations, "mark_used"):
            # A custom mutable association implementation cannot prove the
            # same no-write invariant as AssociationOverlay.  Fail closed.
            raise RuntimeError(
                "strict trace requires AssociationRepository or a read-only AssociationOverlay"
            )
        else:
            # Small dependency-injected unit boundaries sometimes supply no
            # association API at all.  They cannot expose a write hook, so
            # preserve the trace lifecycle test seam while still passing the
            # explicit no-learning flag to the implementation.
            result = execute_impl(allow_association_learning=False)

        result["growth_staging"] = {
            "enabled": False,
            "committed": False,
            "temporary_to_durable_ids": {},
            "reason": "strict_trace_read_only",
        }
        result["strict_trace_read_only"] = True
        return result

    @staticmethod
    def _v3_learning_report(
        status: str,
        reason: str = "",
        *,
        candidate_ids: Sequence[str] = (),
        receipts: Sequence[Mapping[str, object]] = (),
        revisit_contracts: Sequence[Mapping[str, object]] = (),
        revisit_projections: Sequence[Mapping[str, object]] = (),
        rejected_count: int = 0,
        deferred_candidate_options: int = 0,
    ) -> dict[str, object]:
        """Return the deliberately redacted public T13 result shape."""

        normalized_receipts: list[dict[str, object]] = []
        for receipt in receipts:
            try:
                receipt_id = int(receipt.get("receipt_id", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                receipt_id = 0
            normalized_receipts.append(
                {
                    # Candidate IDs and receipt IDs are opaque durable
                    # identifiers.  Never pass a source key, source fact,
                    # query text, or vector through this response field.
                    "candidate_id": str(
                        receipt.get("creation_request_id", "") or ""
                    ),
                    "receipt_id": receipt_id if receipt_id > 0 else None,
                    "status": str(receipt.get("status", "") or ""),
                    "ready_for_revisit": bool(
                        receipt.get("ready_for_revisit", False)
                    ),
                }
            )
        normalized_contracts: list[dict[str, object]] = []
        for contract in revisit_contracts:
            try:
                receipt_id = int(contract.get("receipt_id", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                receipt_id = 0
            contract_status = str(contract.get("status", "") or "")
            if contract_status not in {
                "created",
                "idempotent",
                "deferred_not_ready",
                "skipped_invalid_draft",
                "skipped_invalid_receipt",
                "skipped_missing_receipt",
                "skipped_receipt_mismatch",
                "skipped_association_mismatch",
                "failed",
            }:
                contract_status = "failed"
            normalized_contracts.append(
                {
                    "candidate_id": str(contract.get("candidate_id", "") or ""),
                    "receipt_id": receipt_id if receipt_id > 0 else None,
                    "status": contract_status,
                }
            )
        normalized_projections: list[dict[str, object]] = []
        for projection in revisit_projections:
            try:
                receipt_id = int(projection.get("receipt_id", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                receipt_id = 0
            projection_status = str(projection.get("status", "") or "")
            if projection_status not in {
                "ready",
                "deferred_not_ready",
                "failed",
                "rejected_not_reconstructable",
            }:
                projection_status = "failed"
            projection_reason = str(projection.get("reason", "") or "")
            if projection_reason not in {
                "runtime_manifest_ready",
                "runtime_manifest_deferred",
                "runtime_manifest_failed",
                "runtime_projection_not_reconstructable",
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
            }:
                projection_reason = "runtime_manifest_failed"
            normalized_projections.append(
                {
                    "candidate_id": str(projection.get("candidate_id", "") or ""),
                    "receipt_id": receipt_id if receipt_id > 0 else None,
                    "status": projection_status,
                    "reason": projection_reason,
                }
            )
        return {
            "status": str(status),
            "reason": str(reason),
            "candidate_ids": [str(value) for value in candidate_ids if str(value)],
            "receipt_ids": [
                int(item["receipt_id"])
                for item in normalized_receipts
                if isinstance(item.get("receipt_id"), int)
            ],
            "receipts": normalized_receipts,
            # Contract statuses are deliberately opaque.  In particular no
            # source-fact, source key, query text, vector, or contract payload
            # escapes the application finalizer through this result.
            "revisit_contracts": normalized_contracts,
            # Receipt readiness is not an exact-reuse readiness claim.  This
            # independent, redacted outcome makes optional V17 projection and
            # publication observable without exposing a runtime ticket.
            "revisit_projections": normalized_projections,
            "rejected_count": max(0, int(rejected_count)),
            # T13b intentionally commits only one event/plan transaction.
            # This structural count makes a bounded fail-closed deferral
            # observable without exposing targets, slots, or source data.
            "deferred_candidate_options": max(
                0, int(deferred_candidate_options)
            ),
        }

    @staticmethod
    def _v3_learning_vector(
        bundle: QueryVectorBundle,
        physical_id: str,
    ) -> tuple[np.ndarray | None, str]:
        """Validate a persisted-cue input without coercing or re-embedding it."""

        try:
            vector = np.asarray(bundle.vector_for_physical(physical_id))
        except (KeyError, TypeError, ValueError):
            return None, "vector_reference_missing"
        if vector.dtype != np.float32:
            return None, "vector_dtype_invalid"
        if vector.shape != (int(bundle.dimension),):
            return None, "vector_shape_invalid"
        if not np.all(np.isfinite(vector)):
            return None, "vector_nonfinite"
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            return None, "vector_not_normalized"
        # QueryVectorBundle owns this immutable RAM object.  Returning it does
        # not issue an embedding call or mutate the request-local vector.
        return vector, ""

    @staticmethod
    def _v3_learning_binding_for_slot(
        bundle: QueryVectorBundle,
        slot_id: str,
        *,
        allow_whole_question_binding: bool = False,
    ) -> tuple[object | None, str]:
        """Require one exact logical binding; never re-embed or use text fallback.

        A singleton requirement that is literally the raw question may reuse
        the already-observed whole-question vector. This is not a semantic
        nearest-neighbour substitution: it is the same logical input and the
        same immutable physical vector, and it avoids a redundant provider
        embedding merely to duplicate that vector as an ``atomic`` binding.
        """

        bindings = tuple(
            item
            for item in bundle.bindings_for_slot(slot_id)
            if str(item.role) == "atomic"
        )
        if not bindings and allow_whole_question_binding:
            bindings = tuple(
                item
                for item in bundle.logical_bindings
                if str(item.role) == "whole"
                and str(item.physical_id) == str(bundle.whole_physical_id)
            )
        if not bindings:
            return None, "slot_vector_binding_missing"
        if len(bindings) != 1:
            return None, "slot_vector_binding_ambiguous"
        binding = bindings[0]
        if (
            not str(binding.query_id).strip()
            or not str(binding.text_hash).strip()
            or not str(binding.physical_id).strip()
            or str(binding.embedding_space_id) != str(bundle.embedding_space_id)
        ):
            return None, "slot_vector_binding_invalid"
        return binding, ""

    @staticmethod
    def _v3_learning_source_supports(
        contribution: CandidateContribution,
        slot: EvidenceSlot,
    ) -> tuple[ClauseSupport, ...]:
        """Keep only existing source-bound mappings for one frozen slot."""

        supports = tuple(
            item
            for item in contribution.clause_supports
            if item.slot_id == slot.slot_id
            and item.is_required_coverage
            and str(item.mapping_ref).strip()
        )
        if not supports:
            return ()
        if slot.support_mode == "joint":
            covered = {item.clause_id for item in supports}
            if not set(slot.clause_ids).issubset(covered):
                return ()
        return supports

    def _v3_learning_anchors(
        self,
        capture: _V3LearningCapture,
        *,
        target_episode_id: int,
    ) -> tuple[LearningAnchor, ...]:
        """Derive independent base anchors without trusting merged seed origin."""

        # ``initial_candidate_episode_ids`` may include graph-expanded or
        # source-cohort rows.  An anchor is stronger: it must have an explicit
        # direct-retrieval provenance record captured before any association
        # path was expanded.  Missing provenance is intentionally zero, not a
        # compatibility fallback to the broader candidate set.
        base_ids = set(capture.initial_candidate_episode_ids).intersection(
            capture.independent_base_episode_ids
        )
        # A source-bound, independently retrieved Episode that the Q1
        # reranker actually delivered may supply contextual *context* even
        # when it does not itself prove the target slot. It remains bounded
        # to the Q1 delivery set; an arbitrary rank-only candidate cannot be
        # promoted into an anchor.
        delivered_base_ids = set(capture.initial_delivered_episode_ids)
        excluded = {
            int(target_episode_id),
            *capture.contextual_expansion_episode_ids,
            *capture.cue_endpoint_episode_ids,
        }
        anchors: list[LearningAnchor] = []
        for contribution in capture.contributions:
            if (
                contribution.lane not in {"base_source_mapping", "base_rerank"}
                or contribution.episode_id not in base_ids
                or contribution.episode_id in excluded
                or (
                    contribution.lane == "base_rerank"
                    and contribution.episode_id not in delivered_base_ids
                )
            ):
                continue
            # A rank/cosine signal alone is not sufficient to become a
            # learning anchor. Reuse the same full-source fact closure that
            # the selector already validated for its base contribution.
            source_bound_supports = tuple(
                support
                for support in contribution.clause_supports
                if support.is_required_coverage
                and str(support.mapping_ref).strip()
            )
            source_facts = tuple(
                dict.fromkeys(
                    (*contribution.source_facts, *(
                        support.source_fact
                        for support in source_bound_supports
                        if support.source_fact is not None
                    ))
                )
            )
            if not source_facts or not str(contribution.query_ref).strip():
                continue
            provenance_refs = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        contribution.contribution_id,
                        contribution.evidence_ref,
                        *(support.mapping_ref for support in source_bound_supports),
                    )
                    if str(value).strip()
                )
            )
            if not provenance_refs:
                continue
            activation = self._v3_finite_score(
                contribution.feature_map.get("fusion_rank_score", 0.0)
            )
            try:
                anchors.append(
                    LearningAnchor(
                        anchor_type="episode",
                        anchor_id=contribution.episode_id,
                        contribution_id=contribution.contribution_id,
                        source_facts=source_facts,
                        # There is not yet an Episode embedding revision ID in
                        # the schema.  This opaque request-local route ref is
                        # intentionally *not* represented as node-vector
                        # provenance; it only satisfies the immutable receipt
                        # audit field until a future revision identity exists.
                        vector_ref=self._v3_opaque_ref(
                            "base-route-vector-ref-v1", contribution.query_ref
                        ),
                        provenance_refs=provenance_refs,
                        activation=activation,
                        lane="base",
                        independent=True,
                    )
                )
            except (TypeError, ValueError):
                # Malformed local provenance is a reason to omit this route,
                # never a reason to synthesize a replacement anchor.
                continue
        return tuple(anchors)

    def _v3_revisit_contract_draft(
        self,
        *,
        question: str,
        capture: _V3LearningCapture,
        candidate: LearningCandidate,
        plan: LearningCandidatePlan,
        target_contribution: CandidateContribution,
        supports: Sequence[ClauseSupport],
        bundle: QueryVectorBundle,
        domain: str,
        source_request_hash: str,
        context_scope_hash: str | None,
        projection_diagnostic: dict[str, str] | None = None,
    ) -> ContextualRevisitContractDraft | None:
        """Prepare one Q1-only redacted contract draft, or fail closed.

        This boundary has the last safe view of the request's full typed
        requirements, current source closure, vector identities and selected
        anchor.  It deliberately does not write anything and never attempts
        to reconstruct this information from a receipt or public trace.
        """

        def reject(reason: str) -> None:
            if projection_diagnostic is not None:
                projection_diagnostic["ordinary_contract_reason"] = reason
            return None

        requirements = capture.requirements
        if (
            not isinstance(requirements, RequirementResolution)
            or requirements.request_mode != "factual"
            or requirements.status != "resolved"
            or not isinstance(candidate, LearningCandidate)
            or not isinstance(plan, LearningCandidatePlan)
            or candidate.anchor_type != "episode"
            or int(candidate.anchor_id) == int(candidate.target_episode_id)
            or int(candidate.target_episode_id) != int(target_contribution.episode_id)
            or str(candidate.slot_id) != str(target_contribution.slot_id)
            # Automatic Q1 contracts cannot use a bare question as a
            # cross-conversation context identity.  A caller must opt in to
            # an opaque scope; ordinary learning still proceeds without one.
            or context_scope_hash is None
        ):
            return reject("input_or_scope_invalid")

        # A single edge has one need cue.  Do not persist a contract whose
        # multi-slot source-mapping semantics cannot later be supplied as an
        # exact transient ticket without guessing another edge/proof.
        required_slots = tuple(slot for slot in requirements.requirements if slot.required)
        if len(required_slots) != 1 or str(required_slots[0].slot_id) != str(candidate.slot_id):
            return reject("single_required_slot_required")
        slot = required_slots[0]
        required_clause_ids = tuple(sorted({str(value) for value in slot.clause_ids}))
        if not required_clause_ids:
            return reject("required_clause_missing")

        selected_anchors = tuple(
            anchor
            for anchor in plan.selected_anchors
            if int(anchor.anchor_id) == int(candidate.anchor_id)
            and str(anchor.contribution_id) == str(candidate.anchor_contribution_id)
        )
        if len(selected_anchors) != 1:
            return reject("selected_anchor_unavailable")
        anchor = selected_anchors[0]
        if (
            anchor.anchor_type != "episode"
            or not anchor.is_eligible_base_anchor
            or anchor.is_contextual
            or not anchor.independent
            or not math.isfinite(float(anchor.activation))
            or float(anchor.activation) <= 0.0
        ):
            return reject("selected_anchor_invalid")

        # Re-open the full source closure at the precise finalization point.
        # Candidate/source facts captured earlier are not enough: a source can
        # be revised between selector work and this post-answer callback.
        facts, reasons = self._v3_source_fact_closure(
            (int(candidate.anchor_id), int(candidate.target_episode_id))
        )
        anchor_fact = facts.get(int(candidate.anchor_id))
        target_fact = facts.get(int(candidate.target_episode_id))
        if (
            anchor_fact is None
            or target_fact is None
            or reasons.get(int(candidate.anchor_id)) != "source_bound"
            or reasons.get(int(candidate.target_episode_id)) != "source_bound"
            or anchor_fact not in anchor.source_facts
            or anchor_fact not in candidate.source_facts
            or target_fact not in candidate.source_facts
        ):
            return reject("source_closure_unavailable")

        source_supports = tuple(
            support
            for support in supports
            if support.is_required_coverage
            and str(support.slot_id) == str(slot.slot_id)
            and support.source_fact == target_fact
            and str(support.mapping_ref).strip()
        )
        if not source_supports:
            return reject("source_mapping_support_unavailable")
        covered_clause_ids = {
            str(support.clause_id) for support in source_supports if str(support.clause_id)
        }
        if covered_clause_ids != set(required_clause_ids):
            return reject("source_mapping_clause_mismatch")
        verification_refs = tuple(
            sorted(
                {
                    str(support.mapping_ref).strip()
                    for support in source_supports
                    if str(support.mapping_ref).strip()
                }
            )
        )
        if not verification_refs:
            return reject("source_mapping_verification_missing")
        try:
            effective_endpoint_limit = (
                int(capture.endpoint_limit)
                if capture.endpoint_limit is not None
                else int(getattr(self.contextual_matcher, "endpoint_limit", 0))
            )
            if effective_endpoint_limit <= 0:
                return reject("endpoint_limit_invalid")
            budget = EvidenceSelectionBudget(
                episode_limit=max(0, int(self.config.retrieval.answer_episode_limit)),
            )
            if int(budget.episode_limit) <= 0:
                return reject("delivery_budget_invalid")
            mapping_ref = exact_revisit_aggregate_mapping_ref(
                target_episode_id=int(candidate.target_episode_id),
                slot_id=str(slot.slot_id),
                clause_ids=required_clause_ids,
                source_fact_id=target_fact.fact_id,
                verification_refs=verification_refs,
            )
            target_mapping = ExactRevisitTargetMappingDraft(
                target_episode_id=int(candidate.target_episode_id),
                slot_id=str(slot.slot_id),
                clause_ids=required_clause_ids,
                source_fact_id=target_fact.fact_id,
                mapping_ref=mapping_ref,
            )
            source_facts = (anchor_fact, target_fact)
            draft = ContextualRevisitContractDraft(
                creation_request_id=str(candidate.candidate_id),
                source_request_hash=source_request_hash,
                domain=domain,
                anchor_episode_id=int(candidate.anchor_id),
                context_hash=exact_revisit_context_hash(
                    question,
                    context_scope_hash=context_scope_hash,
                ),
                slot_need_bindings=exact_revisit_slot_need_bindings(
                    requirements, bundle
                ),
                requirements_fingerprint=exact_revisit_requirements_fingerprint(
                    requirements
                ),
                source_closure_fingerprint=exact_revisit_source_closure_fingerprint(
                    source_facts
                ),
                retrieval_policy_fingerprint=exact_revisit_policy_fingerprint(
                    self.config,
                    self.contextual_matcher,
                    endpoint_limit=effective_endpoint_limit,
                ),
                budget_fingerprint=exact_revisit_budget_fingerprint(budget),
                anchor_manifest_fingerprint=exact_revisit_anchor_manifest_fingerprint(
                    {int(candidate.anchor_id): float(anchor.activation)}
                ),
                source_fact_refs_fingerprint=exact_revisit_source_fact_refs_fingerprint(
                    source_facts
                ),
                target_mapping=target_mapping,
            )
            if projection_diagnostic is not None:
                projection_diagnostic["ordinary_contract_reason"] = "ready"
            return draft
        except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError) as error:
            # Any absent or ambiguous source/vector/policy input makes this a
            # normal Q1 creation without a Q2 contract, never a synthetic
            # durable mapping.
            return reject("draft_construction_failed_" + type(error).__name__)

    def _v17_revisit_runtime_projection(
        self,
        *,
        question: str,
        capture: _V3LearningCapture,
        candidate: LearningCandidate,
        plan: LearningCandidatePlan,
        target_contribution: CandidateContribution,
        supports: Sequence[ClauseSupport],
        bundle: QueryVectorBundle,
        domain: str,
        source_request_hash: str,
        context_scope_hash: str | None,
        ordinary_draft: ContextualRevisitContractDraft | None,
    ) -> tuple[
        ContextualRevisitContractDraft,
        ContextualRevisitRuntimeSeed,
        ContextualRestrictedRewriteGuardDraft | None,
    ] | None:
        """Project one strictly reconstructable Q1 edge into V17 runtime IDs.

        The ordinary V16 draft remains the qualification baseline.  Only the
        narrow shape below can be rebuilt from a future question without
        retaining planner labels or source prose: one whole-question,
        alternative slot; one independently selected anchor; and byte-equal
        context/need vectors.  This method creates no ticket and does not
        enter the exact-revisit path.  It merely supplies the redacted seed
        and its matching runtime-ID V16 projection to the normal Q1 write.
        """

        requirements = capture.requirements
        normalized_question = normalize_query_text(question)
        if (
            ordinary_draft is None
            or not isinstance(requirements, RequirementResolution)
            or requirements.request_mode != "factual"
            or requirements.status != "resolved"
            or not normalized_question
            or context_scope_hash is None
            or not isinstance(candidate, LearningCandidate)
            or not isinstance(plan, LearningCandidatePlan)
            or candidate.anchor_type != "episode"
            or int(candidate.anchor_id) == int(candidate.target_episode_id)
            or int(candidate.target_episode_id) != int(target_contribution.episode_id)
        ):
            return None

        # A fresh runtime request can reconstruct only a one-slot template
        # whose semantic content is exactly the current whole question. Rich
        # Q1 requirement metadata (subject/relation/time/negation, etc.) is
        # retained on the authoritative Q1 requirement and still governed the
        # source-backed learning decision. The runtime template does not
        # discard or weaken that requirement: its only request text is the
        # same whole question, and Q2 lookup requires an exact whole-question
        # hash plus domain, scope, vector-space, receipt, source and policy
        # revalidation. Original slot/query/clause identifiers are
        # intentionally not reused because they are planner-local.
        required_slots = tuple(
            slot for slot in requirements.requirements if slot.required
        )
        if len(required_slots) != 1:
            return None
        slot = required_slots[0]
        original_clause_ids = tuple(
            sorted({str(value).strip() for value in slot.clause_ids if str(value).strip()})
        )
        if (
            str(slot.slot_id) != str(candidate.slot_id)
            or normalize_query_text(slot.question) != normalized_question
            or str(slot.support_mode) != "alternative"
            or len(original_clause_ids) != 1
        ):
            return None

        # The V17 seed deliberately has room for exactly one independently
        # selected episode anchor.  A plan with another selected anchor cannot
        # be reduced without changing its learning semantics.
        if len(plan.selected_anchors) != 1:
            return None
        anchor = plan.selected_anchors[0]
        if (
            anchor.anchor_type != "episode"
            or int(anchor.anchor_id) != int(candidate.anchor_id)
            or str(anchor.contribution_id) != str(candidate.anchor_contribution_id)
            or not anchor.is_eligible_base_anchor
            or anchor.is_contextual
            or not anchor.independent
            or not math.isfinite(float(anchor.activation))
            or float(anchor.activation) <= 0.0
        ):
            return None

        # Prove that both original logical cues are literally the current
        # question and are represented by the same immutable float32 bytes.
        # Vector equality by numeric value is insufficient here: a -0.0 or
        # altered payload must not become a restart-safe seed by accident.
        question_text_hash = self._request_vector_hash(normalized_question)
        whole_bindings = tuple(
            item
            for item in bundle.logical_bindings
            if str(item.role) == "whole"
            and str(item.physical_id) == str(bundle.whole_physical_id)
        )
        slot_is_whole_question = (
            normalize_query_text(slot.question) == normalized_question
        )
        # A rich planner may retain useful paraphrase/constraint atomic
        # bindings for Q1 retrieval even though its one authoritative slot is
        # literally the raw question.  Exact revisit's reusable template is
        # the raw whole question, so prefer that already-observed binding here
        # rather than rejecting the otherwise valid Q1 merely because an
        # additional atomic paraphrase exists.  This does not discard the
        # subject/relation/time/negation fields on the Q1 requirement, nor
        # does it re-embed or synthesize a need for Q2.
        if slot_is_whole_question and len(whole_bindings) == 1:
            need_binding = whole_bindings[0]
            _need_binding_error = ""
        else:
            need_binding, _need_binding_error = self._v3_learning_binding_for_slot(
                bundle,
                slot.slot_id,
                allow_whole_question_binding=slot_is_whole_question,
            )
        if (
            len(whole_bindings) != 1
            or need_binding is None
            or str(whole_bindings[0].text_hash).casefold() != question_text_hash
            or str(getattr(need_binding, "text_hash", "")).casefold()
            != question_text_hash
        ):
            return None
        context_vector, context_vector_error = self._v3_learning_vector(
            bundle, str(whole_bindings[0].physical_id)
        )
        need_vector, need_vector_error = self._v3_learning_vector(
            bundle, str(getattr(need_binding, "physical_id", ""))
        )
        if (
            context_vector is None
            or need_vector is None
            or context_vector_error
            or need_vector_error
            or context_vector.tobytes(order="C") != need_vector.tobytes(order="C")
        ):
            return None

        # Re-open source closure after the ordinary draft was made.  This is a
        # cheap second current-state check, not a reconstruction from receipt
        # data; any revision between the two checks simply omits the seed.
        facts, reasons = self._v3_source_fact_closure(
            (int(candidate.anchor_id), int(candidate.target_episode_id))
        )
        anchor_fact = facts.get(int(candidate.anchor_id))
        target_fact = facts.get(int(candidate.target_episode_id))
        if (
            anchor_fact is None
            or target_fact is None
            or reasons.get(int(candidate.anchor_id)) != "source_bound"
            or reasons.get(int(candidate.target_episode_id)) != "source_bound"
            or anchor_fact not in anchor.source_facts
            or anchor_fact not in candidate.source_facts
            or target_fact not in candidate.source_facts
        ):
            return None
        source_supports = tuple(
            support
            for support in supports
            if support.is_required_coverage
            and str(support.slot_id) == str(slot.slot_id)
            and support.source_fact == target_fact
            and str(support.mapping_ref).strip()
        )
        if not source_supports:
            return None
        if {
            str(support.clause_id)
            for support in source_supports
            if str(support.clause_id)
        } != set(original_clause_ids):
            return None
        verification_refs = tuple(
            sorted(
                {
                    str(support.mapping_ref).strip()
                    for support in source_supports
                    if str(support.mapping_ref).strip()
                }
            )
        )
        if not verification_refs:
            return None

        try:
            effective_endpoint_limit = (
                int(capture.endpoint_limit)
                if capture.endpoint_limit is not None
                else int(getattr(self.contextual_matcher, "endpoint_limit", 0))
            )
            if effective_endpoint_limit <= 0:
                return None
            budget = EvidenceSelectionBudget(
                episode_limit=max(0, int(self.config.retrieval.answer_episode_limit)),
            )
            if int(budget.episode_limit) <= 0:
                return None
            source_facts = (anchor_fact, target_fact)
            context_hash = exact_revisit_context_hash(
                normalized_question,
                context_scope_hash=context_scope_hash,
            )
            source_closure_fingerprint = exact_revisit_source_closure_fingerprint(
                source_facts
            )
            retrieval_policy_fingerprint = exact_revisit_policy_fingerprint(
                self.config,
                self.contextual_matcher,
                endpoint_limit=effective_endpoint_limit,
            )
            budget_fingerprint = exact_revisit_budget_fingerprint(budget)
            anchor_manifest_fingerprint = exact_revisit_anchor_manifest_fingerprint(
                {int(candidate.anchor_id): float(anchor.activation)}
            )
            source_fact_refs_fingerprint = exact_revisit_source_fact_refs_fingerprint(
                source_facts
            )
            # The supplied ordinary draft must bind the same live Q1 state.
            # This makes the runtime form a projection, never an alternate
            # source of truth that could silently broaden a normal draft.
            if (
                str(ordinary_draft.creation_request_id)
                != str(candidate.candidate_id)
                or str(ordinary_draft.source_request_hash) != str(source_request_hash)
                or str(ordinary_draft.domain) != str(domain)
                or int(ordinary_draft.anchor_episode_id) != int(candidate.anchor_id)
                or str(ordinary_draft.context_hash) != context_hash
                or int(ordinary_draft.target_mapping.target_episode_id)
                != int(candidate.target_episode_id)
                or str(ordinary_draft.target_mapping.source_fact_id)
                != str(target_fact.fact_id)
                or str(ordinary_draft.source_closure_fingerprint)
                != source_closure_fingerprint
                or str(ordinary_draft.retrieval_policy_fingerprint)
                != retrieval_policy_fingerprint
                or str(ordinary_draft.budget_fingerprint) != budget_fingerprint
                or str(ordinary_draft.anchor_manifest_fingerprint)
                != anchor_manifest_fingerprint
                or str(ordinary_draft.source_fact_refs_fingerprint)
                != source_fact_refs_fingerprint
            ):
                return None

            runtime_ref_parts = (
                str(candidate.candidate_id),
                str(source_request_hash),
                str(context_hash),
                str(context_scope_hash),
                str(anchor_fact.fact_id),
                str(target_fact.fact_id),
            )
            runtime_slot_ref = self._v17_revisit_runtime_ref(
                "slot", *runtime_ref_parts
            )
            runtime_query_ref = self._v17_revisit_runtime_ref(
                "query", *runtime_ref_parts
            )
            runtime_clause_ref = self._v17_revisit_runtime_ref(
                "clause", *runtime_ref_parts
            )
            runtime_slot_id = "runtime-slot:" + runtime_slot_ref.rsplit(":", 1)[-1]
            runtime_query_id = "runtime-query:" + runtime_query_ref.rsplit(":", 1)[-1]
            runtime_clause_id = "runtime-clause:" + runtime_clause_ref.rsplit(":", 1)[-1]
            # QueryVectorBundle disambiguates duplicate logical query IDs.
            # Keep the whole binding visibly distinct while deriving it from
            # the same persisted query ref, so the atomic binding preserves
            # the seed's exact runtime query ID without a hidden ``:2``.
            runtime_whole_query_id = (
                "runtime-whole:" + runtime_query_ref.rsplit(":", 1)[-1]
            )
            runtime_slot = EvidenceSlot(
                slot_id=runtime_slot_id,
                question=normalized_question,
                required=True,
                query_id=runtime_query_id,
                query_refs=(runtime_query_id,),
                origin="reused_template",
                support_mode="alternative",
                clause_ids=(runtime_clause_id,),
            )
            runtime_requirements = RequirementResolution(
                request_mode="factual",
                status="resolved",
                requirements=(runtime_slot,),
                # The source template is created locally from the current
                # whole question; no planner replay or stored planner label is
                # necessary for an eligible V17 projection.
                planner_origin="reused_template",
            )
            runtime_bundle = self._new_query_embedding_coordinator().bundle_from_precomputed_vectors(
                (
                    QueryVectorRequest(
                        role="whole",
                        text=normalized_question,
                        query_id=runtime_whole_query_id,
                    ),
                    QueryVectorRequest(
                        role="atomic",
                        text=normalized_question,
                        query_id=runtime_query_id,
                        slot_id=runtime_slot_id,
                    ),
                ),
                {normalized_question: context_vector},
                embedding_space=bundle.embedding_space,
                source_request_hash=source_request_hash,
            )
            runtime_whole_bindings = tuple(
                item
                for item in runtime_bundle.logical_bindings
                if str(item.role) == "whole"
                and str(item.physical_id) == str(runtime_bundle.whole_physical_id)
            )
            runtime_need_binding, _runtime_need_error = self._v3_learning_binding_for_slot(
                runtime_bundle, runtime_slot_id
            )
            runtime_context_vector, runtime_context_error = self._v3_learning_vector(
                runtime_bundle, runtime_bundle.whole_physical_id
            )
            runtime_need_vector, runtime_need_error = self._v3_learning_vector(
                runtime_bundle,
                str(getattr(runtime_need_binding, "physical_id", "")),
            )
            try:
                # Bind the sidecar to the exact canonical bytes that the
                # outer Q1 materialization will hand to the repository, not
                # to the convenience runtime bundle below.  That bundle
                # intentionally re-normalizes precomputed vectors and may
                # differ by one float32 ULP from persisted cue bytes.
                persisted_context_blob = Database.validate_float32_vector(
                    context_vector,
                    int(bundle.dimension),
                    require_normalized=True,
                ).tobytes()
                persisted_need_blob = Database.validate_float32_vector(
                    need_vector,
                    int(bundle.dimension),
                    require_normalized=True,
                ).tobytes()
                runtime_context_blob = Database.validate_float32_vector(
                    runtime_context_vector,
                    int(runtime_bundle.dimension),
                    require_normalized=True,
                ).tobytes()
                runtime_need_blob = Database.validate_float32_vector(
                    runtime_need_vector,
                    int(runtime_bundle.dimension),
                    require_normalized=True,
                ).tobytes()
            except (TypeError, ValueError, FloatingPointError):
                return None
            if (
                str(runtime_bundle.source_request_hash) != str(source_request_hash)
                or str(runtime_bundle.model_id) != str(bundle.model_id)
                or int(runtime_bundle.dimension) != int(bundle.dimension)
                or str(runtime_bundle.embedding_space_id)
                != str(bundle.embedding_space_id)
                or len(runtime_whole_bindings) != 1
                or runtime_need_binding is None
                or str(runtime_whole_bindings[0].text_hash).casefold()
                != question_text_hash
                or str(getattr(runtime_need_binding, "text_hash", "")).casefold()
                != question_text_hash
                or runtime_context_vector is None
                or runtime_need_vector is None
                or runtime_context_error
                or runtime_need_error
                or persisted_context_blob != persisted_need_blob
                or runtime_context_blob != runtime_need_blob
            ):
                return None
            runtime_context_vector_fingerprint = self._v17_cue_vector_fingerprint(
                persisted_context_blob
            )
            runtime_need_vector_fingerprint = self._v17_cue_vector_fingerprint(
                persisted_need_blob
            )

            runtime_bindings = exact_revisit_slot_need_bindings(
                runtime_requirements, runtime_bundle
            )
            runtime_requirements_fingerprint = exact_revisit_requirements_fingerprint(
                runtime_requirements
            )
            runtime_mapping_ref = exact_revisit_aggregate_mapping_ref(
                target_episode_id=int(candidate.target_episode_id),
                slot_id=runtime_slot_id,
                clause_ids=(runtime_clause_id,),
                source_fact_id=target_fact.fact_id,
                verification_refs=verification_refs,
            )
            runtime_target_mapping = ExactRevisitTargetMappingDraft(
                target_episode_id=int(candidate.target_episode_id),
                slot_id=runtime_slot_id,
                clause_ids=(runtime_clause_id,),
                source_fact_id=target_fact.fact_id,
                mapping_ref=runtime_mapping_ref,
            )
            runtime_draft = ContextualRevisitContractDraft(
                creation_request_id=str(candidate.candidate_id),
                source_request_hash=source_request_hash,
                domain=domain,
                anchor_episode_id=int(candidate.anchor_id),
                context_hash=context_hash,
                slot_need_bindings=runtime_bindings,
                requirements_fingerprint=runtime_requirements_fingerprint,
                source_closure_fingerprint=source_closure_fingerprint,
                retrieval_policy_fingerprint=retrieval_policy_fingerprint,
                budget_fingerprint=budget_fingerprint,
                anchor_manifest_fingerprint=anchor_manifest_fingerprint,
                source_fact_refs_fingerprint=source_fact_refs_fingerprint,
                target_mapping=runtime_target_mapping,
            )
            runtime_seed = ContextualRevisitRuntimeSeed(
                creation_request_id=str(candidate.candidate_id),
                domain=domain,
                context_scope_hash=str(context_scope_hash),
                context_hash=context_hash,
                source_request_hash=source_request_hash,
                context_cue_text_hash=question_text_hash,
                need_cue_text_hash=question_text_hash,
                slot_need_bindings=runtime_bindings,
                requirements_fingerprint=runtime_requirements_fingerprint,
                source_closure_fingerprint=source_closure_fingerprint,
                retrieval_policy_fingerprint=retrieval_policy_fingerprint,
                budget_fingerprint=budget_fingerprint,
                anchor_manifest_fingerprint=anchor_manifest_fingerprint,
                source_fact_refs_fingerprint=source_fact_refs_fingerprint,
                anchor_episode_id=int(candidate.anchor_id),
                anchor_activation=float(anchor.activation),
                anchor_source_fact_id=anchor_fact.fact_id,
                target_episode_id=int(candidate.target_episode_id),
                target_source_fact_id=target_fact.fact_id,
                target_mapping_ref=runtime_mapping_ref,
                runtime_slot_ref=runtime_slot_ref,
                runtime_query_ref=runtime_query_ref,
                runtime_clause_ref=runtime_clause_ref,
                endpoint_limit=effective_endpoint_limit,
                episode_limit=int(budget.episode_limit),
                source_fact_limit=budget.source_fact_limit,
                delivery_token_limit=budget.delivery_token_limit,
            )
            # Keep the Q1 v16 compatibility draft and the V17 seed provably
            # identical at their only shared contract boundary.  The numeric
            # association ID is intentionally a harmless deterministic probe:
            # both functions attach it only while hashing the same target proof.
            if (
                runtime_seed.slot_need_bindings != runtime_bindings
                or runtime_seed.requirements_fingerprint
                != runtime_requirements_fingerprint
                or runtime_draft.slot_need_bindings != runtime_bindings
                or runtime_draft.requirements_fingerprint
                != runtime_requirements_fingerprint
                or runtime_draft.target_mapping.slot_id != runtime_seed.runtime_slot_id
                or runtime_draft.target_mapping.clause_ids
                != (runtime_seed.runtime_clause_id,)
                or runtime_draft.target_mapping.source_fact_id
                != runtime_seed.target_source_fact_id
                or runtime_draft.target_mapping.mapping_ref
                != runtime_seed.target_mapping_ref
                or runtime_draft.source_fact_roles_fingerprint(1)
                != runtime_seed.source_fact_roles_fingerprint(1)
            ):
                return None
            # This optional T16 guard is deliberately independent from the
            # V17 exact seed.  It accepts only a fully consumed restricted
            # grammar and a process-local HMAC key.  A guard-generation miss
            # must never make the already-safe exact Q2 projection unavailable.
            try:
                commitment_key = (
                    RestrictedRewriteCommitmentKey.from_environment()
                    if bool(
                        getattr(
                            self.config.retrieval,
                            "contextual_restricted_rewrite_enabled",
                            False,
                        )
                    )
                    else None
                )
                restricted_rewrite_guard = restricted_rewrite_guard_draft(
                    question=question,
                    requirements=requirements,
                    creation_request_id=str(candidate.candidate_id),
                    domain=domain,
                    context_scope_hash=str(context_scope_hash),
                    runtime_seed=runtime_seed,
                    context_cue_vector_fingerprint=(
                        runtime_context_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=runtime_need_vector_fingerprint,
                    model_id=str(bundle.model_id),
                    embedding_space_id=str(bundle.embedding_space_id),
                    dimension=int(bundle.dimension),
                    dtype="float32",
                    commitment_key=commitment_key,
                )
            except (TypeError, ValueError, UnicodeError):
                restricted_rewrite_guard = None
            return (
                runtime_draft,
                runtime_seed,
                restricted_rewrite_guard,
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            FloatingPointError,
        ):
            # This remains optional Q1 enhancement material.  A malformed or
            # no-longer-reconstructable projection leaves the normal V16 draft
            # untouched rather than making edge learning fail.
            return None

    def _v3_rerank_success_refs(
        self,
        result: Mapping[str, object],
        target_episode_id: int,
    ) -> tuple[str, ...]:
        """Record only an observed successful expensive rerank stage."""

        trace = result.get("rerank_trace")
        reranked = result.get("reranked_episode_ids")
        if not isinstance(trace, Mapping) or not isinstance(reranked, list):
            return ()
        if not bool(trace.get("enabled", False)) or trace.get("error"):
            return ()
        try:
            delivered_by_rerank = int(target_episode_id) in {
                int(value) for value in reranked
            }
        except (TypeError, ValueError):
            return ()
        trace_identity = str(trace.get("input_hash", "") or "").strip()
        if not delivered_by_rerank or not trace_identity:
            return ()
        return (
            self._v3_opaque_ref(
                "rerank-success-v1", trace_identity, int(target_episode_id)
            ),
        )

    @staticmethod
    def _v3_learning_receipts_from_callback(value: object) -> tuple[Mapping[str, object], ...]:
        """Accept the application facade's mapping while tolerating test fakes."""

        if isinstance(value, Mapping):
            value = value.get("receipts", ())
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))

    @staticmethod
    def _v3_learning_revisit_contracts_from_callback(
        value: object,
    ) -> tuple[Mapping[str, object], ...]:
        """Read only the app's redacted post-publication status records."""

        if not isinstance(value, Mapping):
            return ()
        rows = value.get("revisit_contracts", ())
        if not isinstance(rows, (list, tuple)):
            return ()
        return tuple(item for item in rows if isinstance(item, Mapping))

    @staticmethod
    def _v3_learning_revisit_projections_from_callback(
        value: object,
    ) -> tuple[Mapping[str, object], ...]:
        """Read redacted V17 projection/publication outcomes from the app."""

        if not isinstance(value, Mapping):
            return ()
        rows = value.get("revisit_projections", ())
        if not isinstance(rows, (list, tuple)):
            return ()
        return tuple(item for item in rows if isinstance(item, Mapping))

    def _finalize_v3_query_learning(
        self,
        question: str,
        result: Mapping[str, object],
        *,
        enabled: bool,
        learning_request_id: str | None,
        generate_answer: bool,
        frozen_plan: Mapping[str, object] | None,
        strict_vector_bundle: bool,
        contextual_domain: str | None,
        contextual_revisit_scope_hash: str | None,
        trace_bridge: QueryTraceBridge | None,
    ) -> dict[str, object]:
        """Create bounded source-verified candidates only after a live answer.

        This is intentionally a post-query boundary.  It receives the runtime
        selector capture, never reverses the redacted JSON trace, and passes
        all writes through the injected application callback.
        """

        if not enabled:
            return self._v3_learning_report("skipped", "explicit_opt_in_required")
        request_id = str(learning_request_id or "").strip()
        if not request_id:
            return self._v3_learning_report("skipped", "learning_request_id_required")
        if self.contextual_learning_finalizer is None:
            return self._v3_learning_report("skipped", "application_finalizer_unavailable")
        if trace_bridge is not None:
            return self._v3_learning_report("skipped", "strict_trace_read_only")
        if frozen_plan is not None or strict_vector_bundle:
            return self._v3_learning_report("skipped", "frozen_or_strict_replay")
        if not generate_answer:
            return self._v3_learning_report("skipped", "answer_generation_required")
        if not isinstance(contextual_domain, str):
            return self._v3_learning_report("skipped", "explicit_domain_required")
        domain = contextual_domain.strip()
        if not domain or "\n" in domain or "\r" in domain or "," in domain:
            return self._v3_learning_report("skipped", "explicit_domain_required")
        if (
            not self.config.retrieval.contextual_association_enabled
            or self.contextual_matcher is None
        ):
            return self._v3_learning_report("skipped", "contextual_v3_not_enabled")
        capture = self._v3_learning_capture
        if capture is None:
            return self._v3_learning_report("skipped", "v3_selector_not_observed")
        if capture.shadow:
            return self._v3_learning_report("skipped", "shadow_query")
        direct_base_mode = bool(capture.direct_base_validation_completed)
        if not capture.target_gate_completed and not direct_base_mode:
            return self._v3_learning_report(
                "skipped", "target_or_direct_base_gate_not_completed"
            )
        if capture.delivery_loss or capture.missing_required_clauses:
            return self._v3_learning_report("skipped", "selector_delivery_incomplete")
        bundle = capture.bundle
        if bundle is None:
            return self._v3_learning_report("skipped", "query_vector_bundle_missing")
        if (
            str(bundle.model_id) != str(self.config.model.embedding_model)
            or int(bundle.dimension) != int(self.config.model.embedding_dimension)
            or not str(bundle.embedding_space_id).strip()
        ):
            return self._v3_learning_report("skipped", "query_vector_bundle_incompatible")

        # An audit result is meaningful only for an answer that was actually
        # delivered by this query.  ``generate_answer=True`` is intent, not
        # proof: mocks, cancellations, or an empty model response must not
        # turn an otherwise-valid audit payload into a durable learning write.
        answer = result.get("answer")
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or result.get("answer_generation_skipped") is not False
        ):
            return self._v3_learning_report(
                "skipped", "answer_delivery_not_confirmed"
            )
        answer_terminal_state = result.get("answer_terminal_state")
        if (
            not isinstance(answer_terminal_state, Mapping)
            or answer_terminal_state.get("terminal_state") != "completed_verified"
        ):
            return self._v3_learning_report(
                "skipped", "answer_terminal_not_verified"
            )
        audits = result.get("answer_audits")
        if (
            not isinstance(audits, list)
            or not audits
            or not isinstance(audits[-1], Mapping)
            or audits[-1].get("valid") is not True
        ):
            return self._v3_learning_report("skipped", "answer_guard_failed")
        try:
            delivered_ids = tuple(
                sorted({int(value) for value in result.get("episode_ids", []) if int(value) > 0})
            )
        except (TypeError, ValueError):
            return self._v3_learning_report("skipped", "delivered_episode_ids_invalid")
        if delivered_ids != capture.final_selected_episode_ids:
            return self._v3_learning_report("skipped", "selector_delivery_mismatch")
        if not delivered_ids:
            return self._v3_learning_report("skipped", "no_delivered_evidence")

        whole_bindings = tuple(
            item
            for item in bundle.logical_bindings
            if str(item.role) == "whole"
            and str(item.physical_id) == str(bundle.whole_physical_id)
        )
        if len(whole_bindings) != 1:
            return self._v3_learning_report("skipped", "context_vector_binding_ambiguous")
        context_binding = whole_bindings[0]
        if (
            not str(context_binding.query_id).strip()
            or not str(context_binding.text_hash).strip()
            or str(context_binding.embedding_space_id) != str(bundle.embedding_space_id)
        ):
            return self._v3_learning_report("skipped", "context_vector_binding_invalid")
        context_vector, context_vector_error = self._v3_learning_vector(
            bundle, context_binding.physical_id
        )
        if context_vector is None:
            return self._v3_learning_report("skipped", context_vector_error)

        required_slots = tuple(item for item in capture.slots if item.required)
        if not required_slots:
            return self._v3_learning_report("skipped", "required_slots_missing")
        slot_bindings: dict[str, object] = {}
        slot_vectors: dict[str, np.ndarray] = {}
        for slot in required_slots:
            if slot.slot_id in slot_bindings:
                return self._v3_learning_report("skipped", "slot_id_ambiguous")
            binding, binding_error = self._v3_learning_binding_for_slot(
                bundle,
                slot.slot_id,
                allow_whole_question_binding=(
                    len(required_slots) == 1
                    and normalize_query_text(slot.question)
                    == normalize_query_text(question)
                ),
            )
            if binding is None:
                return self._v3_learning_report("skipped", binding_error)
            need_vector, vector_error = self._v3_learning_vector(
                bundle, str(getattr(binding, "physical_id", ""))
            )
            if need_vector is None:
                return self._v3_learning_report("skipped", vector_error)
            slot_bindings[slot.slot_id] = binding
            slot_vectors[slot.slot_id] = need_vector

        source_request_hash = "sha256:" + self._request_vector_hash(question)
        if str(bundle.source_request_hash) != source_request_hash:
            return self._v3_learning_report("skipped", "bundle_request_hash_mismatch")
        learning_request_hash = self._v3_opaque_ref(
            "t13b-query-learning-request-v1",
            request_id,
            source_request_hash,
            domain,
            bundle.embedding_space_id,
            bundle.model_id,
            bundle.dimension,
            context_binding.query_id,
            context_binding.physical_id,
            *(
                f"{slot_id}:{getattr(binding, 'query_id', '')}:{getattr(binding, 'physical_id', '')}"
                for slot_id, binding in sorted(slot_bindings.items())
            ),
        )

        contextual_ids = set(capture.contextual_expansion_episode_ids)
        cue_endpoint_ids = set(capture.cue_endpoint_episode_ids)
        direct_base_target_ids = set(
            capture.initial_candidate_episode_ids
        ).intersection(capture.independent_base_episode_ids)
        delivered_set = set(delivered_ids)
        target_options: list[tuple[EvidenceSlot, CandidateContribution, tuple[ClauseSupport, ...]]] = []
        seen_targets: set[tuple[str, int]] = set()
        for slot in required_slots:
            for contribution in sorted(
                capture.contributions,
                key=lambda item: (item.slot_id, item.episode_id, item.contribution_id),
            ):
                key = (slot.slot_id, int(contribution.episode_id))
                if key in seen_targets:
                    continue
                if (
                    contribution.lane != "base_source_mapping"
                    or contribution.slot_id != slot.slot_id
                    or contribution.episode_id not in delivered_set
                    or contribution.episode_id in contextual_ids
                    or contribution.episode_id in cue_endpoint_ids
                    or (
                        direct_base_mode
                        and contribution.episode_id not in direct_base_target_ids
                    )
                ):
                    continue
                supports = self._v3_learning_source_supports(contribution, slot)
                if not supports:
                    continue
                seen_targets.add(key)
                target_options.append((slot, contribution, supports))

        rejected_count = 0
        deferred_candidate_options = 0
        selected_event: RecallLearningEvent | None = None
        selected_plan: LearningCandidatePlan | None = None
        selected_materializations: dict[str, dict[str, object]] | None = None

        # ``RecallLearningEvent`` has one target/slot contract.  It would be
        # unsafe to run several such events and describe the whole request as
        # one atomic creation: a later failure could leave an earlier target
        # durable.  Until a future grouped-event contract can prove a single
        # transaction across slots, choose at most one admissible plan.  That
        # one plan can still contain up to four independently selected anchors
        # and is finalized by the repository in one transaction.
        for option_index, (slot, target_contribution, supports) in enumerate(
            target_options
        ):
            target_id = int(target_contribution.episode_id)
            anchors = self._v3_learning_anchors(
                capture, target_episode_id=target_id
            )
            if not anchors:
                rejected_count += 1
                continue
            target_facts = tuple(
                dict.fromkeys(
                    support.source_fact
                    for support in supports
                    if support.source_fact is not None
                )
            )
            verification_refs = tuple(
                dict.fromkeys(
                    str(support.mapping_ref)
                    for support in supports
                    if str(support.mapping_ref).strip()
                )
            )
            target_provenance_refs = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        target_contribution.contribution_id,
                        target_contribution.evidence_ref,
                        *verification_refs,
                    )
                    if str(value).strip()
                )
            )
            if not target_facts or not verification_refs or not target_provenance_refs:
                rejected_count += 1
                continue
            costly_refs = self._v3_rerank_success_refs(result, target_id)
            try:
                event = RecallLearningEvent(
                    request_id=request_id,
                    request_hash=learning_request_hash,
                    domain=domain,
                    target_episode_ids=(target_id,),
                    initial_candidate_episode_ids=capture.initial_candidate_episode_ids,
                    initial_delivered_episode_ids=capture.initial_delivered_episode_ids,
                    final_selected_episode_ids=capture.final_selected_episode_ids,
                    final_delivered_episode_ids=delivered_ids,
                    contextual_expansion_episode_ids=capture.contextual_expansion_episode_ids,
                    source_request_hash=source_request_hash,
                    slot_id=slot.slot_id,
                    context_query_id=str(context_binding.query_id),
                    need_query_id=str(getattr(slot_bindings[slot.slot_id], "query_id", "")),
                    context_vector_ref=str(context_binding.physical_id),
                    need_vector_ref=str(getattr(slot_bindings[slot.slot_id], "physical_id", "")),
                    source_facts=target_facts,
                    verification_refs=verification_refs,
                    verification_status="source_bound",
                    target_provenance_refs=target_provenance_refs,
                    costly_stage_refs=costly_refs,
                    answer_guard_passed=True,
                )
                plan = plan_learning_candidates(
                    event,
                    anchors,
                    max_candidates=4,
                    max_anchors=4,
                )
            except (TypeError, ValueError):
                rejected_count += 1
                continue
            if not plan.candidates:
                rejected_count += 1
                continue
            # Contract drafting is deliberately best-effort and entirely
            # request-local.  A missing draft must not make the already-safe
            # Q1 edge plan fail; it simply means this receipt cannot advertise
            # an exact revisit input it cannot prove.
            revisit_contract_drafts: dict[str, ContextualRevisitContractDraft] = {}
            revisit_runtime_seeds: dict[str, ContextualRevisitRuntimeSeed] = {}
            revisit_projection_markers: dict[str, str] = {}
            revisit_projection_reasons: dict[str, str] = {}
            revisit_restricted_rewrite_guards: dict[
                str, ContextualRestrictedRewriteGuardDraft
            ] = {}
            for candidate in plan.candidates:
                ordinary_contract_diagnostic: dict[str, str] = {}
                ordinary_draft = self._v3_revisit_contract_draft(
                    question=question,
                    capture=capture,
                    candidate=candidate,
                    plan=plan,
                    target_contribution=target_contribution,
                    supports=supports,
                    bundle=bundle,
                    domain=domain,
                    source_request_hash=source_request_hash,
                    context_scope_hash=contextual_revisit_scope_hash,
                    projection_diagnostic=ordinary_contract_diagnostic,
                )
                runtime_projection = self._v17_revisit_runtime_projection(
                    question=question,
                    capture=capture,
                    candidate=candidate,
                    plan=plan,
                    target_contribution=target_contribution,
                    supports=supports,
                    bundle=bundle,
                    domain=domain,
                    source_request_hash=source_request_hash,
                    context_scope_hash=contextual_revisit_scope_hash,
                    ordinary_draft=ordinary_draft,
                )
                if runtime_projection is not None:
                    (
                        runtime_draft,
                        runtime_seed,
                        restricted_rewrite_guard,
                    ) = runtime_projection
                    # The runtime-ID V16 draft is intentionally paired with
                    # its V17 seed.  The repository can then create/promote
                    # its one canonical contract without translating planner
                    # labels after the Q1 request has ended.
                    revisit_contract_drafts[candidate.candidate_id] = runtime_draft
                    revisit_runtime_seeds[candidate.candidate_id] = runtime_seed
                    revisit_projection_markers[candidate.candidate_id] = "projected"
                    revisit_projection_reasons[candidate.candidate_id] = (
                        "runtime_seed_prepared"
                    )
                    if restricted_rewrite_guard is not None:
                        revisit_restricted_rewrite_guards[candidate.candidate_id] = (
                            restricted_rewrite_guard
                        )
                elif ordinary_draft is not None:
                    # Most normal Q1 candidates are not safely reducible to a
                    # whole-question runtime template.  Preserve their
                    # existing V16 compatibility behavior with no V17 seed.
                    revisit_contract_drafts[candidate.candidate_id] = ordinary_draft
                    revisit_projection_markers[candidate.candidate_id] = (
                        "rejected_not_reconstructable"
                    )
                    revisit_projection_reasons[candidate.candidate_id] = (
                        "ordinary_contract_"
                        + ordinary_contract_diagnostic.get(
                            "ordinary_contract_reason", "runtime_projection_unavailable"
                        )
                    )
                else:
                    # Keep the optional exact-revisit outcome observable even
                    # when ordinary V16 drafting also cannot prove a contract.
                    revisit_projection_markers[candidate.candidate_id] = (
                        "rejected_not_reconstructable"
                    )
                    revisit_projection_reasons[candidate.candidate_id] = (
                        "ordinary_contract_"
                        + ordinary_contract_diagnostic.get(
                            "ordinary_contract_reason", "runtime_projection_unavailable"
                        )
                    )
            selected_event = event
            selected_plan = plan
            selected_materializations = {
                candidate.candidate_id: {
                    "domain": domain,
                    "model_id": str(bundle.model_id),
                    "dimension": int(bundle.dimension),
                    "context_vector": context_vector,
                    "need_vector": slot_vectors[slot.slot_id],
                    "context_text_hash": str(context_binding.text_hash),
                    "need_text_hash": str(
                        getattr(slot_bindings[slot.slot_id], "text_hash", "")
                    ),
                    "embedding_space_id": str(bundle.embedding_space_id),
                    "embedding_space": bundle.embedding_space,
                    # Receipt creation never persists request/source text as a
                    # convenience label.  Source facts and hashes are enough.
                    "context_display_text": "",
                    "need_display_text": "",
                    # This deliberately says only whether V17 projection was
                    # possible from the live Q1 state. The application turns
                    # a projected seed into a distinct ready/deferred/failed
                    # publication result after canonical receipt checks.
                    "revisit_projection": revisit_projection_markers.get(
                        candidate.candidate_id,
                        "rejected_not_reconstructable",
                    ),
                    "revisit_projection_reason": revisit_projection_reasons.get(
                        candidate.candidate_id,
                        "runtime_projection_unavailable",
                    ),
                    # This typed object is process-local and redacted.  The
                    # application may use it only after a canonical receipt
                    # becomes ready; it cannot reconstruct a missing draft.
                    **(
                        {"revisit_contract_draft": revisit_contract_drafts[candidate.candidate_id]}
                        if candidate.candidate_id in revisit_contract_drafts
                        else {}
                    ),
                    **(
                        {"revisit_runtime_seed": revisit_runtime_seeds[candidate.candidate_id]}
                        if candidate.candidate_id in revisit_runtime_seeds
                        else {}
                    ),
                    **(
                        {"restricted_rewrite_guard": revisit_restricted_rewrite_guards[candidate.candidate_id]}
                        if candidate.candidate_id in revisit_restricted_rewrite_guards
                        else {}
                    ),
                }
                for candidate in plan.candidates
            }
            deferred_candidate_options = len(target_options) - option_index - 1
            break

        if (
            selected_event is None
            or selected_plan is None
            or selected_materializations is None
        ):
            return self._v3_learning_report(
                "skipped",
                "no_eligible_source_bound_candidate",
                rejected_count=rejected_count,
            )
        planned_ids = [candidate.candidate_id for candidate in selected_plan.candidates]
        try:
            callback_result = self.contextual_learning_finalizer(
                selected_event, selected_plan, selected_materializations
            )
        except Exception:
            # Repository finalization is transactional.  An application
            # callback failure therefore cannot be presented as ready.
            return self._v3_learning_report(
                "failed",
                "application_finalizer_failed",
                candidate_ids=planned_ids,
                rejected_count=rejected_count,
                deferred_candidate_options=deferred_candidate_options,
            )
        receipt_rows = self._v3_learning_receipts_from_callback(callback_result)
        revisit_contract_rows = self._v3_learning_revisit_contracts_from_callback(
            callback_result
        )
        revisit_projection_rows = self._v3_learning_revisit_projections_from_callback(
            callback_result
        )
        received_ids = {
            str(item.get("creation_request_id", "") or "")
            for item in receipt_rows
        }
        if set(planned_ids) != received_ids:
            return self._v3_learning_report(
                "failed",
                "application_finalizer_receipt_mismatch",
                candidate_ids=planned_ids,
                receipts=receipt_rows,
                rejected_count=rejected_count,
                deferred_candidate_options=deferred_candidate_options,
            )
        receipt_statuses = {str(item.get("status", "") or "") for item in receipt_rows}
        status = (
            "ready"
            if receipt_statuses == {"ready"}
            else "committed_pending_index"
            if receipt_statuses and receipt_statuses.issubset({"committed_pending_index", "ready"})
            else "completed_with_nonready_receipts"
        )
        return self._v3_learning_report(
            status,
            "",
            candidate_ids=planned_ids,
            receipts=receipt_rows,
            revisit_contracts=revisit_contract_rows,
            revisit_projections=revisit_projection_rows,
            rejected_count=rejected_count,
            deferred_candidate_options=deferred_candidate_options,
        )

    @staticmethod
    def _v17_cue_vector_fingerprint(blob: object) -> str:
        """Return the durable-safe digest for one raw persisted cue blob."""

        raw = bytes(blob)
        if not raw:
            raise ValueError("automatic revisit cue vector is empty")
        return "cue-vector:sha256:" + hashlib.sha256(raw).hexdigest()

    def _v17_automatic_exact_revisit(
        self,
        question: str,
        *,
        intent_override: QueryIntent | dict | None,
        followup_queries_override: list[str] | None,
        frozen_plan: dict | None,
        query_embeddings_override: dict[str, np.ndarray] | None,
        query_vector_bundle: QueryVectorBundle | None,
        strict_vector_bundle: bool,
        contextual_domain: str | None,
        contextual_endpoint_limit: int | None,
        contextual_evaluation_as_of: str | None,
        contextual_revisit_scope_hash: str | None,
        trace_bridge: QueryTraceBridge | None,
        deadline_at: float | None,
    ) -> _V17AutomaticExactRevisit | None:
        """Reconstruct the narrow V17 Q2 input without a provider call.

        A V17 row is only a redacted durable *seed*.  This method reads its
        current cue blobs and creates the same one-slot, one-vector runtime
        shape that Q1 projected.  It deliberately does not return an object
        to the public API and it does not itself decide evidence: the normal
        T15 exact lane must still re-run matcher, gate, source closure and
        selector before it can produce a result.
        """

        def ensure_deadline(stage: str) -> None:
            if deadline_at is not None and monotonic() >= deadline_at:
                raise TimeoutError(
                    f"query deadline exceeded before automatic_revisit_{stage}"
                )

        ensure_deadline("preflight")
        # Automatic reuse is intentionally stricter than a caller supplied
        # transient ticket.  Any replay/debug/override surface could change
        # the request semantics without a new V17 projection, so it stays on
        # the ordinary retrieval path.
        if (
            trace_bridge is not None
            or frozen_plan is not None
            or strict_vector_bundle
            or intent_override is not None
            or followup_queries_override is not None
            or query_embeddings_override is not None
            or query_vector_bundle is not None
            or contextual_endpoint_limit is not None
            or contextual_evaluation_as_of is not None
            or not self.config.retrieval.contextual_association_enabled
            or self.config.retrieval.contextual_association_shadow
            or self.contextual_matcher is None
            or not isinstance(contextual_domain, str)
            or contextual_revisit_scope_hash is None
        ):
            return None

        normalized_question = normalize_query_text(question)
        domain = str(contextual_domain).strip()
        if (
            not normalized_question
            or not domain
            or "\n" in domain
            or "\r" in domain
            or "," in domain
        ):
            return None
        repository = self._exact_revisit_contract_repository()
        if repository is None:
            return None

        try:
            source_request_hash = exact_revisit_request_hash(normalized_question)
            context_hash = exact_revisit_context_hash(
                normalized_question,
                context_scope_hash=contextual_revisit_scope_hash,
            )
            coordinator = self._new_query_embedding_coordinator()
            current_space = coordinator.embedding_space()
            lookup = ContextualRevisitRuntimeManifestLookup(
                domain=domain,
                context_scope_hash=contextual_revisit_scope_hash,
                context_hash=context_hash,
                source_request_hash=source_request_hash,
                model_id=str(self.config.model.embedding_model),
                embedding_space_id=str(current_space.canonical_id),
                dimension=int(self.config.model.embedding_dimension),
            )
            manifest = repository.find_contextual_revisit_runtime_manifest(lookup)
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            return None

        ensure_deadline("manifest_lookup")

        try:
            if (
                not isinstance(manifest, ContextualRevisitRuntimeManifest)
                or manifest.state != "ready"
                or int(manifest.manifest_id) <= 0
                or not str(manifest.manifest_fingerprint).strip()
            ):
                return None
            seed = manifest.seed
            if (
                str(seed.domain) != domain
                or str(seed.context_scope_hash) != str(contextual_revisit_scope_hash)
                or str(seed.context_hash) != context_hash
                or str(seed.source_request_hash) != source_request_hash
                or str(manifest.model_id) != str(self.config.model.embedding_model)
                or int(manifest.dimension) != int(self.config.model.embedding_dimension)
                or str(manifest.dtype) != "float32"
                or str(manifest.embedding_space_id) != str(current_space.canonical_id)
                or int(manifest.context_cue_id) == int(manifest.need_cue_id)
                or int(manifest.contract_id or 0) <= 0
                or not str(manifest.contract_fingerprint).strip()
                or manifest.to_revisit_contract().contract_fingerprint
                != str(manifest.contract_fingerprint)
            ):
                return None

            # The whole question is the only textual material V17 may rebuild.
            # Both cue text hashes must be exactly that normalized question; no
            # planner phrase or source text is ever read back from storage.
            question_text_hash = self._request_vector_hash(normalized_question)
            if (
                str(seed.context_cue_text_hash) != question_text_hash
                or str(seed.need_cue_text_hash) != question_text_hash
                or str(seed.support_mode) != "alternative"
                or int(seed.episode_limit)
                != int(self.config.retrieval.answer_episode_limit)
            ):
                return None
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
        ):
            return None

        ensure_deadline("cue_lookup")
        try:
            cue_rows = {
                int(row["id"]): row
                for row in repository.load_contextual_cue_pair(
                    context_cue_id=int(manifest.context_cue_id),
                    need_cue_id=int(manifest.need_cue_id),
                    domain=domain,
                )
            }
            if set(cue_rows) != {
                int(manifest.context_cue_id),
                int(manifest.need_cue_id),
            }:
                return None
            ensure_deadline("cue_read")
            context_row = cue_rows[int(manifest.context_cue_id)]
            need_row = cue_rows[int(manifest.need_cue_id)]

            def valid_cue(row, *, cue_kind: str, text_hash: str) -> bool:
                return bool(
                    str(row["domain"] or "") == domain
                    and str(row["cue_kind"] or "") == cue_kind
                    and str(row["model_id"] or "")
                    == str(manifest.model_id)
                    and int(row["dimension"] or 0) == int(manifest.dimension)
                    and str(row["dtype"] or "") == "float32"
                    and str(row["embedding_space_id"] or "")
                    == str(manifest.embedding_space_id)
                    and str(row["text_hash"] or "") == text_hash
                )

            if not valid_cue(
                context_row,
                cue_kind="context",
                text_hash=seed.context_cue_text_hash,
            ) or not valid_cue(
                need_row,
                cue_kind="need",
                text_hash=seed.need_cue_text_hash,
            ):
                return None
            context_blob = bytes(context_row["vector_blob"])
            need_blob = bytes(need_row["vector_blob"])
            if (
                self._v17_cue_vector_fingerprint(context_blob)
                != str(manifest.context_cue_vector_fingerprint)
                or self._v17_cue_vector_fingerprint(need_blob)
                != str(manifest.need_cue_vector_fingerprint)
                # Q1 qualifies a runtime seed only when these are literally
                # the same serialized float32 vector, not merely close.
                or context_blob != need_blob
            ):
                return None
            ensure_deadline("cue_validation")
            context_vector = decode_embedding(
                context_blob, int(manifest.dimension), renormalize=False
            )
            need_vector = decode_embedding(
                need_blob, int(manifest.dimension), renormalize=False
            )
            if (
                context_vector.dtype != np.float32
                or need_vector.dtype != np.float32
                or context_vector.shape != (int(manifest.dimension),)
                or need_vector.shape != (int(manifest.dimension),)
                or not np.all(np.isfinite(context_vector))
                or not np.all(np.isfinite(need_vector))
                or not math.isclose(
                    float(np.linalg.norm(context_vector)),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(np.linalg.norm(need_vector)),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                )
                or not np.array_equal(context_vector, need_vector)
            ):
                return None
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            return None

        try:
            # Preserve the raw persisted float32 bytes after validating them.
            # The normal coordinator's precomputed-vector convenience method
            # re-normalizes, which can change a valid float32 payload by one
            # ULP.  Here Q1's byte-equality condition is part of the V17
            # safety contract, so build the single physical bundle directly.
            context_vector.setflags(write=False)
            runtime_digest = str(seed.runtime_query_ref).rsplit(":", 1)[-1]
            whole_query_id = "runtime-whole:" + runtime_digest
            physical_id = "query-vector:sha256:" + hashlib.sha256(
                (
                    str(current_space.canonical_id)
                    + "\0"
                    + question_text_hash
                ).encode("utf-8")
            ).hexdigest()
            runtime_bundle = QueryVectorBundle(
                model_id=str(self.config.model.embedding_model),
                dimension=int(manifest.dimension),
                whole=context_vector,
                queries=(
                    QueryVector(
                        query_id=whole_query_id,
                        text_hash=question_text_hash,
                        role="whole",
                        vector=context_vector,
                        text=normalized_question,
                        physical_id=physical_id,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                    QueryVector(
                        query_id=seed.runtime_query_id,
                        text_hash=question_text_hash,
                        role="atomic",
                        vector=context_vector,
                        slot_id=seed.runtime_slot_id,
                        text=normalized_question,
                        physical_id=physical_id,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                ),
                embedding_space=current_space,
                physical_vectors=(
                    PhysicalQueryVector(
                        physical_id=physical_id,
                        text_hash=question_text_hash,
                        vector=context_vector,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                ),
                source_request_hash=source_request_hash,
                schema_version=2,
                whole_physical_id=physical_id,
            )
            runtime_slot = EvidenceSlot(
                slot_id=seed.runtime_slot_id,
                question=normalized_question,
                required=True,
                query_id=seed.runtime_query_id,
                query_refs=(seed.runtime_query_id,),
                origin="reused_template",
                support_mode="alternative",
                clause_ids=(seed.runtime_clause_id,),
            )
            requirements = RequirementResolution(
                request_mode="factual",
                status="resolved",
                requirements=(runtime_slot,),
                planner_origin="reused_template",
            )
            intent = QueryIntent(search_queries=[normalized_question])
            if infer_request_mode(normalized_question, intent) != "factual":
                return None
            budget = EvidenceSelectionBudget(
                episode_limit=int(seed.episode_limit),
                source_fact_limit=seed.source_fact_limit,
                delivery_token_limit=seed.delivery_token_limit,
            )
            proof = ExactRevisitSourceMappingProof(
                association_id=int(manifest.association_id),
                target_episode_id=int(seed.target_episode_id),
                slot_id=seed.runtime_slot_id,
                clause_ids=(seed.runtime_clause_id,),
                source_fact_id=seed.target_source_fact_id,
                mapping_ref=seed.target_mapping_ref,
            )
            revisit_input = ExactRevisitInput(
                domain=domain,
                context_hash=context_hash,
                requirements=requirements,
                intent=intent,
                query_vector_bundle=runtime_bundle,
                budget=budget,
                retrieval_policy_fingerprint=seed.retrieval_policy_fingerprint,
                source_closure_fingerprint=seed.source_closure_fingerprint,
                source_fact_roles_fingerprint=manifest.source_fact_roles_fingerprint,
                source_fact_refs_fingerprint=seed.source_fact_refs_fingerprint,
                evaluation_as_of=self._resolve_contextual_evaluation_as_of(None),
                endpoint_limit=int(seed.endpoint_limit),
                anchor_activations={
                    int(seed.anchor_episode_id): float(seed.anchor_activation)
                },
                base_episode_ids=(int(seed.anchor_episode_id),),
                base_slot_support={},
                target_source_mapping_proofs=(proof,),
            )
            if (
                int(runtime_bundle.physical_count) != 1
                or int(runtime_bundle.logical_count) != 2
                or str(runtime_bundle.source_request_hash) != source_request_hash
                or str(runtime_bundle.embedding_space_id)
                != str(manifest.embedding_space_id)
                or not np.array_equal(
                    runtime_bundle.vector_for_physical(physical_id), context_vector
                )
                or revisit_input.slot_need_bindings != seed.slot_need_bindings
                or revisit_input.requirements_fingerprint
                != seed.requirements_fingerprint
                or revisit_input.budget_fingerprint != seed.budget_fingerprint
                or revisit_input.anchor_manifest_fingerprint
                != seed.anchor_manifest_fingerprint
                or revisit_input.retrieval_policy_fingerprint
                != exact_revisit_policy_fingerprint(
                    self.config,
                    self.contextual_matcher,
                    endpoint_limit=int(seed.endpoint_limit),
                )
            ):
                return None
            ensure_deadline("runtime_reconstruction")
            return _V17AutomaticExactRevisit(
                revisit_input=revisit_input,
                manifest_id=int(manifest.manifest_id),
                manifest_fingerprint=str(manifest.manifest_fingerprint),
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            FloatingPointError,
            RuntimeError,
        ):
            # An automatic V17 attempt is a cache-like optimization.  Any
            # uncertainty leaves the ordinary path authoritative.
            return None

    def _t16_automatic_restricted_rewrite_revisit(
        self,
        question: str,
        *,
        intent_override: QueryIntent | dict | None,
        followup_queries_override: list[str] | None,
        frozen_plan: dict | None,
        query_embeddings_override: dict[str, np.ndarray] | None,
        query_vector_bundle: QueryVectorBundle | None,
        strict_vector_bundle: bool,
        contextual_domain: str | None,
        contextual_endpoint_limit: int | None,
        contextual_evaluation_as_of: str | None,
        contextual_revisit_scope_hash: str | None,
        trace_bridge: QueryTraceBridge | None,
        deadline_at: float | None,
    ) -> _T16RestrictedRewriteRevisit | None:
        """Rebuild a local T15 input from one fully parsed HMAC rewrite.

        This is deliberately narrower than V17 exact revisit.  It accepts two
        controlled English surface forms only, makes no new embedding,
        planner, or model call, and falls through on every unknown
        token/grammar/key/sidecar condition.  Its local matcher may use the
        old cue vector only through the private HMAC-authenticated capability
        constructed below; it is never presented as an ordinary embedding for
        arbitrary current text.
        """

        def ensure_deadline(stage: str) -> None:
            if deadline_at is not None and monotonic() >= deadline_at:
                raise TimeoutError(
                    f"query deadline exceeded before restricted_rewrite_{stage}"
                )

        ensure_deadline("preflight")
        if (
            trace_bridge is not None
            or frozen_plan is not None
            or strict_vector_bundle
            or intent_override is not None
            or followup_queries_override is not None
            or query_embeddings_override is not None
            or query_vector_bundle is not None
            or contextual_endpoint_limit is not None
            or contextual_evaluation_as_of is not None
            or not bool(
                getattr(
                    self.config.retrieval,
                    "contextual_restricted_rewrite_enabled",
                    False,
                )
            )
            or not self.config.retrieval.contextual_association_enabled
            or self.config.retrieval.contextual_association_shadow
            or self.contextual_matcher is None
            or not isinstance(contextual_domain, str)
            or contextual_revisit_scope_hash is None
        ):
            return None

        normalized_question = normalize_query_text(question)
        domain = str(contextual_domain).strip()
        if (
            not normalized_question
            or not domain
            or "\n" in domain
            or "\r" in domain
            or "," in domain
        ):
            return None
        key = RestrictedRewriteCommitmentKey.from_environment()
        ir = parse_restricted_rewrite_question(normalized_question)
        if key is None or ir is None:
            return None
        try:
            rewrite_commitment = key.commitment(
                context_scope_hash=contextual_revisit_scope_hash,
                ir=ir,
            )
            repository = self._exact_revisit_contract_repository()
            if repository is None:
                return None
            guard_lookup = ContextualRestrictedRewriteGuardLookup(
                association_id=None,
                domain=domain,
                context_scope_hash=contextual_revisit_scope_hash,
                rewrite_commitment=rewrite_commitment,
                commitment_key_id=key.key_id,
                grammar_version=ir.grammar_version,
            )
            restored = repository.find_contextual_restricted_rewrite_guard(
                guard_lookup
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            return None
        ensure_deadline("guard_lookup")
        if restored is None:
            return None
        guard, manifest = restored

        try:
            coordinator = self._new_query_embedding_coordinator()
            current_space = coordinator.embedding_space()
            if (
                not isinstance(guard, ContextualRestrictedRewriteGuard)
                or not isinstance(manifest, ContextualRevisitRuntimeManifest)
                or manifest.state != "ready"
                or int(manifest.manifest_id) <= 0
                or not str(manifest.manifest_fingerprint).strip()
                or str(guard.rewrite_commitment) != rewrite_commitment
                or str(guard.binding_commitment or "") == ""
                or str(guard.manifest_binding_commitment or "") == ""
                or str(guard.ready_manifest_commitment or "") == ""
                or not str(manifest.binding_fingerprint or "")
                or str(guard.context_cue_vector_fingerprint)
                != str(manifest.context_cue_vector_fingerprint)
                or str(guard.need_cue_vector_fingerprint)
                != str(manifest.need_cue_vector_fingerprint)
                or str(guard.model_id) != str(manifest.model_id)
                or str(guard.embedding_space_id)
                != str(manifest.embedding_space_id)
                or int(guard.dimension) != int(manifest.dimension)
                or str(guard.dtype) != str(manifest.dtype)
                or str(guard.commitment_key_id) != key.key_id
                or str(guard.grammar_version) != ir.grammar_version
                or int(guard.creation_receipt_id)
                != int(manifest.creation_receipt_id)
                or int(guard.association_id) != int(manifest.association_id)
                or str(guard.seed_fingerprint) != str(manifest.seed.seed_fingerprint)
                or str(guard.domain) != domain
                or str(guard.context_scope_hash)
                != str(contextual_revisit_scope_hash)
                or str(manifest.model_id) != str(self.config.model.embedding_model)
                or int(manifest.dimension) != int(self.config.model.embedding_dimension)
                or str(manifest.dtype) != "float32"
                or str(manifest.embedding_space_id) != str(current_space.canonical_id)
                or int(manifest.context_cue_id) == int(manifest.need_cue_id)
                or int(manifest.contract_id or 0) <= 0
                or not str(manifest.contract_fingerprint).strip()
                or manifest.to_revisit_contract().contract_fingerprint
                != str(manifest.contract_fingerprint)
            ):
                return None
            seed = manifest.seed
            if (
                str(seed.domain) != domain
                or str(seed.context_scope_hash)
                != str(contextual_revisit_scope_hash)
                or str(seed.support_mode) != "alternative"
                or int(seed.episode_limit)
                != int(self.config.retrieval.answer_episode_limit)
            ):
                return None
            expected_binding = key.binding_commitment(
                context_scope_hash=contextual_revisit_scope_hash,
                rewrite_commitment=rewrite_commitment,
                creation_request_id=seed.creation_request_id,
                domain=domain,
                seed_fingerprint=seed.seed_fingerprint,
                context_cue_vector_fingerprint=(
                    manifest.context_cue_vector_fingerprint
                ),
                need_cue_vector_fingerprint=(
                    manifest.need_cue_vector_fingerprint
                ),
                model_id=manifest.model_id,
                embedding_space_id=manifest.embedding_space_id,
                dimension=manifest.dimension,
                dtype=manifest.dtype,
                grammar_version=ir.grammar_version,
                signature_version=guard.signature_version,
            )
            if not hmac.compare_digest(
                str(guard.binding_commitment), str(expected_binding)
            ):
                return None
            expected_manifest_binding = key.manifest_binding_commitment(
                context_scope_hash=contextual_revisit_scope_hash,
                rewrite_commitment=rewrite_commitment,
                creation_request_id=seed.creation_request_id,
                domain=domain,
                seed_fingerprint=seed.seed_fingerprint,
                context_cue_vector_fingerprint=(
                    manifest.context_cue_vector_fingerprint
                ),
                need_cue_vector_fingerprint=(
                    manifest.need_cue_vector_fingerprint
                ),
                model_id=manifest.model_id,
                embedding_space_id=manifest.embedding_space_id,
                dimension=manifest.dimension,
                dtype=manifest.dtype,
                grammar_version=ir.grammar_version,
                signature_version=guard.signature_version,
                manifest_binding_fingerprint=manifest.binding_fingerprint,
            )
            if not hmac.compare_digest(
                str(guard.manifest_binding_commitment),
                str(expected_manifest_binding),
            ):
                return None
            expected_ready_manifest_commitment = key.ready_manifest_commitment(
                context_scope_hash=contextual_revisit_scope_hash,
                binding_commitment=guard.binding_commitment,
                manifest_binding_commitment=guard.manifest_binding_commitment,
                manifest_fingerprint=manifest.manifest_fingerprint,
                signature_version=guard.signature_version,
            )
            if not hmac.compare_digest(
                str(guard.ready_manifest_commitment),
                str(expected_ready_manifest_commitment),
            ):
                return None
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
        ):
            return None

        ensure_deadline("cue_lookup")
        try:
            cue_rows = {
                int(row["id"]): row
                for row in repository.load_contextual_cue_pair(
                    context_cue_id=int(manifest.context_cue_id),
                    need_cue_id=int(manifest.need_cue_id),
                    domain=domain,
                )
            }
            if set(cue_rows) != {
                int(manifest.context_cue_id),
                int(manifest.need_cue_id),
            }:
                return None
            context_row = cue_rows[int(manifest.context_cue_id)]
            need_row = cue_rows[int(manifest.need_cue_id)]

            def valid_cue(row, *, cue_kind: str, text_hash: str) -> bool:
                return bool(
                    str(row["domain"] or "") == domain
                    and str(row["cue_kind"] or "") == cue_kind
                    and str(row["model_id"] or "") == str(manifest.model_id)
                    and int(row["dimension"] or 0) == int(manifest.dimension)
                    and str(row["dtype"] or "") == "float32"
                    and str(row["embedding_space_id"] or "")
                    == str(manifest.embedding_space_id)
                    and str(row["text_hash"] or "") == text_hash
                )

            if not valid_cue(
                context_row,
                cue_kind="context",
                text_hash=seed.context_cue_text_hash,
            ) or not valid_cue(
                need_row,
                cue_kind="need",
                text_hash=seed.need_cue_text_hash,
            ):
                return None
            context_blob = bytes(context_row["vector_blob"])
            need_blob = bytes(need_row["vector_blob"])
            if (
                self._v17_cue_vector_fingerprint(context_blob)
                != str(manifest.context_cue_vector_fingerprint)
                or self._v17_cue_vector_fingerprint(context_blob)
                != str(guard.context_cue_vector_fingerprint)
                or self._v17_cue_vector_fingerprint(need_blob)
                != str(manifest.need_cue_vector_fingerprint)
                or self._v17_cue_vector_fingerprint(need_blob)
                != str(guard.need_cue_vector_fingerprint)
                or context_blob != need_blob
            ):
                return None
            context_vector = decode_embedding(
                context_blob, int(manifest.dimension), renormalize=False
            )
            need_vector = decode_embedding(
                need_blob, int(manifest.dimension), renormalize=False
            )
            if (
                context_vector.dtype != np.float32
                or need_vector.dtype != np.float32
                or context_vector.shape != (int(manifest.dimension),)
                or need_vector.shape != (int(manifest.dimension),)
                or not np.all(np.isfinite(context_vector))
                or not np.all(np.isfinite(need_vector))
                or not math.isclose(
                    float(np.linalg.norm(context_vector)),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(np.linalg.norm(need_vector)),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                )
                or not np.array_equal(context_vector, need_vector)
            ):
                return None
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            return None

        try:
            # The vector has been authenticated against the *origin* cue blob
            # and may only be associated with current text inside this private
            # HMAC capability.  No public bundle/override path can construct
            # this pairing, and `_try_exact_revisit_v3` rechecks the capability
            # before both of its full local passes.
            context_vector.setflags(write=False)
            question_text_hash = self._request_vector_hash(normalized_question)
            source_request_hash = exact_revisit_request_hash(normalized_question)
            context_hash = exact_revisit_context_hash(
                normalized_question,
                context_scope_hash=contextual_revisit_scope_hash,
            )
            runtime_digest = str(seed.runtime_query_ref).rsplit(":", 1)[-1]
            whole_query_id = "runtime-whole:" + runtime_digest
            physical_id = "query-vector:sha256:" + hashlib.sha256(
                (
                    str(current_space.canonical_id)
                    + "\0"
                    + question_text_hash
                ).encode("utf-8")
            ).hexdigest()
            runtime_bundle = QueryVectorBundle(
                model_id=str(self.config.model.embedding_model),
                dimension=int(manifest.dimension),
                whole=context_vector,
                queries=(
                    QueryVector(
                        query_id=whole_query_id,
                        text_hash=question_text_hash,
                        role="whole",
                        vector=context_vector,
                        text=normalized_question,
                        physical_id=physical_id,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                    QueryVector(
                        query_id=seed.runtime_query_id,
                        text_hash=question_text_hash,
                        role="atomic",
                        vector=context_vector,
                        slot_id=seed.runtime_slot_id,
                        text=normalized_question,
                        physical_id=physical_id,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                ),
                embedding_space=current_space,
                physical_vectors=(
                    PhysicalQueryVector(
                        physical_id=physical_id,
                        text_hash=question_text_hash,
                        vector=context_vector,
                        embedding_space_id=str(current_space.canonical_id),
                    ),
                ),
                source_request_hash=source_request_hash,
                schema_version=2,
                whole_physical_id=physical_id,
            )
            runtime_slot = EvidenceSlot(
                slot_id=seed.runtime_slot_id,
                question=normalized_question,
                required=True,
                query_id=seed.runtime_query_id,
                query_refs=(seed.runtime_query_id,),
                # `EvidenceSlot` intentionally exposes only ordinary planner
                # origins.  The private rewrite authorization is carried by
                # `_T16RestrictedRewriteRevisit`, never smuggled into a
                # request-scoped slot field.
                origin="reused_template",
                support_mode="alternative",
                clause_ids=(seed.runtime_clause_id,),
            )
            requirements = RequirementResolution(
                request_mode="factual",
                status="resolved",
                requirements=(runtime_slot,),
                planner_origin="restricted_rewrite",
            )
            intent = QueryIntent(search_queries=[normalized_question])
            if infer_request_mode(normalized_question, intent) != "factual":
                return None
            budget = EvidenceSelectionBudget(
                episode_limit=int(seed.episode_limit),
                source_fact_limit=seed.source_fact_limit,
                delivery_token_limit=seed.delivery_token_limit,
            )
            proof = ExactRevisitSourceMappingProof(
                association_id=int(manifest.association_id),
                target_episode_id=int(seed.target_episode_id),
                slot_id=seed.runtime_slot_id,
                clause_ids=(seed.runtime_clause_id,),
                source_fact_id=seed.target_source_fact_id,
                mapping_ref=seed.target_mapping_ref,
            )
            revisit_input = ExactRevisitInput(
                domain=domain,
                context_hash=context_hash,
                requirements=requirements,
                intent=intent,
                query_vector_bundle=runtime_bundle,
                budget=budget,
                retrieval_policy_fingerprint=seed.retrieval_policy_fingerprint,
                source_closure_fingerprint=seed.source_closure_fingerprint,
                source_fact_roles_fingerprint=manifest.source_fact_roles_fingerprint,
                source_fact_refs_fingerprint=seed.source_fact_refs_fingerprint,
                evaluation_as_of=self._resolve_contextual_evaluation_as_of(None),
                endpoint_limit=int(seed.endpoint_limit),
                anchor_activations={
                    int(seed.anchor_episode_id): float(seed.anchor_activation)
                },
                base_episode_ids=(int(seed.anchor_episode_id),),
                base_slot_support={},
                target_source_mapping_proofs=(proof,),
            )
            if (
                int(runtime_bundle.physical_count) != 1
                or int(runtime_bundle.logical_count) != 2
                or str(runtime_bundle.source_request_hash) != source_request_hash
                or str(runtime_bundle.embedding_space_id)
                != str(manifest.embedding_space_id)
                or not np.array_equal(
                    runtime_bundle.vector_for_physical(physical_id), context_vector
                )
                or revisit_input.budget_fingerprint != seed.budget_fingerprint
                or revisit_input.anchor_manifest_fingerprint
                != seed.anchor_manifest_fingerprint
                or revisit_input.retrieval_policy_fingerprint
                != exact_revisit_policy_fingerprint(
                    self.config,
                    self.contextual_matcher,
                    endpoint_limit=int(seed.endpoint_limit),
                )
            ):
                return None
            ensure_deadline("runtime_reconstruction")
            return _T16RestrictedRewriteRevisit(
                revisit_input=revisit_input,
                manifest_id=int(manifest.manifest_id),
                manifest_fingerprint=str(manifest.manifest_fingerprint),
                creation_receipt_id=int(guard.creation_receipt_id),
                guard_fingerprint=str(guard.guard_fingerprint),
                rewrite_commitment=str(guard.rewrite_commitment),
                binding_commitment=str(guard.binding_commitment),
                manifest_binding_commitment=str(guard.manifest_binding_commitment),
                ready_manifest_commitment=str(guard.ready_manifest_commitment),
                context_cue_vector_fingerprint=str(
                    guard.context_cue_vector_fingerprint
                ),
                need_cue_vector_fingerprint=str(guard.need_cue_vector_fingerprint),
                model_id=str(guard.model_id),
                embedding_space_id=str(guard.embedding_space_id),
                dimension=int(guard.dimension),
                dtype=str(guard.dtype),
                commitment_key_id=str(guard.commitment_key_id),
                grammar_version=str(guard.grammar_version),
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            FloatingPointError,
            RuntimeError,
        ):
            return None

    def _exact_revisit_contract_repository(self) -> AssociationRepository | None:
        """Return a durable read view only when it is safe to probe v16."""

        if isinstance(self.associations, AssociationRepository):
            return self.associations
        if isinstance(self.associations, AssociationOverlay):
            return self.associations.repository
        # A staged view can contain request-local edge mutations.  Its durable
        # contract lookup is not the same retrieval state, so it cannot power
        # an exact shortcut.
        return None

    @staticmethod
    def _exact_revisit_overlay_hides(
        associations: object,
        association_id: int,
    ) -> bool:
        """Do not let a durable receipt escape an active edge overlay mask."""

        if not isinstance(associations, AssociationOverlay):
            return False
        return bool(
            int(association_id) in associations.hidden_ids
            or int(association_id) in associations.restored_rows
        )

    def _exact_revisit_delivery_guard(
        self,
        *,
        repository: AssociationRepository,
        contract,
        source_closure_fingerprint: str,
        source_fact_refs_fingerprint: str,
        source_fact_roles_fingerprint: str,
        selected_episode_ids: Sequence[int],
        expected_runtime_manifest_id: int | None = None,
        expected_runtime_manifest_fingerprint: str | None = None,
        restricted_rewrite: _T16RestrictedRewriteRevisit | None = None,
    ) -> str | None:
        """Return a short-lived, redacted guard for exact-result delivery.

        Exact revisit intentionally uses ordinary short SQLite reads rather
        than holding a reader transaction across local selection or answer
        generation.  The first exact pass records this guard, then a second
        complete pass reruns the matcher/gate/closure/selector and must
        reproduce one current coherent snapshot before response assembly.  A
        source, edge, contract, or publication change between the passes
        therefore becomes an ordinary exact miss instead of a mixed-view
        answer.  The result path performs one further revalidation directly
        before invoking an answer model; this is a snapshot-consistency check,
        not a claim that an unrelated future database commit cannot happen
        after the check returns.

        This value is process-local only: it contains no prose, vectors, or
        source text and is never emitted in a result or persisted.
        """

        try:
            if (expected_runtime_manifest_id is None) != (
                expected_runtime_manifest_fingerprint is None
            ):
                return None
            current_contract = repository.load_contextual_revisit_contract(
                int(contract.creation_receipt_id)
            )
            if (
                current_contract is None
                or str(current_contract.contract_fingerprint)
                != str(contract.contract_fingerprint)
            ):
                return None
            runtime_manifest_payload: dict[str, object] | None = None
            if expected_runtime_manifest_id is not None:
                runtime_manifest = (
                    repository.load_contextual_revisit_runtime_manifest(
                        int(contract.creation_receipt_id)
                    )
                )
                if (
                    runtime_manifest is None
                    or runtime_manifest.state != "ready"
                    or int(runtime_manifest.manifest_id)
                    != int(expected_runtime_manifest_id)
                    or str(runtime_manifest.manifest_fingerprint)
                    != str(expected_runtime_manifest_fingerprint)
                    or str(runtime_manifest.contract_fingerprint)
                    != str(contract.contract_fingerprint)
                    or runtime_manifest.to_revisit_contract().contract_fingerprint
                    != str(contract.contract_fingerprint)
                ):
                    return None
                runtime_manifest_payload = {
                    "manifest_fingerprint": str(runtime_manifest.manifest_fingerprint),
                    "manifest_id": int(runtime_manifest.manifest_id),
                }
            restricted_rewrite_payload: dict[str, object] | None = None
            if restricted_rewrite is not None:
                restored = repository.load_contextual_restricted_rewrite_guard(
                    int(contract.creation_receipt_id)
                )
                if restored is None:
                    return None
                guard, guarded_manifest = restored
                if (
                    int(guard.creation_receipt_id)
                    != int(restricted_rewrite.creation_receipt_id)
                    or str(guard.guard_fingerprint)
                    != str(restricted_rewrite.guard_fingerprint)
                    or str(guard.rewrite_commitment)
                    != str(restricted_rewrite.rewrite_commitment)
                    or str(guard.binding_commitment)
                    != str(restricted_rewrite.binding_commitment)
                    or str(guard.manifest_binding_commitment)
                    != str(restricted_rewrite.manifest_binding_commitment)
                    or not str(guard.ready_manifest_commitment)
                    or str(guard.ready_manifest_commitment)
                    != str(restricted_rewrite.ready_manifest_commitment)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(restricted_rewrite.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(restricted_rewrite.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(restricted_rewrite.model_id)
                    or str(guard.embedding_space_id)
                    != str(restricted_rewrite.embedding_space_id)
                    or int(guard.dimension) != int(restricted_rewrite.dimension)
                    or str(guard.dtype) != str(restricted_rewrite.dtype)
                    or str(guard.commitment_key_id)
                    != str(restricted_rewrite.commitment_key_id)
                    or str(guard.grammar_version)
                    != str(restricted_rewrite.grammar_version)
                    or int(guarded_manifest.manifest_id)
                    != int(restricted_rewrite.manifest_id)
                    or str(guarded_manifest.manifest_fingerprint)
                    != str(restricted_rewrite.manifest_fingerprint)
                    or str(guarded_manifest.contract_fingerprint)
                    != str(contract.contract_fingerprint)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(guarded_manifest.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(guarded_manifest.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(guarded_manifest.model_id)
                    or str(guard.embedding_space_id)
                    != str(guarded_manifest.embedding_space_id)
                    or int(guard.dimension) != int(guarded_manifest.dimension)
                    or str(guard.dtype) != str(guarded_manifest.dtype)
                    or not str(guarded_manifest.binding_fingerprint or "")
                ):
                    return None
                key = RestrictedRewriteCommitmentKey.from_environment()
                if (
                    not bool(
                        getattr(
                            self.config.retrieval,
                            "contextual_restricted_rewrite_enabled",
                            False,
                        )
                    )
                    or key is None
                    or str(key.key_id) != str(guard.commitment_key_id)
                ):
                    return None
                expected_binding = key.binding_commitment(
                    context_scope_hash=guard.context_scope_hash,
                    rewrite_commitment=guard.rewrite_commitment,
                    creation_request_id=guarded_manifest.seed.creation_request_id,
                    domain=guard.domain,
                    seed_fingerprint=guarded_manifest.seed.seed_fingerprint,
                    context_cue_vector_fingerprint=(
                        guarded_manifest.context_cue_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=(
                        guarded_manifest.need_cue_vector_fingerprint
                    ),
                    model_id=guarded_manifest.model_id,
                    embedding_space_id=guarded_manifest.embedding_space_id,
                    dimension=guarded_manifest.dimension,
                    dtype=guarded_manifest.dtype,
                    grammar_version=guard.grammar_version,
                    signature_version=guard.signature_version,
                )
                if not hmac.compare_digest(
                    str(guard.binding_commitment), str(expected_binding)
                ):
                    return None
                expected_manifest_binding = key.manifest_binding_commitment(
                    context_scope_hash=guard.context_scope_hash,
                    rewrite_commitment=guard.rewrite_commitment,
                    creation_request_id=guarded_manifest.seed.creation_request_id,
                    domain=guard.domain,
                    seed_fingerprint=guarded_manifest.seed.seed_fingerprint,
                    context_cue_vector_fingerprint=(
                        guarded_manifest.context_cue_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=(
                        guarded_manifest.need_cue_vector_fingerprint
                    ),
                    model_id=guarded_manifest.model_id,
                    embedding_space_id=guarded_manifest.embedding_space_id,
                    dimension=guarded_manifest.dimension,
                    dtype=guarded_manifest.dtype,
                    grammar_version=guard.grammar_version,
                    signature_version=guard.signature_version,
                    manifest_binding_fingerprint=guarded_manifest.binding_fingerprint,
                )
                if not hmac.compare_digest(
                    str(guard.manifest_binding_commitment),
                    str(expected_manifest_binding),
                ):
                    return None
                expected_ready_manifest_commitment = key.ready_manifest_commitment(
                    context_scope_hash=guard.context_scope_hash,
                    binding_commitment=guard.binding_commitment,
                    manifest_binding_commitment=guard.manifest_binding_commitment,
                    manifest_fingerprint=guarded_manifest.manifest_fingerprint,
                    signature_version=guard.signature_version,
                )
                if not hmac.compare_digest(
                    str(guard.ready_manifest_commitment),
                    str(expected_ready_manifest_commitment),
                ):
                    return None
                restricted_rewrite_payload = {
                    "binding_commitment": str(guard.binding_commitment),
                    "manifest_binding_commitment": str(
                        guard.manifest_binding_commitment
                    ),
                    "ready_manifest_commitment": str(
                        guard.ready_manifest_commitment
                    ),
                    "commitment_key_id": str(guard.commitment_key_id),
                    "context_cue_vector_fingerprint": str(
                        guard.context_cue_vector_fingerprint
                    ),
                    "dimension": int(guard.dimension),
                    "embedding_space_id": str(guard.embedding_space_id),
                    "grammar_version": str(guard.grammar_version),
                    "guard_fingerprint": str(guard.guard_fingerprint),
                    "model_id": str(guard.model_id),
                    "need_cue_vector_fingerprint": str(
                        guard.need_cue_vector_fingerprint
                    ),
                    "rewrite_commitment": str(guard.rewrite_commitment),
                }
            edge = repository.get(int(contract.association_id))
            if edge is None:
                return None
            edge_id = int(self._record_value(edge, "id"))
            from_id = int(self._record_value(edge, "from_id"))
            to_id = int(self._record_value(edge, "to_id"))
            context_cue_id = int(self._record_value(edge, "context_cue_id"))
            need_cue_id = int(self._record_value(edge, "need_cue_id"))
            utility_weight = float(self._record_value(edge, "utility_weight"))
            if (
                edge_id != int(contract.association_id)
                or context_cue_id != int(contract.context_cue_id)
                or need_cue_id != int(contract.need_cue_id)
                or str(self._record_value(edge, "from_type")) != "episode"
                or str(self._record_value(edge, "to_type")) != "episode"
                or str(self._record_value(edge, "association_mode"))
                != "contextual_recall"
                or from_id <= 0
                or to_id <= 0
                or not math.isfinite(utility_weight)
            ):
                return None
            publication = repository.get_contextual_index_publication()
            publication_epoch = int(publication["index_epoch"])
            publication_space = str(publication["embedding_space_id"] or "")
            if (
                publication_epoch < int(contract.ready_index_epoch)
                or publication_space != str(contract.embedding_space_id)
            ):
                return None
            normalized_selected_ids = tuple(
                sorted({int(value) for value in selected_episode_ids})
            )
            if not normalized_selected_ids or any(
                value <= 0 for value in normalized_selected_ids
            ):
                return None
            payload = {
                "contract_fingerprint": str(contract.contract_fingerprint),
                "edge": {
                    "association_mode": str(
                        self._record_value(edge, "association_mode")
                    ),
                    "context_cue_id": context_cue_id,
                    "expires_at": str(self._record_value(edge, "expires_at") or ""),
                    "from_id": from_id,
                    "from_type": str(self._record_value(edge, "from_type")),
                    "id": edge_id,
                    "lifecycle_state": str(
                        self._record_value(edge, "lifecycle_state")
                    ),
                    "need_cue_id": need_cue_id,
                    "source_request_hash": str(
                        self._record_value(edge, "source_request_hash") or ""
                    ),
                    "to_id": to_id,
                    "to_type": str(self._record_value(edge, "to_type")),
                    "updated_at": str(self._record_value(edge, "updated_at") or ""),
                    "utility_weight": utility_weight,
                },
                "publication": {
                    "context_cue_count": int(publication["context_cue_count"]),
                    "embedding_space_id": publication_space,
                    "index_epoch": publication_epoch,
                    "need_cue_count": int(publication["need_cue_count"]),
                    "published_at": str(publication["published_at"] or ""),
                },
                "runtime_manifest": runtime_manifest_payload,
                "restricted_rewrite": restricted_rewrite_payload,
                "selected_episode_ids": list(normalized_selected_ids),
                "source_closure_fingerprint": str(source_closure_fingerprint),
                "source_fact_refs_fingerprint": str(source_fact_refs_fingerprint),
                "source_fact_roles_fingerprint": str(source_fact_roles_fingerprint),
                "version": "exact-revisit-delivery-guard-v1",
            }
            material = json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return "exact-revisit-delivery:sha256:" + hashlib.sha256(
                material
            ).hexdigest()
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RuntimeError,
            sqlite3.DatabaseError,
        ):
            return None

    def _exact_revisit_source_mapping_contributions(
        self,
        *,
        proofs: Sequence[ExactRevisitSourceMappingProof],
        association_id: int,
        accepted_hits: Sequence[ContextualSlotHit],
        slots: Sequence[EvidenceSlot],
        source_facts: Mapping[int, SourceFactRef],
    ) -> list[CandidateContribution] | None:
        """Rebuild only separately proven source mappings for one exact edge.

        The ordinary contextual contribution remains ranking-only.  These
        edge-scoped contributions exist only when the caller supplies a
        previous source mapping proof whose fact identity is confirmed again
        from the full current source closure.  Keeping ``edge_id`` preserves
        contribution-mask semantics.
        """

        slot_by_id = {str(slot.slot_id): slot for slot in slots}
        hit_by_key: dict[tuple[int, int, str], ContextualSlotHit] = {}
        for hit in accepted_hits:
            key = (
                int(hit.association_id),
                int(hit.target_episode_id),
                str(hit.matched_slot_id),
            )
            previous = hit_by_key.get(key)
            if previous is None or float(hit.total_score) > float(previous.total_score):
                hit_by_key[key] = hit
        result: list[CandidateContribution] = []
        for proof in proofs:
            if int(proof.association_id) != int(association_id):
                return None
            key = (
                int(proof.association_id),
                int(proof.target_episode_id),
                str(proof.slot_id),
            )
            hit = hit_by_key.get(key)
            slot = slot_by_id.get(str(proof.slot_id))
            fact = source_facts.get(int(proof.target_episode_id))
            if (
                hit is None
                or slot is None
                or fact is None
                or fact.fact_id != str(proof.source_fact_id)
                or tuple(sorted(slot.clause_ids)) != tuple(proof.clause_ids)
            ):
                return None
            try:
                result.append(
                    CandidateContribution(
                        contribution_id=exact_revisit_mapping_contribution_id(proof),
                        episode_id=int(proof.target_episode_id),
                        slot_id=str(proof.slot_id),
                        lane="contextual_source_mapping",
                        edge_id=int(association_id),
                        query_ref=self._v3_opaque_ref(
                            "exact-revisit-mapping-query",
                            proof.association_id,
                            proof.target_episode_id,
                            proof.slot_id,
                        ),
                        rank_features={
                            "current_relevance_estimate": float(
                                hit.target_support_score
                            ),
                            "contextual_combined_score": float(hit.total_score),
                        },
                        source_facts=(fact,),
                        clause_supports=tuple(
                            ClauseSupport(
                                slot_id=slot.slot_id,
                                clause_id=clause_id,
                                source_fact=fact,
                                support_mode=slot.support_mode,
                                verification_status="source_bound",
                                mapping_ref=str(proof.mapping_ref),
                            )
                            for clause_id in proof.clause_ids
                        ),
                        source_span_refs=(fact.fact_id,),
                        evidence_ref=fact.fact_id,
                    )
                )
            except (TypeError, ValueError):
                return None
        return result or None

    def _exact_revisit_result(
        self,
        *,
        question: str,
        generate_answer: bool,
        revisit_input: ExactRevisitInput,
        contract,
        evaluation_as_of: str,
        deadline_seconds: float | None,
        deadline_at: float | None,
        ensure_deadline: Callable[[str], None],
        started_at: float,
        episodes: list[dict],
        selected_episodes: list[dict],
        attribution,
        gate_result: ContextualTargetGateResult,
        source_reasons: Mapping[int, str],
        merged_trace_rows: Sequence[Mapping[str, object]],
        matcher_context_hit_count: int,
        matcher_need_hit_count: int,
        matcher_proposal_count: int,
        revisit_kind: Literal["exact", "restricted_rewrite"] = "exact",
        pre_answer_delivery_revalidation: Callable[[], bool] | None = None,
    ) -> dict:
        """Return a normal-shaped response without answer/candidate caching.

        ``evaluation_as_of`` is the wall-clock snapshot captured for this
        request, not the timestamp carried by the historical ticket.  A
        ticket can prove that a candidate contract once existed, but it must
        never reactivate an edge or source revision that is no longer valid.
        """

        selected_ids = tuple(int(item["id"]) for item in selected_episodes)
        if selected_ids != tuple(attribution.treatment.selected_episode_ids):
            # This should be structurally impossible after
            # ``_v3_selected_episode_rows``.  Refuse rather than reporting a
            # selector receipt that differs from delivered evidence.
            raise ValueError("exact revisit delivery does not match selector")

        trace: dict[str, object] = {
            "enabled": True,
            "backend": "contextual_exact_revisit_contribution_selector_v3",
            "matcher_backend": "contextual_double_key",
            "evaluation_as_of": evaluation_as_of,
            "requirements_status": revisit_input.requirements.status,
            "authoritative_requirements": revisit_input.requirements.as_trace_payload(),
            "base_endpoint_manifest": list(revisit_input.base_episode_ids),
            "independent_base_endpoint_manifest": list(
                revisit_input.anchor_activation_map
            ),
            "attached_edges": [int(contract.association_id)],
            "attached_episode_ids": [
                int(item.target_episode_id) for item in gate_result.hits
            ],
            "matcher_context_hit_count": max(0, int(matcher_context_hit_count)),
            "matcher_need_hit_count": max(0, int(matcher_need_hit_count)),
            "matcher_proposal_count": max(0, int(matcher_proposal_count)),
            "target_gate": gate_result.as_trace_payload(),
            "target_checks": [
                item.as_trace_payload() for item in gate_result.outcomes
            ],
            "merged_candidates": [dict(item) for item in merged_trace_rows],
            "source_provenance_status": [
                {"episode_id": int(episode_id), "status": str(status)}
                for episode_id, status in sorted(source_reasons.items())
            ],
            "selected_contextual_contribution_ids": [
                contribution.contribution_id
                for aggregate in attribution.treatment.selected
                for contribution in aggregate.contributions
                if contribution.is_contextual
            ],
            "ranking_only_contextual_contribution_ids": [
                contribution.contribution_id
                for aggregate in aggregate_contributions(
                    contribution
                    for aggregate in attribution.treatment.selected
                    for contribution in aggregate.contributions
                )
                for contribution in aggregate.contributions
                if contribution.is_contextual
                and not any(
                    support.is_required_coverage
                    for support in contribution.clause_supports
                )
            ],
            "external_mask": {
                "edge_ids": list(revisit_input.masked_edge_ids),
                "contribution_ids": list(revisit_input.masked_contribution_ids),
            },
            "shadow": False,
            "compatibility_projection": "v3_contribution_selector",
        }
        self._v3_record_counterfactual_trace(trace, attribution)
        trace.update(self._query_vector_trace_metadata(revisit_input.query_vector_bundle))

        answer_paths: list[dict] = []
        chronology_notes: list[str] = []
        if generate_answer:
            # The shortcut does not get to turn a bounded request into an
            # unbounded answer call.  Check both sides because the answer
            # model itself cannot be cancelled reliably at this layer.
            ensure_deadline("exact_revisit_answer_generation")
            if (
                pre_answer_delivery_revalidation is not None
                and not pre_answer_delivery_revalidation()
            ):
                raise ValueError(
                    "exact revisit evidence changed before answer generation"
                )
            ensure_deadline("exact_revisit_answer_generation")
            answer, answer_audits, answer_revision_count = (
                self._generate_audited_answer(
                    question,
                    revisit_input.intent,
                    selected_episodes,
                    [],
                    answer_paths,
                    chronology_notes,
                    deadline_at=deadline_at,
                )
            )
            ensure_deadline("exact_revisit_answer_generation")
            answer_terminal_state = deepcopy(
                getattr(self, "_last_answer_execution_state", {})
            )
        else:
            answer, answer_audits, answer_revision_count = "", [], 0
            answer_terminal_state = {
                "terminal_state": "not_generated",
                "reason": "generate_answer_false",
            }

        ensure_deadline("exact_revisit_response")
        elapsed_seconds = max(0.0, perf_counter() - started_at)

        selector_trace = {
            key: deepcopy(value)
            for key, value in trace.items()
            if key
            in {
                "backend",
                "matcher_backend",
                "evaluation_as_of",
                "requirements_status",
                "authoritative_requirements",
                "base_endpoint_manifest",
                "independent_base_endpoint_manifest",
                "attached_edges",
                "attached_episode_ids",
                "matcher_context_hit_count",
                "matcher_need_hit_count",
                "matcher_proposal_count",
                "target_gate",
                "target_checks",
                "merged_candidates",
                "source_provenance_status",
                "selected_contextual_contribution_ids",
                "ranking_only_contextual_contribution_ids",
                "masked_selector",
                "treatment_selector",
                "contribution_counterfactual",
                "external_mask",
                "compatibility_projection",
            }
        }
        evidence_slot_trace = {
            "slot_selector_v2": selector_trace,
            "slot_selector_v3": deepcopy(selector_trace),
            "revisit_probe": {
                "status": (
                    "restricted_rewrite_hit"
                    if revisit_kind == "restricted_rewrite"
                    else "exact_hit"
                ),
                "mode": revisit_kind,
                "contract_fingerprint": str(contract.contract_fingerprint),
                "association_id": int(contract.association_id),
                "planner_skipped": True,
                "reranker_skipped": True,
                "embedding_skipped": True,
                "answer_cache_used": False,
                "executed_modules": {
                    "planner": False,
                    "embedding": False,
                    "reranker": False,
                    "contextual_matcher": True,
                    "target_gate": True,
                    "source_closure": True,
                    "contribution_selector": True,
                    "answer_generation": bool(generate_answer),
                },
            },
        }
        used_edge_ids = tuple(
            sorted(
                {
                    int(contribution.edge_id)
                    for aggregate in attribution.treatment.selected
                    for contribution in aggregate.contributions
                    if contribution.edge_id is not None
                }
            )
        )
        episode_fields = (
            "id",
            "score",
            "text",
            "participants",
            "source_key",
            "segment_index",
            "story_time_text",
            "story_order",
            "timeline_scope",
            "evidence_origin",
            "epistemic_status",
            "generation",
            "epistemic_note",
        )
        result = {
            "question": question,
            "query_plan_frozen": False,
            "query_plan_id": None,
            "intent": asdict(revisit_input.intent),
            "authoritative_requirements": revisit_input.requirements.as_result_dict(),
            "authoritative_requirements_trace": revisit_input.requirements.as_trace_payload(),
            "query_vector_bundle": revisit_input.query_vector_bundle.metadata(),
            "followup_search_queries": [],
            "followup_planner_invoked": False,
            "followup_planning_mode": "exact_revisit",
            "followup_planning_reason": "exact_revisit_preflight",
            "atomic_anchor_episode_ids": list(revisit_input.base_episode_ids),
            "candidate_episode_ids": [int(item["id"]) for item in episodes],
            "reranked_episode_ids": [],
            "rerank_trace": {
                "enabled": False,
                "reason": "exact_revisit_selector",
                "planner_skipped": True,
                "reranker_skipped": True,
                "answer_cache_used": False,
            },
            "evidence_slot_trace": evidence_slot_trace,
            "rerank_frozen_before_growth": True,
            "source_key_cohort": {"enabled": False, "reason": "exact_revisit"},
            "answer": answer,
            "answer_audits": answer_audits,
            "answer_revision_count": answer_revision_count,
            "answer_terminal_state": answer_terminal_state,
            "answer_generation_skipped": not generate_answer,
            "episode_ids": list(selected_ids),
            "concept_ids": [],
            "association_ids": list(used_edge_ids),
            "association_usage_recorded": False,
            "new_association_ids": [],
            "reinforced_association_ids": [],
            "growth_utility_gate": {
                "enabled": False,
                "reason": "exact_revisit_read_only",
                "retained_created_ids": [],
                "retained_reinforced_ids": [],
                "removed_created_ids": [],
                "restored_reinforced_ids": [],
                "premise_closure_ids": [],
            },
            "growth_counterfactual_utility": {
                "enabled": False,
                "reason": "exact_revisit_read_only",
            },
            "association_cue_ids": [
                int(contract.context_cue_id),
                int(contract.need_cue_id),
            ],
            "association_cue_entries": [],
            "association_capsule_fast_lane": False,
            "contextual_association": {
                "enabled": True,
                "backend": str(trace["backend"]),
                "matcher_backend": str(trace["matcher_backend"]),
                "evaluation_as_of": evaluation_as_of,
                "context_prototype_hits": int(trace["matcher_context_hit_count"]),
                "need_prototype_hits": int(trace["matcher_need_hit_count"]),
                "candidate_count": len(gate_result.hits),
                "selected_count": len(
                    trace["selected_contextual_contribution_ids"]
                ),
                "treatment_selected_count": len(
                    attribution.treatment.selected_episode_ids
                ),
                "shadow": False,
                "new_slot_count": int(trace.get("new_slot_count", 0)),
                "harm_count": int(bool(trace.get("harm", False))),
                "attached_edges": list(trace["attached_edges"]),
                "attached_episode_ids": list(trace["attached_episode_ids"]),
                "masked_episode_ids": list(trace.get("masked_selected_episode_ids", [])),
                "treatment_episode_ids": list(
                    trace.get("treatment_selected_episode_ids", [])
                ),
                "external_calls": 0,
                "target_gate": deepcopy(trace["target_gate"]),
                "compatibility_projection": str(trace["compatibility_projection"]),
                "contribution_selector": deepcopy(trace.get("treatment_selector", {})),
                "masked_contribution_selector": deepcopy(trace.get("masked_selector", {})),
                "contribution_counterfactual": deepcopy(
                    trace.get("contribution_counterfactual", {})
                ),
                "merged_candidates": deepcopy(trace["merged_candidates"]),
                "source_provenance_status": deepcopy(trace["source_provenance_status"]),
                "selected_contextual_contribution_ids": list(
                    trace["selected_contextual_contribution_ids"]
                ),
                "ranking_only_contextual_contribution_ids": list(
                    trace["ranking_only_contextual_contribution_ids"]
                ),
            },
            "chronology_notes": chronology_notes,
            "evidence_episodes": [
                {key: item[key] for key in episode_fields if key in item}
                for item in selected_episodes
            ],
            "evidence_concepts": [],
            "association_paths": answer_paths,
            "paragraph_retrieval_enabled": self.paragraph_retrieval_enabled,
            "sparse_retrieval_enabled": self.sparse_retrieval_enabled,
            "query_embedding_cache": {"hit_count": 0, "miss_count": 0},
            "exact_revisit": {
                "status": "hit",
                "mode": revisit_kind,
                "contract_fingerprint": str(contract.contract_fingerprint),
                "association_id": int(contract.association_id),
                "evaluation_as_of": evaluation_as_of,
                "planner_skipped": True,
                "reranker_skipped": True,
                "embedding_skipped": True,
                "answer_cache_used": False,
                "executed_modules": {
                    "planner": False,
                    "embedding": False,
                    "reranker": False,
                    "contextual_matcher": True,
                    "target_gate": True,
                    "source_closure": True,
                    "contribution_selector": True,
                    "answer_generation": bool(generate_answer),
                },
            },
            "timings": {
                "version": "query-stage-timing-v1",
                "phases_seconds": {
                    "exact_revisit": round(max(0.0, float(elapsed_seconds)), 6)
                },
                "measured_seconds": round(max(0.0, float(elapsed_seconds)), 6),
                "unattributed_seconds": 0.0,
                "total_seconds": round(max(0.0, float(elapsed_seconds)), 6),
                "deadline_seconds": deadline_seconds,
            },
        }
        # The outer public boundary removes this internal carrier on every
        # path.  It exists only long enough to form a Source-bearing
        # ``stop_after='evidence'`` response; normal exact results and logs
        # retain their historical compact evidence shape.
        result["_runtime_selected_evidence"] = selected_episodes
        return result

    def _try_exact_revisit_v3(
        self,
        question: str,
        *,
        generate_answer: bool,
        exact_revisit_input: ExactRevisitInput | None,
        intent_override: QueryIntent | dict | None,
        followup_queries_override: list[str] | None,
        frozen_plan: dict | None,
        query_embeddings_override: dict[str, np.ndarray] | None,
        query_vector_bundle: QueryVectorBundle | None,
        strict_vector_bundle: bool,
        contextual_domain: str | None,
        contextual_endpoint_limit: int | None,
        contextual_evaluation_as_of: str | None,
        contextual_revisit_scope_hash: str | None,
        trace_bridge: QueryTraceBridge | None,
        deadline_seconds: float | None,
        deadline_at: float | None,
        expected_runtime_manifest_id: int | None = None,
        expected_runtime_manifest_fingerprint: str | None = None,
        restricted_rewrite: _T16RestrictedRewriteRevisit | None = None,
        _delivery_guard: str | None = None,
        _started_at: float | None = None,
        _emit_contextual_event: bool = True,
    ) -> dict | None:
        """Try one fully revalidated local V3 exact revisit, or return ``None``.

        Returning ``None`` is deliberately ordinary control flow: all misses
        fall through to the existing planner/retrieval pipeline.  This method
        never embeds, invokes a planner/reranker, writes an association, or
        turns a contextual relevance score into factual coverage.
        """

        started = perf_counter() if _started_at is None else _started_at

        def ensure_deadline(stage: str) -> None:
            if deadline_at is not None and monotonic() >= deadline_at:
                raise TimeoutError(f"query deadline exceeded before {stage}")

        ensure_deadline("exact_revisit_preflight")
        if (expected_runtime_manifest_id is None) != (
            expected_runtime_manifest_fingerprint is None
        ):
            return None
        if restricted_rewrite is not None and not isinstance(
            restricted_rewrite, _T16RestrictedRewriteRevisit
        ):
            return None
        if restricted_rewrite is not None and (
            expected_runtime_manifest_id is None
            or int(expected_runtime_manifest_id)
            != int(restricted_rewrite.manifest_id)
            or str(expected_runtime_manifest_fingerprint)
            != str(restricted_rewrite.manifest_fingerprint)
        ):
            return None
        if (
            exact_revisit_input is None
            or not isinstance(exact_revisit_input, ExactRevisitInput)
            or trace_bridge is not None
            or frozen_plan is not None
            or strict_vector_bundle
            or intent_override is not None
            or followup_queries_override is not None
            or query_embeddings_override
            or query_vector_bundle is not None
            or not self.config.retrieval.contextual_association_enabled
            or self.config.retrieval.contextual_association_shadow
            or self.contextual_matcher is None
        ):
            return None
        # A public ``contextual_evaluation_as_of`` is an explicit historical
        # replay request.  Exact revisit has no replay mode: it only serves a
        # live request after current lifecycle/source revalidation.  Let the
        # normal path own its documented replay behavior instead.
        if contextual_evaluation_as_of is not None:
            return None
        if contextual_domain is not None and str(contextual_domain).strip() != exact_revisit_input.domain:
            return None
        if (
            contextual_endpoint_limit is not None
            and int(contextual_endpoint_limit) != int(exact_revisit_input.endpoint_limit)
        ):
            return None
        try:
            # Keep malformed ticket metadata fail-closed, but do not grant it
            # authority over the current database view.  In particular, an
            # old ticket timestamp must not make a now-expired edge visible.
            self._resolve_contextual_evaluation_as_of(
                exact_revisit_input.evaluation_as_of
            )
            evaluation_as_of = self._resolve_contextual_evaluation_as_of(None)
            ensure_deadline("exact_revisit_ticket_validation")
            if exact_revisit_input.context_hash != exact_revisit_context_hash(
                question,
                context_scope_hash=contextual_revisit_scope_hash,
            ):
                return None
            bundle = exact_revisit_input.query_vector_bundle
            if bundle.source_request_hash != exact_revisit_request_hash(question):
                return None
            coordinator = self._new_query_embedding_coordinator()
            current_space = coordinator.embedding_space()
            if (
                str(bundle.model_id) != str(self.config.model.embedding_model)
                or int(bundle.dimension) != int(self.config.model.embedding_dimension)
                or str(bundle.embedding_space_id) != str(current_space.canonical_id)
            ):
                return None
            if int(exact_revisit_input.budget.episode_limit) != int(
                self.config.retrieval.answer_episode_limit
            ):
                return None
            if exact_revisit_input.retrieval_policy_fingerprint != exact_revisit_policy_fingerprint(
                self.config,
                self.contextual_matcher,
                endpoint_limit=exact_revisit_input.endpoint_limit,
            ):
                return None
            repository = self._exact_revisit_contract_repository()
            if repository is None:
                return None
            if restricted_rewrite is None:
                contract = repository.find_contextual_revisit_contract(
                    exact_revisit_input.contract_lookup()
                )
            else:
                # This is the only deliberate departure from V16's normal
                # question/vector equality check.  It is not a switch on the
                # public ticket: a current full grammar parse and scoped HMAC
                # must reproduce the private capability, then the sidecar's
                # ready V17 manifest supplies the canonical V16 contract.
                key = RestrictedRewriteCommitmentKey.from_environment()
                ir = parse_restricted_rewrite_question(question)
                if (
                    not bool(
                        getattr(
                            self.config.retrieval,
                            "contextual_restricted_rewrite_enabled",
                            False,
                        )
                    )
                    or key is None
                    or ir is None
                    or contextual_revisit_scope_hash is None
                    or str(key.key_id)
                    != str(restricted_rewrite.commitment_key_id)
                    or str(ir.grammar_version)
                    != str(restricted_rewrite.grammar_version)
                    or str(
                        key.commitment(
                            context_scope_hash=contextual_revisit_scope_hash,
                            ir=ir,
                        )
                    )
                    != str(restricted_rewrite.rewrite_commitment)
                    or not str(restricted_rewrite.binding_commitment)
                    or not str(restricted_rewrite.manifest_binding_commitment)
                    or not str(restricted_rewrite.ready_manifest_commitment)
                ):
                    return None
                restored = repository.load_contextual_restricted_rewrite_guard(
                    int(restricted_rewrite.creation_receipt_id)
                )
                if restored is None:
                    return None
                guard, restricted_manifest = restored
                if (
                    int(guard.creation_receipt_id)
                    != int(restricted_rewrite.creation_receipt_id)
                    or str(guard.guard_fingerprint)
                    != str(restricted_rewrite.guard_fingerprint)
                    or str(guard.rewrite_commitment)
                    != str(restricted_rewrite.rewrite_commitment)
                    or str(guard.binding_commitment)
                    != str(restricted_rewrite.binding_commitment)
                    or str(guard.manifest_binding_commitment)
                    != str(restricted_rewrite.manifest_binding_commitment)
                    or not str(guard.ready_manifest_commitment)
                    or str(guard.ready_manifest_commitment)
                    != str(restricted_rewrite.ready_manifest_commitment)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(restricted_rewrite.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(restricted_rewrite.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(restricted_rewrite.model_id)
                    or str(guard.embedding_space_id)
                    != str(restricted_rewrite.embedding_space_id)
                    or int(guard.dimension) != int(restricted_rewrite.dimension)
                    or str(guard.dtype) != str(restricted_rewrite.dtype)
                    or str(guard.commitment_key_id)
                    != str(restricted_rewrite.commitment_key_id)
                    or str(guard.grammar_version)
                    != str(restricted_rewrite.grammar_version)
                    or int(restricted_manifest.manifest_id)
                    != int(restricted_rewrite.manifest_id)
                    or str(restricted_manifest.manifest_fingerprint)
                    != str(restricted_rewrite.manifest_fingerprint)
                    or int(restricted_manifest.association_id)
                    != int(guard.association_id)
                    or str(guard.context_cue_vector_fingerprint)
                    != str(restricted_manifest.context_cue_vector_fingerprint)
                    or str(guard.need_cue_vector_fingerprint)
                    != str(restricted_manifest.need_cue_vector_fingerprint)
                    or str(guard.model_id) != str(restricted_manifest.model_id)
                    or str(guard.embedding_space_id)
                    != str(restricted_manifest.embedding_space_id)
                    or int(guard.dimension) != int(restricted_manifest.dimension)
                    or str(guard.dtype) != str(restricted_manifest.dtype)
                    or str(restricted_manifest.contract_fingerprint or "") == ""
                    or str(restricted_manifest.binding_fingerprint or "") == ""
                ):
                    return None
                expected_binding = key.binding_commitment(
                    context_scope_hash=contextual_revisit_scope_hash,
                    rewrite_commitment=guard.rewrite_commitment,
                    creation_request_id=restricted_manifest.seed.creation_request_id,
                    domain=guard.domain,
                    seed_fingerprint=restricted_manifest.seed.seed_fingerprint,
                    context_cue_vector_fingerprint=(
                        restricted_manifest.context_cue_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=(
                        restricted_manifest.need_cue_vector_fingerprint
                    ),
                    model_id=restricted_manifest.model_id,
                    embedding_space_id=restricted_manifest.embedding_space_id,
                    dimension=restricted_manifest.dimension,
                    dtype=restricted_manifest.dtype,
                    grammar_version=guard.grammar_version,
                    signature_version=guard.signature_version,
                )
                if not hmac.compare_digest(
                    str(guard.binding_commitment), str(expected_binding)
                ):
                    return None
                expected_manifest_binding = key.manifest_binding_commitment(
                    context_scope_hash=contextual_revisit_scope_hash,
                    rewrite_commitment=guard.rewrite_commitment,
                    creation_request_id=restricted_manifest.seed.creation_request_id,
                    domain=guard.domain,
                    seed_fingerprint=restricted_manifest.seed.seed_fingerprint,
                    context_cue_vector_fingerprint=(
                        restricted_manifest.context_cue_vector_fingerprint
                    ),
                    need_cue_vector_fingerprint=(
                        restricted_manifest.need_cue_vector_fingerprint
                    ),
                    model_id=restricted_manifest.model_id,
                    embedding_space_id=restricted_manifest.embedding_space_id,
                    dimension=restricted_manifest.dimension,
                    dtype=restricted_manifest.dtype,
                    grammar_version=guard.grammar_version,
                    signature_version=guard.signature_version,
                    manifest_binding_fingerprint=(
                        restricted_manifest.binding_fingerprint
                    ),
                )
                if not hmac.compare_digest(
                    str(guard.manifest_binding_commitment),
                    str(expected_manifest_binding),
                ):
                    return None
                expected_ready_manifest_commitment = key.ready_manifest_commitment(
                    context_scope_hash=contextual_revisit_scope_hash,
                    binding_commitment=guard.binding_commitment,
                    manifest_binding_commitment=guard.manifest_binding_commitment,
                    manifest_fingerprint=restricted_manifest.manifest_fingerprint,
                    signature_version=guard.signature_version,
                )
                if not hmac.compare_digest(
                    str(guard.ready_manifest_commitment),
                    str(expected_ready_manifest_commitment),
                ):
                    return None
                contract = restricted_manifest.to_revisit_contract()
                current_contract = repository.load_contextual_revisit_contract(
                    int(contract.creation_receipt_id)
                )
                if current_contract != contract:
                    return None
            ensure_deadline("exact_revisit_contract_lookup")
            if contract is None:
                return None
            if expected_runtime_manifest_id is not None:
                runtime_manifest = repository.load_contextual_revisit_runtime_manifest(
                    int(contract.creation_receipt_id)
                )
                if (
                    runtime_manifest is None
                    or runtime_manifest.state != "ready"
                    or int(runtime_manifest.manifest_id)
                    != int(expected_runtime_manifest_id)
                    or str(runtime_manifest.manifest_fingerprint)
                    != str(expected_runtime_manifest_fingerprint)
                    or str(runtime_manifest.contract_fingerprint)
                    != str(contract.contract_fingerprint)
                    or runtime_manifest.to_revisit_contract().contract_fingerprint
                    != str(contract.contract_fingerprint)
                ):
                    return None
            if (
                contract.source_fact_roles_fingerprint
                != exact_revisit_input.source_fact_roles_fingerprint
                or contract.source_fact_refs_fingerprint
                != exact_revisit_input.source_fact_refs_fingerprint
                or int(contract.association_id) in set(exact_revisit_input.masked_edge_ids)
                or self._exact_revisit_overlay_hides(
                    self.associations, int(contract.association_id)
                )
            ):
                return None

            base_nodes = [
                TraversedNode(
                    "episode",
                    int(episode_id),
                    float(exact_revisit_input.anchor_activation_map.get(episode_id, 0.001)),
                )
                for episode_id in exact_revisit_input.base_episode_ids
            ]
            base_episodes, _ = self._materialize_nodes(base_nodes, include_sources=True)
            if {
                int(item["id"]) for item in base_episodes
            } != set(exact_revisit_input.base_episode_ids):
                return None
            slots = list(exact_revisit_input.requirements.requirements)
            base_contributions, _base_reasons, _base_trace_rows = (
                self._v3_build_candidate_contributions(
                    episodes=base_episodes,
                    slots=slots,
                    slot_support=exact_revisit_input.base_slot_support_map,
                    base_episode_ids=exact_revisit_input.base_episode_ids,
                )
            )
            ensure_deadline("exact_revisit_base_validation")
            base_attribution = contribution_contextual_attribution(
                base_contributions,
                slots,
                exact_revisit_input.budget,
            )
            unresolved_slot_ids = set(base_attribution.masked.missing_required_clauses)
            if not unresolved_slot_ids:
                return None
            unresolved_slots = [
                slot for slot in slots if str(slot.slot_id) in unresolved_slot_ids
            ]
            proofs = tuple(exact_revisit_input.target_source_mapping_proofs)
            if {proof.slot_id for proof in proofs} != unresolved_slot_ids:
                return None
            if any(
                int(proof.association_id) != int(contract.association_id)
                for proof in proofs
            ):
                return None
            # The transient ticket must not use a contextual endpoint as an
            # active anchor for the same repair.  This is checked separately
            # from the matcher\'s per-proposal self-edge guard because a
            # multi-anchor ticket could otherwise hide the circularity.
            if any(
                int(proof.target_episode_id)
                in exact_revisit_input.anchor_activation_map
                for proof in proofs
            ):
                return None

            preliminary = self.contextual_recall(
                bundle,
                domain=exact_revisit_input.domain,
                endpoint_limit=exact_revisit_input.endpoint_limit,
                active_anchor_ids=exact_revisit_input.anchor_activation_map,
                unresolved_slot_ids=sorted(unresolved_slot_ids),
                evaluation_as_of=evaluation_as_of,
                emit_event=_emit_contextual_event,
            )
            ensure_deadline("exact_revisit_matcher")
            raw_proposals = preliminary.get("pre_target_proposals", [])
            if not isinstance(raw_proposals, (list, tuple)) or not all(
                isinstance(item, ContextualPreTargetProposal) for item in raw_proposals
            ):
                return None
            raw_context_hits = preliminary.get("context_hits", [])
            raw_need_hits = preliminary.get("need_hits", [])
            matcher_context_hit_count = (
                len(raw_context_hits)
                if isinstance(raw_context_hits, (list, tuple))
                else 0
            )
            matcher_need_hit_count = (
                sum(
                    len(item)
                    for item in raw_need_hits
                    if isinstance(item, (list, tuple))
                )
                if isinstance(raw_need_hits, (list, tuple))
                else 0
            )
            proof_keys = {
                (proof.association_id, proof.target_episode_id, proof.slot_id)
                for proof in proofs
            }
            expected_bindings = {
                slot.slot_id: next(
                    item
                    for item in bundle.bindings_for_slot(slot.slot_id)
                    if item.role == "atomic"
                )
                for slot in unresolved_slots
            }
            proposals = [
                proposal
                for proposal in raw_proposals
                if (
                    int(proposal.association_id) == int(contract.association_id)
                    and int(proposal.context_cue_id) == int(contract.context_cue_id)
                    and int(proposal.need_cue_id) == int(contract.need_cue_id)
                    and int(proposal.anchor_episode_id)
                    in exact_revisit_input.anchor_activation_map
                    and int(proposal.target_episode_id) != int(proposal.anchor_episode_id)
                    and (
                        int(proposal.association_id),
                        int(proposal.target_episode_id),
                        str(proposal.matched_slot_id),
                    )
                    in proof_keys
                    and str(proposal.matched_slot_id) in expected_bindings
                    and str(proposal.matched_query_id)
                    == str(expected_bindings[str(proposal.matched_slot_id)].query_id)
                    and str(proposal.matched_physical_id)
                    == str(expected_bindings[str(proposal.matched_slot_id)].physical_id)
                    and str(proposal.embedding_space_id) == str(bundle.embedding_space_id)
                )
            ]
            if not proposals:
                return None
            gate_result = self._gate_contextual_targets(
                proposals=proposals,
                bundle=bundle,
                unresolved_slots=unresolved_slots,
                endpoint_limit=exact_revisit_input.endpoint_limit,
                evaluation_as_of=evaluation_as_of,
            )
            ensure_deadline("exact_revisit_target_gate")
            accepted_keys = {
                (
                    int(hit.association_id),
                    int(hit.target_episode_id),
                    str(hit.matched_slot_id),
                )
                for hit in gate_result.hits
            }
            if proof_keys != accepted_keys:
                return None
            target_scores: dict[int, float] = {}
            for hit in gate_result.hits:
                target_scores[int(hit.target_episode_id)] = max(
                    target_scores.get(int(hit.target_episode_id), 0.0),
                    float(hit.target_support_score),
                )
            recovered, _ = self._materialize_nodes(
                [
                    TraversedNode("episode", episode_id, score)
                    for episode_id, score in sorted(target_scores.items())
                    if episode_id not in set(exact_revisit_input.base_episode_ids)
                ],
                include_sources=True,
            )
            episodes_by_id = {
                int(item["id"]): item for item in [*base_episodes, *recovered]
            }
            if not set(target_scores).issubset(episodes_by_id):
                return None
            episodes = [episodes_by_id[key] for key in sorted(episodes_by_id)]
            target_relevance = {
                (item.matched_slot_id, int(item.target_episode_id)): float(
                    item.relevance_estimate
                )
                for item in gate_result.outcomes
                if item.accepted
                and item.relevance_estimate is not None
                and math.isfinite(float(item.relevance_estimate))
            }
            contributions, source_reasons, trace_rows = (
                self._v3_build_candidate_contributions(
                    episodes=episodes,
                    slots=slots,
                    slot_support=exact_revisit_input.base_slot_support_map,
                    contextual_hits=gate_result.hits,
                    target_relevance_scores=target_relevance,
                    base_episode_ids=exact_revisit_input.base_episode_ids,
                )
            )
            facts, fresh_reasons = self._v3_source_fact_closure(
                [int(item["id"]) for item in episodes]
            )
            ensure_deadline("exact_revisit_source_closure")
            expected_episode_ids = set(exact_revisit_input.base_episode_ids).union(
                target_scores
            )
            if (
                set(facts) != expected_episode_ids
                or any(
                    fresh_reasons.get(episode_id) != "source_bound"
                    for episode_id in expected_episode_ids
                )
            ):
                return None
            proof_contributions = self._exact_revisit_source_mapping_contributions(
                proofs=proofs,
                association_id=int(contract.association_id),
                accepted_hits=gate_result.hits,
                slots=slots,
                source_facts=facts,
            )
            if proof_contributions is None:
                return None
            all_contributions = [*contributions, *proof_contributions]
            source_facts = tuple(facts[key] for key in sorted(facts))
            fresh_closure_fingerprint = exact_revisit_source_closure_fingerprint(
                source_facts
            )
            fresh_refs_fingerprint = exact_revisit_source_fact_refs_fingerprint(
                source_facts
            )
            fresh_roles_fingerprint = exact_revisit_mapping_roles_fingerprint(
                facts,
                base_slot_support=exact_revisit_input.base_slot_support_map,
                target_source_mapping_proofs=proofs,
            )
            if (
                fresh_closure_fingerprint
                != exact_revisit_input.source_closure_fingerprint
                or fresh_closure_fingerprint != contract.source_closure_fingerprint
                or fresh_refs_fingerprint
                != exact_revisit_input.source_fact_refs_fingerprint
                or fresh_refs_fingerprint != contract.source_fact_refs_fingerprint
                or fresh_roles_fingerprint
                != exact_revisit_input.source_fact_roles_fingerprint
                or fresh_roles_fingerprint != contract.source_fact_roles_fingerprint
            ):
                return None
            proof_ids = {
                exact_revisit_mapping_contribution_id(proof) for proof in proofs
            }
            # An externally supplied contribution mask is an evaluation
            # contract, not a hint to ignore only the factual mapping.  If it
            # names *any* route carried by this durable edge (including the
            # ordinary relevance-only contextual route), the exact shortcut
            # cannot honestly stand in for the masked query.  Fall through so
            # the full path records the appropriate counterfactual instead.
            contracted_contribution_ids = {
                contribution.contribution_id
                for contribution in all_contributions
                if int(contribution.edge_id or 0) == int(contract.association_id)
            }
            if set(exact_revisit_input.masked_contribution_ids).intersection(
                proof_ids | contracted_contribution_ids
            ):
                return None
            effective_aggregates = mask_contribution_aggregates(
                all_contributions,
                contribution_ids=exact_revisit_input.masked_contribution_ids,
                edge_ids=exact_revisit_input.masked_edge_ids,
            )
            effective_contributions = [
                contribution
                for aggregate in effective_aggregates
                for contribution in aggregate.contributions
            ]
            attribution = contribution_contextual_attribution(
                effective_contributions,
                slots,
                exact_revisit_input.budget,
            )
            ensure_deadline("exact_revisit_contribution_selector")
            selected_proof_ids = set(
                attribution.treatment.selected_contribution_ids
            ).intersection(proof_ids)
            if not selected_proof_ids:
                return None
            selected_episodes = self._v3_selected_episode_rows(
                attribution.treatment,
                episodes,
            )
            if tuple(int(item["id"]) for item in selected_episodes) != tuple(
                attribution.treatment.selected_episode_ids
            ):
                return None
            delivery_guard = self._exact_revisit_delivery_guard(
                repository=repository,
                contract=contract,
                source_closure_fingerprint=fresh_closure_fingerprint,
                source_fact_refs_fingerprint=fresh_refs_fingerprint,
                source_fact_roles_fingerprint=fresh_roles_fingerprint,
                selected_episode_ids=attribution.treatment.selected_episode_ids,
                expected_runtime_manifest_id=expected_runtime_manifest_id,
                expected_runtime_manifest_fingerprint=(
                    expected_runtime_manifest_fingerprint
                ),
                restricted_rewrite=restricted_rewrite,
            )
            ensure_deadline("exact_revisit_delivery_revalidation")
            if delivery_guard is None:
                return None
            if _delivery_guard is None:
                # Do not hold a SQLite transaction or an index read lock while
                # recomputing the route.  A second full local pass gives the
                # delivered evidence one fresh, coherent validation attempt.
                # A third compact revalidation occurs immediately before an
                # optional answer-model invocation; these are snapshots, not
                # a long-lived database lock across a model request.
                return self._try_exact_revisit_v3(
                    question,
                    generate_answer=generate_answer,
                    exact_revisit_input=exact_revisit_input,
                    intent_override=intent_override,
                    followup_queries_override=followup_queries_override,
                    frozen_plan=frozen_plan,
                    query_embeddings_override=query_embeddings_override,
                    query_vector_bundle=query_vector_bundle,
                    strict_vector_bundle=strict_vector_bundle,
                    contextual_domain=contextual_domain,
                    contextual_endpoint_limit=contextual_endpoint_limit,
                    contextual_evaluation_as_of=contextual_evaluation_as_of,
                    contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                    trace_bridge=trace_bridge,
                    deadline_seconds=deadline_seconds,
                    deadline_at=deadline_at,
                    expected_runtime_manifest_id=expected_runtime_manifest_id,
                    expected_runtime_manifest_fingerprint=(
                        expected_runtime_manifest_fingerprint
                    ),
                    restricted_rewrite=restricted_rewrite,
                    _delivery_guard=delivery_guard,
                    _started_at=started,
                    _emit_contextual_event=_emit_contextual_event,
                )
            if delivery_guard != _delivery_guard:
                return None
            def pre_answer_delivery_revalidation() -> bool:
                return bool(
                    self._exact_revisit_delivery_guard(
                        repository=repository,
                        contract=contract,
                        source_closure_fingerprint=fresh_closure_fingerprint,
                        source_fact_refs_fingerprint=fresh_refs_fingerprint,
                        source_fact_roles_fingerprint=fresh_roles_fingerprint,
                        selected_episode_ids=(
                            attribution.treatment.selected_episode_ids
                        ),
                        expected_runtime_manifest_id=expected_runtime_manifest_id,
                        expected_runtime_manifest_fingerprint=(
                            expected_runtime_manifest_fingerprint
                        ),
                        restricted_rewrite=restricted_rewrite,
                    )
                    == _delivery_guard
                )
            return self._exact_revisit_result(
                question=question,
                generate_answer=generate_answer,
                revisit_input=exact_revisit_input,
                contract=contract,
                evaluation_as_of=evaluation_as_of,
                deadline_seconds=deadline_seconds,
                deadline_at=deadline_at,
                ensure_deadline=ensure_deadline,
                started_at=started,
                episodes=episodes,
                selected_episodes=selected_episodes,
                attribution=attribution,
                gate_result=gate_result,
                source_reasons={**source_reasons, **fresh_reasons},
                merged_trace_rows=[*trace_rows, *(
                    self._v3_contribution_trace_payload(
                        contribution,
                        fresh_reasons.get(
                            contribution.episode_id, "source_closure_missing"
                        ),
                    )
                    for contribution in proof_contributions
                )],
                matcher_context_hit_count=matcher_context_hit_count,
                matcher_need_hit_count=matcher_need_hit_count,
                matcher_proposal_count=len(raw_proposals),
                revisit_kind=(
                    "restricted_rewrite"
                    if restricted_rewrite is not None
                    else "exact"
                ),
                pre_answer_delivery_revalidation=pre_answer_delivery_revalidation,
            )
        except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError):
            # A malformed caller ticket or stale/mismatched local state is a
            # cache miss, not a partly trusted answer route.
            return None

    def _try_contextual_revisit_only(
        self,
        question: str,
        *,
        contextual_domain: str | None,
        contextual_revisit_scope_hash: str | None,
        deadline_seconds: float | None,
        deadline_at: float | None,
    ) -> dict | None:
        """Run only the automatic local revisit lanes, never ordinary retrieval.

        This is deliberately narrower than :meth:`query`: it fixes every
        replay, trace, override, learning, and answer-generation input to the
        read-only automatic-revisit shape.  A non-hit is ordinary cache-like
        control flow and returns ``None`` here; it must not fall through to
        ``_query_impl``.  Keeping that miss behavior in this private helper
        lets the public preflight boundary remain useful to callers that want
        to decide whether to run their own planner.
        """

        automatic_revisit = self._v17_automatic_exact_revisit(
            question,
            intent_override=None,
            followup_queries_override=None,
            frozen_plan=None,
            query_embeddings_override=None,
            query_vector_bundle=None,
            strict_vector_bundle=False,
            contextual_domain=contextual_domain,
            contextual_endpoint_limit=None,
            contextual_evaluation_as_of=None,
            contextual_revisit_scope_hash=contextual_revisit_scope_hash,
            trace_bridge=None,
            deadline_at=deadline_at,
        )
        restricted_rewrite = (
            self._t16_automatic_restricted_rewrite_revisit(
                question,
                intent_override=None,
                followup_queries_override=None,
                frozen_plan=None,
                query_embeddings_override=None,
                query_vector_bundle=None,
                strict_vector_bundle=False,
                contextual_domain=contextual_domain,
                contextual_endpoint_limit=None,
                contextual_evaluation_as_of=None,
                contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                trace_bridge=None,
                deadline_at=deadline_at,
            )
            if automatic_revisit is None
            else None
        )
        effective_exact_input = (
            automatic_revisit.revisit_input
            if automatic_revisit is not None
            else (
                restricted_rewrite.revisit_input
                if restricted_rewrite is not None
                else None
            )
        )
        return self._try_exact_revisit_v3(
            question,
            # This API is a preflight, not a replacement answer endpoint.
            # Avoiding generation makes a hit as provider-free as the local
            # automatic lanes and ensures a miss has no model fallback.
            generate_answer=False,
            exact_revisit_input=effective_exact_input,
            intent_override=None,
            followup_queries_override=None,
            frozen_plan=None,
            query_embeddings_override=None,
            query_vector_bundle=None,
            strict_vector_bundle=False,
            contextual_domain=contextual_domain,
            contextual_endpoint_limit=None,
            contextual_evaluation_as_of=None,
            contextual_revisit_scope_hash=contextual_revisit_scope_hash,
            trace_bridge=None,
            deadline_seconds=deadline_seconds,
            deadline_at=deadline_at,
            expected_runtime_manifest_id=(
                automatic_revisit.manifest_id
                if automatic_revisit is not None
                else (
                    restricted_rewrite.manifest_id
                    if restricted_rewrite is not None
                    else None
                )
            ),
            expected_runtime_manifest_fingerprint=(
                automatic_revisit.manifest_fingerprint
                if automatic_revisit is not None
                else (
                    restricted_rewrite.manifest_fingerprint
                    if restricted_rewrite is not None
                    else None
                )
            ),
            restricted_rewrite=restricted_rewrite,
            _emit_contextual_event=False,
        )

    def try_contextual_revisit(
        self,
        question: str,
        *,
        contextual_domain: str,
        contextual_revisit_scope_hash: str,
        deadline_seconds: float | None = None,
    ) -> dict | None:
        """Return a verified automatic contextual-revisit result, or ``None``.

        Unlike :meth:`query`, this public preflight never runs ordinary
        planning, embedding, reranking, answer generation, trace emission,
        or contextual learning after a miss.  It only attempts the existing
        exact and restricted-rewrite automatic routes against current local
        state.  Deadline handling intentionally uses the same absolute budget
        convention as ``query`` so a slow local manifest read cannot restart
        the caller's budget before validation.
        """

        # This public boundary must not expose request-local diagnostics or a
        # selector capture from a preceding ordinary query when its local
        # preflight is a miss.  It never creates a learning capture itself.
        self._v3_learning_capture = None
        last_query_embeddings = getattr(self, "last_query_embeddings", None)
        if isinstance(last_query_embeddings, dict):
            last_query_embeddings.clear()
        else:
            self.last_query_embeddings = {}
        self.last_query_embedding_cache_trace = {"hits": [], "misses": []}
        deadline_at = (
            monotonic() + max(0.001, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        )
        with self._provider_deadline_scope(deadline_at):
            return self._try_contextual_revisit_only(
                question,
                contextual_domain=contextual_domain,
                contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                deadline_seconds=deadline_seconds,
                deadline_at=deadline_at,
            )

    def query(
        self,
        question: str,
        *,
        generate_answer: bool = True,
        stop_after: Literal["evidence"] | None = None,
        intent_override: QueryIntent | dict | None = None,
        followup_queries_override: list[str] | None = None,
        frozen_plan: dict | None = None,
        deadline_seconds: float | None = None,
        query_embeddings_override: dict[str, np.ndarray] | None = None,
        query_vector_bundle: QueryVectorBundle | None = None,
        strict_vector_bundle: bool = False,
        contextual_domain: str | None = None,
        contextual_endpoint_limit: int | None = None,
        contextual_evaluation_as_of: str | None = None,
        contextual_revisit_scope_hash: str | None = None,
        trace_bridge: QueryTraceBridge | None = None,
        contextual_learning: bool = False,
        learning_request_id: str | None = None,
        exact_revisit_input: ExactRevisitInput | None = None,
        warm_episode_activations: dict[int, float] | None = None,
    ) -> dict:
        """Execute a query, optionally finalizing a strict source-bound V3 plan.

        The caller is responsible for constructing the bridge with real
        database/config/permission hashes.  The regular query API remains
        unchanged when no bridge is supplied; the legacy ``JsonlEventLogger``
        is not substituted or repurposed.  Durable contextual learning is
        independently opt-in: it requires both ``contextual_learning=True``
        and a stable caller-provided ``learning_request_id``.  Direct engine
        users with no application finalizer remain read-only.

        ``stop_after="evidence"`` is the public, read-only retrieval boundary
        used by diagnostic Q2 comparisons.  It materializes the same selected
        Source excerpts that a normal answer would receive, but it never
        generates/audits an answer, grows an association, marks an edge used,
        or enters the learning finalizer.  ``generate_answer=False`` retains
        its historical retrieval-only semantics; callers that need the
        stronger no-write contract must use ``stop_after`` explicitly.

        ``exact_revisit_input`` is an independently opt-in transient V3
        contract.  It can skip planning only after rerunning local matcher,
        target/source provenance, and contribution-selector checks.

        ``warm_episode_activations`` carries up to 32 bounded Episode search
        hints from an earlier retrieval round.  It bypasses automatic exact
        revisit so the new question can traverse the graph with both fresh
        and retained activation.  The activation trace is not evidence.
        """

        if stop_after not in (None, "evidence"):
            raise ValueError("stop_after must be None or 'evidence'")
        warm_episode_activations = self._normalize_warm_episode_activations(
            warm_episode_activations
        )
        if warm_episode_activations is not None and exact_revisit_input is not None:
            raise ValueError(
                "warm_episode_activations cannot be combined with exact_revisit_input"
            )
        evidence_only = stop_after == "evidence"
        if evidence_only:
            if contextual_learning or learning_request_id is not None:
                raise ValueError(
                    "evidence-only query cannot enable contextual learning"
                )
            # ``stop_after`` owns the answer boundary.  This intentionally
            # overrides the public default of ``generate_answer=True`` so
            # ``query(question, stop_after='evidence')`` is safe by default.
            generate_answer = False

        # QueryEngine is request-local in production but many focused tests
        # reuse it.  Never let a selector capture from a prior request become
        # input to this request's finalizer.
        self._v3_learning_capture = None
        # The automatic exact routes can return before ``_query_impl`` gets a
        # chance to reset request diagnostics.  Clear them at the public
        # boundary as well, so a private automatic hit never leaves a prior
        # request's vectors or cache trace observable on a reused engine.
        # A few boundary-only tests deliberately build a minimal engine via
        # ``object.__new__`` so they can exercise trace failure semantics
        # without initializing indexes.  Preserve that supported boundary:
        # diagnostics are optional state at entry, but are always reset to an
        # empty mapping before either an automatic route or ordinary query
        # work can observe them.
        last_query_embeddings = getattr(self, "last_query_embeddings", None)
        if isinstance(last_query_embeddings, dict):
            last_query_embeddings.clear()
        else:
            self.last_query_embeddings = {}
        self.last_query_embedding_cache_trace = {"hits": [], "misses": []}
        # An exact probe (including the private V17 automatic probe) may spend
        # part of the caller's budget before it falls through.  Pin one
        # absolute deadline at the public boundary so neither a slow miss nor
        # a manifest lookup can silently restart the timer for ordinary
        # retrieval.  This is also harmless for a request with no eligible
        # exact route: ``_query_impl`` receives the same absolute deadline it
        # would otherwise derive immediately after entry.
        exact_request_deadline_at = (
            monotonic() + max(0.001, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        )

        def ensure_public_deadline(stage: str) -> None:
            """Reject a late result before it crosses any write boundary.

            Provider clients can discard a late HTTP response, but local
            association commits and the V3 learning finalizer happen after
            that client call returns.  They must use the same absolute
            deadline rather than treating a successful model response as
            permission to write after the request has expired.
            """

            if (
                exact_request_deadline_at is not None
                and monotonic() >= exact_request_deadline_at
            ):
                raise TimeoutError(f"query deadline exceeded before {stage}")

        def execute_impl(
            *,
            allow_association_learning: bool = True,
            record_association_use: bool = True,
        ) -> dict:
            return self._query_impl(
                question,
                generate_answer=generate_answer,
                intent_override=intent_override,
                followup_queries_override=followup_queries_override,
                frozen_plan=frozen_plan,
                deadline_seconds=deadline_seconds,
                query_embeddings_override=query_embeddings_override,
                query_vector_bundle=query_vector_bundle,
                strict_vector_bundle=strict_vector_bundle,
                contextual_domain=contextual_domain,
                contextual_endpoint_limit=contextual_endpoint_limit,
                contextual_evaluation_as_of=contextual_evaluation_as_of,
                allow_association_learning=allow_association_learning,
                record_association_use=record_association_use,
                trace_bridge=trace_bridge,
                request_deadline_at=exact_request_deadline_at,
                warm_episode_activations=warm_episode_activations,
            )

        def execute() -> dict:
            # Probe before ordinary growth staging creates a mutable overlay.
            # A caller-supplied ticket retains its existing behavior.  Without
            # one, V17 may privately reconstruct a single exact input from a
            # ready scope-bound manifest.  Every invalid/incomplete condition
            # simply falls through to normal planner/retrieval below; no
            # ticket or manifest identity enters the public result.
            automatic_revisit = (
                self._v17_automatic_exact_revisit(
                    question,
                    intent_override=intent_override,
                    followup_queries_override=followup_queries_override,
                    frozen_plan=frozen_plan,
                    query_embeddings_override=query_embeddings_override,
                    query_vector_bundle=query_vector_bundle,
                    strict_vector_bundle=strict_vector_bundle,
                    contextual_domain=contextual_domain,
                    contextual_endpoint_limit=contextual_endpoint_limit,
                    contextual_evaluation_as_of=contextual_evaluation_as_of,
                    contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                    trace_bridge=trace_bridge,
                    deadline_at=exact_request_deadline_at,
                )
                if exact_revisit_input is None and warm_episode_activations is None
                else None
            )
            restricted_rewrite = (
                self._t16_automatic_restricted_rewrite_revisit(
                    question,
                    intent_override=intent_override,
                    followup_queries_override=followup_queries_override,
                    frozen_plan=frozen_plan,
                    query_embeddings_override=query_embeddings_override,
                    query_vector_bundle=query_vector_bundle,
                    strict_vector_bundle=strict_vector_bundle,
                    contextual_domain=contextual_domain,
                    contextual_endpoint_limit=contextual_endpoint_limit,
                    contextual_evaluation_as_of=contextual_evaluation_as_of,
                    contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                    trace_bridge=trace_bridge,
                    deadline_at=exact_request_deadline_at,
                )
                if exact_revisit_input is None
                and automatic_revisit is None
                and warm_episode_activations is None
                else None
            )
            effective_exact_input = (
                automatic_revisit.revisit_input
                if automatic_revisit is not None
                else (
                    restricted_rewrite.revisit_input
                    if restricted_rewrite is not None
                    else exact_revisit_input
                )
            )
            exact_result = self._try_exact_revisit_v3(
                question,
                generate_answer=generate_answer,
                exact_revisit_input=effective_exact_input,
                intent_override=intent_override,
                followup_queries_override=followup_queries_override,
                frozen_plan=frozen_plan,
                query_embeddings_override=query_embeddings_override,
                query_vector_bundle=query_vector_bundle,
                strict_vector_bundle=strict_vector_bundle,
                contextual_domain=contextual_domain,
                contextual_endpoint_limit=contextual_endpoint_limit,
                contextual_evaluation_as_of=contextual_evaluation_as_of,
                contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                trace_bridge=trace_bridge,
                deadline_seconds=deadline_seconds,
                deadline_at=exact_request_deadline_at,
                expected_runtime_manifest_id=(
                    automatic_revisit.manifest_id
                    if automatic_revisit is not None
                    else (
                        restricted_rewrite.manifest_id
                        if restricted_rewrite is not None
                        else None
                    )
                ),
                expected_runtime_manifest_fingerprint=(
                    automatic_revisit.manifest_fingerprint
                    if automatic_revisit is not None
                    else (
                        restricted_rewrite.manifest_fingerprint
                        if restricted_rewrite is not None
                        else None
                    )
                ),
                restricted_rewrite=restricted_rewrite,
            ) if warm_episode_activations is None else None
            if exact_result is not None:
                return exact_result
            if evidence_only:
                # Do not create a staging overlay merely to commit an empty
                # result.  The implementation still executes the production
                # retrieval/selection path, but every business mutation is
                # disabled at its narrow owning boundary.
                result = execute_impl(
                    allow_association_learning=False,
                    record_association_use=False,
                )
                result.setdefault(
                    "growth_staging",
                    {
                        "enabled": False,
                        "committed": False,
                        "reason": "evidence_only_read_only",
                        "temporary_to_durable_ids": {},
                    },
                )
                return result
            if trace_bridge is not None:
                return self._execute_strict_trace_read_only(execute_impl)
            if (
                self.config.retrieval.growth_max_rounds <= 0
                or not self.config.retrieval.growth_staging_enabled
                or isinstance(self.associations, StagedAssociationOverlay)
            ):
                result = execute_impl()
                result.setdefault(
                    "growth_staging",
                    {
                        "enabled": False,
                        "committed": False,
                        "temporary_to_durable_ids": {},
                    },
                )
                return result

            durable_associations = self.associations
            durable_traverser = self.traverser
            durable_growth = self.growth
            durable_chronology = self.chronology
            staged = StagedAssociationOverlay(durable_associations)
            self.associations = staged
            self.traverser = GraphTraverser(staged)
            self.growth = AssociationGrowthEngine(
                self.model, staged, self.episodes, self.config.weights, self.logger
            )
            self.chronology = ChronologyService(self.episodes, staged, self.logger)
            try:
                result = execute_impl()
                ensure_public_deadline("association_growth_commit")
                mapping = staged.commit(
                    require_live=lambda: ensure_public_deadline(
                        "association_growth_commit"
                    )
                )
                return self._remap_staged_result(result, mapping)
            finally:
                self.associations = durable_associations
                self.traverser = durable_traverser
                self.growth = durable_growth
                self.chronology = durable_chronology

        if trace_bridge is not None:
            trace_bridge.start()
        binding = (
            trace_bridge.bind_model(self.model)
            if trace_bridge is not None
            else nullcontext()
        )
        provider_deadline_binding = self._provider_deadline_scope(
            exact_request_deadline_at
        )
        try:
            with binding:
                with provider_deadline_binding:
                    with transaction_liveness(
                        lambda: ensure_public_deadline("sqlite_commit")
                    ):
                        result = execute()
            selected_evidence_runtime = result.pop(
                "_runtime_selected_evidence", ()
            )
            if evidence_only:
                result["execution_profile"] = (
                    "input_frozen"
                    if bool(result.get("query_plan_frozen", False))
                    else "live_evidence"
                )
                result["evidence_result"] = self._evidence_only_result_payload(
                    result,
                    selected_evidence_runtime,
                )
            # A provider can return successfully just after its deadline.
            # Check again before exposing an answer or executing any durable
            # learning path outside the provider context manager.
            ensure_public_deadline("post_query_result")
            exact_revisit_hit = bool(
                isinstance(result.get("exact_revisit"), Mapping)
                and result["exact_revisit"].get("status") == "hit"
            )
            if exact_revisit_hit and (
                contextual_learning or learning_request_id is not None
            ):
                # Exact revisit is deliberately a zero-write retrieval action.
                # Do not let an unrelated learning opt-in turn it into a
                # second post-answer transaction.
                result["contextual_learning"] = self._v3_learning_report(
                    "skipped", "exact_revisit_read_only"
                )
                self._v3_learning_capture = None
            elif contextual_learning or learning_request_id is not None:
                try:
                    ensure_public_deadline("contextual_learning_finalization")
                    with transaction_liveness(
                        lambda: ensure_public_deadline(
                            "contextual_learning_commit"
                        )
                    ):
                        result["contextual_learning"] = (
                            self._finalize_v3_query_learning(
                                question,
                                result,
                                enabled=bool(contextual_learning),
                                learning_request_id=learning_request_id,
                                generate_answer=generate_answer,
                                frozen_plan=frozen_plan,
                                strict_vector_bundle=strict_vector_bundle,
                                contextual_domain=contextual_domain,
                                contextual_revisit_scope_hash=contextual_revisit_scope_hash,
                                trace_bridge=trace_bridge,
                            )
                        )
                finally:
                    # The callback receives the only supported route to
                    # runtime vectors/source facts.  Discard it before the
                    # result leaves this boundary, irrespective of an error.
                    self._v3_learning_capture = None
            else:
                self._v3_learning_capture = None
            ensure_public_deadline("query_return")
            if trace_bridge is not None:
                trace_bridge.complete_success()
            return result
        except BaseException as exc:
            # A failed or expired query must not leave request-local
            # selector/source objects reachable by a later reused test or
            # service engine, and it must never enter the finalizer.
            self._v3_learning_capture = None
            if trace_bridge is not None:
                trace_bridge.complete_failure(exc)
            raise

    def _query_impl(
        self,
        question: str,
        *,
        generate_answer: bool = True,
        intent_override: QueryIntent | dict | None = None,
        followup_queries_override: list[str] | None = None,
        frozen_plan: dict | None = None,
        deadline_seconds: float | None = None,
        query_embeddings_override: dict[str, np.ndarray] | None = None,
        query_vector_bundle: QueryVectorBundle | None = None,
        strict_vector_bundle: bool = False,
        contextual_domain: str | None = None,
        contextual_endpoint_limit: int | None = None,
        contextual_evaluation_as_of: str | None = None,
        allow_association_learning: bool = True,
        record_association_use: bool = True,
        trace_bridge: QueryTraceBridge | None = None,
        request_deadline_at: float | None = None,
        warm_episode_activations: dict[int, float] | None = None,
    ) -> dict:
        self.last_query_embeddings.clear()
        self.last_query_embedding_cache_trace = {"hits": [], "misses": []}
        query_started_at = perf_counter()
        deadline_at = (
            request_deadline_at
            if request_deadline_at is not None
            else (
                monotonic() + max(0.001, float(deadline_seconds))
                if deadline_seconds is not None
                else None
            )
        )

        def ensure_deadline(stage: str) -> None:
            if deadline_at is not None and monotonic() >= deadline_at:
                raise TimeoutError(f"query deadline exceeded before {stage}")
        phase_seconds = {
            name: 0.0
            for name in (
                "intent_parse",
                "initial_embedding",
                "initial_retrieval",
                "initial_graph_expansion",
                "prepared_early_contextual",
                "contextual_association",
                "followup_planning",
                "followup_embedding",
                "followup_retrieval",
                "followup_graph_expansion",
                "source_cohort",
                "evidence_preparation",
                "evidence_rerank",
                "association_growth",
                "final_selection",
                "growth_utility",
                "answer_generation",
            )
        }
        if strict_vector_bundle and query_embeddings_override:
            # A legacy string→vector map does not carry an embedding-space
            # contract or logical slot identity, so replay must reject it even
            # if a frozen plan would otherwise avoid looking at the map.
            raise ValueError(
                "strict vector replay does not allow legacy query_embeddings_override"
            )
        if frozen_plan is not None:
            if intent_override is not None or followup_queries_override is not None:
                raise ValueError(
                    "frozen_plan cannot be combined with intent/follow-up overrides"
                )
            self._validate_query_plan(question, frozen_plan)
        # A normal request freezes exactly one UTC instant before retrieval so
        # contextual edge expiry, target version checks, and later trace data
        # cannot observe a moving wall clock. Frozen/replay callers may supply
        # their own already-pinned value (or preserve the one in their plan);
        # legacy frozen plans still bypass the v3 contextual lane below.
        frozen_evaluation_as_of = (
            frozen_plan.get("contextual_evaluation_as_of", frozen_plan.get("evaluation_as_of"))
            if frozen_plan is not None
            else None
        )
        requested_contextual_as_of = (
            contextual_evaluation_as_of
            if contextual_evaluation_as_of is not None
            else frozen_evaluation_as_of
        )
        request_contextual_evaluation_as_of = (
            self._resolve_contextual_evaluation_as_of(requested_contextual_as_of)
            if (frozen_plan is None or requested_contextual_as_of is not None)
            else None
        )
        if self.logger:
            self.logger.emit(
                "query_started",
                question=question,
                intent_overridden=intent_override is not None,
                followup_queries_overridden=followup_queries_override is not None,
                frozen_query_plan_id=(
                    str(frozen_plan["plan_id"]) if frozen_plan is not None else None
                ),
            )
        phase_started = perf_counter()
        ensure_deadline("intent_parse")
        if frozen_plan is not None:
            intent = QueryIntent.from_dict(dict(frozen_plan["intent"]))
        elif isinstance(intent_override, QueryIntent):
            intent = intent_override
        elif isinstance(intent_override, dict):
            intent = QueryIntent.from_dict(intent_override)
        else:
            intent = self._parse_intent(question)
        # An intent override freezes stochastic model planning, not the current
        # deterministic retrieval policy.  Re-apply structural answer slots so
        # controlled A/B runs and production overrides do not silently bypass
        # newer evidence protections.  A full frozen_plan remains byte-stable.
        if frozen_plan is None:
            intent.search_queries = list(
                dict.fromkeys(
                    [
                        *intent.search_queries,
                        *structural_queries(question, intent),
                    ]
                )
            )
        # Freeze the request's evidence obligations before the first vector
        # operation.  Planner atomic questions and deterministic structural
        # constraints may be requirements; the whole question, later follow-up
        # queries, rerank expansion, and contextual matching remain discovery
        # hints only.  In particular, no candidate record is available to this
        # resolver, so an empty candidate pool cannot shrink the denominator.
        if frozen_plan is not None:
            requirement_planner_origin = "persisted"
        elif intent_override is not None:
            requirement_planner_origin = "explicit"
        elif callable(getattr(self.model, "trace_request", None)):
            requirement_planner_origin = "cloud"
        else:
            requirement_planner_origin = "local"
        # Discovery may use many paraphrases, but a singleton controlled run
        # must freeze the *actual* raw atomic query that the reranker uses.
        # Taking planner item zero instead is not equivalent: the planner may
        # lead with a paraphrase, while the reranker maps its coverage to the
        # user question. That mismatch produces an artificial unresolved
        # requirement despite source-bound coverage. The original ``intent``
        # remains untouched below for vector/sparse retrieval, so this never
        # silently narrows recall. Normal multi-slot configurations retain
        # their existing planner-prefix behavior.
        rerank_atomic_limit = int(self.config.retrieval.rerank_atomic_query_limit)
        requirement_queries = (
            [question]
            if rerank_atomic_limit == 1
            else limit_rerank_atomic_queries(
                list(intent.search_queries),
                rerank_atomic_limit,
            )
        )
        requirement_intent = replace(
            intent,
            search_queries=requirement_queries,
        )
        authoritative_requirements = resolve_authoritative_requirements(
            question,
            requirement_intent,
            request_mode=infer_request_mode(question, intent),
            planner_origin=requirement_planner_origin,
        )
        requirements_trace_event_id: str | None = None
        if trace_bridge is not None:
            # This is the first v3 stage the engine has actually observed:
            # requirements are frozen before any vector/candidate operation.
            requirements_trace_event_id = self._emit_trace_requirements_resolved(
                trace_bridge,
                authoritative_requirements,
            )
        self._record_phase(phase_seconds, "intent_parse", phase_started)
        supplied_query_vector_bundle = query_vector_bundle
        embedding_coordinator = self._new_query_embedding_coordinator()
        source_request_hash = "sha256:" + self._request_vector_hash(question)
        initial_vector_specs: list[QueryVectorRequest] = []
        initial_vector_trace_checkpoint = (
            trace_bridge.checkpoint() if trace_bridge is not None else None
        )
        initial_vector_origins: dict[str, str] = {}
        initial_persisted_text_hashes: set[str] = set()
        if supplied_query_vector_bundle is not None:
            initial_vector_origins.update(
                {
                    str(item.physical_id): "persisted"
                    for item in supplied_query_vector_bundle.physical_vectors
                }
            )
            initial_persisted_text_hashes.update(
                str(item.text_hash)
                for item in supplied_query_vector_bundle.physical_vectors
            )
        # Historical direct overrides have no durable provider receipt.  The
        # bundle still records them explicitly as request-cache material when
        # a normal (non-strict) query uses that compatibility route.
        legacy_override_hashes = {
            self._request_vector_hash(str(text))
            for text in (query_embeddings_override or {})
            if normalize_query_text(str(text))
        }

        vector_trace_event_id: str | None = None
        vector_trace_origins: dict[str, str] = {}
        vector_trace_stage_index = 0

        def emit_vector_bundle_if_observed(
            bundle: QueryVectorBundle | None,
            *,
            before_observations: QueryTraceObservationCheckpoint | None,
            inherited_origins: dict[str, str],
            extra_provider_parent_event_ids: Sequence[str] = (),
        ) -> None:
            nonlocal vector_trace_event_id, vector_trace_origins, vector_trace_stage_index
            if trace_bridge is None or bundle is None:
                return
            if requirements_trace_event_id is None or before_observations is None:
                raise TracePersistenceError(
                    "strict trace vector stage lacks a requirements observation"
                )
            origins = dict(inherited_origins)
            for physical in bundle.physical_vectors:
                if (
                    str(physical.text_hash) in legacy_override_hashes
                    and str(physical.physical_id) not in origins
                ):
                    origins[str(physical.physical_id)] = "request_cache"
                if str(physical.text_hash) in initial_persisted_text_hashes:
                    origins[str(physical.physical_id)] = "persisted"
            provider_delta = trace_bridge.observations_since(before_observations)
            default_origin = (
                "provider"
                if int(provider_delta.embedding_logical_batches) > 0
                else "local_encoder"
            )
            for physical in bundle.physical_vectors:
                origins.setdefault(str(physical.physical_id), default_origin)
            vector_trace_event_id = self._emit_trace_vector_bundle_ready(
                trace_bridge,
                bundle,
                causal_parent_event_id=(
                    vector_trace_event_id or requirements_trace_event_id
                ),
                before_vector_observations=before_observations,
                physical_origins=origins,
                strict_single_batch=strict_vector_bundle,
                stage_index=vector_trace_stage_index,
                extra_provider_parent_event_ids=extra_provider_parent_event_ids,
            )
            vector_trace_origins = origins
            vector_trace_stage_index += 1

        frozen_followup_seed_hits: list[SearchHit] = []
        frozen_source_cohort_seed_hits: list[SearchHit] = []
        frozen_final_seed_hits: list[SearchHit] = []
        frozen_seed_stage_replay: dict[str, object] = {
            "classification": "live_request",
            "initial_seed_source": "live_initial_retrieval",
            "followup_seed_source": "live_followup_retrieval",
            "source_cohort_seed_source": "live_source_cohort_retrieval",
        }
        if frozen_plan is not None:
            initial_queries = [str(value) for value in frozen_plan["initial_queries"]]
            initial_episode_anchor_ids = [
                int(value)
                for value in frozen_plan.get("initial_episode_anchor_ids", [])
            ]
            frozen_final_seed_hits = self._frozen_search_hits(
                frozen_plan.get("final_seed_hits")
            )
            initial_payload_present = isinstance(
                frozen_plan.get("initial_seed_hits"), list
            )
            initial_seed_hits = self._frozen_search_hits(
                frozen_plan.get("initial_seed_hits")
            )
            frozen_followup_seed_hits = self._frozen_search_hits(
                frozen_plan.get("followup_seed_hits")
            )
            source_cohort_payload_present = isinstance(
                frozen_plan.get("source_cohort_seed_hits"), list
            )
            frozen_source_cohort_seed_hits = self._frozen_search_hits(
                frozen_plan.get("source_cohort_seed_hits")
            )
            if initial_payload_present:
                seeds = initial_seed_hits
                frozen_seed_stage_replay = {
                    "classification": (
                        "stage_complete_frozen_replay"
                        if source_cohort_payload_present
                        else "initial_and_followup_saved_final_delta_reconstructed"
                    ),
                    "initial_seed_source": "saved_initial_seed_hits",
                    "initial_seed_count": len(initial_seed_hits),
                    "initial_seed_hit_sources": deepcopy(
                        frozen_plan.get("initial_seed_hit_sources", [])
                    ),
                    "followup_seed_source": "saved_followup_seed_hits",
                    "followup_seed_count": len(frozen_followup_seed_hits),
                    "source_cohort_seed_source": (
                        "saved_source_cohort_seed_hits"
                        if source_cohort_payload_present
                        else "reconstructed_final_seed_delta_not_historical_stage"
                    ),
                    "source_cohort_seed_count": len(frozen_source_cohort_seed_hits),
                    "final_seed_count": len(frozen_final_seed_hits),
                }
            else:
                # Historical plans that stored only final seeds remain useful
                # for legacy reproduction, but must never be cited as an
                # independent prepared-early input or early-work saving.
                seeds = frozen_final_seed_hits
                frozen_seed_stage_replay = {
                    "classification": "legacy_final_input_replay",
                    "initial_seed_source": "not_observed_final_seed_fallback",
                    "initial_seed_count": 0,
                    "followup_seed_source": "not_observed",
                    "followup_seed_count": 0,
                    "source_cohort_seed_source": "not_observed",
                    "source_cohort_seed_count": 0,
                    "final_seed_count": len(frozen_final_seed_hits),
                }
            active_cue_ids = [
                int(value)
                for value in frozen_plan.get("association_cue_association_ids", [])
            ]
            active_cue_entries = [
                dict(item)
                for item in frozen_plan.get("association_cue_entries", [])
            ]
            initial_rankings = deepcopy(frozen_plan.get("initial_rankings", {}))
            episode_anchor_ids = [
                int(value) for value in frozen_plan.get("episode_anchor_ids", [])
            ]
            emit_vector_bundle_if_observed(
                query_vector_bundle,
                before_observations=initial_vector_trace_checkpoint,
                inherited_origins=initial_vector_origins,
            )
        else:
            initial_episode_anchor_ids = []
            initial_queries = list(dict.fromkeys([question, *intent.search_queries]))
            initial_vector_specs = [
                *self._request_vector_specs(
                    [question],
                    role="whole",
                    authoritative_requirements=authoritative_requirements,
                ),
                *self._request_vector_specs(
                    intent.search_queries,
                    role="atomic",
                    authoritative_requirements=authoritative_requirements,
                ),
            ]
            initial_embedding_started = perf_counter()
            query_vector_bundle = self._prepare_request_vector_bundle(
                embedding_coordinator,
                initial_vector_specs,
                source_bundles=(
                    (supplied_query_vector_bundle,)
                    if supplied_query_vector_bundle is not None
                    else ()
                ),
                legacy_overrides=query_embeddings_override,
                provider_purpose="initial_vector_retrieval",
                source_request_hash=source_request_hash,
                strict_vector_bundle=strict_vector_bundle,
            )
            self._record_phase(
                phase_seconds,
                "initial_embedding",
                initial_embedding_started,
            )
            emit_vector_bundle_if_observed(
                query_vector_bundle,
                before_observations=initial_vector_trace_checkpoint,
                inherited_origins=initial_vector_origins,
            )
            initial_vector_overrides = self._bundle_overrides_for_queries(
                query_vector_bundle,
                initial_queries,
            )
            seeds, active_cue_ids, active_cue_entries, initial_rankings = (
                self._vector_seed_hits_with_cues(
                    initial_queries,
                    initial_episode_anchor_ids,
                    self.config.retrieval.answer_whole_question_anchor_episodes,
                    question,
                    phase_seconds=phase_seconds,
                    timing_prefix="initial",
                    query_embeddings_override=initial_vector_overrides,
                    provider_purpose="initial_vector_retrieval",
                )
            )
            initial_finalize_started = perf_counter()
            for entity in intent.target_entities:
                for row in self.concepts.find_by_alias(entity):
                    seeds.append(
                        SearchHit(
                            "concept",
                            int(row["canonical_concept_id"] or row["id"]),
                            1.0,
                        )
                    )
            seeds = self._merge_hits(seeds)
            episode_anchor_ids = list(initial_episode_anchor_ids)
            self._record_phase(
                phase_seconds,
                "initial_retrieval",
                initial_finalize_started,
            )
        # Carry prior-round attention into the same bounded graph traversal as
        # fresh vector/sparse seeds.  Keep the fresh snapshot separate: warm
        # Episode IDs are search hints, not independent current-query anchors
        # for contextual provenance or learning.
        fresh_query_seeds = list(seeds)
        if warm_episode_activations is not None:
            seeds = self._merge_hits(
                seeds,
                [
                    SearchHit("episode", episode_id, score)
                    for episode_id, score in sorted(
                        warm_episode_activations.items()
                    )
                ],
            )
        contextual_trace = {
            "enabled": False,
            "backend": "contextual_double_key",
            "evaluation_as_of": request_contextual_evaluation_as_of,
            "context_hits": [],
            "need_hits": [],
            "attached_edges": [],
            "attached_episode_ids": [],
            "external_calls": 0,
        }
        # Keep a request-local provenance split.  It is deliberately plain
        # metadata (no vectors) so it is safe to place in the trace and to
        # hand to the background plasticity worker.
        base_episode_ids_before_contextual = sorted(
            {
                int(item.node_id)
                for item in fresh_query_seeds
                if item.node_type == "episode"
            }
        )
        # Keep strict Source closures request-local. Metadata validation may
        # cheaply inspect a candidate first, but only this cache can make a
        # candidate eligible to alter the ordinary seed pool.
        prepared_early_source_cache: dict[
            int, tuple[SourceFactRef | None, str]
        ] = {}
        base_seeds_before_prepared_early = list(seeds)
        base_anchor_activations = self._base_anchor_activations(fresh_query_seeds)
        # A relation-cue endpoint may participate in ordinary retrieval, but
        # it is not an independent anchor for a prepared contextual proposal.
        # Compute the exclusion before the broad traversal so both the early
        # observation and the later residual selector use the same rule.
        cue_anchor_exclusions = self._association_cue_episode_ids(
            active_cue_entries
        )
        if query_vector_bundle is not None:
            contextual_trace.update(
                self._query_vector_trace_metadata(query_vector_bundle)
            )
            # The ordinary treatment remains deferred until masked residual
            # selection below.  The separately configured W06 shadow probe
            # observes only a pre-target proposal and never seeds traversal.
            contextual_trace["deferred_until_residual_slots"] = True
        prepared_early_started = perf_counter()
        ensure_deadline("prepared_early_contextual")
        prepared_early_trace = self._prepared_early_contextual_proposal(
            bundle=query_vector_bundle,
            domain=contextual_domain,
            endpoint_limit=contextual_endpoint_limit,
            anchor_activations=base_anchor_activations,
            authoritative_requirements=authoritative_requirements,
            evaluation_as_of=request_contextual_evaluation_as_of,
            cue_endpoint_episode_ids=cue_anchor_exclusions,
        )
        prepared_early_trace["input_stage_replay"] = frozen_seed_stage_replay
        # The experimental candidate-pool lane is deliberately tiny: a
        # source-validated, non-base prepared target may seed the *ordinary*
        # graph traversal. It remains a contextual provenance route, is not
        # an answer or an early stop, and is excluded from base-route
        # attribution below. This prevents an edge-origin candidate from
        # becoming an apparent independent retrieval result merely because it
        # participated in traversal.
        prepared_early_contextual_episode_ids: tuple[int, ...] = ()
        prepared_early_contextual_derived_episode_ids: tuple[int, ...] = ()
        prepared_early_root_edge_ids_by_episode: dict[int, tuple[int, ...]] = {}
        injected_hits: list[SearchHit] = []
        if bool(
            self.config.retrieval.contextual_prepared_early_candidate_pool_enabled
        ):
            accepted_candidates = prepared_early_trace.get(
                "accepted_candidate_order", []
            )
            if not isinstance(accepted_candidates, list):
                accepted_candidates = []
            target_ids: list[int] = []
            for candidate in accepted_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                try:
                    target_episode_id = int(candidate.get("target_episode_id", 0))
                except (TypeError, ValueError):
                    continue
                if target_episode_id > 0:
                    target_ids.append(target_episode_id)
            strict_facts, strict_reasons = self._v3_source_fact_closure(
                target_ids,
                cache=prepared_early_source_cache,
            )
            strict_bound_ids = set(strict_facts)
            prepared_early_trace["source_binding_status"] = (
                "strict_source_bound"
                if target_ids and all(target_id in strict_bound_ids for target_id in target_ids)
                else "strict_source_binding_rejected"
                if target_ids
                else "no_metadata_validated_candidate"
            )
            prepared_early_trace["source_binding_validated"] = bool(
                target_ids and all(target_id in strict_bound_ids for target_id in target_ids)
            )
            prepared_early_trace["candidate_pool_strict_source_status"] = [
                {
                    "episode_id": target_episode_id,
                    "status": strict_reasons.get(
                        target_episode_id, "source_closure_missing"
                    ),
                    "source_revision_id": (
                        strict_facts[target_episode_id].source_revision_id
                        if target_episode_id in strict_facts
                        else ""
                    ),
                }
                for target_episode_id in sorted(set(target_ids))
            ]
            strict_candidates: list[Mapping[str, object]] = []
            for candidate in accepted_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                try:
                    target_episode_id = int(candidate.get("target_episode_id", 0))
                except (TypeError, ValueError):
                    continue
                if target_episode_id in strict_bound_ids:
                    strict_candidates.append(candidate)
            injected_hits, prepared_early_contextual_episode_ids = (
                self._prepared_early_candidate_pool_hits(
                    strict_candidates,
                    set(base_episode_ids_before_contextual),
                )
            )
            eligible_ids = list(prepared_early_contextual_episode_ids)
            root_edge_sets: dict[int, set[int]] = {
                episode_id: set() for episode_id in eligible_ids
            }
            for candidate in strict_candidates:
                try:
                    target_episode_id = int(candidate.get("target_episode_id", 0))
                    association_id = int(candidate.get("association_id", 0))
                except (TypeError, ValueError):
                    continue
                if target_episode_id in root_edge_sets and association_id > 0:
                    root_edge_sets[target_episode_id].add(association_id)
            prepared_early_root_edge_ids_by_episode = {
                episode_id: tuple(sorted(edge_ids))
                for episode_id, edge_ids in sorted(root_edge_sets.items())
            }
            prepared_early_trace["candidate_pool_eligible_episode_ids"] = (
                eligible_ids
            )
            if injected_hits:
                seeds = self._merge_hits(seeds, injected_hits)
                prepared_early_trace["candidate_pool_mutated"] = True
                prepared_early_trace["candidate_pool_injected_episode_ids"] = (
                    eligible_ids
                )
                prepared_early_trace["candidate_pool_injection_reason"] = (
                    "source_validated_nonbase_prepared_candidates_seeded"
                )
                prepared_early_trace["executed_modules"] = {
                    **dict(prepared_early_trace.get("executed_modules", {})),
                    "prepared_early_candidate_pool": True,
                }
            elif prepared_early_trace.get("source_metadata_validated"):
                prepared_early_trace["candidate_pool_injection_reason"] = (
                    "strict_source_bound_targets_already_independent_base_anchors"
                    if strict_bound_ids
                    else "metadata_validated_but_strict_source_binding_rejected"
                )
            else:
                prepared_early_trace["candidate_pool_injection_reason"] = (
                    "no_source_validated_prepared_candidate"
                )
        contextual_trace["prepared_early"] = prepared_early_trace
        self._record_phase(
            phase_seconds,
            "prepared_early_contextual",
            prepared_early_started,
        )
        all_new_associations: list[int] = []
        all_reinforced_associations: list[int] = []
        reinforced_before: dict[int, dict] = {}
        seen_growth_fingerprints: set[tuple] = set()
        phase_started = perf_counter()
        ensure_deadline("initial_graph_expansion")
        traversed, paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        cue_scores = {
            int(item["association_id"]): float(item.get("cosine", 0.0))
            for item in active_cue_entries
        }
        paths.extend(
            self._explicit_association_paths(active_cue_ids, cue_scores)
        )
        cue_fast_endpoint_keys = self._association_cue_fast_endpoint_keys(
            active_cue_entries
        )
        traversed = self._truncate_traversed_nodes(
            traversed,
            self.config.retrieval.candidate_limit,
            cue_fast_endpoint_keys,
        )
        # This bounded readout is request-local attention for a later gap
        # query.  It includes propagation from both current-query and warm
        # seeds, but says nothing about whether an Episode proves a claim.
        initial_graph_episode_activations = [
            {
                "episode_id": int(item.node_id),
                "score": min(1.0, max(0.0, float(item.score))),
            }
            for item in sorted(
                traversed,
                key=lambda node: (-float(node.score), int(node.node_id)),
            )
            if item.node_type == "episode"
            and math.isfinite(float(item.score))
            and float(item.score) > 0.0
        ][:32]
        if injected_hits:
            # Candidate treatment adds a root seed, so ordinary traversal no
            # longer tells us by itself whether a later Episode had an
            # independent route. Replaying the same bounded graph locally
            # once without injected roots, plus once per injected root, gives
            # a small provenance partition without a model call or DB write.
            dependency_started = perf_counter()
            ensure_deadline("prepared_early_candidate_dependency")
            base_only_nodes, _base_only_paths = self.traverser.expand(
                base_seeds_before_prepared_early,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            base_only_nodes = self._truncate_traversed_nodes(
                base_only_nodes,
                self.config.retrieval.candidate_limit,
                cue_fast_endpoint_keys,
            )
            base_only_episode_ids = {
                int(item.node_id)
                for item in base_only_nodes
                if item.node_type == "episode"
            }
            materialized_episode_ids = {
                int(item.node_id)
                for item in traversed
                if item.node_type == "episode"
            }
            root_edge_sets_by_episode: dict[int, set[int]] = {}
            root_closure_episode_ids: set[int] = set()
            for injected_hit in injected_hits:
                root_nodes, _root_paths = self.traverser.expand(
                    [injected_hit],
                    self.config.retrieval.graph_beam_width,
                    self.config.retrieval.graph_max_hops,
                )
                root_nodes = self._truncate_traversed_nodes(
                    root_nodes,
                    self.config.retrieval.candidate_limit,
                    cue_fast_endpoint_keys,
                )
                root_episode_ids = {
                    int(item.node_id)
                    for item in root_nodes
                    if item.node_type == "episode"
                }.intersection(materialized_episode_ids)
                root_closure_episode_ids.update(root_episode_ids)
                for episode_id in root_episode_ids:
                    root_edge_sets_by_episode.setdefault(episode_id, set()).update(
                        prepared_early_root_edge_ids_by_episode.get(
                            int(injected_hit.node_id), ()
                        )
                    )
            derived_only_ids = tuple(
                sorted(root_closure_episode_ids.difference(base_only_episode_ids))
            )
            overlap_ids = tuple(
                sorted(root_closure_episode_ids.intersection(base_only_episode_ids))
            )
            prepared_early_contextual_derived_episode_ids = derived_only_ids
            prepared_early_root_edge_ids_by_episode = {
                episode_id: tuple(sorted(root_edge_sets_by_episode.get(episode_id, set())))
                for episode_id in sorted(root_closure_episode_ids)
            }
            prepared_early_trace["candidate_pool_dependency_partition"] = {
                "method": "bounded_base_only_and_per_root_graph_replay_v1",
                "scope": "initial_graph_expansion_only",
                "base_only_episode_ids": sorted(base_only_episode_ids),
                "root_closure_episode_ids": sorted(root_closure_episode_ids),
                "derived_only_episode_ids": list(derived_only_ids),
                "independent_overlap_episode_ids": list(overlap_ids),
                "root_edge_ids_by_episode": {
                    str(episode_id): list(edge_ids)
                    for episode_id, edge_ids in sorted(
                        prepared_early_root_edge_ids_by_episode.items()
                    )
                },
            }
            self._record_phase(
                phase_seconds,
                "prepared_early_candidate_dependency",
                dependency_started,
            )
        else:
            prepared_early_trace["candidate_pool_dependency_partition"] = {
                "method": "not_run_no_injected_candidate",
                "scope": "initial_graph_expansion_only",
                "base_only_episode_ids": [],
                "root_closure_episode_ids": [],
                "derived_only_episode_ids": [],
                "independent_overlap_episode_ids": [],
                "root_edge_ids_by_episode": {},
            }
        self._record_phase(
            phase_seconds,
            "initial_graph_expansion",
            phase_started,
        )
        phase_started = perf_counter()
        ensure_deadline("followup_planning")
        followup_planning_trace_checkpoint = (
            trace_bridge.checkpoint() if trace_bridge is not None else None
        )
        followup_planner_invoked = False
        missing_initial_followup_slots: list[str] = []
        if frozen_plan is not None:
            followup_queries = [
                str(value) for value in frozen_plan.get("followup_queries", [])
            ]
            followup_planning_reason = "frozen_plan"
        elif followup_queries_override is not None:
            followup_queries = list(dict.fromkeys(followup_queries_override))
            followup_planning_reason = "explicit_override"
        else:
            should_plan_followup, followup_planning_reason = (
                self._followup_planning_decision(question, intent)
            )
            if (
                str(self.config.retrieval.followup_planning_mode)
                .strip()
                .casefold()
                == "missing_slots"
            ):
                missing_initial_followup_slots = self._initial_missing_followup_slots(
                    initial_queries,
                    initial_rankings,
                )
                should_plan_followup = bool(missing_initial_followup_slots)
                followup_planning_reason = (
                    "initial_candidate_slots_missing"
                    if should_plan_followup
                    else "all_initial_candidate_slots_present"
                )
            followup_planner_invoked = should_plan_followup
            followup_queries = (
                self._plan_followup_queries(question, intent, traversed)
                if should_plan_followup
                else []
            )
        self._record_phase(phase_seconds, "followup_planning", phase_started)
        followup_rankings: dict[str, list] = {
            "episode": [],
            "concept": [],
            "paragraph": [],
            "paragraph_episode_expansion": [],
        }
        if frozen_plan is not None:
            followup_rankings = deepcopy(
                frozen_plan.get("followup_rankings", followup_rankings)
            )
            if frozen_seed_stage_replay.get("classification") != "legacy_final_input_replay":
                seeds = self._merge_hits(seeds, frozen_followup_seed_hits)
                traversed, paths = self.traverser.expand(
                    seeds,
                    self.config.retrieval.graph_beam_width,
                    self.config.retrieval.graph_max_hops,
                )
                paths.extend(
                    self._explicit_association_paths(active_cue_ids, cue_scores)
                )
                traversed = self._truncate_traversed_nodes(
                    traversed,
                    self.config.retrieval.candidate_limit,
                    cue_fast_endpoint_keys,
                )
                frozen_seed_stage_replay["followup_seed_replay_applied"] = True
            else:
                frozen_seed_stage_replay["followup_seed_replay_applied"] = False
        elif followup_queries:
            followup_vector_specs = self._request_vector_specs(
                followup_queries,
                role="followup",
                authoritative_requirements=authoritative_requirements,
            )
            followup_vector_trace_checkpoint = (
                trace_bridge.checkpoint() if trace_bridge is not None else None
            )
            followup_planning_provider_delta = (
                trace_bridge.observations_since(followup_planning_trace_checkpoint)
                if (
                    trace_bridge is not None
                    and followup_planning_trace_checkpoint is not None
                )
                else None
            )
            followup_embedding_started = perf_counter()
            query_vector_bundle = self._prepare_request_vector_bundle(
                embedding_coordinator,
                [*initial_vector_specs, *followup_vector_specs],
                source_bundles=tuple(
                    bundle
                    for bundle in (
                        supplied_query_vector_bundle,
                        query_vector_bundle,
                    )
                    if bundle is not None
                ),
                legacy_overrides=query_embeddings_override,
                provider_purpose="followup_vector_retrieval",
                source_request_hash=source_request_hash,
                strict_vector_bundle=strict_vector_bundle,
            )
            self._record_phase(
                phase_seconds,
                "followup_embedding",
                followup_embedding_started,
            )
            emit_vector_bundle_if_observed(
                query_vector_bundle,
                before_observations=followup_vector_trace_checkpoint,
                inherited_origins=vector_trace_origins,
                extra_provider_parent_event_ids=(
                    followup_planning_provider_delta.provider_event_ids
                    if followup_planning_provider_delta is not None
                    else ()
                ),
            )
            followup_vector_overrides = self._bundle_overrides_for_queries(
                query_vector_bundle,
                followup_queries,
            )
            # The initial metadata was recorded before follow-up planning.
            # Refresh it with the expanded logical bundle; no raw query text
            # enters the trace-shaped result.
            contextual_trace.update(
                self._query_vector_trace_metadata(query_vector_bundle)
            )
            followup_episode_anchor_ids: list[int] = []
            (
                followup_hits,
                followup_cue_ids,
                followup_cue_entries,
                followup_rankings,
            ) = (
                self._vector_seed_hits_with_cues(
                    followup_queries,
                    followup_episode_anchor_ids,
                    cue_scope_question=question,
                    phase_seconds=phase_seconds,
                    timing_prefix="followup",
                    query_embeddings_override=followup_vector_overrides,
                    provider_purpose="followup_vector_retrieval",
                )
            )
            phase_started = perf_counter()
            ensure_deadline("followup_retrieval")
            active_cue_ids = list(
                dict.fromkeys([*active_cue_ids, *followup_cue_ids])
            )
            active_cue_entries = [
                *active_cue_entries,
                *[
                    item
                    for item in followup_cue_entries
                    if int(item["association_id"])
                    not in {
                        int(existing["association_id"])
                        for existing in active_cue_entries
                    }
                ],
            ]
            cue_scores = {
                int(item["association_id"]): float(item.get("cosine", 0.0))
                for item in active_cue_entries
            }
            seeds = self._merge_hits(seeds, followup_hits)
            episode_anchor_ids = self._interleave_anchor_ids(
                followup_episode_anchor_ids,
                initial_episode_anchor_ids,
            )
            traversed, paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            paths.extend(
                self._explicit_association_paths(active_cue_ids, cue_scores)
            )
            cue_fast_endpoint_keys = self._association_cue_fast_endpoint_keys(
                active_cue_entries
            )
            traversed = self._truncate_traversed_nodes(
                traversed,
                self.config.retrieval.candidate_limit,
                cue_fast_endpoint_keys,
            )
            self._record_phase(
                phase_seconds,
                "followup_graph_expansion",
                phase_started,
            )
        if self.logger:
            self.logger.emit(
                "followup_queries_planned",
                question=question,
                followup_queries=followup_queries,
                planner_invoked=followup_planner_invoked,
                planning_mode=self.config.retrieval.followup_planning_mode,
                planning_reason=followup_planning_reason,
                missing_initial_slot_count=len(missing_initial_followup_slots),
            )

        phase_started = perf_counter()
        ensure_deadline("source_cohort")
        association_capsule_fast_lane = bool(
            initial_rankings.get("association_capsule_fast_lane")
            or followup_rankings.get("association_capsule_fast_lane")
        )
        if association_capsule_fast_lane:
            source_key_cohort_trace = {
                "enabled": False,
                "reason": "association_capsule_fast_lane",
                "supported_source_keys": [],
                "added_episode_ids": [],
                "boosted_episode_ids": [],
                "skipped_source_keys": [],
            }
        elif frozen_plan is not None:
            source_key_cohort_trace = deepcopy(
                frozen_plan.get("source_key_cohort", {})
            )
            if frozen_seed_stage_replay.get("classification") != "legacy_final_input_replay":
                if frozen_seed_stage_replay.get("source_cohort_seed_source") == (
                    "saved_source_cohort_seed_hits"
                ):
                    replayed_cohort_hits = frozen_source_cohort_seed_hits
                else:
                    existing_keys = {
                        (hit.node_type, int(hit.node_id)) for hit in seeds
                    }
                    replayed_cohort_hits = [
                        hit
                        for hit in frozen_final_seed_hits
                        if (hit.node_type, int(hit.node_id)) not in existing_keys
                    ]
                seeds = self._merge_hits(seeds, replayed_cohort_hits)
                traversed, paths = self.traverser.expand(
                    seeds,
                    self.config.retrieval.graph_beam_width,
                    self.config.retrieval.graph_max_hops,
                )
                paths.extend(
                    self._explicit_association_paths(active_cue_ids, cue_scores)
                )
                traversed = self._truncate_traversed_nodes(
                    traversed,
                    self.config.retrieval.candidate_limit,
                    cue_fast_endpoint_keys,
                )
                frozen_seed_stage_replay["source_cohort_seed_replay_applied"] = True
                frozen_seed_stage_replay["replayed_source_cohort_seed_count"] = len(
                    replayed_cohort_hits
                )
            else:
                frozen_seed_stage_replay["source_cohort_seed_replay_applied"] = False
        else:
            cohort_hits, source_key_cohort_trace = self._source_key_cohort_hits(
                initial_episode_anchor_ids,
                traversed,
            )
            if cohort_hits:
                seeds = self._merge_hits(seeds, cohort_hits)
                traversed, paths = self.traverser.expand(
                    seeds,
                    self.config.retrieval.graph_beam_width,
                    self.config.retrieval.graph_max_hops,
                )
                paths.extend(
                    self._explicit_association_paths(active_cue_ids, cue_scores)
                )
                traversed = self._truncate_traversed_nodes(
                    traversed,
                    self.config.retrieval.candidate_limit,
                    cue_fast_endpoint_keys,
                )
        self._record_phase(phase_seconds, "source_cohort", phase_started)
        if self.logger:
            self.logger.emit(
                "source_key_cohort_retrieval",
                question=question,
                **source_key_cohort_trace,
            )

        # Freeze the direct/static evidence lane before query-time growth. The
        # graph may add a bounded bridge later, but it cannot trigger a second
        # stochastic rerank that silently replaces already selected evidence.
        phase_started = perf_counter()
        base_episodes, _base_concepts = self._materialize_nodes(
            traversed, include_sources=True
        )
        base_episode_limit = min(
            self.config.retrieval.answer_episode_limit,
            len(base_episodes),
        )
        paragraph_context_by_source = self._paragraph_context_by_source(
            [
                *initial_rankings.get("paragraph", []),
                *followup_rankings.get("paragraph", []),
            ]
        )
        self._record_phase(
            phase_seconds,
            "evidence_preparation",
            phase_started,
        )
        evidence_floor_trace: dict = {}
        observed_rerank_input_fingerprint = (
            self._rerank_candidate_input_fingerprint(
                base_episodes,
                paragraph_context_by_source,
            )
        )
        if frozen_plan is not None:
            reranked_episode_ids = [
                int(value) for value in frozen_plan.get("reranked_episode_ids", [])
            ]
            rerank_trace = deepcopy(frozen_plan.get("rerank_trace", {}))
            expected_fingerprint = str(
                rerank_trace.get("candidate_input_fingerprint", "")
            )
            rerank_reuse = {
                "observed_candidate_input_fingerprint": (
                    observed_rerank_input_fingerprint
                ),
                "frozen_candidate_input_fingerprint": expected_fingerprint,
                "historical_rerank_mapping_status": (
                    self._rerank_slot_mapping_status(rerank_trace)
                ),
                "candidate_pool_changed": bool(
                    prepared_early_contextual_derived_episode_ids
                ),
            }
            rerank_reuse["status"] = self._frozen_rerank_reuse_decision(
                expected_fingerprint,
                observed_rerank_input_fingerprint,
            )
            if rerank_reuse["status"] != "frozen_input_fingerprint_matched":
                # Candidate derivation answers a provenance question; it is
                # not an input-identity shortcut.  A saved rerank decision can
                # be replayed only when the actual prompt-facing candidates
                # and paragraph context fingerprint match.  Old plans without
                # a fingerprint remain replayable for historical comparison,
                # but cannot grant a new mapping, early stop, or learning
                # qualification.
                reranked_episode_ids = []
                rerank_trace = {
                    "enabled": False,
                    "reason": "frozen_rerank_not_reused_without_matching_input",
                    "candidate_input_fingerprint": (
                        observed_rerank_input_fingerprint
                    ),
                    "candidate_input_reuse": rerank_reuse,
                    "historical_rerank": {
                        "enabled": bool(rerank_trace.get("enabled", False)),
                        "had_error": bool(str(rerank_trace.get("error", "")).strip()),
                        "mapping_status": rerank_reuse[
                            "historical_rerank_mapping_status"
                        ],
                    },
                    "merged_coverage": {"coverage": []},
                    "deterministic_evidence_floor": {},
                }
            prepared_early_trace["rerank_input_reuse"] = rerank_reuse
        else:
            phase_started = perf_counter()
            evidence_floor_trace = self._hybrid_evidence_floor_trace(
                question,
                initial_rankings,
                followup_rankings,
                initial_queries,
                followup_queries,
            )
            constraint_candidate_ids = self._constraint_candidate_ids(
                question,
                initial_queries,
                initial_rankings,
                followup_queries,
                followup_rankings,
            )
            self._record_phase(
                phase_seconds,
                "evidence_preparation",
                phase_started,
            )
            phase_started = perf_counter()
            ensure_deadline("evidence_rerank")
            reranked_episode_ids, rerank_trace = self._rerank_answer_episodes(
                question,
                intent,
                list(
                    dict.fromkeys(
                        [question, *intent.search_queries, *followup_queries]
                    )
                ),
                base_episodes,
                base_episode_limit,
                [*constraint_candidate_ids, *episode_anchor_ids],
                paragraph_context_by_source,
                required_candidate_ids=(
                    evidence_floor_trace["selected_episode_ids"]
                ),
                answer_slot_anchor_ids=[
                    int(value)
                    for slot in evidence_floor_trace.get(
                        "constraint_slots", []
                    )
                    if str(slot.get("query", "")).startswith(
                        "__answer_slot__ "
                    )
                    for value in slot.get("floor_episode_ids", [])
                ],
                association_cue_entries=active_cue_entries,
            )
            self._record_phase(
                phase_seconds,
                "evidence_rerank",
                phase_started,
            )
            phase_started = perf_counter()
            evidence_floor_trace = self._admit_evidence_floor_trace(
                evidence_floor_trace,
                rerank_trace.get("required_evidence_floor_ids", []),
            )
            rerank_trace["deterministic_evidence_floor"] = evidence_floor_trace
            rerank_trace["constraint_candidate_ids"] = constraint_candidate_ids
            rerank_trace["candidate_input_fingerprint"] = (
                observed_rerank_input_fingerprint
            )
            self._record_phase(
                phase_seconds,
                "evidence_preparation",
                phase_started,
            )
        frozen_coverage_groups = (
            rerank_trace.get("merged_coverage", {})
            .get("coverage", [])
        )
        counterfactual_replay_plan = (
            frozen_plan
            if frozen_plan is not None
            else {
                "version": 4,
                "question": question,
                "intent": asdict(intent),
                "initial_queries": initial_queries,
                "followup_queries": followup_queries,
                "final_seed_hits": self._serialize_hits(seeds),
                "episode_anchor_ids": episode_anchor_ids,
                "association_cue_entries": active_cue_entries,
                "association_cue_association_ids": active_cue_ids,
                "reranked_episode_ids": reranked_episode_ids,
                "rerank_trace": rerank_trace,
                "configuration": self._replay_configuration(),
            }
        )
        phase_started = perf_counter()
        ensure_deadline("association_growth")
        for round_index in range(
            self.config.retrieval.growth_max_rounds
            if allow_association_learning
            else 0
        ):
            episodes, concepts = self._materialize_nodes(traversed)
            growth_episode_limit = min(
                self.config.retrieval.growth_episode_limit,
                len(episodes),
            )
            growth_episodes, growth_paths = self._select_answer_evidence(
                episodes,
                paths,
                growth_episode_limit,
                30,
                set([*all_new_associations, *all_reinforced_associations]),
                question,
                reranked_episode_ids or episode_anchor_ids,
                self.config.retrieval.learned_bridge_slots,
                self.config.retrieval.learned_bridge_min_query_relevance,
                self.config.retrieval.learned_bridge_duplicate_threshold,
                frozen_coverage_groups,
            )
            if self.logger:
                self.logger.emit(
                    "growth_evidence_selected",
                    question=question,
                    round=round_index + 1,
                    episode_limit=growth_episode_limit,
                    episode_ids=[item["id"] for item in growth_episodes],
                    path_association_ids=[
                        item.get("association_id") for item in growth_paths
                    ],
                )
            growth_nodes = [
                *[
                    {
                        "type": "episode",
                        "id": item["id"],
                        "text": item["text"],
                        "participants": item.get("participants", []),
                        "source_key": item.get("source_key", ""),
                    }
                    for item in growth_episodes
                ],
                *[
                    {
                        "type": "concept",
                        "id": item["id"],
                        "text": f"{item['canonical_name']}：{item['description']}",
                    }
                    for item in concepts[:20]
                ],
            ]
            outcome = self.growth.grow(
                question,
                growth_nodes,
                growth_paths,
                seen_growth_fingerprints,
            )
            all_new_associations.extend(outcome.created_ids)
            all_reinforced_associations.extend(outcome.reinforced_ids)
            for association_id, row in outcome.reinforced_before.items():
                reinforced_before.setdefault(int(association_id), dict(row))
            if not outcome.changed:
                break
            if self.logger:
                self.logger.emit(
                    "graph_expansion",
                    question=question,
                    round=round_index + 1,
                    created_association_ids=outcome.created_ids,
                    reinforced_association_ids=outcome.reinforced_ids,
                )
            traversed, paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            paths.extend(
                self._explicit_association_paths(active_cue_ids, cue_scores)
            )
            traversed = self._truncate_traversed_nodes(
                traversed,
                self.config.retrieval.candidate_limit,
                cue_fast_endpoint_keys,
            )
        self._record_phase(
            phase_seconds,
            "association_growth",
            phase_started,
        )
        phase_started = perf_counter()
        ensure_deadline("final_selection")
        late_nodes, late_paths = self._late_learned_bridge_closure(
            question,
            reranked_episode_ids or episode_anchor_ids,
            traversed,
        )
        if late_nodes:
            traversed = [*traversed, *late_nodes]
        if late_paths:
            paths.extend(late_paths)
        episodes, concepts = self._materialize_nodes(traversed, include_sources=True)
        changed_association_ids = [
            *all_new_associations,
            *all_reinforced_associations,
        ]
        paths.extend(
            self._explicit_association_paths(
                [*active_cue_ids, *changed_association_ids],
                cue_scores,
            )
        )
        episode_limit = min(self.config.retrieval.answer_episode_limit, len(episodes))
        capsule_episode_ids = {
            int(value)
            for capsule in rerank_trace.get("association_capsules", [])
            for value in capsule.get("endpoint_episode_ids", [])
        }
        if association_capsule_fast_lane:
            # An audited capsule already names the small premise closure that
            # answered an earlier query.  Sending the ordinary broad Top-K to
            # the chat model defeats its purpose as a latency/token cache.
            # Keep a few slots for evidence floors, but cap the prompt-facing
            # evidence independently of the broad candidate pool.
            episode_limit = min(
                episode_limit,
                max(
                    len(capsule_episode_ids),
                    int(
                        self.config.retrieval.association_cue_fast_path_evidence_limit
                    ),
                ),
            )
        selected_episodes, answer_paths = self._select_answer_evidence(
            episodes,
            paths,
            episode_limit,
            self.config.retrieval.answer_path_limit,
            set([*active_cue_ids, *changed_association_ids]),
            question,
            reranked_episode_ids or episode_anchor_ids,
            self.config.retrieval.learned_bridge_slots,
            self.config.retrieval.learned_bridge_min_query_relevance,
            self.config.retrieval.learned_bridge_duplicate_threshold,
            frozen_coverage_groups,
            cue_scores,
            capsule_episode_ids,
        )
        # M2/M3: apply deterministic slot coverage after the ordinary
        # candidate/rerank work is frozen.  This gives the masked selector and
        # contextual treatment exactly the same base candidate universe.
        contextual_started = perf_counter()
        # A frozen plan freezes stochastic planner/retrieval input; it does
        # not make the local contextual selector ineligible.  In particular,
        # a restored, contract-bound vector bundle must still reach the same
        # residual-slot and source-provenance checks as a live request.  The
        # previous frozen-plan guard silently bypassed this public Q2 lane,
        # making an available edge indistinguishable from one rejected by the
        # matcher. Do not synthesize vectors here: without a supplied bundle
        # the historical no-selection behavior remains.
        if frozen_plan is None or query_vector_bundle is not None:
            # Freeze only direct vector/sparse seed endpoints for T13 anchor
            # provenance. ``base_episodes`` is used solely to confirm the
            # seed still materialized; it is not proof of independence because
            # it also contains graph-expanded/source-cohort rows. Association
            # cue endpoints are explicitly excluded even if they were merged
            # into the seed list for ordinary retrieval.
            materialized_base_ids = {
                int(item["id"])
                for item in base_episodes
                if isinstance(item, dict) and int(item.get("id", 0) or 0) > 0
            }
            learning_independent_base_ids = tuple(
                sorted(
                    set(base_episode_ids_before_contextual)
                    .intersection(materialized_base_ids)
                    .difference(cue_anchor_exclusions)
                )
            )
            # This same direct snapshot explains candidate-missing/dropped.
            # The V3 selector still receives the full pre-context universe,
            # so this stricter creation provenance cannot change delivery.
            learning_initial_candidate_ids = learning_independent_base_ids
            learning_initial_candidate_set = set(learning_initial_candidate_ids)
            learning_initial_delivered_ids = tuple(
                sorted(
                    {
                        int(item["id"])
                        for item in selected_episodes
                        if isinstance(item, dict)
                        and int(item.get("id", 0) or 0) in learning_initial_candidate_set
                    }
                )
            )
            selected_episodes, contextual_trace = self._select_contextual_slots(
                episodes=episodes,
                baseline_selected=selected_episodes,
                rerank_trace=rerank_trace,
                reranked_episode_ids=reranked_episode_ids,
                bundle=query_vector_bundle,
                domain=contextual_domain,
                endpoint_limit=contextual_endpoint_limit,
                anchor_activations=base_anchor_activations,
                authoritative_requirements=authoritative_requirements,
                evaluation_as_of=request_contextual_evaluation_as_of,
                cue_endpoint_episode_ids=cue_anchor_exclusions,
                learning_initial_candidate_episode_ids=learning_initial_candidate_ids,
                learning_independent_base_episode_ids=learning_independent_base_ids,
                learning_initial_delivered_episode_ids=learning_initial_delivered_ids,
                prepared_early_contextual_episode_ids=(
                    prepared_early_contextual_episode_ids
                ),
                prepared_early_contextual_derived_episode_ids=(
                    prepared_early_contextual_derived_episode_ids
                ),
                source_fact_cache=prepared_early_source_cache,
            )
        # The late selector returns its own trace object.  Preserve the
        # earlier observation as a distinct stage rather than overwriting it
        # with an apparently late-only record.
        contextual_trace["prepared_early"] = prepared_early_trace
        self._record_phase(
            phase_seconds,
            "contextual_association",
            contextual_started,
        )
        if self.config.retrieval.graph_max_hops > 0:
            chronology = self.chronology.order(
                [item["id"] for item in selected_episodes]
            )
            episode_map = {item["id"]: item for item in selected_episodes}
            ordered_episodes = [
                episode_map[node_id]
                for node_id in chronology.ordered_ids
                if node_id in episode_map
            ]
            chronology_notes = chronology.notes
        else:
            ordered_episodes = selected_episodes
            chronology_notes = [
                "纯向量基线：未读取 Association，Episode 保持向量相似度顺序。"
            ]
        self._record_phase(
            phase_seconds,
            "final_selection",
            phase_started,
        )
        provisional_used_edge_ids = list(
            dict.fromkeys(
                int(item["association_id"])
                for item in answer_paths
                if "association_id" in item
            )
        )
        # Persist a redacted, diagnostic-only retrieval view before answer
        # generation.  An answer/audit timeout must not erase the already
        # completed candidate, selection, and source-provenance observations.
        # This event has no authority to create an edge or mark one used.
        self._emit_answer_checkpoint(
            "retrieval_checkpoint",
            question=question,
            authoritative_requirements=asdict(authoritative_requirements),
            candidate_episode_ids=[
                int(item["id"]) for item in episodes if "id" in item
            ],
            reranked_episode_ids=list(reranked_episode_ids),
            selected_episode_ids=[
                int(item["id"]) for item in ordered_episodes if "id" in item
            ],
            selected_evidence=ordered_episodes,
            source_provenance_status=contextual_trace.get(
                "source_provenance_status", []
            ),
            contextual_contribution_ids=contextual_trace.get(
                "selected_contextual_contribution_ids", []
            ),
            request_deadline_remaining_seconds=(
                None if deadline_at is None else round(max(0.0, deadline_at - monotonic()), 6)
            ),
        )
        phase_started = perf_counter()
        ensure_deadline("answer_generation")
        growth_counterfactual_utility = self._counterfactual_growth_utility(
            counterfactual_replay_plan,
            all_new_associations,
            all_reinforced_associations,
            reinforced_before,
        )
        changed_set = {
            int(value)
            for value in [*all_new_associations, *all_reinforced_associations]
        }
        persistable_changed = {
            int(value)
            for value in growth_counterfactual_utility.get(
                "persistable_changed_ids", []
            )
        }
        persistence_edge_ids = [
            value
            for value in provisional_used_edge_ids
            if value not in changed_set or value in persistable_changed
        ]
        if self.config.retrieval.growth_persist_only_used:
            growth_utility_gate = self._prune_unused_growth(
                all_new_associations,
                all_reinforced_associations,
                reinforced_before,
                persistence_edge_ids,
            )
            all_new_associations = list(
                growth_utility_gate["retained_created_ids"]
            )
            all_reinforced_associations = list(
                growth_utility_gate["retained_reinforced_ids"]
            )
        else:
            growth_utility_gate = {
                "enabled": False,
                "retained_created_ids": list(
                    dict.fromkeys(all_new_associations)
                ),
                "retained_reinforced_ids": list(
                    dict.fromkeys(all_reinforced_associations)
                ),
                "removed_created_ids": [],
                "restored_reinforced_ids": [],
                "premise_closure_ids": [],
            }
        durable_changed_ids = {
            int(value)
            for value in [*all_new_associations, *all_reinforced_associations]
        }
        removed_changed_ids = changed_set.difference(durable_changed_ids)
        if removed_changed_ids:
            answer_paths = [
                item
                for item in answer_paths
                if int(item.get("association_id", -1)) not in removed_changed_ids
            ]
            if self.config.retrieval.graph_max_hops > 0:
                chronology = self.chronology.order(
                    [item["id"] for item in selected_episodes]
                )
                episode_map = {item["id"]: item for item in selected_episodes}
                ordered_episodes = [
                    episode_map[node_id]
                    for node_id in chronology.ordered_ids
                    if node_id in episode_map
                ]
                chronology_notes = chronology.notes
        used_edge_ids = list(
            dict.fromkeys(
                int(item["association_id"])
                for item in answer_paths
                if "association_id" in item
            )
        )
        # Counterfactual pruning can be non-trivial.  Recheck immediately at
        # the durable utility mutation so a request that expires during local
        # CPU work cannot increment an association's usage statistics.
        # ``stop_after='evidence'`` follows this same retrieval path but must
        # remain observational: a selected edge is reported as participation,
        # never as a use/utility write.
        if record_association_use:
            ensure_deadline("growth_utility_write")
            self.associations.mark_used(
                used_edge_ids,
                require_live=lambda: ensure_deadline("growth_utility_write"),
            )
        self._record_phase(
            phase_seconds,
            "growth_utility",
            phase_started,
        )
        phase_started = perf_counter()
        if generate_answer:
            answer, answer_audits, answer_revision_count = (
                self._generate_audited_answer(
                    question,
                    intent,
                    ordered_episodes,
                    concepts[: self.config.retrieval.answer_concept_limit],
                    answer_paths,
                    chronology_notes,
                    deadline_at=deadline_at,
                )
            )
            answer_terminal_state = deepcopy(
                getattr(self, "_last_answer_execution_state", {})
            )
        else:
            answer, answer_audits, answer_revision_count = "", [], 0
            answer_terminal_state = {
                "terminal_state": "not_generated",
                "reason": "generate_answer_false",
            }
        self._record_phase(
            phase_seconds,
            "answer_generation",
            phase_started,
        )
        evidence_slot_trace = self._final_evidence_slot_trace(
            rerank_trace,
            [int(item["id"]) for item in ordered_episodes],
        )
        selector_trace_payload = {
            key: deepcopy(value)
            for key, value in contextual_trace.items()
            if key in {
                "slots",
                "masked_selected_episode_ids",
                "masked_missing_slots",
                "treatment_selected_episode_ids",
                "new_slots",
                "lost_slots",
                "new_slot_count",
                "harm",
                "strict_attribution",
                "shadow",
                "reason",
                "requirements_status",
                "authoritative_requirements",
                "base_support_mapping_status",
                "evaluation_as_of",
                "target_gate",
                "target_checks",
                "compatibility_projection",
                "merged_candidates",
                "merged_aggregates",
                "masked_selector",
                "treatment_selector",
                "source_provenance_status",
                "selected_contextual_contribution_ids",
                "ranking_only_contextual_contribution_ids",
                "matcher_backend",
                "contribution_counterfactual",
                "base_endpoint_manifest",
                "prepared_early",
            }
        }
        # Keep the established receipt key for historical readers while
        # exposing the contribution-aware shape under its explicit v3 name.
        evidence_slot_trace["slot_selector_v2"] = selector_trace_payload
        if str(contextual_trace.get("backend", "")).endswith("_v3"):
            evidence_slot_trace["slot_selector_v3"] = deepcopy(
                selector_trace_payload
            )
        result = {
            "question": question,
            "query_plan_frozen": frozen_plan is not None,
            "query_plan_id": (
                str(frozen_plan["plan_id"]) if frozen_plan is not None else None
            ),
            "intent": asdict(intent),
            # Local diagnostics retain the frozen request contract; the
            # parallel trace payload is hash-only and is safe for a future
            # explicit v3 ``requirements_resolved`` event.
            "authoritative_requirements": (
                authoritative_requirements.as_result_dict()
            ),
            "authoritative_requirements_trace": (
                authoritative_requirements.as_trace_payload()
            ),
            "query_vector_bundle": (
                query_vector_bundle.metadata()
                if query_vector_bundle is not None
                else None
            ),
            "followup_search_queries": followup_queries,
            "episode_activation_trace": {
                "role": "retrieval_attention_only",
                "warm_input": [
                    {"episode_id": episode_id, "score": score}
                    for episode_id, score in sorted(
                        (warm_episode_activations or {}).items()
                    )
                ],
                "initial_graph_top_episodes": initial_graph_episode_activations,
            },
            "followup_planner_invoked": followup_planner_invoked,
            "followup_planning_mode": (
                self.config.retrieval.followup_planning_mode
            ),
            "followup_planning_reason": followup_planning_reason,
            "followup_missing_initial_slot_count": len(
                missing_initial_followup_slots
            ),
            "atomic_anchor_episode_ids": episode_anchor_ids,
            "candidate_episode_ids": [
                int(value)
                for value in (
                    rerank_trace.get("candidate_episode_ids")
                    or [
                        item.node_id
                        for item in traversed
                        if item.node_type == "episode"
                    ][: self.config.retrieval.rerank_candidate_limit]
                )
            ],
            "reranked_episode_ids": reranked_episode_ids,
            "rerank_trace": rerank_trace,
            "evidence_slot_trace": evidence_slot_trace,
            "rerank_frozen_before_growth": True,
            "source_key_cohort": source_key_cohort_trace,
            "answer": answer,
            "answer_audits": answer_audits,
            "answer_revision_count": answer_revision_count,
            "answer_terminal_state": answer_terminal_state,
            "answer_generation_skipped": not generate_answer,
            "episode_ids": [item["id"] for item in ordered_episodes],
            "concept_ids": [
                item["id"]
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_ids": used_edge_ids,
            "association_usage_recorded": bool(record_association_use),
            "new_association_ids": list(dict.fromkeys(all_new_associations)),
            "reinforced_association_ids": list(
                dict.fromkeys(all_reinforced_associations)
            ),
            "growth_utility_gate": growth_utility_gate,
            "growth_counterfactual_utility": growth_counterfactual_utility,
            "association_cue_ids": active_cue_ids,
            "association_cue_entries": active_cue_entries,
            "association_capsule_fast_lane": association_capsule_fast_lane,
            "contextual_association": {
                "enabled": bool(contextual_trace.get("enabled", False)),
                "backend": str(contextual_trace.get("backend", "contextual_double_key")),
                "matcher_backend": str(
                    contextual_trace.get("matcher_backend", "")
                ),
                "reason": str(contextual_trace.get("reason", "not_observed")),
                "masked_missing_slots": list(
                    contextual_trace.get("masked_missing_slots", [])
                ),
                "evaluation_as_of": contextual_trace.get("evaluation_as_of"),
                "context_prototype_hits": len(contextual_trace.get("context_hits", [])),
                "need_prototype_hits": sum(
                    len(item) for item in contextual_trace.get("need_hits", [])
                ),
                "candidate_count": len(contextual_trace.get("hits", [])),
                "selected_count": (
                    len(
                        contextual_trace.get(
                            "selected_contextual_contribution_ids", []
                        )
                    )
                    if str(contextual_trace.get("backend", "")).endswith("_v3")
                    else len(contextual_trace.get("strict_attribution", []))
                ),
                "treatment_selected_count": int(
                    contextual_trace.get("selected_count", 0)
                ),
                "shadow": bool(contextual_trace.get("shadow", False)),
                "new_slot_count": int(contextual_trace.get("new_slot_count", 0)),
                "harm_count": int(contextual_trace.get("harm_count", 0)),
                "attached_edges": [
                    int(value) for value in contextual_trace.get("attached_edges", [])
                ],
                "attached_episode_ids": [
                    int(value) for value in contextual_trace.get("attached_episode_ids", [])
                ],
                "masked_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "masked_selected_episode_ids", []
                    )
                ],
                "treatment_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "treatment_selected_episode_ids", []
                    )
                ],
                "external_calls": 0,
                "context_query_id": str(
                    contextual_trace.get("context_query_id", "")
                ),
                "query_vectors": list(
                    contextual_trace.get("query_vectors", [])
                ),
                "query_vector_bundle": contextual_trace.get(
                    "query_vector_bundle"
                ),
                "base_episode_ids": base_episode_ids_before_contextual,
                "base_endpoint_manifest": list(
                    contextual_trace.get("base_endpoint_manifest", [])
                ),
                "contextual_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "attached_episode_ids", []
                    )
                ],
                # Utility learning consumes only this deterministic
                # single-edge/leave-one-out receipt; it never infers success
                # from the final answer prose or an unmasked batch result.
                "strict_attribution": list(
                    contextual_trace.get("strict_attribution", [])
                ),
                "target_gate": deepcopy(contextual_trace.get("target_gate", {})),
                "compatibility_projection": str(
                    contextual_trace.get("compatibility_projection", "")
                ),
                "contribution_selector": deepcopy(
                    contextual_trace.get("treatment_selector", {})
                ),
                "masked_contribution_selector": deepcopy(
                    contextual_trace.get("masked_selector", {})
                ),
                "contribution_counterfactual": deepcopy(
                    contextual_trace.get("contribution_counterfactual", {})
                ),
                "merged_candidates": deepcopy(
                    contextual_trace.get("merged_candidates", [])
                ),
                "merged_aggregates": deepcopy(
                    contextual_trace.get("merged_aggregates", [])
                ),
                "source_provenance_status": deepcopy(
                    contextual_trace.get("source_provenance_status", [])
                ),
                "selected_contextual_contribution_ids": list(
                    contextual_trace.get(
                        "selected_contextual_contribution_ids", []
                    )
                ),
                "ranking_only_contextual_contribution_ids": list(
                    contextual_trace.get(
                        "ranking_only_contextual_contribution_ids", []
                    )
                ),
                "prepared_early": deepcopy(
                    contextual_trace.get("prepared_early", {})
                ),
            },
            "chronology_notes": chronology_notes,
            "evidence_episodes": [
                {
                    key: item[key]
                    for key in (
                        "id",
                        "score",
                        "text",
                        "participants",
                        "source_key",
                        "segment_index",
                        "story_time_text",
                        "story_order",
                        "timeline_scope",
                        "evidence_origin",
                        "epistemic_status",
                        "generation",
                        "epistemic_note",
                    )
                }
                for item in ordered_episodes
            ],
            "evidence_concepts": [
                {
                    key: item[key]
                    for key in ("id", "score", "canonical_name", "description")
                }
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_paths": answer_paths,
            "paragraph_retrieval_enabled": self.paragraph_retrieval_enabled,
            "sparse_retrieval_enabled": self.sparse_retrieval_enabled,
            "query_embedding_cache": {
                "hit_count": len(
                    set(self.last_query_embedding_cache_trace["hits"])
                ),
                "miss_count": len(
                    set(self.last_query_embedding_cache_trace["misses"])
                ),
            },
        }
        total_seconds = round(perf_counter() - query_started_at, 6)
        measured_seconds = round(sum(phase_seconds.values()), 6)
        result["timings"] = {
            "version": "query-stage-timing-v1",
            "phases_seconds": phase_seconds,
            "measured_seconds": measured_seconds,
            "unattributed_seconds": round(
                max(0.0, total_seconds - measured_seconds),
                6,
            ),
            "total_seconds": total_seconds,
            "deadline_seconds": deadline_seconds,
        }
        if self.logger:
            self.logger.emit(
                "answer_generated" if generate_answer else "retrieval_completed",
                result=result,
            )
        # This is deliberately added after the normal event-log payload.  It
        # carries bounded authorized Source excerpts for the public
        # evidence-only return value, but must not turn a general retrieval
        # log into an exportable raw-source payload.
        result["_runtime_selected_evidence"] = ordered_episodes
        return result
