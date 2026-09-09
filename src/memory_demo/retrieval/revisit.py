"""Strict, request-local inputs for an exact V3 contextual revisit.

This module intentionally contains no database, model, or answer-cache API.
The durable v16 contract is redacted; it can only be used to compare these
caller-supplied runtime values, never to reconstruct them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
import os
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from memory_demo.embeddings import normalize_embedding, normalize_query_text
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.types import (
    CandidateContribution,
    ContextualRestrictedRewriteGuard,
    ContextualRestrictedRewriteGuardDraft,
    ContextualRevisitContractLookup,
    ContextualRevisitRuntimeManifest,
    ContextualRevisitRuntimeSeed,
    ContextualRevisitSlotNeedBinding,
    EvidenceSelectionBudget,
    QueryIntent,
    QueryVectorBundle,
    SourceFactRef,
)


_OPAQUE_DIGEST_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}:sha256:[0-9a-f]{64}$"
)
_REQUEST_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESTRICTED_REWRITE_GRAMMAR_VERSION = "restricted-rewrite-grammar-v1"
_RESTRICTED_REWRITE_SIGNATURE_VERSION = "contextual-restricted-rewrite-guard-v5"
_RESTRICTED_REWRITE_TERM_RE = re.compile(r"^[a-z][a-z'\-]*(?: [a-z][a-z'\-]*)*$")
_RESTRICTED_REWRITE_OF_TEMPLATE_RE = re.compile(
    r"^(?:what is|tell me) the (?P<predicate>[a-z][a-z'\-]*(?: [a-z][a-z'\-]*){0,3}) of (?P<subject>[a-z][a-z'\-]*(?: [a-z][a-z'\-]*){0,5})\??$"
)
_RESTRICTED_REWRITE_FOR_TEMPLATE_RE = re.compile(
    r"^for (?P<subject>[a-z][a-z'\-]*(?: [a-z][a-z'\-]*){0,5}), (?:what is|tell me) the (?P<predicate>[a-z][a-z'\-]*(?: [a-z][a-z'\-]*){0,3})\??$"
)
# This is a conservative deny-list, not a general natural-language semantic
# verifier.  A fixed word-order swap is authorized only when neither field
# carries one of these known semantic-risk markers; every unrecognized form
# remains an ordinary retrieval request and is not promised equivalent.
_RESTRICTED_REWRITE_REJECT_TOKENS = frozenset(
    {
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "before",
        "between",
        "can",
        "could",
        "current",
        "currently",
        "during",
        "each",
        "either",
        "every",
        "except",
        "first",
        "former",
        "from",
        "he",
        "her",
        "his",
        "how",
        "if",
        "in",
        "it",
        "its",
        "last",
        "later",
        "less",
        "may",
        "more",
        "most",
        "must",
        "never",
        "no",
        "none",
        "not",
        "now",
        "only",
        "of",
        "or",
        "our",
        "previous",
        "second",
        "she",
        "should",
        "since",
        "than",
        "that",
        "theirs",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "today",
        "tomorrow",
        "unless",
        "until",
        "versus",
        "was",
        "were",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "without",
        "would",
        "yesterday",
        "you",
    }
)


def _canonical_digest(namespace: str, payload: object) -> str:
    """Make one opaque, durable-safe digest without returning its material."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require_opaque_digest(value: object, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _OPAQUE_DIGEST_RE.fullmatch(normalized):
        raise ValueError(f"exact revisit {field_name} must be an opaque sha256 digest")
    return normalized


def exact_revisit_context_hash(
    question: str,
    *,
    context_scope_hash: str | None = None,
) -> str:
    """Bind an exact revisit input to question plus an optional opaque scope.

    A caller may keep legacy question-only exact tickets by omitting the
    scope.  New automatically-created contracts require a caller-supplied
    opaque scope so a generic question cannot cross a conversation/user
    boundary merely because its wording is identical.
    """

    normalized = normalize_query_text(str(question or ""))
    if not normalized:
        raise ValueError("exact revisit question is required")
    if context_scope_hash is None:
        # Preserve v16's already-persisted manual-ticket spelling.  Only a
        # caller that explicitly supplies a scope receives the stronger,
        # scope-bound context identity below.
        return _canonical_digest("revisit-context", {"question": normalized})
    scope = _require_opaque_digest(context_scope_hash, "context_scope_hash")
    return _canonical_digest(
        "revisit-context", {"question": normalized, "context_scope_hash": scope}
    )


def exact_revisit_request_hash(question: str) -> str:
    """Match QueryEngine's request-vector provenance spelling exactly."""

    normalized = normalize_query_text(str(question or ""))
    if not normalized:
        raise ValueError("exact revisit question is required")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RestrictedRewriteIR:
    """One fully-consumed, positive, single-predicate question IR.

    It exists only in process memory while calculating an HMAC commitment.
    The explicit ``none`` values are intentional: a later grammar version
    cannot silently treat a newly introduced time/number/negation/object
    constraint as absent.
    """

    subject: str
    predicate: str
    grammar_version: str = _RESTRICTED_REWRITE_GRAMMAR_VERSION
    answer_variable_role: str = "object"
    polarity: str = "positive"
    temporal: str = "none"
    numeric: str = "none"
    comparison: str = "none"
    quantifier: str = "none"
    modality: str = "none"
    condition: str = "none"
    reference: str = "none"

    def canonical_payload(self) -> dict[str, str]:
        return {
            "answer_variable_role": self.answer_variable_role,
            "comparison": self.comparison,
            "condition": self.condition,
            "grammar_version": self.grammar_version,
            "modality": self.modality,
            "numeric": self.numeric,
            "polarity": self.polarity,
            "predicate": self.predicate,
            "quantifier": self.quantifier,
            "reference": self.reference,
            "subject": self.subject,
            "temporal": self.temporal,
        }


@dataclass(frozen=True, slots=True)
class RestrictedRewriteCommitmentKey:
    """A process-local HMAC key; its secret is never serialised or logged."""

    key_id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        key_id = str(self.key_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", key_id):
            raise ValueError("restricted rewrite commitment key id is invalid")
        secret = bytes(self.secret or b"")
        if len(secret) < 16:
            raise ValueError("restricted rewrite commitment key is too short")
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "secret", secret)

    @classmethod
    def from_environment(cls) -> "RestrictedRewriteCommitmentKey | None":
        """Load an opt-in key without putting it in application config/trace."""

        key_id = str(
            os.environ.get("MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY_ID", "")
        ).strip()
        secret = os.environ.get("MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY", "")
        if not key_id or not secret:
            return None
        try:
            return cls(key_id=key_id, secret=secret.encode("utf-8"))
        except (TypeError, ValueError, UnicodeError):
            return None

    def _scope_key(self, context_scope_hash: str) -> tuple[str, bytes]:
        """Derive a scope-local signing key without retaining request prose."""

        scope = _require_opaque_digest(context_scope_hash, "context_scope_hash")
        return scope, hmac.new(
            self.secret,
            b"memory-demo/restricted-rewrite/scope/v1\0" + scope.encode("ascii"),
            hashlib.sha256,
        ).digest()

    def commitment(self, *, context_scope_hash: str, ir: RestrictedRewriteIR) -> str:
        """Create a scope-derived root commitment over every explicit IR atom."""

        _scope, scope_key = self._scope_key(context_scope_hash)
        atoms = {
            field_name: hmac.new(
                scope_key,
                (
                    b"memory-demo/restricted-rewrite/atom/v1\0"
                    + field_name.encode("ascii")
                    + b"\0"
                    + str(value).encode("utf-8")
                ),
                hashlib.sha256,
            ).hexdigest()
            for field_name, value in ir.canonical_payload().items()
        }
        root_payload = {
            "atom_commitments": atoms,
            "commitment_key_id": self.key_id,
            "grammar_version": ir.grammar_version,
            "signature_version": _RESTRICTED_REWRITE_SIGNATURE_VERSION,
        }
        root = hmac.new(
            scope_key,
            (
                b"memory-demo/restricted-rewrite/root/v1\0"
                + json.dumps(
                    root_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).hexdigest()
        return "restricted-rewrite:sha256:" + root

    def binding_commitment(
        self,
        *,
        context_scope_hash: str,
        rewrite_commitment: str,
        creation_request_id: str,
        domain: str,
        seed_fingerprint: str,
        context_cue_vector_fingerprint: str,
        need_cue_vector_fingerprint: str,
        model_id: str,
        embedding_space_id: str,
        dimension: int,
        dtype: str,
        grammar_version: str,
        signature_version: str,
    ) -> str:
        """Sign the semantic root to one request-local V17 seed.

        The root stays separately indexable for the deliberately tiny grammar,
        but it is not by itself enough to authorize a replay.  This second
        HMAC prevents a database-only attacker from copying a valid root and
        recomputing public SHA-256 fingerprints on another receipt/manifest.
        The seed fingerprint is already a typed projection of the candidate's
        request and endpoints, and repository lookups re-bind it to the live
        manifest and receipt before this value is trusted.
        """

        scope, scope_key = self._scope_key(context_scope_hash)
        root = _require_opaque_digest(rewrite_commitment, "rewrite_commitment")
        request_id = str(creation_request_id or "").strip()
        normalized_domain = str(domain or "").strip()
        seed = _require_opaque_digest(seed_fingerprint, "seed_fingerprint")
        context_vector = _require_opaque_digest(
            context_cue_vector_fingerprint,
            "context_cue_vector_fingerprint",
        )
        need_vector = _require_opaque_digest(
            need_cue_vector_fingerprint,
            "need_cue_vector_fingerprint",
        )
        normalized_model = str(model_id or "").strip()
        normalized_space = str(embedding_space_id or "").strip()
        try:
            normalized_dimension = int(dimension)
        except (TypeError, ValueError) as error:
            raise ValueError("restricted rewrite provenance dimension is invalid") from error
        normalized_dtype = str(dtype or "").strip()
        grammar = str(grammar_version or "").strip()
        signature = str(signature_version or "").strip()
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", request_id)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", normalized_domain)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", normalized_model)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", normalized_space)
            or normalized_dimension <= 0
            or normalized_dtype != "float32"
            or grammar != _RESTRICTED_REWRITE_GRAMMAR_VERSION
            or signature != _RESTRICTED_REWRITE_SIGNATURE_VERSION
        ):
            raise ValueError("restricted rewrite binding input is invalid")
        payload = {
            "commitment_key_id": self.key_id,
            "context_cue_vector_fingerprint": context_vector,
            "context_scope_hash": scope,
            "creation_request_id": request_id,
            "dimension": normalized_dimension,
            "domain": normalized_domain,
            "dtype": normalized_dtype,
            "embedding_space_id": normalized_space,
            "grammar_version": grammar,
            "model_id": normalized_model,
            "need_cue_vector_fingerprint": need_vector,
            "rewrite_commitment": root,
            "seed_fingerprint": seed,
            "signature_version": signature,
        }
        digest = hmac.new(
            scope_key,
            (
                b"memory-demo/restricted-rewrite/binding/v1\0"
                + json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).hexdigest()
        return "restricted-rewrite-binding:sha256:" + digest

    def manifest_binding_commitment(
        self,
        *,
        context_scope_hash: str,
        rewrite_commitment: str,
        creation_request_id: str,
        domain: str,
        seed_fingerprint: str,
        context_cue_vector_fingerprint: str,
        need_cue_vector_fingerprint: str,
        model_id: str,
        embedding_space_id: str,
        dimension: int,
        dtype: str,
        grammar_version: str,
        signature_version: str,
        manifest_binding_fingerprint: str,
    ) -> str:
        """HMAC the repository-built pending-manifest authority.

        The public runtime-manifest binding contains receipt/association/cue
        identities, source-role closure, validity interval, the seed, and its
        vector provenance.  It is constructed by the repository only after
        those rows exist.  Deriving a short-lived second key from the verified
        first HMAC lets the repository sign that canonical value in its Q1
        transaction without receiving or persisting the environment secret.
        """

        scope, scope_key = self._scope_key(context_scope_hash)
        binding = self.binding_commitment(
            context_scope_hash=context_scope_hash,
            rewrite_commitment=rewrite_commitment,
            creation_request_id=creation_request_id,
            domain=domain,
            seed_fingerprint=seed_fingerprint,
            context_cue_vector_fingerprint=context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=need_cue_vector_fingerprint,
            model_id=model_id,
            embedding_space_id=embedding_space_id,
            dimension=dimension,
            dtype=dtype,
            grammar_version=grammar_version,
            signature_version=signature_version,
        )
        manifest = _require_opaque_digest(
            manifest_binding_fingerprint,
            "manifest_binding_fingerprint",
        )
        signature = str(signature_version or "").strip()
        if signature != _RESTRICTED_REWRITE_SIGNATURE_VERSION:
            raise ValueError("restricted rewrite manifest binding version is invalid")
        post_bind_payload = {
            "binding_commitment": binding,
            "commitment_key_id": self.key_id,
            "context_scope_hash": scope,
            "signature_version": signature,
            "version": "restricted-rewrite-post-bind-key-v1",
        }
        post_bind_key = hmac.new(
            scope_key,
            (
                b"memory-demo/restricted-rewrite/post-bind-key/v1\0"
                + json.dumps(
                    post_bind_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).digest()
        payload = {
            "binding_commitment": binding,
            "commitment_key_id": self.key_id,
            "context_scope_hash": scope,
            "manifest_binding_fingerprint": manifest,
            "signature_version": signature,
            "version": "restricted-rewrite-manifest-binding-v1",
        }
        digest = hmac.new(
            post_bind_key,
            (
                b"memory-demo/restricted-rewrite/manifest-binding/v1\0"
                + json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).hexdigest()
        return "restricted-rewrite-manifest-binding:sha256:" + digest

    def ready_manifest_commitment(
        self,
        *,
        context_scope_hash: str,
        binding_commitment: str,
        manifest_binding_commitment: str,
        manifest_fingerprint: str,
        signature_version: str,
    ) -> str:
        """HMAC the canonical ready manifest after V16 publication.

        ``manifest_fingerprint`` already binds the immutable pending binding,
        ready timestamp/epoch/publication fingerprint, and the one canonical
        V16 contract.  Its public SHA-256 alone is forgeable by a raw database
        writer, so derive a third domain-separated key from the two verified
        prior HMACs before authorizing the ready state.
        """

        scope, scope_key = self._scope_key(context_scope_hash)
        binding = _require_opaque_digest(binding_commitment, "binding_commitment")
        pending_binding = _require_opaque_digest(
            manifest_binding_commitment,
            "manifest_binding_commitment",
        )
        ready_manifest = _require_opaque_digest(
            manifest_fingerprint, "manifest_fingerprint"
        )
        signature = str(signature_version or "").strip()
        if signature != _RESTRICTED_REWRITE_SIGNATURE_VERSION:
            raise ValueError("restricted rewrite ready binding version is invalid")
        post_ready_payload = {
            "binding_commitment": binding,
            "commitment_key_id": self.key_id,
            "context_scope_hash": scope,
            "manifest_binding_commitment": pending_binding,
            "signature_version": signature,
            "version": "restricted-rewrite-post-ready-key-v1",
        }
        post_ready_key = hmac.new(
            scope_key,
            (
                b"memory-demo/restricted-rewrite/post-ready-key/v1\0"
                + json.dumps(
                    post_ready_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).digest()
        payload = {
            "binding_commitment": binding,
            "commitment_key_id": self.key_id,
            "context_scope_hash": scope,
            "manifest_binding_commitment": pending_binding,
            "manifest_fingerprint": ready_manifest,
            "signature_version": signature,
            "version": "restricted-rewrite-ready-manifest-v1",
        }
        digest = hmac.new(
            post_ready_key,
            (
                b"memory-demo/restricted-rewrite/ready-manifest/v1\0"
                + json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            hashlib.sha256,
        ).hexdigest()
        return "restricted-rewrite-ready-manifest:sha256:" + digest


def restricted_rewrite_manifest_binding_signer(
    commitment_key: RestrictedRewriteCommitmentKey | None,
) -> Callable[
    [ContextualRestrictedRewriteGuardDraft, ContextualRevisitRuntimeSeed, str],
    str,
] | None:
    """Return a process-only Q1 signer; never serialize its closure or key.

    The repository invokes this only after it has canonically re-read the
    pending runtime manifest inside the same transaction that creates the
    cues, edge, receipt, manifest, and guard.  The callable itself is kept in
    the in-process materialization map and is deliberately absent from the
    typed draft, its storage payload, logs, and results.
    """

    if commitment_key is None:
        return None

    def sign(
        guard: ContextualRestrictedRewriteGuardDraft,
        runtime_seed: ContextualRevisitRuntimeSeed,
        manifest_binding_fingerprint: str,
    ) -> str:
        if not isinstance(guard, ContextualRestrictedRewriteGuardDraft):
            raise TypeError("restricted rewrite manifest signer needs a typed draft")
        if not isinstance(runtime_seed, ContextualRevisitRuntimeSeed):
            raise TypeError("restricted rewrite manifest signer needs a typed seed")
        if (
            guard.creation_request_id != runtime_seed.creation_request_id
            or guard.domain != runtime_seed.domain
            or guard.context_scope_hash != runtime_seed.context_scope_hash
        ):
            raise ValueError("restricted rewrite draft does not bind its runtime seed")
        expected_binding = commitment_key.binding_commitment(
            context_scope_hash=guard.context_scope_hash,
            rewrite_commitment=guard.rewrite_commitment,
            creation_request_id=guard.creation_request_id,
            domain=guard.domain,
            seed_fingerprint=runtime_seed.seed_fingerprint,
            context_cue_vector_fingerprint=guard.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=guard.need_cue_vector_fingerprint,
            model_id=guard.model_id,
            embedding_space_id=guard.embedding_space_id,
            dimension=guard.dimension,
            dtype=guard.dtype,
            grammar_version=guard.grammar_version,
            signature_version=guard.signature_version,
        )
        if not hmac.compare_digest(expected_binding, guard.binding_commitment):
            raise ValueError("restricted rewrite draft first binding is invalid")
        return commitment_key.manifest_binding_commitment(
            context_scope_hash=guard.context_scope_hash,
            rewrite_commitment=guard.rewrite_commitment,
            creation_request_id=guard.creation_request_id,
            domain=guard.domain,
            seed_fingerprint=runtime_seed.seed_fingerprint,
            context_cue_vector_fingerprint=guard.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=guard.need_cue_vector_fingerprint,
            model_id=guard.model_id,
            embedding_space_id=guard.embedding_space_id,
            dimension=guard.dimension,
            dtype=guard.dtype,
            grammar_version=guard.grammar_version,
            signature_version=guard.signature_version,
            manifest_binding_fingerprint=manifest_binding_fingerprint,
        )

    return sign


def restricted_rewrite_ready_manifest_signer(
    commitment_key: RestrictedRewriteCommitmentKey | None,
) -> Callable[
    [ContextualRestrictedRewriteGuard, ContextualRevisitRuntimeManifest], str
] | None:
    """Return the process-only signer for one pending-to-ready transition.

    It is intentionally invoked only inside the repository transaction that
    has just constructed and re-bound a ready manifest.  It cannot be used to
    repair or retrospectively bless an already-ready sidecar.
    """

    if commitment_key is None:
        return None

    def sign(
        guard: ContextualRestrictedRewriteGuard,
        manifest: ContextualRevisitRuntimeManifest,
    ) -> str:
        if not isinstance(guard, ContextualRestrictedRewriteGuard):
            raise TypeError("restricted rewrite ready signer needs a typed guard")
        if not isinstance(manifest, ContextualRevisitRuntimeManifest):
            raise TypeError("restricted rewrite ready signer needs a typed manifest")
        if guard.ready_manifest_commitment:
            raise ValueError("restricted rewrite guard is already ready-bound")
        if (
            manifest.state != "ready"
            or int(guard.creation_receipt_id) != int(manifest.creation_receipt_id)
            or int(guard.association_id) != int(manifest.association_id)
            or str(guard.domain) != str(manifest.seed.domain)
            or str(guard.context_scope_hash) != str(manifest.seed.context_scope_hash)
            or str(guard.seed_fingerprint) != str(manifest.seed.seed_fingerprint)
            or str(guard.commitment_key_id) != str(commitment_key.key_id)
        ):
            raise ValueError("restricted rewrite ready signer binding is invalid")
        expected_binding = commitment_key.binding_commitment(
            context_scope_hash=guard.context_scope_hash,
            rewrite_commitment=guard.rewrite_commitment,
            creation_request_id=manifest.seed.creation_request_id,
            domain=guard.domain,
            seed_fingerprint=manifest.seed.seed_fingerprint,
            context_cue_vector_fingerprint=(
                manifest.context_cue_vector_fingerprint
            ),
            need_cue_vector_fingerprint=manifest.need_cue_vector_fingerprint,
            model_id=manifest.model_id,
            embedding_space_id=manifest.embedding_space_id,
            dimension=manifest.dimension,
            dtype=manifest.dtype,
            grammar_version=guard.grammar_version,
            signature_version=guard.signature_version,
        )
        if not hmac.compare_digest(expected_binding, guard.binding_commitment):
            raise ValueError("restricted rewrite guard first binding is invalid")
        expected_pending_binding = commitment_key.manifest_binding_commitment(
            context_scope_hash=guard.context_scope_hash,
            rewrite_commitment=guard.rewrite_commitment,
            creation_request_id=manifest.seed.creation_request_id,
            domain=guard.domain,
            seed_fingerprint=manifest.seed.seed_fingerprint,
            context_cue_vector_fingerprint=(
                manifest.context_cue_vector_fingerprint
            ),
            need_cue_vector_fingerprint=manifest.need_cue_vector_fingerprint,
            model_id=manifest.model_id,
            embedding_space_id=manifest.embedding_space_id,
            dimension=manifest.dimension,
            dtype=manifest.dtype,
            grammar_version=guard.grammar_version,
            signature_version=guard.signature_version,
            manifest_binding_fingerprint=manifest.binding_fingerprint,
        )
        if not hmac.compare_digest(
            expected_pending_binding, guard.manifest_binding_commitment
        ):
            raise ValueError("restricted rewrite guard pending binding is invalid")
        return commitment_key.ready_manifest_commitment(
            context_scope_hash=guard.context_scope_hash,
            binding_commitment=guard.binding_commitment,
            manifest_binding_commitment=guard.manifest_binding_commitment,
            manifest_fingerprint=manifest.manifest_fingerprint,
            signature_version=guard.signature_version,
        )

    return sign


def _restricted_rewrite_term(value: str, *, maximum_words: int) -> str | None:
    normalized = normalize_query_text(value).casefold()
    if not _RESTRICTED_REWRITE_TERM_RE.fullmatch(normalized):
        return None
    words = tuple(normalized.split())
    if not words or len(words) > maximum_words:
        return None
    if any(word in _RESTRICTED_REWRITE_REJECT_TOKENS for word in words):
        return None
    return normalized


def parse_restricted_rewrite_question(question: str) -> RestrictedRewriteIR | None:
    """Parse only two controlled surface forms, otherwise return a safe miss.

    This is intentionally *not* a natural-language semantic parser.  The two
    accepted forms differ only in an explicit subject/predicate word order:
    ``What is the <predicate> of <subject>?`` and
    ``For <subject>, what is the <predicate>?``.  Every other question,
    including non-English text, aliases, pronouns, dates, numbers, negation,
    comparisons, conditions, modality, quantifiers, and multi-object wording
    is rejected before any lookup can occur.
    """

    normalized = normalize_query_text(str(question or "")).casefold()
    if not normalized or len(normalized) > 180 or any(char.isdigit() for char in normalized):
        return None
    match = _RESTRICTED_REWRITE_OF_TEMPLATE_RE.fullmatch(normalized)
    if match is None:
        match = _RESTRICTED_REWRITE_FOR_TEMPLATE_RE.fullmatch(normalized)
    if match is None:
        return None
    subject = _restricted_rewrite_term(match.group("subject"), maximum_words=6)
    predicate = _restricted_rewrite_term(match.group("predicate"), maximum_words=4)
    if subject is None or predicate is None:
        return None
    return RestrictedRewriteIR(subject=subject, predicate=predicate)


def restricted_rewrite_guard_draft(
    *,
    question: str,
    requirements: RequirementResolution,
    creation_request_id: str,
    domain: str,
    context_scope_hash: str,
    runtime_seed: ContextualRevisitRuntimeSeed,
    context_cue_vector_fingerprint: str,
    need_cue_vector_fingerprint: str,
    model_id: str,
    embedding_space_id: str,
    dimension: int,
    dtype: str,
    commitment_key: RestrictedRewriteCommitmentKey | None,
) -> ContextualRestrictedRewriteGuardDraft | None:
    """Produce a Q1 guard only for a local grammar and configured HMAC key."""

    if (
        commitment_key is None
        or not isinstance(requirements, RequirementResolution)
        or not isinstance(runtime_seed, ContextualRevisitRuntimeSeed)
    ):
        return None
    normalized_question = normalize_query_text(str(question or ""))
    if (
        requirements.request_mode != "factual"
        or requirements.status != "resolved"
        or len(requirements.requirements) != 1
    ):
        return None
    slot = requirements.requirements[0]
    if (
        not slot.required
        or str(slot.support_mode) != "alternative"
        or normalize_query_text(str(slot.question or "")) != normalized_question
    ):
        return None
    ir = parse_restricted_rewrite_question(normalized_question)
    if ir is None:
        return None
    try:
        commitment = commitment_key.commitment(
            context_scope_hash=context_scope_hash,
            ir=ir,
        )
        if (
            str(runtime_seed.creation_request_id) != str(creation_request_id)
            or str(runtime_seed.domain) != str(domain)
            or str(runtime_seed.context_scope_hash) != str(context_scope_hash)
            or str(runtime_seed.context_hash)
            != exact_revisit_context_hash(
                normalized_question,
                context_scope_hash=context_scope_hash,
            )
            or str(runtime_seed.source_request_hash)
            != exact_revisit_request_hash(normalized_question)
        ):
            return None
        binding_commitment = commitment_key.binding_commitment(
            context_scope_hash=context_scope_hash,
            rewrite_commitment=commitment,
            creation_request_id=creation_request_id,
            domain=domain,
            seed_fingerprint=runtime_seed.seed_fingerprint,
            context_cue_vector_fingerprint=context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=need_cue_vector_fingerprint,
            model_id=model_id,
            embedding_space_id=embedding_space_id,
            dimension=dimension,
            dtype=dtype,
            grammar_version=ir.grammar_version,
            signature_version=_RESTRICTED_REWRITE_SIGNATURE_VERSION,
        )
        return ContextualRestrictedRewriteGuardDraft(
            creation_request_id=creation_request_id,
            domain=domain,
            context_scope_hash=context_scope_hash,
            rewrite_commitment=commitment,
            binding_commitment=binding_commitment,
            context_cue_vector_fingerprint=context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=need_cue_vector_fingerprint,
            model_id=model_id,
            embedding_space_id=embedding_space_id,
            dimension=dimension,
            dtype=dtype,
            commitment_key_id=commitment_key.key_id,
            grammar_version=ir.grammar_version,
        )
    except (TypeError, ValueError, UnicodeError):
        return None


def exact_revisit_requirements_fingerprint(
    requirements: RequirementResolution,
) -> str:
    """Fingerprint every coverage-relevant request obligation.

    The output is opaque; it is safe to compare with v16 but is deliberately
    not a serialised replacement for the live RequirementResolution.
    """

    if not isinstance(requirements, RequirementResolution):
        raise TypeError("exact revisit requires RequirementResolution")
    payload = {
        "request_mode": str(requirements.request_mode),
        "status": str(requirements.status),
        "requirements": [
            {
                "slot_id": slot.slot_id,
                "question_hash": hashlib.sha256(
                    normalize_query_text(slot.question).encode("utf-8")
                ).hexdigest(),
                "required": bool(slot.required),
                "query_id": slot.query_id,
                "query_refs": list(slot.query_refs),
                "origin": slot.origin,
                "support_mode": slot.support_mode,
                "clause_ids": list(slot.clause_ids),
                "subject_terms": list(slot.subject_terms),
                "object_terms": list(slot.object_terms),
                "relation_hint": slot.relation_hint,
                "temporal_hint": slot.temporal_hint,
                "negation_hint": slot.negation_hint,
                "modality_hint": slot.modality_hint,
                "epistemic_hint": slot.epistemic_hint,
            }
            for slot in requirements.requirements
        ],
    }
    return _canonical_digest("revisit-requirements", payload)


def _strict_bundle_vector(bundle: QueryVectorBundle, physical_id: str) -> np.ndarray:
    """Validate a supplied vector without normalising/coercing it in place."""

    try:
        raw = np.asarray(bundle.vector_for_physical(physical_id))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("exact revisit vector reference is unavailable") from error
    if raw.dtype != np.float32:
        raise ValueError("exact revisit vectors must be float32")
    if raw.shape != (int(bundle.dimension),):
        raise ValueError("exact revisit vector shape is invalid")
    if not np.all(np.isfinite(raw)):
        raise ValueError("exact revisit vector contains NaN or infinity")
    normalized = normalize_embedding(raw, int(bundle.dimension))
    if not np.allclose(raw, normalized, rtol=1e-5, atol=1e-6):
        raise ValueError("exact revisit vector must already be normalized")
    return raw


def exact_revisit_slot_need_bindings(
    requirements: RequirementResolution,
    bundle: QueryVectorBundle,
) -> tuple[ContextualRevisitSlotNeedBinding, ...]:
    """Bind each required need to an observed request-local vector.

    The ordinary form is one atomic planner binding per required slot.  A
    deliberately narrower form also exists for the V17 exact-revisit shape:
    one required slot whose own question is literally the request's whole
    question may bind that already-observed whole vector.  This preserves
    rich requirement fields while allowing Q1 retrieval to retain additional
    planner paraphrases; it neither reconstructs a future need nor chooses
    among those paraphrases.
    """

    if not isinstance(requirements, RequirementResolution):
        raise TypeError("exact revisit requires RequirementResolution")
    if not isinstance(bundle, QueryVectorBundle):
        raise TypeError("exact revisit requires QueryVectorBundle")
    if int(bundle.dimension) <= 0 or not str(bundle.embedding_space_id).strip():
        raise ValueError("exact revisit bundle metadata is invalid")

    whole_bindings = tuple(
        item
        for item in bundle.logical_bindings
        if item.role == "whole" and item.physical_id == bundle.whole_physical_id
    )
    if not whole_bindings:
        raise ValueError("exact revisit bundle needs a whole-query binding")
    _strict_bundle_vector(bundle, bundle.whole_physical_id)
    required_slots = tuple(slot for slot in requirements.requirements if slot.required)

    result: list[ContextualRevisitSlotNeedBinding] = []
    for slot in required_slots:
        if not str(slot.slot_id).strip() or not str(slot.query_id).strip():
            raise ValueError("exact revisit required slot lacks a stable query identity")
        bindings = tuple(
            item
            for item in bundle.bindings_for_slot(slot.slot_id)
            if item.role == "atomic"
        )
        uses_whole_question_binding = (
            len(bindings) != 1
            and len(required_slots) == 1
            and len(whole_bindings) == 1
            and normalize_query_text(slot.question)
            == normalize_query_text(whole_bindings[0].text)
        )
        if len(bindings) == 1:
            binding = bindings[0]
        elif uses_whole_question_binding:
            # Multiple Q1-only planner retrieval cues are not candidate
            # future needs.  The unique whole-question binding is already
            # present in this request and is the only safe exact template.
            binding = whole_bindings[0]
        else:
            raise ValueError(
                "exact revisit required slot needs exactly one atomic vector binding"
            )
        if (
            (
                not uses_whole_question_binding
                and str(binding.query_id) != str(slot.query_id)
            )
            or str(binding.embedding_space_id) != str(bundle.embedding_space_id)
            or not str(binding.text_hash).strip()
            or not str(binding.physical_id).strip()
        ):
            raise ValueError("exact revisit atomic slot binding is inconsistent")
        _strict_bundle_vector(bundle, binding.physical_id)
        result.append(
            ContextualRevisitSlotNeedBinding(
                slot_id=_canonical_digest("revisit-slot", slot.slot_id),
                need_query_id=_canonical_digest("revisit-need-query", binding.query_id),
                need_hash=_canonical_digest(
                    "revisit-need", {"text_hash": str(binding.text_hash)}
                ),
            )
        )
    if not result:
        raise ValueError("exact revisit needs at least one required slot")
    return tuple(sorted(result, key=lambda item: item.slot_id))


def exact_revisit_budget_fingerprint(budget: EvidenceSelectionBudget) -> str:
    if not isinstance(budget, EvidenceSelectionBudget):
        raise TypeError("exact revisit budget must use EvidenceSelectionBudget")
    return _canonical_digest(
        "revisit-budget",
        {
            "episode_limit": int(budget.episode_limit),
            "source_fact_limit": budget.source_fact_limit,
            "delivery_token_limit": budget.delivery_token_limit,
        },
    )


def exact_revisit_anchor_manifest_fingerprint(
    anchor_activations: Mapping[int, float] | Sequence[tuple[int, float]],
) -> str:
    rows = _normalise_anchor_activations(anchor_activations)
    return _canonical_digest(
        "revisit-anchor-manifest",
        [{"episode_id": episode_id, "activation": activation} for episode_id, activation in rows],
    )


def exact_revisit_source_closure_fingerprint(
    source_facts: Sequence[SourceFactRef],
) -> str:
    """Hash complete fact identities, never a source key alone."""

    rows = _normalise_source_facts(source_facts)
    return _canonical_digest(
        "revisit-source-closure",
        [
            {
                "source_revision_id": item.source_revision_id,
                "record_span": list(item.record_span),
                "span_hash": item.span_hash,
                "raw_span_hash": item.raw_span_hash,
            }
            for item in rows
        ],
    )


def exact_revisit_source_fact_refs_fingerprint(
    source_facts: Sequence[SourceFactRef],
) -> str:
    return _canonical_digest(
        "revisit-source-fact-refs",
        [item.fact_id for item in _normalise_source_facts(source_facts)],
    )


def exact_revisit_source_fact_roles_fingerprint(
    contributions: Sequence[CandidateContribution],
) -> str:
    """Bind source facts to current base/contextual roles without prose."""

    rows: list[dict[str, object]] = []
    for contribution in sorted(
        contributions,
        key=lambda item: (
            int(item.episode_id),
            str(item.lane),
            -1 if item.edge_id is None else int(item.edge_id),
            str(item.contribution_id),
        ),
    ):
        rows.append(
            {
                "episode_id": int(contribution.episode_id),
                "lane": str(contribution.lane),
                "edge_id": contribution.edge_id,
                "slot_id": str(contribution.slot_id),
                "source_fact_ids": [item.fact_id for item in contribution.source_facts],
                "supports": [
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
            }
        )
    return _canonical_digest("revisit-source-fact-roles", rows)


@dataclass(frozen=True, slots=True)
class ExactRevisitSourceMappingProof:
    """One historic source-grounded target mapping held only in RAM.

    A contextual cosine match never becomes factual support.  This proof is
    the separate, pre-existing mapping that allows a source-bound support
    contribution to be rebuilt after the current source closure has confirmed
    the exact fact identity again.  It contains no source/query/answer prose.
    """

    association_id: int
    target_episode_id: int
    slot_id: str
    clause_ids: tuple[str, ...]
    source_fact_id: str
    mapping_ref: str

    def __post_init__(self) -> None:
        for field_name in ("association_id", "target_episode_id"):
            try:
                value = int(getattr(self, field_name))
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"exact revisit mapping proof {field_name} must be positive"
                ) from error
            if value <= 0:
                raise ValueError(
                    f"exact revisit mapping proof {field_name} must be positive"
                )
            object.__setattr__(self, field_name, value)
        slot_id = str(self.slot_id or "").strip()
        if not slot_id:
            raise ValueError("exact revisit mapping proof slot_id is required")
        raw_clause_ids = self.clause_ids
        if isinstance(raw_clause_ids, str):
            raw_clause_ids = (raw_clause_ids,)
        clause_ids = tuple(
            sorted({str(item).strip() for item in raw_clause_ids if str(item).strip()})
        )
        if not clause_ids:
            raise ValueError("exact revisit mapping proof clause_ids are required")
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "clause_ids", clause_ids)
        object.__setattr__(
            self,
            "source_fact_id",
            _require_opaque_digest(self.source_fact_id, "mapping_proof.source_fact_id"),
        )
        object.__setattr__(
            self,
            "mapping_ref",
            _require_opaque_digest(self.mapping_ref, "mapping_proof.mapping_ref"),
        )


def exact_revisit_aggregate_mapping_ref(
    *,
    target_episode_id: int,
    slot_id: str,
    clause_ids: Sequence[str],
    source_fact_id: str,
    verification_refs: Sequence[str],
) -> str:
    """Bind pre-existing target mappings into one opaque exact-lane proof ID.

    The normal V3 selector can retain several source-grounded mapping refs for
    one target/slot.  T15's exact proof is deliberately one edge × target ×
    slot record, so this helper combines *only those existing opaque refs*;
    it never turns a matcher score into a factual mapping.
    """

    try:
        target_id = int(target_episode_id)
    except (TypeError, ValueError) as error:
        raise TypeError("exact revisit target episode id is invalid") from error
    if target_id <= 0:
        raise ValueError("exact revisit target episode id must be positive")
    normalized_slot = str(slot_id or "").strip()
    if not normalized_slot:
        raise ValueError("exact revisit target mapping slot is required")
    normalized_clauses = tuple(
        sorted({str(value).strip() for value in clause_ids if str(value).strip()})
    )
    if not normalized_clauses:
        raise ValueError("exact revisit target mapping clauses are required")
    fact_id = _require_opaque_digest(source_fact_id, "target_mapping.source_fact_id")
    refs = tuple(sorted({str(value).strip() for value in verification_refs if str(value).strip()}))
    if not refs:
        raise ValueError("exact revisit target mapping needs verification refs")
    return _canonical_digest(
        "revisit-target-mapping",
        {
            "target_episode_id": target_id,
            "slot_id": normalized_slot,
            "clause_ids": list(normalized_clauses),
            "source_fact_id": fact_id,
            "verification_refs": list(refs),
        },
    )


@dataclass(frozen=True, slots=True)
class ExactRevisitTargetMappingDraft:
    """Redacted engine-to-application target proof before an edge ID exists.

    It intentionally contains no source/query/answer prose or vectors.  The
    application may attach the canonical ready receipt's association ID, but
    must never infer a target, fact, or mapping ref that the engine omitted.
    """

    target_episode_id: int
    slot_id: str
    clause_ids: tuple[str, ...]
    source_fact_id: str
    mapping_ref: str

    def __post_init__(self) -> None:
        try:
            target_episode_id = int(self.target_episode_id)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit draft target episode id is invalid") from error
        if target_episode_id <= 0:
            raise ValueError("exact revisit draft target episode id must be positive")
        slot_id = str(self.slot_id or "").strip()
        if not slot_id:
            raise ValueError("exact revisit draft slot is required")
        raw_clause_ids = self.clause_ids
        if isinstance(raw_clause_ids, str):
            raw_clause_ids = (raw_clause_ids,)
        clause_ids = tuple(
            sorted({str(value).strip() for value in raw_clause_ids if str(value).strip()})
        )
        if not clause_ids:
            raise ValueError("exact revisit draft clauses are required")
        object.__setattr__(self, "target_episode_id", target_episode_id)
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "clause_ids", clause_ids)
        object.__setattr__(
            self,
            "source_fact_id",
            _require_opaque_digest(self.source_fact_id, "draft.source_fact_id"),
        )
        object.__setattr__(
            self,
            "mapping_ref",
            _require_opaque_digest(self.mapping_ref, "draft.mapping_ref"),
        )

    def proof_for_association(self, association_id: int) -> ExactRevisitSourceMappingProof:
        """Attach only a verified canonical association identity."""

        return ExactRevisitSourceMappingProof(
            association_id=association_id,
            target_episode_id=self.target_episode_id,
            slot_id=self.slot_id,
            clause_ids=self.clause_ids,
            source_fact_id=self.source_fact_id,
            mapping_ref=self.mapping_ref,
        )


@dataclass(frozen=True, slots=True)
class ContextualRevisitContractDraft:
    """Complete redacted Q1 material needed to write one ready v16 contract.

    This is deliberately transient.  The engine creates it only from live
    selector/source-closure/vector objects, and the application later joins it
    to the canonical ready receipt.  It is not a durable Q2 ticket and cannot
    reconstruct requirements, vectors, or answer text after a restart.
    """

    creation_request_id: str
    source_request_hash: str
    domain: str
    anchor_episode_id: int
    context_hash: str
    slot_need_bindings: tuple[ContextualRevisitSlotNeedBinding, ...]
    requirements_fingerprint: str
    source_closure_fingerprint: str
    retrieval_policy_fingerprint: str
    budget_fingerprint: str
    anchor_manifest_fingerprint: str
    source_fact_refs_fingerprint: str
    target_mapping: ExactRevisitTargetMappingDraft

    def __post_init__(self) -> None:
        creation_request_id = str(self.creation_request_id or "").strip()
        if not creation_request_id or "\n" in creation_request_id or "\r" in creation_request_id:
            raise ValueError("exact revisit draft creation request id is required")
        source_request_hash = str(self.source_request_hash or "").strip()
        if not _REQUEST_HASH_RE.fullmatch(source_request_hash):
            raise ValueError("exact revisit draft source request hash is invalid")
        domain = str(self.domain or "").strip()
        if not domain or "\n" in domain or "\r" in domain or "," in domain:
            raise ValueError("exact revisit draft domain is invalid")
        try:
            anchor_episode_id = int(self.anchor_episode_id)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit draft anchor episode id is invalid") from error
        if anchor_episode_id <= 0:
            raise ValueError("exact revisit draft anchor episode id must be positive")
        if not isinstance(self.target_mapping, ExactRevisitTargetMappingDraft):
            raise TypeError("exact revisit draft needs a typed target mapping")
        if anchor_episode_id == int(self.target_mapping.target_episode_id):
            raise ValueError("exact revisit draft cannot use a self target")
        raw_bindings = tuple(self.slot_need_bindings or ())
        if not raw_bindings or any(
            not isinstance(item, ContextualRevisitSlotNeedBinding)
            for item in raw_bindings
        ):
            raise ValueError("exact revisit draft needs typed slot bindings")
        bindings = tuple(
            sorted(
                raw_bindings,
                key=lambda item: (item.slot_id, item.need_query_id, item.need_hash),
            )
        )
        if len({item.slot_id for item in bindings}) != len(bindings):
            raise ValueError("exact revisit draft slot bindings are ambiguous")
        object.__setattr__(self, "creation_request_id", creation_request_id)
        object.__setattr__(self, "source_request_hash", source_request_hash)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "anchor_episode_id", anchor_episode_id)
        object.__setattr__(self, "slot_need_bindings", bindings)
        for field_name in (
            "context_hash",
            "requirements_fingerprint",
            "source_closure_fingerprint",
            "retrieval_policy_fingerprint",
            "budget_fingerprint",
            "anchor_manifest_fingerprint",
            "source_fact_refs_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_opaque_digest(getattr(self, field_name), field_name),
            )

    def source_fact_roles_fingerprint(self, association_id: int) -> str:
        """Derive the edge-bound role fingerprint without filling any gaps."""

        proof = self.target_mapping.proof_for_association(association_id)
        return exact_revisit_mapping_roles_fingerprint_from_fact_ids(
            {int(proof.target_episode_id): str(proof.source_fact_id)},
            base_slot_support={},
            target_source_mapping_proofs=(proof,),
        )


def exact_revisit_mapping_roles_fingerprint_from_fact_ids(
    source_fact_ids_by_episode: Mapping[int, str],
    *,
    base_slot_support: Mapping[int, Sequence[str]],
    target_source_mapping_proofs: Sequence[ExactRevisitSourceMappingProof],
) -> str:
    """Fingerprint source-mapping roles from already-verified opaque fact IDs.

    The application uses this after it has attached the canonical receipt's
    association ID to a typed engine draft.  It is intentionally not a source
    closure constructor: absent or malformed fact IDs remain blank, causing a
    later exact input to mismatch rather than silently acquire support.
    """

    facts: dict[int, str] = {}
    for raw_episode_id, raw_fact_id in source_fact_ids_by_episode.items():
        try:
            episode_id = int(raw_episode_id)
        except (TypeError, ValueError):
            continue
        if episode_id <= 0:
            continue
        fact_id = str(raw_fact_id or "").strip()
        if fact_id and _OPAQUE_DIGEST_RE.fullmatch(fact_id):
            facts[episode_id] = fact_id
    rows: list[dict[str, object]] = []
    for raw_episode_id, slot_ids in sorted(base_slot_support.items()):
        episode_id = int(raw_episode_id)
        fact_id = facts.get(episode_id, "")
        if not fact_id:
            rows.append(
                {
                    "role": "base_source_mapping",
                    "episode_id": episode_id,
                    "slot_ids": sorted(str(item) for item in slot_ids),
                    "source_fact_id": "",
                }
            )
            continue
        rows.append(
            {
                "role": "base_source_mapping",
                "episode_id": episode_id,
                "slot_ids": sorted(str(item) for item in slot_ids),
                "source_fact_id": fact_id,
            }
        )
    for proof in sorted(
        target_source_mapping_proofs,
        key=lambda item: (
            item.association_id,
            item.target_episode_id,
            item.slot_id,
            item.mapping_ref,
        ),
    ):
        rows.append(
            {
                "role": "contextual_source_mapping",
                "association_id": int(proof.association_id),
                "episode_id": int(proof.target_episode_id),
                "slot_id": proof.slot_id,
                "clause_ids": list(proof.clause_ids),
                "expected_source_fact_id": proof.source_fact_id,
                "current_source_fact_id": facts.get(
                    int(proof.target_episode_id), ""
                ),
                "mapping_ref": proof.mapping_ref,
            }
        )
    return _canonical_digest("revisit-source-fact-roles", rows)


def exact_revisit_mapping_roles_fingerprint(
    source_facts_by_episode: Mapping[int, SourceFactRef],
    *,
    base_slot_support: Mapping[int, Sequence[str]],
    target_source_mapping_proofs: Sequence[ExactRevisitSourceMappingProof],
) -> str:
    """Bind full current fact IDs to source-mapping roles, not ranking scores."""

    fact_ids: dict[int, str] = {}
    for raw_episode_id, fact in source_facts_by_episode.items():
        try:
            episode_id = int(raw_episode_id)
        except (TypeError, ValueError):
            continue
        if episode_id <= 0 or not isinstance(fact, SourceFactRef):
            continue
        fact_ids[episode_id] = fact.fact_id
    return exact_revisit_mapping_roles_fingerprint_from_fact_ids(
        fact_ids,
        base_slot_support=base_slot_support,
        target_source_mapping_proofs=target_source_mapping_proofs,
    )


def exact_revisit_mapping_contribution_id(
    proof: ExactRevisitSourceMappingProof,
) -> str:
    """Stable opaque ID for the separately verified, edge-scoped mapping."""

    if not isinstance(proof, ExactRevisitSourceMappingProof):
        raise TypeError("exact revisit mapping contribution needs a typed proof")
    return _canonical_digest(
        "revisit-source-mapping-contribution",
        {
            "association_id": proof.association_id,
            "target_episode_id": proof.target_episode_id,
            "slot_id": proof.slot_id,
            "clause_ids": list(proof.clause_ids),
            "source_fact_id": proof.source_fact_id,
            "mapping_ref": proof.mapping_ref,
        },
    )


def exact_revisit_policy_fingerprint(
    config,
    matcher,
    *,
    endpoint_limit: int,
) -> str:
    """Fingerprint only knobs that change this local exact lane's meaning."""

    retrieval = getattr(config, "retrieval", None)
    if retrieval is None or matcher is None:
        raise ValueError("exact revisit policy requires retrieval config and matcher")
    return _canonical_digest(
        "revisit-policy",
        {
            "answer_episode_limit": int(retrieval.answer_episode_limit),
            "contextual_enabled": bool(retrieval.contextual_association_enabled),
            "contextual_shadow": bool(retrieval.contextual_association_shadow),
            "endpoint_limit": int(endpoint_limit),
            "context_threshold": float(getattr(matcher, "context_threshold", math.nan)),
            "need_threshold": float(getattr(matcher, "need_threshold", math.nan)),
            "combine_mode": str(getattr(matcher, "combine_mode", "")),
            "edge_top_k": int(getattr(matcher, "edge_top_k", 0)),
            "embedding_space_id": str(getattr(matcher, "embedding_space_id", "")),
        },
    )


def _normalise_anchor_activations(
    value: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized: dict[int, float] = {}
    for raw_id, raw_score in items:
        if isinstance(raw_id, bool):
            raise TypeError("exact revisit anchor ids must be positive integers")
        try:
            episode_id = int(raw_id)
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit anchor activations are invalid") from error
        if episode_id <= 0 or not math.isfinite(score) or score <= 0.0:
            raise ValueError("exact revisit anchor activations must be positive finite")
        previous = normalized.get(episode_id)
        if previous is not None and not math.isclose(previous, score):
            raise ValueError("exact revisit anchor activation is ambiguous")
        normalized[episode_id] = score
    if not normalized:
        raise ValueError("exact revisit needs at least one independent anchor")
    return tuple(sorted(normalized.items()))


def _normalise_episode_ids(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    normalized: set[int] = set()
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise TypeError(f"exact revisit {field_name} must contain positive integers")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"exact revisit {field_name} must contain positive integers"
            ) from error
        if value <= 0:
            raise ValueError(f"exact revisit {field_name} must contain positive integers")
        normalized.add(value)
    if not normalized:
        raise ValueError(f"exact revisit {field_name} cannot be empty")
    return tuple(sorted(normalized))


def _normalise_slot_support(
    value: Mapping[int, Sequence[str]] | Sequence[tuple[int, Sequence[str]]],
    base_episode_ids: Sequence[int],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    base_ids = set(base_episode_ids)
    normalized: dict[int, tuple[str, ...]] = {}
    for raw_episode_id, raw_slots in items:
        try:
            episode_id = int(raw_episode_id)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit base support episode id is invalid") from error
        if episode_id not in base_ids:
            raise ValueError("exact revisit base support is outside the base manifest")
        if isinstance(raw_slots, str):
            raw_slots = (raw_slots,)
        slots = tuple(sorted({str(item).strip() for item in raw_slots if str(item).strip()}))
        if not slots:
            raise ValueError("exact revisit base support slot set cannot be empty")
        existing = normalized.get(episode_id)
        if existing is not None and existing != slots:
            raise ValueError("exact revisit base support is ambiguous")
        normalized[episode_id] = slots
    return tuple(sorted(normalized.items()))


def _normalise_mask_ids(values: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise TypeError("exact revisit masked edge ids must be positive integers")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit masked edge ids must be positive integers") from error
        if value <= 0:
            raise ValueError("exact revisit masked edge ids must be positive")
        normalized.add(value)
    return tuple(sorted(normalized))


def _normalise_mask_contributions(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _normalise_source_facts(
    values: Sequence[SourceFactRef],
) -> tuple[SourceFactRef, ...]:
    by_identity: dict[tuple[str, tuple[str, ...], str, str], SourceFactRef] = {}
    for item in values:
        if not isinstance(item, SourceFactRef):
            raise TypeError("exact revisit source closure needs SourceFactRef values")
        by_identity.setdefault(item.identity_key, item)
    if not by_identity:
        raise ValueError("exact revisit source closure cannot be empty")
    return tuple(
        by_identity[key]
        for key in sorted(by_identity, key=lambda key: (key[0], key[1], key[2], key[3]))
    )


@dataclass(frozen=True, slots=True)
class ExactRevisitInput:
    """Complete caller-supplied runtime material for the T15 exact path.

    This object is intentionally not durable and never contains an answer.
    It may contain process-local vectors and requirement text because the
    engine needs them to rerun the normal local matcher, target gate and V3
    selector.  The v16 contract stores only the opaque fingerprints produced
    from this material.
    """

    domain: str
    context_hash: str
    requirements: RequirementResolution
    intent: QueryIntent
    query_vector_bundle: QueryVectorBundle
    budget: EvidenceSelectionBudget
    retrieval_policy_fingerprint: str
    source_closure_fingerprint: str
    source_fact_roles_fingerprint: str
    source_fact_refs_fingerprint: str
    evaluation_as_of: str
    endpoint_limit: int
    anchor_activations: Mapping[int, float] | Sequence[tuple[int, float]]
    base_episode_ids: Sequence[int]
    base_slot_support: (
        Mapping[int, Sequence[str]] | Sequence[tuple[int, Sequence[str]]]
    ) = field(default_factory=tuple)
    target_source_mapping_proofs: Sequence[ExactRevisitSourceMappingProof] = field(
        default_factory=tuple
    )
    masked_edge_ids: Sequence[int] = field(default_factory=tuple)
    masked_contribution_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        domain = str(self.domain or "").strip()
        if not domain or "\n" in domain or "\r" in domain or "," in domain:
            raise ValueError("exact revisit domain is required")
        if not isinstance(self.requirements, RequirementResolution):
            raise TypeError("exact revisit requires RequirementResolution")
        if (
            self.requirements.request_mode != "factual"
            or self.requirements.status != "resolved"
        ):
            raise ValueError(
                "exact revisit requires a resolved factual requirement contract"
            )
        if not isinstance(self.intent, QueryIntent):
            raise TypeError("exact revisit requires QueryIntent")
        if not isinstance(self.query_vector_bundle, QueryVectorBundle):
            raise TypeError("exact revisit requires QueryVectorBundle")
        if not isinstance(self.budget, EvidenceSelectionBudget):
            raise TypeError("exact revisit requires EvidenceSelectionBudget")
        try:
            endpoint_limit = int(self.endpoint_limit)
        except (TypeError, ValueError) as error:
            raise TypeError("exact revisit endpoint_limit must be positive") from error
        if endpoint_limit <= 0:
            raise ValueError("exact revisit endpoint_limit must be positive")
        evaluation_as_of = str(self.evaluation_as_of or "").strip()
        if not evaluation_as_of:
            raise ValueError("exact revisit evaluation_as_of is required")
        base_ids = _normalise_episode_ids(self.base_episode_ids, "base_episode_ids")
        anchors = _normalise_anchor_activations(self.anchor_activations)
        if not {episode_id for episode_id, _ in anchors}.issubset(set(base_ids)):
            raise ValueError("exact revisit anchors must be in the base manifest")
        supports = _normalise_slot_support(self.base_slot_support, base_ids)
        required_slot_ids = {
            str(slot.slot_id) for slot in self.requirements.requirements if slot.required
        }
        if any(
            slot_id not in required_slot_ids
            for _episode_id, slot_ids in supports
            for slot_id in slot_ids
        ):
            raise ValueError("exact revisit base support names an unknown required slot")
        raw_proofs = tuple(self.target_source_mapping_proofs or ())
        if not raw_proofs or any(
            not isinstance(item, ExactRevisitSourceMappingProof) for item in raw_proofs
        ):
            raise ValueError("exact revisit needs source mapping proofs")
        proofs = tuple(
            sorted(
                raw_proofs,
                key=lambda item: (
                    item.association_id,
                    item.target_episode_id,
                    item.slot_id,
                    item.mapping_ref,
                ),
            )
        )
        proof_keys = {
            (item.association_id, item.target_episode_id, item.slot_id)
            for item in proofs
        }
        if len(proof_keys) != len(proofs):
            raise ValueError("exact revisit source mapping proof is ambiguous")
        if any(item.slot_id not in required_slot_ids for item in proofs):
            raise ValueError("exact revisit mapping proof names an unknown required slot")

        # Copy the mutable compatibility intent so the public caller cannot
        # alter answer semantics after a supposedly frozen input was accepted.
        intent = QueryIntent.from_dict(asdict(self.intent))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "context_hash", _require_opaque_digest(self.context_hash, "context_hash"))
        object.__setattr__(
            self,
            "retrieval_policy_fingerprint",
            _require_opaque_digest(
                self.retrieval_policy_fingerprint, "retrieval_policy_fingerprint"
            ),
        )
        for field_name in (
            "source_closure_fingerprint",
            "source_fact_roles_fingerprint",
            "source_fact_refs_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_opaque_digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "endpoint_limit", endpoint_limit)
        object.__setattr__(self, "evaluation_as_of", evaluation_as_of)
        object.__setattr__(self, "anchor_activations", anchors)
        object.__setattr__(self, "base_episode_ids", base_ids)
        object.__setattr__(self, "base_slot_support", supports)
        object.__setattr__(self, "target_source_mapping_proofs", proofs)
        object.__setattr__(self, "masked_edge_ids", _normalise_mask_ids(self.masked_edge_ids))
        object.__setattr__(
            self,
            "masked_contribution_ids",
            _normalise_mask_contributions(self.masked_contribution_ids),
        )
        # Ensure the object is structurally complete at the public boundary;
        # engine also repeats the check immediately before matching.
        exact_revisit_slot_need_bindings(self.requirements, self.query_vector_bundle)

    @property
    def anchor_activation_map(self) -> dict[int, float]:
        return {int(episode_id): float(score) for episode_id, score in self.anchor_activations}

    @property
    def base_slot_support_map(self) -> dict[int, set[str]]:
        return {
            int(episode_id): set(slot_ids)
            for episode_id, slot_ids in self.base_slot_support
        }

    @property
    def anchor_manifest_fingerprint(self) -> str:
        return exact_revisit_anchor_manifest_fingerprint(self.anchor_activations)

    @property
    def requirements_fingerprint(self) -> str:
        return exact_revisit_requirements_fingerprint(self.requirements)

    @property
    def budget_fingerprint(self) -> str:
        return exact_revisit_budget_fingerprint(self.budget)

    @property
    def slot_need_bindings(self) -> tuple[ContextualRevisitSlotNeedBinding, ...]:
        return exact_revisit_slot_need_bindings(
            self.requirements, self.query_vector_bundle
        )

    def contract_lookup(self) -> ContextualRevisitContractLookup:
        """Build the only typed, exact repository lookup accepted by v16."""

        return ContextualRevisitContractLookup(
            domain=self.domain,
            model_id=str(self.query_vector_bundle.model_id),
            embedding_space_id=str(self.query_vector_bundle.embedding_space_id),
            dimension=int(self.query_vector_bundle.dimension),
            dtype="float32",
            context_hash=self.context_hash,
            slot_need_bindings=self.slot_need_bindings,
            requirements_fingerprint=self.requirements_fingerprint,
            source_closure_fingerprint=self.source_closure_fingerprint,
            retrieval_policy_fingerprint=self.retrieval_policy_fingerprint,
            budget_fingerprint=self.budget_fingerprint,
            anchor_manifest_fingerprint=self.anchor_manifest_fingerprint,
        )
