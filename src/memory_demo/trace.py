"""Strict, content-addressed v3 recall-trace persistence.

The existing :class:`JsonlEventLogger` remains the operational audit log.  This
module writes a separate, schema-validated trace that is suitable for a
reproducible benchmark run.  It intentionally does not infer stages that the
current retrieval engine has not observed yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from importlib import resources
import io
import json
import os
from pathlib import Path
import re
from threading import Lock, RLock
from time import monotonic
from typing import Any
from uuid import uuid4

import numpy as np

from memory_demo.event_log import redact_for_export
from memory_demo.llm.client import (
    ProviderCallObservation,
    ProviderTraceSink,
    TracePersistenceError,
)

try:  # The dependency is declared in pyproject.toml, but keep app import lazy.
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:  # pragma: no cover - exercised by packaging, not unit tests
    Draft202012Validator = None
    SchemaError = Exception


TRACE_VERSION = "aevnema.recall-trace.v3"
TRACE_CONTRACT_RESOURCE = "contracts/aevnema_v3_trace_contract.schema.json"
TRACE_CONTRACT_UPSTREAM_SHA256 = (
    "b666b72935aded65d951e9fed3ba20bff10a5b19d2210fb9c59029012d9efc86"
)
TRACE_EVENT_TYPES = frozenset(
    {
        "request_received",
        "requirements_resolved",
        "vector_bundle_ready",
        "revisit_probe",
        "base_retrieval",
        "anchors_selected",
        "edge_scored",
        "target_checked",
        "candidate_merged",
        "selector_step",
        "recovery_step",
        "mask_evaluated",
        "evidence_delivered",
        "provider_call",
        "response_checked",
        "learning_committed",
        "request_completed",
    }
)

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_HASH_REFERENCE_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
# Trace identifiers are receipts, not user content.  Keep them compact and
# opaque so a caller cannot accidentally turn an identifier or stage label
# into another channel for a prompt, URL, or credential.
_TRACE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_TRACE_STAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,95}\Z")
# The provider labels in the approved integration are finite, machine labels:
# e.g. ``model_client:embedding``, ``contextual:query_vector_bundle`` and
# ``association_growth_primary_audit``.  Permit those namespaces without
# accepting free-form text in a durable trace receipt.
_TRACE_PURPOSE_PATTERN = re.compile(
    r"[a-z][a-z0-9_]{0,63}(?::[a-z0-9][a-z0-9_-]{0,63}){0,3}\Z"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[^\s,;]+")
_SECRET_KEY_NAMES = {
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "token",
    "secret",
    "password",
    "passwd",
    "cookie",
    "setcookie",
    "credential",
    "credentials",
    "privatekey",
}
_RAW_CONTENT_KEY_NAMES = {
    "content",
    "message",
    "messages",
    "input",
    "prompt",
    "system",
    "user",
    "text",
    "sourcetext",
    "sourceexcerpt",
    "excerpt",
    "documents",
    "document",
    "answer",
    "response",
}
_URL_KEY_NAMES = {"url", "uri", "baseurl", "endpointurl", "targeturl"}
_VISIBILITIES = frozenset({"full_local", "audit", "shareable"})


class TraceContractError(ValueError):
    """A record violates the vendored v3 trace contract."""


class TraceStateError(RuntimeError):
    """A trace session was used in an impossible lifecycle order."""


class TracePrivacyError(TraceContractError):
    """A trace record attempted to put protected raw content in JSONL."""


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _is_secret_key(value: Any) -> bool:
    normalized = _normalized_key(value)
    return (
        normalized in _SECRET_KEY_NAMES
        or normalized.endswith("apikey")
        or normalized.endswith("accesstoken")
        or normalized.endswith("refreshtoken")
        or normalized.endswith("idtoken")
        or normalized.endswith("authorization")
        or normalized.endswith("credential")
        or normalized.endswith("credentials")
        or normalized.endswith("privatekey")
        or normalized.endswith("password")
        or normalized.endswith("cookie")
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TraceContractError(
            "trace JSON must contain only finite, serializable values"
        ) from exc


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace one small receipt and force its file data to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact_ref_dict(value: "ArtifactRef | Mapping[str, Any]") -> dict[str, Any]:
    if isinstance(value, ArtifactRef):
        return value.as_dict()
    if not isinstance(value, Mapping):
        raise TraceContractError("artifact_refs must contain ArtifactRef or mapping values")
    return {str(key): item for key, item in value.items()}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object only when every key occurs once.

    Python's default decoder silently takes the last value for a duplicate
    key.  That makes a signed-looking persisted receipt ambiguous to tools
    that choose the first value, so every persisted JSON object must reject
    duplicates before schema validation begins.
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Do not echo the key: a tampered key can itself contain sensitive
            # text and exception logs are not part of the trace contract.
            raise TraceContractError("persisted JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(_value: str) -> None:
    raise TraceContractError("persisted JSON must not contain non-finite numbers")


def _load_strict_persisted_json(raw: bytes | str, *, source: str) -> Any:
    """Decode a trace receipt with unambiguous, standards-compliant JSON."""

    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except TraceContractError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceContractError(f"invalid JSON in persisted {source}") from exc


def _read_strict_persisted_json(path: Path, *, source: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TraceStateError(f"cannot read persisted {source}") from exc
    return _load_strict_persisted_json(raw, source=source)


def _assert_trace_identifier(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not _TRACE_IDENTIFIER_PATTERN.fullmatch(value):
        raise TracePrivacyError(
            f"{field} must be a short opaque trace identifier, not free-form content"
        )


def _assert_trace_stage(value: Any) -> None:
    if not isinstance(value, str) or not _TRACE_STAGE_PATTERN.fullmatch(value):
        raise TracePrivacyError("trace stage must be a short opaque machine label")


def _assert_trace_purpose(value: Any) -> None:
    if not isinstance(value, str) or not _TRACE_PURPOSE_PATTERN.fullmatch(value):
        raise TracePrivacyError("provider purpose must be an approved machine label")


def _artifact_ref_fingerprint(
    reference: Mapping[str, Any],
) -> tuple[str, str, str, str, str, bool]:
    """Validate one content-addressed reference and return its stable identity."""

    artifact_id = reference.get("artifact_id")
    path_value = reference.get("path")
    digest = reference.get("sha256")
    media_type = reference.get("media_type")
    visibility = reference.get("visibility")
    redacted = reference.get("redacted")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise TraceContractError("artifact reference sha256 must be a lowercase SHA-256")
    if artifact_id != f"sha256:{digest}":
        raise TraceContractError("artifact_id must exactly match its sha256")
    if path_value != f"artifacts/{digest}":
        raise TraceContractError("artifact path must exactly match its sha256")
    if not isinstance(media_type, str) or not media_type:
        raise TraceContractError("artifact media_type must not be empty")
    if visibility not in _VISIBILITIES:
        raise TraceContractError("artifact reference has an unknown visibility")
    if not isinstance(redacted, bool):
        raise TraceContractError("artifact redacted must be a boolean")
    if visibility != "full_local" and not redacted:
        raise TracePrivacyError(
            "audit/shareable artifact references must be explicitly redacted"
        )
    return (
        str(artifact_id),
        str(path_value),
        digest,
        media_type,
        str(visibility),
        redacted,
    )


def _validate_artifact_references(
    references: Sequence[Mapping[str, Any]],
    *,
    catalog: dict[str, tuple[str, str, str, str, str, bool]] | None = None,
) -> None:
    """Require canonical references and a non-ambiguous artifact catalogue.

    Exact references may appear in separate events (a later stage can point to
    the same query artifact), but a single record cannot repeat a reference
    and one content hash cannot change its declared metadata over the run.
    """

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for reference in references:
        fingerprint = _artifact_ref_fingerprint(reference)
        artifact_id, path_value = fingerprint[:2]
        if artifact_id in seen_ids or path_value in seen_paths:
            raise TraceContractError("record contains a duplicate artifact reference")
        seen_ids.add(artifact_id)
        seen_paths.add(path_value)
        if catalog is not None:
            existing = catalog.get(artifact_id)
            if existing is not None and existing != fingerprint:
                raise TraceContractError(
                    "artifact reference metadata conflicts with an earlier record"
                )
            catalog[artifact_id] = fingerprint


def _artifact_catalog_after(
    catalog: Mapping[str, tuple[str, str, str, str, str, bool]],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str, str, str, str, bool]]:
    updated = dict(catalog)
    _validate_artifact_references(references, catalog=updated)
    return updated


def _is_artifact_pointer_key(value: Any) -> bool:
    """Return whether a payload key carries one or more artifact identifiers.

    The contract currently uses both singular (``query_artifact_id``) and
    plural (``comparison_artifact_ids``) spellings.  Treat both as pointers so
    a future nested receipt field cannot become an unchecked reference merely
    because it is introduced below an existing payload object.
    """

    key = str(value).casefold()
    return (
        key == "artifact_id"
        or key.endswith("_artifact_id")
        or key == "artifact_ids"
        or key.endswith("_artifact_ids")
    )


def _iter_payload_artifact_pointers(value: Any) -> Sequence[str]:
    """Collect artifact IDs recursively from a payload or case summary.

    Only string values attached to an artifact-pointer key are yielded.  The
    JSON Schema remains responsible for the exact type of each known field;
    this traversal supplies the cross-record closure check the schema cannot
    express.
    """

    pointers: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                if _is_artifact_pointer_key(raw_key):
                    if isinstance(nested, str):
                        pointers.append(nested)
                    elif isinstance(nested, (list, tuple)):
                        pointers.extend(
                            value for value in nested if isinstance(value, str)
                        )
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(pointers)


def _validate_artifact_pointer_closure(
    value: Any,
    *,
    allowed_artifact_ids: set[str],
    source: str,
) -> None:
    """Require every artifact pointer to resolve to a verified reference.

    Event payloads must carry their own declared reference so each JSONL line
    is locally auditable.  A case summary has no ``artifact_refs`` field in
    the upstream contract and is therefore allowed to point at the already
    verified, run-wide catalogue instead.
    """

    for artifact_id in _iter_payload_artifact_pointers(value):
        if artifact_id not in allowed_artifact_ids:
            raise TraceContractError(
                f"artifact pointer in {source} is not declared by a verified artifact reference"
            )


def _assert_event_lifecycle_shape(record: Mapping[str, Any]) -> None:
    """Validate lifecycle facts contained in one trace event alone."""

    sequence = record["sequence"]
    event_type = record["event_type"]
    parents = record["parent_event_ids"]
    if len(parents) != len(set(parents)):
        raise TraceContractError("trace event parent_event_ids must be unique")
    if sequence == 0:
        if event_type != "request_received":
            raise TraceContractError("the first request event must be request_received")
        if parents:
            raise TraceContractError("request_received must not have parent events")
    else:
        if event_type == "request_received":
            raise TraceContractError("request_received may be emitted only once")
        if not parents:
            raise TraceContractError("non-initial trace events require explicit parents")


def _assert_trace_safe(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """Reject accidental raw credentials, prompts, source text, and URLs.

    The JSON schema describes structure, not confidentiality.  The one
    `requirement.question` field that the contract retains must be an opaque
    `sha256:<digest>` reference; its full text belongs in a `full_local`
    artifact instead.
    """

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = _normalized_key(key)
            current_path = (*path, key)
            if _is_secret_key(key):
                raise TracePrivacyError(
                    f"protected credential field is forbidden in trace JSONL: {'.'.join(current_path)}"
                )
            if normalized in _URL_KEY_NAMES or normalized in {"headers", "header"}:
                raise TracePrivacyError(
                    f"URL/header field is forbidden in trace JSONL: {'.'.join(current_path)}"
                )
            if normalized == "question":
                if not isinstance(item, str) or not _HASH_REFERENCE_PATTERN.fullmatch(item):
                    raise TracePrivacyError(
                        "requirement.question must be an opaque sha256 reference; "
                        "store raw question text only as a full_local artifact"
                    )
                continue
            if normalized == "purpose":
                _assert_trace_purpose(item)
                continue
            if normalized in _RAW_CONTENT_KEY_NAMES:
                raise TracePrivacyError(
                    f"raw content field is forbidden in trace JSONL: {'.'.join(current_path)}"
                )
            _assert_trace_safe(item, path=current_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_trace_safe(item, path=(*path, str(index)))
        return
    if isinstance(value, str) and _BEARER_PATTERN.search(value):
        raise TracePrivacyError(
            f"bearer credential is forbidden in trace JSONL: {'.'.join(path)}"
        )


def _load_contract_bytes(contract_path: str | Path | None = None) -> bytes:
    if contract_path is not None:
        return Path(contract_path).read_bytes()
    resource = resources.files("memory_demo").joinpath(TRACE_CONTRACT_RESOURCE)
    return resource.read_bytes()


def load_trace_contract(contract_path: str | Path | None = None) -> dict[str, Any]:
    """Load the exact vendored Draft 2020-12 contract resource."""

    raw = _load_contract_bytes(contract_path)
    digest = sha256(raw).hexdigest()
    if contract_path is None and digest != TRACE_CONTRACT_UPSTREAM_SHA256:
        raise TraceContractError(
            "vendored trace contract hash differs from the approved plan artifact"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - immutable resource
        raise TraceContractError("vendored trace contract is not JSON") from exc
    if value.get("$id") != "urn:aevnema:recall-trace:v3:2026-09-05":
        raise TraceContractError("unexpected v3 trace contract identity")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A content-addressed artifact reference accepted by the v3 contract."""

    artifact_id: str
    path: str
    sha256: str
    media_type: str
    visibility: str
    redacted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "visibility": self.visibility,
            "redacted": self.redacted,
        }


class ArtifactStore:
    """Content-addressed, atomically persisted artifacts for one trace run."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "artifacts"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise TraceStateError("trace artifacts directory must not be a symbolic link")
        self._lock = Lock()

    def put_bytes(
        self,
        value: bytes,
        *,
        media_type: str,
        visibility: str,
        redacted: bool,
    ) -> ArtifactRef:
        if visibility not in _VISIBILITIES:
            raise TraceContractError(f"unknown artifact visibility: {visibility}")
        if not media_type:
            raise TraceContractError("artifact media_type must not be empty")
        if visibility != "full_local" and not redacted:
            raise TracePrivacyError(
                "audit/shareable artifacts must be explicitly marked redacted"
            )
        data = bytes(value)
        digest = sha256(data).hexdigest()
        path = self.directory / digest
        with self._lock:
            if path.is_symlink():
                raise TraceStateError("trace artifacts must not be symbolic links")
            if path.exists():
                if not path.is_file():
                    raise TraceStateError("trace artifact path is not a regular file")
                if sha256(path.read_bytes()).hexdigest() != digest:
                    raise TraceStateError(f"artifact hash collision at {path}")
            else:
                _atomic_write_bytes(path, data)
        return ArtifactRef(
            artifact_id=f"sha256:{digest}",
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=digest,
            media_type=str(media_type),
            visibility=str(visibility),
            redacted=bool(redacted),
        )

    def put_json(
        self,
        value: Any,
        *,
        visibility: str = "full_local",
        redacted: bool | None = None,
    ) -> ArtifactRef:
        if visibility not in _VISIBILITIES:
            raise TraceContractError(f"unknown artifact visibility: {visibility}")
        if visibility != "full_local" and redacted is False:
            raise TracePrivacyError(
                "audit/shareable JSON is always redacted; use full_local for raw content"
            )
        should_redact = visibility != "full_local" or bool(redacted)
        stored = redact_for_export(value) if should_redact else value
        return self.put_bytes(
            _canonical_json_bytes(stored),
            media_type="application/json",
            visibility=visibility,
            redacted=should_redact,
        )

    def put_text(self, value: str, *, visibility: str = "full_local") -> ArtifactRef:
        if visibility != "full_local":
            raise TracePrivacyError(
                "raw text artifacts are full_local only; create a redacted JSON derivation for sharing"
            )
        return self.put_bytes(
            str(value).encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            visibility=visibility,
            redacted=False,
        )

    def put_numpy(self, value: np.ndarray, *, visibility: str = "full_local") -> ArtifactRef:
        if visibility != "full_local":
            raise TracePrivacyError("vectors are full_local artifacts only")
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise TraceContractError("trace vector artifacts must have dtype float32")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return self.put_bytes(
            buffer.getvalue(),
            media_type="application/x-npy",
            visibility=visibility,
            redacted=False,
        )


ArtifactFingerprint = tuple[str, str, str, str, str, bool]


@dataclass(frozen=True, slots=True)
class PersistedTraceRun:
    """One fully verified, read-only v3 trace receipt.

    The mappings are parsed from disk by :func:`validate_persisted_trace_run`.
    Callers should treat them as evidence, rather than mutable working state.
    ``artifact_catalog`` contains only canonical references whose on-disk
    content hash and no-symlink path have already been verified.
    """

    run_dir: Path
    manifest: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    artifact_catalog: Mapping[str, ArtifactFingerprint]


def _new_trace_contract_validator(
    contract_path: str | Path | None = None,
) -> Any:
    """Build the one schema validator used for writes and persisted receipts."""

    if Draft202012Validator is None:
        raise TraceContractError(
            "jsonschema is required to validate Aevnema v3 recall traces"
        )
    contract = load_trace_contract(contract_path)
    try:
        Draft202012Validator.check_schema(contract)
    except SchemaError as exc:
        raise TraceContractError("vendored trace contract is invalid") from exc
    return Draft202012Validator(contract)


def _validate_trace_record(
    record: Mapping[str, Any],
    *,
    validator: Any,
    expected_record_type: str | None = None,
) -> None:
    """Validate one record independently of whether it is live or persisted."""

    if not isinstance(record, Mapping):
        raise TraceContractError("v3 trace records must be JSON objects")
    if (
        expected_record_type is not None
        and record.get("record_type") != expected_record_type
    ):
        raise TraceContractError(
            f"expected {expected_record_type}, got {record.get('record_type')!r}"
        )
    errors = sorted(
        validator.iter_errors(dict(record)),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise TraceContractError(
            f"v3 trace contract violation at {location}: {first.message}"
        )

    record_type = record["record_type"]
    references = record.get("artifact_refs", [])
    if not isinstance(references, list) or any(
        not isinstance(reference, Mapping) for reference in references
    ):
        raise TraceContractError("artifact_refs must be an array of objects")
    _validate_artifact_references(references)
    _assert_trace_safe(record)

    if record_type == "run_manifest":
        run_id = record["run_id"]
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise TraceContractError("run_id must be a safe single path component")
        _assert_trace_identifier(record["arm"], field="arm")
        for module_name in record["actual_modules"]:
            _assert_trace_identifier(module_name, field="actual_module")
        return

    if record_type == "case_summary":
        _assert_trace_identifier(record["run_id"], field="run_id")
        for field in ("family_id", "q1_request_id"):
            _assert_trace_identifier(record[field], field=field)
        for field in ("q2_request_ids", "creation_receipt_ids"):
            for value in record[field]:
                _assert_trace_identifier(value, field=field)
        return

    if record_type != "trace_event":
        # The contract has only the three record kinds above.  Retain a
        # defensive branch so this helper remains safe if a custom contract
        # is supplied by a test or future migration.
        return

    run_id = record["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise TraceContractError("run_id must be a safe single path component")
    for field in ("request_id", "event_id", "scope_id"):
        _assert_trace_identifier(record[field], field=field)
    _assert_trace_stage(record["stage"])
    for parent_event_id in record["parent_event_ids"]:
        _assert_trace_identifier(parent_event_id, field="parent_event_id")
    _assert_event_lifecycle_shape(record)
    _validate_artifact_pointer_closure(
        record["payload"],
        allowed_artifact_ids={str(reference["artifact_id"]) for reference in references},
        source="trace event payload",
    )
    if record["event_type"] == "provider_call":
        provider = record["payload"]
        _assert_trace_identifier(provider["call_id"], field="provider call_id")
        logical_batch_id = provider["logical_batch_id"]
        if logical_batch_id is not None:
            _assert_trace_identifier(
                logical_batch_id, field="provider logical_batch_id"
            )
        _assert_trace_purpose(provider["purpose"])


def _resolve_persisted_run_dir(run_dir: str | Path) -> Path:
    supplied = Path(run_dir)
    if supplied.is_symlink():
        raise TraceStateError("trace run directory must not be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise TraceStateError("trace run directory does not exist") from exc
    if not resolved.is_dir():
        raise TraceStateError("trace run path is not a directory")
    return resolved


def _read_regular_persisted_json(path: Path, *, source: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise TraceStateError(f"persisted {source} must not be a symbolic link")
    try:
        if not path.is_file():
            raise TraceStateError(f"persisted {source} is missing or not a regular file")
    except OSError as exc:
        raise TraceStateError(f"cannot inspect persisted {source}") from exc
    value = _read_strict_persisted_json(path, source=source)
    if not isinstance(value, Mapping):
        raise TraceContractError(f"persisted {source} must be a JSON object")
    return value


def _read_persisted_events(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read JSONL through the same strict decoder used by every receipt."""

    if path.is_symlink():
        raise TraceStateError("persisted events JSONL must not be a symbolic link")
    try:
        if not path.exists():
            return ()
        if not path.is_file():
            raise TraceStateError("persisted events JSONL is not a regular file")
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TraceStateError("cannot read persisted events JSONL") from exc

    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise TraceContractError(
                f"blank JSONL record at events line {line_number}"
            )
        record = _load_strict_persisted_json(
            raw_line, source=f"events line {line_number}"
        )
        if not isinstance(record, Mapping):
            raise TraceContractError(
                f"persisted events line {line_number} must be a JSON object"
            )
        records.append(record)
    return tuple(records)


def _validate_persisted_lifecycle(
    manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    """Validate causal parentage, request isolation, scope binding, and closure."""

    seen_global: set[str] = set()
    seen_per_request: dict[str, set[str]] = {}
    expected_sequence: dict[str, int] = {}
    request_scopes: dict[str, str] = {}
    completed: set[str] = set()
    for record in events:
        if record["run_id"] != manifest["run_id"]:
            raise TraceStateError("event run_id differs from manifest")
        request_id = str(record["request_id"])
        event_id = str(record["event_id"])
        if event_id in seen_global:
            raise TraceStateError("duplicate persisted event_id")
        expected = expected_sequence.get(request_id, 0)
        if record["sequence"] != expected:
            raise TraceStateError("request event sequence is not contiguous")
        if request_id in completed:
            raise TraceStateError("persisted request has an event after request_completed")
        scope_id = str(record["scope_id"])
        if expected == 0:
            # `_assert_event_lifecycle_shape` has already established a root;
            # retain the binding so interleaved events cannot change scope.
            request_scopes[request_id] = scope_id
        elif request_scopes.get(request_id) != scope_id:
            raise TraceStateError(
                "persisted request changed scope_id during its lifecycle"
            )
        parents = set(record["parent_event_ids"])
        if not parents.issubset(seen_per_request.get(request_id, set())):
            raise TraceStateError("persisted event has a foreign or future parent")
        if record["event_type"] == "request_completed":
            if request_id in completed:
                raise TraceStateError("persisted request has multiple request_completed events")
            completed.add(request_id)
        seen_global.add(event_id)
        seen_per_request.setdefault(request_id, set()).add(event_id)
        expected_sequence[request_id] = expected + 1
    if manifest["status"] == "completed" and set(expected_sequence) != completed:
        raise TraceStateError(
            "completed run contains a request without request_completed"
        )


def _verify_persisted_artifact_catalog(
    run_dir: Path, catalog: Mapping[str, ArtifactFingerprint]
) -> None:
    """Hash every canonical artifact once and reject all symlink indirection."""

    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.is_symlink():
        raise TraceStateError("trace artifacts directory must not be a symbolic link")
    try:
        if not artifacts_dir.is_dir():
            raise TraceStateError("trace run is missing its artifacts directory")
        resolved_artifacts_dir = artifacts_dir.resolve(strict=True)
    except OSError as exc:
        raise TraceStateError("cannot inspect trace artifacts directory") from exc

    for _artifact_id, fingerprint in catalog.items():
        _artifact_id, _path_value, digest, _media_type, _visibility, _redacted = fingerprint
        artifact_path = artifacts_dir / digest
        if artifact_path.is_symlink():
            raise TraceStateError("trace artifacts must not be symbolic links")
        try:
            if not artifact_path.is_file():
                raise TraceStateError("trace artifact is missing")
            resolved_artifact = artifact_path.resolve(strict=True)
        except OSError as exc:
            raise TraceStateError("cannot inspect trace artifact") from exc
        if resolved_artifact.parent != resolved_artifacts_dir:
            raise TraceStateError("trace artifact resolves outside the artifacts directory")
        try:
            digest_on_disk = sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise TraceStateError("cannot read trace artifact") from exc
        if digest_on_disk != digest:
            raise TraceStateError("trace artifact hash mismatch")


def validate_persisted_trace_run(
    run_dir: str | Path,
    *,
    contract_path: str | Path | None = None,
) -> PersistedTraceRun:
    """Strictly verify an on-disk run before any replay or rendering reads it.

    This is deliberately the shared persisted-validation boundary: the trace
    writer uses it for post-write verification and the Markdown renderer uses
    it before creating output.  No caller may substitute a looser JSON,
    privacy, schema, artifact, or request-causality interpretation.
    """

    resolved_run_dir = _resolve_persisted_run_dir(run_dir)
    validator = _new_trace_contract_validator(contract_path)
    manifest = _read_regular_persisted_json(
        resolved_run_dir / "run_manifest.json", source="run manifest"
    )
    _validate_trace_record(
        manifest, validator=validator, expected_record_type="run_manifest"
    )
    events = _read_persisted_events(resolved_run_dir / "events.jsonl")
    for record in events:
        _validate_trace_record(
            record, validator=validator, expected_record_type="trace_event"
        )
    _validate_persisted_lifecycle(manifest, events)

    artifact_catalog: dict[str, ArtifactFingerprint] = {}
    for record in (manifest, *events):
        references = record.get("artifact_refs", [])
        _validate_artifact_references(references, catalog=artifact_catalog)
    _verify_persisted_artifact_catalog(resolved_run_dir, artifact_catalog)
    return PersistedTraceRun(
        run_dir=resolved_run_dir,
        manifest=manifest,
        events=events,
        artifact_catalog=dict(artifact_catalog),
    )


def validate_case_summary(
    case_summary: Mapping[str, Any],
    *,
    trace_run: PersistedTraceRun,
    contract_path: str | Path | None = None,
) -> None:
    """Validate a case summary against the already verified run catalogue.

    Unlike JSONL events, the upstream case-summary schema has no local
    ``artifact_refs`` member.  Its artifact pointers are therefore closed over
    ``trace_run.artifact_catalog``, which only exists after
    :func:`validate_persisted_trace_run` has checked hashes and paths.
    """

    validator = _new_trace_contract_validator(contract_path)
    _validate_trace_record(
        case_summary, validator=validator, expected_record_type="case_summary"
    )
    if case_summary["run_id"] != trace_run.manifest["run_id"]:
        raise TraceStateError("case summary run_id differs from verified trace run")
    _validate_artifact_pointer_closure(
        case_summary,
        allowed_artifact_ids=set(trace_run.artifact_catalog),
        source="case summary",
    )


class RecallTraceWriter:
    """Write one isolated v3 trace run without changing legacy audit logs.

    A new run begins conservatively as ``incomplete``.  It becomes
    ``completed`` only through :meth:`finalize`, after every registered request
    has emitted its terminal ``request_completed`` event.  Consequently an
    abrupt process exit can never leave a false completed receipt.
    """

    def __init__(
        self,
        output_root: str | Path,
        manifest: Mapping[str, Any],
        *,
        contract_path: str | Path | None = None,
    ):
        self._contract_path = contract_path
        self._contract_bytes = _load_contract_bytes(contract_path)
        self.contract_sha256 = sha256(self._contract_bytes).hexdigest()
        if contract_path is None and self.contract_sha256 != TRACE_CONTRACT_UPSTREAM_SHA256:
            raise TraceContractError(
                "vendored trace contract hash differs from the approved plan artifact"
            )
        self.contract = load_trace_contract(contract_path)
        self._validator = _new_trace_contract_validator(contract_path)
        self._lock = RLock()
        self._closed = False
        self._requests: dict[str, TraceRequestSession] = {}
        self._event_ids: set[str] = set()
        self._artifact_catalog: dict[
            str, tuple[str, str, str, str, str, bool]
        ] = {}

        initial_manifest = self._normalize_manifest(manifest)
        self._validate_record(initial_manifest, expected_record_type="run_manifest")
        run_id = initial_manifest["run_id"]
        self.run_dir = Path(output_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.artifacts = ArtifactStore(self.run_dir)
        contract_ref = self.artifacts.put_bytes(
            self._contract_bytes,
            media_type="application/schema+json",
            visibility="audit",
            redacted=True,
        )
        initial_manifest["artifact_refs"] = [
            *initial_manifest["artifact_refs"],
            contract_ref.as_dict(),
        ]
        initial_manifest["status"] = "incomplete"
        self._validate_record(initial_manifest, expected_record_type="run_manifest")
        self._artifact_catalog = _artifact_catalog_after(
            self._artifact_catalog, initial_manifest["artifact_refs"]
        )
        self._manifest = initial_manifest
        self._write_manifest()

    def _normalize_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(manifest, Mapping):
            raise TraceContractError("run manifest must be a mapping")
        result = {str(key): value for key, value in manifest.items()}
        result["record_type"] = result.get("record_type", "run_manifest")
        result["trace_version"] = result.get("trace_version", TRACE_VERSION)
        result["artifact_refs"] = [
            _artifact_ref_dict(value) for value in result.get("artifact_refs", [])
        ]
        return result

    def _validate_record(
        self, record: Mapping[str, Any], *, expected_record_type: str | None = None
    ) -> None:
        _validate_trace_record(
            record,
            validator=self._validator,
            expected_record_type=expected_record_type,
        )

    def _write_manifest(self) -> None:
        if self.manifest_path.is_symlink():
            raise TraceStateError("trace manifest path must not be a symbolic link")
        _atomic_write_bytes(self.manifest_path, _canonical_json_bytes(self._manifest))

    def _append_event(self, record: dict[str, Any]) -> None:
        if self.events_path.is_symlink():
            raise TraceStateError("trace events path must not be a symbolic link")
        line = _canonical_json_bytes(record) + b"\n"
        with self.events_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def start_request(
        self, *, scope_id: str, request_id: str | None = None
    ) -> "TraceRequestSession":
        selected_id = (
            request_id if request_id is not None else f"request-{uuid4().hex}"
        )
        _assert_trace_identifier(selected_id, field="request_id")
        _assert_trace_identifier(scope_id, field="scope_id")
        with self._lock:
            if self._closed:
                raise TraceStateError("cannot start a request after the run is finalized")
            if selected_id in self._requests:
                raise TraceStateError(f"duplicate trace request_id: {selected_id}")
            session = TraceRequestSession(self, selected_id, scope_id)
            self._requests[selected_id] = session
            return session

    def _emit_from_session(
        self,
        session: "TraceRequestSession",
        *,
        sequence: int,
        event_type: str,
        stage: str,
        payload: Mapping[str, Any],
        artifact_refs: Sequence[ArtifactRef | Mapping[str, Any]],
        parent_event_ids: Sequence[str],
        elapsed_ms: float,
    ) -> str:
        if not isinstance(event_type, str):
            raise TraceContractError("trace event_type must be a string")
        if event_type not in TRACE_EVENT_TYPES:
            raise TraceContractError(f"unknown v3 trace event type: {event_type}")
        _assert_trace_stage(stage)
        references = [_artifact_ref_dict(value) for value in artifact_refs]
        if any(not isinstance(value, str) for value in parent_event_ids):
            raise TraceContractError("trace event parent_event_ids must be strings")
        parents = list(parent_event_ids)
        with self._lock:
            if self._closed:
                raise TraceStateError("cannot emit an event after the run is finalized")
            if any(parent not in session._event_ids for parent in parents):
                raise TraceStateError("trace event parent must belong to the same request")
            event_id = f"event-{uuid4().hex}"
            record = {
                "record_type": "trace_event",
                "trace_version": TRACE_VERSION,
                "run_id": self._manifest["run_id"],
                "request_id": session.request_id,
                "event_id": event_id,
                "sequence": int(sequence),
                "parent_event_ids": parents,
                "event_type": event_type,
                "elapsed_ms": round(max(0.0, float(elapsed_ms)), 6),
                "stage": str(stage),
                "scope_id": session.scope_id,
                "artifact_refs": references,
                "payload": {str(key): value for key, value in payload.items()},
            }
            self._validate_record(record, expected_record_type="trace_event")
            if event_id in self._event_ids:
                raise TraceStateError("duplicate generated trace event_id")
            next_artifact_catalog = _artifact_catalog_after(
                self._artifact_catalog, references
            )
            self._append_event(record)
            self._event_ids.add(event_id)
            self._artifact_catalog = next_artifact_catalog
            return event_id

    def finalize(self, status: str = "completed") -> None:
        if status not in {"completed", "incomplete", "invalid_setup"}:
            raise TraceContractError(f"unsupported terminal run status: {status}")
        with self._lock:
            if self._closed:
                raise TraceStateError("trace run is already finalized")
            if status == "completed":
                incomplete = [
                    request_id
                    for request_id, session in self._requests.items()
                    if not session.completed
                ]
                if incomplete:
                    raise TraceStateError(
                        "cannot mark trace completed before request_completed: "
                        + ", ".join(sorted(incomplete))
                    )
            self._manifest["status"] = status
            self._validate_record(self._manifest, expected_record_type="run_manifest")
            self._write_manifest()
            self._closed = True

    def validate_persisted(self) -> None:
        """Use the shared strict persisted-validation boundary for this run."""

        persisted = validate_persisted_trace_run(
            self.run_dir, contract_path=self._contract_path
        )
        if persisted.manifest["run_id"] != self._manifest["run_id"]:
            raise TraceStateError("persisted manifest run_id differs from active writer")

    def __enter__(self) -> "RecallTraceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._closed:
            self.finalize("incomplete")
        return False


class TraceRequestSession:
    """Thread-safe request-local sequence and causal-parent coordinator."""

    def __init__(self, writer: RecallTraceWriter, request_id: str, scope_id: str):
        self.writer = writer
        self.request_id = request_id
        self.scope_id = scope_id
        self._started_at = monotonic()
        self._lock = Lock()
        self._sequence = 0
        self._event_ids: set[str] = set()
        self.completed = False

    def emit(
        self,
        event_type: str,
        *,
        stage: str,
        payload: Mapping[str, Any],
        artifact_refs: Sequence[ArtifactRef | Mapping[str, Any]] = (),
        parent_event_ids: Sequence[str] = (),
    ) -> str:
        if not isinstance(payload, Mapping):
            raise TraceContractError("trace event payload must be a mapping")
        with self._lock:
            if self.completed:
                raise TraceStateError("cannot emit after request_completed")
            if self._sequence == 0 and event_type != "request_received":
                raise TraceStateError("the first request event must be request_received")
            if self._sequence > 0 and event_type == "request_received":
                raise TraceStateError("request_received may be emitted only once")
            if any(not isinstance(value, str) for value in parent_event_ids):
                raise TraceContractError("trace event parent_event_ids must be strings")
            parents = tuple(parent_event_ids)
            if self._sequence == 0 and parents:
                raise TraceStateError("request_received must not have parent events")
            if self._sequence > 0 and not parents:
                raise TraceStateError("non-initial request events require an explicit parent")
            if len(parents) != len(set(parents)):
                raise TraceStateError("trace event parent_event_ids must be unique")
            event_id = self.writer._emit_from_session(
                self,
                sequence=self._sequence,
                event_type=event_type,
                stage=stage,
                payload=payload,
                artifact_refs=artifact_refs,
                parent_event_ids=parents,
                elapsed_ms=(monotonic() - self._started_at) * 1000.0,
            )
            self._event_ids.add(event_id)
            self._sequence += 1
            if event_type == "request_completed":
                self.completed = True
            return event_id


@dataclass(frozen=True, slots=True)
class ProviderObservationCheckpoint:
    """An immutable, request-local snapshot of actual provider observations.

    The checkpoint deliberately contains only durable receipt identifiers and
    aggregate counters.  It never carries a prompt, source excerpt, endpoint,
    request body, or response body.  A caller may use
    :attr:`causal_parent_event_ids` when it has *actually* observed provider
    work that a later stage depends on; it is not an instruction to invent a
    provider dependency for every stage.
    """

    request_id: str
    request_event_id: str
    provider_event_ids: tuple[str, ...]
    cloud_logical_calls: int
    cloud_http_attempts: int
    contextual_http_attempts: int
    embedding_logical_batches: int
    fallback_count: int
    fallback_kind: str | None

    @property
    def causal_parent_event_ids(self) -> tuple[str, ...]:
        """Return the root plus provider events observed at this checkpoint."""

        return (self.request_event_id, *self.provider_event_ids)

    def completion_summary(self) -> dict[str, Any]:
        """Return the schema-facing aggregate fields without mutable state."""

        return {
            "cloud_logical_calls": self.cloud_logical_calls,
            "cloud_http_attempts": self.cloud_http_attempts,
            "contextual_http_attempts": self.contextual_http_attempts,
            "embedding_logical_batches": self.embedding_logical_batches,
            "fallback_kind": self.fallback_kind,
            "provider_event_ids": self.provider_event_ids,
        }


@dataclass(frozen=True, slots=True)
class ProviderObservationDelta:
    """Actual provider observations added after one ledger checkpoint.

    The delta is useful for a stage that directly follows one model boundary:
    its :attr:`causal_parent_event_ids` contains the request root and only the
    provider-call receipts emitted since that boundary's checkpoint.  It is
    still the caller's responsibility to decide whether that causal link was
    actually used and to name those parents explicitly.
    """

    request_id: str
    request_event_id: str
    provider_event_ids: tuple[str, ...]
    cloud_logical_calls: int
    cloud_http_attempts: int
    contextual_http_attempts: int
    embedding_logical_batches: int
    fallback_kind: str | None

    @property
    def causal_parent_event_ids(self) -> tuple[str, ...]:
        """Return the request root plus only provider events in this delta."""

        return (self.request_event_id, *self.provider_event_ids)


class RequestProviderLedger(ProviderTraceSink):
    """Request-exclusive provider ledger used by the strict v3 trace.

    ``ProviderCallAccounting`` remains a useful process/import aggregate, but
    it cannot be snapshotted for a request receipt: concurrent requests would
    then be charged to whichever receipt happened to finish.  This ledger is
    bound through ``ModelClient.trace_request`` and records only observations
    delivered inside that request's context.
    """

    def __init__(self, session: TraceRequestSession, request_event_id: str):
        self.session = session
        self.request_event_id = str(request_event_id)
        if not self.request_event_id:
            raise TraceContractError("provider ledger needs a request_received event id")
        self._lock = Lock()
        self._logical_batches: dict[str, tuple[str, str]] = {}
        self._attempts: dict[str, int] = {}
        self._provider_event_ids: list[str] = []
        self._sent_attempts = 0
        self._contextual_sent_attempts = 0
        self._fallback_count = 0
        # A sink can fail after a provider has already answered.  The client
        # latches this through ``trace_persistence_failed`` so no later
        # best-effort branch can turn the damaged receipt into completion.
        self._faulted = False

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._faulted

    def trace_persistence_failed(self, exc: BaseException) -> None:
        """Latch a writer failure without retaining its potentially raw text."""

        del exc
        with self._lock:
            self._faulted = True

    def logical_batch_started(
        self,
        *,
        logical_batch_id: str,
        operation: str,
        requested_model: str,
    ) -> None:
        batch_id = str(logical_batch_id).strip()
        if not batch_id:
            raise TraceContractError("logical_batch_id must not be empty")
        values = (str(operation), str(requested_model))
        with self._lock:
            try:
                existing = self._logical_batches.get(batch_id)
                if existing is not None and existing != values:
                    raise TraceStateError(
                        "one logical provider batch cannot change operation or requested model"
                    )
                self._logical_batches.setdefault(batch_id, values)
            except Exception:
                self._faulted = True
                raise

    @staticmethod
    def _is_contextual_purpose(purpose: str) -> bool:
        """Count only an explicit contextual label, never a guessed route."""

        normalized = str(purpose).strip().casefold()
        return normalized == "contextual" or normalized.startswith("contextual:")

    def provider_call_finished(self, observation: ProviderCallObservation) -> None:
        """Persist one provider event and update this request's receipt only."""

        if not isinstance(observation, ProviderCallObservation):
            raise TypeError("provider trace sink requires ProviderCallObservation")
        batch_id = observation.logical_batch_id
        with self._lock:
            try:
                if batch_id is None:
                    attempt = 1
                else:
                    normalized_batch_id = str(batch_id).strip()
                    if not normalized_batch_id:
                        raise TraceContractError("logical_batch_id must not be blank")
                    self._logical_batches.setdefault(
                        normalized_batch_id,
                        (str(observation.operation), str(observation.requested_model)),
                    )
                    attempt = self._attempts.get(normalized_batch_id, 0) + 1
                    self._attempts[normalized_batch_id] = attempt
                payload = {
                    "call_id": str(observation.call_id),
                    "logical_batch_id": batch_id,
                    "attempt": attempt,
                    "role": str(observation.role),
                    "requested_model": str(observation.requested_model),
                    "actual_model": observation.actual_model,
                    "purpose": str(observation.purpose),
                    "status": str(observation.status),
                    "queue_ms": float(observation.queue_ms),
                    "network_ms": (
                        None
                        if observation.network_ms is None
                        else float(observation.network_ms)
                    ),
                    # The current requests client is non-streaming, so it cannot
                    # honestly observe time to first token.
                    "first_token_ms": None,
                    "total_ms": float(observation.total_ms),
                    "prompt_tokens": observation.prompt_tokens,
                    "completion_tokens": observation.completion_tokens,
                    "http_status": observation.http_status,
                    "error_class": observation.error_class,
                    "late_result_discarded": bool(observation.late_result_discarded),
                }
                event_id = self.session.emit(
                    "provider_call",
                    stage="provider",
                    payload=payload,
                    parent_event_ids=(self.request_event_id,),
                )
                self._provider_event_ids.append(event_id)
                if observation.sent:
                    self._sent_attempts += 1
                    if self._is_contextual_purpose(observation.purpose):
                        self._contextual_sent_attempts += 1
            except Exception:
                self._faulted = True
                raise

    def fallback_used(
        self,
        *,
        logical_batch_id: str,
        from_model: str,
        to_model: str,
    ) -> None:
        del from_model, to_model
        batch_id = str(logical_batch_id).strip()
        if not batch_id:
            raise TraceContractError("fallback needs a logical_batch_id")
        with self._lock:
            try:
                if batch_id not in self._logical_batches:
                    raise TraceStateError("fallback belongs to an unknown logical provider batch")
                self._fallback_count += 1
            except Exception:
                self._faulted = True
                raise

    def _completion_summary_locked(self) -> dict[str, Any]:
        """Build a receipt summary while the ledger lock is held."""

        embedding_batches = sum(
            1
            for operation, _model in self._logical_batches.values()
            if operation == "embedding"
        )
        return {
            "cloud_logical_calls": len(self._logical_batches),
            "cloud_http_attempts": self._sent_attempts,
            "contextual_http_attempts": self._contextual_sent_attempts,
            "embedding_logical_batches": embedding_batches,
            "fallback_kind": "model_fallback" if self._fallback_count else None,
            "provider_event_ids": tuple(self._provider_event_ids),
        }

    def checkpoint(self) -> ProviderObservationCheckpoint:
        """Freeze the provider observations that exist for this request now.

        The ledger lock makes the counters and event-id list one coherent
        snapshot even when another query is using the same ``ModelClient``.
        A faulted ledger cannot produce a checkpoint: later stages must not
        turn an unreceipted provider failure into an apparently valid trace.
        """

        with self._lock:
            if self._faulted:
                raise TracePersistenceError("strict provider trace is faulted")
            summary = self._completion_summary_locked()
            return ProviderObservationCheckpoint(
                request_id=self.session.request_id,
                request_event_id=self.request_event_id,
                provider_event_ids=tuple(summary["provider_event_ids"]),
                cloud_logical_calls=int(summary["cloud_logical_calls"]),
                cloud_http_attempts=int(summary["cloud_http_attempts"]),
                contextual_http_attempts=int(summary["contextual_http_attempts"]),
                embedding_logical_batches=int(
                    summary["embedding_logical_batches"]
                ),
                fallback_count=self._fallback_count,
                fallback_kind=summary["fallback_kind"],
            )

    def _validate_checkpoint_locked(
        self, checkpoint: ProviderObservationCheckpoint
    ) -> None:
        """Reject a foreign, impossible, or rewritten ledger checkpoint."""

        if not isinstance(checkpoint, ProviderObservationCheckpoint):
            raise TypeError("checkpoint must be ProviderObservationCheckpoint")
        if checkpoint.request_id != self.session.request_id:
            raise TraceStateError("provider checkpoint belongs to a different request")
        if checkpoint.request_event_id != self.request_event_id:
            raise TraceStateError("provider checkpoint belongs to a different ledger")
        previous_event_ids = checkpoint.provider_event_ids
        if len(previous_event_ids) > len(self._provider_event_ids) or tuple(
            self._provider_event_ids[: len(previous_event_ids)]
        ) != previous_event_ids:
            raise TraceStateError("provider checkpoint is not an earlier ledger state")
        current_summary = self._completion_summary_locked()
        if any(
            previous > current
            for previous, current in (
                (checkpoint.cloud_logical_calls, current_summary["cloud_logical_calls"]),
                (checkpoint.cloud_http_attempts, current_summary["cloud_http_attempts"]),
                (
                    checkpoint.contextual_http_attempts,
                    current_summary["contextual_http_attempts"],
                ),
                (
                    checkpoint.embedding_logical_batches,
                    current_summary["embedding_logical_batches"],
                ),
                (checkpoint.fallback_count, self._fallback_count),
            )
        ):
            raise TraceStateError("provider checkpoint has impossible counters")

    def observations_since(
        self, checkpoint: ProviderObservationCheckpoint
    ) -> ProviderObservationDelta:
        """Return only real provider observations emitted after ``checkpoint``.

        The result is request-local and immutable.  It is not an aggregate
        reconstructed from another request, and a trace persistence fault
        fails closed before a caller can use a partial receipt as evidence.
        """

        with self._lock:
            if self._faulted:
                raise TracePersistenceError("strict provider trace is faulted")
            self._validate_checkpoint_locked(checkpoint)
            summary = self._completion_summary_locked()
            new_event_ids = tuple(self._provider_event_ids[len(checkpoint.provider_event_ids) :])
            return ProviderObservationDelta(
                request_id=self.session.request_id,
                request_event_id=self.request_event_id,
                provider_event_ids=new_event_ids,
                cloud_logical_calls=(
                    int(summary["cloud_logical_calls"])
                    - checkpoint.cloud_logical_calls
                ),
                cloud_http_attempts=(
                    int(summary["cloud_http_attempts"])
                    - checkpoint.cloud_http_attempts
                ),
                contextual_http_attempts=(
                    int(summary["contextual_http_attempts"])
                    - checkpoint.contextual_http_attempts
                ),
                embedding_logical_batches=(
                    int(summary["embedding_logical_batches"])
                    - checkpoint.embedding_logical_batches
                ),
                fallback_kind=(
                    "model_fallback"
                    if self._fallback_count > checkpoint.fallback_count
                    else None
                ),
            )

    def completion_summary(self) -> dict[str, Any]:
        """Return only observations from this request, under one lock."""

        with self._lock:
            return self._completion_summary_locked()


@dataclass(frozen=True, slots=True)
class QueryTraceObservationCheckpoint:
    """Request-scoped observations available to one later trace-stage write.

    ``observed_parent_event_ids`` is an availability set, not an automatic
    parent list.  The integration must still name the causal parents it
    actually used when calling :meth:`QueryTraceBridge.emit_observed_stage`.
    That prevents a final result from being reverse-engineered into a false
    sequence of v3 stages.
    """

    provider: ProviderObservationCheckpoint
    observed_stage_event_ids: tuple[str, ...]

    @property
    def request_id(self) -> str:
        return self.provider.request_id

    @property
    def request_event_id(self) -> str:
        return self.provider.request_event_id

    @property
    def provider_event_ids(self) -> tuple[str, ...]:
        return self.provider.provider_event_ids

    @property
    def observed_parent_event_ids(self) -> tuple[str, ...]:
        """Return only event IDs known to have happened at checkpoint time."""

        return tuple(
            dict.fromkeys(
                (
                    self.request_event_id,
                    *self.observed_stage_event_ids,
                    *self.provider_event_ids,
                )
            )
        )


class QueryTraceBridge:
    """Explicit bridge between one query boundary and a v3 trace session.

    The caller supplies the fully evidenced ``request_received`` payload and
    its query artifact reference.  The bridge refuses to manufacture database
    hashes, permission hashes, or request scopes from incomplete engine state.
    It observes only provider work that actually reaches :meth:`ModelClient._post`.
    """

    def __init__(
        self,
        session: TraceRequestSession,
        *,
        request_payload: Mapping[str, Any],
        request_artifact_refs: Sequence[ArtifactRef | Mapping[str, Any]],
    ):
        self.session = session
        self._request_payload = {
            str(key): value for key, value in request_payload.items()
        }
        self._request_artifact_refs = tuple(request_artifact_refs)
        self._lock = Lock()
        self._started_at: float | None = None
        self._request_event_id: str | None = None
        self._ledger: RequestProviderLedger | None = None
        self._observed_stage_event_ids: list[str] = []
        self._completed = False

    @property
    def request_event_id(self) -> str:
        if self._request_event_id is None:
            raise TraceStateError("query trace bridge has not started")
        return self._request_event_id

    @property
    def ledger(self) -> RequestProviderLedger:
        if self._ledger is None:
            raise TraceStateError("query trace bridge has not started")
        return self._ledger

    def start(self) -> str:
        with self._lock:
            if self._request_event_id is not None:
                raise TraceStateError("query trace bridge was started twice")
            request_event_id = self.session.emit(
                "request_received",
                stage="request",
                payload=self._request_payload,
                artifact_refs=self._request_artifact_refs,
            )
            self._request_event_id = request_event_id
            self._ledger = RequestProviderLedger(self.session, request_event_id)
            self._started_at = monotonic()
            return request_event_id

    def checkpoint_observations(self) -> QueryTraceObservationCheckpoint:
        """Freeze request/provider/stage observations available at this point.

        This is a snapshot boundary, not a stage synthesizer.  An engine can
        write a full-local evidence artifact after it has real results, obtain
        this checkpoint, then pass its relevant explicit parents and artifact
        reference to :meth:`emit_observed_stage`.  Provider work that happens
        after this call is deliberately absent from the snapshot.
        """

        with self._lock:
            if self._request_event_id is None:
                raise TraceStateError("cannot checkpoint a trace bridge before start")
            if self._completed:
                raise TraceStateError("cannot checkpoint a completed trace bridge")
            provider = self.ledger.checkpoint()
            return QueryTraceObservationCheckpoint(
                provider=provider,
                observed_stage_event_ids=tuple(self._observed_stage_event_ids),
            )

    # Keep the short spelling available for stage integrations while making
    # the longer public name self-documenting at call sites.
    def checkpoint(self) -> QueryTraceObservationCheckpoint:
        return self.checkpoint_observations()

    def _validate_observation_checkpoint(
        self, checkpoint: QueryTraceObservationCheckpoint
    ) -> None:
        if not isinstance(checkpoint, QueryTraceObservationCheckpoint):
            raise TypeError("observation_checkpoint must be QueryTraceObservationCheckpoint")
        if checkpoint.request_id != self.session.request_id:
            raise TraceStateError(
                "observation checkpoint belongs to a different trace request"
            )
        if checkpoint.request_event_id != self.request_event_id:
            raise TraceStateError(
                "observation checkpoint belongs to a different trace bridge"
            )

    def observations_since(
        self, checkpoint: QueryTraceObservationCheckpoint
    ) -> ProviderObservationDelta:
        """Return provider receipts actually emitted after a bridge checkpoint.

        The returned delta carries only the new provider event IDs.  A caller
        can use its explicit causal parents for a directly following stage,
        or take a fresh bridge checkpoint when that stage also depends on an
        earlier observed stage event.
        """

        with self._lock:
            if self._request_event_id is None:
                raise TraceStateError("cannot compare observations before bridge start")
            if self._completed:
                raise TraceStateError("cannot compare observations after completion")
            self._validate_observation_checkpoint(checkpoint)
            return self.ledger.observations_since(checkpoint.provider)

    def _validate_observation_delta(self, delta: ProviderObservationDelta) -> None:
        if not isinstance(delta, ProviderObservationDelta):
            raise TypeError(
                "observation_checkpoint must be QueryTraceObservationCheckpoint "
                "or ProviderObservationDelta"
            )
        if delta.request_id != self.session.request_id:
            raise TraceStateError(
                "provider observation delta belongs to a different trace request"
            )
        if delta.request_event_id != self.request_event_id:
            raise TraceStateError(
                "provider observation delta belongs to a different trace bridge"
            )

    def emit_observed_stage(
        self,
        event_type: str,
        *,
        stage: str,
        payload: Mapping[str, Any],
        artifact_refs: Sequence[ArtifactRef | Mapping[str, Any]] = (),
        parent_event_ids: Sequence[str],
        observation_checkpoint: (
            QueryTraceObservationCheckpoint | ProviderObservationDelta | None
        ) = None,
    ) -> str:
        """Persist one stage only when the caller has observed every field.

        The bridge intentionally has no defaults for payloads or parents.  A
        query integration must provide schema-complete data at the point it is
        observed; it may not reconstruct a fictitious path from a final query
        result.  ``request_received`` and ``request_completed`` remain owned
        by the bridge lifecycle, and ``provider_call`` remains owned by the
        real provider ledger.  A stage hook therefore cannot forge a root,
        terminal, or provider observation.  When an
        ``observation_checkpoint`` is supplied, every explicitly named parent
        must have been observed at that checkpoint or in that provider delta;
        the bridge never fills in parents automatically.
        """

        if event_type in {"request_received", "request_completed", "provider_call"}:
            raise TraceStateError(
                "query trace lifecycle/provider events are owned by the bridge ledger"
            )
        parents = tuple(parent_event_ids)
        if any(not isinstance(value, str) for value in parents):
            raise TraceContractError("trace event parent_event_ids must be strings")
        if not parents:
            raise TraceStateError("an observed trace stage requires explicit parents")
        with self._lock:
            if self._request_event_id is None:
                raise TraceStateError("cannot emit a trace stage before bridge start")
            if self._completed:
                raise TraceStateError("cannot emit a trace stage after completion")
            if self.ledger.faulted:
                raise TracePersistenceError("strict provider trace is faulted")
            if observation_checkpoint is not None:
                if isinstance(observation_checkpoint, QueryTraceObservationCheckpoint):
                    self._validate_observation_checkpoint(observation_checkpoint)
                    available_parents = set(
                        observation_checkpoint.observed_parent_event_ids
                    )
                else:
                    self._validate_observation_delta(observation_checkpoint)
                    available_parents = set(
                        observation_checkpoint.causal_parent_event_ids
                    )
                if not set(parents).issubset(available_parents):
                    raise TraceStateError(
                        "observed stage parents are not present in the supplied "
                        "observation checkpoint"
                    )
            event_id = self.session.emit(
                event_type,
                stage=stage,
                payload=payload,
                artifact_refs=artifact_refs,
                parent_event_ids=parents,
            )
            self._observed_stage_event_ids.append(event_id)
            return event_id

    @contextmanager
    def bind_model(self, model: Any):
        """Bind the model's real HTTP hook to this request-local ledger."""

        trace_request = getattr(model, "trace_request", None)
        if not callable(trace_request):
            raise TraceStateError(
                "strict query trace needs a model with trace_request support"
            )
        with trace_request(self.ledger):
            yield self.ledger

    def _complete(
        self,
        *,
        route: str,
        status: str,
        reason_code: str,
    ) -> str:
        with self._lock:
            if self._request_event_id is None or self._started_at is None:
                raise TraceStateError("cannot complete a trace bridge before start")
            if self._completed:
                raise TraceStateError("query trace bridge is already completed")
            checkpoint = self.ledger.checkpoint()
            summary = checkpoint.completion_summary()
            event_id = self.session.emit(
                "request_completed",
                stage="completion",
                payload={
                    "route": route,
                    "status": status,
                    "total_ms": round(
                        max(0.0, (monotonic() - self._started_at) * 1000.0),
                        6,
                    ),
                    "cloud_logical_calls": summary["cloud_logical_calls"],
                    "cloud_http_attempts": summary["cloud_http_attempts"],
                    "contextual_http_attempts": summary[
                        "contextual_http_attempts"
                    ],
                    "embedding_logical_batches": summary[
                        "embedding_logical_batches"
                    ],
                    # A renderer may list unobserved v3 stages.  Do not claim
                    # a stage was intentionally skipped unless a future engine
                    # integration directly observes that decision.
                    "actually_skipped_stages": [],
                    "learning_receipt_ids": [],
                    "fallback_kind": summary["fallback_kind"],
                    "reason_code": str(reason_code),
                },
                parent_event_ids=(
                    self.request_event_id,
                    *self._observed_stage_event_ids,
                    *summary["provider_event_ids"],
                ),
            )
            self._completed = True
            return event_id

    def complete_success(self) -> str:
        return self._complete(
            route="resolve",
            status="completed",
            reason_code="legacy_query_pipeline_completed",
        )

    def complete_failure(self, exc: BaseException) -> str | None:
        """Record a normal technical failure; preserve abrupt-stop truthfully.

        A process interrupt may occur after a provider request has started but
        before controlled work has joined.  It must leave the run incomplete,
        rather than manufacturing a terminal receipt.
        """

        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            return None
        if isinstance(exc, TracePersistenceError) or self.ledger.faulted:
            return None
        if isinstance(exc, (TraceContractError, TraceStateError, TracePrivacyError)):
            # Persistence itself failed; a second trace write cannot make that
            # trace trustworthy and must not hide the original exception.
            return None
        return self._complete(
            route="failed",
            status="technical_failure",
            reason_code=f"query_exception_{type(exc).__name__}",
        )
