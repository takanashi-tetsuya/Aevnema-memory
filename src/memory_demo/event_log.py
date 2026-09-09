from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np


REDACTED_VALUE = "[REDACTED]"

# Event logs and progress ledgers are routinely retained longer and shared
# more widely than the process that created them.  Keep the key list explicit
# so a new configuration field cannot become exportable merely because it was
# added to a dataclass.
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
_CONTENT_KEY_NAMES = {
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
    "question",
    "answer",
    "response",
}
_URL_KEY_NAMES = {"url", "uri", "baseurl", "endpointurl", "targeturl"}
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


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


def _content_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, dict):
        return sum(_content_length(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_content_length(item) for item in value)
    return 0


def _redact_url(value: str) -> str:
    """Keep a URL's route while removing credentials and query values."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return REDACTED_VALUE
    if not parsed.scheme and not parsed.netloc:
        return _BEARER_PATTERN.sub(r"\1" + REDACTED_VALUE, value)
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        return REDACTED_VALUE
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def redact_for_export(value: Any, *, key: str | None = None) -> Any:
    """Return a recursively redacted, JSON-compatible export value.

    This is deliberately used for logs and persisted operational snapshots,
    not for in-memory request payloads.  It removes credentials and source or
    prompt bodies while retaining lengths and safe execution metadata.
    """

    normalized_key = _normalized_key(key) if key is not None else ""
    if key is not None and _is_secret_key(key):
        return REDACTED_VALUE
    if normalized_key in _CONTENT_KEY_NAMES:
        return {"redacted": True, "char_count": _content_length(value)}
    if normalized_key in _URL_KEY_NAMES and isinstance(value, str):
        return _redact_url(value)
    if normalized_key in {"headers", "header"}:
        return REDACTED_VALUE
    if is_dataclass(value):
        return redact_for_export(asdict(value), key=key)
    if isinstance(value, dict):
        return {
            str(item_key): redact_for_export(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_for_export(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return _BEARER_PATTERN.sub(r"\1" + REDACTED_VALUE, value)
    return value


def redact_for_answer_evidence_checkpoint(value: Any, *, key: str | None = None) -> Any:
    """Redact credentials while retaining the narrow, locally auditable evidence.

    Operational JSONL deliberately hides all prompt and answer bodies.  An
    answer-boundary checkpoint has a different, explicitly authorized purpose:
    it preserves only the question, selected Source excerpts, answer revision,
    and audit result needed to reconstruct one experiment.  It still redacts
    credentials, headers, and URL query values recursively.
    """

    normalized_key = _normalized_key(key) if key is not None else ""
    if key is not None and _is_secret_key(key):
        return REDACTED_VALUE
    if normalized_key in _URL_KEY_NAMES and isinstance(value, str):
        return _redact_url(value)
    if normalized_key in {"headers", "header"}:
        return REDACTED_VALUE
    if is_dataclass(value):
        return redact_for_answer_evidence_checkpoint(asdict(value), key=key)
    if isinstance(value, dict):
        return {
            str(item_key): redact_for_answer_evidence_checkpoint(
                item_value, key=str(item_key)
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_for_answer_evidence_checkpoint(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return _BEARER_PATTERN.sub(r"\1" + REDACTED_VALUE, value)
    return value


def safe_config_snapshot(config: Any) -> dict[str, Any]:
    """Create an allow-listed configuration receipt with no credentials.

    Paths and unknown future root fields are intentionally omitted.  The
    snapshot describes execution-relevant behavior, not a machine or account.
    """

    raw = asdict(config) if is_dataclass(config) else dict(config)
    model = dict(raw.get("model") or {})
    allowed_model_keys = (
        "base_url",
        "embedding_model",
        "reranker_model",
        "reasoning_model",
        "fallback_model",
        "reasoning_max_tokens",
        "reasoning_enable_thinking",
        "embedding_dimension",
        "timeout_seconds",
        "max_retries",
        "max_concurrent_requests",
    )
    snapshot: dict[str, Any] = {
        "snapshot_schema": "memory_demo.safe_config.v1",
        "prompt_version": raw.get("prompt_version"),
        "optimization_profile": raw.get("optimization_profile"),
        "database_filename": Path(str(raw.get("database_path", ""))).name,
        "model": {
            key: model.get(key)
            for key in allowed_model_keys
            if key in model
        },
        "model_api_key_configured": bool(model.get("api_key")),
    }
    for section in (
        "segment",
        "paragraph",
        "ingestion",
        "concept_extraction",
        "retrieval",
        "weights",
    ):
        if section in raw:
            snapshot[section] = raw[section]
    return redact_for_export(snapshot)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return {"bytes_length": len(value)}
    raise TypeError(f"cannot serialize {type(value).__name__}")


class JsonlEventLogger:
    def __init__(self, path: str | Path, *, answer_evidence_enabled: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.answer_evidence_enabled = bool(answer_evidence_enabled)
        self._lock = Lock()

    @property
    def answer_evidence_path(self) -> Path:
        """Local-only companion stream for explicitly scoped answer evidence."""

        return self.path.with_name(self.path.stem + ".answer-evidence.jsonl")

    def _write(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(
            record,
            ensure_ascii=False,
            default=_json_default,
            separators=(",", ":"),
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def emit(self, event: str, **payload: Any) -> None:
        record = redact_for_export({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        })
        self._write(self.path, record)

    def emit_answer_evidence_checkpoint(self, event: str, **payload: Any) -> None:
        """Persist an allow-listed answer-boundary evidence receipt locally.

        This intentionally does not relax the ordinary event-log redaction
        policy.  The companion file is created only by QueryEngine's answer
        boundary and has no provider request, arbitrary prompt, configuration,
        or general payload channel.
        """

        allowed_events = {
            "answer_input_checkpoint",
            "answer_revision_checkpoint",
        }
        allowed_payload_keys = {
            "stage",
            "question",
            "requirements",
            "version_binding",
            "selected_evidence",
            "prompt_evidence_assertions",
            "answer",
            "answer_revision_hash",
            "audit",
            "audit_index",
            "revision_count",
            "remaining_budget_seconds",
            "terminal_reason",
            "error_type",
        }
        if event not in allowed_events:
            raise ValueError(f"unsupported answer evidence event: {event}")
        if not self.answer_evidence_enabled:
            raise RuntimeError(
                "answer evidence checkpoints require explicit experimental enablement"
            )
        unknown = set(payload).difference(allowed_payload_keys)
        if unknown:
            raise ValueError(
                "answer evidence checkpoint contains unapproved fields: "
                + ", ".join(sorted(unknown))
            )
        record = redact_for_answer_evidence_checkpoint({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "schema": "memory_demo.answer_boundary_evidence.v1",
            "local_only": True,
            **payload,
        })
        self._write(self.answer_evidence_path, record)
