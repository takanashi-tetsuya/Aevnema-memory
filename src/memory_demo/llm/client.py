from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Protocol
from threading import BoundedSemaphore, Condition, Event, Lock, Thread, local
from uuid import uuid4

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool

from memory_demo.config import ModelConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.runtime.deadline import DeadlineBudget


# A process-wide gate protects Windows from bursts of simultaneous TCP/TLS
# handshakes when several ModelClient instances start with empty pools. It is
# deliberately narrower than the request semaphore: established pooled
# connections continue to serve concurrent requests.
_CONNECTION_ESTABLISHMENT_LOCK = Lock()


class _SerializedHTTPSConnection(HTTPSConnection):
    """Serialize only physical HTTPS connection establishment."""

    def connect(self) -> None:
        with _CONNECTION_ESTABLISHMENT_LOCK:
            super().connect()


class _SerializedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _SerializedHTTPSConnection


class _SerializedConnectionHTTPAdapter(HTTPAdapter):
    """Adapter whose HTTPS pools construct sockets one at a time."""

    @staticmethod
    def _install_serialized_https_pool(manager: Any) -> None:
        manager.pool_classes_by_scheme = manager.pool_classes_by_scheme.copy()
        manager.pool_classes_by_scheme["https"] = _SerializedHTTPSConnectionPool

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)
        self._install_serialized_https_pool(self.poolmanager)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        self._install_serialized_https_pool(manager)
        return manager


def _payload_content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_payload_content_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(_payload_content_chars(item) for item in value.values())
    return 0


def _request_log_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    inputs = payload.get("input")
    return {
        "model": payload.get("model"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "input_count": len(inputs) if isinstance(inputs, list) else int(inputs is not None),
        "input_chars": _payload_content_chars(messages)
        + _payload_content_chars(inputs),
        "max_tokens": payload.get("max_tokens"),
        "top_n": payload.get("top_n"),
    }


def _response_log_summary(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    data = payload.get("data")
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "data_count": len(data) if isinstance(data, list) else 0,
        "output_chars": _payload_content_chars(choices),
        "usage": payload.get("usage"),
    }


class ModelClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


class ModelTransportUnavailable(ModelClientError):
    """A local network resource failure that must stop batch dispatch."""


class ModelDeadlineExceeded(TimeoutError):
    """The client-side request budget ended before work could be used.

    This intentionally is not a ``ModelClientError``: retry/fallback loops
    must not reinterpret an exhausted caller deadline as a provider failure.
    A sent request can still finish remotely, but its result is discarded and
    never reaches parsing, learning, or answer delivery.
    """

    def __init__(self, stage: str, *, late_result_discarded: bool = False) -> None:
        super().__init__(f"provider deadline exceeded before {stage}")
        self.stage = str(stage)
        self.late_result_discarded = bool(late_result_discarded)


class TracePersistenceError(RuntimeError):
    """A request-local provider trace sink could not persist an observation.

    This is deliberately neither :class:`ModelClientError` nor
    :class:`ValueError`.  A durable trace failure is not a provider failure:
    retrying or trying a fallback model after a real provider response would
    create unreceipted work.  Query-level best-effort paths also use this
    distinct signal to leave the strict trace incomplete rather than
    manufacturing a successful terminal receipt.
    """


@dataclass(frozen=True, slots=True)
class ProviderCallObservation:
    """One attempted provider request, without request/response bodies."""

    call_id: str
    logical_batch_id: str | None
    operation: str
    endpoint: str
    requested_model: str
    actual_model: str | None
    role: str
    purpose: str
    status: str
    sent: bool
    queue_ms: float
    network_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    http_status: int | None
    error_class: str | None
    late_result_discarded: bool = False


class ProviderTraceSink(Protocol):
    """Optional request-local observer; it must never contain raw payloads."""

    def logical_batch_started(
        self,
        *,
        logical_batch_id: str,
        operation: str,
        requested_model: str,
    ) -> None: ...

    def provider_call_finished(self, observation: ProviderCallObservation) -> None: ...

    def fallback_used(
        self,
        *,
        logical_batch_id: str,
        from_model: str,
        to_model: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderCallScope:
    """One caller-visible operation, including all of its retries/fallbacks."""

    logical_batch_id: str
    operation: str
    model: str


class ProviderCallAccounting:
    """Thread-safe accounting of logical work and actual HTTP attempts."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._logical_batches = 0
        self._http_attempts = 0
        self._http_succeeded = 0
        self._http_failed = 0
        self._retries_scheduled = 0
        self._fallbacks = 0
        self._operations: Counter[str] = Counter()
        self._endpoints: Counter[str] = Counter()
        self._outcomes: Counter[str] = Counter()

    def begin(self, operation: str, model: str) -> ProviderCallScope:
        scope = ProviderCallScope(uuid4().hex, str(operation), str(model))
        with self._lock:
            self._logical_batches += 1
            self._operations[scope.operation] += 1
        return scope

    def record_http_attempt(self, scope: ProviderCallScope, endpoint: str) -> None:
        del scope  # The id is retained in the event stream, not an unbounded map.
        with self._lock:
            self._http_attempts += 1
            self._endpoints[str(endpoint)] += 1

    def record_http_result(self, scope: ProviderCallScope, outcome: str) -> None:
        del scope
        with self._lock:
            self._outcomes[str(outcome)] += 1
            if outcome == "success":
                self._http_succeeded += 1
            else:
                self._http_failed += 1

    def record_retry(self, scope: ProviderCallScope) -> None:
        del scope
        with self._lock:
            self._retries_scheduled += 1

    def record_fallback(self, scope: ProviderCallScope) -> None:
        del scope
        with self._lock:
            self._fallbacks += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "logical_batches": self._logical_batches,
                "http_attempts": self._http_attempts,
                "http_succeeded": self._http_succeeded,
                "http_failed": self._http_failed,
                "retries_scheduled": self._retries_scheduled,
                "fallbacks": self._fallbacks,
                "operations": dict(sorted(self._operations.items())),
                "endpoints": dict(sorted(self._endpoints.items())),
                "outcomes": dict(sorted(self._outcomes.items())),
            }

    def reset(self) -> None:
        with self._lock:
            self._logical_batches = 0
            self._http_attempts = 0
            self._http_succeeded = 0
            self._http_failed = 0
            self._retries_scheduled = 0
            self._fallbacks = 0
            self._operations.clear()
            self._endpoints.clear()
            self._outcomes.clear()


class CampaignBudgetError(RuntimeError):
    """A local campaign-budget control signal, never a provider response."""


class CampaignBudgetExhausted(CampaignBudgetError):
    """No further provider HTTP attempt may be dispatched for this campaign.

    This deliberately does not inherit :class:`ModelClientError`.  Retry,
    repair, and fallback loops must stop rather than interpreting an exhausted
    campaign allowance as a reason to try another paid provider request.
    """


class CampaignBudgetConfigurationError(CampaignBudgetError):
    """The durable campaign receipt cannot safely be opened or updated."""


@dataclass(frozen=True, slots=True)
class CampaignHttpReservation:
    """A conservative, durable reservation for one actual HTTP dispatch."""

    campaign_id: str
    ordinal: int
    remaining_after_reservation: int


class CampaignHttpBudget:
    """Small shared hard-stop for one explicitly configured model campaign.

    The object is intentionally opt-in: ordinary application calls keep their
    existing behaviour until a controlled runner opens a named campaign and
    supplies it to every ``ModelClient`` it creates.  A single process is the
    dispatcher for a campaign.  Within that process, ``open`` returns one
    shared object for each receipt path, and the receipt makes a later process
    continue from the already-reserved count.

    A reservation is made immediately before transport dispatch and is never
    released.  That is conservative by design: cancellation, timeout, or an
    unknown provider outcome can still have consumed billable work.
    """

    schema = "aevnema.campaign_http_budget.v1"
    _opened_lock = Lock()
    _opened: dict[Path, "CampaignHttpBudget"] = {}

    def __init__(
        self,
        *,
        campaign_id: str,
        max_http_attempts: int,
        receipt_path: Path,
    ) -> None:
        normalized_campaign_id = str(campaign_id).strip()
        if not normalized_campaign_id:
            raise ValueError("campaign_id must not be empty")
        if int(max_http_attempts) < 1:
            raise ValueError("max_http_attempts must be at least one")
        self.campaign_id = normalized_campaign_id
        self.max_http_attempts = int(max_http_attempts)
        self.receipt_path = Path(receipt_path).expanduser().resolve()
        self._lock = Lock()
        self._reserved_http_attempts = 0
        self._load_or_create()

    @classmethod
    def open(
        cls,
        *,
        campaign_id: str,
        max_http_attempts: int,
        receipt_path: str | Path,
    ) -> "CampaignHttpBudget":
        """Open the process-shared budget for a controlled campaign.

        The path is part of the identity so an unrelated campaign cannot
        silently inherit this campaign's allowance.  Reopening the same path
        with a different id or cap is rejected instead of resetting it.
        """

        path = Path(receipt_path).expanduser().resolve()
        with cls._opened_lock:
            existing = cls._opened.get(path)
            if existing is not None:
                if (
                    existing.campaign_id != str(campaign_id).strip()
                    or existing.max_http_attempts != int(max_http_attempts)
                ):
                    raise CampaignBudgetConfigurationError(
                        "campaign receipt is already open with different settings"
                    )
                return existing
            budget = cls(
                campaign_id=campaign_id,
                max_http_attempts=max_http_attempts,
                receipt_path=path,
            )
            cls._opened[path] = budget
            return budget

    @classmethod
    def _clear_process_cache_for_test(cls) -> None:
        """Simulate a fresh dispatcher in local persistence tests only."""

        with cls._opened_lock:
            cls._opened.clear()

    def _validated_payload(self, payload: object) -> int:
        if not isinstance(payload, dict):
            raise CampaignBudgetConfigurationError("campaign receipt is not an object")
        if payload.get("schema") != self.schema:
            raise CampaignBudgetConfigurationError("campaign receipt schema mismatch")
        if payload.get("campaign_id") != self.campaign_id:
            raise CampaignBudgetConfigurationError("campaign receipt id mismatch")
        if payload.get("max_http_attempts") != self.max_http_attempts:
            raise CampaignBudgetConfigurationError("campaign receipt cap mismatch")
        reserved = payload.get("reserved_http_attempts")
        if isinstance(reserved, bool) or not isinstance(reserved, int):
            raise CampaignBudgetConfigurationError("campaign receipt count is invalid")
        if reserved < 0 or reserved > self.max_http_attempts:
            raise CampaignBudgetConfigurationError("campaign receipt count is out of range")
        return reserved

    def _payload_locked(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "max_http_attempts": self.max_http_attempts,
            "reserved_http_attempts": self._reserved_http_attempts,
        }

    def _persist_locked(self) -> None:
        try:
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.receipt_path.with_name(
                f".{self.receipt_path.name}.{uuid4().hex}.tmp"
            )
            temporary.write_text(
                json.dumps(self._payload_locked(), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.receipt_path)
        except OSError as exc:
            raise CampaignBudgetConfigurationError(
                "campaign receipt could not be persisted before dispatch"
            ) from exc

    def _load_or_create(self) -> None:
        with self._lock:
            try:
                if self.receipt_path.exists():
                    self._reserved_http_attempts = self._validated_payload(
                        json.loads(self.receipt_path.read_text(encoding="utf-8"))
                    )
                    return
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignBudgetConfigurationError(
                    "campaign receipt could not be read safely"
                ) from exc
            self._persist_locked()

    def reserve(self) -> CampaignHttpReservation:
        """Atomically reserve one irreversible real provider dispatch."""

        with self._lock:
            if self._reserved_http_attempts >= self.max_http_attempts:
                raise CampaignBudgetExhausted(
                    "campaign HTTP budget exhausted before provider dispatch"
                )
            self._reserved_http_attempts += 1
            self._persist_locked()
            return CampaignHttpReservation(
                campaign_id=self.campaign_id,
                ordinal=self._reserved_http_attempts,
                remaining_after_reservation=(
                    self.max_http_attempts - self._reserved_http_attempts
                ),
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                **self._payload_locked(),
                "remaining_http_attempts": (
                    self.max_http_attempts - self._reserved_http_attempts
                ),
                "receipt_filename": self.receipt_path.name,
            }


_PROCESS_PROVIDER_ACCOUNTING = ProviderCallAccounting()
_ACTIVE_PROVIDER_SCOPE: ContextVar[tuple[object, ProviderCallScope] | None] = ContextVar(
    "memory_demo_active_provider_scope", default=None
)
_ACTIVE_PROVIDER_TRACE: ContextVar[tuple[object, ProviderTraceSink] | None] = ContextVar(
    "memory_demo_active_provider_trace", default=None
)
_ACTIVE_PROVIDER_PURPOSE: ContextVar[tuple[object, str, str] | None] = ContextVar(
    "memory_demo_active_provider_purpose", default=None
)
_ACTIVE_PROVIDER_DEADLINE: ContextVar[tuple[object, DeadlineBudget] | None] = ContextVar(
    "memory_demo_active_provider_deadline", default=None
)
_ACTIVE_PROVIDER_WORKLOAD: ContextVar[tuple[object, str] | None] = ContextVar(
    "memory_demo_active_provider_workload", default=None
)


class _ProviderRequestScheduler:
    """A process-local provider gate with foreground-over-import admission.

    The provider cap is shared by clients that use the same endpoint and
    credential.  This matters when an importer and an interactive query live
    in one Python process: per-client semaphores alone could each admit a full
    configured batch.  Running work is never pre-empted, but when a slot is
    released an interactive waiter is admitted before a background/import
    waiter.

    A later client may only make the shared cap stricter.  Divergent client
    settings should never cause the process to exceed the most conservative
    caller's declared provider budget.  A fresh process reads the configured
    cap again, which is also how ordinary application starts are isolated.
    """

    def __init__(self, capacity: int) -> None:
        self._condition = Condition(Lock())
        self._capacity = max(1, int(capacity))
        self._active = 0
        self._interactive_waiters = 0
        self._background_waiters = 0

    def restrict(self, capacity: int) -> None:
        """Apply a safe, process-lifetime upper bound for this provider."""

        with self._condition:
            self._capacity = min(self._capacity, max(1, int(capacity)))
            self._condition.notify_all()

    def acquire(self, *, priority: str, deadline_at: float | None) -> bool:
        """Acquire one slot, returning ``False`` if its deadline elapsed."""

        if priority not in {"interactive", "background"}:
            raise ValueError(f"unsupported provider priority: {priority!r}")
        with self._condition:
            if priority == "interactive":
                self._interactive_waiters += 1
            else:
                self._background_waiters += 1
            acquired = False
            try:
                while True:
                    foreground_waiting = self._interactive_waiters > 0
                    may_enter = self._active < self._capacity and (
                        priority == "interactive" or not foreground_waiting
                    )
                    if may_enter:
                        self._active += 1
                        acquired = True
                        return True
                    if deadline_at is None:
                        self._condition.wait()
                        continue
                    remaining = float(deadline_at) - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=remaining)
            finally:
                if priority == "interactive":
                    self._interactive_waiters -= 1
                else:
                    self._background_waiters -= 1
                # A background waiter may become eligible just because the
                # final interactive waiter departed due to deadline expiry.
                if not acquired:
                    self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("provider scheduler release without acquisition")
            self._active -= 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        """Return metadata-only scheduler state for diagnostics/tests."""

        with self._condition:
            return {
                "capacity": int(self._capacity),
                "active": int(self._active),
                "interactive_waiters": int(self._interactive_waiters),
                "background_waiters": int(self._background_waiters),
            }


_PROCESS_PROVIDER_SCHEDULERS_LOCK = Lock()
_PROCESS_PROVIDER_SCHEDULERS: dict[tuple[str, bytes], _ProviderRequestScheduler] = {}


def _provider_scheduler_key(config: ModelConfig) -> tuple[str, bytes]:
    """Identify a provider budget without retaining/logging a raw API key."""

    endpoint = str(config.base_url).rstrip("/")
    credential_digest = hashlib.sha256(
        str(config.api_key).encode("utf-8")
    ).digest()
    return endpoint, credential_digest


def _process_provider_scheduler(config: ModelConfig) -> _ProviderRequestScheduler:
    key = _provider_scheduler_key(config)
    requested_capacity = max(1, int(config.max_concurrent_requests))
    with _PROCESS_PROVIDER_SCHEDULERS_LOCK:
        scheduler = _PROCESS_PROVIDER_SCHEDULERS.get(key)
        if scheduler is None:
            scheduler = _ProviderRequestScheduler(requested_capacity)
            _PROCESS_PROVIDER_SCHEDULERS[key] = scheduler
        else:
            scheduler.restrict(requested_capacity)
        return scheduler


class _DeadlineIsolatedPost:
    """Run one blocking transport call without letting it outlive its lease.

    ``requests`` has connect/read timeouts, but neither is a total wall-clock
    deadline when a peer keeps delivering tiny chunks.  A foreground caller
    can therefore stop waiting at its absolute deadline while this helper
    retains the actual provider/local slots until the physical request exits.
    The eventual response is closed and never returned to parsing/retry logic.
    """

    def __init__(self, invoke: Callable[[], Any]) -> None:
        self._invoke = invoke
        self._lock = Lock()
        self._done = Event()
        self._completed = False
        self._response: Any | None = None
        self._error: BaseException | None = None
        self._abandoned_callback: (
            Callable[[Any | None, BaseException | None], None] | None
        ) = None

    def start(self) -> None:
        Thread(
            target=self._run,
            name="memory-provider-deadline",
            daemon=True,
        ).start()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout=max(0.0, float(timeout)))

    def result(self) -> Any:
        with self._lock:
            if not self._completed:
                raise RuntimeError("isolated provider call has not completed")
            response = self._response
            error = self._error
        if error is not None:
            raise error
        return response

    def abandon(
        self,
        callback: Callable[[Any | None, BaseException | None], None],
    ) -> None:
        """Make one cleanup owner for a result the foreground will discard.

        If the worker is still running, it owns the callback once it exits.
        If it won the race just before abandonment, perform the same cleanup
        synchronously instead.  In both cases the foreground must relinquish
        its lease ownership; otherwise an interrupt in that tiny race can
        leave a completed response unclosed.
        """

        with self._lock:
            if not self._completed:
                self._abandoned_callback = callback
                return
            response = self._response
            error = self._error
        callback(response, error)

    def _run(self) -> None:
        response: Any | None = None
        error: BaseException | None = None
        try:
            response = self._invoke()
        except BaseException as exc:  # delivered only to the active caller
            error = exc
        with self._lock:
            self._response = response
            self._error = error
            self._completed = True
            callback = self._abandoned_callback
        self._done.set()
        if callback is not None:
            try:
                callback(response, error)
            except Exception:
                # Cleanup is deliberately best-effort. The foreground request
                # has already received its deadline failure and must not see a
                # late worker exception or raw provider diagnostic.
                pass


def provider_call_accounting_snapshot() -> dict[str, Any]:
    """Return process-wide actual provider-call counters for run manifests."""

    return _PROCESS_PROVIDER_ACCOUNTING.snapshot()


def reset_provider_call_accounting_for_tests() -> None:
    """Reset the default process ledger; intended only for isolated tests."""

    _PROCESS_PROVIDER_ACCOUNTING.reset()


def _quote_bare_line_ids(text: str) -> str:
    """Quote JSON tokens such as L0007 without touching string contents."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        match = re.match(r"[Ll]0*\d+", text[index:])
        if match:
            end = index + len(match.group(0))
            previous = text[index - 1] if index else ""
            following = text[end] if end < len(text) else ""
            if (
                (not previous or not (previous.isalnum() or previous == "_"))
                and (not following or not (following.isalnum() or following == "_"))
            ):
                output.append(json.dumps(match.group(0)))
                index = end
                continue
        output.append(character)
        index += 1
    return "".join(output)


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    line_id_normalized = _quote_bare_line_ids(cleaned)
    if line_id_normalized != cleaned:
        candidates.append(line_id_normalized)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        positions = [
            position
            for position in (candidate.find("{"), candidate.find("["))
            if position >= 0
        ]
        for position in sorted(positions):
            try:
                value, _ = decoder.raw_decode(candidate[position:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("model response does not contain valid JSON")


class ModelClient:
    """Minimal OpenAI-compatible HTTP client with logging and fallback."""

    semantic_output_format = "natural_text"

    def __init__(
        self,
        config: ModelConfig,
        logger: JsonlEventLogger | None = None,
        *,
        accounting: ProviderCallAccounting | None = None,
        campaign_budget: CampaignHttpBudget | None = None,
    ):
        self.config = config
        self.logger = logger
        # Do not use ``id(self)`` in ContextVars: a copied/long-lived context
        # can outlive a client, after which CPython may reuse that integer for
        # another client.  This opaque token cannot collide by identity.
        self._provider_context_identity = object()
        self._accounting = accounting or _PROCESS_PROVIDER_ACCOUNTING
        self._campaign_budget = campaign_budget
        request_limit = max(1, int(config.max_concurrent_requests))
        # Keep the per-client semaphore for the client's own Session/adapter
        # pool, then apply a second, process-wide provider gate immediately
        # before dispatch.  The latter is what makes import and query clients
        # share one real upstream budget.
        self._request_semaphore = BoundedSemaphore(request_limit)
        self._provider_scheduler = _process_provider_scheduler(config)
        # urllib.request opens a new connection for every request.  A single
        # import client is shared by file workers, so use one shared adapter
        # (and therefore one connection pool) while keeping Session instances
        # thread-local.  This avoids a 16-worker retry burst repeatedly
        # creating Windows sockets, yet does not share mutable Session state
        # across threads.
        self._http_adapter = _SerializedConnectionHTTPAdapter(
            pool_connections=1,
            pool_maxsize=request_limit,
            pool_block=True,
            max_retries=0,
        )
        self._thread_sessions = local()
        self._transport_lock = Lock()
        self._transport_blocked_until = 0.0
        self._transport_error = ""

    def provider_call_snapshot(self) -> dict[str, Any]:
        """Expose actual provider work for a run receipt or test assertion."""

        return self._accounting.snapshot()

    def provider_limit_snapshot(self) -> dict[str, int]:
        """Expose only shared queue counters; no endpoint/key material leaks."""

        return self._provider_scheduler.snapshot()

    def campaign_budget_snapshot(self) -> dict[str, object] | None:
        """Return the attached controlled-campaign receipt, if one exists."""

        if self._campaign_budget is None:
            return None
        return self._campaign_budget.snapshot()

    def _current_deadline_budget(self) -> DeadlineBudget | None:
        """Return this client's request-local budget without cross-talk."""

        current = _ACTIVE_PROVIDER_DEADLINE.get()
        if current is None or current[0] is not self._provider_context_identity:
            return None
        return current[1]

    def _remaining_deadline_seconds(self) -> float | None:
        budget = self._current_deadline_budget()
        return None if budget is None else budget.remaining()

    def _require_deadline(self, stage: str) -> float | None:
        """Fail before local work that can no longer be delivered."""

        budget = self._current_deadline_budget()
        if budget is None:
            return None
        remaining = budget.remaining()
        if remaining <= 0.0:
            raise ModelDeadlineExceeded(stage)
        return remaining

    @contextmanager
    def deadline_budget(self, deadline_at: float | None):
        """Bind one absolute monotonic client-side deadline to this request.

        Nesting deliberately replaces an outer value only inside the nested
        scope.  A shared client may serve imports and interactive queries on
        different threads, so storing the deadline on ``self`` would let one
        request shorten another request's budget.
        """

        if deadline_at is None:
            yield None
            return
        budget = DeadlineBudget(float(deadline_at))
        token = _ACTIVE_PROVIDER_DEADLINE.set(
            (self._provider_context_identity, budget)
        )
        try:
            yield budget
        finally:
            _ACTIVE_PROVIDER_DEADLINE.reset(token)

    def _current_call_scope(self) -> ProviderCallScope | None:
        current = _ACTIVE_PROVIDER_SCOPE.get()
        if current is None or current[0] is not self._provider_context_identity:
            return None
        return current[1]

    def _current_trace_sink(self) -> ProviderTraceSink | None:
        """Return the request-local observer bound to this client, if any.

        This deliberately uses a :class:`ContextVar` rather than client state:
        one shared import client can therefore serve several worker requests
        without letting one request's receipt absorb another request's calls.
        """

        current = _ACTIVE_PROVIDER_TRACE.get()
        if current is None or current[0] is not self._provider_context_identity:
            return None
        return current[1]

    def _current_provider_purpose(self, operation: str) -> tuple[str, str]:
        """Return an explicit caller purpose, or a conservative local label."""

        current = _ACTIVE_PROVIDER_PURPOSE.get()
        if current is not None and current[0] is self._provider_context_identity:
            return current[1], current[2]
        if operation == "embedding":
            return "embedding", "model_client:embedding"
        if operation == "rerank":
            return "reranker", "model_client:rerank"
        return "other", f"model_client:{operation}"

    def _current_provider_priority(self, role: str) -> str:
        """Resolve a request-local workload class without cross-client state."""

        current = _ACTIVE_PROVIDER_WORKLOAD.get()
        if current is not None and current[0] is self._provider_context_identity:
            return current[1]
        # Explicit ingestion labels are background even when callers do not
        # use the broader workload scope. Unknown work defaults foreground so
        # it cannot accidentally be delayed behind a corpus import.
        return "background" if role == "ingestion" else "interactive"

    @contextmanager
    def trace_request(self, sink: ProviderTraceSink):
        """Bind one request-local trace sink for calls made in this context.

        The sink is intentionally called synchronously.  If durable trace
        persistence fails, that error is not converted to ``ModelClientError``
        and hence cannot trigger a costly provider retry after an already sent
        request.  ``ContextVar`` bindings do not cross executor threads
        automatically; callers that dispatch work must bind explicitly there.
        """

        if sink is None:
            raise ValueError("trace sink must not be None")
        token = _ACTIVE_PROVIDER_TRACE.set((self._provider_context_identity, sink))
        try:
            yield sink
        finally:
            _ACTIVE_PROVIDER_TRACE.reset(token)

    @contextmanager
    def provider_purpose(self, role: str, purpose: str):
        """Attach a truthful semantic role to provider calls in this context."""

        normalized_role = str(role).strip()
        normalized_purpose = str(purpose).strip()
        if normalized_role not in {
            "planner",
            "embedding",
            "reranker",
            "answer",
            "audit",
            "ingestion",
            "other",
        }:
            raise ValueError(f"unsupported provider trace role: {role!r}")
        if not normalized_purpose:
            raise ValueError("provider trace purpose must not be empty")
        token = _ACTIVE_PROVIDER_PURPOSE.set(
            (self._provider_context_identity, normalized_role, normalized_purpose)
        )
        try:
            yield
        finally:
            _ACTIVE_PROVIDER_PURPOSE.reset(token)

    @contextmanager
    def provider_workload(self, workload: str):
        """Classify a bounded unit of work for shared provider admission.

        ``background`` is used by corpus import workers; ordinary query work
        remains ``interactive``. This is scheduling metadata only: it neither
        changes model routing nor bypasses the common HTTP accounting.
        """

        normalized = str(workload).strip().casefold()
        if normalized not in {"interactive", "background"}:
            raise ValueError(f"unsupported provider workload: {workload!r}")
        token = _ACTIVE_PROVIDER_WORKLOAD.set(
            (self._provider_context_identity, normalized)
        )
        try:
            yield
        finally:
            _ACTIVE_PROVIDER_WORKLOAD.reset(token)

    @staticmethod
    def _usage_tokens(value: Any) -> tuple[int | None, int | None]:
        """Extract only provider-reported non-negative token counters."""

        if not isinstance(value, dict):
            return None, None
        usage = value.get("usage")
        if not isinstance(usage, dict):
            return None, None

        def integer(*names: str) -> int | None:
            for name in names:
                candidate = usage.get(name)
                if isinstance(candidate, bool):
                    continue
                if isinstance(candidate, int) and candidate >= 0:
                    return candidate
            return None

        return (
            integer("prompt_tokens", "input_tokens"),
            integer("completion_tokens", "output_tokens"),
        )

    @staticmethod
    def _response_model(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        model = value.get("model")
        if not isinstance(model, str) or not model.strip():
            return None
        return model.strip()

    @staticmethod
    def _http_status(value: Any) -> int | None:
        candidate = getattr(value, "status_code", None)
        if isinstance(candidate, bool):
            return None
        if isinstance(candidate, int) and 100 <= candidate <= 599:
            return candidate
        return None

    @staticmethod
    def _mark_trace_sink_fault(
        sink: ProviderTraceSink, exc: BaseException
    ) -> None:
        """Let strict sinks latch a failure without exposing its contents.

        ``ProviderTraceSink`` deliberately stays usable for small observers in
        tests and integrations, so this acknowledgement hook is optional.
        A failing acknowledgement cannot safely repair the original durable
        write, and must not mask it.
        """

        acknowledge = getattr(sink, "trace_persistence_failed", None)
        if not callable(acknowledge):
            return
        try:
            acknowledge(exc)
        except Exception:
            pass

    def _notify_trace_sink(
        self,
        sink: ProviderTraceSink | None,
        method_name: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Call a trace sink without turning persistence failure into retry.

        Provider traces are receipts, not best-effort telemetry.  A sink
        failure therefore uses a dedicated signal which the retry/fallback
        loops do not classify as an ordinary model or JSON failure.
        """

        if sink is None:
            return
        callback = getattr(sink, method_name, None)
        if not callable(callback):
            error = TypeError(f"provider trace sink lacks {method_name}")
            self._mark_trace_sink_fault(sink, error)
            raise TracePersistenceError("provider trace persistence failed") from error
        try:
            callback(*args, **kwargs)
        except TracePersistenceError as exc:
            self._mark_trace_sink_fault(sink, exc)
            raise
        except Exception as exc:
            self._mark_trace_sink_fault(sink, exc)
            raise TracePersistenceError("provider trace persistence failed") from exc

    def _notify_provider_observation(
        self,
        sink: ProviderTraceSink | None,
        *,
        call_id: str,
        scope: ProviderCallScope,
        endpoint: str,
        requested_model: str,
        role: str,
        purpose: str,
        status: str,
        sent: bool,
        queue_ms: float,
        network_ms: float | None,
        total_started_at: float,
        actual_model: str | None = None,
        response_value: Any = None,
        http_status: int | None = None,
        error: BaseException | None = None,
        late_result_discarded: bool = False,
    ) -> None:
        """Emit a body-free observation without changing retry semantics."""

        if sink is None:
            return
        prompt_tokens, completion_tokens = self._usage_tokens(response_value)
        self._notify_trace_sink(
            sink,
            "provider_call_finished",
            ProviderCallObservation(
                call_id=call_id,
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                endpoint=str(endpoint),
                requested_model=requested_model,
                actual_model=actual_model,
                role=role,
                purpose=purpose,
                status=status,
                sent=bool(sent),
                queue_ms=round(max(0.0, queue_ms), 6),
                network_ms=(
                    None
                    if network_ms is None
                    else round(max(0.0, network_ms), 6)
                ),
                total_ms=round(
                    max(0.0, (time.monotonic() - total_started_at) * 1000.0),
                    6,
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                http_status=http_status,
                error_class=(type(error).__name__ if error is not None else None),
                late_result_discarded=bool(late_result_discarded),
            ),
        )

    @contextmanager
    def _logical_batch(self, operation: str, model: str):
        scope = self._accounting.begin(operation, model)
        token = _ACTIVE_PROVIDER_SCOPE.set((self._provider_context_identity, scope))
        try:
            sink = self._current_trace_sink()
            self._notify_trace_sink(
                sink,
                "logical_batch_started",
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                requested_model=scope.model,
            )
            if self.logger:
                self.logger.emit(
                    "provider_call",
                    phase="logical_batch_started",
                    logical_batch_id=scope.logical_batch_id,
                    operation=scope.operation,
                    model=scope.model,
                )
            yield scope
        finally:
            _ACTIVE_PROVIDER_SCOPE.reset(token)

    def _standalone_call_scope(
        self, endpoint: str, payload: dict[str, Any]
    ) -> ProviderCallScope:
        scope = self._accounting.begin(
            f"raw:{endpoint}", str(payload.get("model", ""))
        )
        sink = self._current_trace_sink()
        self._notify_trace_sink(
            sink,
            "logical_batch_started",
            logical_batch_id=scope.logical_batch_id,
            operation=scope.operation,
            requested_model=scope.model or "unspecified",
        )
        if self.logger:
            self.logger.emit(
                "provider_call",
                phase="logical_batch_started",
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                model=scope.model,
            )
        return scope

    def _record_http_attempt(
        self, scope: ProviderCallScope, endpoint: str
    ) -> None:
        self._accounting.record_http_attempt(scope, endpoint)
        if self.logger:
            self.logger.emit(
                "provider_call",
                phase="http_attempt",
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                model=scope.model,
                endpoint=endpoint,
            )

    def _record_http_result(
        self, scope: ProviderCallScope, endpoint: str, outcome: str
    ) -> None:
        self._accounting.record_http_result(scope, outcome)
        if self.logger:
            self.logger.emit(
                "provider_call",
                phase="http_result",
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                model=scope.model,
                endpoint=endpoint,
                outcome=outcome,
            )

    def _record_retry(self) -> None:
        scope = self._current_call_scope()
        if scope is not None:
            self._accounting.record_retry(scope)

    def _record_fallback(self, from_model: str, to_model: str) -> None:
        scope = self._current_call_scope()
        if scope is not None:
            self._accounting.record_fallback(scope)
            sink = self._current_trace_sink()
            self._notify_trace_sink(
                sink,
                "fallback_used",
                logical_batch_id=scope.logical_batch_id,
                from_model=str(from_model),
                to_model=str(to_model),
            )

    @staticmethod
    def _is_socket_resource_error(exc: BaseException) -> bool:
        """Identify local Winsock access denial before an HTTP response exists."""

        return "winerror 10013" in str(exc).casefold()

    def _transport_circuit_error(self) -> ModelTransportUnavailable | None:
        with self._transport_lock:
            remaining = self._transport_blocked_until - time.monotonic()
            error = self._transport_error
        if remaining <= 0:
            return None
        return ModelTransportUnavailable(
            "local socket transport circuit is open; "
            f"resume only after {max(1, int(remaining + 0.999))} seconds: {error}",
            retry_after=remaining,
        )

    def _open_transport_circuit(
        self, exc: BaseException
    ) -> ModelTransportUnavailable:
        # Give Windows and any endpoint security filter time to release or
        # re-admit connections.  All concurrent workers observe this one
        # circuit instead of independently retrying the same local failure.
        cooldown_seconds = 30.0
        with self._transport_lock:
            self._transport_blocked_until = max(
                self._transport_blocked_until,
                time.monotonic() + cooldown_seconds,
            )
            self._transport_error = str(exc)
        return ModelTransportUnavailable(
            "local socket access was denied (WinError 10013); "
            "transport circuit opened to stop retry amplification",
            retry_after=cooldown_seconds,
        )

    def _session(self) -> requests.Session:
        session = getattr(self._thread_sessions, "session", None)
        if session is None:
            session = requests.Session()
            session.mount("https://", self._http_adapter)
            session.mount("http://", self._http_adapter)
            self._thread_sessions.session = session
        return session

    def _emit_model_payload(
        self, event: str, *, request_id: str, endpoint: str, payload: Any
    ) -> bool:
        """Keep complete payloads in a private opt-in stream, never in audit JSONL."""

        if not self.logger or os.getenv(
            "MEMORY_LOG_MODEL_PAYLOADS", "false"
        ).strip().casefold() not in {"true", "1", "yes", "on"}:
            return False
        try:
            self.logger.emit_model_payload(
                event, request_id=request_id, endpoint=endpoint, payload=payload
            )
        except Exception as exc:
            # A diagnostic write failure must not retry a completed paid call.
            self.logger.emit(
                "model_payload_log_failed",
                request_id=request_id,
                endpoint=endpoint,
                error_type=type(exc).__name__,
            )
            return False
        return True

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        scope = self._current_call_scope() or self._standalone_call_scope(
            endpoint, payload
        )
        trace_sink = self._current_trace_sink()
        requested_model = str(payload.get("model") or scope.model or "unspecified")
        role, purpose = self._current_provider_purpose(scope.operation)
        call_id = uuid4().hex
        call_started_at = time.monotonic()
        queue_ms = 0.0
        network_ms: float | None = None

        def observe(
            *,
            status: str,
            sent: bool,
            actual_model: str | None = None,
            response_value: Any = None,
            http_status: int | None = None,
            error: BaseException | None = None,
            late_result_discarded: bool = False,
        ) -> None:
            self._notify_provider_observation(
                trace_sink,
                call_id=call_id,
                scope=scope,
                endpoint=endpoint,
                requested_model=requested_model,
                role=role,
                purpose=purpose,
                status=status,
                sent=sent,
                queue_ms=queue_ms,
                network_ms=network_ms,
                total_started_at=call_started_at,
                actual_model=actual_model,
                response_value=response_value,
                http_status=http_status,
                error=error,
                late_result_discarded=late_result_discarded,
            )

        try:
            self._require_deadline("provider_preflight")
        except ModelDeadlineExceeded as error:
            observe(status="rejected_before_send", sent=False, error=error)
            raise
        if not self.config.api_key:
            error = ModelClientError("SILICONFLOW_API_KEY is not configured")
            observe(status="rejected_before_send", sent=False, error=error)
            raise error
        blocked = self._transport_circuit_error()
        if blocked is not None:
            observe(status="rejected_before_send", sent=False, error=blocked)
            raise blocked
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            observe(status="rejected_before_send", sent=False, error=exc)
            raise
        target = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        request_id = call_id
        if self.logger:
            payload_logged = self._emit_model_payload(
                "llm_request",
                request_id=request_id,
                endpoint=endpoint,
                payload=payload,
            )
            self.logger.emit(
                "llm_request",
                request_id=request_id,
                logical_batch_id=scope.logical_batch_id,
                operation=scope.operation,
                endpoint=endpoint,
                payload=_request_log_summary(payload),
                content_logged=False,
                payload_companion_logged=payload_logged,
            )
        response: Any | None = None
        sent = False
        network_started_at: float | None = None
        semaphore_acquired = False
        provider_slot_acquired = False
        try:
            # Limit only active network requests, not file preparation or
            # database writes.  Queue acquisition is inside the same absolute
            # request budget as provider I/O: a request that can no longer be
            # delivered must never create a late HTTP attempt merely because a
            # worker freed a socket after its deadline.
            queued_at = time.monotonic()
            remaining = self._remaining_deadline_seconds()
            semaphore_acquired = (
                self._request_semaphore.acquire(timeout=max(0.0, remaining))
                if remaining is not None
                else self._request_semaphore.acquire()
            )
            queue_ms = (time.monotonic() - queued_at) * 1000.0
            if not semaphore_acquired:
                error = ModelDeadlineExceeded("provider_queue")
                observe(status="rejected_before_send", sent=False, error=error)
                raise error
            try:
                remaining = self._require_deadline("provider_dispatch")
            except ModelDeadlineExceeded as error:
                observe(status="rejected_before_send", sent=False, error=error)
                raise
            deadline_budget = self._current_deadline_budget()
            provider_slot_acquired = self._provider_scheduler.acquire(
                priority=self._current_provider_priority(role),
                deadline_at=(
                    deadline_budget.deadline_at
                    if deadline_budget is not None
                    else None
                ),
            )
            queue_ms = (time.monotonic() - queued_at) * 1000.0
            if not provider_slot_acquired:
                error = ModelDeadlineExceeded("provider_queue")
                observe(status="rejected_before_send", sent=False, error=error)
                raise error
            try:
                remaining = self._require_deadline("provider_dispatch")
            except ModelDeadlineExceeded as error:
                observe(status="rejected_before_send", sent=False, error=error)
                raise
            # Reserve immediately before the transport boundary.  The receipt
            # records dispatched work conservatively: it is intentionally not
            # released for a timeout, interrupt, or otherwise unknown result.
            # This exception is not a ModelClientError, so no retry/repair or
            # fallback loop can turn an exhausted campaign into another call.
            if self._campaign_budget is not None:
                try:
                    self._campaign_budget.reserve()
                except CampaignBudgetExhausted as error:
                    observe(status="rejected_before_send", sent=False, error=error)
                    if self.logger:
                        self.logger.emit(
                            "campaign_budget_exhausted",
                            campaign_id=self._campaign_budget.campaign_id,
                            endpoint=endpoint,
                        )
                    raise
            self._record_http_attempt(scope, endpoint)
            sent = True
            network_started_at = time.monotonic()

            def dispatch_post() -> Any:
                return self._session().post(
                    target,
                    data=body,
                    headers=headers,
                    # ``requests`` cannot be force-cancelled remotely. This
                    # bounds connect/read waits; the deadline isolation below
                    # separately enforces the caller-visible total budget.
                    timeout=(
                        min(float(self.config.timeout_seconds), remaining)
                        if remaining is not None
                        else self.config.timeout_seconds
                    ),
                )

            if deadline_budget is None:
                response = dispatch_post()
            else:
                isolated_post = _DeadlineIsolatedPost(dispatch_post)
                isolated_post.start()

                def cleanup_late_transport(
                    late_response: Any | None,
                    _late_error: BaseException | None,
                ) -> None:
                    try:
                        if late_response is not None:
                            late_response.close()
                    finally:
                        # The physical HTTP operation, not the foreground
                        # wait, owns these leases after abandonment.
                        self._provider_scheduler.release()
                        self._request_semaphore.release()

                try:
                    finished_in_budget = isolated_post.wait(float(remaining))
                except BaseException:
                    # A caller interrupt is also an abandonment boundary. Do
                    # not release its physical socket lease prematurely while
                    # the isolated worker may still be using it.
                    isolated_post.abandon(cleanup_late_transport)
                    provider_slot_acquired = False
                    semaphore_acquired = False
                    raise
                if finished_in_budget:
                    response = isolated_post.result()
                else:
                    # Transfer both actual transport leases to the worker (or
                    # clean them synchronously if it completed in this race).
                    # Releasing them here would allow new calls to exceed the
                    # configured limit while the old socket still runs.
                    isolated_post.abandon(cleanup_late_transport)
                    provider_slot_acquired = False
                    semaphore_acquired = False
                    network_ms = (time.monotonic() - network_started_at) * 1000.0
                    error = ModelDeadlineExceeded(
                        "provider_response", late_result_discarded=True
                    )
                    self._record_http_result(scope, endpoint, "late_discarded")
                    observe(
                        status="late_discarded",
                        sent=True,
                        error=error,
                        late_result_discarded=True,
                    )
                    raise error
            network_ms = (time.monotonic() - network_started_at) * 1000.0
            if self._remaining_deadline_seconds() is not None and (
                self._remaining_deadline_seconds() <= 0.0
            ):
                error = ModelDeadlineExceeded(
                    "provider_response", late_result_discarded=True
                )
                status_code = self._http_status(response)
                try:
                    response.close()
                finally:
                    self._record_http_result(scope, endpoint, "late_discarded")
                    observe(
                        status="late_discarded",
                        sent=True,
                        http_status=status_code,
                        error=error,
                        late_result_discarded=True,
                    )
                raise error
            try:
                response.raise_for_status()
                value = response.json()
                if self._remaining_deadline_seconds() is not None and (
                    self._remaining_deadline_seconds() <= 0.0
                ):
                    error = ModelDeadlineExceeded(
                        "provider_response_parse", late_result_discarded=True
                    )
                    self._record_http_result(scope, endpoint, "late_discarded")
                    observe(
                        status="late_discarded",
                        sent=True,
                        response_value=value,
                        http_status=self._http_status(response),
                        error=error,
                        late_result_discarded=True,
                    )
                    raise error
                self._record_http_result(scope, endpoint, "success")
            finally:
                response.close()
        except ModelDeadlineExceeded:
            raise
        except requests.HTTPError as exc:
            response = exc.response
            status_code = self._http_status(response)
            try:
                body_length = len(response.content) if response is not None else 0
            except (AttributeError, TypeError):
                body_length = 0
            details = (
                f"HTTP {status_code if status_code is not None else '?'}: "
                f"provider response body withheld ({body_length} chars)"
            )
            self._record_http_result(
                scope,
                endpoint,
                f"http_{status_code}" if status_code is not None else "http_error",
            )
            observe(
                status="failed",
                sent=True,
                http_status=status_code,
                error=exc,
            )
            retry_after = None
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except (AttributeError, TypeError, ValueError):
                pass
            if self.logger:
                self.logger.emit(
                    "llm_error",
                    request_id=request_id,
                    logical_batch_id=scope.logical_batch_id,
                    endpoint=endpoint,
                    error=(
                        details
                    ),
                )
            raise ModelClientError(
                details,
                status_code=status_code,
                retry_after=retry_after,
                retryable=(
                    response is not None
                    and (
                        response.status_code in {408, 409, 425, 429}
                        or 500 <= response.status_code < 600
                    )
                ),
            ) from exc
        except (
            requests.RequestException,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            if sent and network_ms is None and network_started_at is not None:
                network_ms = (time.monotonic() - network_started_at) * 1000.0
            outcome = (
                "timeout"
                if isinstance(exc, (requests.Timeout, TimeoutError))
                else "transport_error"
            )
            if sent:
                self._record_http_result(scope, endpoint, outcome)
            observe(
                status=(
                    "timeout"
                    if isinstance(exc, (requests.Timeout, TimeoutError))
                    else "failed"
                ),
                sent=sent,
                http_status=self._http_status(response),
                error=exc,
            )
            if self._is_socket_resource_error(exc):
                transport_error = self._open_transport_circuit(exc)
                if self.logger:
                    self.logger.emit(
                        "transport_circuit_opened",
                        request_id=request_id,
                        logical_batch_id=scope.logical_batch_id,
                        endpoint=endpoint,
                        error=str(exc),
                        cooldown_seconds=transport_error.retry_after,
                    )
                raise transport_error from exc
            if self.logger:
                self.logger.emit(
                    "llm_error",
                    request_id=request_id,
                    logical_batch_id=scope.logical_batch_id,
                    endpoint=endpoint,
                    error=str(exc),
                )
            raise ModelClientError(str(exc)) from exc
        finally:
            if provider_slot_acquired:
                self._provider_scheduler.release()
            if semaphore_acquired:
                self._request_semaphore.release()
        observe(
            status="succeeded",
            sent=True,
            actual_model=self._response_model(value),
            response_value=value,
            http_status=self._http_status(response),
        )
        if self.logger:
            payload_logged = self._emit_model_payload(
                "llm_response",
                request_id=request_id,
                endpoint=endpoint,
                payload=value,
            )
            self.logger.emit(
                "llm_response",
                request_id=request_id,
                logical_batch_id=scope.logical_batch_id,
                endpoint=endpoint,
                payload=_response_log_summary(value),
                content_logged=False,
                payload_companion_logged=payload_logged,
            )
        return value

    @staticmethod
    def _retry_delay(exc: ModelClientError, attempt: int) -> float:
        """Spread concurrent retries, especially after provider rate limiting."""
        if exc.retry_after is not None:
            return min(max(exc.retry_after, 0.0), 120.0)
        if exc.status_code == 429:
            return min(8.0 * (2**attempt) + random.uniform(0.0, 4.0), 60.0)
        if exc.status_code is not None and 500 <= exc.status_code < 600:
            return min(2.0**attempt + random.uniform(0.0, 1.0), 15.0)
        return min(2.0**attempt, 4.0)

    def _sleep_for_retry(self, delay: float) -> None:
        """Wait only when the shared request budget can still afford it."""

        wait = max(0.0, float(delay))
        remaining = self._remaining_deadline_seconds()
        if remaining is not None:
            # Do not consume the last instant and then begin an inevitably
            # late fallback/HTTP call.  The tiny guard covers scheduler jitter
            # without pretending a synchronous sleep is precisely cancellable.
            if remaining <= 0.0 or wait >= max(0.0, remaining - 0.001):
                raise ModelDeadlineExceeded("provider_retry")
        if wait:
            time.sleep(wait)
        self._require_deadline("provider_retry")

    def _chat_once(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.1,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.reasoning_max_tokens is not None:
            payload["max_tokens"] = self.config.reasoning_max_tokens
        if self.config.reasoning_enable_thinking is not None:
            payload["enable_thinking"] = self.config.reasoning_enable_thinking
        response = self._post("chat/completions", payload)
        self._require_deadline("provider_chat_parse")
        try:
            choice = response["choices"][0]
            content = str(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(
                "unexpected chat response structure", retryable=False
            ) from exc
        if str(choice.get("finish_reason", "")).casefold() == "length":
            raise ModelClientError(
                "chat response was truncated because the output-token limit was reached",
                retryable=False,
            )
        self._require_deadline("provider_chat_delivery")
        return content

    def chat_text(
        self,
        system: str,
        user: str,
        model: str | None = None,
        allow_fallback: bool = True,
        temperature: float = 0.1,
        max_retries: int | None = None,
    ) -> str:
        selected = model or self.config.reasoning_model
        if self._current_call_scope() is None:
            with self._logical_batch("chat_text", selected):
                return self.chat_text(
                    system,
                    user,
                    model=selected,
                    allow_fallback=allow_fallback,
                    temperature=temperature,
                    max_retries=max_retries,
                )
        errors: list[str] = []
        models = [selected]
        if allow_fallback and self.config.fallback_model != selected:
            models.append(self.config.fallback_model)
        retry_limit = (
            self.config.max_retries
            if max_retries is None
            else max(0, int(max_retries))
        )
        for model_name in models:
            for attempt in range(retry_limit + 1):
                try:
                    return self._chat_once(system, user, model_name, temperature)
                except ModelTransportUnavailable:
                    # Both configured models use the same transport.  Trying
                    # a fallback model would only open another socket while
                    # the local circuit is deliberately cooling down.
                    raise
                except ModelDeadlineExceeded:
                    raise
                except ModelClientError as exc:
                    errors.append(f"{model_name} attempt {attempt + 1}: {exc}")
                    delay = self._retry_delay(exc, attempt)
                    will_retry = (
                        exc.retryable and attempt < retry_limit
                    )
                    if self.logger:
                        self.logger.emit(
                            "retry" if will_retry else "model_attempt_failed",
                            model=model_name,
                            attempt=attempt + 1,
                            error=str(exc),
                            retry_after_seconds=delay if will_retry else 0.0,
                            retryable=exc.retryable,
                        )
                    if not exc.retryable:
                        break
                    if will_retry:
                        self._sleep_for_retry(delay)
                        self._record_retry()
            if model_name != models[-1]:
                self._record_fallback(model_name, models[-1])
                if self.logger:
                    self.logger.emit(
                        "fallback", from_model=model_name, to_model=models[-1]
                    )
        raise ModelClientError("; ".join(errors))

    def chat_json(
        self,
        system: str,
        user: str,
        model: str | None = None,
        allow_fallback: bool = True,
        max_retries: int | None = None,
    ) -> Any:
        selected = model or self.config.reasoning_model
        if self._current_call_scope() is None:
            with self._logical_batch("chat_json", selected):
                return self.chat_json(
                    system,
                    user,
                    model=selected,
                    allow_fallback=allow_fallback,
                    max_retries=max_retries,
                )
        models = [selected]
        if allow_fallback and self.config.fallback_model != selected:
            models.append(self.config.fallback_model)
        errors: list[str] = []
        for index, model_name in enumerate(models):
            try:
                text = self.chat_text(
                    system,
                    user,
                    model=model_name,
                    allow_fallback=False,
                    max_retries=max_retries,
                )
            except ModelTransportUnavailable:
                raise
            except ModelDeadlineExceeded:
                raise
            except ModelClientError as exc:
                errors.append(f"{model_name}: {exc}")
            else:
                try:
                    self._require_deadline("provider_json_parse")
                    parsed = extract_json_payload(text)
                    self._require_deadline("provider_json_delivery")
                    return parsed
                except ValueError as first_error:
                    # Ask the original natural-language task again.  The
                    # malformed machine output is local diagnostic data and
                    # must not become a second model prompt.
                    retry_prompt = (
                        str(user)
                        + "\n上一轮回复无法被程序解析。请重新完整回答，"
                          "不要复制或修补上一轮的输出。"
                    )
                    try:
                        repaired = self.chat_text(
                            system,
                            retry_prompt,
                            model=model_name,
                            allow_fallback=False,
                            temperature=0.0,
                            max_retries=max_retries,
                        )
                        self._require_deadline("provider_json_repair_parse")
                        parsed = extract_json_payload(repaired)
                        self._require_deadline("provider_json_repair_delivery")
                        return parsed
                    except ModelTransportUnavailable:
                        raise
                    except ModelDeadlineExceeded:
                        raise
                    except (ModelClientError, ValueError):
                        errors.append(
                            f"{model_name}: invalid JSON after repair: {first_error}"
                        )
            if index + 1 < len(models):
                self._record_fallback(model_name, models[index + 1])
                if self.logger:
                    self.logger.emit(
                        "fallback",
                        from_model=model_name,
                        to_model=models[index + 1],
                        reason="invalid_or_unavailable_json",
                    )
        raise ModelClientError("; ".join(errors) or "model did not return valid JSON")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.embedding_dimension), dtype=np.float32)
        if self._current_call_scope() is None:
            with self._logical_batch("embedding", self.config.embedding_model):
                return self.embed(texts)
        errors: list[str] = []
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._post(
                    "embeddings",
                    {"model": self.config.embedding_model, "input": texts},
                )
                try:
                    rows = sorted(
                        response["data"], key=lambda item: int(item.get("index", 0))
                    )
                    matrix = np.asarray(
                        [item["embedding"] for item in rows], dtype=np.float32
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ModelClientError(
                        "unexpected embedding response structure"
                    ) from exc
                expected = (len(texts), self.config.embedding_dimension)
                if matrix.shape != expected:
                    raise ModelClientError(
                        f"embedding shape {matrix.shape} != {expected}"
                    )
                self._require_deadline("provider_embedding_parse")
                return matrix
            except ModelTransportUnavailable:
                raise
            except ModelDeadlineExceeded:
                raise
            except ModelClientError as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                delay = self._retry_delay(exc, attempt)
                will_retry = (
                    exc.retryable and attempt < self.config.max_retries
                )
                if self.logger:
                    self.logger.emit(
                        "retry" if will_retry else "model_attempt_failed",
                        operation="embedding",
                        model=self.config.embedding_model,
                        attempt=attempt + 1,
                        error=str(exc),
                        retry_after_seconds=delay if will_retry else 0.0,
                        retryable=exc.retryable,
                    )
                if not exc.retryable:
                    break
                if will_retry:
                    self._sleep_for_retry(delay)
                    self._record_retry()
        raise ModelClientError("; ".join(errors))

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[dict[str, float | int]]:
        """Rank documents with SiliconFlow's dedicated rerank endpoint."""
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("rerank query must not be empty")
        if not documents:
            return []
        if self._current_call_scope() is None:
            with self._logical_batch("rerank", self.config.reranker_model):
                return self.rerank(normalized_query, documents, top_n=top_n)
        limit = min(
            len(documents),
            max(1, int(top_n if top_n is not None else len(documents))),
        )
        errors: list[str] = []
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._post(
                    "rerank",
                    {
                        "model": self.config.reranker_model,
                        "query": normalized_query,
                        "documents": [str(value) for value in documents],
                        "top_n": limit,
                        "return_documents": False,
                    },
                )
                raw_results = response.get("results")
                if not isinstance(raw_results, list):
                    raise ModelClientError(
                        "unexpected rerank response structure"
                    )
                ranked: list[dict[str, float | int]] = []
                seen: set[int] = set()
                for raw in raw_results:
                    try:
                        index = int(raw["index"])
                        score = float(raw["relevance_score"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ModelClientError(
                            "unexpected rerank result structure"
                        ) from exc
                    if not 0 <= index < len(documents) or index in seen:
                        raise ModelClientError(
                            "rerank result contains invalid document index"
                        )
                    seen.add(index)
                    ranked.append(
                        {"index": index, "relevance_score": score}
                    )
                self._require_deadline("provider_rerank_parse")
                return ranked
            except ModelTransportUnavailable:
                raise
            except ModelDeadlineExceeded:
                raise
            except ModelClientError as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                delay = self._retry_delay(exc, attempt)
                will_retry = (
                    exc.retryable and attempt < self.config.max_retries
                )
                if self.logger:
                    self.logger.emit(
                        "retry" if will_retry else "model_attempt_failed",
                        operation="rerank",
                        model=self.config.reranker_model,
                        attempt=attempt + 1,
                        error=str(exc),
                        retry_after_seconds=delay if will_retry else 0.0,
                        retryable=exc.retryable,
                    )
                if not exc.retryable:
                    break
                if will_retry:
                    self._sleep_for_retry(delay)
                    self._record_retry()
        raise ModelClientError("; ".join(errors))
