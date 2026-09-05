from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any
from threading import BoundedSemaphore, Lock, local
from uuid import uuid4

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool

from memory_demo.config import ModelConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import JSON_REPAIR_SYSTEM, json_repair_prompt


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

    def __init__(self, config: ModelConfig, logger: JsonlEventLogger | None = None):
        self.config = config
        self.logger = logger
        request_limit = max(1, int(config.max_concurrent_requests))
        self._request_semaphore = BoundedSemaphore(request_limit)
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

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise ModelClientError("SILICONFLOW_API_KEY is not configured")
        blocked = self._transport_circuit_error()
        if blocked is not None:
            raise blocked
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        target = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        request_id = uuid4().hex
        if self.logger:
            self.logger.emit(
                "llm_request",
                request_id=request_id,
                endpoint=endpoint,
                payload=(
                    payload
                    if os.getenv("MEMORY_LOG_MODEL_PAYLOADS", "false").casefold()
                    == "true"
                    else _request_log_summary(payload)
                ),
                content_logged=(
                    os.getenv("MEMORY_LOG_MODEL_PAYLOADS", "false").casefold()
                    == "true"
                ),
            )
        try:
            # Limit only active network requests, not file preparation or
            # database writes.  The shared adapter reuses those bounded
            # connections for retries and subsequent file workers.
            with self._request_semaphore:
                response = self._session().post(
                    target,
                    data=body,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                try:
                    response.raise_for_status()
                    value = response.json()
                finally:
                    response.close()
        except requests.HTTPError as exc:
            response = exc.response
            details = response.text if response is not None else str(exc)
            retry_after = None
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except (AttributeError, TypeError, ValueError):
                pass
            if self.logger:
                self.logger.emit(
                    "llm_error",
                    request_id=request_id,
                    endpoint=endpoint,
                    error=(
                        f"HTTP {response.status_code if response is not None else '?'}: "
                        f"{details}"
                    ),
                )
            raise ModelClientError(
                (
                    f"HTTP {response.status_code if response is not None else '?'}: "
                    f"{details}"
                ),
                status_code=response.status_code if response is not None else None,
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
            if self._is_socket_resource_error(exc):
                transport_error = self._open_transport_circuit(exc)
                if self.logger:
                    self.logger.emit(
                        "transport_circuit_opened",
                        request_id=request_id,
                        endpoint=endpoint,
                        error=str(exc),
                        cooldown_seconds=transport_error.retry_after,
                    )
                raise transport_error from exc
            if self.logger:
                self.logger.emit(
                    "llm_error",
                    request_id=request_id,
                    endpoint=endpoint,
                    error=str(exc),
                )
            raise ModelClientError(str(exc)) from exc
        if self.logger:
            self.logger.emit(
                "llm_response",
                request_id=request_id,
                endpoint=endpoint,
                payload=(
                    value
                    if os.getenv("MEMORY_LOG_MODEL_PAYLOADS", "false").casefold()
                    == "true"
                    else _response_log_summary(value)
                ),
                content_logged=(
                    os.getenv("MEMORY_LOG_MODEL_PAYLOADS", "false").casefold()
                    == "true"
                ),
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
                        time.sleep(delay)
            if self.logger and model_name != models[-1]:
                self.logger.emit("fallback", from_model=model_name, to_model=models[-1])
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
            except ModelClientError as exc:
                errors.append(f"{model_name}: {exc}")
            else:
                try:
                    return extract_json_payload(text)
                except ValueError as first_error:
                    repair_prompt = json_repair_prompt(text)
                    try:
                        repaired = self.chat_text(
                            JSON_REPAIR_SYSTEM,
                            repair_prompt,
                            model=model_name,
                            allow_fallback=False,
                            temperature=0.0,
                            max_retries=max_retries,
                        )
                        return extract_json_payload(repaired)
                    except ModelTransportUnavailable:
                        raise
                    except (ModelClientError, ValueError):
                        errors.append(
                            f"{model_name}: invalid JSON after repair: {first_error}"
                        )
            if index + 1 < len(models) and self.logger:
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
                return matrix
            except ModelTransportUnavailable:
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
                    time.sleep(delay)
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
                return ranked
            except ModelTransportUnavailable:
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
                    time.sleep(delay)
        raise ModelClientError("; ".join(errors))
