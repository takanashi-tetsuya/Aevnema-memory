from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

import numpy as np
import requests

from memory_demo.config import ModelConfig
from memory_demo.llm.client import (
    ModelClient,
    ModelClientError,
    ProviderCallObservation,
    TracePersistenceError,
)


class _Response:
    """Small requests.Response stand-in for the real ModelClient._post hook."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict:
        return self._payload

    def close(self) -> None:
        self.closed = True


class _TraceSink:
    """Thread-safe request-local observer used without any trace writer."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.starts: list[tuple[str, str, str]] = []
        self.observations: list[ProviderCallObservation] = []
        self.fallbacks: list[tuple[str, str, str]] = []

    def logical_batch_started(
        self,
        *,
        logical_batch_id: str,
        operation: str,
        requested_model: str,
    ) -> None:
        with self._lock:
            self.starts.append((logical_batch_id, operation, requested_model))

    def provider_call_finished(self, observation: ProviderCallObservation) -> None:
        with self._lock:
            self.observations.append(observation)

    def fallback_used(
        self,
        *,
        logical_batch_id: str,
        from_model: str,
        to_model: str,
    ) -> None:
        with self._lock:
            self.fallbacks.append((logical_batch_id, from_model, to_model))


class _FailingObservationSink(_TraceSink):
    """A durable observer failure must not trigger repair/retry/fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.faults: list[str] = []

    def provider_call_finished(self, observation: ProviderCallObservation) -> None:
        super().provider_call_finished(observation)
        raise ValueError("private trace writer failure")

    def trace_persistence_failed(self, exc: BaseException) -> None:
        self.faults.append(type(exc).__name__)


class ProviderTraceObservationTests(unittest.TestCase):
    def _client(self, **overrides: object) -> ModelClient:
        config = ModelConfig(
            api_key="trace-test-key",
            embedding_model="trace-embedding-model",
            embedding_dimension=2,
            max_retries=0,
            max_concurrent_requests=2,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return ModelClient(config)

    def test_concurrent_trace_sinks_are_request_local_at_real_post_hook(self) -> None:
        """Two caller contexts sharing one client must not mix observations."""

        client = self._client()
        entered_post = Barrier(2)

        def post(*_args: object, **_kwargs: object) -> _Response:
            # Both calls arrive at the actual HTTP hook concurrently.  This
            # catches accidental use of a process-global/current-last sink.
            entered_post.wait(timeout=3)
            return _Response(
                payload={
                    "model": "served-trace-embedding-model",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 0},
                    "data": [{"index": 0, "embedding": [0.25, 0.75]}],
                }
            )

        first_sink = _TraceSink()
        second_sink = _TraceSink()

        def embed_in_request(sink: _TraceSink, marker: str) -> np.ndarray:
            with client.trace_request(sink), client.provider_purpose(
                "embedding", f"test:parallel:{marker}"
            ):
                return client.embed([f"private source text {marker}"])

        with patch(
            "memory_demo.llm.client.requests.Session.post", side_effect=post
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(embed_in_request, first_sink, "first")
                second = executor.submit(embed_in_request, second_sink, "second")
                self.assertTrue(np.allclose(first.result(timeout=5), [[0.25, 0.75]]))
                self.assertTrue(np.allclose(second.result(timeout=5), [[0.25, 0.75]]))

        for sink, marker in ((first_sink, "first"), (second_sink, "second")):
            self.assertEqual(1, len(sink.starts))
            self.assertEqual(1, len(sink.observations))
            self.assertEqual([], sink.fallbacks)
            logical_batch_id, operation, requested_model = sink.starts[0]
            observation = sink.observations[0]
            self.assertEqual(logical_batch_id, observation.logical_batch_id)
            self.assertEqual("embedding", operation)
            self.assertEqual("embedding", observation.operation)
            self.assertEqual("trace-embedding-model", requested_model)
            self.assertEqual("trace-embedding-model", observation.requested_model)
            self.assertEqual("served-trace-embedding-model", observation.actual_model)
            self.assertEqual("embeddings", observation.endpoint)
            self.assertEqual("embedding", observation.role)
            self.assertEqual(f"test:parallel:{marker}", observation.purpose)
            self.assertEqual("succeeded", observation.status)
            self.assertTrue(observation.sent)
            self.assertGreaterEqual(observation.queue_ms, 0.0)
            self.assertIsNotNone(observation.network_ms)
            self.assertGreaterEqual(observation.network_ms or 0.0, 0.0)
            self.assertGreaterEqual(observation.total_ms, 0.0)
            self.assertEqual(7, observation.prompt_tokens)
            self.assertEqual(0, observation.completion_tokens)
            self.assertEqual(200, observation.http_status)
            self.assertIsNone(observation.error_class)
            self.assertFalse(observation.late_result_discarded)
            # A provider trace must be metadata-only even though `_post`
            # received source text above.
            self.assertNotIn("private source text", repr(observation))

        self.assertNotEqual(first_sink.starts[0][0], second_sink.starts[0][0])

    def test_http_failure_is_observed_after_a_sent_attempt(self) -> None:
        client = self._client()
        sink = _TraceSink()
        failure = _Response(status_code=503, content=b"withheld upstream body")

        with patch(
            "memory_demo.llm.client.requests.Session.post", return_value=failure
        ):
            with client.trace_request(sink), client.provider_purpose(
                "embedding", "test:http-failure"
            ):
                with self.assertRaises(ModelClientError):
                    client.embed(["private source text failure"])

        self.assertTrue(failure.closed)
        self.assertEqual(1, len(sink.starts))
        self.assertEqual(1, len(sink.observations))
        observation = sink.observations[0]
        self.assertEqual(sink.starts[0][0], observation.logical_batch_id)
        self.assertEqual("embedding", observation.operation)
        self.assertEqual("embeddings", observation.endpoint)
        self.assertEqual("embedding", observation.role)
        self.assertEqual("test:http-failure", observation.purpose)
        self.assertEqual("failed", observation.status)
        self.assertTrue(observation.sent)
        self.assertEqual(503, observation.http_status)
        self.assertIsNone(observation.actual_model)
        self.assertIsNotNone(observation.network_ms)
        self.assertIsNotNone(observation.error_class)
        self.assertNotIn("private source text failure", repr(observation))

    def test_transport_failure_preserves_sent_attempt_and_network_timing(self) -> None:
        """A socket/timeout error happens after dispatch, not before it."""

        client = self._client()
        sink = _TraceSink()
        diagnostic = "private transport diagnostic must not enter the trace"

        with patch(
            "memory_demo.llm.client.requests.Session.post",
            side_effect=requests.ConnectionError(diagnostic),
        ):
            with client.trace_request(sink), client.provider_purpose(
                "embedding", "test:transport-failure"
            ):
                with self.assertRaises(ModelClientError):
                    client.embed(["private source text transport failure"])

        self.assertEqual(1, len(sink.starts))
        self.assertEqual(1, len(sink.observations))
        observation = sink.observations[0]
        self.assertEqual("failed", observation.status)
        self.assertTrue(observation.sent)
        self.assertIsNotNone(observation.network_ms)
        self.assertGreaterEqual(observation.network_ms or 0.0, 0.0)
        self.assertIsNone(observation.http_status)
        self.assertEqual("ConnectionError", observation.error_class)
        self.assertNotIn(diagnostic, repr(observation))
        self.assertNotIn("private source text transport failure", repr(observation))

    def test_pre_send_configuration_rejection_is_observed_without_http(self) -> None:
        client = self._client(api_key="")
        sink = _TraceSink()

        with patch("memory_demo.llm.client.requests.Session.post") as post:
            with client.trace_request(sink), client.provider_purpose(
                "embedding", "test:pre-send-rejection"
            ):
                with self.assertRaises(ModelClientError):
                    client.embed(["private source text rejected"])

        post.assert_not_called()
        self.assertEqual(1, len(sink.starts))
        self.assertEqual(1, len(sink.observations))
        observation = sink.observations[0]
        self.assertEqual(sink.starts[0][0], observation.logical_batch_id)
        self.assertEqual("embedding", observation.operation)
        self.assertEqual("embeddings", observation.endpoint)
        self.assertEqual("embedding", observation.role)
        self.assertEqual("test:pre-send-rejection", observation.purpose)
        self.assertEqual("rejected_before_send", observation.status)
        self.assertFalse(observation.sent)
        self.assertIsNone(observation.network_ms)
        self.assertIsNone(observation.http_status)
        self.assertIsNotNone(observation.error_class)
        self.assertNotIn("private source text rejected", repr(observation))

    def test_trace_persistence_failure_cannot_trigger_json_repair_or_fallback(self) -> None:
        """A receipt failure after send must halt before another provider call."""

        client = self._client(max_retries=2)
        sink = _FailingObservationSink()
        response = _Response(
            payload={
                "model": "served-primary-model",
                # If normal handling reached parsing, this invalid JSON would
                # request a repair call.  The failing receipt must stop first.
                "choices": [{"message": {"content": "not valid json"}}],
            }
        )

        with patch(
            "memory_demo.llm.client.requests.Session.post", return_value=response
        ) as post:
            with client.trace_request(sink), client.provider_purpose(
                "planner", "query_intent"
            ):
                with self.assertRaises(TracePersistenceError) as raised:
                    client.chat_json("private system", "private question")

        self.assertEqual("provider trace persistence failed", str(raised.exception))
        post.assert_called_once()
        self.assertTrue(response.closed)
        self.assertEqual(1, len(sink.starts))
        self.assertEqual(1, len(sink.observations))
        self.assertEqual([], sink.fallbacks)
        self.assertEqual(["ValueError"], sink.faults)
        self.assertNotIn("private trace writer failure", str(raised.exception))
        self.assertNotIn("private question", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
