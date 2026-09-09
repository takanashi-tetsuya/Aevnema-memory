from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Event
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import requests

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.llm.client import (
    ModelClient,
    ModelDeadlineExceeded,
    ProviderCallAccounting,
    ProviderCallObservation,
)
from memory_demo.retrieval.engine import QueryEngine


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.content = b""
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
    def __init__(self) -> None:
        self.observations: list[ProviderCallObservation] = []

    def logical_batch_started(self, **_kwargs) -> None:
        return None

    def provider_call_finished(self, observation: ProviderCallObservation) -> None:
        self.observations.append(observation)

    def fallback_used(self, **_kwargs) -> None:
        return None


class _DeadlineAwareLocalModel:
    """A non-network model that proves QueryEngine binds the same deadline."""

    def __init__(self) -> None:
        self.scopes: list[float] = []
        self.closed = 0

    @contextmanager
    def deadline_budget(self, deadline_at: float):
        self.scopes.append(float(deadline_at))
        try:
            yield None
        finally:
            self.closed += 1


class ProviderDeadlineBudgetT18Tests(unittest.TestCase):
    @staticmethod
    def _client(
        *,
        retries: int = 0,
        concurrency: int = 1,
        base_url: str = "https://t18-local.invalid/v1",
    ) -> ModelClient:
        return ModelClient(
            ModelConfig(
                api_key="local-deadline-test-key",
                base_url=base_url,
                embedding_dimension=2,
                max_retries=retries,
                max_concurrent_requests=concurrency,
            ),
            # Retry assertions are local to each deadline scenario.  The
            # production default is intentionally process-wide, so inject an
            # isolated ledger instead of letting unrelated tests' retries
            # leak into this fixture's observation.
            accounting=ProviderCallAccounting(),
        )

    def test_expired_or_queue_exhausted_request_never_sends_http(self) -> None:
        client = self._client(concurrency=1)
        sink = _TraceSink()
        self.assertTrue(client._request_semaphore.acquire(timeout=0.1))
        try:
            with patch("memory_demo.llm.client.requests.Session.post") as post:
                with client.trace_request(sink), client.deadline_budget(
                    time.monotonic() + 0.02
                ):
                    with self.assertRaisesRegex(
                        ModelDeadlineExceeded, "provider deadline exceeded before provider_queue"
                    ):
                        client.embed(["queue-bound local fixture"])
        finally:
            client._request_semaphore.release()

        post.assert_not_called()
        self.assertEqual(1, len(sink.observations))
        observation = sink.observations[0]
        self.assertEqual("rejected_before_send", observation.status)
        self.assertFalse(observation.sent)
        self.assertEqual("ModelDeadlineExceeded", observation.error_class)

    def test_late_http_response_is_closed_discarded_and_never_retried(self) -> None:
        client = self._client(retries=2)
        sink = _TraceSink()
        response = _Response(
            payload={"data": [{"index": 0, "embedding": [0.25, 0.75]}]}
        )

        def late_post(*_args, **_kwargs):
            time.sleep(0.03)
            return response

        with patch(
            "memory_demo.llm.client.requests.Session.post", side_effect=late_post
        ) as post:
            with client.trace_request(sink), client.deadline_budget(
                time.monotonic() + 0.01
            ):
                with self.assertRaisesRegex(
                    ModelDeadlineExceeded,
                    "provider deadline exceeded before provider_response",
                ) as raised:
                    client.embed(["late local fixture"])
            # Deadline isolation returns promptly, but deliberately retains
            # the physical provider lease until the slow worker exits.
            self.assertEqual(1, client.provider_limit_snapshot()["active"])
            closes_at = time.monotonic() + 1.0
            while not response.closed and time.monotonic() < closes_at:
                time.sleep(0.002)
            self.assertTrue(response.closed)
            self.assertEqual(0, client.provider_limit_snapshot()["active"])
            post.assert_called_once()

        self.assertTrue(raised.exception.late_result_discarded)
        self.assertEqual(1, len(sink.observations))
        observation = sink.observations[0]
        self.assertEqual("late_discarded", observation.status)
        self.assertTrue(observation.sent)
        self.assertTrue(observation.late_result_discarded)
        self.assertEqual("ModelDeadlineExceeded", observation.error_class)
        self.assertEqual(0, client.provider_call_snapshot()["retries_scheduled"])

    def test_caller_interrupt_keeps_physical_leases_until_worker_cleanup(self) -> None:
        """An interrupted waiter must not reopen a socket slot prematurely."""

        client = self._client()
        response = _Response(
            payload={"data": [{"index": 0, "embedding": [0.25, 0.75]}]}
        )
        worker_started = Event()
        release_worker = Event()

        def delayed_post(*_args, **_kwargs):
            worker_started.set()
            if not release_worker.wait(timeout=1.0):
                raise AssertionError("test did not release delayed provider worker")
            return response

        with patch(
            "memory_demo.llm.client.requests.Session.post", side_effect=delayed_post
        ) as post:
            with patch(
                "memory_demo.llm.client._DeadlineIsolatedPost.wait",
                side_effect=KeyboardInterrupt,
            ):
                with client.deadline_budget(time.monotonic() + 0.5):
                    with self.assertRaises(KeyboardInterrupt):
                        client.embed(["interrupt local fixture"])

            self.assertTrue(worker_started.wait(timeout=1.0))
            self.assertEqual(1, client.provider_limit_snapshot()["active"])
            release_worker.set()
            closes_at = time.monotonic() + 1.0
            while not response.closed and time.monotonic() < closes_at:
                time.sleep(0.002)
            self.assertTrue(response.closed)
            self.assertEqual(0, client.provider_limit_snapshot()["active"])
            post.assert_called_once()

    def test_retry_delay_that_exhausts_budget_stops_before_second_attempt(self) -> None:
        client = self._client(retries=2)
        response = _Response(status_code=429, headers={"Retry-After": "1"})
        with patch(
            "memory_demo.llm.client.requests.Session.post", return_value=response
        ) as post:
            with client.deadline_budget(time.monotonic() + 0.02):
                with self.assertRaisesRegex(
                    ModelDeadlineExceeded, "provider deadline exceeded before provider_retry"
                ):
                    client.embed(["retry local fixture"])

        self.assertTrue(response.closed)
        post.assert_called_once()
        self.assertEqual(0, client.provider_call_snapshot()["retries_scheduled"])

    def test_interactive_query_gets_next_shared_provider_slot_before_import(self) -> None:
        """Separate clients sharing a provider must not each spend its cap."""

        base_url = f"https://t18-priority-{uuid4().hex}.invalid/v1"
        import_active = self._client(concurrency=1, base_url=base_url)
        import_waiter = self._client(concurrency=1, base_url=base_url)
        query_client = self._client(concurrency=1, base_url=base_url)
        first_entered = Event()
        release_first = Event()
        dispatch_order: list[str] = []

        def post(*_args, **kwargs):
            payload = json.loads(kwargs["data"])
            label = str(payload["input"][0])
            dispatch_order.append(label)
            if label == "import-active":
                first_entered.set()
                if not release_first.wait(timeout=3):
                    raise AssertionError("test did not release first provider call")
            return _Response(
                payload={"data": [{"index": 0, "embedding": [0.5, 0.5]}]}
            )

        def background_embed(client: ModelClient, label: str) -> np.ndarray:
            with client.provider_workload("background"):
                return client.embed([label])

        def wait_for_waiters(key: str) -> None:
            ends_at = time.monotonic() + 1.0
            while time.monotonic() < ends_at:
                if import_waiter.provider_limit_snapshot()[key] >= 1:
                    return
                time.sleep(0.002)
            self.fail(f"expected a {key} provider waiter")

        with patch("memory_demo.llm.client.requests.Session.post", side_effect=post):
            with ThreadPoolExecutor(max_workers=3) as executor:
                first = executor.submit(background_embed, import_active, "import-active")
                try:
                    self.assertTrue(first_entered.wait(timeout=1))
                    queued_import = executor.submit(
                        background_embed, import_waiter, "import-queued"
                    )
                    wait_for_waiters("background_waiters")
                    query = executor.submit(query_client.embed, ["interactive-query"])
                    wait_for_waiters("interactive_waiters")
                finally:
                    release_first.set()

                self.assertTrue(np.allclose(first.result(timeout=2), [[0.5, 0.5]]))
                self.assertTrue(
                    np.allclose(query.result(timeout=2), [[0.5, 0.5]])
                )
                self.assertTrue(
                    np.allclose(queued_import.result(timeout=2), [[0.5, 0.5]])
                )

        self.assertEqual(
            ["import-active", "interactive-query", "import-queued"],
            dispatch_order,
        )
        self.assertEqual(
            {
                "capacity": 1,
                "active": 0,
                "interactive_waiters": 0,
                "background_waiters": 0,
            },
            query_client.provider_limit_snapshot(),
        )

    def test_query_engine_binds_one_absolute_deadline_to_capable_model(self) -> None:
        model = _DeadlineAwareLocalModel()
        config = AppConfig(model=ModelConfig(embedding_dimension=2))
        config.retrieval.growth_max_rounds = 0
        engine = QueryEngine(
            config,
            model,
            EmbeddingIndex(2),
            EmbeddingIndex(2),
            object(),
            object(),
            object(),
            object(),
        )
        sentinel = {"ok": True}
        before = time.monotonic()
        with patch.object(engine, "_query_impl", return_value=sentinel):
            result = engine.query(
                "deadline binding fixture",
                generate_answer=False,
                deadline_seconds=0.5,
            )

        self.assertIs(sentinel, result)
        self.assertEqual(1, len(model.scopes))
        self.assertGreater(model.scopes[0], before)
        self.assertEqual(1, model.closed)

    def test_deadline_failure_never_reaches_query_learning_finalizer(self) -> None:
        model = _DeadlineAwareLocalModel()
        config = AppConfig(model=ModelConfig(embedding_dimension=2))
        config.retrieval.growth_max_rounds = 0
        engine = QueryEngine(
            config,
            model,
            EmbeddingIndex(2),
            EmbeddingIndex(2),
            object(),
            object(),
            object(),
            object(),
        )

        def late_query(*_args, **_kwargs):
            engine._v3_learning_capture = object()
            raise ModelDeadlineExceeded("provider_response", late_result_discarded=True)

        with patch.object(engine, "_query_impl", side_effect=late_query), patch.object(
            engine, "_finalize_v3_query_learning"
        ) as finalize:
            with self.assertRaises(ModelDeadlineExceeded):
                engine.query(
                    "late result fixture",
                    generate_answer=False,
                    contextual_learning=True,
                    learning_request_id="t18-no-finalize",
                    deadline_seconds=0.5,
                )

        finalize.assert_not_called()
        self.assertIsNone(engine._v3_learning_capture)

    def test_local_result_that_crosses_deadline_never_enters_finalizer(self) -> None:
        """A successful model stage is not permission for a late write."""

        model = _DeadlineAwareLocalModel()
        config = AppConfig(model=ModelConfig(embedding_dimension=2))
        config.retrieval.growth_max_rounds = 0
        engine = QueryEngine(
            config,
            model,
            EmbeddingIndex(2),
            EmbeddingIndex(2),
            object(),
            object(),
            object(),
            object(),
        )

        def completed_query(*_args, **_kwargs):
            engine._v3_learning_capture = object()
            return {"answer": "local result"}

        with patch.object(
            engine, "_v17_automatic_exact_revisit", return_value=None
        ), patch.object(
            engine, "_t16_automatic_restricted_rewrite_revisit", return_value=None
        ), patch.object(engine, "_try_exact_revisit_v3", return_value=None), patch.object(
            engine, "_query_impl", side_effect=completed_query
        ), patch.object(
            engine, "_finalize_v3_query_learning"
        ) as finalize, patch(
            "memory_demo.retrieval.engine.monotonic", side_effect=[100.0, 100.6]
        ):
            with self.assertRaisesRegex(
                TimeoutError, "before post_query_result"
            ):
                engine.query(
                    "late local result fixture",
                    generate_answer=False,
                    contextual_learning=True,
                    learning_request_id="t18-late-local-result",
                    deadline_seconds=0.5,
                )

        finalize.assert_not_called()
        self.assertIsNone(engine._v3_learning_capture)


if __name__ == "__main__":
    unittest.main()
