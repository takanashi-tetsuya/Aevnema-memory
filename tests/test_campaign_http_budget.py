from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from memory_demo.config import ModelConfig
from memory_demo.llm.client import (
    CampaignBudgetExhausted,
    CampaignHttpBudget,
    ModelClient,
    ProviderCallAccounting,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status_code = 200
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.content = b""
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload

    def close(self) -> None:
        self.closed = True


def _config() -> ModelConfig:
    return ModelConfig(
        api_key="local-test-key",
        base_url="https://api.example.test/v1",
        reasoning_model="primary",
        fallback_model="fallback",
        embedding_model="embed",
        embedding_dimension=2,
        max_retries=0,
        max_concurrent_requests=2,
    )


class CampaignHttpBudgetTests(unittest.TestCase):
    def tearDown(self) -> None:
        CampaignHttpBudget._clear_process_cache_for_test()

    def test_n_plus_one_is_rejected_before_transport_and_not_accounted_as_http(self) -> None:
        with TemporaryDirectory() as directory:
            budget = CampaignHttpBudget.open(
                campaign_id="budget-n-plus-one",
                max_http_attempts=1,
                receipt_path=Path(directory) / "campaign.json",
            )
            accounting = ProviderCallAccounting()
            client = ModelClient(_config(), accounting=accounting, campaign_budget=budget)
            response = _Response({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
            with patch("memory_demo.llm.client.requests.Session.post", return_value=response) as post:
                client._post("embeddings", {"model": "embed", "input": ["one"]})
                with self.assertRaises(CampaignBudgetExhausted):
                    client._post("embeddings", {"model": "embed", "input": ["two"]})

            self.assertEqual(1, post.call_count)
            self.assertEqual(1, accounting.snapshot()["http_attempts"])
            self.assertEqual(1, budget.snapshot()["reserved_http_attempts"])
            self.assertEqual(0, budget.snapshot()["remaining_http_attempts"])

    def test_two_clients_racing_for_last_slot_dispatch_exactly_once(self) -> None:
        with TemporaryDirectory() as directory:
            budget = CampaignHttpBudget.open(
                campaign_id="budget-race",
                max_http_attempts=1,
                receipt_path=Path(directory) / "campaign.json",
            )
            barrier = Barrier(2)

            def run_one(label: str) -> str:
                client = ModelClient(_config(), campaign_budget=budget)
                barrier.wait()
                try:
                    client._post("embeddings", {"model": "embed", "input": [label]})
                    return "sent"
                except CampaignBudgetExhausted:
                    return "blocked"

            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=lambda *_args, **_kwargs: _Response({"data": []}),
            ) as post, ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(run_one, ["a", "b"]))

            self.assertEqual(["blocked", "sent"], sorted(outcomes))
            self.assertEqual(1, post.call_count)
            self.assertEqual(1, budget.snapshot()["reserved_http_attempts"])

    def test_json_repair_and_fallback_each_consume_a_real_dispatch(self) -> None:
        def chat_response(content: str) -> _Response:
            return _Response({"choices": [{"message": {"content": content}}]})

        with TemporaryDirectory() as directory:
            budget = CampaignHttpBudget.open(
                campaign_id="budget-repair-fallback",
                max_http_attempts=3,
                receipt_path=Path(directory) / "campaign.json",
            )
            accounting = ProviderCallAccounting()
            client = ModelClient(_config(), accounting=accounting, campaign_budget=budget)
            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=[
                    chat_response("not-json"),
                    chat_response("still-not-json"),
                    chat_response('{"status":"ok"}'),
                ],
            ) as post:
                self.assertEqual({"status": "ok"}, client.chat_json("system", "user"))

            self.assertEqual(3, post.call_count)
            self.assertEqual(3, budget.snapshot()["reserved_http_attempts"])
            snapshot = accounting.snapshot()
            self.assertEqual(3, snapshot["http_attempts"])
            self.assertEqual(1, snapshot["fallbacks"])

    def test_exhaustion_stops_json_repair_before_a_fallback_can_dispatch(self) -> None:
        def chat_response(content: str) -> _Response:
            return _Response({"choices": [{"message": {"content": content}}]})

        with TemporaryDirectory() as directory:
            budget = CampaignHttpBudget.open(
                campaign_id="budget-stop-repair-fallback",
                max_http_attempts=2,
                receipt_path=Path(directory) / "campaign.json",
            )
            client = ModelClient(_config(), campaign_budget=budget)
            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=[chat_response("not-json"), chat_response("still-not-json")],
            ) as post:
                with self.assertRaises(CampaignBudgetExhausted):
                    client.chat_json("system", "user")

            self.assertEqual(2, post.call_count)
            self.assertEqual(2, budget.snapshot()["reserved_http_attempts"])

    def test_persisted_reservation_survives_a_fresh_dispatcher(self) -> None:
        with TemporaryDirectory() as directory:
            receipt = Path(directory) / "campaign.json"
            original = CampaignHttpBudget.open(
                campaign_id="budget-restart",
                max_http_attempts=2,
                receipt_path=receipt,
            )
            original.reserve()
            CampaignHttpBudget._clear_process_cache_for_test()

            resumed = CampaignHttpBudget.open(
                campaign_id="budget-restart",
                max_http_attempts=2,
                receipt_path=receipt,
            )
            self.assertEqual(1, resumed.snapshot()["reserved_http_attempts"])
            resumed.reserve()
            CampaignHttpBudget._clear_process_cache_for_test()
            final = CampaignHttpBudget.open(
                campaign_id="budget-restart",
                max_http_attempts=2,
                receipt_path=receipt,
            )
            self.assertEqual(2, final.snapshot()["reserved_http_attempts"])
            self.assertEqual(0, final.snapshot()["remaining_http_attempts"])
            with self.assertRaises(CampaignBudgetExhausted):
                final.reserve()
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(2, stored["reserved_http_attempts"])


if __name__ == "__main__":
    unittest.main()
