from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import requests

from benchmarks.import_corpus import save_ledger
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.event_log import JsonlEventLogger, safe_config_snapshot
from memory_demo.llm.client import (
    ModelClient,
    ModelClientError,
    ProviderCallAccounting,
)
from memory_demo.repositories.extraction import ExtractionRepository


class _Response:
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


class SecurityObservabilityTests(unittest.TestCase):
    def test_event_logger_redacts_nested_secrets_headers_urls_and_source_text(self):
        secret = "v3-test-secret-value"
        source_text = "不应写入日志的原始剧情文本"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path)
            logger.emit(
                "security_test",
                source_text=source_text,
                nested={"credentials": {"api_key": secret}},
                headers={"Authorization": f"Bearer {secret}"},
                endpoint_url=f"https://user:{secret}@api.example.test/v1?token={secret}",
                error=f"upstream Authorization: Bearer {secret}",
            )
            raw = path.read_text(encoding="utf-8")
            record = json.loads(raw)

        self.assertNotIn(secret, raw)
        self.assertNotIn(source_text, raw)
        self.assertEqual({"redacted": True, "char_count": len(source_text)}, record["source_text"])
        self.assertEqual("[REDACTED]", record["nested"]["credentials"])
        self.assertEqual("[REDACTED]", record["headers"])
        self.assertEqual("https://api.example.test/v1", record["endpoint_url"])
        self.assertIn("Bearer [REDACTED]", record["error"])

    def test_config_and_ledger_snapshots_use_an_allow_list(self):
        secret = "v3-ledger-secret"
        config = AppConfig(database_path=Path("private/location/memory.db"))
        config.model.api_key = secret
        config.model.base_url = (
            f"https://user:{secret}@api.example.test/v1?api_key={secret}"
        )

        snapshot = safe_config_snapshot(config)
        rendered = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("private/location", rendered)
        self.assertEqual("memory.db", snapshot["database_filename"])
        self.assertTrue(snapshot["model_api_key_configured"])
        self.assertNotIn("api_key", snapshot["model"])
        self.assertEqual("https://api.example.test/v1", snapshot["model"]["base_url"])

        with TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.json"
            save_ledger(
                ledger_path,
                {"files": {}, "last_process": {"config": {"api_key": secret}}},
            )
            raw = ledger_path.read_text(encoding="utf-8")
        self.assertNotIn(secret, raw)

    def test_extraction_repository_defense_in_depth_redacts_direct_snapshot_input(self):
        secret = "v3-extraction-run-secret"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "memory.db")
            database.initialize()
            repository = ExtractionRepository(database)
            run_id = repository.start_run(
                {"model": {"api_key": secret}, "headers": {"Authorization": f"Bearer {secret}"}},
                {},
                root / "events.jsonl",
            )
            with database.connection() as connection:
                stored = str(
                    connection.execute(
                        "SELECT config_snapshot FROM extraction_run WHERE id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )

        self.assertNotIn(secret, stored)
        self.assertIn("[REDACTED]", stored)

    def test_provider_accounting_counts_retry_as_one_logical_batch_and_two_http_attempts(self):
        accounting = ProviderCallAccounting()
        client = ModelClient(
            ModelConfig(
                api_key="test-key",
                embedding_dimension=2,
                max_retries=1,
            ),
            accounting=accounting,
        )
        first = _Response(status_code=429, headers={"Retry-After": "0"})
        second = _Response(
            payload={"data": [{"index": 0, "embedding": [0.25, 0.75]}]}
        )

        with patch(
            "memory_demo.llm.client.requests.Session.post",
            side_effect=[first, second],
        ), patch("memory_demo.llm.client.time.sleep"):
            matrix = client.embed(["只编码一次的测试文本"])

        self.assertTrue(np.allclose(matrix, np.asarray([[0.25, 0.75]], dtype=np.float32)))
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        snapshot = accounting.snapshot()
        self.assertEqual(1, snapshot["logical_batches"])
        self.assertEqual(2, snapshot["http_attempts"])
        self.assertEqual(1, snapshot["http_succeeded"])
        self.assertEqual(1, snapshot["http_failed"])
        self.assertEqual(1, snapshot["retries_scheduled"])
        self.assertEqual({"embedding": 1}, snapshot["operations"])
        self.assertEqual({"embeddings": 2}, snapshot["endpoints"])
        self.assertEqual(1, snapshot["outcomes"]["http_429"])
        self.assertEqual(1, snapshot["outcomes"]["success"])

    def test_http_error_and_payload_opt_in_never_write_authorization_or_payload_text(self):
        secret = "v3-provider-secret"
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(log_path)
            accounting = ProviderCallAccounting()
            client = ModelClient(
                ModelConfig(api_key=secret, max_retries=0),
                logger,
                accounting=accounting,
            )
            response = _Response(
                status_code=401,
                content=f"Authorization: Bearer {secret}".encode("utf-8"),
            )
            with patch.dict(os.environ, {"MEMORY_LOG_MODEL_PAYLOADS": "true"}), patch(
                "memory_demo.llm.client.requests.Session.post", return_value=response
            ):
                with self.assertRaises(ModelClientError) as caught:
                    client._post(
                        "chat/completions",
                        {"model": "test", "input": [f"private {secret}"], "headers": {"Authorization": f"Bearer {secret}"}},
                    )
            raw = log_path.read_text(encoding="utf-8")

        self.assertTrue(response.closed)
        self.assertNotIn(secret, raw)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("provider response body withheld", str(caught.exception))
        snapshot = accounting.snapshot()
        self.assertEqual(1, snapshot["logical_batches"])
        self.assertEqual(1, snapshot["http_attempts"])
        self.assertEqual(1, snapshot["http_failed"])
        self.assertEqual(1, snapshot["outcomes"]["http_401"])


if __name__ == "__main__":
    unittest.main()
