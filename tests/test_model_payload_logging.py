from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from memory_demo.config import ModelConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.client import ModelClient


class ModelPayloadLoggingTests(unittest.TestCase):
    def test_opt_in_writes_request_and_response_to_private_companion(self):
        with TemporaryDirectory() as directory:
            logger = JsonlEventLogger(Path(directory) / "query.jsonl")
            client = ModelClient(ModelConfig(api_key="test-key"), logger)
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [{"message": {"content": "model-output"}}],
                "api_key": "provider-secret",
            }
            session = Mock()
            session.post.return_value = response
            with patch.dict(os.environ, {"MEMORY_LOG_MODEL_PAYLOADS": "true"}), patch.object(
                client, "_session", return_value=session
            ):
                client._post(
                    "chat/completions",
                    {"model": "test", "messages": [{"role": "user", "content": "model-input"}]},
                )

            private = [
                json.loads(line)
                for line in logger.model_payload_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["event"] for item in private], ["llm_request", "llm_response"])
            self.assertEqual(private[0]["request_id"], private[1]["request_id"])
            self.assertIn("model-input", json.dumps(private, ensure_ascii=False))
            self.assertIn("model-output", json.dumps(private, ensure_ascii=False))
            self.assertNotIn("provider-secret", json.dumps(private, ensure_ascii=False))
            self.assertEqual(logger.model_payload_path.stat().st_mode & 0o777, 0o600)
            operational = logger.path.read_text(encoding="utf-8")
            self.assertNotIn("model-input", operational)
            self.assertNotIn("model-output", operational)
            self.assertIn('"payload_companion_logged":true', operational)

    def test_disabled_flag_keeps_only_redacted_operational_log(self):
        with TemporaryDirectory() as directory:
            logger = JsonlEventLogger(Path(directory) / "query.jsonl")
            client = ModelClient(ModelConfig(api_key="test-key"), logger)
            with patch.dict(os.environ, {"MEMORY_LOG_MODEL_PAYLOADS": "false"}):
                self.assertFalse(client._emit_model_payload(
                    "llm_request", request_id="a", endpoint="chat/completions", payload={"input": "secret"}
                ))
            self.assertFalse(logger.model_payload_path.exists())
