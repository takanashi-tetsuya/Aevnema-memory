from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.export_q1_q2_case_postmortem import (
    NOT_OBSERVED,
    build_postmortem,
    export_postmortem,
)


class Q1Q2CasePostmortemTests(unittest.TestCase):
    def test_builds_only_observed_http_and_preserves_absent_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "case.json"
            log_path = root / "q1.jsonl"
            case_path.write_text(
                json.dumps(
                    {
                        "q1": {
                            "record": {
                                "status": "failed",
                                "started_at": "2026-09-07T00:00:00+00:00",
                                "elapsed_ms": 120000,
                                "error": {"type": "ModelDeadlineExceeded"},
                                "provider": {
                                    "counts": {"http_attempts": 1, "succeeded": 0, "failed_or_rejected": 1},
                                    "observations": [
                                        {
                                            "logical_batch_id": "rerank-1",
                                            "purpose": "evidence_rerank_selection",
                                            "operation": "chat_json",
                                            "requested_model": "deepseek-ai/DeepSeek-V3.2",
                                            "status": "timeout",
                                            "sent": True,
                                        }
                                    ],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            events = [
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "provider_call", "phase": "logical_batch_started", "logical_batch_id": "rerank-1"},
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "llm_request", "logical_batch_id": "rerank-1", "payload": {"input_chars": 77}, "content_logged": False},
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "provider_call", "phase": "http_attempt", "logical_batch_id": "rerank-1"},
                {"timestamp": "2026-09-07T00:01:31+00:00", "event": "provider_call", "phase": "http_result", "logical_batch_id": "rerank-1", "outcome": "timeout"},
                {"timestamp": "2026-09-07T00:01:31+00:00", "event": "llm_error", "logical_batch_id": "rerank-1", "error": "Read timed out"},
                {"timestamp": "2026-09-07T00:01:31+00:00", "event": "evidence_rerank_failed", "error": "ModelClientError: Read timed out"},
                {"timestamp": "2026-09-07T00:01:32+00:00", "event": "llm_request", "logical_batch_id": "answer-1", "operation": "chat_text", "payload": {"input_chars": 1000}, "content_logged": False},
            ]
            log_path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")

            payload = build_postmortem(case_path, log_path)

            replay = payload["q1"]["http_attempts"][0]
            self.assertEqual("timeout", replay["ledger_status"])
            self.assertEqual(1000.0, replay["elapsed_since_q1_start_ms_at_http_attempt"])
            self.assertEqual(NOT_OBSERVED, replay["remaining_budget_ms"])
            self.assertEqual(NOT_OBSERVED, replay["input_state"]["raw_input"])
            self.assertEqual(NOT_OBSERVED, replay["output_state"]["raw_output"])
            self.assertEqual("ModelClientError: Read timed out", payload["rerank_observation"]["failure_event"])
            self.assertEqual(NOT_OBSERVED, payload["late_answer_request"]["raw_prompt"])

            output = root / "export"
            exported = export_postmortem(case_path=case_path, q1_jsonl_path=log_path, output_dir=output)
            self.assertTrue(Path(exported["full_local"]).is_file())
            self.assertTrue(Path(exported["report"]).is_file())
            self.assertTrue(case_path.is_file())
            with self.assertRaises(FileExistsError):
                export_postmortem(case_path=case_path, q1_jsonl_path=log_path, output_dir=output)

    def test_pairs_primary_and_fallback_as_two_http_attempts_in_one_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "case.json"
            log_path = root / "q1.jsonl"
            case_path.write_text(
                json.dumps(
                    {
                        "q1": {
                            "record": {
                                "status": "failed",
                                "started_at": "2026-09-07T00:00:00+00:00",
                                "provider": {
                                    "counts": {"http_attempts": 2, "succeeded": 1, "failed_or_rejected": 1},
                                    "observations": [
                                        {
                                            "call_id": "primary-call",
                                            "logical_batch_id": "answer-1",
                                            "purpose": "answer_generation",
                                            "operation": "chat_text",
                                            "requested_model": "deepseek-ai/DeepSeek-V3.2",
                                            "actual_model": None,
                                            "status": "timeout",
                                        },
                                        {
                                            "call_id": "fallback-call",
                                            "logical_batch_id": "answer-1",
                                            "purpose": "answer_generation",
                                            "operation": "chat_text",
                                            "requested_model": "deepseek-ai/DeepSeek-V3.2",
                                            "actual_model": "zai-org/GLM-4.5V",
                                            "status": "succeeded",
                                        },
                                    ],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            events = [
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "provider_call", "phase": "logical_batch_started", "logical_batch_id": "answer-1"},
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "llm_request", "request_id": "primary-call", "logical_batch_id": "answer-1", "payload": {"model": "deepseek-ai/DeepSeek-V3.2"}, "content_logged": False},
                {"timestamp": "2026-09-07T00:00:01+00:00", "event": "provider_call", "phase": "http_attempt", "logical_batch_id": "answer-1"},
                {"timestamp": "2026-09-07T00:00:26+00:00", "event": "provider_call", "phase": "http_result", "logical_batch_id": "answer-1", "outcome": "timeout"},
                {"timestamp": "2026-09-07T00:00:26+00:00", "event": "llm_error", "logical_batch_id": "answer-1", "error": "Read timed out"},
                {"timestamp": "2026-09-07T00:00:26+00:00", "event": "fallback", "from_model": "deepseek-ai/DeepSeek-V3.2", "to_model": "zai-org/GLM-4.5V"},
                {"timestamp": "2026-09-07T00:00:26+00:00", "event": "llm_request", "request_id": "fallback-call", "logical_batch_id": "answer-1", "payload": {"model": "zai-org/GLM-4.5V"}, "content_logged": False},
                {"timestamp": "2026-09-07T00:00:26+00:00", "event": "provider_call", "phase": "http_attempt", "logical_batch_id": "answer-1"},
                {"timestamp": "2026-09-07T00:00:36+00:00", "event": "provider_call", "phase": "http_result", "logical_batch_id": "answer-1", "outcome": "success"},
                {"timestamp": "2026-09-07T00:00:36+00:00", "event": "llm_response", "request_id": "fallback-call", "logical_batch_id": "answer-1", "payload": {"model": "zai-org/GLM-4.5V"}, "content_logged": False},
            ]
            log_path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")

            payload = build_postmortem(case_path, log_path)

            attempts = payload["q1"]["http_attempts"]
            self.assertEqual(2, len(attempts))
            self.assertEqual("deepseek-ai/DeepSeek-V3.2", attempts[0]["request_model"])
            self.assertEqual("zai-org/GLM-4.5V", attempts[1]["request_model"])
            self.assertEqual("timeout", attempts[0]["output_state"]["http_outcome"])
            self.assertEqual("success", attempts[1]["output_state"]["http_outcome"])
            self.assertEqual("zai-org/GLM-4.5V", attempts[0]["fallback_after_this_attempt"]["to_model"])
            self.assertEqual("deepseek-ai/DeepSeek-V3.2", attempts[1]["fallback_into_this_attempt"]["from_model"])


if __name__ == "__main__":
    unittest.main()
