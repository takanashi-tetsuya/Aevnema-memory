from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import json

from benchmarks.run_atomic_answer_scope_eval import (
    _extraction_packet,
    _scope_packet,
    _validate_extraction_anchors,
)
from benchmarks.run_atomic_answer_repair_eval import (
    _deterministic_quality,
    _repaired_source_payload,
)
from benchmarks.run_answer_persistence_repair_eval import _load_rows
from config.prompt_config.answer_persistence_repair_prompts import _contract_view


class AtomicAnswerScopeEvalTests(unittest.TestCase):
    def test_extraction_normalizes_unknown_metadata_without_dropping_claim(self) -> None:
        packet = _extraction_packet(
            {
                "claims": [
                    {
                        "id": "C1",
                        "text": "A 帮助了 B",
                        "source_span": "A 帮助了 B",
                        "kind": "unknown-kind",
                        "assertion_mode": "unknown-mode",
                    }
                ]
            }
        )
        self.assertEqual("state", packet["claims"][0]["kind"])
        self.assertEqual("asserted", packet["claims"][0]["assertion_mode"])

    def test_scope_packet_requires_one_decision_per_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing scope decisions"):
            _scope_packet(
                {
                    "overall_verdict": "safe",
                    "decisions": [
                        {"claim_id": "C1", "scope": "supported_fact"}
                    ],
                },
                {"C1", "C2"},
            )

    def test_unsupported_claim_forces_unsafe_verdict(self) -> None:
        packet = _scope_packet(
            {
                "overall_verdict": "safe",
                "decisions": [
                    {"claim_id": "C1", "scope": "unsupported"}
                ],
            },
            {"C1"},
        )
        self.assertEqual("unsafe", packet["overall_verdict"])
        self.assertTrue(packet["decisions"][0]["persistent"])

    def test_extraction_anchor_must_be_an_exact_answer_span(self) -> None:
        packet = {
            "claims": [
                {
                    "id": "C1",
                    "text": "阿洛娜是学生的助手",
                    "source_span": "阿洛娜是学生的助手",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "exact answer source_span"):
            _validate_extraction_anchors(packet, "阿洛娜可以帮老师记录下来")

    def test_extraction_anchor_accepts_a_literal_source_span(self) -> None:
        packet = {
            "claims": [
                {
                    "id": "C1",
                    "text": "阿洛娜提出帮助老师记录",
                    "source_span": "阿洛娜可以帮老师记录下来",
                }
            ]
        }
        self.assertIs(
            packet,
            _validate_extraction_anchors(packet, "阿洛娜可以帮老师记录下来哦！"),
        )
        self.assertEqual("阿洛娜可以帮老师记录下来", packet["claims"][0]["text"])
        self.assertNotIn("normalized_text", packet["claims"][0])

    def test_extraction_anchor_drops_only_invalid_claims(self) -> None:
        packet = {
            "claims": [
                {"id": "C1", "text": "有效", "source_span": "原文"},
                {"id": "C2", "text": "污染", "source_span": "问题里的文字"},
            ]
        }
        result = _validate_extraction_anchors(packet, "这是原文")
        self.assertEqual(["C1"], [claim["id"] for claim in result["claims"]])
        self.assertEqual(["C2"], result["anchor_rejections"])

    def test_contract_view_keeps_request_directives_receipts_and_premises(self) -> None:
        view = _contract_view(
            {
                "requested_actions": [
                    {
                        "action": "memory_write",
                        "target": "private",
                        "public_action_summary": "store marker in private memory",
                    }
                ],
                "runtime_receipts": [
                    {
                        "receipt_id": "R1",
                        "action": "memory_write",
                        "target": "private",
                        "status": "succeeded",
                        "result_summary": "stored",
                    }
                ],
                "question_premises": [
                    {"premise": "A met B", "status": "supported"}
                ],
                "response_directives": [
                    {
                        "kind": "action_outcome",
                        "required": True,
                        "action": "memory_write",
                        "target": "private",
                        "status": "succeeded",
                        "result_summary": "stored",
                        "ignored_internal_field": "must not leak",
                    }
                ],
            }
        )
        self.assertEqual("private", view["requested_actions"][0]["target"])
        self.assertEqual(
            "store marker in private memory",
            view["requested_actions"][0]["public_action_summary"],
        )
        self.assertEqual("memory_write", view["runtime_receipts"][0]["action"])
        self.assertEqual("supported", view["question_premises"][0]["status"])
        self.assertEqual(
            "succeeded", view["response_directives"][0]["status"]
        )
        self.assertNotIn(
            "ignored_internal_field", view["response_directives"][0]
        )

    def test_scope_packet_accepts_receipt_and_question_premise_scopes(self) -> None:
        packet = _scope_packet(
            {
                "overall_verdict": "safe",
                "decisions": [
                    {"claim_id": "C1", "scope": "runtime_receipt"},
                    {"claim_id": "C2", "scope": "question_premise"},
                ],
            },
            {"C1", "C2"},
        )
        self.assertEqual("safe", packet["overall_verdict"])

    def test_self_contained_manifest_supplies_label_and_expected(self) -> None:
        payload = {
            "cases": [
                {
                    "id": "custom:one",
                    "name": "one",
                    "question": "Q",
                    "answer": "A",
                    "contract": {},
                    "local_guard": {"verdict": "safe"},
                    "expected": {},
                    "human_persistence_verdict": "unsafe",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = _load_rows(path, set())
        self.assertEqual("unsafe", rows[0]["human_persistence_verdict"])

    def test_repaired_manifest_is_self_contained_and_targets_safe_answer(self) -> None:
        payload = _repaired_source_payload(
            [
                {
                    "id": "custom:one",
                    "name": "one",
                    "question": "Q",
                    "final_answer": "repaired",
                    "contract": {},
                    "local_guard": {"verdict": "safe"},
                    "expected": {"required": ["repaired"]},
                    "human_persistence_verdict": "unsafe",
                }
            ]
        )
        case = payload["cases"][0]
        self.assertEqual({"required": ["repaired"]}, case["expected"])
        self.assertEqual("safe", case["human_persistence_verdict"])

    def test_empty_deterministic_expectation_is_explicitly_skipped(self) -> None:
        result = _deterministic_quality({}, "free-form answer", {})
        self.assertTrue(result["passed"])
        self.assertTrue(result["skipped"])


if __name__ == "__main__":
    unittest.main()
