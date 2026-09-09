from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.types import QueryIntent


class _RevisionModel:
    def __init__(self, audits: list[bool]) -> None:
        self.audits = list(audits)
        self.calls: list[str] = []

    def chat_text(self, _system: str, _prompt: str) -> str:
        self.calls.append("chat_text")
        return "corrected" if self.calls.count("chat_text") > 1 else "initial"

    def chat_json(self, _system: str, _prompt: str) -> dict:
        self.calls.append("chat_json")
        valid = self.audits.pop(0)
        return {
            "valid": valid,
            "reviews": [
                {
                    "claim": "answer claim",
                    "verdict": "supported_fact" if valid else "unsupported",
                    "reason": "fixture",
                }
            ],
        }


def _engine(model: _RevisionModel, log_path: Path) -> QueryEngine:
    engine = object.__new__(QueryEngine)
    engine.model = model
    engine.logger = JsonlEventLogger(log_path, answer_evidence_enabled=True)
    engine.config = AppConfig()
    engine.config.retrieval.answer_evidence_checkpoint_enabled = True
    return engine


def _intent() -> QueryIntent:
    return QueryIntent(target_entities=["entity"], search_queries=["question"])


def _episodes() -> list[dict]:
    return [{
        "id": 7,
        "text": "literal source evidence",
        "source_key": "main/test.json",
        "source_evidence_delivery": "source_bound",
        "source_evidence_quote_count": 1,
        "source_text": "[record: 7]\nzh-CN: literal source evidence",
    }]


class AnswerBudgetAdmissionTests(unittest.TestCase):
    def test_insufficient_full_chain_budget_skips_correction_and_records_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            model = _RevisionModel([False])
            engine = _engine(model, log_path)

            _answer, audits, revisions = engine._generate_audited_answer(
                "private question", _intent(), _episodes(), [], [], [], deadline_at=0.0
            )

            self.assertEqual(["chat_text", "chat_json"], model.calls)
            self.assertEqual(0, revisions)
            self.assertFalse(audits[0]["valid"])
            self.assertEqual(
                "insufficient_budget_for_correction_reaudit_and_finalization",
                engine._last_answer_execution_state["reason"],
            )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                ["answer_input_checkpoint", "answer_revision_checkpoint", "answer_evidence_audit", "answer_revision_checkpoint", "answer_correction_admission"],
                [row["event"] for row in events],
            )
            admission = events[-1]
            self.assertFalse(admission["admitted"])
            self.assertEqual(80.0, admission["required_seconds"])
            self.assertNotIn('"answer":"initial"', json.dumps(events, ensure_ascii=False))
            self.assertEqual(
                {"redacted": True, "char_count": len("private question")},
                events[0]["question"],
            )
            evidence_events = [
                json.loads(line)
                for line in log_path.with_name(
                    log_path.stem + ".answer-evidence.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    "answer_input_checkpoint",
                    "answer_revision_checkpoint",
                    "answer_revision_checkpoint",
                    "answer_revision_checkpoint",
                ],
                [row["event"] for row in evidence_events],
            )
            answer_input, initial_answer, audit_result, rejected = evidence_events
            self.assertTrue(answer_input["local_only"])
            self.assertEqual("private question", answer_input["question"])
            self.assertEqual(
                "literal source evidence",
                answer_input["selected_evidence"][0]["episode_text"],
            )
            self.assertEqual(
                "[record: 7]\nzh-CN: literal source evidence",
                answer_input["selected_evidence"][0]["source_excerpt"],
            )
            self.assertEqual("initial", initial_answer["answer"])
            self.assertFalse(audit_result["audit"]["valid"])
            self.assertTrue(
                answer_input["prompt_evidence_assertions"]["answer_input"]
                ["all_source_excerpts_present"]
            )
            self.assertTrue(
                audit_result["prompt_evidence_assertions"]["audit_input"]
                ["all_source_excerpts_present"]
            )
            self.assertEqual(
                "insufficient_budget_for_correction_reaudit_and_finalization",
                rejected["terminal_reason"],
            )
            self.assertIn("answer_system_hash", answer_input["version_binding"])

    def test_answer_evidence_companion_still_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(log_path, answer_evidence_enabled=True)

            logger.emit_answer_evidence_checkpoint(
                "answer_input_checkpoint",
                stage="answer_input",
                question="answer boundary question",
                version_binding={"api_key": "do-not-store-this"},
            )

            payload = json.loads(
                log_path.with_name(log_path.stem + ".answer-evidence.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual("[REDACTED]", payload["version_binding"]["api_key"])
            self.assertEqual("answer boundary question", payload["question"])

    def test_answer_evidence_companion_requires_explicit_enablement(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(log_path)

            with self.assertRaisesRegex(RuntimeError, "explicit experimental enablement"):
                logger.emit_answer_evidence_checkpoint(
                    "answer_input_checkpoint",
                    stage="answer_input",
                    question="private question",
                )
            self.assertFalse(
                log_path.with_name(log_path.stem + ".answer-evidence.jsonl").exists()
            )

    def test_one_correction_and_required_reaudit_are_allowed_when_envelope_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _RevisionModel([False, True])
            engine = _engine(model, Path(directory) / "events.jsonl")

            answer, audits, revisions = engine._generate_audited_answer(
                "question", _intent(), _episodes(), [], [], [], deadline_at=10_000_000.0
            )

            self.assertEqual("corrected", answer)
            self.assertEqual([False, True], [audit["valid"] for audit in audits])
            self.assertEqual(1, revisions)
            self.assertEqual("completed_verified", engine._last_answer_execution_state["terminal_state"])

    def test_second_correction_is_not_started_after_a_failed_reaudit(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _RevisionModel([False, False])
            engine = _engine(model, Path(directory) / "events.jsonl")

            _answer, audits, revisions = engine._generate_audited_answer(
                "question", _intent(), _episodes(), [], [], [], deadline_at=10_000_000.0
            )

            self.assertEqual(["chat_text", "chat_json", "chat_text", "chat_json"], model.calls)
            self.assertEqual([False, False], [audit["valid"] for audit in audits])
            self.assertEqual(1, revisions)
            self.assertEqual("completed_limited", engine._last_answer_execution_state["terminal_state"])
            self.assertEqual("correction_revision_limit_reached", engine._last_answer_execution_state["reason"])


if __name__ == "__main__":
    unittest.main()
