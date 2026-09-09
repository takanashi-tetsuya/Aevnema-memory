from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_same_chapter_learning_matrix import (
    MATRIX_SCHEMA,
    _ready_receipt_rows,
    _q2_summary,
    load_frozen_matrix_case,
)


def _manifest() -> dict[str, object]:
    return {
        "schema": MATRIX_SCHEMA,
        "formal_scoring_eligible": False,
        "promotion_prohibited": True,
        "q1": {
            "text": "为什么限制行动？",
            "contextual_domain": "knowledge",
            "contextual_revisit_scope": "test-scope",
        },
        "q2_variants": [
            {
                "id": "same",
                "kind": "same_text",
                "text": "为什么限制行动？",
                "context_mode": "absent",
                "reference_basis": "query_text_only",
            },
            {
                "id": "para",
                "kind": "paraphrase",
                "text": "为何限制大家行动？",
                "context_mode": "absent",
                "reference_basis": "query_text_only",
            },
            {
                "id": "partial",
                "kind": "partial_clue",
                "text": "为什么她这么做？",
                "context_mode": "absent",
                "reference_basis": "query_text_only",
            },
            {
                "id": "neighbor",
                "kind": "near_neighbor_counterexample",
                "text": "她为什么提出联手？",
                "context_mode": "absent",
                "reference_basis": "query_text_only",
            },
        ],
    }


class SameChapterLearningMatrixManifestTests(unittest.TestCase):
    def _load(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return load_frozen_matrix_case(path)

    def test_accepts_the_complete_non_scorable_matrix(self):
        case = self._load(_manifest())
        self.assertEqual("为什么限制行动？", case.q1_text)
        self.assertEqual(
            {"same_text", "paraphrase", "partial_clue", "near_neighbor_counterexample"},
            {item.kind for item in case.variants},
        )
        partial = next(item for item in case.variants if item.kind == "partial_clue")
        self.assertEqual("absent", partial.context_mode)
        self.assertEqual("query_text_only", partial.reference_basis)

    def test_rejects_hidden_context_and_answer_hints(self):
        with self.subTest("partial_context"):
            payload = _manifest()
            variants = payload["q2_variants"]
            assert isinstance(variants, list)
            variants[2]["context_text"] = "hidden antecedent"
            with self.assertRaisesRegex(ValueError, "partial clue"):
                self._load(payload)
        with self.subTest("answer_hint"):
            payload = _manifest()
            payload["answer"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "gold/answer/hint"):
                self._load(payload)

    def test_rejects_nonidentical_same_text(self):
        payload = _manifest()
        variants = payload["q2_variants"]
        assert isinstance(variants, list)
        variants[0]["text"] = "换一个问题"
        with self.assertRaisesRegex(ValueError, "same_text"):
            self._load(payload)

    def test_accepts_a_live_typed_ready_runtime_manifest(self):
        @dataclass(frozen=True)
        class RuntimeManifest:
            state: str

        rows = _ready_receipt_rows(
            {
                "new_receipts": [
                    {
                        "receipt": {"status": "ready", "association_id": 121},
                        "association_id_is_new": True,
                        "runtime_manifest": RuntimeManifest(state="ready"),
                    }
                ]
            }
        )
        self.assertEqual(1, len(rows))

        rejected = _ready_receipt_rows(
            {
                "new_receipts": [
                    {
                        "receipt": {"status": "ready", "association_id": 121},
                        "association_id_is_new": True,
                        "runtime_manifest": RuntimeManifest(state="pending"),
                    }
                ]
            }
        )
        self.assertEqual([], rejected)

    def test_terminal_q2_exception_does_not_claim_an_edge_was_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = _q2_summary(
                {
                    "status": "failed",
                    "elapsed_ms": 12.0,
                    "error": {"type": "NameError", "message": "example"},
                    "provider": {"observations": []},
                },
                log_dir=Path(directory),
                this_run_edge_id=121,
            )
        self.assertEqual(
            "not_observed_due_to_terminal_exception",
            summary["this_run_edge_participated"],
        )
        self.assertEqual(
            "not_observed_due_to_terminal_exception",
            summary["actual_edge_path"],
        )

    def test_evidence_only_summary_keeps_delivery_and_participation_separate_from_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = _q2_summary(
                {
                    "status": "completed",
                    "elapsed_ms": 12.0,
                    "provider": {"observations": []},
                    "result": {
                        "execution_profile": "live_evidence",
                        "candidate_episode_ids": [18, 32],
                        "association_ids": [121],
                        "association_usage_recorded": False,
                        "query_vector_bundle": {"logical_binding_count": 1},
                        "rerank_trace": {"enabled": False},
                        "evidence_slot_trace": {},
                        "timings": {"phases_seconds": {"exact_revisit": 0.01}},
                        "evidence_result": {
                            "evidence_state": "complete",
                            "runtime_coverage_status": "not_observed",
                            "materialized_source_refs": [{"episode_id": 32}],
                            "participation": {"edge_state": "contributed"},
                            "actual_skipped_modules": ["answer_generation"],
                            "executed_modules": {
                                "answer_generation": False,
                                "source_materialization": True,
                            },
                            "delivery_states": {
                                "evidence_materialized": True,
                                "answer_input_sent": False,
                            },
                        },
                    },
                },
                log_dir=Path(directory),
                this_run_edge_id=121,
            )
        self.assertEqual("not_run_evidence_only", summary["ordinary_answer_status"])
        self.assertEqual([32], summary["delivered_episode_ids"])
        self.assertEqual("contributed", summary["this_run_edge_participated"])
        self.assertFalse(summary["association_usage_recorded"])
        self.assertTrue(summary["evidence_materialized"])


if __name__ == "__main__":
    unittest.main()
