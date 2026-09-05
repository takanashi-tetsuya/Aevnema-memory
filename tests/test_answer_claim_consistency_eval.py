from __future__ import annotations

import unittest

from benchmarks.run_answer_claim_consistency_deadline_eval import (
    _aggregate_chunk_results,
    _answer_chunks,
    _local_guard,
)


class AnswerClaimConsistencyEvalTests(unittest.TestCase):
    def test_local_guard_catches_structural_risk_but_not_old_promotion_noise(self) -> None:
        safe = _local_guard(
            {
                "intent_mode": "factual",
                "validation": {"invalid_declared_promotions": ["episode:1"]},
            }
        )
        unsafe = _local_guard(
            {
                "intent_mode": "factual",
                "validation": {"context_as_evidence": ["episode:2"]},
            }
        )
        self.assertEqual("safe", safe["verdict"])
        self.assertEqual("unsafe", unsafe["verdict"])

    def test_creative_without_private_write_is_structurally_unsafe(self) -> None:
        result = _local_guard(
            {
                "intent_mode": "creative",
                "write_candidates": {"private": []},
                "validation": {},
            }
        )
        self.assertEqual("unsafe", result["verdict"])
        self.assertIn("creative_without_private_write", result["reasons"])

    def test_answer_chunks_keep_all_non_whitespace_text(self) -> None:
        source = "第一句。\n\n第二句很长！ Third sentence?"
        chunks = _answer_chunks(source, 10)
        compact_source = "".join(source.split())
        compact_chunks = "".join("".join(chunks).split())
        self.assertEqual(compact_source, compact_chunks)
        self.assertTrue(all(len(value) <= 10 for value in chunks))

    def test_unsafe_chunk_dominates_and_incomplete_safe_chunks_do_not_pass(self) -> None:
        unsafe = _aggregate_chunk_results(
            [
                {"verdict": "safe", "timed_out": False},
                {"verdict": "unsafe", "timed_out": False},
            ]
        )
        incomplete = _aggregate_chunk_results(
            [
                {"verdict": "safe", "timed_out": False},
                {"verdict": "", "timed_out": True},
            ]
        )
        self.assertEqual("unsafe", unsafe["verdict"])
        self.assertEqual("", incomplete["verdict"])
        self.assertTrue(incomplete["timed_out"])


if __name__ == "__main__":
    unittest.main()
