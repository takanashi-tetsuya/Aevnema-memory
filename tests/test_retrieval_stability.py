from __future__ import annotations

import unittest

from memory_demo.retrieval_stability import (
    aggregate_stability_runs,
    score_fixed_candidate_run,
    validate_frozen_candidate_alignment,
)


class RetrievalStabilityTests(unittest.TestCase):
    def test_alignment_rejects_reused_ids_with_different_text(self):
        source_rows = [
            {
                "id": "q1",
                "result": {
                    "rerank_trace": {"candidate_episode_ids": [7, 8]},
                    "evidence_episodes": [
                        {"id": 7, "text": "老师应看到故事的最后。"}
                    ],
                },
            }
        ]

        aligned = validate_frozen_candidate_alignment(
            source_rows,
            {7: "老师应看到故事的最后。", 8: "其他证据"},
        )
        reused = validate_frozen_candidate_alignment(
            source_rows,
            {7: "同一个数字 ID 指向了新内容", 8: "其他证据"},
        )

        self.assertTrue(aligned["passed"])
        self.assertFalse(reused["passed"])
        self.assertEqual(reused["evidence_text_mismatch_count"], 1)

    def test_scores_alternative_groups_and_coverage_disagreement(self):
        score = score_fixed_candidate_run(
            selected_episode_ids=[4, 2],
            candidate_episode_ids=[1, 2, 3, 4],
            required_episode_groups=[[1, 2], [3], [4]],
            rerank_trace={
                "enabled": True,
                "backend": "llm",
                "coverage_audit_performed": True,
                "compressor_performed": True,
                "initial": {
                    "coverage": [
                        {"query": "a", "episode_ids": [2]},
                        {"query": "c", "episode_ids": [4]},
                    ]
                },
                "independent_coverage": {
                    "coverage": [
                        {"query": "b", "episode_ids": [3]},
                        {"query": "c", "episode_ids": [4]},
                    ]
                },
            },
        )

        self.assertEqual(score["selected_hit_count"], 2)
        self.assertAlmostEqual(score["recall_at_20"], 2 / 3)
        self.assertEqual(score["candidate_recall_at_100"], 1.0)
        self.assertEqual(score["coverage_vote_disagreement_count"], 2)
        self.assertEqual(score["estimated_model_calls"], 3)
        self.assertEqual(score["groups"][0]["first_rank"], 2)

    def test_aggregate_reports_slot_frequency_and_question_minimum(self):
        def row(repeat_index: int, selected: list[int]) -> dict:
            return {
                "question_id": "q1",
                "question": "问题",
                "repeat_index": repeat_index,
                "elapsed_seconds": float(repeat_index),
                "score": score_fixed_candidate_run(
                    selected_episode_ids=selected,
                    candidate_episode_ids=[1, 2],
                    required_episode_groups=[[1], [2]],
                    rerank_trace={},
                ),
            }

        summary = aggregate_stability_runs(
            [row(1, [1, 2]), row(2, [1])], expected_repeats=2
        )

        self.assertEqual(summary["completed_runs"], 2)
        self.assertEqual(summary["candidate_recall_at_100_minimum"], 1.0)
        self.assertEqual(summary["selected_recall_at_20_minimum"], 0.5)
        self.assertEqual(summary["unstable_required_slot_count"], 1)
        question = summary["questions"][0]
        self.assertEqual(question["recall_at_20_mean"], 0.75)
        self.assertEqual(question["groups"][1]["hit_rate"], 0.5)
        self.assertEqual(question["groups"][1]["missed_repeat_indices"], [2])


if __name__ == "__main__":
    unittest.main()
