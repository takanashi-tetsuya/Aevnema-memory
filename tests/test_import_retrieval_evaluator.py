from __future__ import annotations

import unittest

from benchmarks.evaluate_import_retrieval import resolve_expected_episode_ids


class ImportRetrievalEvaluatorTests(unittest.TestCase):
    def test_resolves_matcher_against_text_and_evidence(self):
        rows = [
            {
                "id": 7,
                "source_key": "main/a.txt",
                "text": "圣亚提出一位预言者。",
                "evidence_quotes_json": '["セイア: クズノハ。", "ニヤ: 了解。"]',
            },
            {
                "id": 8,
                "source_key": "main/b.txt",
                "text": "无关事件",
                "evidence_quotes_json": "[]",
            },
        ]

        expected, details = resolve_expected_episode_ids(
            rows,
            {
                "id": "q1",
                "expected_episode_matchers": [
                    {
                        "source_key": "main/a.txt",
                        "all_terms": ["クズノハ", "ニヤ"],
                    }
                ],
            },
        )

        self.assertEqual(expected, {7})
        self.assertTrue(details[0]["resolved"])

    def test_unresolved_matcher_is_reported_without_inventing_id(self):
        expected, details = resolve_expected_episode_ids(
            [
                {
                    "id": 1,
                    "source_key": "main/a.txt",
                    "text": "普通事件",
                    "evidence_quotes_json": "[]",
                }
            ],
            {
                "id": "q2",
                "expected_episode_matchers": [
                    {"all_terms": ["不存在的证据"]}
                ],
            },
        )

        self.assertEqual(expected, set())
        self.assertFalse(details[0]["resolved"])

    def test_each_matcher_remains_a_distinct_required_group(self):
        rows = [
            {
                "id": 1,
                "source_key": "main/a.txt",
                "text": "第一项事实",
                "evidence_quotes_json": "[]",
            },
            {
                "id": 2,
                "source_key": "main/a.txt",
                "text": "第二项事实",
                "evidence_quotes_json": "[]",
            },
        ]

        expected, details = resolve_expected_episode_ids(
            rows,
            {
                "expected_episode_matchers": [
                    {"all_terms": ["第一项"]},
                    {"all_terms": ["第二项"]},
                ]
            },
        )

        self.assertEqual(expected, {1, 2})
        self.assertEqual(
            [item["matched_episode_ids"] for item in details], [[1], [2]]
        )


if __name__ == "__main__":
    unittest.main()
