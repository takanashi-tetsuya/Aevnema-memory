from __future__ import annotations

import unittest

from memory_demo.types import QueryIntent


class QueryIntentNormalizationTests(unittest.TestCase):
    def test_null_like_constraints_are_empty(self):
        intent = QueryIntent.from_dict(
            {
                "requested_relation": "None",
                "temporal_constraint": None,
                "causal_constraint": "无明确因果约束",
            }
        )

        self.assertEqual(intent.requested_relation, "")
        self.assertEqual(intent.temporal_constraint, "")
        self.assertEqual(intent.causal_constraint, "")


if __name__ == "__main__":
    unittest.main()
