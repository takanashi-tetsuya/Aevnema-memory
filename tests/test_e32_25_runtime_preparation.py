from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from benchmarks.export_e32_25_runtime_questions import (
    RUNTIME_SCHEMA,
    export_runtime_questions,
    validate_runtime_questions,
)
from benchmarks.prepare_e32_25_evaluator_v2 import build_evaluator_v2


_ROOT = Path(__file__).resolve().parents[1]
_EVALUATOR = (
    _ROOT
    / "validation"
    / "aevnema-v3_2"
    / "v3_2_20260908T133000JST"
    / "reports"
    / "E32-25_next_campaign_preflight"
    / "e32_25_evaluator_only_cases.full_local.json"
)


class E3225RuntimePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = _EVALUATOR.read_bytes()
        self.evaluator = json.loads(self.raw)

    def test_export_keeps_only_frozen_questions_and_runtime_labels(self) -> None:
        runtime = export_runtime_questions(self.evaluator)

        self.assertEqual(RUNTIME_SCHEMA, runtime["schema"])
        self.assertEqual(6, len(runtime["cases"]))
        self.assertEqual(
            [case["q1"] for case in self.evaluator["cases"]],
            [case["q1"] for case in runtime["cases"]],
        )
        for original, exported in zip(self.evaluator["cases"], runtime["cases"]):
            self.assertEqual(
                [original["q2"][variant] for variant in (
                    "same_text",
                    "paraphrase",
                    "partial_clue_with_neutral_context",
                    "near_neighbor_counterexample",
                )],
                [entry["question"] for entry in exported["q2"]],
            )
            self.assertNotIn("source_key", exported)
            self.assertNotIn("support_records", exported)
            self.assertNotIn("provisional_evaluator_claim", exported)
        mary = next(
            case for case in runtime["cases"]
            if case["case_id"] == "E32-25-favor-230082"
        )
        partial = next(
            item for item in mary["q2"]
            if item["variant"] == "partial_clue_with_neutral_context"
        )
        self.assertEqual("related_or_partial_clue_variant", partial["scope_label"])
        self.assertEqual("embedded_in_question", partial["visible_context"]["placement"])

    def test_runtime_consumer_rejects_an_evaluator_or_extra_field(self) -> None:
        runtime = export_runtime_questions(self.evaluator)
        contaminated = copy.deepcopy(runtime)
        contaminated["cases"][0]["source_key"] = "event/answer-location.json"
        with self.assertRaisesRegex(ValueError, "forbidden field|unexpected fields"):
            validate_runtime_questions(contaminated)

        extra_top_level = copy.deepcopy(runtime)
        extra_top_level["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unexpected top-level"):
            validate_runtime_questions(extra_top_level)

    def test_v2_preserves_v1_and_gives_each_near_neighbor_its_own_policy(self) -> None:
        v2 = build_evaluator_v2(self.raw)

        self.assertEqual(
            "sha256:" + hashlib.sha256(self.raw).hexdigest(), v2["parent_sha256"]
        )
        self.assertEqual(
            [case["q1"] for case in self.evaluator["cases"]],
            [case["q1"] for case in v2["cases"]],
        )
        for case in v2["cases"]:
            semantics = case["evaluator_v2"]["q2_variant_semantics"]
            near = semantics["near_neighbor_counterexample"]
            self.assertEqual("distinct_need_own_criterion", near["classification"])
            self.assertIn("provisional_evaluator_claim", near)
            self.assertEqual("alternative_any", near["policy"]["source_relevance"]["support_mode"])
            self.assertEqual("joint_all", near["policy"]["source_complete"]["support_mode"])
        mary = next(case for case in v2["cases"] if case["case_id"] == "E32-25-favor-230082")
        self.assertEqual(
            "method_hint_related_variant",
            mary["evaluator_v2"]["q2_variant_semantics"]
            ["partial_clue_with_neutral_context"]["classification"],
        )


if __name__ == "__main__":
    unittest.main()
