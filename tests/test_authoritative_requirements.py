from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from memory_demo.retrieval.coverage import select_evidence
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import (
    requirement_support_from_records,
    resolve_authoritative_requirements,
)
from memory_demo.trace import _assert_trace_safe
from memory_demo.types import QueryIntent, SlotCandidate


class AuthoritativeRequirementTests(unittest.TestCase):
    @staticmethod
    def _three_slot_resolution():
        return resolve_authoritative_requirements(
            "完整问题只用来发现候选，不能替代三个已规划的需求。",
            QueryIntent(
                target_entities=["甲", "乙"],
                search_queries=["需求一", "需求二", "需求三"],
            ),
            request_mode="factual",
            planner_origin="local",
        )

    def test_three_authoritative_slots_keep_an_unsupported_third_in_denominator(self):
        resolution = self._three_slot_resolution()
        slots, support = QueryEngine._request_evidence_slots(
            {
                "merged_coverage": {
                    "coverage": [
                        {"query": "需求一", "episode_ids": [101]},
                        {"query": "需求二", "episode_ids": [202]},
                        # The third planner requirement has no candidate.
                    ]
                }
            },
            authoritative_requirements=resolution.requirements,
        )

        self.assertEqual("resolved", resolution.status)
        self.assertEqual(3, len(slots))
        self.assertEqual(2, len(support))
        candidates = [
            SlotCandidate(episode_id, frozenset(slot_ids))
            for episode_id, slot_ids in support.items()
        ]
        selected = select_evidence(candidates, slots, budget=3)
        third_slot_id = next(
            item.slot_id for item in slots if item.question == "需求三"
        )
        self.assertEqual({third_slot_id}, set(selected.missing_required))

    def test_empty_candidate_records_do_not_erase_authoritative_slots(self):
        resolution = self._three_slot_resolution()
        slots, support = QueryEngine._request_evidence_slots(
            {"merged_coverage": {"coverage": []}},
            authoritative_requirements=resolution.requirements,
        )

        selection = select_evidence([], slots, budget=3)
        self.assertEqual(3, len(slots))
        self.assertEqual({}, support)
        self.assertEqual(
            {item.slot_id for item in slots}, set(selection.missing_required)
        )

    def test_discovery_hints_never_become_required_slots(self):
        resolved = resolve_authoritative_requirements(
            "整题问题",
            QueryIntent(search_queries=["规划的原子问题"]),
            discovery_hints=[
                "整题问题",
                "后续追问",
                "重排展开问题",
                "上下文匹配词",
            ],
        )
        empty = resolve_authoritative_requirements(
            "整题问题",
            QueryIntent(),
            discovery_hints=["整题问题", "后续追问", "重排展开问题"],
        )

        self.assertEqual(["规划的原子问题"], [item.question for item in resolved.requirements])
        self.assertEqual("unknown", empty.status)
        self.assertEqual((), empty.requirements)

    def test_empty_factual_is_unknown_but_nonfactual_empty_is_not_applicable(self):
        intent = QueryIntent()
        factual = resolve_authoritative_requirements(
            "一个没有规划出事实槽的问题", intent, request_mode="factual"
        )
        conversational = resolve_authoritative_requirements(
            "你好", intent, request_mode="conversational"
        )
        exploratory = resolve_authoritative_requirements(
            "随便探索一下", intent, request_mode="exploratory"
        )

        self.assertEqual("unknown", factual.status)
        self.assertEqual("not_applicable", conversational.status)
        self.assertEqual("not_applicable", exploratory.status)
        self.assertEqual("not_applicable", conversational.planner_origin)

    def test_alternative_and_joint_semantics_are_frozen_and_joint_fails_closed(self):
        joint_clauses = ["joint-clause-a", "joint-clause-b"]
        query_refs = ["query-alt"]
        resolution = resolve_authoritative_requirements(
            "显式需求",
            QueryIntent(),
            planner_origin="explicit",
            requirement_specs=[
                {
                    "slot_id": "alternative-slot",
                    "question": "任选其一的证据",
                    "query_refs": query_refs,
                    "clause_ids": ["alternative-clause"],
                    "origin": "user_explicit",
                    "support_mode": "alternative",
                },
                {
                    "slot_id": "joint-slot",
                    "question": "必须共同成立的证据",
                    "query_refs": ["query-joint"],
                    "clause_ids": joint_clauses,
                    "origin": "user_explicit",
                    "support_mode": "joint",
                },
            ],
        )
        query_refs.append("mutated-after-resolution")
        joint_clauses.append("mutated-after-resolution")
        alternative, joint = resolution.requirements

        self.assertEqual(("query-alt",), alternative.query_refs)
        self.assertEqual(("joint-clause-a", "joint-clause-b"), joint.clause_ids)
        with self.assertRaises(FrozenInstanceError):
            joint.support_mode = "alternative"  # type: ignore[misc]

        support = requirement_support_from_records(
            resolution.requirements,
            [
                {"query": "任选其一的证据", "episode_ids": [1]},
                {
                    "query": "必须共同成立的证据",
                    "episode_ids": [2],
                    "clause_ids": ["joint-clause-a"],
                },
            ],
        )
        self.assertEqual({alternative.slot_id}, support[1])
        self.assertNotIn(2, support)
        complete_support = requirement_support_from_records(
            resolution.requirements,
            [
                {
                    "query": "必须共同成立的证据",
                    "episode_ids": [2],
                    "clause_ids": ["joint-clause-a", "joint-clause-b"],
                }
            ],
        )
        self.assertEqual({joint.slot_id}, complete_support[2])

    def test_production_resolver_has_no_evaluation_gold_input(self):
        with self.assertRaisesRegex(ValueError, "origin"):
            resolve_authoritative_requirements(
                "问题",
                QueryIntent(),
                requirement_specs=[
                    {
                        "slot_id": "bad-origin",
                        "question": "不应进入生产路径的评测输入",
                        "origin": "offline_gold",
                    }
                ],
            )
        source = Path(
            "src/memory_demo/retrieval/query_planning.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("offline_gold", source)

    def test_trace_payload_hashes_all_requirement_text(self):
        raw_question = "私有问题：谁在何时否定了该说法？"
        resolution = resolve_authoritative_requirements(
            raw_question,
            QueryIntent(
                target_entities=["私有人物"],
                search_queries=["私有原子需求"],
                requested_relation="私有关系",
                temporal_constraint="私有时间",
            ),
            planner_origin="local",
        )
        payload = resolution.as_trace_payload()
        serialized = json.dumps(payload, ensure_ascii=False)

        self.assertNotIn(raw_question, serialized)
        self.assertNotIn("私有原子需求", serialized)
        self.assertNotIn("私有人物", serialized)
        requirement = payload["requirements"][0]
        self.assertTrue(requirement["question"].startswith("sha256:"))
        self.assertTrue(requirement["relation_hint"].startswith("sha256:"))
        self.assertTrue(requirement["query_refs"][0].startswith("sha256:"))
        _assert_trace_safe(payload)


if __name__ == "__main__":
    unittest.main()
