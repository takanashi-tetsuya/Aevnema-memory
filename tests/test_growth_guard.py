from __future__ import annotations

import json
import unittest

from memory_demo.associations.growth import AssociationGrowthEngine
from memory_demo.config import WeightConfig


class GrowthModel:
    def chat_json(self, _system, _prompt):
        if "增长关系证据审计器" in _system:
            return {
                "reviews": [
                    {"index": index, "accept": True, "reason": "证据支持"}
                    for index in range(4)
                ]
            }
        common = {"polarity": 1, "llm_score": 0.8, "confidence": 0.8}
        return {
            "relationships": [
                {
                    **common,
                    "from_type": "episode",
                    "from_id": 20,
                    "to_type": "episode",
                    "to_id": 10,
                    "relation_type": "temporal",
                    "relation_key": "before",
                    "relation_text": "错误地声称后续文件发生在前面。",
                },
                {
                    **common,
                    "from_type": "episode",
                    "from_id": 20,
                    "to_type": "concept",
                    "to_id": 1,
                    "relation_type": "identity",
                    "relation_key": "same_as",
                    "relation_text": "错误的跨类型身份边。",
                },
                {
                    **common,
                    "from_type": "episode",
                    "from_id": 20,
                    "to_type": "episode",
                    "to_id": 10,
                    "relation_type": "temporal",
                    "relation_key": "after",
                    "relation_text": "后续文件发生在前面文件之后。",
                },
                {
                    **common,
                    "from_type": "episode",
                    "from_id": 10,
                    "to_type": "episode",
                    "to_id": 20,
                    "relation_type": "temporal",
                    "relation_key": "before",
                    "relation_text": "与上一条等价，不应重复强化。",
                },
            ]
        }


class GrowthRepository:
    def __init__(self):
        self.drafts = []

    def upsert(self, draft):
        self.drafts.append(draft)
        return len(self.drafts)

    def find_exact_id(self, _draft):
        return None


class RoleViolationModel:
    def chat_json(self, system, _prompt):
        if "增长关系证据审计器" in system:
            return {
                "reviews": [
                    {
                        "index": 0,
                        "accept": False,
                        "reason": "节点只说真琴引出茶会成员，不证明真琴属于茶会",
                    }
                ]
            }
        return {
            "relationships": [
                {
                    "from_type": "episode",
                    "from_id": 10,
                    "to_type": "episode",
                    "to_id": 20,
                    "relation_type": "semantic",
                    "relation_key": "deception_enables_attack",
                    "relation_text": "查询综合推论：真琴作为茶会成员协助了袭击。",
                    "polarity": 1,
                    "llm_score": 0.8,
                    "confidence": 0.7,
                }
            ]
        }


class SplitAuditModel:
    def __init__(self):
        self.audit_calls = 0

    def chat_json(self, system, _prompt):
        if "增长关系证据审计器" in system:
            self.audit_calls += 1
            adversarial = "对抗复核" in system
            return {
                "reviews": [
                    {
                        "index": 0,
                        "accept": not adversarial,
                        "reason": (
                            "第一次审计误判为可接受"
                            if not adversarial
                            else "端点只说真琴知道茶会内乱，不证明她是茶会高层"
                        ),
                    }
                ]
            }
        return {
            "relationships": [
                {
                    "from_type": "episode",
                    "from_id": 10,
                    "to_type": "episode",
                    "to_id": 20,
                    "relation_type": "semantic",
                    "relation_key": "political_precondition",
                    "relation_text": "查询综合推论：真琴（茶会高层）帮助了袭击。",
                    "polarity": 1,
                    "llm_score": 0.8,
                    "confidence": 0.7,
                }
            ]
        }


class AcceptedHistoricalModel:
    def __init__(self, relation_key="historical_support_context", relation_text=None):
        self.relation_key = relation_key
        self.relation_text = relation_text or (
            "查询综合推论：节点10证明未花曾支援阿里乌斯；节点20证明阿里乌斯后来参与事件；"
            "两者建立历史联系，但没有证据证明该支援延续到或促成本次事件。"
        )

    def chat_json(self, system, _prompt):
        if "增长关系证据审计器" in system:
            return {
                "reviews": [
                    {"index": 0, "accept": True, "reason": "端点证据支持受限关系"}
                ]
            }
        return {
            "relationships": [
                {
                    "from_type": "episode",
                    "from_id": 10,
                    "to_type": "episode",
                    "to_id": 20,
                    "relation_type": "semantic",
                    "relation_key": self.relation_key,
                    "relation_text": self.relation_text,
                    "polarity": 1,
                    "llm_score": 0.8,
                    "confidence": 0.6,
                }
            ]
        }


class PremiseGrowthModel:
    def __init__(self, premise_id=42):
        self.premise_id = premise_id

    def chat_json(self, system, _prompt):
        if "增长关系证据审计器" in system:
            return {
                "reviews": [
                    {
                        "index": 0,
                        "accept": True,
                        "reason": "端点和显式前提共同支持这条推论",
                    }
                ]
            }
        return {
            "relationships": [
                {
                    "from_type": "episode",
                    "from_id": 10,
                    "to_type": "episode",
                    "to_id": 20,
                    "relation_type": "semantic",
                    "relation_key": "inference_chain_bridge",
                    "relation_text": "查询综合推论：已有桥接关系与节点20共同支持新的联想。",
                    "polarity": 1,
                    "llm_score": 0.8,
                    "confidence": 0.72,
                    "premise_association_ids": [self.premise_id],
                }
            ]
        }


class GrowthEpisodes:
    def __init__(self):
        self.rows = {
            10: {
                "id": 10,
                "source_key": "main/33200.json",
                "text": "未花以共同敌人为理由拉拢并支援阿里乌斯。",
                "segment_index": 0,
                "timeline_scope": "main",
                "story_time_text": "",
                "evidence_origin": "source",
                "epistemic_status": "asserted",
                "generation": 0,
                "epistemic_note": "",
            },
            20: {
                "id": 20,
                "source_key": "main/33210.json",
                "text": "阿里乌斯后来采取了超出未花原计划的行动。",
                "segment_index": 0,
                "timeline_scope": "main",
                "story_time_text": "",
                "evidence_origin": "source",
                "epistemic_status": "asserted",
                "generation": 0,
                "epistemic_note": "",
            },
        }

    def get(self, episode_id):
        return self.rows.get(episode_id)

    def get_many(self, episode_ids):
        return [self.rows[value] for value in episode_ids if value in self.rows]


class GrowthGuardTests(unittest.TestCase):
    def test_direct_candidate_audit_keeps_exact_endpoints(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(), repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.audit_candidates(
            [
                {
                    "claim": (
                        "未花以共同敌人为理由拉拢阿里乌斯，但阿里乌斯后来采取了"
                        "超出其原计划的行动，两者形成控制边界上的对照。"
                    ),
                    "premise_episode_ids": [10, 20],
                    "inference_type": "contrast",
                    "confidence": 0.86,
                }
            ]
        )

        self.assertTrue(outcome.changed)
        self.assertEqual(1, len(repository.drafts))
        draft = repository.drafts[0]
        self.assertEqual((draft.from_id, draft.to_id), (10, 20))
        self.assertEqual("thematic_contrast", draft.relation_key)
        self.assertEqual("dual_accepted", draft.audit_status)
        self.assertEqual(1, draft.generation)

    def test_direct_identity_candidate_records_identity_evidence(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(), repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.audit_candidates(
            [
                {
                    "claim": "日富美曾以黑市代号浮士德召集便利屋68。",
                    "premise_episode_ids": [10, 20],
                    "inference_type": "identity",
                    "confidence": 0.94,
                }
            ]
        )

        self.assertTrue(outcome.changed)
        self.assertEqual(1, len(repository.drafts))
        draft = repository.drafts[0]
        self.assertEqual((draft.from_id, draft.to_id), (10, 20))
        self.assertEqual("semantic", draft.relation_type)
        self.assertEqual("identity_evidence", draft.relation_key)
        self.assertEqual("dual_accepted", draft.audit_status)
        self.assertEqual(1, draft.generation)

    def test_direct_evidence_bridge_keeps_retrieval_only_semantics(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(), repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.audit_candidates(
            [
                {
                    "claim": (
                        "节点10记录一项合作，节点20记录后续行动；此边只用于"
                        "共同检索两项观察，不断言合作导致或指挥了行动。"
                    ),
                    "premise_episode_ids": [10, 20],
                    "inference_type": "evidence_bridge",
                    "confidence": 0.82,
                }
            ]
        )

        self.assertTrue(outcome.changed)
        draft = repository.drafts[0]
        self.assertEqual("semantic", draft.relation_type)
        self.assertEqual("evidence_bridge", draft.relation_key)
        self.assertEqual("supported_inference", draft.claim_level)
        self.assertEqual(1, draft.generation)

    def test_direct_candidate_still_requires_adversarial_acceptance(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            SplitAuditModel(), repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.audit_candidates(
            [
                {
                    "claim": "前一节点为后一节点提供了政治条件。",
                    "premise_episode_ids": [10, 20],
                    "inference_type": "causal",
                    "confidence": 0.86,
                }
            ]
        )

        self.assertFalse(outcome.changed)
        self.assertEqual([], repository.drafts)

    def test_rejects_invalid_growth_and_deduplicates_canonical_temporal_edge(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            GrowthModel(), repository, GrowthEpisodes(), WeightConfig()
        )
        outcome = engine.grow(
            "测试问题",
            [
                {"type": "episode", "id": 10},
                {"type": "episode", "id": 20},
                {"type": "concept", "id": 1},
            ],
            [],
            set(),
        )
        self.assertEqual(outcome.created_ids, [1])
        self.assertEqual(outcome.reinforced_ids, [])
        self.assertTrue(outcome.changed)
        self.assertEqual(len(repository.drafts), 1)
        draft = repository.drafts[0]
        self.assertEqual((draft.from_id, draft.to_id), (10, 20))
        self.assertEqual(draft.relation_key, "before")

    def test_growth_evidence_audit_rejects_role_attribution_violation(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            RoleViolationModel(), repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.grow(
            "哪位茶会高层协助了袭击？",
            [
                {
                    "type": "episode",
                    "id": 10,
                    "text": "真琴同意条约只是为了引出茶会成员。",
                },
                {
                    "type": "episode",
                    "id": 20,
                    "text": "阿里乌斯执行了袭击。",
                },
            ],
            [],
            set(),
        )

        self.assertFalse(outcome.changed)
        self.assertEqual(repository.drafts, [])

    def test_deterministic_role_guard_requires_explicit_role_evidence(self):
        invalid = {
            "from_type": "episode",
            "from_id": 1490,
            "to_type": "episode",
            "to_id": 1595,
            "relation_text": (
                "查询综合推论：节点1490中真琴（茶会高层，格黑娜万魔殿成员）"
                "知道政变计划；节点1595证明阿里乌斯执行袭击。"
            ),
        }
        invalid_nodes = {
            (
                "episode",
                1490,
            ): {
                "text": "据真琴自述，她知道茶会内乱，并与阿里乌斯串通。"
            },
            ("episode", 1595): {"text": "纱织承认阿里乌斯发射导弹。"},
        }
        valid = {
            **invalid,
            "from_id": 1168,
            "relation_text": "查询综合推论：未花（茶会高层）发动了内部政变。",
        }
        valid_nodes = {
            ("episode", 1168): {
                "text": "未花说最终要让小渚下台，由自己做茶会的话事人。"
            },
            ("episode", 1595): {"text": "纱织承认阿里乌斯发射导弹。"},
        }

        rejection = AssociationGrowthEngine._unsupported_role_claim(
            invalid, invalid_nodes
        )

        self.assertIn("茶会高层", rejection)
        self.assertIn(
            "茶会高层",
            AssociationGrowthEngine._unsupported_role_claim(valid, valid_nodes),
        )
        host_evidence = {
            **valid,
            "from_id": 1360,
        }
        host_nodes = {
            ("episode", 1360): {
                "text": "最终，未花为了成为茶会的主持，与阿里乌斯联手了。"
            },
            ("episode", 1595): {"text": "纱织承认阿里乌斯发射导弹。"},
        }
        self.assertIn(
            "茶会高层",
            AssociationGrowthEngine._unsupported_role_claim(host_evidence, host_nodes),
        )
        narrative_role = {
            **valid,
            "from_id": 958,
            "to_id": 1724,
            "relation_text": (
                "正义实现部成员在爆炸后指责格黑娜发射导弹，"
                "与事前戒严形成主题对照。"
            ),
        }
        narrative_nodes = {
            ("episode", 958): {
                "text": "茶会派遣正义实现委员会全体成员保护考试大楼。"
            },
            ("episode", 1724): {
                "text": (
                    "一名正义实现部（Justice Task Force）成员指责"
                    "格黑娜向会场发射导弹。"
                )
            },
        }
        self.assertIsNone(
            AssociationGrowthEngine._unsupported_role_claim(
                narrative_role, narrative_nodes
            )
        )
        copular_invalid = {
            **invalid,
            "relation_text": "真琴是茶会高层，并知道政变计划。",
        }
        self.assertIn(
            "茶会高层",
            AssociationGrowthEngine._unsupported_role_claim(
                copular_invalid, invalid_nodes
            ),
        )

    def test_growth_requires_primary_and_adversarial_audits_to_agree(self):
        repository = GrowthRepository()
        model = SplitAuditModel()
        engine = AssociationGrowthEngine(
            model, repository, GrowthEpisodes(), WeightConfig()
        )

        outcome = engine.grow(
            "哪位茶会高层协助了袭击？",
            [
                {
                    "type": "episode",
                    "id": 10,
                    "text": "真琴知道茶会内乱，并与阿里乌斯串通。",
                },
                {
                    "type": "episode",
                    "id": 20,
                    "text": "阿里乌斯执行了袭击。",
                },
            ],
            [],
            set(),
        )

        self.assertEqual(model.audit_calls, 2)
        self.assertFalse(outcome.changed)
        self.assertEqual(repository.drafts, [])

    def test_safe_historical_context_keeps_provenance(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(), repository, GrowthEpisodes(), WeightConfig()
        )
        nodes = [
            {
                "type": "episode",
                "id": 10,
                "source_key": "main/32170.json",
                "text": "未花承认自己曾暗中支援阿里乌斯。",
            },
            {
                "type": "episode",
                "id": 20,
                "source_key": "main/33070.json",
                "text": "阿里乌斯后来在古圣堂现场出现。",
            },
        ]

        outcome = engine.grow("两段记忆有什么安全的历史联系？", nodes, [], set())

        self.assertTrue(outcome.changed)
        draft = repository.drafts[0]
        self.assertEqual(draft.claim_level, "supported_inference")
        self.assertEqual(draft.audit_status, "dual_accepted")
        self.assertEqual(
            [item["id"] for item in json.loads(draft.evidence_json)],
            [10, 20],
        )
        audit = json.loads(draft.audit_json)[0]
        self.assertTrue(audit["primary_accept"])
        self.assertTrue(audit["adversarial_accept"])

    def test_relation_key_does_not_trigger_a_domain_specific_guard(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(
                relation_key="political_precondition",
                relation_text="查询综合推论：较早支援是后来行动的政治前置条件。",
            ),
            repository,
            GrowthEpisodes(),
            WeightConfig(),
        )
        nodes = [
            {
                "type": "episode",
                "id": 10,
                "source_key": "main/32170.json",
                "text": "未花曾支援阿里乌斯。",
            },
            {
                "type": "episode",
                "id": 20,
                "source_key": "main/33070.json",
                "text": "阿里乌斯后来参与袭击。",
            },
        ]

        outcome = engine.grow("谁协助了后来袭击？", nodes, [], set())

        self.assertTrue(outcome.changed)
        self.assertEqual(repository.drafts[0].claim_level, "supported_inference")

    def test_qualified_cross_source_inference_is_accepted_at_generation_one(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            AcceptedHistoricalModel(
                relation_key="political_precondition",
                relation_text=(
                    "查询综合推论：较早支援可能为后续行动铺路，"
                    "但并不直接证明本次具体机制。"
                ),
            ),
            repository,
            GrowthEpisodes(),
            WeightConfig(),
        )
        nodes = [
            {
                "type": "episode",
                "id": 10,
                "source_key": "main/32170.json",
                "text": "未花曾支援阿里乌斯。",
            },
            {
                "type": "episode",
                "id": 20,
                "source_key": "main/33070.json",
                "text": "阿里乌斯后来参与袭击。",
            },
        ]

        outcome = engine.grow("谁协助了后来袭击？", nodes, [], set())

        self.assertTrue(outcome.changed)
        self.assertEqual(repository.drafts[0].generation, 1)
        self.assertEqual(repository.drafts[0].claim_level, "supported_inference")

    def test_generation_uses_longest_visible_inference_premise(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            PremiseGrowthModel(), repository, GrowthEpisodes(), WeightConfig()
        )
        nodes = [
            {"type": "episode", "id": 10, "text": "直接经验A。"},
            {"type": "episode", "id": 20, "text": "直接经验B。"},
        ]
        edges = [
            {
                "association_id": 42,
                "from": ["episode", 10],
                "to": ["episode", 20],
                "relation_type": "semantic",
                "relation_key": "earlier_inference",
                "relation_text": "此前的二层推论。",
                "generation": 2,
            }
        ]

        outcome = engine.grow("继续沿已有推论联想。", nodes, edges, set())

        self.assertTrue(outcome.changed)
        draft = repository.drafts[0]
        self.assertEqual(draft.generation, 3)
        evidence = json.loads(draft.evidence_json)
        premise = next(item for item in evidence if item["type"] == "association")
        self.assertEqual(premise["id"], 42)
        self.assertEqual(premise["generation"], 2)
        audit = json.loads(draft.audit_json)[0]
        self.assertEqual(audit["premise_association_ids"], [42])
        self.assertEqual(audit["generation"], 3)

    def test_invisible_inference_premise_is_rejected(self):
        repository = GrowthRepository()
        engine = AssociationGrowthEngine(
            PremiseGrowthModel(999), repository, GrowthEpisodes(), WeightConfig()
        )
        outcome = engine.grow(
            "继续沿已有推论联想。",
            [
                {"type": "episode", "id": 10, "text": "直接经验A。"},
                {"type": "episode", "id": 20, "text": "直接经验B。"},
            ],
            [],
            set(),
        )

        self.assertFalse(outcome.changed)
        self.assertEqual(repository.drafts, [])


if __name__ == "__main__":
    unittest.main()
