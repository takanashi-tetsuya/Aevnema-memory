from __future__ import annotations

import unittest

from memory_demo.associations.builder import AssociationBuilder

from memory_demo.llm.validation import (
    parse_concept_batch_payload,
    parse_concept_payload,
    parse_concept_relation_batch_payload,
    parse_episode_relation_batch_payload,
    parse_relationships,
)
from memory_demo.types import ConceptDraft, EpisodeDraft


class ValidationCompatibilityTests(unittest.TestCase):
    def test_batch_concepts_and_relations_preserve_group_identity(self):
        groups, group_errors, errors = parse_concept_batch_payload(
            {
                "episode_concepts": [
                    {
                        "episode_index": 0,
                        "concepts": [
                            {
                                "canonical_name": "日奈",
                                "description": "人物",
                                "embedding_text": "日奈，人物",
                                "aliases": [],
                                "confidence": 0.9,
                            }
                        ],
                    },
                    {"episode_index": 1, "concepts": []},
                ]
            },
            2,
        )
        self.assertFalse(group_errors)
        self.assertFalse(errors)
        self.assertEqual(groups[0][0].canonical_name, "日奈")
        self.assertEqual(groups[1], [])

        relations, relation_errors, errors = parse_episode_relation_batch_payload(
            {
                "episode_relationships": [
                    {"current_id": 2, "relationships": []},
                    {"current_id": 3, "relationships": []},
                ]
            },
            {2, 3},
        )
        self.assertFalse(relation_errors)
        self.assertFalse(errors)
        self.assertEqual(set(relations), {2, 3})

        concept_relations, relation_errors, errors = (
            parse_concept_relation_batch_payload(
                {
                    "concept_relationships": [
                        {"current_id": 10, "relationships": []}
                    ]
                },
                {10},
            )
        )
        self.assertFalse(relation_errors)
        self.assertFalse(errors)
        self.assertEqual(concept_relations[10], [])

    def test_episode_accepts_json_encoded_participant_array(self):
        draft = EpisodeDraft.from_dict(
            {"text": "事件", "participants": '["阿洛娜", "[USERNAME]"]'}
        )
        self.assertEqual(draft.participants, ["阿洛娜", "[USERNAME]"])

    def test_concept_accepts_pair_aliases(self):
        draft = ConceptDraft.from_dict(
            {
                "canonical_name": "老师",
                "description": "身份",
                "embedding_text": "老师，身份",
                "aliases": [["Sensei", "en"]],
            }
        )
        self.assertEqual(draft.aliases, [("Sensei", "en")])

    def test_concept_alias_chain_uses_one_canonical_name(self):
        draft = ConceptDraft.from_dict(
            {
                "canonical_name": "花子 / ハナコ / Hanako",
                "description": "补课部成员",
                "embedding_text": "花子，补课部成员",
                "aliases": [{"alias": "하나코", "language": "ko"}],
            }
        )
        self.assertEqual(draft.canonical_name, "花子")
        self.assertEqual(
            draft.aliases,
            [("ハナコ", "unknown"), ("Hanako", "unknown"), ("하나코", "ko")],
        )

    def test_concept_meta_commentary_is_rejected_for_retry(self):
        concepts, errors = parse_concept_payload(
            {
                "concepts": [
                    {
                        "canonical_name": "佩罗罗奇诺",
                        "description": "Episode 1 未提及该物品，此处修正：应提取其他概念",
                        "embedding_text": "佩罗罗奇诺",
                        "aliases": [],
                    }
                ]
            }
        )
        self.assertEqual(concepts, [])
        self.assertTrue(any("meta-commentary" in error for error in errors))

    def test_relation_accepts_bare_list_and_recovers_coarse_type(self):
        parsed, errors = parse_relationships(
            [
                {
                    "candidate_id": 7,
                    "relation_key": "identity/same_as_candidate",
                    "relation_text": "可能是同一概念",
                    "polarity": 1,
                },
                {
                    "candidate_id": 8,
                    "relation_key": "semantic",
                    "relation_text": "存在语义联系",
                },
            ]
        )
        self.assertFalse(errors)
        self.assertEqual(parsed[0]["relation_type"], "identity")
        self.assertEqual(parsed[0]["relation_key"], "same_as_candidate")
        self.assertEqual(parsed[1]["relation_type"], "semantic")
        self.assertEqual(parsed[1]["relation_key"], "related_to")

        temporal, errors = parse_relationships(
            {
                "relationships": [
                    {
                        "candidate_id": 9,
                        "relation_type": "temporal",
                        "relation_key": "follows_after",
                        "relation_text": "当前事件发生在候选事件之后",
                    }
                ]
            }
        )
        self.assertFalse(errors)
        self.assertEqual(temporal[0]["relation_key"], "after")

        invalid_temporal, errors = parse_relationships(
            {
                "relationships": [
                    {
                        "candidate_id": 10,
                        "relation_type": "temporal",
                        "relation_key": "same_time",
                        "relation_text": "两个条目处于同一时期",
                    }
                ]
            }
        )
        self.assertEqual(invalid_temporal, [])
        self.assertIn("must express before or after", errors[0])

    def test_episode_identity_guard_rejects_person_alias_but_keeps_same_event(self):
        person_alias = {
            "relation_type": "identity",
            "relation_key": "same_as_candidate",
            "relation_text": "两段中的白子与日富美指向同一人物浮士德。",
            "polarity": 1,
        }
        same_event = {
            "relation_type": "identity",
            "relation_key": "same_event_detailed",
            "relation_text": "这是对同一事件的更详细描述。",
            "polarity": 1,
        }
        same_core_fact = {
            "relation_type": "identity",
            "relation_key": "same_as_candidate",
            "relation_text": "两段描述的是不同人物合作的同一核心事实。",
            "polarity": 1,
        }
        self.assertIsNotNone(
            AssociationBuilder.episode_relationship_rejection_reason(person_alias)
        )
        self.assertIsNone(
            AssociationBuilder.episode_relationship_rejection_reason(same_event)
        )
        self.assertIsNotNone(
            AssociationBuilder.episode_relationship_rejection_reason(same_core_fact)
        )

    def test_recall_trigger_requires_explicit_memory_language_in_episode(self):
        relation = {
            "relation_type": "recall_trigger",
            "relation_key": "concept_triggered_memory",
            "relation_text": "引用旧概念触发回忆。",
            "polarity": 1,
        }
        ordinary = {"text": "角色引用了第五条公案来进行类比。"}
        remembered = {"text": "白子因眼前的场景想起了过去被救助的经历。"}
        self.assertIsNotNone(
            AssociationBuilder.episode_relationship_rejection_reason(
                relation, ordinary, ordinary
            )
        )
        self.assertIsNone(
            AssociationBuilder.episode_relationship_rejection_reason(
                relation, ordinary, remembered
            )
        )

    def test_temporal_guard_uses_provenance_and_canonicalizes_direction(self):
        current = {
            "id": 20,
            "source_key": "main/33210.json",
            "segment_index": 0,
            "timeline_scope": "main",
            "story_time_text": "",
        }
        candidate = {
            "id": 10,
            "source_key": "main/33200.json",
            "segment_index": 0,
            "timeline_scope": "main",
            "story_time_text": "",
        }
        contradictory = {
            "relation_type": "temporal",
            "relation_key": "before",
            "relation_text": "当前事件发生在候选之前。",
            "polarity": 1,
        }
        valid = {**contradictory, "relation_key": "after"}
        self.assertIsNotNone(
            AssociationBuilder.episode_relationship_rejection_reason(
                contradictory, current, candidate
            )
        )
        self.assertIsNone(
            AssociationBuilder.episode_relationship_rejection_reason(
                valid, current, candidate
            )
        )
        self.assertEqual(
            AssociationBuilder.normalize_episode_relationship(20, 10, valid),
            (10, 20, "before"),
        )


if __name__ == "__main__":
    unittest.main()
