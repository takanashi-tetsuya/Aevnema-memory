from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest

import numpy as np

from memory_demo.config import AppConfig
from memory_demo.associations.traversal import TraversedNode
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine
from memory_demo.retrieval.query_planning import (
    expand_rerank_atomic_queries,
    limit_rerank_atomic_queries,
    structural_queries,
)
from memory_demo.types import AssociationDraft, EpisodeDraft, QueryIntent

from tests.helpers import FakeModel, test_config


class ConcurrentFakeModel(FakeModel):
    def __init__(self, dimension: int = 8):
        super().__init__(dimension)
        self._active_lock = threading.Lock()
        self.active_episode_calls = 0
        self.max_active_episode_calls = 0
        self.batch_relation_calls = 0

    def chat_json(self, system: str, user: str, **kwargs):
        if "事实提取器" in system:
            with self._active_lock:
                self.active_episode_calls += 1
                self.max_active_episode_calls = max(
                    self.max_active_episode_calls, self.active_episode_calls
                )
            try:
                # Leave enough overlap to make the concurrency assertion stable
                # even while a real evaluation process is using the same CPU.
                time.sleep(0.2)
                return {
                    "episodes": [
                        {
                            "text": "阿洛娜与老师在当前场景中交谈。",
                            "participants": ["阿洛娜", "老师"],
                            "event_type": "交谈",
                            "location_text": "",
                            "story_time_text": "",
                            "timeline_scope": "main",
                            "confidence": 0.9,
                        }
                    ]
                }
            finally:
                with self._active_lock:
                    self.active_episode_calls -= 1
        if "关系判断器" in system and "episode_relationships" in user:
            self.batch_relation_calls += 1
            episode_ids = {
                int(value)
                for value in re.findall(r'"current"\s*:\s*\{\s*"id"\s*:\s*(\d+)', user)
            }
            return {
                "episode_relationships": [
                    {"episode_id": episode_id, "relationships": []}
                    for episode_id in sorted(episode_ids)
                ]
            }
        return super().chat_json(system, user, **kwargs)


class OneTimeConceptFailureModel(FakeModel):
    def __init__(self, dimension: int = 8):
        super().__init__(dimension)
        self.failed_once = False

    def chat_json(self, system: str, user: str, **kwargs):
        if "Concept 提取器" in system and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("temporary malformed concept response")
        return super().chat_json(system, user, **kwargs)


class AlwaysFailFactModel(FakeModel):
    def chat_json(self, system: str, user: str, **kwargs):
        if "事实提取器" in system:
            raise RuntimeError("persistent extraction failure")
        return super().chat_json(system, user, **kwargs)


class InterruptingFactModel(FakeModel):
    def chat_json(self, system: str, user: str, **kwargs):
        if "事实提取器" in system:
            raise KeyboardInterrupt("operator interrupted import")
        return super().chat_json(system, user, **kwargs)


class SinglePassFakeModel(FakeModel):
    def __init__(self, dimension: int = 8):
        super().__init__(dimension)
        self.second_pass_calls = 0

    def chat_json(self, system: str, user: str, **kwargs):
        if "忠实事件提取器" in system:
            if "局部缺口补抽" in user:
                ranges = [
                    (int(start), int(end))
                    for start, end in re.findall(
                        r"L(\d+)-L(\d+)", user.split("SOURCE WITH LINE IDS:", 1)[0]
                    )
                ]
                return {
                    "episodes": [
                        {
                            "text": "??? 向 Morgan 批准了补充请求。",
                            "participants": ["???", "Morgan"],
                            "event_type": "approval supplement",
                            "confidence": 0.9,
                            "evidence_spans": [[start, end]],
                        }
                        for start, end in ranges
                    ]
                }
            evidence_line = re.search(
                r"\[L(\d+)\][^\n]*Morgan received approval", user
            )
            if evidence_line is None:
                raise AssertionError("numbered evidence line was not provided")
            line_number = int(evidence_line.group(1))
            return {
                "episodes": [
                    {
                        "text": "??? 向 Morgan 批准了请求。",
                        "participants": ["???", "Morgan"],
                        "event_type": "approval",
                        "confidence": 0.9,
                        "evidence_spans": [[line_number, line_number]],
                    }
                ]
            }
        if "二次理解器" in system:
            self.second_pass_calls += 1
            raise AssertionError("single-pass profile must skip pass 2")
        if "Concept 提取器" in system:
            indexes = sorted(
                {
                    int(value)
                    for value in re.findall(
                        r'episode_index：\s*(\d+)', user
                    )
                }
            )
            return {
                "episode_concepts": [
                    {"episode_index": index, "concepts": []}
                    for index in indexes
                ]
            }
        return super().chat_json(system, user, **kwargs)


class DocumentMapFakeModel(SinglePassFakeModel):
    def __init__(self, dimension: int = 8):
        super().__init__(dimension)
        self.document_map_calls = 0
        self.extraction_received_map = False

    def chat_json(self, system: str, user: str, **kwargs):
        if "长文档导航地图生成器" in system:
            self.document_map_calls += 1
            evidence_line = re.search(
                r"\[L(\d+)\] [^\\\n]*Morgan received approval", user
            )
            if evidence_line is None:
                raise AssertionError("mapped evidence line was not provided")
            line_number = int(evidence_line.group(1))
            return {
                "overview": "Morgan's approval is the document's main event.",
                "segment_contexts": [
                    {
                        "segment_index": 0,
                        "role_in_document": "approval scene",
                        "event_stages": [
                            {
                                "stage_index": 0,
                                "start_line": line_number,
                                "end_line": line_number,
                                "hint": "an unidentified person approves",
                                "time_mode": "current",
                                "unresolved": [
                                    "approver identity is unknown"
                                ],
                            }
                        ],
                        "participants": ["Morgan", "???"],
                        "timeline_notes": ["current"],
                        "unresolved": ["approver identity is unknown"],
                    }
                ],
            }
        if "最小蕴含审计器" in system:
            indexes = sorted(
                {
                    int(value)
                    for value in re.findall(
                        r'episode_index：\s*(\d+)', user
                    )
                }
            )
            return {
                "reviews": [
                    {
                        "episode_index": index,
                        "verdict": "supported",
                        "unsupported_claims": [],
                        "revised_text": "",
                    }
                    for index in indexes
                ]
            }
        if "忠实事件提取器" in system:
            self.extraction_received_map = (
                self.extraction_received_map
                or "DOCUMENT NAVIGATION" in user
            )
        return super().chat_json(system, user, **kwargs)


class AdaptiveAnchorMapFakeModel(DocumentMapFakeModel):
    def __init__(self, dimension: int = 8):
        super().__init__(dimension)
        self.document_anchor_map_calls = 0

    def chat_json(self, system: str, user: str, **kwargs):
        if "长文档的抽取式导航器" in system:
            self.document_anchor_map_calls += 1
            segments = [
                {"segment_index": int(value)}
                for value in re.findall(
                    r"segment_index：\s*(\d+)", user.split("SEGMENTS:\n", 1)[1]
                )
            ]
            return {
                "segment_anchors": [
                    {
                        "segment_index": int(item["segment_index"]),
                        "role_in_document": (
                            "setup"
                            if position == 0
                            else "continuation"
                        ),
                        "time_mode": "current",
                        "anchor_terms": ["Morgan received approval"],
                        "participants": ["Morgan"],
                    }
                    for position, item in enumerate(segments)
                ]
            }
        return super().chat_json(system, user, **kwargs)


class RevisingAnswerModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.answer_calls = 0

    def chat_text(self, system: str, user: str, **kwargs) -> str:
        if "证据约束的长期记忆回答器" in system:
            self.answer_calls += 1
            return "错误候选人是该组织高层。" if self.answer_calls == 1 else "证据支持的候选人满足身份限定。"
        return super().chat_text(system, user, **kwargs)

    def chat_json(self, system: str, user: str, **kwargs):
        if "答案证据审计器" in system:
            invalid = "错误候选人" in user
            return {
                "valid": not invalid,
                "issues": ([{"claim": "身份", "reason": "Episode 不支持"}] if invalid else []),
                "correction_instructions": (["改用有直接身份证据的人"] if invalid else []),
            }
        return super().chat_json(system, user, **kwargs)


class EventContinuityRevisingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.answer_calls = 0

    def chat_text(self, system: str, user: str, **kwargs) -> str:
        if "证据约束的长期记忆回答器" in system:
            self.answer_calls += 1
            if self.answer_calls == 1:
                return "较早场景的待命命令为后来袭击清空了全部防御。"
            return "较早的长期支援可能构成政治前置条件；本次袭击机制未知。"
        return super().chat_text(system, user, **kwargs)

    def chat_json(self, system: str, user: str, **kwargs):
        if "跨事件因果连续性审计器" in system:
            if "清空了全部防御" in user:
                return {
                    "reviews": [
                        {
                            "claim": "跨事件机制转移",
                            "verdict": "unsupported",
                            "requires_revision": True,
                            "reason": "不同 source_key 未证明同一现场机制",
                        }
                    ],
                    "correction_instructions": ["降级为政治前置条件推论"],
                }
            return {"reviews": [], "correction_instructions": []}
        if "答案证据审计器" in system:
            return {
                "reviews": [
                    {
                        "claim": "存在较早支援",
                        "verdict": "supported_inference",
                        "requires_revision": False,
                        "reason": "端点支持长期支援",
                    }
                ],
                "correction_instructions": [],
            }
        return super().chat_json(system, user, **kwargs)


class PipelineQueryTests(unittest.TestCase):
    def test_late_learned_bridge_keeps_path_when_target_is_already_candidate(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()

        class Associations:
            @staticmethod
            def neighbors(node_type, node_id, limit=24):
                self.assertEqual(node_type, "episode")
                self.assertEqual(limit, 24)
                if node_id != 10:
                    return []
                return [
                    {
                        "id": 99,
                        "from_type": "episode",
                        "from_id": 10,
                        "to_type": "episode",
                        "to_id": 20,
                        "weight": 0.82,
                        "confidence": 0.86,
                        "audit_status": "dual_accepted",
                        "generation": 1,
                        "polarity": 1,
                        "relation_key": "thematic_contrast",
                    }
                ]

        engine.associations = Associations()
        engine._explicit_association_paths = lambda ids: [
            {
                "association_id": ids[0],
                "relation_key": "thematic_contrast",
                "relation_text": "合作设想与实际目标存在失控性反差",
            }
        ]
        traversed = [
            TraversedNode("episode", 10, 0.8, []),
            TraversedNode("episode", 20, 0.1, []),
        ]

        additions, paths = engine._late_learned_bridge_closure(
            "为什么合作设想与实际目标存在失控性反差？",
            [10],
            traversed,
        )

        self.assertEqual([], additions)
        self.assertEqual([99], [path["association_id"] for path in paths])
        self.assertTrue(paths[0]["late_bridge_closure"])

    def test_rerank_precompression_protects_floor_then_keeps_pool_order(self):
        selected, decisions = QueryEngine._precompress_rerank_candidate_ids(
            list(range(1, 11)),
            preferred_ids=[3, 4, 8],
            required_ids=[9, 10],
            limit=5,
        )

        self.assertEqual([9, 10, 1, 2, 3], selected)
        by_id = {item["episode_id"]: item for item in decisions}
        self.assertEqual("required_evidence_floor", by_id[9]["reason"])
        self.assertEqual("atomic_anchor", by_id[3]["reason"])
        self.assertEqual("global_rank", by_id[1]["reason"])
        self.assertEqual("dropped_budget", by_id[8]["reason"])

    def test_rerank_precompression_is_exact_noop_at_pool_size(self):
        pool = [5, 2, 8, 1]

        selected, decisions = QueryEngine._precompress_rerank_candidate_ids(
            pool,
            preferred_ids=[8],
            required_ids=[1],
            limit=len(pool),
        )

        self.assertEqual(pool, selected)
        self.assertTrue(all(item["kept"] for item in decisions))

    def test_llm_rerank_uses_bge_prefilter_before_single_review(self):
        class HybridModel:
            def __init__(self):
                self.rerank_input_count = 0
                self.review_calls = 0

            def rerank(self, query, documents, top_n=None):
                self.rerank_input_count = len(documents)
                return [
                    {"index": index, "relevance_score": float(index)}
                    for index in reversed(range(len(documents)))
                ]

            def chat_json(self, system, user, **kwargs):
                self.review_calls += 1
                return {
                    "selected_episode_ids": [20, 19, 18, 17, 16],
                    "coverage": [],
                    "missing_aspects": [],
                }

        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.model.reranker_model = "bge-test"
        engine.config.retrieval.rerank_enabled = True
        engine.config.retrieval.rerank_backend = "llm"
        engine.config.retrieval.rerank_candidate_limit = 20
        engine.config.retrieval.rerank_precompression_limit = 5
        engine.config.retrieval.rerank_review_mode = "lean"
        engine.config.retrieval.rerank_atomic_query_limit = 4
        engine.config.retrieval.rerank_answer_slot_neighbor_radius = 0
        engine.config.retrieval.rerank_answer_slot_neighbor_total_limit = 0
        engine.model = HybridModel()
        engine.logger = None
        episodes = [
            {
                "id": index,
                "score": 1.0 / index,
                "text": f"证据 {index}",
                "participants": [],
                "source_key": f"source-{index}",
            }
            for index in range(1, 21)
        ]

        selected, trace = engine._rerank_answer_episodes(
            "复杂关系",
            QueryIntent.from_dict({"search_queries": ["复杂关系"]}),
            ["复杂关系"],
            episodes,
            5,
        )

        self.assertEqual(20, engine.model.rerank_input_count)
        self.assertEqual(1, engine.model.review_calls)
        self.assertEqual([20, 19, 18, 17, 16], selected)
        self.assertTrue(trace["bge_prefilter"]["performed"])
        self.assertEqual([20, 19, 18, 17, 16], trace["rerank_input_episode_ids"])
        self.assertEqual("none", trace["review_level"])
        self.assertFalse(trace["coverage_audit_performed"])
        self.assertFalse(trace["compressor_performed"])

    def test_llm_rerank_does_not_pad_a_valid_short_evidence_selection(self):
        class ShortSelectionModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "coverage": [
                        {"query": "narrow relation", "episode_ids": [1]}
                    ],
                    "missing_aspects": [],
                }

        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.rerank_enabled = True
        engine.config.retrieval.rerank_backend = "llm"
        engine.config.retrieval.rerank_review_mode = "lean"
        engine.config.retrieval.rerank_atomic_query_limit = 1
        engine.config.retrieval.rerank_answer_slot_neighbor_radius = 0
        engine.config.retrieval.rerank_answer_slot_neighbor_total_limit = 0
        engine.model = ShortSelectionModel()
        engine.logger = None
        episodes = [
            {
                "id": index,
                "score": 1.0 / index,
                "text": f"evidence {index}",
                "participants": [],
                "source_key": "source",
            }
            for index in range(1, 4)
        ]

        selected, trace = engine._rerank_answer_episodes(
            "narrow relation",
            QueryIntent.from_dict({"search_queries": ["narrow relation"]}),
            ["narrow relation"],
            episodes,
            3,
        )

        self.assertEqual([1], selected)
        self.assertEqual([1], trace["final_episode_ids"])

    def test_rerank_atomic_budget_preserves_special_and_late_queries(self):
        queries = [
            "完整问题",
            *[f"早期查询{index}" for index in range(1, 11)],
            "__answer_slot__ 直接执行者",
            *[f"实体化后续查询{index}" for index in range(1, 8)],
        ]

        limited = limit_rerank_atomic_queries(queries, 12)

        self.assertEqual(12, len(limited))
        self.assertEqual("完整问题", limited[0])
        self.assertIn("__answer_slot__ 直接执行者", limited)
        self.assertIn("早期查询1", limited)
        self.assertIn("实体化后续查询7", limited)

    def test_adaptive_rerank_review_spends_calls_only_on_visible_risk(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        allowed = {1, 2, 3}
        sufficient = {
            "coverage": [
                {"query": "甲", "episode_ids": [1]},
                {"query": "乙", "episode_ids": [2]},
            ],
            "missing_aspects": [],
        }

        self.assertEqual(
            engine._rerank_review_level(["甲", "乙"], sufficient, allowed)[0],
            "none",
        )
        self.assertEqual(
            engine._rerank_review_level(
                ["甲"],
                {"coverage": [{"query": "甲", "episode_ids": [1]}]},
                allowed,
            )[0],
            "compress",
        )
        self.assertEqual(
            engine._rerank_review_level(
                ["甲", "乙"],
                {**sufficient, "missing_aspects": ["丙"]},
                allowed,
            )[0],
            "strict",
        )
        self.assertEqual(
            engine._rerank_review_level(
                [f"槽位{index}" for index in range(24)], sufficient, allowed
            )[0],
            "strict",
        )

    def test_fast_adaptive_trusts_complete_initial_coverage(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.rerank_review_mode = "fast_adaptive"
        sufficient = {
            "coverage": [
                {"query": "爆炸现场", "episode_ids": [1]},
                {"query": "直接执行者", "episode_ids": [2]},
                {"query": "政治前提", "episode_ids": [3]},
            ],
            "missing_aspects": [],
        }

        level, reasons = engine._rerank_review_level(
            ["爆炸现场", "直接执行者", "政治前提"],
            sufficient,
            {1, 2, 3},
            "谁直接执行袭击，与哪位高层暗中支援有什么因果关系？",
        )

        self.assertEqual("none", level)
        self.assertEqual(["initial_coverage_sufficient"], reasons)

    def test_fast_adaptive_preserves_reported_gap_as_uncertainty(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.rerank_review_mode = "fast_adaptive"
        incomplete = {
            "coverage": [
                {"query": "爆炸现场", "episode_ids": [1]},
                {"query": "政治前提", "episode_ids": [2]},
            ],
            "missing_aspects": ["不可由候选证明的具体渗透机制"],
        }

        level, reasons = engine._rerank_review_level(
            ["爆炸现场", "政治前提", "具体渗透机制"],
            incomplete,
            {1, 2},
        )

        self.assertEqual("none", level)
        self.assertIn("initial_reported_missing_aspects", reasons)
        self.assertIn("reported_gap_preserved_as_uncertainty", reasons)

    def test_fast_adaptive_still_compresses_structurally_thin_coverage(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.rerank_review_mode = "fast_adaptive"

        level, reasons = engine._rerank_review_level(
            ["爆炸现场", "直接执行者"],
            {"coverage": [], "missing_aspects": ["直接执行者"]},
            {1, 2},
        )

        self.assertEqual("compress", level)
        self.assertIn("thin_initial_coverage", reasons)
    def test_whole_question_reserves_broader_anchor_budget(self):
        class BatchModel:
            @staticmethod
            def embed(texts):
                return np.asarray(
                    [[1.0, float(index)] for index, _ in enumerate(texts)],
                    dtype=np.float32,
                )

        class SequencedEpisodeIndex:
            def __init__(self):
                self.calls = 0

            def search(self, _vector, _top_k):
                self.calls += 1
                base = 0 if self.calls == 1 else 100
                return [
                    (base + rank, 1.0 - rank / 100.0)
                    for rank in range(1, 13)
                ]

        class EmptyConceptIndex:
            @staticmethod
            def search(_vector, _top_k):
                return []

        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.model.embedding_dimension = 2
        engine.model = BatchModel()
        engine.episode_index = SequencedEpisodeIndex()
        engine.concept_index = EmptyConceptIndex()
        anchors = []

        engine._vector_seed_hits(
            ["完整问题", "原子问题"],
            anchors,
            engine.config.retrieval.answer_whole_question_anchor_episodes,
        )

        self.assertEqual(anchors[:10], list(range(1, 11)))
        self.assertEqual(anchors[10:], list(range(101, 107)))

    def test_followup_anchor_ids_are_interleaved_before_initial_tail(self):
        self.assertEqual(
            QueryEngine._interleave_anchor_ids(
                [201, 202, 203],
                [1, 2, 3, 4, 5],
            ),
            [201, 1, 202, 2, 203, 3, 4, 5],
        )

    def test_entity_resolved_followup_requires_answer_dependent_reference(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.followup_planning_mode = "entity_resolved"
        named_intent = QueryIntent(
            target_entities=["日富美", "梓"],
            search_queries=[
                "日富美如何反驳阿里乌斯的宣言",
                "梓的现场身份是什么",
            ],
        )
        dependent_intent = QueryIntent(
            target_entities=["社团首领"],
            search_queries=[
                "社团首领被谁雇佣",
                "雇主使用什么代号",
                "这名少女喜欢哪个吉祥物",
            ],
        )

        self.assertEqual(
            (False, "intent_queries_already_name_each_relation"),
            engine._followup_planning_decision(
                "请分别说明日富美的反驳和梓的身份。",
                named_intent,
            ),
        )
        self.assertEqual(
            (True, "answer_dependent_reference_detected"),
            engine._followup_planning_decision(
                "她被谁以什么代号雇佣，而雇佣她的这名少女又喜欢哪个吉祥物？",
                dependent_intent,
            ),
        )
        self.assertEqual(
            (True, "high_slot_count_safety_net"),
            engine._followup_planning_decision(
                "请逐项说明十二个独立证据槽。",
                QueryIntent(
                    search_queries=[f"证据槽{index}" for index in range(12)]
                ),
            ),
        )

    def test_missing_slots_followup_uses_only_empty_initial_candidate_slots(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        engine.config.retrieval.followup_planning_mode = "missing_slots"
        queries = ["whole question", "first slot", "second slot"]

        self.assertEqual(
            ["second slot"],
            engine._initial_missing_followup_slots(
                queries,
                {
                    "fused_episode": [[{"id": 1}], [{"id": 2}], []],
                    "atomic_episode": [[], [], []],
                    "sparse_episode": [[], [], []],
                },
            ),
        )
        self.assertEqual(
            [],
            engine._initial_missing_followup_slots(
                queries,
                {
                    "fused_episode": [[{"id": 1}], [{"id": 2}], []],
                    "atomic_episode": [[], [], [{"id": 3}]],
                    "sparse_episode": [[], [], []],
                },
            ),
        )

    def test_query_intent_keeps_twelve_explicit_evidence_slots(self):
        intent = QueryIntent.from_dict(
            {"search_queries": [f"证据槽 {index}" for index in range(14)]}
        )

        self.assertEqual(
            intent.search_queries,
            [f"证据槽 {index}" for index in range(12)],
        )

    def test_structural_queries_do_not_invent_domain_slots(self):
        queries = structural_queries(
            "比较甲乙对阿里乌斯的控制边界与失控迹象",
            QueryIntent(target_entities=["甲", "乙", "阿里乌斯"]),
        )

        self.assertEqual(queries, [])

    def test_quoted_text_adds_generic_answer_slot(self):
        queries = structural_queries(
            "‘憎恨’如何贯穿未花接近阿里乌斯的理由、纱织的宣言？",
            QueryIntent(target_entities=["未花", "阿里乌斯", "纱织"]),
        )

        slots = [
            query for query in queries if query.startswith("__answer_slot__ ")
        ]
        self.assertEqual(1, len(slots))
        self.assertEqual(
            slots[0],
            "__answer_slot__ 原文中“憎恨”对应的事实是什么",
        )
        self.assertFalse(
            QueryEngine._requires_strict_evidence_review(
                "‘憎恨’如何贯穿未花接近阿里乌斯的理由"
            )
        )

    def test_quoted_entity_does_not_trigger_authority_template(self):
        queries = structural_queries(
            "“新的伊甸条约机构”是谁在什么权限解释下宣告的？",
            QueryIntent(target_entities=["新的伊甸条约机构"]),
        )

        slots = [
            query for query in queries if query.startswith("__answer_slot__ ")
        ]
        self.assertEqual(1, len(slots))
        self.assertEqual(
            slots[0],
            "__answer_slot__ 原文中“新的伊甸条约机构”对应的事实是什么",
        )

    def test_story_wording_does_not_create_fixed_endpoint_templates(self):
        queries = structural_queries(
            "串联故事开端/结尾中“老师的义务、支持学生梦想”的表述",
            QueryIntent(target_entities=["老师"]),
        )

        self.assertEqual(
            queries,
            ["__answer_slot__ 原文中“老师的义务、支持学生梦想”对应的事实是什么"],
        )

    def test_negated_or_unproven_quotes_do_not_become_answer_floors(self):
        queries = structural_queries(
            "不要直接回答“真正内心”；并区分“比较事实”和“它直接造成行动”的未经证明因果。",
            QueryIntent(),
        )

        self.assertEqual(
            queries,
            ["__answer_slot__ 原文中“比较事实”对应的事实是什么"],
        )

    def test_comparison_wording_does_not_create_domain_templates(self):
        queries = structural_queries(
            "比较两段剧情：弥奈的顾虑、京剧部事件、未花的行动分别说明什么？",
            QueryIntent(target_entities=["弥奈", "未花"]),
        )

        self.assertEqual(queries, [])

    def test_metaphor_wording_does_not_create_fixed_query(self):
        queries = structural_queries(
            "乐园悖论如何隐喻圣三一内部的信任危机？",
            QueryIntent(target_entities=["乐园悖论", "圣三一"]),
        )

        self.assertEqual(queries, [])

    def test_rerank_atomic_queries_split_residual_full_conjuncts(self):
        queries = expand_rerank_atomic_queries(
            ["梓最初被描述为刚转学、学习跟不上"],
            QueryIntent(target_entities=["梓", "补习部"]),
        )

        self.assertEqual(len(queries), 3)
        self.assertTrue(any("刚转学（独立证据槽" in query for query in queries))
        self.assertTrue(any("学习跟不上（独立证据槽" in query for query in queries))

    def test_rerank_atomic_queries_split_long_explicit_question_list(self):
        query = "、".join(f"必须说明的证据槽{index}" for index in range(6))
        queries = expand_rerank_atomic_queries(
            [query],
            QueryIntent(target_entities=["测试人物"]),
        )

        self.assertEqual(len(queries), 7)
        combined = " ".join(queries)
        self.assertTrue(all(f"证据槽{index}" in combined for index in range(6)))

    def test_rerank_does_not_expand_whole_question_or_structural_slots(self):
        queries = expand_rerank_atomic_queries(
            [
                "整题要求甲、乙、丙",
                "普通子问题甲方、普通子问题乙方",
                "__answer_slot__ 原文中“甲、乙”对应的事实是什么",
            ],
            QueryIntent(target_entities=["测试"]),
            "整题要求甲、乙、丙",
        )

        self.assertEqual(
            queries,
            [
                "整题要求甲、乙、丙",
                "普通子问题甲方、普通子问题乙方",
                "测试：普通子问题甲方（独立证据槽，不能由同列另一条件替代）",
                "测试：普通子问题乙方（独立证据槽，不能由同列另一条件替代）",
                "__answer_slot__ 原文中“甲、乙”对应的事实是什么",
            ],
        )

    def test_rebuttal_wording_keeps_only_quoted_evidence_slot(self):
        queries = structural_queries(
            "某人如何反驳对方关于“努力徒劳”的世界真相？",
            QueryIntent(target_entities=["某人"]),
        )

        self.assertEqual(
            queries,
            ["__answer_slot__ 原文中“努力徒劳”对应的事实是什么"],
        )

    def test_structural_queries_use_only_parsed_intent_constraints(self):
        queries = structural_queries(
            "他们之所以能成功渗透，是因为哪位茶会高层暗中协助？"
            "（提示：与政变有关）",
            QueryIntent(
                requested_relation="执行势力与协助者",
                causal_constraint="袭击成功与高层暗中协助的关系",
            ),
        )

        self.assertEqual(
            queries,
            [
                "__constraint_slot__ 关系约束：执行势力与协助者",
                "__constraint_slot__ 因果约束：袭击成功与高层暗中协助的关系",
            ],
        )

    def test_requested_relation_becomes_generic_constraint_slot(self):
        queries = structural_queries(
            "古圣堂发生爆炸，袭击由哪个分校直接执行？",
            QueryIntent(requested_relation="直接执行者"),
        )

        self.assertEqual(
            queries,
            ["__constraint_slot__ 关系约束：直接执行者"],
        )

    def test_rerank_atomic_queries_do_not_split_classification_choices(self):
        query = "这是角色档案、现场揭露还是戏剧性自我呈现"
        queries = expand_rerank_atomic_queries(
            [query],
            QueryIntent(target_entities=["日富美"]),
        )

        self.assertEqual(queries, [query])

    def test_rerank_atomic_queries_split_paired_subject_ownership(self):
        queries = expand_rerank_atomic_queries(
            ["弥奈与玄龙门成员的顾虑是什么"],
            QueryIntent(target_entities=["弥奈", "玄龙门"]),
        )

        self.assertIn("弥奈本人直接陈述或表现的顾虑是什么", queries)
        self.assertIn("玄龙门成员直接陈述或被转述的顾虑是什么", queries)

    def test_high_risk_causal_question_forces_strict_rerank_review(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        sufficient = {
            "coverage": [
                {"query": "表面设定", "episode_ids": [1]},
                {"query": "真正原因", "episode_ids": [2]},
            ],
            "missing_aspects": [],
        }

        level, reasons = engine._rerank_review_level(
            ["表面设定", "真正原因"],
            sufficient,
            {1, 2},
            "表面说法背后的真正原因和隐藏身份是什么？",
        )

        self.assertEqual(level, "strict")
        self.assertIn("high_risk_contrast_causality_or_projection", reasons)

    def test_high_risk_atomic_floor_round_robins_early_subproblems(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        rankings = [
            [{"id": query * 10 + rank} for rank in range(1, 4)]
            for query in range(1, 9)
        ]

        selected = engine._atomic_evidence_floor_ids(
            "表面说法背后的真正原因是什么？",
            rankings,
        )

        self.assertEqual(selected, [11, 21, 31, 41, 51, 61])

    def test_atomic_floor_reserves_budget_for_resolved_followups(self):
        initial = [[{"id": value}] for value in range(1, 9)]
        followup = [[{"id": value}] for value in range(101, 105)]

        selected = QueryEngine._balanced_atomic_floor_rankings(
            initial,
            followup,
            6,
        )

        self.assertEqual(
            [ranking[0]["id"] for ranking in selected],
            [1, 2, 3, 4, 101, 102],
        )

    def test_selection_floor_replaces_only_non_floor_tail(self):
        selected, changes = QueryEngine._enforce_selection_floor(
            [1, 2, 3, 4],
            [3, 8, 9],
            {1, 2, 3, 4, 8, 9},
            4,
        )

        self.assertEqual(set(selected), {1, 3, 8, 9})
        self.assertEqual(changes, [
            {"add_id": 8, "remove_id": 4},
            {"add_id": 9, "remove_id": 2},
        ])

    def test_answer_slot_floor_keeps_adjacent_source_evidence(self):
        episodes = [
            {"id": 10, "source_key": "main/one.json"},
            {"id": 11, "source_key": "main/one.json"},
            {"id": 12, "source_key": "main/one.json"},
            {"id": 13, "source_key": "main/two.json"},
        ]

        selected = QueryEngine._answer_slot_neighbor_floor_ids(
            [11],
            episodes,
            radius=1,
            total_limit=4,
        )

        self.assertEqual([10, 12], selected)

    def test_hybrid_floor_keeps_sparse_recall_diagnostic_not_mandatory(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        initial = {
            "sparse_episode": [
                [{"id": value} for value in range(100, 120)]
            ],
            "atomic_episode": [
                [{"id": 1}, {"id": 2}],
                [{"id": 3}, {"id": 4}],
            ],
        }

        selected = engine._hybrid_evidence_floor_ids(
            "表面说法背后的真正原因是什么？",
            initial,
            {},
        )

        self.assertEqual(selected, [1, 3])

    def test_cross_constraint_floor_precedes_broad_sparse_floor(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        initial = {
            "sparse_episode": [
                [{"id": value} for value in range(100, 120)],
                [{"id": 1168}, {"id": 1167}, {"id": 1490}],
            ],
            "atomic_episode": [
                [{"id": 1}, {"id": 2}],
                [{"id": 1168}, {"id": 1167}, {"id": 1490}],
            ],
        }
        trace = engine._hybrid_evidence_floor_trace(
            "袭击为何成功，真正原因是什么？",
            initial,
            {},
            ["普通槽", "__constraint_slot__ 政变前置证据"],
            [],
        )

        self.assertEqual(trace["selected_episode_ids"][:1], [1168])
        self.assertEqual(
            trace["constraint_slots"][0]["candidate_episode_ids"],
            [1168],
        )

    def test_explicit_answer_slot_enables_floor_without_other_risk_cues(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        initial = {
            "sparse_episode": [
                [{"id": 300}],
                [{"id": 167}, {"id": 166}],
            ],
            "atomic_episode": [
                [{"id": 300}],
                [{"id": 167}, {"id": 166}],
            ],
        }

        trace = engine._hybrid_evidence_floor_trace(
            "新的机构由谁宣告？",
            initial,
            {},
            ["普通问题", "__answer_slot__ 宣告者与权限依据"],
            [],
        )

        self.assertTrue(trace["enabled"])
        self.assertEqual([167], trace["selected_episode_ids"][:1])
        self.assertEqual(
            [167], trace["constraint_slots"][0]["floor_episode_ids"]
        )

    def test_constraint_candidate_reserve_prioritizes_specific_followups(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        initial = {
            "fused_episode": [[{"id": value} for value in range(100, 140)]],
        }
        followup = {
            "fused_episode": [
                [{"id": 900 + value} for value in range(20)],
                [
                    {"id": 1040},
                    {"id": 1308},
                    {"id": 1167},
                    {"id": 1166},
                    {"id": 1168},
                ],
            ],
        }

        selected = engine._constraint_candidate_ids(
            "结果与哪个前提有关？",
            ["普通背景"],
            initial,
            ["普通后续", "__constraint_slot__ 因果约束：前提与结果的关系"],
            followup,
        )

        self.assertEqual(selected[:5], [1040, 1308, 1167, 1166, 1168])

    def test_final_evidence_slot_trace_records_satisfaction_and_loss(self):
        trace = QueryEngine._final_evidence_slot_trace(
            {
                "deterministic_evidence_floor": {
                    "constraint_slots": [
                        {
                            "query": "政变证据",
                            "floor_episode_ids": [1168, 1167],
                        }
                    ],
                    "selected_episode_ids": [1168, 1167, 1481],
                },
                "merged_coverage": {
                    "coverage": [
                        {
                            "query": "爆炸锚点",
                            "mode": "alternatives",
                            "episode_ids": [1481, 1484],
                        }
                    ]
                },
            },
            [1168, 1481],
        )

        self.assertTrue(trace["deterministic"]["constraint_slots"][0]["satisfied"])
        self.assertEqual(
            trace["deterministic"]["missing_floor_episode_ids"],
            [1167],
        )
        self.assertTrue(trace["coverage_slots"][0]["satisfied"])

    def test_multi_fact_question_gets_per_claim_atomic_coverage_slots(self):
        engine = QueryEngine.__new__(QueryEngine)
        engine.config = AppConfig()
        queries = ["谁直接执行袭击", "哪位高层暗中协助"]
        initial = {
            "sparse_episode": [
                [{"id": 10}],
                [{"id": 20}],
            ],
            "atomic_episode": [
                [{"id": 10}, {"id": 11}],
                [{"id": 20}, {"id": 21}],
            ],
        }

        floor = engine._hybrid_evidence_floor_trace(
            "谁直接执行袭击，哪位高层暗中协助？",
            initial,
            {},
            queries,
            [],
        )
        final = engine._final_evidence_slot_trace(
            {"deterministic_evidence_floor": floor},
            [10, 20],
        )

        self.assertTrue(floor["enabled"])
        self.assertEqual(2, len(floor["atomic_slots"]))
        self.assertTrue(
            all(
                slot["satisfied"]
                for slot in final["deterministic"]["atomic_slots"]
            )
        )

    def test_paired_subject_split_keeps_possessive_predicate(self):
        queries = expand_rerank_atomic_queries(
            ["弥奈与玄龙门成员的顾虑的直接原文事实是什么"],
            QueryIntent(target_entities=["弥奈", "玄龙门"]),
        )

        self.assertIn(
            "弥奈本人直接陈述或表现的顾虑的直接原文事实是什么",
            queries,
        )

    def test_rerank_replacements_are_executed_when_final_ids_drift(self):
        corrected = QueryEngine._apply_rerank_replacements(
            {
                "final_episode_ids": [1, 2, 3],
                "replacements": [
                    {"remove_id": 2, "add_id": 4},
                    {"remove_id": 99, "add_id": 5},
                    {"remove_id": 1, "add_id": 999},
                ],
            },
            [1, 2, 3],
            {1, 2, 3, 4, 5},
            4,
        )

        self.assertEqual(corrected, [1, 4, 3, 5])

    def test_rerank_coverage_shortlist_round_robins_atomic_queries(self):
        ids = QueryEngine._coverage_rerank_ids(
            {
                "coverage": [
                    {"query": "A", "episode_ids": [1, 2, 3]},
                    {"query": "B", "episode_ids": [4, 5]},
                ]
            },
            {1, 2, 3, 4, 5},
            4,
        )

        self.assertEqual(ids, [1, 4, 2, 5])

    def test_joint_coverage_protects_every_complementary_episode(self):
        payload = {
            "coverage": [
                {
                    "query": "刚转学并且学习跟不上",
                    "mode": "joint",
                    "episode_ids": [1, 2],
                },
                {
                    "query": "同一事实的改写",
                    "mode": "alternatives",
                    "episode_ids": [3, 4],
                },
            ]
        }

        self.assertEqual(
            QueryEngine._coverage_rerank_ids(payload, {1, 2, 3, 4}, 4),
            [1, 2, 3, 4],
        )
        selected, _ = QueryEngine._enforce_coverage_selection(
            payload,
            [1, 3, 5],
            {1, 2, 3, 4, 5},
            3,
        )
        self.assertEqual(set(selected), {1, 2, 3})

    def test_independent_coverage_slots_are_interleaved_with_primary_slots(self):
        merged = QueryEngine._merge_coverage_payloads(
            {
                "coverage": [
                    {"query": "已有槽", "episode_ids": [1], "mode": "alternatives"}
                ],
                "missing_aspects": ["仍缺事实"],
            },
            {
                "coverage": [
                    {"query": "补充槽", "episode_ids": [2], "mode": "joint"}
                ],
                "missing_aspects": [],
            },
        )

        self.assertEqual(
            [item["query"] for item in merged["coverage"]],
            ["已有槽", "补充槽"],
        )
        self.assertEqual(merged["coverage"][1]["mode"], "joint")
        self.assertEqual(merged["missing_aspects"], ["仍缺事实"])

    def test_independent_coverage_unions_ids_for_the_same_slot(self):
        merged = QueryEngine._merge_coverage_payloads(
            {
                "coverage": [
                    {"query": "同一槽", "episode_ids": [1], "mode": "alternatives"}
                ]
            },
            {
                "coverage": [
                    {"query": "同一槽", "episode_ids": [2], "mode": "alternatives"}
                ]
            },
        )

        self.assertEqual(len(merged["coverage"]), 1)
        self.assertEqual(merged["coverage"][0]["episode_ids"], [1, 2])

    def test_independent_coverage_merges_parenthetical_mode_disagreement(self):
        merged = QueryEngine._merge_coverage_payloads(
            {
                "coverage": [
                    {"query": "同一槽（具体事实）", "episode_ids": [1], "mode": "joint"}
                ]
            },
            {
                "coverage": [
                    {"query": "同一槽", "episode_ids": [2], "mode": "alternatives"}
                ]
            },
        )

        self.assertEqual(len(merged["coverage"]), 1)
        self.assertEqual(merged["coverage"][0]["mode"], "alternatives")
        self.assertEqual(merged["coverage"][0]["episode_ids"], [1, 2])

    def test_rerank_coverage_enforcement_replaces_only_unprotected_tail(self):
        selected, changes = QueryEngine._enforce_coverage_selection(
            {
                "coverage": [
                    {"query": "A", "episode_ids": [1, 2]},
                    {"query": "B", "episode_ids": [3]},
                    {"query": "C", "episode_ids": [4, 5]},
                ]
            },
            [1, 6, 7],
            {1, 2, 3, 4, 5, 6, 7},
            3,
        )

        self.assertEqual(set(selected), {1, 3, 4})
        self.assertEqual(
            changes,
            [
                {"remove_id": 7, "add_id": 3},
                {"remove_id": 6, "add_id": 4},
            ],
        )

    def test_final_coverage_cannot_be_erased_by_low_precision_floor(self):
        final_ids, floor_changes, coverage_changes = (
            QueryEngine._enforce_final_selection_constraints(
                selected_ids=[1, 2, 3],
                required_ids=[8],
                coverage_payload={
                    "coverage": [
                        {"query": "explicit answer slot", "episode_ids": [3]},
                    ]
                },
                allowed_ids={1, 2, 3, 8},
                coverage_allowed_ids={1, 2, 3, 8},
                limit=3,
            )
        )

        self.assertEqual({1, 2, 3}, set(final_ids))
        self.assertEqual([{"add_id": 8, "remove_id": 3}], floor_changes)
        self.assertEqual([{"remove_id": 8, "add_id": 3}], coverage_changes)

    def test_structured_answer_audit_accepts_qualified_inference(self):
        audit = QueryEngine._normalize_answer_audit(
            {
                "reviews": [
                    {
                        "claim": "支援可能为后续行动创造条件",
                        "verdict": "supported_inference",
                        "requires_revision": True,
                        "reason": "前提有证据且已标注推论",
                    }
                ]
            }
        )

        self.assertTrue(audit["valid"])
        self.assertFalse(audit["reviews"][0]["requires_revision"])
        self.assertEqual(audit["issues"], [])

    def test_structured_answer_audit_rejects_unsupported_claim(self):
        audit = QueryEngine._normalize_answer_audit(
            {
                "reviews": [
                    {
                        "claim": "错误的身份归属",
                        "verdict": "unsupported",
                        "requires_revision": False,
                        "reason": "Episode 不支持",
                    }
                ]
            }
        )

        self.assertFalse(audit["valid"])
        self.assertTrue(audit["reviews"][0]["requires_revision"])

    def test_event_audit_result_is_not_overridden_by_answer_keywords(self):
        audit = QueryEngine._normalize_answer_audit(
            {
                "reviews": [
                    {
                        "claim": "长期支援被误读为具体必要条件",
                        "verdict": "unsupported",
                        "reason": "具体机制没有证据",
                        "audit_scope": "event_continuity",
                    }
                ]
            },
            "长期支援可能构成政治铺路；本次渗透的具体路线和执行机制未知。",
        )

        self.assertFalse(audit["valid"])
        self.assertEqual(audit["reviews"][0]["verdict"], "unsupported")

    def test_event_audit_moderation_keeps_actual_strong_causation_invalid(self):
        audit = QueryEngine._normalize_answer_audit(
            {
                "reviews": [
                    {
                        "claim": "早期命令导致后来渗透",
                        "verdict": "unsupported",
                        "reason": "跨事件机制未证实",
                        "audit_scope": "event_continuity",
                    }
                ]
            },
            "他们之所以成功渗透，是因为早期命令清除了防御障碍；具体路线未知。",
        )

        self.assertFalse(audit["valid"])

    def test_invalid_answer_is_revised_and_reaudited(self):
        engine = object.__new__(QueryEngine)
        engine.model = RevisingAnswerModel()
        engine.logger = None
        answer, audits, revision_count = engine._generate_audited_answer(
            "哪位组织高层曾经暗中支援？",
            QueryIntent(
                target_entities=["组织高层"],
                search_queries=["组织高层暗中支援"],
            ),
            [
                {
                    "id": 1,
                    "text": "证据支持的候选人是该组织高层，并承认暗中支援。",
                    "participants": ["证据支持的候选人"],
                    "source_key": "main/test.json",
                }
            ],
            [],
            [],
            [],
        )

        self.assertEqual(revision_count, 1)
        self.assertEqual([audit["valid"] for audit in audits], [False, True])
        self.assertIn("证据支持的候选人", answer)

    def test_cross_event_mechanism_is_revised_to_qualified_precondition(self):
        engine = object.__new__(QueryEngine)
        engine.model = EventContinuityRevisingModel()
        engine.logger = None
        answer, audits, revision_count = engine._generate_audited_answer(
            "谁协助后来的势力成功渗透，原因是什么？",
            QueryIntent(
                search_queries=["谁协助渗透"],
                causal_constraint="较早支援是否导致后来渗透",
            ),
            [
                {
                    "id": 1,
                    "text": "甲在较早场景长期支援乙。",
                    "participants": ["甲", "乙"],
                    "source_key": "main/earlier.json",
                },
                {
                    "id": 2,
                    "text": "乙在后来场景发起袭击。",
                    "participants": ["乙"],
                    "source_key": "main/later.json",
                },
            ],
            [],
            [],
            [],
        )

        self.assertEqual(revision_count, 1)
        self.assertEqual([audit["valid"] for audit in audits], [False, True])
        self.assertTrue(audits[0]["event_continuity_reviews"])
        self.assertIn("本次袭击机制未知", answer)

    def test_answer_evidence_includes_preferred_path_episode_endpoints(self):
        episodes = [
            {"id": episode_id, "score": 1.0 - episode_id / 100.0}
            for episode_id in range(1, 11)
        ]
        paths = [
            {
                "association_id": 99,
                "from": ["episode", 1],
                "to": ["episode", 10],
                "path_score": 0.1,
            },
            {
                "association_id": 1,
                "from": ["episode", 2],
                "to": ["episode", 3],
                "path_score": 0.9,
            },
        ]

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes, paths, episode_limit=5, path_limit=2,
            preferred_association_ids={99},
        )

        selected_ids = {item["id"] for item in selected}
        self.assertIn(10, selected_ids)
        self.assertEqual(answer_paths[0]["association_id"], 99)
        for path in answer_paths:
            for endpoint in (path["from"], path["to"]):
                if endpoint[0] == "episode":
                    self.assertIn(endpoint[1], selected_ids)

    def test_answer_evidence_reserves_atomic_query_anchors(self):
        episodes = [
            {"id": episode_id, "score": 1.0 - episode_id / 100.0}
            for episode_id in range(1, 31)
        ]

        selected, _ = QueryEngine._select_answer_evidence(
            episodes,
            [],
            episode_limit=10,
            path_limit=0,
            preferred_episode_ids=[29, 30],
        )

        selected_ids = {item["id"] for item in selected}
        self.assertIn(29, selected_ids)
        self.assertIn(30, selected_ids)
        self.assertEqual(len(selected_ids), 10)

    def test_answer_evidence_preserves_reranker_order(self):
        episodes = [
            {"id": 1, "score": 0.99, "text": "广泛但不直接的背景"},
            {"id": 2, "score": 0.60, "text": "问题的直接证据"},
            {"id": 3, "score": 0.80, "text": "补充证据"},
        ]

        selected, _ = QueryEngine._select_answer_evidence(
            episodes,
            [],
            episode_limit=3,
            path_limit=0,
            question="问题的直接证据是什么？",
            preferred_episode_ids=[2, 3, 1],
        )

        self.assertEqual([item["id"] for item in selected], [2, 3, 1])

    def test_legacy_graph_path_does_not_displace_base_top_k(self):
        episodes = [
            {"id": episode_id, "score": 1.0 - episode_id / 100.0}
            for episode_id in range(1, 11)
        ]
        paths = [
            {
                "association_id": 50,
                "from": ["episode", 1],
                "to": ["episode", 10],
                "relation_key": "legacy_background",
                "relation_text": "与当前问题主题相近的旧背景关系",
                "audit_status": "dual_accepted",
                "created_reason": "导入阶段 Episode 候选关系判断",
                "path_score": 1.0,
            }
        ]

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=5,
            path_limit=2,
            question="当前问题主题",
            preferred_episode_ids=[1, 2, 3, 4, 5],
        )

        self.assertEqual({item["id"] for item in selected}, {1, 2, 3, 4, 5})
        self.assertNotIn(50, [path["association_id"] for path in answer_paths])

    def test_learned_bridge_replaces_semantically_redundant_episode(self):
        episodes = [
            {"id": 1, "score": 1.0, "text": "玄龙门成员担心外校来访月影祭"},
            {"id": 2, "score": 0.9, "text": "弥奈担心外校影响传统月影祭"},
            {"id": 3, "score": 0.8, "text": "未花利用阿里乌斯政变"},
            {"id": 4, "score": 0.7, "text": "京剧部公开弹劾妃咲"},
            {"id": 5, "score": 0.6, "text": "弥奈说明传统与外校月影祭顾虑"},
        ]
        paths = [
            {
                "association_id": 90,
                "from": ["episode", 3],
                "to": ["episode", 5],
                "relation_key": "governance_concern_bridge",
                "relation_text": "弥奈的外校月影祭顾虑与治理选择有关",
                "audit_status": "dual_accepted",
                "created_reason": "查询中自主增长：治理比较",
                "path_score": 0.9,
            }
        ]

        selected, paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=4,
            path_limit=2,
            question="比较外校月影祭顾虑、京剧部事件与阿里乌斯政变",
            preferred_episode_ids=[1, 2, 3, 4],
            learned_bridge_slots=1,
            learned_bridge_duplicate_threshold=1.0,
            coverage_groups=[
                {"mode": "alternatives", "episode_ids": [1, 2]},
                {"mode": "alternatives", "episode_ids": [3]},
                {"mode": "alternatives", "episode_ids": [4]},
            ],
        )

        selected_ids = {item["id"] for item in selected}
        self.assertEqual(selected_ids, {1, 3, 4, 5})
        self.assertTrue(paths[0]["learned_bridge_slot_used"])

    def test_late_bridge_allows_shared_entities_without_treating_them_as_duplicate(self):
        episodes = [
            {"id": 1, "score": 1.0, "text": "未花支持阿里乌斯对付共同敌人"},
            {"id": 3, "score": 0.9, "text": "圣三一学生讨论阿里乌斯危机"},
            {"id": 2, "score": 0.2, "text": "阿里乌斯小队准备破坏圣娅光环"},
        ]
        paths = [
            {
                "association_id": 91,
                "from": ["episode", 1],
                "to": ["episode", 2],
                "relation_key": "thematic_contrast",
                "relation_text": "合作设想与阿里乌斯实际目标存在反差",
                "audit_status": "dual_accepted",
                "created_reason": "查询中自主增长：回答后固定端点候选审计",
                "path_score": 0.9,
                "late_bridge_closure": True,
            }
        ]

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=2,
            path_limit=2,
            question="阿里乌斯的合作设想与实际目标为什么存在反差？",
            preferred_episode_ids=[1, 3],
            learned_bridge_slots=1,
            learned_bridge_duplicate_threshold=0.05,
        )

        self.assertEqual({item["id"] for item in selected}, {1, 2})
        self.assertTrue(answer_paths[0]["learned_bridge_slot_used"])

    def test_late_bridge_extends_fully_protected_base_top_k(self):
        episodes = [
            {"id": 1, "score": 1.0, "text": "未花提出与阿里乌斯合作"},
            {"id": 3, "score": 0.9, "text": "另一条不可删除的基础证据"},
            {"id": 2, "score": 0.2, "text": "阿里乌斯实际准备破坏圣娅光环"},
        ]
        paths = [
            {
                "association_id": 92,
                "from": ["episode", 1],
                "to": ["episode", 2],
                "relation_key": "thematic_contrast",
                "relation_text": "合作设想与实际目标存在反差",
                "audit_status": "dual_accepted",
                "created_reason": "查询中自主增长：回答后固定端点候选审计",
                "path_score": 1.0,
            },
            {
                "association_id": 92,
                "from": ["episode", 1],
                "to": ["episode", 2],
                "relation_key": "thematic_contrast",
                "relation_text": "合作设想与实际目标存在反差",
                "audit_status": "dual_accepted",
                "created_reason": "查询中自主增长：回答后固定端点候选审计",
                "path_score": 0.9,
                "late_bridge_closure": True,
            }
        ]

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=2,
            path_limit=2,
            question="合作设想与实际目标为什么存在反差？",
            preferred_episode_ids=[1, 3],
            learned_bridge_slots=1,
            learned_bridge_duplicate_threshold=0.05,
            coverage_groups=[
                {"mode": "joint", "episode_ids": [1]},
                {"mode": "joint", "episode_ids": [3]},
            ],
        )

        self.assertEqual({item["id"] for item in selected}, {1, 2, 3})
        self.assertTrue(answer_paths[0]["learned_bridge_slot_used"])

    def test_atomic_anchors_leave_budget_for_graph_path_endpoints(self):
        episodes = [
            {"id": episode_id, "score": 1.0 - episode_id / 100.0}
            for episode_id in range(1, 31)
        ]
        paths = [
            {
                "association_id": 99,
                "from": ["episode", 29],
                "to": ["episode", 30],
                "relation_key": "evidence_bridge",
                "relation_text": "跨片段证据桥",
                "path_score": 0.9,
            }
        ]

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=10,
            path_limit=2,
            preferred_association_ids={99},
            preferred_episode_ids=list(range(1, 21)),
        )

        selected_ids = {item["id"] for item in selected}
        self.assertLessEqual(len(selected_ids.intersection(range(1, 21))), 8)
        self.assertIn(29, selected_ids)
        self.assertIn(30, selected_ids)
        self.assertEqual(answer_paths[0]["association_id"], 99)

    def test_answer_path_budget_reserves_query_relevant_informative_edges(self):
        episodes = [
            {"id": episode_id, "score": 1.0 - episode_id / 100.0}
            for episode_id in range(1, 31)
        ]
        paths = [
            {
                "association_id": edge_id,
                "from": ["episode", 1],
                "to": ["concept", edge_id],
                "relation_key": "involves",
                "relation_text": "该 Episode 涉及某个普通概念",
                "path_score": 0.95 - edge_id / 1000.0,
            }
            for edge_id in range(1, 31)
        ]
        paths.append(
            {
                "association_id": 99,
                "from": ["episode", 2],
                "to": ["episode", 30],
                "relation_key": "evidence_bridge",
                "relation_text": "补习部真正政治原因是排查阻止伊甸条约的叛徒",
                "audit_status": "dual_accepted",
                "created_reason": "查询中自主增长：补习部叛徒问题",
                "path_score": 0.1,
            }
        )
        paths.extend(
            {
                "association_id": 100 + edge_id,
                "from": ["episode", 2],
                "to": ["episode", 30],
                "relation_key": f"background_{edge_id}",
                "relation_text": "另一条有证据端点但与当前问题无直接词项重合的关系",
                "path_score": 0.2 - edge_id / 1000.0,
            }
            for edge_id in range(10)
        )

        selected, answer_paths = QueryEngine._select_answer_evidence(
            episodes,
            paths,
            episode_limit=10,
            path_limit=8,
            question="补习部成立的真正政治原因和叛徒身份是什么？",
        )

        self.assertIn(99, [path["association_id"] for path in answer_paths])
        self.assertIn(30, [episode["id"] for episode in selected])
        self.assertLessEqual(
            sum(path["relation_key"] == "involves" for path in answer_paths),
            2,
        )

    def test_concurrent_source_preparation_and_relation_batching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story = root / "main" / "many-segments.txt"
            story.parent.mkdir()
            story.write_text(
                "\n\n".join(
                    f"场景{index}：阿洛娜、老师与[USERNAME]交流。"
                    + "对话。" * 70
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            model = ConcurrentFakeModel(config.model.embedding_dimension)
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(root / "logs" / "concurrent.jsonl"),
                prepare_workers=3,
                relation_batch_size=12,
            )

            summary = pipeline.import_path(story)

            self.assertGreater(int(summary["sources"]), 1)
            self.assertEqual(summary["episodes"], summary["sources"])
            self.assertGreater(model.max_active_episode_calls, 1)
            self.assertEqual(model.batch_relation_calls, 1)
            self.assertEqual(summary["file_results"][0]["status"], "completed")
            self.assertTrue(
                summary["file_results"][0]["relations_finalized"]
            )
            events = [
                json.loads(line)["event"]
                for line in (root / "logs" / "concurrent.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertLess(events.index("file_finished"), events.index("run_finished"))

    def test_relation_tail_failure_prevents_completed_file_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story = root / "main" / "tail-failure.txt"
            story.parent.mkdir()
            story.write_text("阿洛娜与老师讨论后续安排。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "tail-failure.jsonl"),
                relation_batch_size=99,
            )
            original_flush = pipeline._flush_relation_batches

            def fail_only_corpus_tail(
                concept_jobs,
                episode_groups,
                *,
                force,
                executor=None,
                pending_futures=None,
            ):
                if force:
                    raise RuntimeError("simulated relation tail failure")
                return original_flush(
                    concept_jobs,
                    episode_groups,
                    force=force,
                    executor=executor,
                    pending_futures=pending_futures,
                )

            pipeline._flush_relation_batches = fail_only_corpus_tail

            summary = pipeline.import_path(story)

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["files_completed"], 0)
            self.assertEqual(summary["files_partial"], 1)
            self.assertEqual(summary["failed_relation_batches"], 1)
            self.assertEqual(summary["file_results"][0]["status"], "partial")
            self.assertFalse(
                summary["file_results"][0]["relations_finalized"]
            )

    def test_end_to_end_import_and_growth_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story = root / "main" / "sample.txt"
            story.parent.mkdir()
            story.write_text(
                "阿洛娜第一次在什亭之匣中见到[USERNAME]。\n\n双方开始交流。",
                encoding="utf-8",
            )
            logger = JsonlEventLogger(root / "logs" / "test.jsonl")
            model = FakeModel(config.model.embedding_dimension)
            episode_index = EmbeddingIndex(config.model.embedding_dimension)
            concept_index = EmbeddingIndex(config.model.embedding_dimension)
            pipeline = ImportPipeline(
                config,
                db,
                model,
                logger,
                episode_index,
                concept_index,
            )
            summary = pipeline.import_path(story)
            self.assertEqual(summary["failed_tasks"], 0)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["revised_episodes"], 1)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            sources = SourceRepository(db)
            associations = AssociationRepository(db, config.weights)
            self.assertEqual(episodes.count(), 1)
            self.assertEqual(concepts.count(), 1)
            self.assertEqual(len(concepts.find_by_alias("Arona")), 1)

            engine = QueryEngine(
                config,
                model,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
                logger,
            )
            question = "阿洛娜与老师第一次见面是什么时候？"
            plan = engine.build_query_plan(question)
            result = engine.query(question, frozen_plan=plan)
            self.assertIn("最初联系", result["answer"])
            self.assertTrue(result["query_plan_frozen"])
            self.assertEqual(result["query_plan_id"], plan["plan_id"])
            self.assertEqual(result["new_association_ids"], [])
            self.assertTrue(result["rerank_frozen_before_growth"])
            self.assertTrue(result["growth_utility_gate"]["enabled"])
            self.assertFalse(
                result["growth_counterfactual_utility"]["causal_utility_observed"]
            )
            self.assertTrue(result["growth_utility_gate"]["removed_created_ids"])
            self.assertTrue(result["evidence_episodes"])
            self.assertTrue(result["evidence_concepts"])
            self.assertIn("association_paths", result)

            self.assertEqual(result["answer_revision_count"], 0)
            self.assertTrue(result["answer_audits"][0]["valid"])
            self.assertEqual(len(result["intent"]["search_queries"]), 3)
            self.assertEqual(result["followup_search_queries"], [])
            timings = result["timings"]
            self.assertEqual("query-stage-timing-v1", timings["version"])
            self.assertEqual(
                {
                    "intent_parse",
                    "initial_embedding",
                    "initial_retrieval",
                    "initial_graph_expansion",
                    "prepared_early_contextual",
                    "followup_planning",
                    "followup_embedding",
                    "followup_retrieval",
                    "followup_graph_expansion",
                    "source_cohort",
                    "evidence_preparation",
                    "evidence_rerank",
                    "contextual_association",
                    "association_growth",
                    "final_selection",
                    "growth_utility",
                    "answer_generation",
                },
                set(timings["phases_seconds"]),
            )
            self.assertGreater(timings["total_seconds"], 0.0)
            self.assertAlmostEqual(
                timings["total_seconds"],
                timings["measured_seconds"] + timings["unattributed_seconds"],
                places=4,
            )
            self.assertGreaterEqual(associations.stats()["edges"], 1)
            self.assertTrue((root / "logs" / "test.jsonl").exists())

            reinforced = engine.query(question, frozen_plan=plan)
            self.assertEqual(reinforced["new_association_ids"], [])
            self.assertEqual(reinforced["reinforced_association_ids"], [])
            self.assertFalse(
                reinforced["growth_counterfactual_utility"][
                    "causal_utility_observed"
                ]
            )

    def test_pass1_retries_before_persisting_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.ingestion.task_max_attempts = 2
            db = Database(config.database_path)
            db.initialize()
            story = root / "retry.txt"
            story.write_text("阿洛娜见到老师。", encoding="utf-8")
            model = OneTimeConceptFailureModel(config.model.embedding_dimension)
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(root / "logs" / "retry.jsonl"),
            )

            summary = pipeline.import_path(story)

            self.assertEqual(summary["failed_tasks"], 0)
            self.assertTrue(model.failed_once)
            self.assertEqual(SourceRepository(db).count(), 1)
            self.assertEqual(EpisodeRepository(db).count(), 1)
            with db.connection() as connection:
                task = connection.execute(
                    "SELECT status, retry_count FROM extraction_task"
                ).fetchone()
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["retry_count"], 1)

    def test_failed_extraction_does_not_leave_empty_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.ingestion.task_max_attempts = 2
            db = Database(config.database_path)
            db.initialize()
            story = root / "broken.txt"
            story.write_text("需要提取的文本。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                AlwaysFailFactModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "broken.jsonl"),
            )

            summary = pipeline.import_path(story)

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failed_tasks"], 1)
            self.assertEqual(summary["files_failed"], 1)
            self.assertEqual(SourceRepository(db).count(), 0)
            self.assertEqual(summary["failure_details"][0]["stage"], "pass1")

    def test_clean_empty_episode_is_skipped_only_after_independent_audit(self):
        class AuditedEmptyModel(FakeModel):
            def __init__(self, config):
                super().__init__(config.embedding_dimension)
                self.config = config

            def chat_json(self, system: str, user: str, **kwargs):
                if "事实提取器" in system:
                    return {"episodes": []}
                if "空 Episode 对抗审核器" in system:
                    reviews = []
                    for line in user.splitlines():
                        matched = re.match(r"\[L(\d+)\]\s+(.*)$", line)
                        if matched is None:
                            continue
                        line_number, raw_line = int(matched.group(1)), matched.group(2)
                        if raw_line.startswith("unknown:"):
                            reviews.append(
                                {
                                    "start_line": line_number,
                                    "end_line": line_number,
                                    "kind": "control",
                                    "quote": raw_line,
                                    "reason": "engine display-state command",
                                }
                            )
                    return {
                        "contract_version": "empty_episode_adversarial_audit_v1",
                        "verdict": "safe_skip",
                        "source_kind": "control_only",
                        "reason": "control-only source",
                        "line_reviews": reviews,
                        "required_ranges": [],
                    }
                return super().chat_json(system, user, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.model.reasoning_model = "primary-extractor"
            config.model.fallback_model = "independent-reviewer"
            db = Database(config.database_path)
            db.initialize()
            story = root / "control.txt"
            story.write_text("#clearST", encoding="utf-8")
            log_path = root / "logs" / "empty-audit.jsonl"
            pipeline = ImportPipeline(
                config,
                db,
                AuditedEmptyModel(config.model),
                JsonlEventLogger(log_path),
            )

            summary = pipeline.import_path(story)

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["skipped_tasks"], 1)
            self.assertEqual(summary["sources"], 0)
            self.assertEqual(summary["episodes"], 0)
            self.assertEqual(summary["file_results"][0]["segments_skipped"], 1)
            self.assertEqual(summary["file_results"][0]["status"], "completed")
            self.assertEqual(SourceRepository(db).count(), 0)
            self.assertEqual(EpisodeRepository(db).count(), 0)
            with db.connection() as connection:
                task = connection.execute(
                    "SELECT status, source_id, error_summary FROM extraction_task"
                ).fetchone()
            self.assertEqual(task["status"], "skipped")
            self.assertIsNone(task["source_id"])
            self.assertIn("safe_skip", task["error_summary"])
            audit_events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "empty_episode_skip_confirmed"
            ]
            self.assertEqual(len(audit_events), 1)
            self.assertIn("source_sha256", audit_events[0])
            self.assertNotIn("#clearST", json.dumps(audit_events[0]))

    def test_directory_import_reports_empty_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "01_good.txt").write_text("阿洛娜见到老师。", encoding="utf-8")
            (corpus / "02_empty.txt").write_text("", encoding="utf-8")
            (corpus / "notes.bin").write_bytes(b"ignored")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "mixed.jsonl"),
            )

            summary = pipeline.import_path(corpus)

            self.assertEqual(summary["files"], 2)
            self.assertEqual(summary["files_completed"], 1)
            self.assertEqual(summary["files_failed"], 1)
            self.assertEqual(summary["unsupported_files"], ["notes.bin"])
            self.assertEqual(summary["status"], "partial")
            self.assertIn("no usable text blocks", summary["failure_details"][0]["error"])

    def test_malformed_json_fails_one_file_without_stopping_later_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "01_good.txt").write_text(
                "阿洛娜见到[USERNAME]（老师），并开始交流。",
                encoding="utf-8",
            )
            (corpus / "02_broken.json").write_text(
                '{"records": [', encoding="utf-8"
            )
            (corpus / "03_good.txt").write_text(
                "阿洛娜见到[USERNAME]（老师），并开始交流。",
                encoding="utf-8",
            )
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "malformed-json.jsonl"),
            )

            summary = pipeline.import_path(corpus)

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["files_completed"], 2)
            self.assertEqual(summary["files_failed"], 1)
            self.assertEqual(summary["file_results"][2]["status"], "completed")
            self.assertEqual(EpisodeRepository(db).count(), 2)

    def test_source_root_is_used_consistently_for_unsupported_file_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            corpus = root / "knowledge" / "volume"
            corpus.mkdir(parents=True)
            (corpus / "01_good.txt").write_text(
                "阿洛娜见到老师。", encoding="utf-8"
            )
            (corpus / "notes.bin").write_bytes(b"ignored")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "root-path.jsonl"),
            )

            summary = pipeline.import_path(corpus, source_root=root)

            self.assertEqual(summary["status"], "partial")
            self.assertEqual(
                summary["unsupported_files"], ["knowledge/volume/notes.bin"]
            )
            with db.connection() as connection:
                source_key = connection.execute(
                    "SELECT source_key FROM episode"
                ).fetchone()["source_key"]
            self.assertEqual(source_key, "knowledge/volume/01_good.txt")

    def test_interrupted_import_closes_run_and_running_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story = root / "interrupt.txt"
            story.write_text("阿洛娜见到老师。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                InterruptingFactModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "interrupt.jsonl"),
            )

            with self.assertRaisesRegex(KeyboardInterrupt, "operator interrupted"):
                pipeline.import_path(story)

            with db.connection() as connection:
                run = connection.execute(
                    "SELECT status, summary_json FROM extraction_run"
                ).fetchone()
                task = connection.execute(
                    "SELECT status FROM extraction_task"
                ).fetchone()
            self.assertEqual(run["status"], "interrupted")
            self.assertEqual(task["status"], "interrupted")
            self.assertEqual(json.loads(run["summary_json"])["status"], "interrupted")

    def test_single_pass_profile_persists_once_and_skips_second_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.ingestion.episode_extraction_profile = (
                "single_pass_evidence"
            )
            config.ingestion.episode_audit_always = True
            config.ingestion.episode_factual_audit_mode = "always"
            db = Database(config.database_path)
            db.initialize()
            story = root / "single-pass.txt"
            story.write_text(
                "Morgan received approval from ???.", encoding="utf-8"
            )
            model = SinglePassFakeModel(config.model.embedding_dimension)
            log_path = root / "logs" / "single-pass.jsonl"
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(log_path),
            )

            summary = pipeline.import_path(story)
            events = [
                json.loads(line)["event"]
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["status"], "completed", summary)
            self.assertEqual(EpisodeRepository(db).count(), 1)
            self.assertEqual(model.second_pass_calls, 0)
            self.assertIn("single_pass_episode_evidence_validated", events)
            self.assertIn("second_pass_skipped", events)
            self.assertNotIn("episode_factual_audit_started", events)

    def test_document_map_profile_builds_navigation_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.ingestion.episode_extraction_profile = (
                "document_map_assisted"
            )
            config.prompt_version = "v4.5_document_map_planned_audited"
            db = Database(config.database_path)
            db.initialize()
            story = root / "mapped.txt"
            story.write_text(
                "Morgan received approval from ???.", encoding="utf-8"
            )
            model = DocumentMapFakeModel(config.model.embedding_dimension)
            log_path = root / "logs" / "mapped.jsonl"
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(log_path),
            )

            summary = pipeline.import_path(story)
            events = [
                json.loads(line)["event"]
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["status"], "completed", summary)
            self.assertEqual(model.document_map_calls, 1)
            self.assertTrue(model.extraction_received_map)
            self.assertEqual(model.second_pass_calls, 0)
            self.assertIn("document_map_built", events)
            self.assertIn("single_pass_episode_evidence_validated", events)
            self.assertIn("second_pass_skipped", events)

    def test_adaptive_anchor_map_runs_only_after_file_is_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            config.ingestion.episode_extraction_profile = (
                "adaptive_anchor_map"
            )
            config.prompt_version = "v4.7_adaptive_literal_anchor_audited"
            db = Database(config.database_path)
            db.initialize()
            story = root / "split.txt"
            story.write_text(
                "\n\n".join(
                    f"Morgan received approval from ???. Detail {index}."
                    for index in range(12)
                ),
                encoding="utf-8",
            )
            model = AdaptiveAnchorMapFakeModel(
                config.model.embedding_dimension
            )
            log_path = root / "logs" / "anchored.jsonl"
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(log_path),
            )

            summary = pipeline.import_path(story)
            events = [
                json.loads(line)["event"]
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["status"], "completed", summary)
            self.assertGreater(summary["sources"], 1)
            self.assertEqual(model.document_anchor_map_calls, 1)
            self.assertTrue(model.extraction_received_map)
            self.assertIn("document_anchor_map_triggered", events)
            self.assertIn("document_anchor_map_built", events)
            self.assertIn("second_pass_skipped", events)

    def test_second_pass_is_limited_to_episodes_from_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            source_id = SourceRepository(db).insert("旧批次原文")
            vector = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            old_text = "旧批次的[USERNAME]标记必须保持不变。"
            EpisodeRepository(db).insert(
                source_id,
                "same.txt",
                0,
                EpisodeDraft(text=old_text, confidence=0.9),
                vector,
            )
            story = root / "same.txt"
            story.write_text("新批次的阿洛娜见到老师。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "run-boundary.jsonl"),
            )

            pipeline.import_path(story)

            self.assertEqual(EpisodeRepository(db).get(1)["text"], old_text)

    def test_second_pass_cannot_reintroduce_identity_absent_from_source(self):
        class HallucinatingSecondPassModel(FakeModel):
            def chat_json(self, system: str, user: str, **kwargs):
                if "二次理解器" in system:
                    return {
                        "episode": {
                            "text": "InventedName approved Morgan's request.",
                            "participants": ["InventedName", "Morgan"],
                            "event_type": "approval",
                            "confidence": 0.9,
                        }
                    }
                if "Concept 提取器" in system:
                    return {
                        "episode_concepts": [
                            {"episode_index": 0, "concepts": []}
                        ]
                    }
                if "事实提取器" in system:
                    return {
                        "episodes": [
                            {
                                "text": "??? approved Morgan's request.",
                                "participants": ["???", "Morgan"],
                                "event_type": "approval",
                                "confidence": 0.9,
                            }
                        ]
                    }
                return super().chat_json(system, user, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story = root / "unknown-speaker.txt"
            story.write_text(
                "Morgan received an approval from an unidentified person.",
                encoding="utf-8",
            )
            pipeline = ImportPipeline(
                config,
                db,
                HallucinatingSecondPassModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "second-pass-grounding.jsonl"),
            )

            summary = pipeline.import_path(story)
            row = EpisodeRepository(db).get(1)

            self.assertEqual(summary["status"], "completed")
            self.assertIn("???", row["text"])
            self.assertNotIn("InventedName", row["text"])

    def test_growth_utility_gate_removes_unused_and_restores_reinforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            source_id = sources.insert("测试原文")
            episodes = EpisodeRepository(db)
            vector = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            first = episodes.insert(
                source_id,
                "main/test.json",
                0,
                EpisodeDraft("直接证据甲"),
                vector,
            )
            second = episodes.insert(
                source_id,
                "main/test.json",
                0,
                EpisodeDraft("直接证据乙"),
                vector,
            )
            associations = AssociationRepository(db, config.weights)
            reinforced_draft = AssociationDraft(
                "episode", first, "episode", second,
                "semantic", "existing_bridge", "已有关系",
                weight=0.6, confidence=0.8,
            )
            reinforced_id = associations.upsert(reinforced_draft)
            before = dict(associations.get(reinforced_id))
            associations.upsert(reinforced_draft)
            created_id = associations.upsert(
                AssociationDraft(
                    "episode", first, "episode", second,
                    "semantic", "unused_bridge", "未使用的新关系",
                    weight=0.7, confidence=0.8,
                )
            )
            engine = object.__new__(QueryEngine)
            engine.associations = associations
            engine.logger = None

            gate = engine._prune_unused_growth(
                [created_id],
                [reinforced_id],
                {reinforced_id: before},
                [],
            )

            self.assertIsNone(associations.get(created_id))
            restored = associations.get(reinforced_id)
            self.assertEqual(float(restored["weight"]), float(before["weight"]))
            self.assertEqual(
                int(restored["evidence_count"]), int(before["evidence_count"])
            )
            self.assertEqual(gate["removed_created_ids"], [created_id])
            self.assertEqual(gate["restored_reinforced_ids"], [reinforced_id])

    def test_counterfactual_growth_requires_new_direct_episode_evidence(self):
        changed_ids = {91, 92}
        treatment = {
            "episode_ids": [1, 2],
            "evidence_episodes": [
                {"id": 1, "source_key": "main/a.json"},
                {"id": 2, "source_key": "main/b.json"},
            ],
            "association_paths": [
                {"association_id": 91},
                {"association_id": 7},
            ],
        }
        masked = {
            "episode_ids": [1],
            "evidence_episodes": [
                {"id": 1, "source_key": "main/a.json"},
            ],
            "association_paths": [{"association_id": 7}],
        }

        useful = QueryEngine._counterfactual_utility_from_results(
            treatment, masked, changed_ids
        )
        no_gain = QueryEngine._counterfactual_utility_from_results(
            {**treatment, "episode_ids": [1]}, masked, changed_ids
        )

        self.assertTrue(useful["causal_utility_observed"])
        self.assertEqual(useful["additional_episode_ids"], [2])
        self.assertEqual(useful["additional_source_keys"], ["main/b.json"])
        self.assertEqual(useful["persistable_changed_ids"], [91])
        self.assertFalse(no_gain["causal_utility_observed"])
        self.assertEqual(no_gain["persistable_changed_ids"], [])

    def test_source_root_preserves_corpus_relative_key_for_single_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story_root = root / "public" / "story"
            story = story_root / "main" / "sample.txt"
            story.parent.mkdir(parents=True)
            story.write_text("阿洛娜见到老师。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "root.jsonl"),
            )
            pipeline.import_path(story, source_root=story_root)
            row = EpisodeRepository(db).get(1)
            self.assertEqual(row["source_key"], "main/sample.txt")
            self.assertEqual(row["timeline_scope"], "main")

    def test_corpus_path_overrides_model_invented_timeline_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            story_root = root / "story"
            story = story_root / "favor" / "10000" / "sample.txt"
            story.parent.mkdir(parents=True)
            story.write_text("阿露与老师交谈。", encoding="utf-8")
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(config.model.embedding_dimension),
                JsonlEventLogger(root / "logs" / "scope.jsonl"),
            )

            pipeline.import_path(story, source_root=story_root)

            row = EpisodeRepository(db).get(1)
            self.assertEqual(row["source_key"], "favor/10000/sample.txt")
            self.assertEqual(row["timeline_scope"], "favor:10000")


if __name__ == "__main__":
    unittest.main()

