from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from tests.helpers import FakeModel, test_config
from memory_demo.associations import AssociationBuilder
from memory_demo.config import ParagraphConfig
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.paragraphs import ParagraphSegmenter
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.llm.prompts import concept_prompt
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine
from memory_demo.types import ConceptDraft, EpisodeDraft, ParagraphDraft


class Stage6ParagraphConceptTests(unittest.TestCase):
    def test_source_views_share_one_embedding_request(self):
        class CountingModel(FakeModel):
            def __init__(self, dimension: int):
                super().__init__(dimension)
                self.embedding_batches: list[list[str]] = []

            def embed(self, texts: list[str]) -> np.ndarray:
                self.embedding_batches.append(list(texts))
                return super().embed(texts)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=8)
            config.paragraph.enabled = True
            db = Database(config.database_path)
            db.initialize()
            model = CountingModel(8)
            pipeline = ImportPipeline(
                config,
                db,
                model,
                JsonlEventLogger(root / "logs" / "combined.jsonl"),
            )
            source_text = "阿洛娜与老师在什亭之匣中交谈。"
            episode_drafts = [
                EpisodeDraft(
                    "阿洛娜与老师在什亭之匣中交谈。",
                    participants=["阿洛娜", "老师"],
                )
            ]
            concept_groups = [
                [
                    ConceptDraft(
                        canonical_name="阿洛娜",
                        description="什亭之匣中的人工智能少女",
                        embedding_text="阿洛娜，什亭之匣中的人工智能少女",
                    )
                ]
            ]

            (
                paragraph_drafts,
                paragraph_vectors,
                episode_vectors,
                concept_vectors,
            ) = pipeline._prepare_source_embeddings(
                source_text,
                episode_drafts,
                concept_groups,
            )
            source_id = pipeline.sources.insert(source_text)
            run_id = pipeline.extractions.start_run({}, {}, root / "logs" / "run.jsonl")
            pipeline._insert_paragraphs(
                source_id,
                "main/a.txt",
                0,
                source_text,
                prepared_drafts=paragraph_drafts,
                prepared_vectors=paragraph_vectors,
            )
            pipeline._insert_episode_batch(
                run_id,
                source_id,
                "main/a.txt",
                0,
                episode_drafts,
                prepared_concept_groups=concept_groups,
                prepared_episode_vectors=episode_vectors,
                prepared_concept_vectors=concept_vectors,
            )

            self.assertEqual(len(model.embedding_batches), 1)
            self.assertEqual(len(model.embedding_batches[0]), 3)
            self.assertEqual(pipeline.paragraphs.count(), 1)
            self.assertEqual(pipeline.episodes.count(), 1)
            self.assertEqual(pipeline.concepts.count(), 1)
            self.assertEqual(pipeline.associations.stats()["edges"], 1)

    def test_paragraph_segmenter_preserves_records_and_overlap(self):
        source = (
            "[source_key: main/test.json]\n[segment_index: 0]\n\n"
            + "\n\n".join(
                f"[record: {index}]\n[speaker_raw: 学生]\nzh-CN: 第{index}段" + "内容" * 45
                for index in range(1, 7)
            )
        )
        segmenter = ParagraphSegmenter(
            ParagraphConfig(
                enabled=True,
                target_chars=260,
                max_chars=360,
                overlap_chars=130,
                minimum_chars=80,
            )
        )
        paragraphs = segmenter.segment(source)

        self.assertGreater(len(paragraphs), 1)
        self.assertTrue(all(item.text.startswith("[source_key:") for item in paragraphs))
        self.assertTrue(all("[record:" in item.text for item in paragraphs))
        record_sets = [
            {
                line
                for line in item.text.splitlines()
                if line.startswith("[record:")
            }
            for item in paragraphs
        ]
        self.assertTrue(
            any(left.intersection(right) for left, right in zip(record_sets, record_sets[1:]))
        )

    def test_paragraph_embedding_text_removes_scaffolding_but_keeps_languages(self):
        text = (
            "[source_key: main/a.json]\n[segment_index: 1]\n\n"
            "[speaker_alias_legend]\n아루: zh-CN=阿露 | en=Aru\n\n"
            "[record: 3]\n[speaker_raw: 아루]\n"
            "[script_raw: 1;아루;00;原始指令\n"
            "#na;아루;不应进入 embedding\n"
            "#fontsize;55]\n"
            "zh-CN: 阿露点了情侣专用菜单。\n"
            "en: Aru ordered the couples-only menu."
        )
        cleaned = ParagraphSegmenter.embedding_text(text)

        self.assertNotIn("source_key", cleaned)
        self.assertNotIn("speaker_alias_legend", cleaned)
        self.assertNotIn("script_raw", cleaned)
        self.assertNotIn("不应进入 embedding", cleaned)
        self.assertNotIn("zh-CN=阿露", cleaned)
        self.assertIn("speaker: 아루", cleaned)
        self.assertIn("zh-CN: 阿露点了情侣专用菜单。", cleaned)
        self.assertIn("en: Aru ordered the couples-only menu.", cleaned)

    def test_paragraph_retrieval_adds_source_episode_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.paragraph.enabled = True
            config.retrieval.episode_top_k = 1
            config.retrieval.concept_top_k = 0
            config.retrieval.paragraph_top_k = 1
            config.retrieval.paragraph_episode_expansion_limit = 2
            config.retrieval.paragraph_rrf_weight = 0.35
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            paragraphs = ParagraphRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            paragraph_index = EmbeddingIndex(4)

            source_one = sources.insert("关键词只在原文段落中出现")
            source_two = sources.insert("直接 Episode 命中")
            episode_one_vector = np.array([0.2, 0.98, 0.0, 0.0], dtype=np.float32)
            episode_two_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            episode_one = episodes.insert(
                source_one,
                "main/a.json",
                0,
                EpisodeDraft("摘要没有保留关键词"),
                encode_embedding(episode_one_vector, 4),
            )
            episode_two = episodes.insert(
                source_two,
                "main/b.json",
                0,
                EpisodeDraft("直接相似摘要"),
                encode_embedding(episode_two_vector, 4),
            )
            episode_index.upsert(episode_one, episode_one_vector)
            episode_index.upsert(episode_two, episode_two_vector)
            paragraph_ids = paragraphs.insert_many(
                source_one,
                "main/a.json",
                0,
                [ParagraphDraft(0, "精确关键词原文")],
                [encode_embedding([1.0, 0.0, 0.0, 0.0], 4)],
            )
            paragraph_index.upsert(paragraph_ids[0], [1.0, 0.0, 0.0, 0.0])
            engine = QueryEngine(
                config,
                model=None,
                episode_index=episode_index,
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
                paragraph_index=paragraph_index,
                paragraphs=paragraphs,
            )

            hits, rankings = engine._vector_seed_hits_from_matrix(
                ["精确关键词"],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self.assertEqual(rankings["episode"][0][0]["id"], episode_two)
            self.assertEqual(rankings["paragraph"][0][0]["id"], paragraph_ids[0])
            self.assertIn(episode_one, {hit.node_id for hit in hits})

            config.paragraph.enabled = False
            disabled_hits, disabled_rankings = engine._vector_seed_hits_from_matrix(
                ["精确关键词"],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            self.assertEqual(disabled_rankings["paragraph"], [[]])
            self.assertEqual({hit.node_id for hit in disabled_hits}, {episode_two})

    def test_recall_only_paragraph_does_not_change_existing_anchor_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.paragraph.enabled = True
            config.retrieval.episode_top_k = 2
            config.retrieval.concept_top_k = 0
            config.retrieval.paragraph_top_k = 1
            config.retrieval.paragraph_episode_expansion_limit = 2
            config.retrieval.paragraph_recall_only = True
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            paragraphs = ParagraphRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            episode_index = EmbeddingIndex(4)
            concept_index = EmbeddingIndex(4)
            paragraph_index = EmbeddingIndex(4)
            source_id = sources.insert("原文")
            first = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("第一条"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            second = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("第二条"),
                encode_embedding([0.8, 0.6, 0.0, 0.0], 4),
            )
            episode_index.upsert(first, [1.0, 0.0, 0.0, 0.0])
            episode_index.upsert(second, [0.8, 0.6, 0.0, 0.0])
            paragraph_id = paragraphs.insert_many(
                source_id,
                "main/a.json",
                0,
                [ParagraphDraft(0, "偏向第二条的原文")],
                [encode_embedding([0.8, 0.6, 0.0, 0.0], 4)],
            )[0]
            paragraph_index.upsert(paragraph_id, [0.8, 0.6, 0.0, 0.0])
            engine = QueryEngine(
                config,
                model=None,
                episode_index=episode_index,
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
                paragraph_index=paragraph_index,
                paragraphs=paragraphs,
            )
            anchors: list[int] = []
            engine._vector_seed_hits_from_matrix(
                ["查询"],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                anchors,
            )

            self.assertEqual(anchors[:2], [first, second])

    def test_paragraph_backfill_is_retry_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=8)
            config.paragraph.enabled = True
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            source_id = sources.insert(
                "[source_key: main/a.json]\n\n[record: 1]\nzh-CN: 古圣堂地下仍是废墟。"
            )
            episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("古圣堂地下仍是废墟"),
                encode_embedding(np.arange(1, 9, dtype=np.float32), 8),
            )
            pipeline = ImportPipeline(
                config,
                db,
                FakeModel(8),
                JsonlEventLogger(root / "logs" / "paragraphs.jsonl"),
            )

            first = pipeline.backfill_paragraphs()
            second = pipeline.backfill_paragraphs()

            self.assertEqual(first["sources"], 1)
            self.assertEqual(first["paragraphs"], 1)
            self.assertEqual(second, {"sources": 0, "paragraphs": 0, "failed_sources": 0})

    def test_fine_grained_concept_prompt_is_explicitly_reversible(self):
        conservative = concept_prompt("事件", ["人物"], profile="conservative")
        fine_grained = concept_prompt(
            "事件",
            ["人物"],
            profile="fine_grained",
            target_min=4,
            target_max=10,
        )

        self.assertIn("conservative 档", conservative)
        self.assertIn("fine_grained 档", fine_grained)
        self.assertIn("4—10", fine_grained)
        self.assertNotEqual(conservative, fine_grained)

    def test_concept_augmentation_can_reuse_old_edge_without_reinforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            source_id = sources.insert("阿洛娜和老师开始交流。")
            episode_id = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("阿洛娜和老师开始交流。"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            concept_id = concepts.insert(
                ConceptDraft(
                    canonical_name="阿洛娜",
                    description="什亭之匣中的少女",
                    embedding_text="阿洛娜，什亭之匣中的少女",
                    confidence=0.9,
                ),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            builder = AssociationBuilder(
                FakeModel(4),
                associations,
                episodes,
                concepts,
                config.weights,
            )

            association_id = builder.link_episode_concept(
                episode_id, concept_id, 0.9, "阿洛娜"
            )
            before = dict(associations.get(association_id))
            reused_id = builder.link_episode_concept(
                episode_id,
                concept_id,
                0.9,
                "阿洛娜",
                reinforce_existing=False,
            )
            after_reuse = dict(associations.get(association_id))

            self.assertEqual(reused_id, association_id)
            self.assertEqual(after_reuse, before)

            builder.link_episode_concept(episode_id, concept_id, 0.9, "阿洛娜")
            after_reinforcement = dict(associations.get(association_id))
            self.assertGreater(after_reinforcement["weight"], before["weight"])
            self.assertEqual(
                after_reinforcement["evidence_count"],
                before["evidence_count"] + 1,
            )

    def test_concept_promotion_gate_logs_transient_and_persists_durable_group(self):
        class AdmissionModel(FakeModel):
            def chat_json(self, system: str, user: str, **kwargs):
                if "Concept 提取器" in system:
                    return {
                        "episode_concepts": [
                            {
                                "episode_index": 0,
                                "concepts": [
                                    {
                                        "canonical_name": "持续信任",
                                        "description": "老师持续相信学生的信念",
                                        "embedding_text": "老师对学生的持续信任",
                                        "aliases": [],
                                        "confidence": 0.9,
                                    },
                                    {
                                        "canonical_name": "一次性挥手",
                                        "description": "人物在当前场景挥了一次手",
                                        "embedding_text": "一次性挥手动作",
                                        "aliases": [],
                                        "confidence": 0.8,
                                    },
                                    {
                                        "canonical_name": "阿洛娜",
                                        "description": "什亭之匣中的少女",
                                        "embedding_text": "阿洛娜，什亭之匣中的少女",
                                        "aliases": [],
                                        "confidence": 0.9,
                                    },
                                ],
                            },
                            {
                                "episode_index": 1,
                                "concepts": [
                                    {
                                        "canonical_name": "持续信任",
                                        "description": "老师即使面对风险仍相信学生",
                                        "embedding_text": "老师持续相信学生",
                                        "aliases": [],
                                        "confidence": 0.95,
                                    }
                                ],
                            },
                        ]
                    }
                if "Concept 准入审计器" in system:
                    candidate_indexes = sorted(
                        {int(value) for value in re.findall(r'"candidate_index":\s*(\d+)', user)}
                    )
                    return {
                        "decisions": [
                            {
                                "candidate_index": candidate_index,
                                "action": "promote",
                                "existing_concept_id": None,
                                "reason": "跨 Episode 重复的稳定语义",
                                "confidence": 0.99,
                            }
                            for candidate_index in candidate_indexes
                        ]
                    }
                return super().chat_json(system, user, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.concept_extraction.profile = "fine_grained"
            config.concept_extraction.promotion_gate_enabled = True
            config.concept_extraction.promotion_min_distinct_episodes = 2
            config.concept_extraction.promotion_singleton_reuse_similarity_floor = 1.1
            config.retrieval.concept_relation_min_similarity = 1.1
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            source_id = sources.insert("老师相信学生。人物随后挥手。")
            episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("老师相信学生，阿洛娜在场。"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("老师即使面对风险仍相信学生。"),
                encode_embedding([0.0, 1.0, 0.0, 0.0], 4),
            )
            concepts.insert(
                ConceptDraft(
                    canonical_name="阿洛娜",
                    description="什亭之匣中的少女",
                    embedding_text="阿洛娜，什亭之匣中的少女",
                    confidence=0.9,
                ),
                encode_embedding([0.0, 0.0, 1.0, 0.0], 4),
            )
            pipeline = ImportPipeline(
                config,
                db,
                AdmissionModel(4),
                JsonlEventLogger(root / "logs" / "admission.jsonl"),
                prepare_workers=2,
            )
            pipeline.rebuild_indexes()

            result = pipeline.augment_concepts()

            self.assertEqual(result["existing_groups_reused"], 1)
            self.assertEqual(result["promoted_groups"], 1)
            self.assertEqual(result["transient_groups"], 1)
            self.assertEqual(result["prefiltered_transient_groups"], 1)
            self.assertEqual(result["llm_admission_groups"], 1)
            self.assertEqual(concepts.count(), 2)
            self.assertTrue(concepts.find_by_alias("持续信任"))
            self.assertFalse(concepts.find_by_alias("一次性挥手"))

    def test_concept_vector_seed_requires_cross_episode_graph_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root, dimension=4)
            config.retrieval.sparse_enabled = False
            config.retrieval.episode_top_k = 0
            config.retrieval.concept_top_k = 2
            config.retrieval.concept_seed_min_reachable_episodes = 2
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            episodes = EpisodeRepository(db)
            concepts = ConceptRepository(db)
            associations = AssociationRepository(db, config.weights)
            source_id = sources.insert("两个 Episode")
            first = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("第一段"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            second = episodes.insert(
                source_id,
                "main/a.json",
                0,
                EpisodeDraft("第二段"),
                encode_embedding([0.0, 1.0, 0.0, 0.0], 4),
            )
            singleton = concepts.insert(
                ConceptDraft("单条标签", "只重述第一段", "单条标签"),
                encode_embedding([1.0, 0.0, 0.0, 0.0], 4),
            )
            bridge = concepts.insert(
                ConceptDraft("跨段主题", "连接两段", "跨段主题"),
                encode_embedding([0.9, 0.1, 0.0, 0.0], 4),
            )
            builder = AssociationBuilder(
                None, associations, episodes, concepts, config.weights
            )
            builder.link_episode_concept(first, singleton, 0.9, "单条标签")
            builder.link_episode_concept(first, bridge, 0.9, "跨段主题")
            builder.link_episode_concept(second, bridge, 0.9, "跨段主题")
            concept_index = EmbeddingIndex(4)
            concept_index.upsert(singleton, [1.0, 0.0, 0.0, 0.0])
            concept_index.upsert(bridge, [0.9, 0.1, 0.0, 0.0])
            engine = QueryEngine(
                config,
                model=None,
                episode_index=EmbeddingIndex(4),
                concept_index=concept_index,
                episodes=episodes,
                concepts=concepts,
                sources=sources,
                associations=associations,
            )

            hits, rankings = engine._vector_seed_hits_from_matrix(
                ["主题"],
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )

            self.assertEqual(
                rankings["concept_raw"][0][0]["id"], singleton
            )
            self.assertEqual(
                [item["id"] for item in rankings["concept"][0]], [bridge]
            )
            self.assertEqual(
                {(hit.node_type, hit.node_id) for hit in hits},
                {("concept", bridge)},
            )


if __name__ == "__main__":
    unittest.main()
