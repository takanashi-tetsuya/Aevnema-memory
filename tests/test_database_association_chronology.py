from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
import unittest

import numpy as np

from memory_demo.chronology import ChronologyService
from memory_demo.associations import AssociationBuilder
from memory_demo.associations.traversal import GraphTraverser
from memory_demo.concepts import ConceptResolver
from memory_demo.config import WeightConfig
from memory_demo.database import Database, SCHEMA_VERSION
from memory_demo.embeddings import encode_embedding
from memory_demo.embeddings.index import EmbeddingIndex
from memory_demo.llm.client import ModelClientError
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
    TemporalCycleError,
)
from memory_demo.repositories.concept import normalize_alias
from memory_demo.types import AssociationDraft, ConceptDraft, EpisodeDraft, SearchHit


class DatabaseAssociationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp.name) / "memory.db")
        self.db.initialize()
        self.sources = SourceRepository(self.db)
        self.episodes = EpisodeRepository(self.db)
        self.associations = AssociationRepository(self.db)
        source_id = self.sources.insert("原始文本")
        vector = encode_embedding(np.arange(1, 9, dtype=np.float32), 8)
        self.first = self.episodes.insert(
            source_id,
            "main/a.json",
            5,
            EpisodeDraft("较早事件", timeline_scope="main"),
            vector,
        )
        self.second = self.episodes.insert(
            source_id,
            "main/a.json",
            1,
            EpisodeDraft("较晚事件", timeline_scope="main"),
            vector,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_association_reinforcement_and_chronology(self):
        draft = AssociationDraft(
            "episode",
            self.first,
            "episode",
            self.second,
            "temporal",
            "before",
            "较早事件发生在较晚事件之前",
            weight=0.8,
            confidence=0.9,
        )
        association_id = self.associations.upsert(draft)
        initial = float(self.associations.get(association_id)["weight"])
        self.assertEqual(self.associations.upsert(draft), association_id)
        reinforced = self.associations.get(association_id)
        self.assertGreater(float(reinforced["weight"]), initial)
        self.assertEqual(int(reinforced["evidence_count"]), 2)
        result = ChronologyService(self.episodes, self.associations).order(
            [self.second, self.first]
        )
        self.assertEqual(result.ordered_ids, [self.first, self.second])

    def test_transactions_queue_concurrent_writers_in_process(self):
        """File-level parallel imports must not race SQLite's single writer."""

        with self.db.transaction() as connection:
            connection.execute(
                "CREATE TABLE concurrent_writer_test (value INTEGER NOT NULL)"
            )

        active_lock = Lock()
        active_writers = 0
        maximum_active_writers = 0

        def write(value: int) -> None:
            nonlocal active_writers, maximum_active_writers
            with self.db.transaction() as connection:
                with active_lock:
                    active_writers += 1
                    maximum_active_writers = max(
                        maximum_active_writers, active_writers
                    )
                try:
                    sleep(0.01)
                    connection.execute(
                        "INSERT INTO concurrent_writer_test(value) VALUES(?)", (value,)
                    )
                finally:
                    with active_lock:
                        active_writers -= 1

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(16)))

        with self.db.connection() as connection:
            persisted = connection.execute(
                "SELECT COUNT(*) FROM concurrent_writer_test"
            ).fetchone()[0]
        self.assertEqual(maximum_active_writers, 1)
        self.assertEqual(persisted, 16)

    def test_graph_traversal_batches_neighbor_reads_by_node_type(self):
        association_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "related_to",
                "两个事件相关",
                weight=0.8,
                confidence=0.9,
            )
        )
        calls: list[tuple[str, tuple[int, ...], int]] = []
        original = self.associations.neighbors_many

        def recorded(node_type, node_ids, limit=100):
            ids = tuple(int(value) for value in node_ids)
            calls.append((node_type, ids, limit))
            return original(node_type, ids, limit)

        self.associations.neighbors_many = recorded
        nodes, paths = GraphTraverser(self.associations).expand(
            [
                SearchHit("episode", self.first, 1.0),
                SearchHit("episode", self.second, 0.9),
            ],
            beam_width=8,
            max_hops=1,
        )

        self.assertEqual([("episode", (self.first, self.second), 16)], calls)
        self.assertEqual(
            {self.first, self.second},
            {node.node_id for node in nodes if node.node_type == "episode"},
        )
        self.assertIn(association_id, [row["association_id"] for row in paths])

    def test_unknown_chronology_preserves_relevance_order(self):
        result = ChronologyService(self.episodes, self.associations).order(
            [self.first, self.second]
        )

        self.assertEqual(result.ordered_ids, [self.first, self.second])

    def test_concept_resolver_reuses_shared_incoming_alias(self):
        class EmbeddingModel:
            @staticmethod
            def embed(texts):
                return np.stack(
                    [np.arange(1, 9, dtype=np.float32) for _ in texts]
                )

        resolver = ConceptResolver(
            EmbeddingModel(),
            ConceptRepository(self.db),
            EmbeddingIndex(8),
            object(),
            8,
        )
        concept_ids, relation_jobs = resolver.resolve_many_deferred(
            [
                ConceptDraft(
                    "花子", "补课部成员", "花子，补课部成员", [("Hanako", "en")]
                ),
                ConceptDraft(
                    "ハナコ", "补课部成员", "ハナコ，补课部成员", [("Hanako", "en")]
                ),
            ]
        )

        self.assertEqual(concept_ids[0], concept_ids[1])
        self.assertEqual(ConceptRepository(self.db).count(), 1)
        self.assertEqual(relation_jobs, [])
        aliases = {
            row["alias"]
            for row in ConceptRepository(self.db).list_aliases(concept_ids[0])
        }
        self.assertIn("ハナコ", aliases)

    def test_alias_normalization_collapses_spaces_only_for_cjk(self):
        self.assertEqual(normalize_alias("凯撒 PMC"), normalize_alias("凯撒PMC"))
        self.assertEqual(normalize_alias("Kaiser Corporation"), "kaiser corporation")

    def test_unique_multilingual_aliases_disambiguate_a_broad_name(self):
        class EmbeddingModel:
            @staticmethod
            def embed(texts):
                return np.stack(
                    [np.arange(1, 9, dtype=np.float32) for _ in texts]
                )

        class NoopAssociationBuilder:
            @staticmethod
            def relate_new_concept(*_args, **_kwargs):
                return None

        repository = ConceptRepository(self.db)
        resolver = ConceptResolver(
            EmbeddingModel(),
            repository,
            EmbeddingIndex(8),
            NoopAssociationBuilder(),
            8,
        )
        base_id = resolver.resolve(
            ConceptDraft(
                "砂狼白子",
                "阿拜多斯学生",
                "砂狼白子，阿拜多斯学生",
                [("白子", "zh"), ("Shiroko", "en"), ("シロコ", "ja")],
            )
        )
        repository.insert(
            ConceptDraft(
                "白子 Terror",
                "白子的特定变体",
                "白子 Terror，特定变体",
                [("白子", "zh")],
            ),
            encode_embedding(np.arange(1, 9, dtype=np.float32), 8),
        )

        resolved_id = resolver.resolve(
            ConceptDraft(
                "白子",
                "阿拜多斯学生",
                "白子，阿拜多斯学生",
                [("Shiroko", "en"), ("シロコ", "ja")],
            )
        )

        self.assertEqual(resolved_id, base_id)
        self.assertEqual(repository.count(), 2)

    def test_repository_rejects_edge_that_would_close_temporal_cycle(self):
        third = self.episodes.insert(
            self.sources.insert("第三段原始文本"),
            "main/c.json",
            0,
            EpisodeDraft("最晚事件", timeline_scope="main"),
            encode_embedding(np.arange(1, 9, dtype=np.float32), 8),
        )
        self.associations.upsert(
            AssociationDraft(
                "episode", self.first, "episode", self.second,
                "temporal", "before", "第一件事早于第二件事"
            )
        )
        self.associations.upsert(
            AssociationDraft(
                "episode", self.second, "episode", third,
                "temporal", "before", "第二件事早于第三件事"
            )
        )

        with self.assertRaises(TemporalCycleError) as raised:
            self.associations.upsert(
                AssociationDraft(
                    "episode", third, "episode", self.first,
                    "temporal", "before", "错误地声称第三件事早于第一件事"
                )
            )

        self.assertEqual(raised.exception.earlier_id, third)
        self.assertEqual(raised.exception.later_id, self.first)
        result = ChronologyService(self.episodes, self.associations).order(
            [third, self.second, self.first]
        )
        self.assertFalse(result.has_cycle)
        self.assertEqual(result.ordered_ids, [self.first, self.second, third])

    def test_negative_temporal_relation_does_not_participate_in_cycle_guard(self):
        self.associations.upsert(
            AssociationDraft(
                "episode", self.first, "episode", self.second,
                "temporal", "before", "第一件事早于第二件事"
            )
        )
        association_id = self.associations.upsert(
            AssociationDraft(
                "episode", self.second, "episode", self.first,
                "temporal", "before", "否定第二件事早于第一件事", polarity=-1
            )
        )

        self.assertEqual(int(self.associations.get(association_id)["polarity"]), -1)
        result = ChronologyService(self.episodes, self.associations).order(
            [self.second, self.first]
        )
        self.assertFalse(result.has_cycle)
        self.assertEqual(result.ordered_ids, [self.first, self.second])

    def test_negative_relation_is_preserved(self):
        association_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "identity",
                "not_same_as",
                "两个 Episode 不是同一事件",
                polarity=-1,
            )
        )
        self.assertEqual(int(self.associations.get(association_id)["polarity"]), -1)

    def test_oversized_episode_relation_batch_is_split_and_retried(self):
        class SizeLimitedModel:
            def __init__(self):
                self.group_counts: list[int] = []

            def chat_json(self, _system, user, **_kwargs):
                if "分组：" not in user:
                    return {"relationships": []}
                groups = json.loads(user.split("分组：", 1)[1])
                ids = [int(group["current"]["id"]) for group in groups]
                self.group_counts.append(len(ids))
                if len(ids) > 1:
                    raise ModelClientError(
                        "response would exceed output limit", retryable=False
                    )
                return {
                    "episode_relationships": [
                        {"episode_id": episode_id, "relationships": []}
                        for episode_id in ids
                    ]
                }

        third = self.episodes.insert(
            self.sources.insert("第三段原始文本"),
            "main/b.json",
            0,
            EpisodeDraft("候选事件", timeline_scope="main"),
            encode_embedding(np.arange(1, 9, dtype=np.float32), 8),
        )
        model = SizeLimitedModel()
        builder = AssociationBuilder(
            model,
            self.associations,
            self.episodes,
            ConceptRepository(self.db),
            WeightConfig(),
        )

        created = builder.relate_episode_batches(
            {
                self.first: [(third, 0.8)],
                self.second: [(third, 0.7)],
            }
        )

        self.assertEqual(created, [])
        self.assertEqual(model.group_counts, [2, 1, 1])

    def test_reinforcement_retains_original_and_query_growth_reasons(self):
        base = AssociationDraft(
            "episode",
            self.first,
            "episode",
            self.second,
            "semantic",
            "evidence_bridge",
            "初始关系",
            created_reason="导入阶段 Episode 候选关系判断",
        )
        association_id = self.associations.upsert(base)
        learned = AssociationDraft(
            "episode",
            self.first,
            "episode",
            self.second,
            "semantic",
            "evidence_bridge",
            "查询再次支持该关系",
            audit_status="dual_accepted",
            claim_level="supported_inference",
            generation=1,
            created_reason="查询中自主增长：后续问题",
        )
        self.assertEqual(self.associations.upsert(learned), association_id)
        reason = str(self.associations.get(association_id)["created_reason"])
        self.assertIn("导入阶段 Episode 候选关系判断", reason)
        self.assertIn("查询中自主增长：后续问题", reason)

    def test_reversed_semantic_edge_with_same_key_is_reinforced(self):
        first_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "thematic_contrast",
                "两个场景形成对比。",
                weight=0.7,
                confidence=0.7,
            )
        )
        reversed_draft = AssociationDraft(
            "episode",
            self.second,
            "episode",
            self.first,
            "semantic",
            "thematic_contrast",
            "从另一端描述同一个对比。",
            weight=0.8,
            confidence=0.8,
        )
        self.assertEqual(self.associations.find_exact_id(reversed_draft), first_id)
        self.assertEqual(self.associations.upsert(reversed_draft), first_id)
        self.assertEqual(self.associations.stats()["edges"], 1)
        self.assertEqual(int(self.associations.get(first_id)["evidence_count"]), 2)

    def test_association_preserves_structured_growth_provenance(self):
        first_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "historical_support_context",
                "两段记忆形成受限的历史联系。",
                confidence=0.6,
                claim_level="historical_context",
                audit_status="dual_accepted",
                evidence_json=json.dumps(
                    [{"type": "episode", "id": self.first}],
                    ensure_ascii=False,
                ),
                audit_json=json.dumps(
                    [{"primary_accept": True, "adversarial_accept": True}],
                    ensure_ascii=False,
                ),
            )
        )
        row = self.associations.get(first_id)

        self.assertEqual(row["claim_level"], "historical_context")
        self.assertEqual(row["audit_status"], "dual_accepted")
        self.assertEqual(json.loads(row["evidence_json"])[0]["id"], self.first)
        self.assertTrue(json.loads(row["audit_json"])[0]["primary_accept"])

        self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "historical_support_context",
                "后来获得更直接的支持。",
                claim_level="supported_inference",
                evidence_json=json.dumps(
                    [{"type": "episode", "id": self.second}],
                    ensure_ascii=False,
                ),
            )
        )
        reinforced = self.associations.get(first_id)
        self.assertEqual(reinforced["claim_level"], "supported_inference")
        self.assertEqual(
            {item["id"] for item in json.loads(reinforced["evidence_json"])},
            {self.first, self.second},
        )

    def test_generation_tracks_closest_derivation_independently_of_confidence(self):
        association_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "inference_distance",
                "由较长推断链得到的关系。",
                confidence=0.95,
                generation=3,
                claim_level="supported_inference",
            )
        )
        row = self.associations.get(association_id)
        self.assertEqual(int(row["generation"]), 3)
        self.assertEqual(float(row["confidence"]), 0.95)

        self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "inference_distance",
                "另一条更远但可信的推断链。",
                confidence=0.99,
                generation=5,
                claim_level="supported_inference",
            )
        )
        row = self.associations.get(association_id)
        self.assertEqual(int(row["generation"]), 3)
        self.assertEqual(float(row["confidence"]), 0.99)

        self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "semantic",
                "inference_distance",
                "后来找到更接近直接经验的证明。",
                confidence=0.7,
                generation=1,
                claim_level="supported_inference",
            )
        )
        row = self.associations.get(association_id)
        self.assertEqual(int(row["generation"]), 1)
        self.assertEqual(float(row["confidence"]), 0.99)

        with self.assertRaisesRegex(ValueError, "generation"):
            self.associations.upsert(
                AssociationDraft(
                    "episode",
                    self.first,
                    "episode",
                    self.second,
                    "semantic",
                    "invalid_generation",
                    "非法层级。",
                    generation=-1,
                )
            )

    def test_episode_delete_removes_edges(self):
        association_id = self.associations.upsert(
            AssociationDraft(
                "episode",
                self.first,
                "episode",
                self.second,
                "temporal",
                "before",
                "较早事件发生在较晚事件之前",
            )
        )
        self.assertIsNotNone(self.associations.get(association_id))
        self.assertTrue(self.episodes.delete(self.first))
        self.assertIsNone(self.episodes.get(self.first))
        self.assertIsNone(self.associations.get(association_id))

    def test_schema_v1_database_is_migrated_in_place(self):
        legacy_path = Path(self.temp.name) / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        try:
            connection.executescript(
                """
                CREATE TABLE schema_meta(schema_version INTEGER NOT NULL, created_at TEXT NOT NULL);
                INSERT INTO schema_meta VALUES(1, 'legacy');
                CREATE TABLE association(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_type TEXT NOT NULL,
                    from_id INTEGER NOT NULL,
                    to_type TEXT NOT NULL,
                    to_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    relation_key TEXT NOT NULL,
                    relation_text TEXT NOT NULL,
                    polarity INTEGER NOT NULL DEFAULT 1,
                    weight REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    created_reason TEXT NOT NULL DEFAULT '',
                    last_used TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        legacy = Database(legacy_path)
        legacy.initialize()
        with legacy.connection() as connection:
            version = connection.execute(
                "SELECT schema_version FROM schema_meta"
            ).fetchone()[0]
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(association)")
            }
            episode_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(episode)")
            }
            self.assertEqual(version, SCHEMA_VERSION)
        self.assertTrue(
            {
                "claim_level",
                "audit_status",
                "evidence_json",
                "audit_json",
                "generation",
            }
            .issubset(columns)
        )
        self.assertTrue(
            {
                "evidence_quotes_json",
                "evidence_spans_json",
                "evidence_basis",
            }.issubset(episode_columns)
        )
        with legacy.connection() as connection:
            paragraph_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='paragraph'"
            ).fetchone()
        self.assertIsNotNone(paragraph_table)
        with legacy.connection() as connection:
            fts_tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name IN ('episode_fts', 'source_fts')"
                )
            }
        self.assertEqual(fts_tables, {"episode_fts", "source_fts"})

    def test_episode_persists_exact_evidence_contract(self):
        draft = EpisodeDraft(
            text="Morgan accepted the request.",
            participants=["Morgan"],
            evidence_quotes=["Morgan: accepted the request."],
            evidence_spans=[(4, 4)],
        )
        episode_id = self.episodes.insert(
            int(self.episodes.get(self.first)["source_id"]),
            "meeting.txt",
            0,
            draft,
            encode_embedding(
                np.ones(8, dtype=np.float32),
                8,
            ),
        )

        row = self.episodes.get(episode_id)

        self.assertEqual(
            json.loads(row["evidence_quotes_json"]),
            ["Morgan: accepted the request."],
        )
        self.assertEqual(json.loads(row["evidence_spans_json"]), [[4, 4]])
        self.assertEqual(
            row["evidence_basis"], "reasoning_view_nonempty_lines_v1"
        )

    def test_backup_preserves_legacy_source_schema(self):
        legacy_path = Path(self.temp.name) / "frozen-v1.db"
        target_path = Path(self.temp.name) / "evaluation-copy.db"
        connection = sqlite3.connect(legacy_path)
        try:
            connection.executescript(
                """
                CREATE TABLE schema_meta(schema_version INTEGER NOT NULL, created_at TEXT NOT NULL);
                INSERT INTO schema_meta VALUES(1, 'legacy');
                CREATE TABLE marker(value TEXT NOT NULL);
                INSERT INTO marker VALUES('unchanged');
                """
            )
            connection.commit()
        finally:
            connection.close()

        Database(legacy_path).backup_to(target_path)

        source = sqlite3.connect(legacy_path)
        try:
            self.assertEqual(
                source.execute("SELECT schema_version FROM schema_meta").fetchone()[0],
                1,
            )
            self.assertEqual(
                source.execute("SELECT value FROM marker").fetchone()[0],
                "unchanged",
            )
        finally:
            source.close()
        target = sqlite3.connect(target_path)
        try:
            self.assertEqual(
                target.execute("SELECT schema_version FROM schema_meta").fetchone()[0],
                1,
            )
            self.assertEqual(
                target.execute("SELECT value FROM marker").fetchone()[0],
                "unchanged",
            )
        finally:
            target.close()


if __name__ == "__main__":
    unittest.main()
