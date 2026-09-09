from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from memory_demo.association_overlay import (
    AssociationDelta,
    AssociationOverlay,
    StagedAssociationOverlay,
)
from memory_demo.config import AppConfig
from memory_demo.database import Database, transaction_liveness
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
from benchmarks.support.stage5 import (
    audit_association_delta,
    mechanism_check,
    score_answer_safety,
    score_evidence_retrieval,
)
from memory_demo.types import AssociationDraft, EpisodeDraft

from tests.helpers import FakeModel, test_config


class Stage5CausalityTests(unittest.TestCase):
    def test_staged_growth_does_not_write_until_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            sources = SourceRepository(db)
            source_id = sources.insert("原始证据")
            episodes = EpisodeRepository(db)
            blob = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            first = episodes.insert(
                source_id, "main/test.json", 0, EpisodeDraft("证据甲"), blob
            )
            second = episodes.insert(
                source_id, "main/test.json", 1, EpisodeDraft("证据乙"), blob
            )
            durable = AssociationRepository(db, config.weights)
            staged = StagedAssociationOverlay(durable)
            temporary_id = staged.upsert(
                AssociationDraft(
                    "episode",
                    first,
                    "episode",
                    second,
                    "semantic",
                    "evidence_bridge",
                    "查询综合推论：两条证据形成桥接。",
                    weight=0.7,
                    confidence=0.8,
                    generation=1,
                    claim_level="supported_inference",
                    audit_status="dual_accepted",
                )
            )

            self.assertLess(temporary_id, 0)
            self.assertEqual(durable.stats()["edges"], 0)
            self.assertIsNotNone(staged.get(temporary_id))

            staged.mark_used([temporary_id])
            mapping = staged.commit()

            self.assertIn(temporary_id, mapping)
            self.assertGreater(mapping[temporary_id], 0)
            self.assertEqual(durable.stats()["edges"], 1)
            self.assertEqual(int(durable.get(mapping[temporary_id])["use_count"]), 1)

    def test_staged_commit_rolls_back_every_prior_draft_on_later_failure(self):
        """A failed staged commit must not leave a partially learned graph."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            source_id = SourceRepository(db).insert("原始证据")
            episodes = EpisodeRepository(db)
            blob = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            first = episodes.insert(
                source_id, "main/test.json", 0, EpisodeDraft("证据甲"), blob
            )
            second = episodes.insert(
                source_id, "main/test.json", 1, EpisodeDraft("证据乙"), blob
            )
            durable = AssociationRepository(db, config.weights)
            staged = StagedAssociationOverlay(durable)
            valid_id = staged.upsert(
                AssociationDraft(
                    "episode", first, "episode", second,
                    "semantic", "valid_first", "应当回滚的有效草稿",
                    generation=0,
                )
            )
            # The overlay accepts speculative rows; the durable transaction
            # must reject the missing endpoint and roll back ``valid_id`` too.
            staged.upsert(
                AssociationDraft(
                    "episode", first, "episode", 9_999_999,
                    "semantic", "invalid_later", "不存在的终点",
                    generation=1,
                )
            )
            staged.mark_used([valid_id])

            with self.assertRaisesRegex(ValueError, "to node does not exist"):
                staged.commit()

            self.assertEqual(0, durable.stats()["edges"])

    def test_staged_commit_rolls_back_if_request_expires_before_sqlite_commit(self):
        """A liveness failure at COMMIT must publish no staged association."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            source_id = SourceRepository(db).insert("原始证据")
            episodes = EpisodeRepository(db)
            blob = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            first = episodes.insert(
                source_id, "main/test.json", 0, EpisodeDraft("证据甲"), blob
            )
            second = episodes.insert(
                source_id, "main/test.json", 1, EpisodeDraft("证据乙"), blob
            )
            durable = AssociationRepository(db, config.weights)
            staged = StagedAssociationOverlay(durable)
            temporary_id = staged.upsert(
                AssociationDraft(
                    "episode",
                    first,
                    "episode",
                    second,
                    "semantic",
                    "late_commit",
                    "不应在请求失效后提交。",
                    generation=0,
                )
            )
            staged.mark_used([temporary_id])
            checks = 0

            def require_live() -> None:
                nonlocal checks
                checks += 1
                # The first checks allow staged SQL statements to run.  The
                # transaction hook immediately before SQLite COMMIT must
                # roll all of them back once the request expires.
                if checks == 4:
                    raise TimeoutError("query deadline exceeded before commit")

            with self.assertRaisesRegex(TimeoutError, "before commit"):
                staged.commit(require_live=require_live)

            self.assertEqual(4, checks)
            self.assertEqual(0, durable.stats()["edges"])

    def test_mark_used_rolls_back_if_request_expires_before_sqlite_commit(self):
        """Usage accounting follows the same deadline-safe write rule."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            db = Database(config.database_path)
            db.initialize()
            source_id = SourceRepository(db).insert("原始证据")
            episodes = EpisodeRepository(db)
            blob = encode_embedding(
                np.ones(config.model.embedding_dimension, dtype=np.float32),
                config.model.embedding_dimension,
            )
            first = episodes.insert(
                source_id, "main/test.json", 0, EpisodeDraft("证据甲"), blob
            )
            second = episodes.insert(
                source_id, "main/test.json", 1, EpisodeDraft("证据乙"), blob
            )
            durable = AssociationRepository(db, config.weights)
            association_id = durable.upsert(
                AssociationDraft(
                    "episode",
                    first,
                    "episode",
                    second,
                    "semantic",
                    "late_utility",
                    "不应在请求失效后增加使用计数。",
                    generation=0,
                )
            )
            checks = 0

            def require_live() -> None:
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise TimeoutError("query deadline exceeded before utility commit")

            with self.assertRaisesRegex(TimeoutError, "utility commit"):
                # This mirrors a generic application finalizer: it retains
                # the historical repository method signature, while the
                # request-scoped ContextVar still fences its SQLite commit.
                with transaction_liveness(require_live):
                    durable.mark_used([association_id])

            self.assertEqual(2, checks)
            self.assertEqual(0, int(durable.get(association_id)["use_count"]))

    def _fixture(self, root: Path):
        config = test_config(root)
        config.retrieval.growth_max_rounds = 0
        story = root / "main" / "stage5.txt"
        story.parent.mkdir(parents=True)
        story.write_text(
            "阿洛娜第一次在什亭之匣中见到[USERNAME]。\n\n双方开始交流。",
            encoding="utf-8",
        )
        db = Database(config.database_path)
        db.initialize()
        model = FakeModel(config.model.embedding_dimension)
        episode_index = EmbeddingIndex(config.model.embedding_dimension)
        concept_index = EmbeddingIndex(config.model.embedding_dimension)
        pipeline = ImportPipeline(
            config,
            db,
            model,
            JsonlEventLogger(root / "logs" / "fixture.jsonl"),
            episode_index,
            concept_index,
        )
        pipeline.import_path(story)
        repositories = (
            EpisodeRepository(db),
            ConceptRepository(db),
            SourceRepository(db),
            AssociationRepository(db, config.weights),
        )
        return config, db, model, episode_index, concept_index, repositories

    def test_delta_and_overlay_hide_created_restore_reinforced_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _, _, _, repositories = self._fixture(root)
            _, _, _, associations = repositories
            base = AssociationDraft(
                from_type="episode",
                from_id=1,
                to_type="concept",
                to_id=1,
                relation_type="semantic",
                relation_key="stage5_bridge",
                relation_text="旧关系",
                weight=0.4,
                confidence=0.6,
            )
            reinforced_id = associations.upsert(base)
            before = associations.snapshot()
            associations.upsert(
                replace(
                    base,
                    relation_text="强化后的关系",
                    weight=0.9,
                    confidence=0.9,
                )
            )
            created_id = associations.upsert(
                AssociationDraft(
                    from_type="episode",
                    from_id=1,
                    to_type="concept",
                    to_id=1,
                    relation_type="semantic",
                    relation_key="stage5_created",
                    relation_text="新关系",
                    weight=0.8,
                    confidence=0.8,
                )
            )
            after = associations.snapshot()
            delta = AssociationDelta.capture(
                before, after, [created_id], [reinforced_id]
            )
            overlay = AssociationOverlay.from_delta(associations, delta)
            persisted_before_read = associations.snapshot()

            self.assertIsNone(overlay.get(created_id))
            restored = overlay.get(reinforced_id)
            self.assertEqual(restored, before[reinforced_id])
            neighbor_ids = {
                int(row["id"]) for row in overlay.neighbors("episode", 1)
            }
            self.assertNotIn(created_id, neighbor_ids)
            self.assertIn(reinforced_id, neighbor_ids)
            overlay.mark_used([reinforced_id])
            self.assertEqual(associations.snapshot(), persisted_before_read)
            self.assertEqual(
                AssociationDelta.from_dict(delta.to_dict()).to_dict(),
                delta.to_dict(),
            )

    def test_delta_treats_created_then_reinforced_as_one_creation(self):
        after = {12: {"id": 12, "weight": 0.9}}
        delta = AssociationDelta.capture({}, after, [12], [12])
        self.assertEqual([item["id"] for item in delta.created], [12])
        self.assertEqual(delta.reinforced, [])

    def test_replay_is_deterministic_and_does_not_call_model_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                config,
                _,
                model,
                episode_index,
                concept_index,
                repositories,
            ) = self._fixture(root)
            episodes, concepts, sources, associations = repositories
            engine = QueryEngine(
                config,
                model,
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                associations,
                None,
            )
            bundle = engine.build_replay_bundle("阿洛娜与老师何时建立联系？")
            snapshot = associations.snapshot()

            class NoModelCalls:
                logger = None

                def embed(self, *_args, **_kwargs):
                    raise AssertionError("replay unexpectedly used model.embed")

                def chat_json(self, *_args, **_kwargs):
                    raise AssertionError("replay unexpectedly used model.chat_json")

                def chat_text(self, *_args, **_kwargs):
                    raise AssertionError("replay unexpectedly used model.chat_text")

            replay_engine = QueryEngine(
                config,
                NoModelCalls(),
                episode_index,
                concept_index,
                episodes,
                concepts,
                sources,
                AssociationOverlay(associations),
                None,
            )
            first = replay_engine.replay_retrieval(bundle)
            second = replay_engine.replay_retrieval(bundle)

            self.assertEqual(first, second)
            self.assertEqual(associations.snapshot(), snapshot)
            self.assertEqual(len(bundle["initial_query_embeddings_float32"][0]), 8)
            self.assertTrue(first["episode_ids"])

    def test_replay_rejects_configuration_drift(self):
        engine = object.__new__(QueryEngine)
        engine.config = AppConfig()
        with self.assertRaises(ValueError):
            engine.replay_retrieval(
                {
                    "version": 1,
                    "configuration": {},
                    "question": "x",
                    "final_seed_hits": [],
                }
            )

    def test_stage5_score_ignores_association_growth_counts(self):
        criterion = {
            "required_episode_groups": [[1, 2], [3]],
            "required_sources": ["main/a.json"],
        }
        base = {
            "episode_ids": [1, 3],
            "candidate_episode_ids": [1, 3, 4],
            "evidence_episodes": [{"id": 1, "source_key": "main/a.json"}],
        }
        noisy = {
            **base,
            "new_association_ids": list(range(1000)),
            "reinforced_association_ids": list(range(1000, 2000)),
            "reuse_count": 999999,
        }
        self.assertEqual(
            score_evidence_retrieval(base, criterion),
            score_evidence_retrieval(noisy, criterion),
        )

    def test_historical_context_delta_is_safe_but_external_source_fails(self):
        row = {
            "id": 9,
            "claim_level": "historical_context",
            "audit_status": "dual_accepted",
            "relation_key": "historical_bridge",
            "relation_text": "有证据的历史联系",
            "evidence_json": (
                '[{"source_key":"main/a.json"},{"source_key":"main/b.json"}]'
            ),
            "audit_json": (
                '[{"primary_accept":true,"adversarial_accept":true}]'
            ),
        }
        delta = AssociationDelta(created=[{"id": 9, "before": None, "after": row}], reinforced=[])
        self.assertTrue(
            audit_association_delta(delta, {"main/a.json", "main/b.json"})[
                "passed"
            ]
        )
        self.assertFalse(
            audit_association_delta(delta, {"main/a.json"})["passed"]
        )

    def test_mechanism_requires_delta_path_and_mask_change(self):
        criterion = {
            "required_episode_groups": [[1], [2]],
            "required_sources": [],
        }
        row = {"id": 10}
        delta = AssociationDelta(
            created=[{"id": 10, "before": None, "after": row}], reinforced=[]
        )
        treatment = {
            "episode_ids": [1, 2],
            "candidate_episode_ids": [1, 2],
            "association_paths": [
                {
                    "association_id": 10,
                    "from": ["episode", 1],
                    "to": ["episode", 2],
                }
            ],
        }
        masked = {
            "episode_ids": [1],
            "candidate_episode_ids": [1],
            "association_paths": [],
        }
        result = mechanism_check(treatment, masked, delta, criterion)
        self.assertTrue(result["delta_used_in_treatment_paths"])
        self.assertTrue(result["delta_paths_touch_required_episode"])
        self.assertTrue(result["rank_or_selection_changed_after_mask"])

    def test_missing_label_term_is_incomplete_but_not_unsafe(self):
        result = {
            "answer": "古圣堂与阿里乌斯、地下墓穴和伊甸条约有关，属于推测，机制未知。",
            "answer_audits": [{"valid": True}],
            "evidence_episodes": [{"source_key": "main/a.json"}],
        }
        criterion = {
            "required_terms": ["古圣堂", "阿里乌斯", "地下墓穴", "伊甸条约"],
            "required_term_groups": [
                ["事实", "原文"],
                ["推测"],
                ["未知"],
            ],
        }
        score = score_answer_safety(result, criterion, {"main/a.json"})
        self.assertTrue(score["safety_passed"])
        self.assertFalse(score["semantic_terms_passed"])
        self.assertFalse(score["passed"])


if __name__ == "__main__":
    unittest.main()
