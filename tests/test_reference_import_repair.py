from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from benchmarks.repair_explicit_evidence import repair_explicit_evidence
from memory_demo.database import Database
from memory_demo.repositories.episode import EpisodeRepository
from memory_demo.repositories.association import AssociationRepository
from memory_demo.repositories.concept import ConceptRepository
from memory_demo.repositories.source import SourceRepository
from memory_demo.types import AssociationDraft, ConceptDraft, EpisodeDraft


class ReferenceImportRepairTests(unittest.TestCase):
    def test_title_extra_is_removed_before_mixed_record_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            database = Database(path)
            database.initialize()
            source_id = SourceRepository(database).insert(
                "[record: 0]\nunknown: 《资料标题》\n\n"
                "[record: 1]\nunknown: [资料类型：官方剧情事实]\n事实甲\n\n"
                "[record: 2]\nunknown: [资料类型：剧情解读]\n推论乙"
            )
            episodes = EpisodeRepository(database)
            meta_id = episodes.insert(
                source_id,
                "mixed.txt",
                0,
                EpisodeDraft(
                    text="《资料标题》作为作品标题被正式确认。",
                    event_type="作品标题确认",
                ),
                b"\x00\x00\x00\x00",
            )
            fact_id = episodes.insert(
                source_id,
                "mixed.txt",
                1,
                EpisodeDraft(text="事实甲。", event_type="剧情事实"),
                b"\x00\x00\x00\x00",
            )
            inference_id = episodes.insert(
                source_id,
                "mixed.txt",
                2,
                EpisodeDraft(text="推论乙。", event_type="剧情解读"),
                b"\x00\x00\x00\x00",
            )
            title_concept_id = ConceptRepository(database).insert(
                ConceptDraft(
                    canonical_name="资料标题",
                    description="纯标题副产物",
                    embedding_text="资料标题",
                ),
                b"\x00\x00\x00\x00",
            )
            AssociationRepository(database).upsert(
                AssociationDraft(
                    from_type="episode",
                    from_id=meta_id,
                    to_type="concept",
                    to_id=title_concept_id,
                    relation_type="semantic",
                    relation_key="involves",
                    relation_text="标题涉及标题概念",
                )
            )

            preview = repair_explicit_evidence(path)

            self.assertEqual(preview["ambiguous_source_ids"], [])
            self.assertEqual(preview["meta_episode_ids"], [meta_id])
            self.assertEqual(preview["orphaned_concept_ids"], [title_concept_id])
            self.assertEqual(preview["episode_ids"], [inference_id])

            applied = repair_explicit_evidence(path, apply=True)
            self.assertTrue(applied["applied"])
            with database.connection() as connection:
                remaining = connection.execute(
                    "SELECT id, evidence_origin, epistemic_status, generation "
                    "FROM episode ORDER BY id"
                ).fetchall()
                fts_ids = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT rowid FROM episode_fts ORDER BY rowid"
                    )
                }
                concept_exists = connection.execute(
                    "SELECT 1 FROM concept WHERE id = ?", (title_concept_id,)
                ).fetchone()
            self.assertEqual([int(row["id"]) for row in remaining], [fact_id, inference_id])
            self.assertEqual(remaining[0]["evidence_origin"], "source")
            self.assertEqual(remaining[0]["generation"], 0)
            self.assertEqual(remaining[1]["evidence_origin"], "importer")
            self.assertEqual(remaining[1]["epistemic_status"], "speculative")
            self.assertEqual(remaining[1]["generation"], 1)
            self.assertEqual(fts_ids, {fact_id, inference_id})
            self.assertIsNone(concept_exists)


if __name__ == "__main__":
    unittest.main()
