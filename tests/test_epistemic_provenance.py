from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from memory_demo.adapters.blue_archive import BlueArchiveJsonAdapter
from memory_demo.adapters.text import TextAdapter
from memory_demo.associations.growth import AssociationGrowthEngine
from memory_demo.database import Database
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.ingestion.segmenter import NaturalSegmenter
from memory_demo.llm.prompts import (
    ANSWER_SYSTEM,
    EPISODE_SYSTEM,
    GROWTH_SYSTEM,
    episode_prompt,
)
from memory_demo.config import SegmentConfig
from memory_demo.repositories.episode import EpisodeRepository
from memory_demo.types import EpisodeDraft


class EpistemicProvenanceTests(unittest.TestCase):
    def test_unannotated_document_defaults_to_asserted_not_speculative(self):
        draft = EpisodeDraft.from_dict({"text": "地球围绕太阳运行。"})
        self.assertEqual(draft.evidence_origin, "source")
        self.assertEqual(draft.epistemic_status, "asserted")
        self.assertEqual(draft.generation, 0)

    def test_episode_keeps_origin_status_and_inference_distance_orthogonal(self):
        source_guess = EpisodeDraft.from_dict(
            {
                "text": "角色推测门后可能有人。",
                "evidence_origin": "source",
                "epistemic_status": "speculative",
                "generation": 0,
            }
        )
        importer_guess = EpisodeDraft.from_dict(
            {
                "text": "导入者推测门后可能有人。",
                "evidence_origin": "importer",
                "epistemic_status": "speculative",
                "generation": 0,
            }
        )
        self.assertEqual(source_guess.generation, 0)
        self.assertEqual(importer_guess.generation, 1)
        self.assertEqual(importer_guess.evidence_origin, "importer")
        self.assertEqual(importer_guess.epistemic_status, "speculative")

    def test_text_directive_is_removed_from_content_but_reaches_reasoning_view(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text(
                '[[memory {"origin":"importer","status":"speculative",'
                '"generation":1,"note":"导入者的解释"}]]\n'
                "这可能是未花策划的。",
                encoding="utf-8",
            )
            blocks = TextAdapter("zh-CN").read_blocks(path)
            self.assertEqual(blocks[0].languages["zh-CN"], "这可能是未花策划的。")
            self.assertEqual(blocks[0].metadata["evidence_origin"], "importer")
            segments = NaturalSegmenter(SegmentConfig()).segment("notes.txt", blocks)
            source = segments[0].raw_text
            self.assertIn("[evidence_origin: importer]", source)
            self.assertIn("[epistemic_status: speculative]", source)
            compact = MemoryExtractor.compact_source_for_reasoning(source)
            self.assertIn("[evidence_origin: importer]", compact)
            self.assertIn("[epistemic_note: 导入者的解释]", compact)
            self.assertNotIn("[[memory", compact)

    def test_json_document_annotation_can_be_overridden_per_record(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "story.json"
            path.write_text(
                json.dumps(
                    {
                        "_memory": {"origin": "source", "status": "observed"},
                        "content": [
                            {"TextCn": "亲眼看到爆炸。"},
                            {
                                "TextCn": "导入者认为另有炸药。",
                                "_memory": {
                                    "origin": "importer",
                                    "status": "speculative",
                                    "generation": 1,
                                },
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            blocks = BlueArchiveJsonAdapter().read_blocks(path)
            self.assertEqual(blocks[0].metadata["evidence_origin"], "source")
            self.assertEqual(blocks[0].metadata["epistemic_status"], "observed")
            self.assertEqual(blocks[1].metadata["evidence_origin"], "importer")
            self.assertEqual(blocks[1].metadata["evidence_generation"], 1)

    def test_episode_repository_round_trips_epistemic_fields(self):
        with TemporaryDirectory() as directory:
            db = Database(Path(directory) / "memory.db")
            db.initialize()
            with db.transaction() as connection:
                source_id = int(
                    connection.execute(
                        "INSERT INTO source(raw_text) VALUES(?)",
                        ("source",),
                    ).lastrowid
                )
            repository = EpisodeRepository(db)
            episode_id = repository.insert(
                source_id,
                "notes.txt",
                0,
                EpisodeDraft.from_dict(
                    {
                        "text": "导入者推测另有原因。",
                        "evidence_origin": "importer",
                        "epistemic_status": "speculative",
                        "generation": 1,
                        "epistemic_note": "人工批注",
                    }
                ),
                b"\x00\x00\x00\x00",
            )
            row = repository.get(episode_id)
            self.assertEqual(row["evidence_origin"], "importer")
            self.assertEqual(row["epistemic_status"], "speculative")
            self.assertEqual(row["generation"], 1)
            self.assertEqual(row["epistemic_note"], "人工批注")

    def test_growth_generation_inherits_episode_distance(self):
        relation = {
            "from_type": "episode",
            "from_id": 1,
            "to_type": "episode",
            "to_id": 2,
            "_premise_edges": [{"generation": 2}],
        }
        node_map = {
            ("episode", 1): {"generation": 4},
            ("episode", 2): {"generation": 1},
        }
        self.assertEqual(
            AssociationGrowthEngine._generation(relation, node_map), 5
        )

    def test_extraction_growth_and_answer_prompts_share_the_same_boundary(self):
        extraction = episode_prompt("source", "scope")
        self.assertIn("evidence_origin", extraction)
        self.assertIn("epistemic_status", extraction)
        self.assertIn("默认使用 source+asserted", EPISODE_SYSTEM)
        self.assertIn("新边的 generation 会继承端点", GROWTH_SYSTEM)
        self.assertIn("speculative 只支持", ANSWER_SYSTEM)


if __name__ == "__main__":
    unittest.main()
