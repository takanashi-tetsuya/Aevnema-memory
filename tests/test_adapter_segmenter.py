from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memory_demo.adapters.blue_archive import (
    BlueArchiveJsonAdapter,
    clean_game_text,
    parse_korean_text,
    parse_speaker,
)
from memory_demo.adapters.text import TextAdapter
from memory_demo.config import SegmentConfig
from memory_demo.ingestion.segmenter import NaturalSegmenter, render_block
from memory_demo.types import NormalizedBlock


class AdapterSegmenterTests(unittest.TestCase):
    def test_segment_max_chars_includes_headers_and_alias_legend(self):
        blocks = [
            NormalizedBlock(
                record_index=index,
                speaker_raw=f"speaker-{index}",
                languages={"unknown": "x" * 90},
                boundary_score=2,
                metadata={"speaker_aliases": {"zh-CN": f"角色{index}"}},
            )
            for index in range(8)
        ]
        segmenter = NaturalSegmenter(
            SegmentConfig(
                target_chars=300,
                max_chars=360,
                overlap_chars=80,
                minimum_blocks=1,
            )
        )

        segments = segmenter.segment("nested/a-long-source-name.txt", blocks)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment.raw_text) <= 360 for segment in segments))

    def test_oversized_record_is_not_copied_as_overlap(self):
        blocks = [
            NormalizedBlock(
                record_index=index,
                speaker_raw=f"speaker-{index}",
                languages={"unknown": marker * 900},
                boundary_score=0,
            )
            for index, marker in enumerate(("A", "B", "C"))
        ]
        segments = NaturalSegmenter(
            SegmentConfig(
                target_chars=1_000,
                max_chars=1_200,
                overlap_chars=100,
                minimum_blocks=1,
            )
        ).segment("large-records.txt", blocks)

        self.assertEqual(len(segments), 3)
        combined = "\n".join(segment.raw_text for segment in segments)
        self.assertEqual(combined.count("A" * 100), 9)
        self.assertEqual(combined.count("B" * 100), 9)
        self.assertEqual(combined.count("C" * 100), 9)
        self.assertEqual(
            [(segment.first_record, segment.last_record) for segment in segments],
            [(0, 0), (1, 1), (2, 2)],
        )

    def test_text_adapter_supports_utf16_and_gb18030(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            utf16 = root / "utf16.txt"
            gb18030 = root / "gb18030.txt"
            utf16.write_bytes("第一段。\n\n第二段。".encode("utf-16"))
            gb18030.write_bytes("阿拜多斯。".encode("gb18030"))

            utf16_blocks = TextAdapter("zh-CN").read_blocks(utf16)
            gb_blocks = TextAdapter("zh-CN").read_blocks(gb18030)

            self.assertEqual(
                [block.languages["zh-CN"] for block in utf16_blocks],
                ["第一段。", "第二段。"],
            )
            self.assertEqual(gb_blocks[0].languages["zh-CN"], "阿拜多斯。")

    def test_text_adapter_supports_utf32_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utf32.txt"
            path.write_bytes("圣三一。".encode("utf-32"))

            blocks = TextAdapter("zh-CN").read_blocks(path)

            self.assertEqual(blocks[0].languages["zh-CN"], "圣三一。")

    def test_text_adapter_preserves_standalone_dialogue_speakers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.txt"
            path.write_text(
                "Scene note.\n\n"
                "Morgan：\nThe chair appointed me.\n\n"
                "River:\nWho made that decision?\n\n"
                "Morgan：\nThe board did.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)

            self.assertEqual(
                [block.speaker_raw for block in blocks],
                ["", "Morgan", "River", "Morgan"],
            )
            self.assertEqual(
                blocks[1].languages["en"], "The chair appointed me."
            )

    def test_text_adapter_does_not_treat_sparse_labels_as_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text(
                "Status:\nReady.\n\n"
                "This is ordinary prose.\n\n"
                "Owner:\nMorgan.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)

            self.assertTrue(all(not block.speaker_raw for block in blocks))

    def test_text_adapter_rejects_binary_nul_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.txt"
            path.write_bytes(b"not\x00text")

            with self.assertRaisesRegex(UnicodeError, "control bytes"):
                TextAdapter().read_blocks(path)

    def test_text_adapter_rejects_non_nul_binary_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.txt"
            path.write_bytes(b"not\x01text")

            with self.assertRaisesRegex(UnicodeError, "0x01"):
                TextAdapter().read_blocks(path)

    def test_text_scene_headings_become_non_overlapping_hard_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.txt"
            path.write_text(
                "--- 【Scene A】 ---\n"
                "Alex: The board appointed Morgan.\n"
                "Morgan: Understood.\n"
                "--- 【Scene B】 ---\n"
                "River: The launch was cancelled.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)
            segments = NaturalSegmenter(
                SegmentConfig(
                    target_chars=10_000,
                    max_chars=10_000,
                    overlap_chars=2_000,
                    minimum_blocks=1,
                )
            ).segment("script.txt", blocks)

            self.assertEqual(len(segments), 2)
            self.assertIn("Scene A", segments[0].raw_text)
            self.assertNotIn("Scene B", segments[0].raw_text)
            self.assertIn("Scene B", segments[1].raw_text)
            self.assertNotIn("Morgan: Understood", segments[1].raw_text)

    def test_repeated_sections_mark_short_prefix_as_front_matter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.txt"
            path.write_text(
                "Document title\n\n"
                "Compiled from earlier notes.\n\n"
                "【Fact A】\nAlpha.\n\n"
                "【Fact B】\nBeta.\n\n"
                "【Fact C】\nGamma.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)
            rendered = NaturalSegmenter(
                SegmentConfig(target_chars=10_000, max_chars=10_000)
            ).segment("knowledge.txt", blocks)

            self.assertEqual(
                [block.metadata.get("document_role") for block in blocks[:2]],
                ["front_matter", "front_matter"],
            )
            self.assertIsNone(blocks[2].metadata.get("document_role"))
            self.assertEqual(
                blocks[2].metadata.get("document_style"), "reference"
            )
            self.assertNotIn("Document title", rendered[0].raw_text)
            self.assertNotIn("Compiled from earlier notes", rendered[0].raw_text)
            self.assertIn("Fact A", rendered[0].raw_text)
            self.assertIn("[document_style: reference]", rendered[0].raw_text)

    def test_single_section_transcript_marks_prefix_and_preserves_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single-turn.txt"
            path.write_text(
                "Transcript title\nCollection 824\n\n"
                "This file contains one original dialogue turn.\n\n"
                "━━━━━━━━━━\nTurn 001  Restoration\n━━━━━━━━━━\n\n"
                "Speakers in this turn: Ayumu\n\n"
                "Ayumu：\nThank you for your work, Sensei.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)
            rendered = NaturalSegmenter(
                SegmentConfig(target_chars=10_000, max_chars=10_000)
            ).segment("single-turn.txt", blocks)

            self.assertEqual(
                [block.metadata.get("document_role") for block in blocks[:2]],
                ["front_matter", "front_matter"],
            )
            self.assertEqual(blocks[-1].speaker_raw, "Ayumu")
            self.assertNotIn("Transcript title", rendered[0].raw_text)
            self.assertIn("[speaker_raw: Ayumu]", rendered[0].raw_text)
            self.assertIn("Thank you for your work", rendered[0].raw_text)

    def test_repeated_decorated_choices_do_not_turn_dialogue_into_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogue-with-choices.txt"
            path.write_text(
                "Morgan：\nWe need to choose.\n\n"
                "【Teacher's optional answer】\nProceed carefully.\n\n"
                "River：\nI agree.\n\n"
                "【Teacher's optional answer】\nAsk for more evidence.\n\n"
                "Morgan：\nThen we have a plan.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)

            self.assertTrue(all(block.metadata.get("document_style") != "reference" for block in blocks))
            self.assertEqual(
                [block.speaker_raw for block in blocks if block.speaker_raw],
                ["Morgan", "River", "Morgan"],
            )

    def test_unicode_rule_scene_heading_becomes_hard_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.txt"
            path.write_text(
                "Morgan：\nFirst scene.\n\n"
                "━━━━━━━━━━\nScene 002\n━━━━━━━━━━\n\n"
                "River：\nSecond scene.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)

            self.assertEqual(blocks[1].boundary_score, 3)
            self.assertEqual(blocks[1].metadata.get("document_role"), "front_matter")
            self.assertEqual(blocks[2].boundary_score, 3)
            segments = NaturalSegmenter(
                SegmentConfig(
                    target_chars=10_000,
                    max_chars=10_000,
                    overlap_chars=2_000,
                    minimum_blocks=1,
                )
            ).segment("script.txt", blocks)
            self.assertEqual(len(segments), 2)
            self.assertNotIn("First scene", segments[1].raw_text)

    def test_markdown_headings_are_generic_hard_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.md"
            path.write_text(
                "# Installation\nInstall the package.\n"
                "## Removal\nRemove only the selected package.\n",
                encoding="utf-8",
            )

            blocks = TextAdapter("en").read_blocks(path)
            segments = NaturalSegmenter(
                SegmentConfig(
                    target_chars=10_000,
                    max_chars=10_000,
                    overlap_chars=2_000,
                    minimum_blocks=1,
                )
            ).segment("manual.md", blocks)

            self.assertEqual(len(segments), 2)
            self.assertIn("Installation", segments[0].raw_text)
            self.assertIn("Removal", segments[1].raw_text)

    def test_normalized_source_metadata_is_not_mistaken_for_scene_heading(self):
        sections = TextAdapter._split_structural_sections(
            "[source_key: sample.txt]\n"
            "[segment_index: 0]\n\n"
            "[record: 0]\n"
            "[speaker_raw: Morgan]\n"
            "unknown: Morgan approved the request."
        )

        self.assertEqual(len(sections), 1)
        self.assertFalse(sections[0][1])

    def test_multiline_script_uses_dialogue_actor_not_first_portrait(self):
        script = (
            "1;아리우스 학생 A;01\n"
            "5;아리우스 학생 B;01\n"
            "3;미카;02;그래서 내가 아리우스를 남몰래 지원해준 거야."
        )

        self.assertEqual(parse_speaker(script), "미카")
        self.assertEqual(
            parse_korean_text(script),
            "그래서 내가 아리우스를 남몰래 지원해준 거야.",
        )

    def test_parse_speaker_handles_narration_and_ordinary_dialogue(self):
        self.assertEqual(parse_speaker("#na;히나;출발"), "히나")
        self.assertEqual(parse_speaker("3;히나;00;출발\n#3;a"), "히나")
        self.assertEqual(parse_speaker("#all;hide"), "")
        self.assertEqual(
            parse_korean_text("5;호시노;12;저번에 한 번 붙을 뻔했잖아?\n#5;em;[!]"),
            "저번에 한 번 붙을 뻔했잖아?",
        )
        self.assertEqual(parse_korean_text("#all;hide"), "")

    def test_blue_archive_next_episode_preview_is_not_an_extractable_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "story.json"
            path.write_text(
                json.dumps(
                    {
                        "content": [
                            {
                                "ScriptKr": "#na;히나;출발",
                                "TextCn": "日奈出发。",
                            },
                            {"ScriptKr": "#all;hide"},
                            {"ScriptKr": "#continued"},
                            {
                                "ScriptKr": "#nextepisode;다음화;합숙, 시작합니다!",
                                "TextCn": "下一话；合宿，开始！",
                                "TextEn": "Next Episode; Boot Camp, Here We Come!",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            blocks = BlueArchiveJsonAdapter().read_blocks(path)
            segments = NaturalSegmenter(SegmentConfig()).segment(
                "main/story.json", blocks
            )

            self.assertEqual([block.record_index for block in blocks], [0])
            self.assertEqual(len(segments), 1)
            self.assertNotIn("Next Episode", segments[0].raw_text)
            self.assertNotIn("下一话", segments[0].raw_text)

            between_scenes = Path(directory) / "between-scenes.json"
            between_scenes.write_text(
                json.dumps(
                    {
                        "content": [
                            {"ScriptKr": "#na;日奈;前一幕", "TextCn": "前一幕。"},
                            {
                                "ScriptKr": "#nextepisode;다음화;다음 이야기",
                                "TextCn": "下一话；下一段故事。",
                            },
                            {"ScriptKr": "#na;星野;后一幕", "TextCn": "后一幕。"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            between_blocks = BlueArchiveJsonAdapter().read_blocks(between_scenes)
            between_segments = NaturalSegmenter(
                SegmentConfig(target_chars=10_000, max_chars=10_000)
            ).segment("main/between-scenes.json", between_blocks)

            self.assertEqual([block.record_index for block in between_blocks], [0, 2])
            self.assertEqual(between_blocks[1].boundary_score, 3)
            self.assertEqual(len(between_segments), 2)

            mixed_record = Path(directory) / "mixed-record.json"
            mixed_record.write_text(
                json.dumps(
                    {
                        "content": [
                            {
                                "ScriptKr": "#nextepisode;预告\n#na;日奈;实际剧情",
                                "TextCn": "日奈实际说了这句话。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            mixed_blocks = BlueArchiveJsonAdapter().read_blocks(mixed_record)
            self.assertEqual(len(mixed_blocks), 1)
            self.assertEqual(mixed_blocks[0].record_index, 0)

    def test_blue_archive_localized_control_cards_are_not_story_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = root / "controls.json"
            controls.write_text(
                json.dumps(
                    {
                        "content": [
                            {
                                "ScriptKr": "#na;미나;……후후.",
                                "TextCn": "……呵。",
                            },
                            {
                                "ScriptKr": "#all;hide",
                                "TextCn": "#st;[0,0];instant;1;",
                                "TextEn": "#st;[0,0];instant;1;",
                            },
                            {
                                "ScriptKr": "#st;[0,0];instant;1;",
                                "TextCn": "#clearST",
                                "TextEn": "#clearST",
                            },
                            {
                                "ScriptKr": "#na;미나;계속하자.",
                                "TextCn": "继续吧。",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            blocks = BlueArchiveJsonAdapter().read_blocks(controls)
            segments = NaturalSegmenter(
                SegmentConfig(target_chars=10_000, max_chars=10_000)
            ).segment("favor/controls.json", blocks)

            self.assertEqual([block.record_index for block in blocks], [0, 3])
            self.assertEqual(blocks[1].boundary_score, 3)
            self.assertEqual(len(segments), 2)
            self.assertNotIn("#clearST", "\n".join(segments[0].raw_text for _ in [0]))

            story_command = root / "story-command.json"
            story_command.write_text(
                json.dumps(
                    {
                        "content": [
                            {
                                "ScriptKr": "#na;미나;실제 대사",
                                "TextCn": "实际台词。",
                            },
                            {
                                "ScriptKr": "#st;[0,0];instant;1;",
                                "TextCn": "这不是控制文本。",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retained = BlueArchiveJsonAdapter().read_blocks(story_command)
            self.assertEqual([block.record_index for block in retained], [0, 1])

    def test_generic_json_lists_and_record_containers_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            list_path = root / "list.json"
            list_path.write_text(
                json.dumps(
                    [
                        {"title": "人物", "text": "阿洛娜是向导。"},
                        {"body": "老师来到基沃托斯。"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            records_path = root / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "summary": "该说法属于社区推测。",
                                "_memory": {"epistemic_status": "speculative"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            list_blocks = BlueArchiveJsonAdapter().read_blocks(list_path)
            record_blocks = BlueArchiveJsonAdapter().read_blocks(records_path)

            self.assertEqual(len(list_blocks), 2)
            self.assertIn("title: 人物", list_blocks[0].languages["unknown"])
            self.assertIn("阿洛娜是向导", list_blocks[0].languages["unknown"])
            self.assertEqual(
                record_blocks[0].metadata["epistemic_status"], "speculative"
            )

    def test_generic_json_without_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numbers.json"
            path.write_text('{"items": [1, 2, 3]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "usable textual JSON"):
                BlueArchiveJsonAdapter().read_blocks(path)

    def test_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            nonfinite = root / "nonfinite.json"
            duplicate.write_text(
                '{"text":"first","text":"second"}', encoding="utf-8"
            )
            nonfinite.write_text(
                '{"text":"fact","score":NaN}', encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                BlueArchiveJsonAdapter().read_blocks(duplicate)
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                BlueArchiveJsonAdapter().read_blocks(nonfinite)

    def test_generic_json_bounds_a_single_large_text_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            path.write_text(
                json.dumps({"text": "很长的事实。" * 800}, ensure_ascii=False),
                encoding="utf-8",
            )

            blocks = BlueArchiveJsonAdapter().read_blocks(path)

            self.assertGreater(len(blocks), 1)
            self.assertLessEqual(max(len(block.languages["unknown"]) for block in blocks), 2_000)

    def test_generic_json_preserves_custom_nested_and_scalar_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            path.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "name": "阿鲁",
                                "description": "便利屋68的社长。",
                                "age": 16,
                                "active": True,
                            }
                        ],
                        "world": {
                            "academy": "格黑娜",
                            "district_count": 3,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            blocks = BlueArchiveJsonAdapter().read_blocks(path)

            rendered = "\n".join(block.languages["unknown"] for block in blocks)
            self.assertIn("section: characters", rendered)
            self.assertIn("name: 阿鲁", rendered)
            self.assertIn("便利屋68的社长", rendered)
            self.assertIn("age: 16", rendered)
            self.assertIn("active: True", rendered)
            self.assertIn("academy: 格黑娜", rendered)
            self.assertIn("section: world", rendered)
            self.assertIn("district_count: 3", rendered)
            self.assertLess(rendered.index("name: 阿鲁"), rendered.index("academy: 格黑娜"))

    def test_structured_reference_documents_use_bounded_nonoverlapping_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encyclopedia.txt"
            paragraphs = ["角色百科"] + [
                f"[资料类型：角色百科 | 实体：角色{index}]\n角色{index}的事实。"
                for index in range(25)
            ]
            path.write_text("\n\n".join(paragraphs), encoding="utf-8")
            blocks = TextAdapter("zh-CN").read_blocks(path)
            segments = NaturalSegmenter(
                SegmentConfig(
                    target_chars=100_000,
                    max_chars=100_000,
                    overlap_chars=5_000,
                    minimum_blocks=1,
                    reference_max_blocks=10,
                )
            ).segment("encyclopedia.txt", blocks)

            self.assertEqual(len(segments), 3)
            combined = "\n".join(segment.raw_text for segment in segments)
            self.assertEqual(combined.count("[资料类型："), 25)
            for index in range(25):
                self.assertEqual(combined.count(f"实体：角色{index}]"), 1)

    def test_explicit_reference_theory_header_sets_evidence_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theories.txt"
            path.write_text(
                "推测资料\n\n"
                "[资料类型：推测 | 置信度：低 | 主题：循环]\n"
                "有人推测时间发生了循环。\n\n"
                "[资料类型：官方剧情事实 | 置信度：高]\n"
                "角色明确说自己来到这里。",
                encoding="utf-8",
            )

            blocks = TextAdapter("zh-CN").read_blocks(path)

            theory = blocks[1].metadata
            self.assertEqual(theory["evidence_origin"], "importer")
            self.assertEqual(theory["epistemic_status"], "speculative")
            self.assertEqual(theory["evidence_generation"], 1)
            self.assertNotIn("epistemic_status", blocks[2].metadata)

    def test_ascii_colon_reference_headers_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theories.txt"
            path.write_text(
                "[资料类型: 推测 | 置信度: 低]\n理论甲。\n\n"
                "[资料类型: 官方剧情事实 | 置信度: 高]\n事实乙。",
                encoding="utf-8",
            )

            blocks = TextAdapter("zh-CN").read_blocks(path)

            self.assertEqual(blocks[0].metadata["document_style"], "reference")
            self.assertEqual(blocks[0].metadata["epistemic_status"], "speculative")
            self.assertNotIn("epistemic_status", blocks[1].metadata)

    def test_bundled_student_catalog_adds_multilingual_speaker_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public"
            catalog = public / "config" / "yaml" / "students.yaml"
            catalog.parent.mkdir(parents=True)
            catalog.write_text(
                """- id: 10004
  familyName:
    cn: 空崎
    en: Sorasaki
    kr: 소라사키
  name:
    cn: 日奈
    jp: ヒナ
    en: Hina
    kr: 히나
""",
                encoding="utf-8",
            )
            story = public / "story" / "main" / "sample.json"
            story.parent.mkdir(parents=True)
            story.write_text(
                json.dumps(
                    {
                        "content": [
                            {"ScriptKr": "#na;히나;대사", "TextCn": "出发。"},
                            {
                                "ScriptKr": "3;통신히나;00;대사",
                                "TextCn": "通过通信继续说。",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            blocks = BlueArchiveJsonAdapter().read_blocks(story)
            self.assertEqual(blocks[0].speaker_raw, "히나")
            self.assertEqual(blocks[0].metadata["speaker_aliases"]["zh-CN"], "日奈")
            self.assertEqual(blocks[1].speaker_raw, "통신히나")
            self.assertEqual(blocks[1].metadata["speaker_aliases"]["zh-CN"], "日奈")
            rendered = render_block(blocks[0])
            self.assertIn("zh-CN=日奈", rendered)
            self.assertIn("ko=히나", rendered)
            segments = NaturalSegmenter(
                SegmentConfig(target_chars=200, max_chars=500, overlap_chars=20)
            ).segment("main/sample.json", blocks)
            self.assertEqual(segments[0].raw_text.count("zh-CN=日奈"), 2)
            self.assertIn("통신히나: zh-CN=日奈", segments[0].raw_text)
            self.assertIn("[speaker_alias_legend]", segments[0].raw_text)

    def test_cleaning_preserves_username(self):
        self.assertEqual(
            clean_game_text("[FF6666]你好，[USERNAME]。[-]"),
            "你好，[USERNAME]。",
        )

    def test_multilingual_parse_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "story.json"
            content = []
            for index in range(8):
                content.append(
                    {
                        "GroupId": 1,
                        "ScriptKr": f"#na;角色{index};대사",
                        "TextCn": f"[FF6666]中文剧情第{index}段[-]" * 5,
                        "TextJp": f"日本語{index}",
                        "TextEn": f"English {index}",
                        "TextTw": "",
                        "TextTh": "",
                        "Transition": 1 if index in {0, 4} else 0,
                        "BGName": 0,
                    }
                )
            path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
            blocks = BlueArchiveJsonAdapter().read_blocks(path)
            self.assertEqual(len(blocks), 8)
            self.assertEqual(blocks[0].speaker_raw, "角色0")
            self.assertIn("ja", blocks[0].languages)
            segmenter = NaturalSegmenter(
                SegmentConfig(
                    target_chars=250,
                    max_chars=350,
                    overlap_chars=100,
                    minimum_blocks=1,
                )
            )
            segments = segmenter.segment("main/story.json", blocks)
            self.assertGreaterEqual(len(segments), 2)
            self.assertIn("[source_key: main/story.json]", segments[0].raw_text)
            self.assertTrue(
                set(range(segments[0].first_record, segments[0].last_record + 1))
                & set(range(segments[1].first_record, segments[1].last_record + 1))
            )


if __name__ == "__main__":
    unittest.main()
