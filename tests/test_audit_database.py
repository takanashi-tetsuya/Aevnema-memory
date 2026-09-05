from __future__ import annotations

import json
import unittest

from benchmarks.audit_database import audit_episode_participants


class EpisodeParticipantAuditTests(unittest.TestCase):
    def test_detects_unsupported_participant_without_name_dictionary(self):
        episodes = [
            {
                "id": 7,
                "source_id": 3,
                "text": "甲向乙报告。",
                "participants_json": json.dumps(["甲", "模型补出的丙"]),
            }
        ]

        grounding, completeness = audit_episode_participants(
            episodes,
            {3: "[speaker_raw: 甲]\n甲: 我向乙报告。\n[speaker_raw: 乙]\n乙: 收到。"},
        )

        self.assertEqual(grounding[0]["participant"], "模型补出的丙")
        self.assertEqual(completeness, [])

    def test_detects_source_speaker_named_in_text_but_missing_from_participants(self):
        episodes = [
            {
                "id": 9,
                "source_id": 4,
                "text": "角色甲向角色乙报告后，角色乙同意调查。",
                "participants_json": json.dumps(["角色甲"]),
            }
        ]

        grounding, completeness = audit_episode_participants(
            episodes,
            {
                4: "[speaker_raw: 角色甲]\n角色甲: 我来报告。\n"
                "[speaker_raw: 角色乙 / Character B]\n角色乙 / Character B: 我会调查。"
            },
        )

        self.assertEqual(grounding, [])
        self.assertEqual(
            completeness[0]["speaker_label"], "角色乙 / Character B"
        )

    def test_unknown_participant_is_allowed(self):
        episodes = [
            {
                "id": 11,
                "source_id": 5,
                "text": "未标注发言者提出建议。",
                "participants_json": json.dumps(["未标注发言者"]),
            }
        ]

        grounding, completeness = audit_episode_participants(
            episodes,
            {5: "unknown: 提出建议。"},
        )

        self.assertEqual(grounding, [])
        self.assertEqual(completeness, [])

    def test_completeness_uses_episode_evidence_not_other_source_speakers(self):
        episodes = [
            {
                "id": 12,
                "source_id": 6,
                "text": "阿罗娜说她准备联系リン。",
                "participants_json": json.dumps(["アロナ / 阿罗娜"]),
                "evidence_quotes_json": json.dumps(
                    ["アロナ(阿罗娜): 連邦生徒会のリンさんに連絡します。"]
                ),
            }
        ]

        grounding, completeness = audit_episode_participants(
            episodes,
            {
                6: "アロナ(阿罗娜): 連邦生徒会のリンさんに連絡します。\n"
                "リン: 承知しました。"
            },
        )

        self.assertEqual(grounding, [])
        self.assertEqual(completeness, [])

    def test_completeness_rejects_same_script_name_substrings(self):
        episodes = [
            {
                "id": 13,
                "source_id": 7,
                "text": "ウミカが支援を申し出た。",
                "participants_json": json.dumps(["ウミカ"]),
                "evidence_quotes_json": "[]",
            }
        ]

        grounding, completeness = audit_episode_participants(
            episodes,
            {7: "ミカ: 別の場面。\nウミカ: 支援します。"},
        )

        self.assertEqual(grounding, [])
        self.assertEqual(completeness, [])


if __name__ == "__main__":
    unittest.main()
