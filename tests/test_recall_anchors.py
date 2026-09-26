import hashlib
import inspect
import unittest

from memory_demo.retrieval.recall_anchors import EpisodeAnchorIndex


def source(*texts):
    return "\n\n".join(f"[record: {10 + i * 7}]\nzh-CN: {text}" for i, text in enumerate(texts))


class EpisodeAnchorTests(unittest.TestCase):
    def index(self, raw, **texts):
        return EpisodeAnchorIndex({1: raw}, {int(eid): {"source_id": 1, "text": text} for eid, text in texts.items()})

    def test_strongest_anchor_is_not_first_record(self):
        index = self.index(source("蘋果樹下聊天", "地下通道入口", "地下通道"), **{"1": "地下通道入口"})
        anchors = index.anchors(1)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["record_index"], 17)
        self.assertGreater(anchors[0]["score"], 0)
        blocked = [(anchors[0]["context_start"], anchors[0]["context_end"])]
        self.assertIsNone(index.window(1, unavailable=blocked, max_chars=1000))

    def test_neighbour_negation_remains_in_complete_raw_window(self):
        raw = source("地下通道連到倉庫", "不，剛才的說法只是猜測。", "窗外下起大雨")
        index = self.index(raw, **{"1": "地下通道連到倉庫"})
        records = index.records(1)
        budget = records[1]["context_end"] - records[0]["context_start"]
        window = index.window(1, unavailable=[], max_chars=budget)
        self.assertEqual((window["start"], window["end"]), (records[0]["context_start"], records[1]["context_end"]))
        self.assertIn("只是猜測", raw[window["start"]:window["end"]])

    def test_same_source_different_episode_and_snapshot_identity(self):
        raw = source("紅色蘋果", "藍色海洋")
        index = self.index(raw, **{"1": "紅色蘋果", "2": "藍色海洋"})
        self.assertNotEqual(index.anchors(1)[0]["record_id"], index.anchors(2)[0]["record_id"])
        self.assertEqual(index.anchors(1)[0]["source_sha256"], index.anchors(2)[0]["source_sha256"])
        self.assertNotEqual(index.anchors(1)[0]["episode_text_sha256"], index.anchors(2)[0]["episode_text_sha256"])
        self.assertNotEqual(index.fingerprint, self.index(raw, **{"1": "紅色蘋果"}).fingerprint)

    def test_no_positive_match_and_single_character_query(self):
        index = self.index(source("紅色蘋果"), **{"1": "quantum", "2": "紅"})
        self.assertEqual(index.anchors(1), [])
        self.assertEqual(index.anchors(2), [])
        self.assertIsNone(index.window(1, unavailable=[], max_chars=100))

    def test_initialization_rejects_stale_source_hash_or_wrong_binding(self):
        raw = source("紅色蘋果")
        with self.assertRaisesRegex(ValueError, "hash"):
            EpisodeAnchorIndex({1: raw}, {1: {"source_id": 1, "text": "紅色", "source_sha256": "stale"}})
        with self.assertRaisesRegex(ValueError, "binding"):
            EpisodeAnchorIndex({1: raw}, {1: {"source_id": 2, "text": "紅色"}})
        digest = hashlib.sha256(raw.encode()).hexdigest()
        self.assertTrue(EpisodeAnchorIndex({1: raw}, {1: {"source_id": 1, "text": "紅色", "source_sha256": digest}}).anchors(1))

    def test_input_mutation_and_return_value_mutation(self):
        raw = source("紅色蘋果")
        sources, episodes = {1: raw}, {1: {"source_id": 1, "text": "紅色蘋果"}}
        index = EpisodeAnchorIndex(sources, episodes)
        index.records(1)[0]["text"] = "altered copy"
        index.anchors(1)[0]["score"] = 0
        self.assertEqual(index.records(1)[0]["text"], "紅色蘋果")
        self.assertGreater(index.anchors(1)[0]["score"], 0)
        sources[1] += "changed"
        with self.assertRaisesRegex(ValueError, "Source changed"):
            index.window(1, unavailable=[], max_chars=100)
        sources[1] = raw
        episodes[1]["text"] = "藍色海洋"
        with self.assertRaisesRegex(ValueError, "Episode changed"):
            index.anchors(1)
        with self.assertRaisesRegex(ValueError, "Episode changed"):
            index.records(1)

    def test_unavailable_record_and_separator_cannot_be_crossed(self):
        raw = source("第一個窗口", "紅色蘋果", "中間記錄", "最後窗口")
        index = self.index(raw, **{"1": "紅色蘋果"})
        records = index.records(1)
        blocked = [(records[0]["context_start"], records[0]["context_end"]),
                   (records[1]["context_end"], records[2]["context_start"])]
        window = index.window(1, unavailable=blocked, max_chars=10000)
        self.assertEqual((window["start"], window["end"]), (records[1]["context_start"], records[1]["context_end"]))

    def test_complete_record_overflow_is_explicit(self):
        raw = source("完整記錄不能被切開" * 8, "其他內容")
        index = self.index(raw, **{"1": "完整記錄不能被切開"})
        anchor = index.anchors(1)[0]
        window = index.window(1, unavailable=[], max_chars=5)
        self.assertEqual((window["start"], window["end"]), (anchor["context_start"], anchor["context_end"]))
        self.assertTrue(window["overflow"])
        self.assertEqual(window["overflow_chars"], window["end"] - window["start"] - 5)

    def test_aliases_help_locate_without_rewriting_text(self):
        raw = "[speaker_alias_legend]\nrawA: en=Alice\nrawB: en=Bob\n\n[record: 4]\n[speaker_raw: rawA]\nzh-CN: ……\n\n[record: 20]\n[speaker_raw: rawB]\nzh-CN: ……"
        index = self.index(raw, **{"1": "ＡＬＩＣＥ"})
        anchor = index.anchors(1)[0]
        self.assertEqual(anchor["record_index"], 4)
        self.assertEqual(anchor["text"], "……")
        self.assertEqual(raw[anchor["start"]:anchor["end"]], anchor["text"])

    def test_interface_has_no_question_or_gold_input(self):
        self.assertEqual(list(inspect.signature(EpisodeAnchorIndex).parameters), ["sources", "episodes"])
        self.assertEqual(list(inspect.signature(EpisodeAnchorIndex.window).parameters), ["self", "eid", "unavailable", "max_chars"])
        with self.assertRaises(TypeError):
            self.index(source("紅色蘋果"), **{"1": "紅色蘋果"}).anchors(1, question="紅色")

    def test_bad_budget_and_intervals_rejected(self):
        index = self.index(source("紅色蘋果"), **{"1": "紅色蘋果"})
        for budget in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError):
                index.window(1, unavailable=[], max_chars=budget)
        for spans in [[(True, 2)], [(3, 2)], [(-1, 1)], [(0, 9999)]]:
            with self.assertRaises(ValueError):
                index.window(1, unavailable=spans, max_chars=100)


if __name__ == "__main__":
    unittest.main()
