"""Synthetic retrieval checks; no benchmark names, desired IDs or plot rules."""
from dataclasses import asdict
import json
import unittest

from memory_demo.retrieval.recall_candidates import RecallCandidateIndex


def make_index(texts, sources=None, source_ids=None):
    episodes = {eid: {"source_id": (source_ids or {}).get(eid, eid), "text": text}
                for eid, text in texts.items()}
    raw = {episode["source_id"]: episode["text"] for episode in episodes.values()} if sources is None else sources
    return RecallCandidateIndex(episodes, raw)


class RecallCandidateTests(unittest.TestCase):
    def test_mixed_literal_clue_recovers_episode_absent_from_dense_ranking(self):
        index = make_index({1: "Budget planning meeting", 2: "Orchard gate log",
                            3: "在青铜灯塔发现了一把 violet key。"})
        ranked = index.rank_details("violet key 在青铜灯塔", dense_order=[1, 2])
        self.assertIn(3, ranked.episode_ids[:2])
        self.assertEqual(ranked.scores[3].dense_rrf, 0)
        self.assertGreater(ranked.scores[3].episode_bm25, 0)
        self.assertEqual(set(ranked.episode_ids), {1, 2, 3})

    def test_literal_source_evidence_is_searchable_when_summary_omits_it(self):
        index = make_index({1: "A short meeting summary", 2: "Routine maintenance"},
                           {1: "They left the heliograph inside the cabinet.", 2: "The engine runs normally."})
        ranked = index.rank_details("heliograph")
        self.assertEqual(ranked.episode_ids[0], 1)
        self.assertEqual(ranked.scores[1].episode_bm25, 0)
        self.assertGreater(ranked.scores[1].source_bm25, 0)

    def test_independent_weak_requirements_accumulate_without_discarding_either(self):
        index = make_index({1: "amber beacon", 2: "silver orchard", 3: "amber silver"})
        ranked = index.rank_details("", needs=["amber", "silver"])
        self.assertEqual(ranked.episode_ids[0], 3)
        self.assertEqual(ranked.scores[3].matched_queries, 2)
        self.assertEqual(ranked.scores[1].matched_queries, 1)
        self.assertEqual(ranked.scores[2].matched_queries, 1)
        self.assertGreater(ranked.scores[3].lexical_rrf, ranked.scores[1].lexical_rrf)

    def test_duplicate_queries_and_repeated_rank_entries_do_not_add_votes(self):
        index = make_index({1: "Violet beacon", 2: "orchard"})
        first = index.rank_details("violet", dense_order=[2, 1], graph_order=[1])
        repeated = index.rank_details("VIOLET", cues=[" violet ", "VIOLET"], needs=["violet"],
                                      dense_order=[2, 2, 1, 1], graph_order=[1, 1])
        self.assertEqual(first, repeated)

    def test_zero_lexical_signal_preserves_dense_order_and_all_remaining_candidates(self):
        index = make_index({1: "orchard", 2: "beacon", 3: "cabinet", 4: "engine"})
        ranked = index.rank_details("unseenquasar", dense_order=[3, 1, 999, 3])
        self.assertEqual(ranked.episode_ids, [3, 1, 2, 4])
        self.assertTrue(all(score.lexical_rrf == 0 for score in ranked.scores.values()))
        self.assertEqual(index.rank("unseenquasar", graph_order=[4, 2]), [4, 2, 1, 3])

    def test_common_word_repetition_does_not_outvote_specific_literal_clue(self):
        index = make_index({1: "common " * 5000, 2: "common quartz beacon", 3: "common maintenance"})
        ranked = index.rank_details("common quartz beacon")
        self.assertEqual(ranked.episode_ids[0], 2)
        self.assertGreater(ranked.scores[2].episode_bm25, ranked.scores[1].episode_bm25)

    def test_long_source_with_one_match_is_length_normalized(self):
        index = make_index({1: "first summary", 2: "second summary"},
                           {1: "filler " * 5000 + "heliograph", 2: "heliograph"})
        ranked = index.rank_details("heliograph")
        self.assertGreater(ranked.scores[2].source_bm25, ranked.scores[1].source_bm25)
        self.assertEqual(ranked.episode_ids[0], 2)
        self.assertEqual(len(ranked.episode_ids), 2)

    def test_many_episodes_from_one_source_do_not_create_extra_lexical_votes(self):
        base = make_index({1: "violet beacon", 2: "violet orchard"})
        texts = {1: "violet beacon", 2: "violet orchard", **{eid: "violet beacon" for eid in range(3, 53)}}
        expanded = make_index(texts, {1: "violet beacon", 2: "violet orchard"},
                              {eid: 1 for eid in range(3, 53)})
        before, after = base.rank_details("violet"), expanded.rank_details("violet")
        self.assertEqual(before.scores[1].lexical_rrf, after.scores[1].lexical_rrf)
        self.assertEqual(before.scores[2].lexical_rrf, after.scores[2].lexical_rrf)
        self.assertEqual(len(after.source_episode_ids), 2)
        self.assertEqual({after.scores[eid].source_id for eid in after.source_episode_ids}, {1, 2})
        self.assertEqual(len(after.episode_ids), 52)

    def test_each_source_representative_is_its_highest_scoring_episode(self):
        index = make_index({10: "general introduction", 11: "violet beacon", 20: "orchard"},
                           {1: "general introduction and violet beacon", 2: "orchard"},
                           {10: 1, 11: 1, 20: 2})
        ranked = index.rank_details("violet beacon")
        self.assertEqual(ranked.source_episode_ids, [11, 20])
        self.assertIn(10, ranked.episode_ids)
        self.assertEqual(index.rank("violet beacon"), ranked.episode_ids)

    def test_chinese_short_phrases_and_normalized_english_are_supported(self):
        index = make_index({1: "青铜灯塔旁的工具柜", 2: "A VIOLET key was found", 3: "例行检查"})
        self.assertEqual(index.rank("灯塔")[0], 1)
        self.assertEqual(index.rank("柜")[0], 1)
        self.assertEqual(index.rank("ｖｉｏｌｅｔ KEY")[0], 2)

    def test_empty_index_and_score_diagnostics_are_json_serializable(self):
        self.assertEqual(make_index({}).rank("any clue", dense_order=[1]), [])
        index = make_index({1: "violet beacon", 2: "routine"})
        details = index.rank_details("violet")
        json.dumps({eid: asdict(score) for eid, score in details.scores.items()}, allow_nan=False)

    def test_source_snapshot_is_immutable_and_missing_source_is_rejected(self):
        episodes, sources = {1: {"source_id": 2, "text": "old clue"}}, {2: "old clue"}
        index = RecallCandidateIndex(episodes, sources)
        episodes[1]["text"] = "newtoken"
        sources[2] = "newtoken"
        self.assertEqual(index.rank_details("newtoken").scores[1].lexical_rrf, 0)
        with self.assertRaises(ValueError):
            RecallCandidateIndex({1: {"source_id": 2, "text": "text"}}, {})


if __name__ == "__main__":
    unittest.main()
