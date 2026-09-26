"""Offline tests for grounded follow-up scheduling and graph epoch transitions."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import unittest

import numpy as np

from memory_demo.associations.spreading import SpreadingRecall, SpreadingSeed
from memory_demo.retrieval.recall_cues import (
    CueLimits, enqueue_followup_cues, expand_pending_cues,
    pending_cue_texts, restart_spreading_epoch,
)


class RecallCueTests(unittest.TestCase):
    def setUp(self):
        self.sources = {1: "序言。甲从档案室找到蓝色信封。尾声。", 2: "乙认出蓝色信封来自旧校舍。", 3: "乙联系了丙。"}
        self.episodes = {11: {"source_id": 1}, 12: {"source_id": 1},
                         21: {"source_id": 2}, 31: {"source_id": 3}}
        self.visible = [{"source_id": 1, "episode_id": 11, "start": 3, "end": 16,
                         "text": self.sources[1][3:16]}]
        self.proposal = {"cue": "蓝色信封", "source_id": 1, "episode_id": 11,
                         "quote": "甲从档案室找到蓝色信封。"}
        self.matrix = np.array([[1, 0], [1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)

    def enqueue(self, proposed=None, queue=None, **kwargs):
        return enqueue_followup_cues(queue, [self.proposal] if proposed is None else proposed,
                                     visible=self.visible, sources=self.sources,
                                     episodes=self.episodes, **kwargs)

    def root(self, sid):
        return "source:" + hashlib.sha256(self.sources[sid].encode("utf-8")).hexdigest()

    def expand(self, queue, vectors, **kwargs):
        options = {"episode_ids": [11, 12, 21, 31], "episode_matrix": self.matrix,
                   "episodes": self.episodes, "sources": self.sources,
                   "existing_seeds": [SpreadingSeed("episode", 11, 0.7, self.root(1))],
                   "seeds_per_cue": 1}
        options.update(kwargs)
        return expand_pending_cues(queue, vectors, **options)

    def test_literal_phrase_keeps_absolute_evidence_and_reports_counts(self):
        result = self.enqueue()
        self.assertEqual(result.stats, {"accepted": 1, "rejected": 0, "reasons": {}})
        cue = result.queue["pending"][0]
        self.assertEqual(self.sources[1][cue["start"]:cue["end"]], cue["quote"])
        self.assertEqual(pending_cue_texts(result.queue), ["蓝色信封"])
        self.assertEqual(json.loads(json.dumps(result.queue, ensure_ascii=False)), result.queue)

    def test_rejects_guessed_answer_wrong_source_and_unseen_episode(self):
        values = [{**self.proposal, "cue": "信封证明丙是幕后黑手"},
                  {**self.proposal, "source_id": 2},
                  {**self.proposal, "episode_id": 12},
                  {**self.proposal, "quote": "不存在的蓝色信封"}]
        result = self.enqueue(values)
        self.assertEqual(result.queue["pending"], [])
        self.assertEqual(result.stats["accepted"], 0)
        self.assertEqual(result.stats["rejected"], 4)
        self.assertEqual(result.stats["reasons"]["cue_not_literal_quote_substring"], 1)
        self.assertEqual(result.stats["reasons"]["source_episode_mismatch"], 1)
        self.assertEqual(result.stats["reasons"]["quote_not_uniquely_visible"], 2)

    def test_merged_visible_window_allows_only_explicit_same_source_episodes(self):
        self.visible[0]["episode_ids"] = [11, 12, 21]
        result = self.enqueue([{**self.proposal, "episode_id": 12},
                               {**self.proposal, "episode_id": 21, "cue": "档案室"}])
        self.assertEqual(result.stats["accepted"], 1)
        self.assertEqual(result.queue["pending"][0]["episode_id"], 12)
        self.assertEqual(result.stats["reasons"], {"source_episode_mismatch": 1})

    def test_forged_visible_text_and_ambiguous_quotes_are_rejected(self):
        self.visible[0]["text"] += "伪造"
        self.assertEqual(self.enqueue().stats["rejected"], 1)
        self.sources[1] = "蓝色信封蓝色信封"
        self.visible = [{"source_id": 1, "episode_id": 11, "start": 0,
                         "end": len(self.sources[1]), "text": self.sources[1]}]
        self.proposal["quote"] = "蓝色信封"
        self.assertEqual(self.enqueue().stats["rejected"], 1)

    def test_overlapping_windows_with_one_absolute_span_are_not_ambiguous(self):
        self.visible.append({"source_id": 1, "episode_id": 11, "start": 0,
                             "end": len(self.sources[1]), "text": self.sources[1]})
        self.assertEqual(self.enqueue().stats["accepted"], 1)

    def test_deduplicates_against_initial_pending_and_processed_cues(self):
        excluded = self.enqueue(initial_cues=["蓝色信封"])
        self.assertEqual(excluded.stats["reasons"], {"duplicate": 1})
        first = self.enqueue()
        repeated = self.enqueue(queue=first.queue)
        self.assertEqual(repeated.stats["reasons"], {"duplicate": 1})
        done = self.expand(first.queue, [[0, 1]])
        roundtrip = json.loads(json.dumps(done.queue))
        self.assertEqual(pending_cue_texts(roundtrip), [])
        self.assertEqual(self.enqueue(queue=roundtrip).stats["reasons"], {"duplicate": 1})
        self.sources[1] = "ＡＢＣ abc"
        self.visible = [{"source_id": 1, "episode_id": 11, "start": 0,
                         "end": len(self.sources[1]), "text": self.sources[1]}]
        variants = [{**self.proposal, "cue": text, "quote": text} for text in ("ＡＢＣ", "abc")]
        self.assertEqual(self.enqueue(variants).stats["reasons"], {"duplicate": 1})

    def test_total_limit_includes_processed_cues_and_batch_limit_is_separate(self):
        proposals = [{**self.proposal, "cue": cue} for cue in ("档案室", "蓝色信封", "找到")]
        limits = CueLimits(max_total=2, max_per_wave=1)
        first = self.enqueue(proposals, limits=limits)
        self.assertEqual(first.stats["reasons"], {"capacity_reached": 2})
        processed = self.expand(first.queue, [[0, 1]])
        second = self.enqueue(proposals[1:], queue=processed.queue, limits=limits)
        self.assertEqual(len(second.queue["pending"]), 1)
        third = self.enqueue(proposals, queue=second.queue, limits=limits)
        self.assertEqual(third.stats["reasons"], {"capacity_reached": 3})
        self.assertEqual(len(third.queue["pending"]) + len(third.queue["processed"]), 2)

    def test_invalid_fields_have_diagnostics_and_length_limits(self):
        self.assertEqual(self.enqueue("bad").stats["reasons"], {"invalid_container": 1})
        result = self.enqueue([None, {}, {**self.proposal, "cue": "甲"},
                               {**self.proposal, "source_id": True}])
        self.assertEqual(result.stats["rejected"], 4)
        self.assertEqual(sum(result.stats["reasons"].values()), 4)
        self.assertEqual(self.enqueue(limits=CueLimits(max_cue_chars=3)).stats["reasons"], {"text_length": 1})
        self.assertEqual(self.enqueue(limits=CueLimits(max_quote_chars=3)).stats["reasons"], {"text_length": 1})

    def test_new_vector_finds_new_source_without_reembedding_old_cues(self):
        queue = self.enqueue().queue
        model_calls = []

        def embed(texts):
            model_calls.append(texts)
            return [[0, 1] for _ in texts]

        result = self.expand(queue, embed(pending_cue_texts(queue)))
        self.assertEqual(model_calls, [["蓝色信封"]])
        self.assertEqual({s.node_id for s in result.seeds}, {11, 21})
        self.assertEqual(result.ranked_episode_ids[0], 21)
        self.assertEqual(result.processed_count, 1)
        self.assertEqual(result.scored_count, 4)
        self.assertTrue(result.changed)
        self.assertEqual(result.queue["pending"], [])
        self.assertEqual(len(queue["pending"]), 1, "transition must not mutate its input")
        empty = self.expand(result.queue, [], existing_seeds=result.seeds)
        self.assertFalse(empty.changed)
        self.assertEqual(empty.scored_count, 0)

    def test_failed_embedding_result_does_not_consume_pending_or_mutate_matrix(self):
        queue = self.enqueue().queue
        before = deepcopy(queue)
        matrix_before = self.matrix.copy()
        for vectors in ([], [[float("nan"), 1]], [[0, 0]], [[1, 0, 0]]):
            with self.assertRaises(ValueError):
                self.expand(queue, vectors)
            self.assertEqual(queue, before)
            np.testing.assert_array_equal(self.matrix, matrix_before)
        self.sources[1] += "改写"
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.expand(queue, [[0, 1]])
        self.assertEqual(queue, before)

    def test_repeated_source_roots_use_max_strength_not_extra_independence(self):
        queue = self.enqueue().queue
        old = [SpreadingSeed("episode", 11, 0.7, "question:one"),
               SpreadingSeed("episode", 11, 0.5, "question:two")]
        result = self.expand(queue, [[1, 0]], existing_seeds=old, seeds_per_cue=2)
        self.assertEqual(len(result.seeds), 2)
        self.assertEqual({seed.root_id for seed in result.seeds}, {self.root(1)})
        self.assertEqual(next(seed.activation for seed in result.seeds if seed.node_id == 11), 1.0)
        graph = SpreadingRecall([("episode", 11), ("episode", 12)], [
            {"id": 1, "from_type": "episode", "from_id": 11, "to_type": "episode", "to_id": 12, "weight": 0.5}])
        recalled = graph.search(result.seeds)
        self.assertTrue(all(len(node.contributions) == 1 for node in recalled.nodes))

    def test_tied_matrix_rows_use_episode_id_as_secondary_order(self):
        queue = self.enqueue().queue
        result = self.expand(queue, [[1, 0]], episode_ids=[12, 11, 21, 31])
        self.assertEqual(result.ranked_episode_ids[:2], [11, 12])

    def test_batch_scores_cover_all_episodes_and_take_best_positive_similarity(self):
        proposals = [self.proposal, {**self.proposal, "cue": "档案室"}]
        queue = self.enqueue(proposals).queue
        result = self.expand(queue, [[1, 0], [0, 1]])
        self.assertEqual(set(result.episode_scores), {11, 12, 21, 31})
        self.assertEqual([result.episode_scores[eid] for eid in (11, 12, 21)], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(result.episode_scores[31], 0.8)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in result.episode_scores.values()))
        self.assertEqual(result.ranked_episode_ids, [11, 12, 21, 31])
        self.assertEqual(result.scored_count, 8)

    def test_nonpositive_batch_returns_explicit_zero_scores_and_empty_batch_no_scores(self):
        queue = self.enqueue().queue
        result = self.expand(queue, [[0, -1]])
        self.assertEqual(result.episode_scores, {11: 0.0, 12: 0.0, 21: 0.0, 31: 0.0})
        self.assertFalse(result.changed)
        empty = self.expand(result.queue, [], existing_seeds=result.seeds)
        self.assertEqual(empty.episode_scores, {})

    def test_float_roundoff_above_unit_similarity_is_clamped_for_persisted_scores(self):
        matrix = self.matrix.copy()
        matrix[0, 0] = np.nextafter(np.float32(1), np.float32(2))
        result = self.expand(self.enqueue().queue, [[1, 0]], episode_matrix=matrix)
        self.assertEqual(result.episode_scores[11], 1.0)

    def test_epoch_restart_preserves_evidence_deadline_and_counts_repeated_work(self):
        graph = SpreadingRecall([("episode", i) for i in (11, 21, 31)], [
            {"id": 1, "from_type": "episode", "from_id": 11, "to_type": "episode", "to_id": 21, "weight": 0.5},
            {"id": 2, "from_type": "episode", "from_id": 21, "to_type": "episode", "to_id": 31, "weight": 0.5}])
        old = [SpreadingSeed("episode", 11, 0.7, self.root(1))]
        checkpoint = graph.search(old, max_expansions=1).checkpoint
        updated = self.expand(self.enqueue().queue, [[0, 1]], existing_seeds=old).seeds
        with self.assertRaises(ValueError):
            graph.search(updated, checkpoint=checkpoint)
        state = {"seeds": [asdict(s) for s in old], "spreading": checkpoint,
                 "source_offsets": {"1": len(self.sources[1])}, "facts": [{"quote": "已验证原文"}],
                 "used_edge_ids": [1], "elapsed_seconds": 58.0, "deadline": 360.0,
                 "metrics": {"edge_expansions": 1, "model_batches": 2}}
        before = deepcopy(state)
        restarted = restart_spreading_epoch(state, updated)
        self.assertEqual(state, before)
        for field in ("source_offsets", "facts", "used_edge_ids", "elapsed_seconds", "deadline"):
            self.assertEqual(restarted[field], state[field])
        self.assertIsNone(restarted["spreading"])
        self.assertEqual(restarted["spreading_expansion_base"], 1)
        self.assertEqual(restarted["metrics"]["spreading_restarts"], 1)
        next_wave = graph.search(updated, max_expansions=2)
        restarted["spreading"] = next_wave.checkpoint
        restarted["metrics"]["edge_expansions"] = restarted["spreading_expansion_base"] + next_wave.expansions
        newer = [*updated, SpreadingSeed("episode", 31, 0.5, self.root(3))]
        again = restart_spreading_epoch(restarted, newer)
        self.assertEqual(again["spreading_expansion_base"], 3)
        self.assertEqual(again["metrics"]["edge_expansions"], 3)
        self.assertEqual(restart_spreading_epoch(again, newer), again)


if __name__ == "__main__":
    unittest.main()
