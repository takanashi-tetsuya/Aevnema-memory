"""Real SQLite/graph follow-up integration with scripted, offline reviewers."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.embeddings.codec import encode_embedding
from memory_demo.retrieval.progressive import ProgressiveRecall
from memory_demo.retrieval.recall_policy import RecallPolicy
import test_progressive_recall as fixtures

FOLLOWUP_CUE = "蓝鸟钟楼"
ROOT_QUOTE = "甲寄出了一封求助信，信封上写着蓝鸟钟楼。"


class FollowupModel(fixtures.EvidenceReviewModel):
    def __init__(self, root_id, target_id, clock, *, cue_mode="valid"):
        super().__init__(root_id, target_id, clock)
        self.cue_mode = cue_mode
        self.embedding_texts = []
        self.fail_followup_embedding = False
        self.expire_followup_embedding = False
        self.embedding_vectors = None

    def embed(self, cues):
        self.embedding_calls += 1
        self.embedding_texts.append(list(cues))
        if self.embedding_calls > 1 and self.fail_followup_embedding:
            raise TimeoutError("scripted follow-up embedding timeout")
        if self.embedding_calls > 1 and self.expire_followup_embedding:
            self.clock.value += 21.0
        vector = (self.embedding_vectors[self.embedding_calls - 1] if self.embedding_vectors is not None
                  else [1.0, 0.0] if self.embedding_calls == 1 else [0.0, 1.0])
        return [np.array(vector, dtype=np.float32) for _ in cues]

    def chat_json(self, system, content):
        result = super().chat_json(system, content)
        if system == fixtures.RECALL_PLAN_SYSTEM and self.cue_mode == "initial_duplicate":
            result["cues"] = [FOLLOWUP_CUE]
        if system != fixtures.RECALL_MAP_SYSTEM or self.cue_mode == "none":
            return result
        payload = json.loads(content)
        if self.cue_mode == "two_batches" and self.map_calls > 2:
            return result
        if self.cue_mode == "two_batches" and self.map_calls == 2:
            for source in payload["sources"]:
                if self.target_id in source.get("episode_ids", [source["episode_id"]]):
                    result["followup_cues"] = [{"cue": "读完甲的信", "source_id": source["source_id"],
                                                "episode_id": self.target_id, "quote": source["text"]}]
                    return result
        for source in payload["sources"]:
            identifiers = source.get("episode_ids", [source["episode_id"]])
            if self.root_id not in identifiers:
                continue
            result["followup_cues"] = [{
                "cue": "幕后黑手" if self.cue_mode == "ungrounded" else FOLLOWUP_CUE,
                "source_id": source["source_id"], "episode_id": self.root_id,
                "quote": ROOT_QUOTE,
            }]
            break
        return result


class ProgressiveFollowupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProgressiveRecallIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        # The answer has no graph path from the original seed. Intervening
        # memories remain reachable but do not match either vector cue.
        with f.db.connection() as connection:
            connection.execute("DELETE FROM association WHERE id=?", (f.edge_ids[-1],))
            for position, text in ((0, ROOT_QUOTE), (3, "乙在蓝鸟钟楼读完甲的信后答应了求助，并安排见面。")):
                connection.execute("UPDATE source SET raw_text=? WHERE id=?", (text, f.source_ids[position]))
                connection.execute("UPDATE episode SET text=? WHERE id=?", (text, f.episode_ids[position]))
            for eid in f.episode_ids[1:3]:
                connection.execute("UPDATE episode SET embedding=? WHERE id=?",
                                   (encode_embedding([-1.0, 0.0], 2), eid))
            connection.commit()
        policy = replace(RecallPolicy.for_request(f.question, "light", learn=False),
                         sources_per_review=1, wave_expansions=1)
        # Bounded waves keep the graph frontier open, so a disconnected target
        # cannot silently enter through the complete-graph vector fallback.
        policy_patch = patch.object(RecallPolicy, "for_request", return_value=policy)
        policy_patch.start()
        self.addCleanup(policy_patch.stop)
        self.model = FollowupModel(f.episode_ids[0], f.episode_ids[-1], f.clock)
        self.service = self.new_service()

    def new_service(self):
        f = self.fixture
        return ProgressiveRecall(f.config, f.db, self.model, clock=f.clock)

    def query(self, *, service=None, **kwargs):
        return (service or self.service).query(self.fixture.question, mode="light", learn=False, **kwargs)

    def checkpoint(self, result):
        return self.service.sessions.read(result["session_id"])

    def assert_target_not_discovered(self, checkpoint):
        target = self.fixture.episode_ids[-1]
        self.assertFalse(any(entry["node"] == ["episode", target]
                             for entry in checkpoint["spreading"]["supports"]))

    def test_literal_followup_adds_disconnected_target_seed_and_resolves_question(self):
        first = self.query(max_waves=1)
        self.assertEqual(first["status"], "wave_budget", first)
        initial = self.checkpoint(first)
        target = self.fixture.episode_ids[-1]
        self.assert_target_not_discovered(initial)
        self.assertNotIn(target, {seed["node_id"] for seed in initial["seeds"]})
        self.assertEqual([cue["cue"] for cue in initial["cue_queue"]["pending"]], [FOLLOWUP_CUE])
        self.assertEqual(self.model.embedding_calls, 1)

        final = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)

        self.assertEqual(final["status"], "complete", final)
        checkpoint = self.checkpoint(final)
        self.assertIn(target, {seed["node_id"] for seed in checkpoint["seeds"]})
        self.assertEqual(self.model.embedding_texts[1:], [[FOLLOWUP_CUE]])
        self.assertEqual(final["metrics"]["followup_cues_processed"], 1)
        self.assertEqual(final["metrics"]["spreading_restarts"], 1)
        self.assertEqual(checkpoint["cue_queue"]["pending"], [])
        self.assertEqual(len(checkpoint["cue_queue"]["processed"]), 1)
        self.assertIn(target, {fact["episode_id"] for fact in final["evidence"]})

    def test_without_new_cue_same_two_waves_do_not_reach_disconnected_answer(self):
        self.model.cue_mode = "none"
        result = self.query(max_waves=2)
        self.assertEqual(result["status"], "wave_budget", result)
        self.assertFalse(result["complete"])
        self.assert_target_not_discovered(self.checkpoint(result))
        self.assertEqual(self.model.embedding_calls, 1)
        self.assertEqual(result["metrics"]["spreading_restarts"], 0)

    def test_resume_after_success_does_not_reembed_processed_cue(self):
        first = self.query(max_waves=1)
        resumed = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)
        self.assertTrue(resumed["complete"], resumed)
        before_calls = self.model.embedding_calls
        before_queue = self.checkpoint(resumed)["cue_queue"]

        again = self.query(service=self.new_service(), resume=resumed["session_id"], max_waves=1)

        self.assertTrue(again["complete"], again)
        self.assertEqual(self.model.embedding_calls, before_calls)
        self.assertEqual(self.checkpoint(again)["cue_queue"], before_queue)
        self.assertEqual(sum(system == fixtures.RECALL_PLAN_SYSTEM for system, _ in self.model.calls), 1)

    def test_embedding_timeout_keeps_pending_and_can_resume_with_one_consumption(self):
        first = self.query(max_waves=1)
        before = self.checkpoint(first)
        self.model.fail_followup_embedding = True

        failed = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)

        self.assertEqual(failed["status"], "time_budget", failed)
        interrupted = self.checkpoint(failed)
        for field in ("cue_queue", "seeds", "spreading", "source_offsets", "facts", "spreading_expansion_base"):
            self.assertEqual(interrupted[field], before[field], field)
        self.assertEqual(interrupted["metrics"]["followup_cues_processed"], 0)
        self.model.fail_followup_embedding = False
        resumed = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)
        self.assertTrue(resumed["complete"], resumed)
        self.assertEqual(self.model.embedding_texts[1:], [[FOLLOWUP_CUE], [FOLLOWUP_CUE]])
        self.assertEqual(resumed["metrics"]["followup_cues_processed"], 1)
        self.assertEqual(resumed["metrics"]["spreading_restarts"], 1)

    def test_late_embedding_result_is_not_committed_or_given_a_new_epoch_deadline(self):
        first = self.query(max_waves=1)
        before = self.checkpoint(first)
        self.model.expire_followup_embedding = True
        restored = self.new_service()

        failed = self.query(service=restored, resume=first["session_id"], max_waves=1)

        self.assertEqual(failed["status"], "time_budget", failed)
        self.assertEqual(restored._deadline, 20.0)
        checkpoint = self.checkpoint(failed)
        self.assertEqual(checkpoint["cue_queue"], before["cue_queue"])
        self.assertEqual(checkpoint["seeds"], before["seeds"])
        self.assertEqual(checkpoint["metrics"]["spreading_restarts"], 0)

    def test_epoch_restart_preserves_read_sources_and_facts_and_counts_all_work(self):
        first = self.query(max_waves=1)
        before = self.checkpoint(first)
        root_sid = str(self.fixture.source_ids[0])
        first_fact_ids = {fact["fact_id"] for fact in before["facts"]}
        self.assertEqual(before["metrics"]["edge_expansions"], 1)

        resumed = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)

        self.assertTrue(resumed["complete"], resumed)
        after = self.checkpoint(resumed)
        self.assertEqual(after["source_offsets"][root_sid], before["source_offsets"][root_sid])
        self.assertTrue(first_fact_ids.issubset({fact["fact_id"] for fact in after["facts"]}))
        self.assertEqual(after["spreading_expansion_base"], 1)
        self.assertEqual(after["metrics"]["edge_expansions"], 2)
        self.assertEqual(after["metrics"]["edge_expansions"],
                         after["spreading_expansion_base"] + after["spreading"]["expansions"])
        root_read_windows = [window for trace in after["review_trace"] if trace["committed"]
                             for window in trace["windows"] if str(window["source_id"]) == root_sid]
        self.assertEqual(len(root_read_windows), 1, "original evidence may be reviewed as context without rereading its consumed Source")

    def test_ungrounded_or_initial_duplicate_cue_does_not_request_new_vector(self):
        for mode, reason in (("ungrounded", "cue_not_literal_quote_substring"),
                             ("initial_duplicate", "duplicate")):
            with self.subTest(mode=mode):
                self.model = FollowupModel(self.fixture.episode_ids[0], self.fixture.episode_ids[-1],
                                           self.fixture.clock, cue_mode=mode)
                service = self.new_service()
                result = self.query(service=service, max_waves=2)
                self.assertEqual(result["status"], "wave_budget", result)
                self.assertEqual(self.model.embedding_calls, 1)
                checkpoint = self.checkpoint(result)
                self.assertEqual(checkpoint["cue_queue"]["pending"], [])
                self.assertEqual(checkpoint["cue_queue"]["processed"], [])
                reasons = [trace["cue_validation"]["reasons"] for trace in checkpoint["review_trace"] if trace["committed"]]
                self.assertTrue(any(item.get(reason, 0) > 0 for item in reasons), reasons)

    def test_zero_positive_followup_preserves_existing_fallback_scores_and_order(self):
        f = self.fixture
        with f.db.connection() as connection:
            connection.execute("UPDATE episode SET embedding=? WHERE id=?",
                               (encode_embedding([0.6, 0.8], 2), f.episode_ids[-1]))
            connection.commit()
        self.model.embedding_vectors = [[1, 0], [0, -1]]
        first = self.query(max_waves=1)
        self.assertEqual(first["status"], "wave_budget", first)
        before = self.checkpoint(first)
        self.assertEqual(before["fallback_episode_ids"],
                         [f.episode_ids[0], f.episode_ids[3], f.episode_ids[1], f.episode_ids[2]])

        resumed = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)

        self.assertTrue(resumed["complete"], resumed)
        after = self.checkpoint(resumed)
        self.assertEqual(after["fallback_episode_scores"], before["fallback_episode_scores"])
        self.assertEqual(after["fallback_episode_ids"], before["fallback_episode_ids"])
        self.assertEqual(after["metrics"]["spreading_restarts"], 0)
        self.assertEqual(after["metrics"]["followup_cues_processed"], 1)
        self.assertEqual(self.model.embedding_calls, 2)

    def test_multiple_followup_batches_preserve_each_episodes_historical_best_score(self):
        self.model.cue_mode = "two_batches"
        self.model.scenario = "no_semantic_coverage"
        self.model.embedding_vectors = [[1, 0], [0, 1], [-1, 0]]
        first = self.query(max_waves=1)
        self.assertEqual(first["status"], "wave_budget", first)
        first_state = self.checkpoint(first)

        second = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)

        self.assertEqual(second["status"], "wave_budget", second)
        second_state = self.checkpoint(second)
        self.assertEqual([cue["cue"] for cue in second_state["cue_queue"]["pending"]], ["读完甲的信"])
        third = self.query(service=self.new_service(), resume=first["session_id"], max_waves=1)
        self.assertEqual(third["status"], "wave_budget", third)
        third_state = self.checkpoint(third)
        ids = self.fixture.episode_ids
        self.assertEqual(first_state["fallback_episode_scores"][str(ids[0])], 1.0)
        self.assertEqual(second_state["fallback_episode_scores"][str(ids[0])], 1.0)
        self.assertEqual(second_state["fallback_episode_scores"][str(ids[3])], 1.0)
        self.assertEqual(third_state["fallback_episode_scores"], {str(eid): 1.0 for eid in ids})
        self.assertEqual(third_state["fallback_episode_ids"], sorted(ids))
        self.assertEqual(set(third_state["fallback_episode_ids"]), set(first_state["fallback_episode_ids"]))
        self.assertEqual(self.model.embedding_texts[1:], [[FOLLOWUP_CUE], ["读完甲的信"]])
        self.assertEqual(third_state["metrics"]["followup_cues_processed"], 2)
        self.assertEqual(third_state["metrics"]["spreading_restarts"], 2)


if __name__ == "__main__":
    unittest.main()
