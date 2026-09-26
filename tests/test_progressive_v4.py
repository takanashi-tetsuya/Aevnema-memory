"""Focused review must retain provenance and reject older checkpoints."""
import json
import unittest
from config.prompt_config import recall_v3_prompts, recall_v4_prompts
from memory_demo.retrieval.progressive_v4 import ProgressiveRecallV4
import test_progressive_v3 as fixtures


class FocusedReviewer(fixtures.Reviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.actual = []

    def chat_json(self, system, content):
        self.actual.append((system, (getattr(content, "structured_payload", None) or json.loads(content))))
        if system == recall_v4_prompts.MAP:
            system = recall_v3_prompts.MAP
        elif system == recall_v4_prompts.NEED:
            system = recall_v3_prompts.NEED
        return super().chat_json(system, content)


class ProgressiveV4Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.old_service = self.service
        self.model = FocusedReviewer(self.clock)
        self.service = ProgressiveRecallV4(self.config, self.db, self.model, clock=self.clock)

    def test_current_target_and_original_qualifiers_are_in_actual_trace(self):
        question = "求助信后来怎么样？只谈收到当晚，不推断此后。"
        result = self.service.query(question, learn=False)
        self.assertEqual(result["review_protocol_version"], 4)
        self.assertTrue(result["complete"], result)
        state = self.service.sessions.read(result["session_id"])
        calls = [(system, payload) for system, payload in self.model.actual
                 if system == recall_v4_prompts.NEED]
        self.assertEqual(len(calls), 1)
        payload = calls[0][1]
        self.assertEqual(payload["question"], state["needs"][0])
        self.assertNotIn("needs", payload)
        self.assertIn(question, json.dumps(payload["request_context"], ensure_ascii=False))
        traces = [t for t in state["review_trace"] if t["stage"] == "need"]
        self.assertEqual(traces[0]["request"], payload)
        self.assertEqual({f["episode_id"] for f in result["evidence"]}, set(self.eids))

    def test_v3_checkpoint_cannot_be_silently_reinterpreted_as_v4(self):
        result = self.old_service.query("求助信后来怎么样？", learn=False)
        before = len(self.model.actual)
        with self.assertRaisesRegex(ValueError, "another review protocol"):
            self.service.query("求助信后来怎么样？", resume=result["session_id"], learn=False)
        self.assertEqual(len(self.model.actual), before)


if __name__ == "__main__":
    unittest.main()
