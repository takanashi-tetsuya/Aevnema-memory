"""Stage failures over real SQLite must not erase or certify pending evidence."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from config.prompt_config import recall_v3_prompts as prompts
from memory_demo.database import Database
from memory_demo.embeddings.codec import encode_embedding
from memory_demo.repositories import AssociationRepository, EpisodeRepository, SourceRepository
from memory_demo.retrieval.progressive_v3 import ProgressiveRecallV3
from memory_demo.types import AssociationDraft, EpisodeDraft


class Clock:
    value = 0.0
    def __call__(self):
        return self.value


class Reviewer:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.fact_delay = 0
        self.invalid_need = False
        self.with_link = False
        self.link_delay = 0
        self.reject_first = False

    def embed(self, cues):
        return [np.array([1., 0.], dtype=np.float32) for _ in cues]

    def chat_json(self, system, content):
        p = (getattr(content, "structured_payload", None) or json.loads(content))
        self.calls.append(system)
        if system == prompts.PLAN:
            return {"needs": ["求助信得到什么回应"], "cues": ["求助信"]}
        if system == prompts.MAP:
            eids = [r["episode_ids"][0] for r in p["records"]]
            return {"facts": [{"record_ids": [r["id"]], "episode_id": r["episode_ids"][0],
                "interpretation": "原文中的求助或答复", "need_indices": [0]} for r in p["records"]],
                "links": [{"from_episode_id": eids[0], "to_episode_id": eids[1], "rationale": "求助与回应"}]
                if self.with_link and len(eids) == 2 else [], "followup_cues": []}
        if system == prompts.FACTS:
            self.clock.value += self.fact_delay
            self.fact_delay = 0
            return {"fact_decisions": [{"fact_id": f["fact_id"], "decision": "reject" if self.reject_first and i == 0 else "accept",
                "statement": "原文所述事件", "reason": "按离线fixture审查相关性", "episode_alignment": "supported"} for i, f in enumerate(p["facts"])]}
        if system == prompts.NEED:
            if self.invalid_need:
                return {"status": "supported", "fact_ids": ["F999"], "answer": "错误引用", "reason": "错误引用"}
            return {"status": "supported", "fact_ids": [f["fact_id"] for f in p["facts"]],
                    "answer": "乙答应帮助", "reason": "求助和回应均有记录"}
        if system == prompts.LINKS:
            self.clock.value += self.link_delay
            self.link_delay = 0
            return {"link_decisions": [{"link_id": link["link_id"], "decision": "accept",
                    "reason": "两端原文记录求助及回应"} for link in p["links"]]}
        raise AssertionError("unexpected review stage")


class ProgressiveV3Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        self.db = Database(path / "memory.sqlite")
        self.db.initialize()
        sources, episodes = SourceRepository(self.db), EpisodeRepository(self.db)
        self.eids = []
        for i, text in enumerate(("甲寄出了求助信。", "乙读完信后答应帮助甲。")):
            sid = sources.insert(text)
            self.eids.append(episodes.insert(sid, f"example/{i}", 0, EpisodeDraft(text),
                encode_embedding([1., 0.] if i == 0 else [0., 1.], 2)))
        AssociationRepository(self.db).upsert(AssociationDraft("episode", self.eids[0],
            "episode", self.eids[1], "semantic", "existing", "既有联想", weight=.8, confidence=1.))
        self.config = SimpleNamespace(database_path=self.db.path, log_dir=path / "logs",
            model=SimpleNamespace(embedding_dimension=2, embedding_model="offline"))
        self.clock = Clock()
        self.model = Reviewer(self.clock)
        self.service = ProgressiveRecallV3(self.config, self.db, self.model, clock=self.clock)

    def test_record_evidence_is_delivered_and_bound_to_source(self):
        result = self.service.query("求助信后来怎么样？", learn=False)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["review_protocol_version"], 3)
        self.assertEqual({f["episode_id"] for f in result["evidence"]}, set(self.eids))
        state = self.service.sessions.read(result["session_id"])
        self.assertEqual(state["version"], 3)
        self.assertIsNone(state["review_work"])
        for f in result["evidence"]:
            self.assertIn(f["quote"], f["claim"])
        self.assertEqual(result["learning"]["status"], "disabled")

    def test_need_failure_preserves_verified_facts_without_completion(self):
        self.model.invalid_need = True
        result = self.service.query("求助信后来怎么样？", learn=False, max_waves=1)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertTrue(all(a["status"] == "unknown" for a in result["need_assessments"]))
        self.assertEqual(result["stage_errors"][-1]["stage"], "need")

    def test_global_timeout_resumes_candidates_without_remapping(self):
        self.model.fact_delay = 10
        first = self.service.query("求助信后来怎么样？", timeout_seconds=5, learn=False)
        self.assertEqual(first["status"], "time_budget")
        self.assertEqual(first["evidence"], [])
        self.assertEqual(first["pending_evidence_count"], 2)
        pending = self.service.sessions.read(first["session_id"])
        self.assertEqual(len(pending["source_offsets"]), 2)
        resumed_service = ProgressiveRecallV3(self.config, self.db, self.model, clock=self.clock)
        final = resumed_service.query("求助信后来怎么样？", resume=first["session_id"], learn=False)
        self.assertTrue(final["complete"], final)
        self.assertEqual(self.model.calls.count(prompts.MAP), 1)
        self.assertEqual(self.model.calls.count(prompts.PLAN), 1)

    def test_local_stage_deadline_retains_pending_for_another_wave(self):
        self.model.fact_delay = 50
        first = self.service.query("求助信后来怎么样？", learn=False, max_waves=1)
        self.assertEqual(first["status"], "wave_budget", first)
        self.assertEqual(first["evidence"], [])
        self.assertEqual(first["pending_evidence_count"], 2)
        self.assertEqual(first["stage_errors"][-1]["type"], "TimeoutError")
        second = self.service.query("求助信后来怎么样？", resume=first["session_id"], learn=False)
        self.assertTrue(second["complete"], second)

    def test_nested_evidence_is_rechecked_before_delivery(self):
        result = self.service.query("求助信后来怎么样？", learn=False)
        state = self.service.sessions.read(result["session_id"])
        state["facts"][0]["evidence"][0]["quote"] = "原文没有这句话"
        with self.assertRaisesRegex(ValueError, "binding changed"):
            self.service._validate_delivered_sources(state)

    def test_resolved_needs_do_not_skip_unfinished_link_stage_on_resume(self):
        self.model.with_link = True
        self.model.link_delay = 10
        first = self.service.query("求助信后来怎么样？", timeout_seconds=5, learn=False)
        self.assertEqual(first["status"], "time_budget", first)
        pending = self.service.sessions.read(first["session_id"])
        self.assertEqual(pending["resolved_needs"], [0])
        self.assertEqual(pending["review_work"]["phase"], "links")
        second = self.service.query("求助信后来怎么样？", resume=first["session_id"], learn=False)
        self.assertTrue(second["complete"], second)
        self.assertEqual(self.model.calls.count(prompts.LINKS), 2)
        self.assertEqual(self.model.calls.count(prompts.MAP), 1)
        final = self.service.sessions.read(first["session_id"])
        self.assertEqual(len(final["verified_links"]), 1)

    def test_committed_fact_stage_is_not_repeated_after_cursor_write_interrupt(self):
        self.model.reject_first = True
        original_write = self.service.sessions.write
        interrupted = []
        def write_then_interrupt(state):
            original_write(state)
            work = state.get("review_work") or {}
            if not interrupted and "facts:0" in work.get("committed_stages", {}) and work["batch_cursor"] == 0:
                interrupted.append(True)
                raise KeyboardInterrupt()
        self.service.sessions.write = write_then_interrupt
        first = self.service.query("求助信后来怎么样？", learn=False)
        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(len(first["evidence"]), 1)
        restored = ProgressiveRecallV3(self.config, self.db, self.model, clock=self.clock)
        final = restored.query("求助信后来怎么样？", resume=first["session_id"], learn=False)
        self.assertTrue(final["complete"], final)
        self.assertEqual(self.model.calls.count(prompts.FACTS), 1)
        self.assertEqual(self.model.calls.count(prompts.MAP), 1)


if __name__ == "__main__":
    unittest.main()
