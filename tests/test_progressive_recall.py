"""Offline integration tests over real SQLite evidence and deterministic models."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from config.prompt_config.progressive_recall_prompts import (
    RECALL_MAP_SYSTEM, RECALL_PLAN_SYSTEM, RECALL_VERIFY_SYSTEM,
)
from memory_demo.app import MemoryApplication
from memory_demo.associations.feedback import SourceEvidence, VerifiedRecallLink, source_sha256
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.embeddings.codec import encode_embedding
from memory_demo.repositories import AssociationRepository, EpisodeRepository, SourceRepository
from memory_demo.retrieval.progressive import ProgressiveRecall
from memory_demo.types import AssociationDraft, EpisodeDraft


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

def model_payload(content):
    """Inspect local test metadata; the provider only receives natural text."""
    value = getattr(content, "structured_payload", None)
    return value if value is not None else json.loads(content)


def fixture_review_verdict(payload, *, accepted_fact_indices, covered_needs, accepted_link_indices):
    """Translate only these explicitly scripted legacy fixtures into v2 verdicts.

    This helper is never used for malformed-schema tests or production responses.
    Related accepted facts remain partial when the fixture does not certify the
    complete requirement. An unsupported legacy coverage vote creates no proof.
    """
    facts, links = payload["proposal"]["facts"], payload["proposal"]["links"]
    accepted = set(accepted_fact_indices)
    accepted_links = set(accepted_link_indices)
    needs = []
    for index in range(len(payload["needs"])):
        identifiers = [fact["fact_id"] for i, fact in enumerate(facts)
                       if i in accepted and index in fact["need_indices"]]
        status = "supported" if index in covered_needs and identifiers else "partial" if identifiers else "unknown"
        needs.append({"need_index": index, "status": status, "fact_ids": identifiers,
                      "reason": "The scripted source evidence supports this extent of the requirement."})
    return {
        "fact_decisions": [{"fact_id": fact["fact_id"], "decision": "accept" if i in accepted else "reject",
                            "reason": "The scripted source supports the claim." if i in accepted
                            else "The scripted source contradicts this claim or does not support its attribution."}
                           for i, fact in enumerate(facts)],
        "link_decisions": [{"link_id": link["link_id"], "decision": "accept" if i in accepted_links else "reject",
                            "reason": "Both quoted endpoints support this retrieval association." if i in accepted_links
                            else "The scripted review does not support this association."}
                           for i, link in enumerate(links)],
        "need_decisions": needs,
    }


def complete_v2_fixture_verdict(payload, *, need_statuses=None):
    """Build explicit v2 responses for new tests; no legacy conversion occurs."""
    facts, links = payload["proposal"]["facts"], payload["proposal"]["links"]
    needs = []
    for index in range(len(payload["needs"])):
        refs = [f["fact_id"] for f in facts if index in f["need_indices"]]
        status = (need_statuses or {}).get(index, "supported" if refs else "unknown")
        needs.append({"need_index": index, "status": status,
                      "fact_ids": [] if status == "unknown" else refs,
                      "reason": "The explicit v2 fixture records this bounded source conclusion."})
    return {"fact_decisions": [{"fact_id": f["fact_id"], "decision": "accept",
                                 "reason": "The fixture Source supports this exact attributed claim."} for f in facts],
            "link_decisions": [{"link_id": link["link_id"], "decision": "accept",
                                 "reason": "The quoted endpoints support a retrieval association."} for link in links],
            "need_decisions": needs}


class EvidenceReviewModel:
    """A scripted reviewer, not a semantic-quality claim about a production LLM.

    Mapping deliberately overclaims coverage in failure scenarios. A separate
    verifier response must control completion, and exact quotes must first pass
    local validation. The graph itself remains fully real and unmocked.
    """

    def __init__(self, root_id, target_id, clock):
        self.root_id, self.target_id, self.clock = root_id, target_id, clock
        self.scenario = "normal"
        self.calls = []
        self.embedding_calls = 0
        self.map_calls = 0
        self.timeout_on_map = None
        self.cancel_on_map = None
        self.cancelled = False

    def embed(self, cues):
        self.embedding_calls += 1
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in cues]

    def chat_json(self, system, content):
        payload = model_payload(content)
        self.calls.append((system, payload))
        if system == RECALL_PLAN_SYSTEM:
            need = "乙如何回应甲的求助？"
            if self.scenario == "joint_incomplete":
                need = "乙是否既答应了求助，也完成了实际行动？"
            elif self.scenario == "negation":
                need = "乙是否明确拒绝了甲的求助？"
            return {"needs": [need], "cues": ["甲送出的求助信"]}
        if system == RECALL_MAP_SYSTEM:
            self.map_calls += 1
            if self.map_calls == self.timeout_on_map:
                self.clock.value += 10
            if self.map_calls == self.cancel_on_map:
                self.cancelled = True
            facts = []
            seen = set()
            for source in payload["sources"]:
                eid = source["episode_id"]
                if eid not in {self.root_id, self.target_id} or eid in seen:
                    continue
                seen.add(eid)
                target = eid == self.target_id
                quote = source["text"]
                claim = "乙答应了甲的求助，并安排见面。" if target else "甲寄出了一封求助信。"
                if target and self.scenario == "wrong_quote":
                    quote = "这段文字根本没有在原文里出现。"
                if target and self.scenario == "negation":
                    claim = "乙拒绝了甲的求助。"
                facts.append({"source_id": source["source_id"], "episode_id": eid,
                              "quote": quote, "claim": claim, "need_indices": [0] if target else []})
            links = []
            if {self.root_id, self.target_id}.issubset({f["episode_id"] for f in facts}):
                links = [{"from_episode_id": self.root_id, "to_episode_id": self.target_id,
                          "rationale": "The letter and its response are supported by the two quoted sources."}]
            return {"facts": facts, "links": links}
        if system == RECALL_VERIFY_SYSTEM:
            facts = payload["proposal"]["facts"]
            accepted = [i for i, fact in enumerate(facts)
                        if not ("拒绝" in fact["claim"] and "答应" in fact["quote"])]
            accepted_ids = {facts[i]["episode_id"] for i in accepted}
            coverage = [0] if self.target_id in accepted_ids else []
            if self.scenario in {"no_semantic_coverage", "joint_incomplete"}:
                coverage = []
            links = [i for i, link in enumerate(payload["proposal"]["links"])
                     if {link["from_episode_id"], link["to_episode_id"]}.issubset(accepted_ids)]
            return fixture_review_verdict(payload, accepted_fact_indices=accepted,
                                          covered_needs=coverage, accepted_link_indices=links)
        raise AssertionError("unexpected model prompt")


class ProgressiveRecallIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.db = Database(directory / "memory.sqlite")
        self.db.initialize()
        self.sources, self.episodes = SourceRepository(self.db), EpisodeRepository(self.db)
        self.associations = AssociationRepository(self.db)
        texts = ["甲寄出了一封求助信。", "值日生整理了教室。", "列车正常运行。", "乙读完甲的信后答应了求助，并安排见面。"]
        self.source_ids = [self.sources.insert(text) for text in texts]
        self.episode_ids = [self.episodes.insert(
            source_id, f"main/{i}.json", 0, EpisodeDraft(text),
            encode_embedding([1.0, 0.0] if i == 0 else [0.0, 1.0], 2),
        ) for i, (source_id, text) in enumerate(zip(self.source_ids, texts))]
        self.edge_ids = [self.associations.upsert(AssociationDraft(
            "episode", self.episode_ids[i], "episode", self.episode_ids[i + 1],
            "semantic", f"existing-{i}", "既有联想", weight=0.8, confidence=1.0,
        )) for i in range(3)]
        self.config = SimpleNamespace(
            log_dir=directory / "logs", database_path=self.db.path,
            model=SimpleNamespace(embedding_dimension=2, embedding_model="offline-fixture"),
        )
        self.clock = ManualClock()
        self.model = EvidenceReviewModel(self.episode_ids[0], self.episode_ids[-1], self.clock)
        self.recall = ProgressiveRecall(self.config, self.db, self.model, clock=self.clock)
        self.question = "那封求助信后来得到了怎样的回应？"

    def weight(self, identifier):
        return float(self.associations.get(identifier)["weight"])

    def plan_calls(self):
        return sum(system == RECALL_PLAN_SYSTEM for system, _ in self.model.calls)

    def test_first_review_learns_then_paraphrase_reuses_with_less_work_and_feedback(self):
        first = self.recall.query(self.question, mode="deep")
        self.assertTrue(first["complete"], first)
        self.assertEqual(first["missing_requirements"], [])
        self.assertEqual(first["learning"]["status"], "applied")
        shortcuts = self.recall.feedback.valid_learned_edge_ids()
        self.assertEqual(len(shortcuts), 1)
        shortcut_id = next(iter(shortcuts))
        self.assertEqual(self.weight(shortcut_id), 0.2)
        existing_weights = [self.weight(edge) for edge in self.edge_ids]

        second = self.recall.query("再回想一下，甲的求助最后获得了什么答复？", mode="deep")
        self.assertTrue(second["complete"], second)
        self.assertEqual(second["needs"], first["needs"])
        self.assertEqual(second["missing_requirements"], [])
        self.assertEqual(second["used_edge_ids"], [shortcut_id])
        self.assertLess(second["metrics"]["edge_expansions"], first["metrics"]["edge_expansions"])
        self.assertLess(second["metrics"]["source_windows_reviewed"], first["metrics"]["source_windows_reviewed"])
        target_quotes = lambda result: {f["quote"] for f in result["evidence"] if f["episode_id"] == self.episode_ids[-1]}
        self.assertEqual(target_quotes(second), target_quotes(first))
        positive = self.recall.apply_feedback(second["session_id"], positive=True, feedback_id="positive")
        self.assertTrue(positive["applied"])
        self.assertAlmostEqual(self.weight(shortcut_id), 0.36)
        retry = self.recall.apply_feedback(second["session_id"], positive=True, feedback_id="positive")
        self.assertFalse(retry["applied"])
        self.recall.apply_feedback(second["session_id"], positive=False, feedback_id="negative")
        self.assertAlmostEqual(self.weight(shortcut_id), 0.288)
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], existing_weights)

    def test_wave_resume_keeps_plan_embedding_sources_and_graph_progress(self):
        partial = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        self.assertEqual(partial["status"], "wave_budget", partial)
        self.assertFalse(partial["complete"])
        self.assertTrue(partial["resumable"])
        self.assertEqual(len(partial["evidence"]), 1)
        self.assertEqual(self.plan_calls(), 1)
        self.assertEqual(self.model.embedding_calls, 1)
        before = self.recall.sessions.read(partial["session_id"])
        self.assertIsNotNone(before["spreading"])
        self.assertEqual(len(before["source_offsets"]), 2)
        # A fresh service instance proves persistence rather than object memory.
        restored = ProgressiveRecall(self.config, self.db, self.model, clock=self.clock)
        final = restored.query(self.question, mode="light", resume=partial["session_id"], max_waves=1, learn=False)
        self.assertTrue(final["complete"], final)
        self.assertEqual(final["session_id"], partial["session_id"])
        self.assertEqual(self.plan_calls(), 1)
        self.assertEqual(self.model.embedding_calls, 1)
        self.assertEqual(final["metrics"]["embedding_batches"], 1)
        self.assertEqual(final["metrics"]["source_windows_reviewed"], 4)

    def test_timeout_retains_partial_evidence_and_retries_uncommitted_source_wave(self):
        self.model.timeout_on_map = 2
        partial = self.recall.query(self.question, mode="light", timeout_seconds=5, learn=False)
        self.assertEqual(partial["status"], "time_budget", partial)
        self.assertEqual(len(partial["evidence"]), 1)
        self.assertEqual(partial["metrics"]["source_windows_reviewed"], 2)
        resumed = self.recall.query(self.question, mode="light", timeout_seconds=5,
                                    resume=partial["session_id"], learn=False)
        self.assertTrue(resumed["complete"], resumed)
        self.assertEqual(resumed["metrics"]["source_windows_reviewed"], 4)
        self.assertEqual(self.plan_calls(), 1)
        self.assertEqual(self.model.embedding_calls, 1)

    def test_cancellation_retains_partial_evidence_and_can_resume(self):
        self.model.cancel_on_map = 2
        partial = self.recall.query(self.question, mode="light", learn=False,
                                    cancelled=lambda: self.model.cancelled)
        self.assertEqual(partial["status"], "cancelled", partial)
        self.assertEqual(len(partial["evidence"]), 1)
        self.assertFalse(partial["complete"])
        self.model.cancelled = False
        resumed = self.recall.query(self.question, mode="light", learn=False,
                                    resume=partial["session_id"], cancelled=lambda: self.model.cancelled)
        self.assertTrue(resumed["complete"], resumed)
        self.assertEqual(self.model.embedding_calls, 1)
        self.assertEqual(self.plan_calls(), 1)

    def test_changed_source_invalidates_checkpoint_shortcut_and_feedback(self):
        first = self.recall.query(self.question)
        self.assertTrue(first["complete"], first)
        shortcut = next(iter(self.recall.feedback.valid_learned_edge_ids()))
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=raw_text || '后来补充了日期。' WHERE id=?", (self.source_ids[-1],))
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())
        with self.assertRaisesRegex(ValueError, "changed"):
            self.recall.query(self.question, resume=first["session_id"])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.recall.apply_feedback(first["session_id"], positive=True, feedback_id="stale")
        fresh = self.recall.query("重新检查求助信的答复。", learn=False)
        self.assertTrue(fresh["complete"], fresh)
        self.assertNotIn(shortcut, fresh["used_edge_ids"])
        self.assertEqual(fresh["metrics"]["shortcut_candidates"], 0)
        self.assertGreater(fresh["metrics"]["edge_expansions"], 0)

    def test_wrong_quote_cannot_enter_evidence_or_close_requirement(self):
        self.model.scenario = "wrong_quote"
        result = self.recall.query(self.question)
        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "search_exhausted", result)
        self.assertEqual(len(result["missing_requirements"]), 1)
        self.assertNotIn(self.episode_ids[-1], {f["episode_id"] for f in result["evidence"]})
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())
        verified_proposals = [payload["proposal"]["facts"] for system, payload in self.model.calls if system == RECALL_VERIFY_SYSTEM]
        self.assertTrue(all("根本没有" not in fact["quote"] for facts in verified_proposals for fact in facts))

    def test_independent_verifier_rejects_negation_even_when_mapper_claims_coverage(self):
        self.model.scenario = "negation"
        result = self.recall.query(self.question)
        self.assertFalse(result["complete"])
        self.assertNotIn("拒绝", result["answer"])
        self.assertEqual(result["status"], "search_exhausted", result)
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_joint_requirement_rejection_prevents_completion_and_learning(self):
        self.model.scenario = "joint_incomplete"
        result = self.recall.query("是否既答应求助，也完成了实际行动？")
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["status"], "search_exhausted", result)
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_source_closed_facts_without_semantic_coverage_are_not_complete(self):
        self.model.scenario = "no_semantic_coverage"
        result = self.recall.query(self.question)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(len(result["missing_requirements"]), 1)
        self.assertEqual(result["status"], "search_exhausted", result)
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_repeated_relation_reviews_across_waves_learn_one_shortcut(self):
        first = self.recall.query(self.question)
        self.assertTrue(first["complete"], first)
        action_id = self.episode_ids[2]

        class MultipleRequirementsModel(EvidenceReviewModel):
            def chat_json(inner, system, content):
                response = super().chat_json(system, content)
                payload = model_payload(content)
                if system == RECALL_PLAN_SYSTEM:
                    response["needs"].append("列车是否正常运行？")
                elif system == RECALL_MAP_SYSTEM:
                    for source in payload["sources"]:
                        if source["episode_id"] == action_id:
                            response["facts"].append({
                                "source_id": source["source_id"], "episode_id": action_id,
                                "quote": source["text"], "claim": "列车正常运行。", "need_indices": [1],
                            })
                    for link in response["links"]:
                        link["rationale"] += f" Review wave {inner.map_calls}."
                elif system == RECALL_VERIFY_SYSTEM:
                    if any(f["episode_id"] == action_id for f in payload["proposal"]["facts"]):
                        response["need_decisions"][1]["status"] = "supported"
                return response

        model = MultipleRequirementsModel(self.episode_ids[0], self.episode_ids[-1], self.clock)
        recall = ProgressiveRecall(self.config, self.db, model, clock=self.clock)
        result = recall.query("回想求助信的答复和列车的运行情况。", mode="light")
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["learning"]["status"], "applied")
        self.assertEqual(len(recall.feedback.valid_learned_edge_ids()), 1)
        self.assertGreaterEqual(result["metrics"]["review_waves"], 3)

    def test_verified_endpoints_do_not_certify_unrelated_intermediate_path_edges(self):
        original_weights = [self.weight(edge) for edge in self.edge_ids]
        result = self.recall.query(self.question)
        self.assertTrue(result["complete"], result)
        self.assertEqual(len(result["evidence"]), 2)
        # Search reached the reply via classroom duty and a train. Reviewing
        # the letter and reply validates A <-> D, not those unrelated bridges.
        self.assertGreater(result["metrics"]["edge_expansions"], 0)
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], original_weights)
        valid_shortcuts = self.recall.feedback.valid_learned_edge_ids()
        self.assertEqual(len(valid_shortcuts), 1)
        self.assertEqual(set(result["learning"]["association_ids"]), valid_shortcuts)
        with self.db.connection() as connection:
            changes = connection.execute("SELECT association_id FROM recall_feedback_change").fetchall()
        self.assertEqual({int(row[0]) for row in changes}, valid_shortcuts)

    def test_later_counterevidence_withdraws_prior_coverage_facts_and_links(self):
        disputed_id = self.episode_ids[1]
        action_id = self.episode_ids[2]
        correction_id = self.episode_ids[3]
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=? WHERE id=?",
                               ("乙当时说：我答应了求助。", self.source_ids[1]))
            connection.execute("UPDATE source SET raw_text=? WHERE id=?",
                               ("乙随后澄清：之前说答应只是误会，我没有接受求助。", self.source_ids[3]))

        class RevisingReviewer(EvidenceReviewModel):
            def __init__(inner, *args):
                super().__init__(*args)
                inner.saw_prior_fact_in_final_review = False
                inner.saw_prior_link_in_final_review = False

            def chat_json(inner, system, content):
                payload = model_payload(content)
                inner.calls.append((system, payload))
                if system == RECALL_PLAN_SYSTEM:
                    return {"needs": ["乙是否实际同意求助？", "列车是否正常运行？"], "cues": ["甲的求助信"]}
                correction_visible = any(s["episode_id"] == correction_id for s in payload["sources"])
                if system == RECALL_MAP_SYSTEM:
                    inner.map_calls += 1
                    facts = []
                    for source in payload["sources"]:
                        eid = source["episode_id"]
                        # Once the correction arrives, the mapper no longer
                        # repeats the earlier claim. The service must still
                        # show that prior claim and link to the final verifier.
                        if correction_visible and eid in {inner.root_id, disputed_id}:
                            continue
                        claims = {inner.root_id: "甲寄出了一封求助信。", disputed_id: "乙实际同意了求助。",
                                  action_id: "列车正常运行。", correction_id: "乙后来明确否认接受了求助。"}
                        facts.append({"source_id": source["source_id"], "episode_id": eid,
                                      "quote": source["text"], "claim": claims[eid],
                                      "need_indices": [0] if eid == disputed_id else [1] if eid == action_id else []})
                    links = [] if correction_visible else [{"from_episode_id": inner.root_id,
                        "to_episode_id": disputed_id, "rationale": "先前以答应求助作为回应关系。"}]
                    return {"facts": facts, "links": links}
                if system == RECALL_VERIFY_SYSTEM:
                    facts = payload["proposal"]["facts"]
                    links = payload["proposal"]["links"]
                    if correction_visible:
                        inner.saw_prior_fact_in_final_review = any(f["episode_id"] == disputed_id for f in facts)
                        inner.saw_prior_link_in_final_review = any(l["to_episode_id"] == disputed_id for l in links)
                        response = fixture_review_verdict(payload,
                            accepted_fact_indices=[i for i, f in enumerate(facts) if f["episode_id"] != disputed_id],
                            accepted_link_indices=[], covered_needs=[1])
                        correction_fact_id = next(f["fact_id"] for f in facts if f["episode_id"] == correction_id)
                        disputed_fact_ids = {f["fact_id"] for f in facts if f["episode_id"] == disputed_id}
                        for decision in response["fact_decisions"]:
                            if decision["fact_id"] in disputed_fact_ids:
                                decision.update(reason="乙随后明确澄清之前的答应只是误会，否认实际接受求助。",
                                                basis_fact_ids=[correction_fact_id])
                        for decision in response["link_decisions"]:
                            decision.update(reason="明确的后续澄清推翻以实际答应为前提的联结。",
                                            basis_fact_ids=[correction_fact_id])
                        return response
                    return fixture_review_verdict(payload, accepted_fact_indices=list(range(len(facts))),
                        accepted_link_indices=list(range(len(links))), covered_needs=[0])
                raise AssertionError("unexpected model prompt")

        model = RevisingReviewer(self.episode_ids[0], self.episode_ids[-1], self.clock)
        recall = ProgressiveRecall(self.config, self.db, model, clock=self.clock)
        question = "回想乙是否真的接受了求助，以及当时列车的运行情况。"
        partial = recall.query(question, mode="light", max_waves=1)
        self.assertEqual(partial["status"], "wave_budget", partial)
        checkpoint = recall.sessions.read(partial["session_id"])
        self.assertEqual(checkpoint["covered_needs"], [0])
        self.assertEqual(len(checkpoint["verified_links"]), 1)
        result = recall.query(question, mode="light", resume=partial["session_id"])
        self.assertFalse(result["complete"], result)
        self.assertEqual(result["status"], "search_exhausted")
        self.assertEqual(result["missing_requirements"], ["乙是否实际同意求助？"])
        self.assertNotIn("乙实际同意了求助。", result["answer"])
        self.assertTrue(model.saw_prior_fact_in_final_review)
        self.assertTrue(model.saw_prior_link_in_final_review)
        final_state = recall.sessions.read(partial["session_id"])
        self.assertEqual(final_state["covered_needs"], [1])
        self.assertEqual(final_state["verified_links"], [])
        self.assertEqual(recall.feedback.valid_learned_edge_ids(), set())
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.8] * 3)

    def test_source_changed_during_final_verification_clears_stale_delivery_and_learning(self):
        original_chat = self.model.chat_json

        def changing_reviewer(system, content):
            response = original_chat(system, content)
            if system == RECALL_VERIFY_SYSTEM:
                with self.db.transaction() as connection:
                    connection.execute("UPDATE source SET raw_text=raw_text || '原文被修正。' WHERE id=?",
                                       (self.source_ids[-1],))
            return response

        self.model.chat_json = changing_reviewer
        result = self.recall.query(self.question)
        self.assertEqual(result["status"], "source_changed", result)
        self.assertFalse(result["complete"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["used_edge_ids"], [])
        self.assertEqual(result["missing_requirements"], result["needs"])
        self.assertNotEqual(result["learning"]["status"], "applied")
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.8] * 3)
        checkpoint = self.recall.sessions.read(result["session_id"])
        self.assertEqual(checkpoint["facts"], [])
        self.assertEqual(checkpoint["verified_links"], [])

    def test_identical_data_in_another_database_cannot_receive_session_feedback(self):
        first = self.recall.query(self.question)
        self.assertTrue(first["complete"], first)
        other = Database(Path(self.temp.name) / "another.sqlite")
        with self.db.connection() as source, other.connection() as destination:
            source.backup(destination)
        config = SimpleNamespace(log_dir=self.config.log_dir, database_path=other.path,
                                 model=self.config.model)
        foreign = ProgressiveRecall(config, other, self.model, clock=self.clock)
        with other.connection() as connection:
            before = [tuple(r) for r in connection.execute("SELECT id,weight FROM association ORDER BY id")]
        with self.assertRaisesRegex(ValueError, "another database"):
            foreign.apply_feedback(first["session_id"], positive=True, feedback_id="wrong-database")
        with other.connection() as connection:
            after = [tuple(r) for r in connection.execute("SELECT id,weight FROM association ORDER BY id")]
            events = connection.execute("SELECT COUNT(*) FROM recall_feedback_event WHERE feedback_id='wrong-database'").fetchone()[0]
        self.assertEqual(before, after)
        self.assertEqual(events, 0)

    def test_invalid_requirement_entries_reject_the_entire_plan_without_dropping_needs(self):
        for invalid in ("", "  ", None, 42, {}, False):
            with self.subTest(invalid=invalid):
                model = EvidenceReviewModel(self.episode_ids[0], self.episode_ids[-1], self.clock)
                original_chat = model.chat_json

                def malformed_plan(system, content):
                    response = original_chat(system, content)
                    if system == RECALL_PLAN_SYSTEM:
                        response["needs"] = ["乙如何回应求助？", invalid]
                    return response

                model.chat_json = malformed_plan
                recall = ProgressiveRecall(self.config, self.db, model, clock=self.clock)
                result = recall.query(self.question)
                self.assertEqual(result["status"], "technical_error", result)
                self.assertFalse(result["complete"])
                self.assertEqual(result["error"]["type"], "ValueError")
                self.assertEqual(result["evidence"], [])
                self.assertEqual(result["needs"], [])
                self.assertEqual(model.embedding_calls, 0)
                self.assertEqual([system for system, _ in model.calls], [RECALL_PLAN_SYSTEM])
                self.assertEqual(recall.feedback.valid_learned_edge_ids(), set())

    def test_visible_source_quote_cannot_be_attached_to_an_unseen_episode(self):
        hidden_episode = self.episodes.insert(
            self.source_ids[0], "main/hidden.json", 1, EpisodeDraft("未呈现的另一事件摘要。"),
            encode_embedding([0.0, 1.0], 2),
        )
        original_chat = self.model.chat_json
        verifier_payloads = []

        def mismatched_episode(system, content):
            payload = model_payload(content)
            if system == RECALL_MAP_SYSTEM:
                source = next(s for s in payload["sources"] if s["episode_id"] == self.episode_ids[0])
                self.assertNotIn(hidden_episode, {s["episode_id"] for s in payload["sources"]})
                return {"facts": [{"source_id": source["source_id"], "episode_id": hidden_episode,
                                   "quote": source["text"], "claim": "引文被错误地挂到另一个事件。",
                                   "need_indices": [0]}], "links": []}
            if system == RECALL_VERIFY_SYSTEM:
                verifier_payloads.append(payload)
                return fixture_review_verdict(payload,
                    accepted_fact_indices=list(range(len(payload["proposal"]["facts"]))),
                    covered_needs=[0], accepted_link_indices=[])
            return original_chat(system, content)

        self.model.chat_json = mismatched_episode
        result = self.recall.query(self.question, mode="deep", learn=False, max_waves=1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["evidence"], [])
        # With no admissible candidates the v2 boundary skips the verifier;
        # the forged Episode binding must never reach semantic certification.
        self.assertEqual(verifier_payloads, [])
        self.assertEqual(len(result["missing_requirements"]), 1)

    def test_modified_checkpoint_checksum_is_rejected_before_any_new_model_work(self):
        partial = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        self.assertEqual(partial["status"], "wave_budget", partial)
        checkpoint = Path(partial["checkpoint_path"])
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertIn("checksum", state)
        state["covered_needs"] = list(range(len(state["needs"])))
        checkpoint.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        model_calls = len(self.model.calls)
        embedding_calls = self.model.embedding_calls
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.recall.query(self.question, mode="light", resume=partial["session_id"], learn=False)
        self.assertEqual(len(self.model.calls), model_calls)
        self.assertEqual(self.model.embedding_calls, embedding_calls)

    def test_shortcut_priority_follows_strength_and_suppressed_edge_leaves_early_lane(self):
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=? WHERE id=?",
                               ("甲把求助信交给值日生转交给乙。", self.source_ids[1]))

        def source_evidence(index):
            source = self.sources.get(self.source_ids[index])
            text = str(source["raw_text"])
            return SourceEvidence(self.episode_ids[index], self.source_ids[index], source_sha256(text),
                                  0, len(text), text)

        def verified_link(index):
            return VerifiedRecallLink(self.episode_ids[0], self.episode_ids[index],
                                      (source_evidence(0), source_evidence(index)),
                                      "fixture-source-review", "求助信的转交或回应由两端原文支持。", verified=True)

        handoff = self.recall.feedback.learn_verified("handoff", [verified_link(1)], learning_rate=0.8).association_ids[0]
        reply = self.recall.feedback.learn_verified("reply", [verified_link(3)], learning_rate=0.2).association_ids[0]

        def first_review_order(question):
            before = len(self.model.calls)
            result = self.recall.query(question, mode="light", learn=False)
            self.assertTrue(result["complete"], result)
            mapped = next(payload for system, payload in self.model.calls[before:] if system == RECALL_MAP_SYSTEM)
            return [s["episode_id"] for s in mapped["sources"]], result

        order, _ = first_review_order("信件的答复是什么？")
        self.assertEqual(order, [self.episode_ids[1], self.episode_ids[3]])
        self.recall.feedback.apply_user_feedback("promote-reply", [reply], positive=True, learning_rate=0.9)
        order, _ = first_review_order("重新回想信件的答复。")
        self.assertEqual(order, [self.episode_ids[3], self.episode_ids[1]])
        self.recall.feedback.apply_user_feedback("suppress-handoff", [handoff], positive=False, learning_rate=0.95)
        self.assertLess(self.weight(handoff), 0.05)
        order, result = first_review_order("再次检查信件最终的答复。")
        self.assertEqual(order, [self.episode_ids[3]])
        self.assertEqual(result["metrics"]["shortcut_candidates"], 1)
        self.assertIn(handoff, {int(row["id"]) for row in self.associations.neighbors("episode", self.episode_ids[0])})

    def test_public_application_recall_and_feedback_accept_a_scripted_provider(self):
        config = AppConfig(database_path=self.db.path, log_dir=self.config.log_dir,
                           model=ModelConfig(embedding_dimension=2, embedding_model="offline-fixture"))
        application = MemoryApplication(config)
        result = application.recall(self.question, mode="deep", model=self.model)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["learning"]["status"], "applied")
        shortcut_id = next(iter(self.recall.feedback.valid_learned_edge_ids()))
        feedback = application.recall_feedback(result["session_id"], positive=True, feedback_id="public-positive")
        self.assertTrue(feedback["applied"])
        self.assertEqual([change["association_id"] for change in feedback["changes"]], [shortcut_id])
        self.assertAlmostEqual(self.weight(shortcut_id), 0.36)

    def test_unrelated_later_wave_retains_verified_fact_when_mapper_does_not_repeat_it(self):
        first = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        self.assertEqual(first["status"], "wave_budget", first)
        before = self.recall.sessions.read(first["session_id"])
        original_chat = self.model.chat_json

        def no_new_facts(system, content):
            if system == RECALL_MAP_SYSTEM:
                return {"facts": [], "links": []}
            if system == RECALL_VERIFY_SYSTEM:
                return complete_v2_fixture_verdict(model_payload(content), need_statuses={0: "unknown"})
            return original_chat(system, content)

        self.model.chat_json = no_new_facts
        result = self.recall.query(self.question, mode="light", resume=first["session_id"],
                                   max_waves=1, learn=False)
        after = self.recall.sessions.read(first["session_id"])
        self.assertEqual(result["evidence"], before["facts"])
        self.assertEqual([f["fact_id"] for f in after["facts"]], [f["fact_id"] for f in before["facts"]])
        self.assertGreater(after["metrics"]["source_windows_reviewed"], before["metrics"]["source_windows_reviewed"])
        self.assertFalse(result["complete"])
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_prior_short_quote_replay_restores_its_speaker_record(self):
        raw = "[source_key: main/a.json]\n[record: 1]\n[speaker_raw: 甲]\nzh-CN: 我寄出了求助信。\n"
        with self.db.transaction() as connection:
            connection.execute("UPDATE source SET raw_text=? WHERE id=?", (raw, self.source_ids[0]))
        original_chat = self.model.chat_json
        observed_contexts = []
        resumed = False

        def contextual_reviewer(system, content):
            payload = model_payload(content)
            if system == RECALL_MAP_SYSTEM:
                if resumed:
                    return {"facts": [], "links": []}
                response = original_chat(system, content)
                for fact in response["facts"]:
                    if fact["episode_id"] == self.episode_ids[0]:
                        fact["quote"] = "我寄出了求助信。"
                return response
            if system == RECALL_VERIFY_SYSTEM:
                observed_contexts.append(any(
                    source["source_id"] == self.source_ids[0]
                    and "[speaker_raw: 甲]" in source["text"]
                    and "我寄出了求助信。" in source["text"]
                    for source in payload["sources"]))
                return complete_v2_fixture_verdict(payload, need_statuses={0: "unknown"})
            return original_chat(system, content)

        self.model.chat_json = contextual_reviewer
        first = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        self.assertEqual(first["evidence"][0]["quote"], "我寄出了求助信。")
        resumed = True
        result = self.recall.query(self.question, mode="light", resume=first["session_id"],
                                   max_waves=1, learn=False)
        self.assertEqual(observed_contexts, [True, True])
        self.assertEqual(result["evidence"][0]["fact_id"], first["evidence"][0]["fact_id"])

    def test_needs_context_retains_pending_facts_without_delivery_or_learning(self):
        original_chat = self.model.chat_json

        def request_context(system, content):
            if system == RECALL_VERIFY_SYSTEM:
                response = complete_v2_fixture_verdict(model_payload(content), need_statuses={0: "unknown"})
                for decision in response["fact_decisions"]:
                    decision.update(decision="needs_context", reason="The fixture needs the preceding speaker record.")
                for decision in response["link_decisions"]:
                    decision.update(decision="needs_context", reason="Endpoint attribution remains pending.")
                return response
            return original_chat(system, content)

        self.model.chat_json = request_context
        result = self.recall.query(self.question, mode="deep", max_waves=1)
        state = self.recall.sessions.read(result["session_id"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(len(state["pending_facts"]), 2)
        self.assertEqual(state["covered_needs"], [])
        self.assertEqual(state["resolved_needs"], [])
        self.assertNotEqual(result["learning"]["status"], "applied")
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.8] * 3)

    def test_shared_source_context_reread_preserves_every_pending_episode_binding(self):
        from memory_demo.retrieval.recall_policy import RecallPolicy
        from memory_demo.retrieval.recall_review import stable_fact_id

        raw = "[speaker: A] I agree. [following context]"
        quote = "I agree."
        start = raw.index(quote)
        episodes = {101: {"source_id": 1, "text": "first episode"},
                    102: {"source_id": 1, "text": "second episode"}}
        facts = [{"source_id": 1, "episode_id": eid,
                  "source_sha256": source_sha256(raw),
                  "start": start, "end": start + len(quote), "quote": quote,
                  "claim": "A agrees.", "need_indices": [0]}
                 for eid in episodes]
        state = {"pending_facts": facts, "context_rechecked_fact_ids": []}

        windows = self.recall._context_windows(
            state, {1: raw}, episodes, RecallPolicy.for_request("Who agrees?"))

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["text"], raw)
        self.assertEqual(set(windows[0]["episode_ids"]), set(episodes))
        self.assertEqual(set(windows[0]["recheck_fact_ids"]),
                         {stable_fact_id(fact) for fact in facts})
        self.assertEqual(state["context_rechecked_fact_ids"], [])

    def test_malformed_review_is_atomic_and_keeps_last_verified_state_and_offsets(self):
        for corruption in ("legacy_response", "missing_need", "duplicate_fact", "unknown_fact", "blank_reason", "rejected_support"):
            with self.subTest(corruption=corruption):
                model = EvidenceReviewModel(self.episode_ids[0], self.episode_ids[-1], self.clock)
                recall = ProgressiveRecall(self.config, self.db, model, clock=self.clock)
                first = recall.query(self.question, mode="light", max_waves=1, learn=False)
                before = recall.sessions.read(first["session_id"])
                original_chat = model.chat_json

                def bad_verdict(system, content):
                    if system != RECALL_VERIFY_SYSTEM:
                        return original_chat(system, content)
                    payload = model_payload(content)
                    # Invalid protocol responses never pass through the legacy adapter.
                    if corruption == "legacy_response":
                        return {"accepted_fact_indices": [], "covered_needs": [], "accepted_link_indices": []}
                    response = complete_v2_fixture_verdict(payload)
                    if corruption == "missing_need":
                        response["need_decisions"] = []
                    elif corruption == "duplicate_fact":
                        response["fact_decisions"].append(dict(response["fact_decisions"][0]))
                    elif corruption == "unknown_fact":
                        response["fact_decisions"][0]["fact_id"] = "not-a-candidate-fact"
                    elif corruption == "blank_reason":
                        response["fact_decisions"][0]["reason"] = " "
                    elif corruption == "rejected_support":
                        target_id = next(f["fact_id"] for f in payload["proposal"]["facts"]
                                         if f["episode_id"] == self.episode_ids[-1])
                        next(d for d in response["fact_decisions"] if d["fact_id"] == target_id)["decision"] = "reject"
                    return response

                model.chat_json = bad_verdict
                result = recall.query(self.question, mode="light", resume=first["session_id"], max_waves=1)
                after = recall.sessions.read(first["session_id"])
                self.assertEqual(result["status"], "technical_error", result)
                self.assertEqual(result["error"]["type"], "ReviewSchemaError")
                for field in ("facts", "pending_facts", "source_offsets", "covered_needs", "resolved_needs", "need_assessments"):
                    self.assertEqual(after[field], before[field], field)
                for field in ("source_windows_reviewed", "review_waves"):
                    self.assertEqual(after["metrics"][field], before["metrics"][field], field)
                self.assertEqual(result["evidence"], first["evidence"])
                self.assertEqual(recall.feedback.valid_learned_edge_ids(), set())

    def test_malformed_mapper_keeps_prior_facts_and_source_offsets(self):
        first = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        before = self.recall.sessions.read(first["session_id"])
        original_chat = self.model.chat_json

        def bad_mapper(system, content):
            if system == RECALL_MAP_SYSTEM:
                return {"facts": None, "links": []}
            return original_chat(system, content)

        self.model.chat_json = bad_mapper
        result = self.recall.query(self.question, mode="light", resume=first["session_id"], max_waves=1)
        after = self.recall.sessions.read(first["session_id"])
        self.assertEqual(result["status"], "technical_error", result)
        self.assertEqual(result["error"]["type"], "ReviewSchemaError")
        self.assertEqual(after["source_offsets"], before["source_offsets"])
        self.assertEqual(result["evidence"], first["evidence"])
        self.assertEqual(after["metrics"]["review_waves"], before["metrics"]["review_waves"])
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_supported_partial_unknown_and_refuted_needs_remain_distinct(self):
        original_chat = self.model.chat_json

        def mixed_assessments(system, content):
            payload = model_payload(content)
            if system == RECALL_PLAN_SYSTEM:
                return {"needs": ["甲是否寄信？", "答应以后是否真的执行？", "事件是哪一天？", "乙明确拒绝求助的前提是否成立？"],
                        "cues": ["甲的求助信"]}
            if system == RECALL_MAP_SYSTEM:
                response = original_chat(system, content)
                for fact in response["facts"]:
                    fact["need_indices"] = [0] if fact["episode_id"] == self.episode_ids[0] else [1, 3]
                response["links"] = []
                return response
            if system == RECALL_VERIFY_SYSTEM:
                return complete_v2_fixture_verdict(payload,
                    need_statuses={0: "supported", 1: "partial", 2: "unknown", 3: "refuted"})
            return original_chat(system, content)

        self.model.chat_json = mixed_assessments
        result = self.recall.query(self.question, mode="deep", max_waves=1)
        state = self.recall.sessions.read(result["session_id"])
        self.assertFalse(result["complete"])
        self.assertEqual(state["covered_needs"], [0])
        self.assertEqual(state["resolved_needs"], [0, 3])
        self.assertEqual([item["status"] for item in state["need_assessments"]],
                         ["supported", "partial", "unknown", "refuted"])
        self.assertEqual(result["missing_requirements"], result["needs"][1:3])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_source_refuted_premise_resolves_without_certifying_the_false_claim(self):
        original_chat = self.model.chat_json

        def correcting_premise(system, content):
            payload = model_payload(content)
            if system == RECALL_PLAN_SYSTEM:
                return {"needs": ["乙拒绝甲求助的前提是否成立？"], "cues": ["求助信"]}
            if system == RECALL_MAP_SYSTEM:
                response = original_chat(system, content)
                response["links"] = []
                return response
            if system == RECALL_VERIFY_SYSTEM:
                return complete_v2_fixture_verdict(payload,
                    need_statuses={0: "refuted" if any(f["need_indices"] for f in payload["proposal"]["facts"]) else "unknown"})
            return original_chat(system, content)

        self.model.chat_json = correcting_premise
        result = self.recall.query(self.question, mode="deep")
        state = self.recall.sessions.read(result["session_id"])
        self.assertTrue(result["complete"], result)
        self.assertEqual(state["covered_needs"], [])
        self.assertEqual(state["resolved_needs"], [0])
        self.assertEqual(result["missing_requirements"], [])
        self.assertIn("答应", result["answer"])
        self.assertNotIn("乙拒绝了", result["answer"])
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_valid_v1_checkpoint_requires_fresh_session_without_calls_or_rewrite(self):
        first = self.recall.query(self.question, mode="light", max_waves=1, learn=False)
        path = Path(first["checkpoint_path"])
        state = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(state["version"], 2)
        state["version"] = 1
        for key in ("pending_facts", "evidence_contexts", "need_assessments", "resolved_needs"):
            state.pop(key, None)
        state.pop("checksum")
        state["checksum"] = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest()
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        before = path.read_bytes()
        calls, embeddings = len(self.model.calls), self.model.embedding_calls
        with self.assertRaisesRegex(ValueError, "version|fresh|v1"):
            self.recall.query(self.question, mode="light", resume=first["session_id"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(self.model.calls), calls)
        self.assertEqual(self.model.embedding_calls, embeddings)
        self.assertEqual(self.recall.feedback.valid_learned_edge_ids(), set())

    def test_feedback_without_a_verified_association_is_an_explicit_noop(self):
        result = self.recall.query(self.question, learn=False)
        self.assertTrue(result["complete"], result)
        self.assertTrue(result["evidence"])
        self.assertTrue(result["used_edge_ids"])
        self.assertEqual(result["feedback_edge_ids"], [])
        feedback = self.recall.apply_feedback(result["session_id"], positive=True, feedback_id="no-verified-edge")
        self.assertFalse(feedback["applied"])
        self.assertEqual(feedback["reason"], "no_verified_association")
        self.assertEqual(feedback["changes"], [])
        self.assertEqual([self.weight(edge) for edge in self.edge_ids], [0.8] * 3)
        with self.db.connection() as connection:
            self.assertIsNone(connection.execute("SELECT 1 FROM sqlite_master WHERE name='recall_feedback_event'").fetchone())


if __name__ == "__main__":
    unittest.main()
