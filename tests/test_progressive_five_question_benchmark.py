from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from benchmarks.run_progressive_five_question_benchmark import run, _score_result
from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.embeddings.codec import encode_embedding
from memory_demo.llm import CampaignHttpBudget
from memory_demo.llm.recall_prompts import RECALL_PLAN_SYSTEM, RECALL_MAP_SYSTEM, RECALL_VERIFY_SYSTEM
from memory_demo.repositories import SourceRepository, EpisodeRepository, AssociationRepository
from memory_demo.types import EpisodeDraft, AssociationDraft
from test_progressive_recall import fixture_review_verdict


class ProgressiveFiveQuestionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(CampaignHttpBudget._clear_process_cache_for_test)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "frozen.sqlite"
        db = Database(self.source)
        db.initialize()
        sources, episodes = SourceRepository(db), EpisodeRepository(db)
        root = sources.insert("The letter was sent.")
        target = sources.insert("QUOTE_TERM is present in the reply.")
        self.root_id = episodes.insert(root, "main/a", 0, EpisodeDraft("EPISODE_TERM is in the summary."), encode_embedding([1, 0], 2))
        self.target_id = episodes.insert(target, "main/b", 0, EpisodeDraft("The reply was received."), encode_embedding([0, 1], 2))
        AssociationRepository(db).upsert(AssociationDraft("episode", self.root_id, "episode", self.target_id,
                                                        "semantic", "test", "existing path", weight=0.4, confidence=1.0))
        self.config = AppConfig(database_path=self.source, log_dir=self.directory / "unused",
                               model=ModelConfig(api_key="secret-never-export-this", embedding_dimension=2,
                                                 embedding_model="offline-embedding", reasoning_model="offline-reasoner"))
        self.questions = [{"id": f"q{i}", "label": f"evaluator-label-{i}", "question": f"What happened in question {i}?",
                           "key_evidence_terms": ["EPISODE_TERM", "QUOTE_TERM", "MISSING_TERM"],
                           "expected_answer": "THIS_MUST_NOT_REACH_RUNTIME"} for i in range(5)]
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_text(json.dumps(self.questions), encoding="utf-8")
        self.runtime_calls = []
        self.timeout_first = False

    def factories(self):
        owner = self

        class ObservedApplication(MemoryApplication):
            def recall(app, question, **kwargs):
                with app.db.connection() as connection:
                    learned_before = connection.execute("SELECT COUNT(*) FROM association WHERE association_mode='simple_recall'").fetchone()[0]
                owner.runtime_calls.append({"question": question, "kwargs": kwargs, "database": str(app.db.path),
                                            "learned_before": learned_before})
                return super().recall(question, **kwargs)

        class OfflineProvider:
            def __init__(self, config, logger, *, accounting, campaign_budget):
                self.config, self.logger = config, logger
                self.accounting, self.budget = accounting, campaign_budget
                owner.assertEqual(config.reasoning_max_tokens, 8192)

            def count(self, operation):
                self.budget.reserve()
                scope = self.accounting.begin(operation, "offline")
                self.accounting.record_http_attempt(scope, operation)
                self.accounting.record_http_result(scope, "success")
                self.logger.emit("offline", api_key=self.config.api_key, content="provider body must be hidden")

            def embed(self, cues):
                self.count("embedding")
                return [np.array([1.0, 0.0], dtype=np.float32) for _ in cues]

            def chat_json(self, system, content):
                self.count("chat")
                payload = json.loads(content)
                if system == RECALL_PLAN_SYSTEM:
                    if owner.timeout_first and payload["question"] == owner.questions[0]["question"]:
                        raise TimeoutError("controlled deadline")
                    return {"needs": ["What was the reply?"], "cues": ["letter"]}
                if system == RECALL_MAP_SYSTEM:
                    facts = [{"source_id": s["source_id"], "episode_id": s["episode_id"], "quote": s["text"],
                              "claim": "The source provides this detail.", "need_indices": [0] if s["episode_id"] == owner.target_id else []}
                             for s in payload["sources"]]
                    ids = {f["episode_id"] for f in facts}
                    links = [{"from_episode_id": owner.root_id, "to_episode_id": owner.target_id,
                              "rationale": "The sent letter and its reply."}] if {owner.root_id, owner.target_id}.issubset(ids) else []
                    return {"facts": facts, "links": links}
                if system == RECALL_VERIFY_SYSTEM:
                    facts = payload["proposal"]["facts"]
                    return fixture_review_verdict(payload,
                        accepted_fact_indices=list(range(len(facts))),
                        covered_needs=[0] if any(f["episode_id"] == owner.target_id for f in facts) else [],
                        accepted_link_indices=list(range(len(payload["proposal"]["links"]))))
                raise AssertionError("unexpected prompt")

        return ObservedApplication, OfflineProvider

    def run_campaign(self, output, **kwargs):
        app, model = self.factories()
        return run(self.source, output, manifest=self.manifest, config=self.config,
                   application_factory=app, model_factory=model, **kwargs)

    def test_isolated_five_questions_exact_repeats_and_evaluator_only_keywords(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        output = self.directory / "campaign"
        report = self.run_campaign(output, repeat=True)
        self.assertEqual(report["completed_attempt_receipts"], 10)
        self.assertEqual(len(self.runtime_calls), 10)
        first, repeated = self.runtime_calls[::2], self.runtime_calls[1::2]
        self.assertEqual(len({row["database"] for row in first}), 5)
        self.assertEqual([row["learned_before"] for row in first], [0] * 5)
        self.assertEqual([row["learned_before"] for row in repeated], [1] * 5)
        for original, repeat in zip(first, repeated):
            self.assertEqual(original["database"], repeat["database"])
            self.assertEqual(original["question"], repeat["question"])
        for call in self.runtime_calls:
            self.assertEqual(set(call["kwargs"]), {"mode", "timeout_seconds", "learn", "model", "resume"})
            self.assertEqual(call["kwargs"]["timeout_seconds"], 360.0)
            self.assertEqual(call["kwargs"]["mode"], "deep")
            self.assertNotIn("TERM", call["question"])
            self.assertNotIn("expected_answer", call["kwargs"])
        self.assertEqual(report["summaries"]["first"]["legacy_episode_keywords"], {"matched": 5, "total": 15})
        self.assertEqual(report["summaries"]["first"]["verified_quote_keywords"], {"matched": 5, "total": 15})
        self.assertEqual(report["summaries"]["exact_repeat"]["attempts"], 5)
        self.assertEqual(report["summaries"]["first"]["http_attempt_reservations"], 20)
        self.assertEqual(report["summaries"]["exact_repeat"]["http_attempt_reservations"], 20)
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 40)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
        for path in output.rglob("*.json*"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(self.config.model.api_key, text, str(path))
            self.assertNotIn("provider body must be hidden", text, str(path))
        self.assertEqual(len(list(output.glob("question-*/first/checkpoint.json"))), 5)

    def test_resume_skips_completed_time_budget_attempts_and_protects_existing_output(self):
        self.timeout_first = True
        output = self.directory / "campaign"
        initial = self.run_campaign(output)
        self.assertEqual(initial["rows"][0]["runtime"]["status"], "time_budget")
        calls = len(self.runtime_calls)
        used = initial["campaign_http_budget"]["reserved_http_attempts"]
        resumed = self.run_campaign(output, resume=True)
        self.assertEqual(len(self.runtime_calls), calls)
        self.assertEqual(resumed["completed_attempt_receipts"], 5)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], used)
        with self.assertRaises(FileExistsError):
            self.run_campaign(output)
        result_file = output / "question-01/first/result.json"
        result_file.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_campaign(output, resume=True)
        self.assertEqual(len(self.runtime_calls), calls)

    def test_historical_score_uses_eight_distinct_delivered_episodes_not_eight_facts(self):
        db = Database(self.source)
        repository = EpisodeRepository(db)
        ids = [self.root_id, self.target_id]
        for index in range(7):
            ids.append(repository.insert(1, f"main/extra-{index}", 0,
                                         EpisodeDraft("EIGHTH" if index == 5 else "NINTH" if index == 6 else "other"),
                                         encode_embedding([0, 1], 2)))
        result = {"evidence": [{"episode_id": self.root_id, "quote": "quote-only"}] * 8
                              + [{"episode_id": eid, "quote": "quote-only"} for eid in ids[1:]]}
        question = {"id": "x", "label": "x", "question": "x", "key_evidence_terms": ["EIGHTH", "NINTH", "quote-only"]}
        scored = _score_result(question, result, db, 1.0, "first")
        self.assertEqual(scored["legacy_episode_keywords"]["selected_episode_ids"], ids[:8])
        self.assertEqual(scored["legacy_episode_keywords"]["matched_terms"], ["EIGHTH"])
        self.assertEqual(scored["verified_quote_keywords"]["matched_terms"], ["quote-only"])
        self.assertFalse(scored["semantic_truth"]["evaluated"])

    def test_prepare_only_makes_five_clones_without_calls_then_resume_executes(self):
        output = self.directory / "prepared-campaign"
        prepared = self.run_campaign(output, prepare_only=True)
        self.assertTrue(prepared["prepared_only"])
        self.assertEqual(len(prepared["prepared_questions"]), 5)
        self.assertEqual(len({row["database"] for row in prepared["prepared_questions"]}), 5)
        self.assertTrue(all(Path(row["database"]).is_file() for row in prepared["prepared_questions"]))
        self.assertEqual(prepared["completed_attempt_receipts"], 0)
        self.assertEqual(prepared["campaign_http_budget"]["reserved_http_attempts"], 0)
        self.assertEqual(self.runtime_calls, [])
        self.assertEqual(list(output.rglob("provider-*.jsonl")), [])
        executed = self.run_campaign(output, resume=True)
        self.assertFalse(executed["prepared_only"])
        self.assertEqual(executed["completed_attempt_receipts"], 5)
        self.assertEqual(executed["campaign_http_budget"]["reserved_http_attempts"], 20)
        self.assertEqual(len(self.runtime_calls), 5)


if __name__ == "__main__":
    unittest.main()
