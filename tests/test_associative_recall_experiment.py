from __future__ import annotations

from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import socket
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from benchmarks import run_associative_recall_experiment as runner
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.llm import CampaignHttpBudget
from memory_demo.retrieval.progressive import RecallSessionStore


class AssociativeRecallExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(CampaignHttpBudget._clear_process_cache_for_test)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.sqlite"
        connection = sqlite3.connect(self.source)
        connection.executescript("CREATE TABLE source(id INTEGER PRIMARY KEY,raw_text TEXT);"
                                 "CREATE TABLE episode(id INTEGER PRIMARY KEY,source_id INTEGER,source_key TEXT,segment_index INTEGER,text TEXT);"
                                 "INSERT INTO source VALUES(1,'original source');"
                                 "INSERT INTO episode VALUES(1,1,'main/a.json',0,'summary');")
        connection.close()
        self.questions = [{"id": f"q{i}", "question": f"Question {i}?"} for i in range(3)]
        self.manifest = self.root / "runtime.json"
        self.manifest.write_text(json.dumps({"questions": self.questions}), encoding="utf-8")
        self.config = AppConfig(database_path=self.source, log_dir=self.root / "unused",
                                model=ModelConfig(api_key="DO_NOT_EXPORT_THIS_SECRET", reasoning_model="offline-reasoner"))
        self.now = 1000.0
        self.calls, self.model_calls, self.app_calls = [], [], []
        self.interrupt = False
        self.interrupt_before_checkpoint = False
        self.learning_receipt_committed = False
        self.status = "time_budget"
        self.protocol = 3
        self.fingerprint = {"test-file.py": "fixed-code"}
        self.addCleanup(patch.stopall)
        patch.object(runner, "_code_fingerprints", lambda: deepcopy(self.fingerprint)).start()
        for name in ("connect", "connect_ex"):
            patch.object(socket.socket, name, side_effect=AssertionError("no network allowed")).start()
        patch.object(socket, "getaddrinfo", side_effect=AssertionError("no network allowed")).start()
        self.original_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def factories(self):
        owner = self

        class Application:
            def __init__(self, config):
                owner.app_calls.append(str(config.database_path))
                self.db = object()

        class Provider:
            def __init__(self, config, logger, *, accounting, campaign_budget):
                owner.model_calls.append(config)
                self.budget, self.logger = campaign_budget, logger
                owner.assertEqual(config.reasoning_max_tokens, 8192)
                if owner.interrupt_before_checkpoint:
                    owner.interrupt_before_checkpoint = False
                    raise SystemExit("hard stop before runtime starts")

        class Service:
            protocol_version = owner.protocol

            def __init__(self, config, db, model):
                self.config, self.model = config, model
                self.sessions = RecallSessionStore(config.log_dir / "recall_sessions")

            def _validate_delivered_sources(self, state):
                owner.assertEqual(state["facts"], [])

            def _recover_committed_learning(self, state, *, refresh):
                if not owner.learning_receipt_committed:
                    return False
                owner.assertTrue(refresh)
                state["status"] = "complete"
                state["learning"] = {"status": "applied", "association_ids": [7]}
                return True

            def query(self, question, **kwargs):
                owner.calls.append({"question": question, "kwargs": kwargs, "database": str(self.config.database_path)})
                self.model.budget.reserve()
                self.model.logger.emit("offline", text=self.config.model.api_key)
                if kwargs["resume"]:
                    state = self.sessions.read(kwargs["resume"])
                else:
                    state = {"version": self.protocol_version, "session_id": uuid4().hex,
                             "question": question, "context": "", "needs": ["need"],
                             "facts": [], "pending_facts": [], "resolved_needs": [],
                             "elapsed_seconds": 0.0, "metrics": {}, "learning": {"status": "not_attempted"}}
                state["status"] = "running"
                self.sessions.write(state)
                if owner.interrupt:
                    owner.interrupt = False
                    owner.now += 100.0
                    raise SystemExit("hard stop after request reservation")
                state["status"] = owner.status
                state["elapsed_seconds"] += 2.0
                if self.protocol_version >= 3:
                    state["candidate_episode_ids"] = [1]
                    state["presented_episode_ids"] = [1]
                self.sessions.write(state)
                return runner._recover_result(state, self, self.protocol_version)

        return Application, Provider, Service

    def run_campaign(self, output=None, **kwargs):
        app, model, service = self.factories()
        return runner.run(self.source, output or self.root / "campaign", manifest=self.manifest,
                          config=self.config, application_factory=app, model_factory=model,
                          service_factory=service, wall_clock=lambda: self.now, protocol=self.protocol, **kwargs)

    def test_prepare_only_selects_independent_clones_without_services_or_provider(self):
        report = self.run_campaign(prepare_only=True, question_ids=["q2", "q0"])
        self.assertEqual([q["id"] for q in report["prepared_questions"]], ["q2", "q0"])
        self.assertEqual((self.calls, self.app_calls, self.model_calls), ([], [], []))
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 0)
        self.assertEqual(report["binding"]["runtime_questions"], [self.questions[2], self.questions[0]])
        self.assertFalse(report["binding"]["learn"])
        paths = [Path(q["database"]) for q in report["prepared_questions"]]
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(runner._semantic_snapshot(p) == report["binding"]["source_semantic_sha256"] for p in paths))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_terminal_time_budget_has_receipts_and_resume_never_dispatches(self):
        first = self.run_campaign()
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(c["kwargs"]["learn"] is False for c in self.calls))
        self.assertEqual([c["question"] for c in self.calls], [q["question"] for q in self.questions])
        self.assertEqual(first["campaign_http_budget"]["reserved_http_attempts"], 3)
        self.now += 10000
        resumed = self.run_campaign(resume=True)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(resumed["completed_attempt_receipts"], 3)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], 3)
        attempt = self.root / "campaign/question-01/first"
        receipt = runner._read(attempt / "receipt.json")
        self.assertTrue({"result.json", "provider.json", "checkpoint.json", "invocations.json"} <= receipt["files"].keys())
        self.assertTrue(any(name.startswith("provider-") for name in receipt["files"]))
        for name, digest in receipt["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)
        exported = "\n".join(p.read_text(encoding="utf-8") for p in (self.root / "campaign").rglob("*.json"))
        self.assertNotIn(self.config.model.api_key, exported)

    def test_running_hard_stop_uses_remaining_original_deadline_and_http_budget(self):
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["kwargs"]["timeout_seconds"], 360.0)
        CampaignHttpBudget._clear_process_cache_for_test()
        report = self.run_campaign(resume=True, question_ids=["q0"])
        self.assertEqual(self.calls[1]["kwargs"]["timeout_seconds"], 260.0)
        self.assertIsNotNone(self.calls[1]["kwargs"]["resume"])
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 2)

    def test_expired_running_checkpoint_finishes_without_another_provider(self):
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"])
        self.now += 500
        report = self.run_campaign(resume=True, question_ids=["q0"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(report["rows"][0]["status"], "time_budget")
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)

    def test_expired_before_first_checkpoint_still_commits_empty_terminal_result(self):
        self.interrupt_before_checkpoint = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"])
        self.now += 500
        report = self.run_campaign(resume=True, question_ids=["q0"])
        self.assertEqual(self.calls, [])
        self.assertEqual(report["rows"][0]["status"], "time_budget")
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 0)

    def test_committed_learning_recovers_after_hard_stop_and_expired_deadline(self):
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], learn=True)
        self.learning_receipt_committed = True
        self.now += 500
        report = self.run_campaign(resume=True, question_ids=["q0"], learn=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(report["rows"][0]["status"], "complete")
        result = runner._read(self.root / "campaign/question-01/first/result.json")
        self.assertEqual(result["learning"], {"status": "applied", "association_ids": [7]})

    def test_terminal_checkpoint_missing_receipt_recovers_without_call(self):
        self.run_campaign(question_ids=["q0"])
        attempt = self.root / "campaign/question-01/first"
        (attempt / "receipt.json").unlink()
        report = self.run_campaign(resume=True, question_ids=["q0"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(report["completed_attempt_receipts"], 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])

    def test_v4_terminal_recovery_keeps_review_diagnostics_without_new_call(self):
        self.protocol = 4
        self.run_campaign(question_ids=["q0"])
        attempt = self.root / "campaign/question-01/first"
        (attempt / "receipt.json").unlink()
        self.run_campaign(resume=True, question_ids=["q0"])
        result = runner._read(attempt / "result.json")
        self.assertEqual(result["review_protocol_version"], 4)
        self.assertEqual(result["candidate_episode_ids"], [1])
        self.assertEqual(result["presented_episode_ids"], [1])
        self.assertEqual(result["stage_calls"], [])
        self.assertEqual(result["stage_errors"], [])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)

    def test_v2_missing_diagnostic_is_null(self):
        self.protocol = 2
        self.run_campaign(question_ids=["q0"])
        result = runner._read(self.root / "campaign/question-01/first/result.json")
        self.assertIsNone(result["candidate_episode_ids"])
        self.assertIsNone(result["presented_episode_ids"])
        self.assertEqual(result["review_protocol_version"], 2)

    def test_v7_terminal_recovery_preserves_service_window_annotations(self):
        self.protocol = 7
        base_factories = self.factories

        def annotated_factories():
            app, model, service = base_factories()
            service.annotate_completion = staticmethod(lambda result, state: {
                **result, "window_diagnostics": deepcopy(state.get("window_diagnostics", {}))})
            return app, model, service

        with patch.object(self, "factories", side_effect=annotated_factories):
            self.run_campaign(question_ids=["q0"])
            attempt = self.root / "campaign/question-01/first"
            result = runner._read(attempt / "result.json")
            sessions = RecallSessionStore(attempt / "logs/recall_sessions")
            state = sessions.read(result["session_id"])
            diagnostics = {"read_intervals": {"1": [[5, 12]]}, "unread_prefix": True}
            state["window_diagnostics"] = diagnostics
            (attempt / "receipt.json").unlink()
            sessions.write(state)
            self.now += 10000
            report = self.run_campaign(resume=True, question_ids=["q0"])

        recovered = runner._read(attempt / "result.json")
        self.assertEqual(recovered["review_protocol_version"], 7)
        self.assertEqual(recovered["window_diagnostics"], diagnostics)
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])
        for name, digest in runner._read(attempt / "receipt.json")["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)

    def test_v7_rejects_older_checkpoint_before_provider_dispatch(self):
        self.protocol = 7
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"])
        attempt = self.root / "campaign/question-01/first"
        sessions = RecallSessionStore(attempt / "logs/recall_sessions")
        checkpoint = next((attempt / "logs/recall_sessions").glob("*.json"))
        state = sessions.read(checkpoint.stem)
        state["version"] = 6
        sessions.write(state)

        with self.assertRaisesRegex(ValueError, "checkpoint request or protocol differs"):
            self.run_campaign(resume=True, question_ids=["q0"])

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(runner._read(self.root / "campaign/http-budget.json")["reserved_http_attempts"], 1)

    def test_v8_factory_and_cli_are_opt_in_with_fixed_campaign_options(self):
        from memory_demo.retrieval.progressive_v8 import ProgressiveRecallV8

        self.assertIs(runner._service_factory(8), ProgressiveRecallV8)
        report = {"campaign_id": "offline", "prepared_only": True,
                  "completed_attempt_receipts": 0, "expected_attempts": 3,
                  "stop_reason": None, "campaign_http_budget": {}}
        arguments = ["--source-database", str(self.source), "--output", str(self.root / "cli"),
                     "--manifest", str(self.manifest), "--prepare-only", "--max-http-attempts", "90"]
        with patch.object(runner, "run", return_value=report) as dispatch, patch("sys.stdout", new=StringIO()):
            self.assertEqual(runner.main(arguments + ["--protocol", "8"]), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 8)
            self.assertEqual(dispatch.call_args.kwargs["max_http_attempts"], 90)
            self.assertFalse(dispatch.call_args.kwargs["learn"])
            self.assertEqual(runner.main(arguments), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 3)

    def test_v8_terminal_recovery_preserves_feedback_diagnostics_without_dispatch(self):
        self.protocol = 8
        base_factories = self.factories

        def annotated_factories():
            app, model, service = base_factories()
            service.annotate_completion = staticmethod(lambda result, state: {
                **result, "schedule_protocol": deepcopy(state.get("schedule_protocol", {})),
                "local_review_trace": deepcopy(state.get("local_review_trace", []))})
            return app, model, service

        with patch.object(self, "factories", side_effect=annotated_factories):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
            attempt = self.root / "campaign/question-01/first"
            result = runner._read(attempt / "result.json")
            sessions = RecallSessionStore(attempt / "logs/recall_sessions")
            state = sessions.read(result["session_id"])
            schedule_protocol = {"version": 8, "pending_review_first": True}
            local_review_trace = [{"candidate_id": "candidate-a", "action": "review"},
                                  {"support_id": "support-b", "action": "withdraw"}]
            state["schedule_protocol"] = schedule_protocol
            state["local_review_trace"] = local_review_trace
            (attempt / "receipt.json").unlink()
            sessions.write(state)
            self.now += 10000
            report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)

        recovered = runner._read(attempt / "result.json")
        self.assertEqual(recovered["review_protocol_version"], 8)
        self.assertEqual(recovered["schedule_protocol"], schedule_protocol)
        self.assertEqual(recovered["local_review_trace"], local_review_trace)
        self.assertEqual(recovered["candidate_episode_ids"], [1])
        self.assertEqual(recovered["presented_episode_ids"], [1])
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])
        for name, digest in runner._read(attempt / "receipt.json")["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)

    def test_v8_interruption_retains_original_deadline_and_bound_budget(self):
        self.protocol = 8
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[0]["kwargs"]["timeout_seconds"], 360.0)
        CampaignHttpBudget._clear_process_cache_for_test()
        report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[1]["kwargs"]["timeout_seconds"], 260.0)
        self.assertIsNotNone(self.calls[1]["kwargs"]["resume"])
        self.assertTrue(all(c["kwargs"]["learn"] is False for c in self.calls))
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 2)
        with self.assertRaisesRegex(ValueError, "parameters changed"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=91)
        self.now += 10000
        self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.model_calls), 2)

    def test_v8_ninety_reservations_stop_remaining_questions_and_do_not_reset(self):
        self.protocol = 8
        base_factories = self.factories

        def exhausting_factories():
            app, model, service = base_factories()
            query = service.query

            def consume_budget_then_query(instance, question, **kwargs):
                for _ in range(89):
                    instance.model.budget.reserve()
                return query(instance, question, **kwargs)

            service.query = consume_budget_then_query
            return app, model, service

        with patch.object(self, "factories", side_effect=exhausting_factories):
            report = self.run_campaign(max_http_attempts=90)
            self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 90)
            self.assertEqual(report["stop_reason"], "campaign_http_budget_exhausted")
            self.assertEqual(report["completed_attempt_receipts"], 1)
            self.assertEqual(len(self.calls), 1)
            CampaignHttpBudget._clear_process_cache_for_test()
            resumed = self.run_campaign(resume=True, max_http_attempts=90)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], 90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_v8_rejects_v7_checkpoint_before_provider_dispatch(self):
        self.protocol = 8
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        attempt = self.root / "campaign/question-01/first"
        sessions = RecallSessionStore(attempt / "logs/recall_sessions")
        checkpoint = next((attempt / "logs/recall_sessions").glob("*.json"))
        state = sessions.read(checkpoint.stem)
        state["version"] = 7
        sessions.write(state)
        with self.assertRaisesRegex(ValueError, "checkpoint request or protocol differs"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(runner._read(self.root / "campaign/http-budget.json")["reserved_http_attempts"], 1)

    def test_v9_factory_and_cli_are_opt_in_with_fixed_campaign_options(self):
        from memory_demo.retrieval.progressive_v9 import ProgressiveRecallV9

        self.assertIs(runner._service_factory(9), ProgressiveRecallV9)
        report = {"campaign_id": "offline", "prepared_only": True,
                  "completed_attempt_receipts": 0, "expected_attempts": 3,
                  "stop_reason": None, "campaign_http_budget": {}}
        arguments = ["--source-database", str(self.source), "--output", str(self.root / "cli"),
                     "--manifest", str(self.manifest), "--prepare-only", "--max-http-attempts", "90"]
        with patch.object(runner, "run", return_value=report) as dispatch, patch("sys.stdout", new=StringIO()):
            self.assertEqual(runner.main(arguments + ["--protocol", "9"]), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 9)
            self.assertEqual(dispatch.call_args.kwargs["max_http_attempts"], 90)
            self.assertFalse(dispatch.call_args.kwargs["learn"])
            self.assertEqual(runner.main(arguments), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 3)

    def test_v9_terminal_recovery_preserves_feedback_diagnostics_without_dispatch(self):
        self.protocol = 9
        base_factories = self.factories

        def annotated_factories():
            app, model, service = base_factories()
            service.annotate_completion = staticmethod(lambda result, state: {
                **result, "schedule_protocol": deepcopy(state.get("schedule_protocol", {})),
                "local_review_trace": deepcopy(state.get("local_review_trace", []))})
            return app, model, service

        with patch.object(self, "factories", side_effect=annotated_factories):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
            attempt = self.root / "campaign/question-01/first"
            result = runner._read(attempt / "result.json")
            sessions = RecallSessionStore(attempt / "logs/recall_sessions")
            state = sessions.read(result["session_id"])
            schedule_protocol = {"version": 9, "pending_review_first": True}
            local_review_trace = [{"candidate_id": "candidate-a", "action": "review"},
                                  {"support_id": "support-b", "action": "withdraw"}]
            state["schedule_protocol"] = schedule_protocol
            state["local_review_trace"] = local_review_trace
            (attempt / "receipt.json").unlink()
            sessions.write(state)
            self.now += 10000
            report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)

        recovered = runner._read(attempt / "result.json")
        self.assertEqual(recovered["review_protocol_version"], 9)
        self.assertEqual(recovered["schedule_protocol"], schedule_protocol)
        self.assertEqual(recovered["local_review_trace"], local_review_trace)
        self.assertEqual(recovered["candidate_episode_ids"], [1])
        self.assertEqual(recovered["presented_episode_ids"], [1])
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])
        for name, digest in runner._read(attempt / "receipt.json")["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)

    def test_v9_interruption_retains_original_deadline_and_bound_budget(self):
        self.protocol = 9
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[0]["kwargs"]["timeout_seconds"], 360.0)
        CampaignHttpBudget._clear_process_cache_for_test()
        report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[1]["kwargs"]["timeout_seconds"], 260.0)
        self.assertIsNotNone(self.calls[1]["kwargs"]["resume"])
        self.assertTrue(all(c["kwargs"]["learn"] is False for c in self.calls))
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 2)
        with self.assertRaisesRegex(ValueError, "parameters changed"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=91)
        self.now += 10000
        self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.model_calls), 2)

    def test_v9_ninety_reservations_stop_remaining_questions_and_do_not_reset(self):
        self.protocol = 9
        base_factories = self.factories

        def exhausting_factories():
            app, model, service = base_factories()
            query = service.query

            def consume_budget_then_query(instance, question, **kwargs):
                for _ in range(89):
                    instance.model.budget.reserve()
                return query(instance, question, **kwargs)

            service.query = consume_budget_then_query
            return app, model, service

        with patch.object(self, "factories", side_effect=exhausting_factories):
            report = self.run_campaign(max_http_attempts=90)
            self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 90)
            self.assertEqual(report["stop_reason"], "campaign_http_budget_exhausted")
            self.assertEqual(report["completed_attempt_receipts"], 1)
            self.assertEqual(len(self.calls), 1)
            CampaignHttpBudget._clear_process_cache_for_test()
            resumed = self.run_campaign(resume=True, max_http_attempts=90)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], 90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_v9_rejects_v8_checkpoint_before_provider_dispatch(self):
        self.protocol = 9
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        attempt = self.root / "campaign/question-01/first"
        sessions = RecallSessionStore(attempt / "logs/recall_sessions")
        checkpoint = next((attempt / "logs/recall_sessions").glob("*.json"))
        state = sessions.read(checkpoint.stem)
        state["version"] = 8
        sessions.write(state)
        with self.assertRaisesRegex(ValueError, "checkpoint request or protocol differs"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(runner._read(self.root / "campaign/http-budget.json")["reserved_http_attempts"], 1)

    def test_v10_factory_and_cli_are_opt_in_with_fixed_campaign_options(self):
        from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10

        self.assertIs(runner._service_factory(10), ProgressiveRecallV10)
        report = {"campaign_id": "offline", "prepared_only": True,
                  "completed_attempt_receipts": 0, "expected_attempts": 3,
                  "stop_reason": None, "campaign_http_budget": {}}
        arguments = ["--source-database", str(self.source), "--output", str(self.root / "cli"),
                     "--manifest", str(self.manifest), "--prepare-only", "--max-http-attempts", "90"]
        with patch.object(runner, "run", return_value=report) as dispatch, patch("sys.stdout", new=StringIO()):
            self.assertEqual(runner.main(arguments + ["--protocol", "10"]), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 10)
            self.assertEqual(dispatch.call_args.kwargs["max_http_attempts"], 90)
            self.assertFalse(dispatch.call_args.kwargs["learn"])
            self.assertEqual(runner.main(arguments), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 3)

    def test_v10_terminal_recovery_preserves_feedback_diagnostics_without_dispatch(self):
        self.protocol = 10
        base_factories = self.factories

        def annotated_factories():
            app, model, service = base_factories()
            service.annotate_completion = staticmethod(lambda result, state: {
                **result, "schedule_protocol": deepcopy(state.get("schedule_protocol", {})),
                "local_review_trace": deepcopy(state.get("local_review_trace", [])),
                "contract_anchor_binding": deepcopy(state.get("contract_anchor_binding"))})
            return app, model, service

        with patch.object(self, "factories", side_effect=annotated_factories):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
            attempt = self.root / "campaign/question-01/first"
            result = runner._read(attempt / "result.json")
            sessions = RecallSessionStore(attempt / "logs/recall_sessions")
            state = sessions.read(result["session_id"])
            schedule_protocol = {"version": 10, "pending_review_first": True}
            local_review_trace = [{"candidate_id": "candidate-a", "action": "review"},
                                  {"support_id": "support-b", "action": "withdraw"}]
            state["schedule_protocol"] = schedule_protocol
            state["local_review_trace"] = local_review_trace
            anchor_binding = {"protocol": {"version": 1}, "request_fragments_sha256": "fixture-bound-fragments"}
            state["contract_anchor_binding"] = anchor_binding
            (attempt / "receipt.json").unlink()
            sessions.write(state)
            self.now += 10000
            report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)

        recovered = runner._read(attempt / "result.json")
        self.assertEqual(recovered["review_protocol_version"], 10)
        self.assertEqual(recovered["schedule_protocol"], schedule_protocol)
        self.assertEqual(recovered["local_review_trace"], local_review_trace)
        self.assertEqual(recovered["contract_anchor_binding"], anchor_binding)
        self.assertEqual(recovered["candidate_episode_ids"], [1])
        self.assertEqual(recovered["presented_episode_ids"], [1])
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])
        for name, digest in runner._read(attempt / "receipt.json")["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)

    def test_v10_interruption_retains_original_deadline_and_bound_budget(self):
        self.protocol = 10
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[0]["kwargs"]["timeout_seconds"], 360.0)
        CampaignHttpBudget._clear_process_cache_for_test()
        report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[1]["kwargs"]["timeout_seconds"], 260.0)
        self.assertIsNotNone(self.calls[1]["kwargs"]["resume"])
        self.assertTrue(all(c["kwargs"]["learn"] is False for c in self.calls))
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 2)
        with self.assertRaisesRegex(ValueError, "parameters changed"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=91)
        self.now += 10000
        self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.model_calls), 2)

    def test_v10_ninety_reservations_stop_remaining_questions_and_do_not_reset(self):
        self.protocol = 10
        base_factories = self.factories

        def exhausting_factories():
            app, model, service = base_factories()
            query = service.query

            def consume_budget_then_query(instance, question, **kwargs):
                for _ in range(89):
                    instance.model.budget.reserve()
                return query(instance, question, **kwargs)

            service.query = consume_budget_then_query
            return app, model, service

        with patch.object(self, "factories", side_effect=exhausting_factories):
            report = self.run_campaign(max_http_attempts=90)
            self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 90)
            self.assertEqual(report["stop_reason"], "campaign_http_budget_exhausted")
            self.assertEqual(report["completed_attempt_receipts"], 1)
            self.assertEqual(len(self.calls), 1)
            CampaignHttpBudget._clear_process_cache_for_test()
            resumed = self.run_campaign(resume=True, max_http_attempts=90)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], 90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_v10_rejects_v9_checkpoint_before_provider_dispatch(self):
        self.protocol = 10
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        attempt = self.root / "campaign/question-01/first"
        sessions = RecallSessionStore(attempt / "logs/recall_sessions")
        checkpoint = next((attempt / "logs/recall_sessions").glob("*.json"))
        state = sessions.read(checkpoint.stem)
        state["version"] = 9
        sessions.write(state)
        with self.assertRaisesRegex(ValueError, "checkpoint request or protocol differs"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(runner._read(self.root / "campaign/http-budget.json")["reserved_http_attempts"], 1)

    def test_v11_factory_and_cli_are_opt_in_with_fixed_campaign_options(self):
        from memory_demo.retrieval.progressive_v11 import ProgressiveRecallV11

        self.assertIs(runner._service_factory(11), ProgressiveRecallV11)
        report = {"campaign_id": "offline", "prepared_only": True,
                  "completed_attempt_receipts": 0, "expected_attempts": 3,
                  "stop_reason": None, "campaign_http_budget": {}}
        arguments = ["--source-database", str(self.source), "--output", str(self.root / "cli"),
                     "--manifest", str(self.manifest), "--prepare-only", "--max-http-attempts", "90"]
        with patch.object(runner, "run", return_value=report) as dispatch, patch("sys.stdout", new=StringIO()):
            self.assertEqual(runner.main(arguments + ["--protocol", "11"]), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 11)
            self.assertEqual(dispatch.call_args.kwargs["max_http_attempts"], 90)
            self.assertFalse(dispatch.call_args.kwargs["learn"])
            self.assertEqual(runner.main(arguments), 0)
            self.assertEqual(dispatch.call_args.kwargs["protocol"], 3)

    def test_v11_terminal_recovery_preserves_feedback_diagnostics_without_dispatch(self):
        self.protocol = 11
        base_factories = self.factories

        def annotated_factories():
            app, model, service = base_factories()
            service.annotate_completion = staticmethod(lambda result, state: {
                **result, "schedule_protocol": deepcopy(state.get("schedule_protocol", {})),
                "local_review_trace": deepcopy(state.get("local_review_trace", [])),
                "contract_anchor_binding": deepcopy(state.get("contract_anchor_binding"))})
            return app, model, service

        with patch.object(self, "factories", side_effect=annotated_factories):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
            attempt = self.root / "campaign/question-01/first"
            result = runner._read(attempt / "result.json")
            sessions = RecallSessionStore(attempt / "logs/recall_sessions")
            state = sessions.read(result["session_id"])
            schedule_protocol = {"version": 11, "pending_review_first": True}
            local_review_trace = [{"candidate_id": "candidate-a", "action": "review"},
                                  {"support_id": "support-b", "action": "withdraw"}]
            state["schedule_protocol"] = schedule_protocol
            state["local_review_trace"] = local_review_trace
            anchor_binding = {"protocol": {"version": 1}, "request_fragments_sha256": "fixture-bound-fragments"}
            state["contract_anchor_binding"] = anchor_binding
            (attempt / "receipt.json").unlink()
            sessions.write(state)
            self.now += 10000
            report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)

        recovered = runner._read(attempt / "result.json")
        self.assertEqual(recovered["review_protocol_version"], 11)
        self.assertEqual(recovered["schedule_protocol"], schedule_protocol)
        self.assertEqual(recovered["local_review_trace"], local_review_trace)
        self.assertEqual(recovered["contract_anchor_binding"], anchor_binding)
        self.assertEqual(recovered["candidate_episode_ids"], [1])
        self.assertEqual(recovered["presented_episode_ids"], [1])
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(runner._read(attempt / "provider.json")["recovered_terminal_checkpoint_without_recall"])
        for name, digest in runner._read(attempt / "receipt.json")["files"].items():
            self.assertEqual(runner._digest(attempt / name), digest)

    def test_v11_interruption_retains_original_deadline_and_bound_budget(self):
        self.protocol = 11
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[0]["kwargs"]["timeout_seconds"], 360.0)
        CampaignHttpBudget._clear_process_cache_for_test()
        report = self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(self.calls[1]["kwargs"]["timeout_seconds"], 260.0)
        self.assertIsNotNone(self.calls[1]["kwargs"]["resume"])
        self.assertTrue(all(c["kwargs"]["learn"] is False for c in self.calls))
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 2)
        with self.assertRaisesRegex(ValueError, "parameters changed"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=91)
        self.now += 10000
        self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.model_calls), 2)

    def test_v11_ninety_reservations_stop_remaining_questions_and_do_not_reset(self):
        self.protocol = 11
        base_factories = self.factories

        def exhausting_factories():
            app, model, service = base_factories()
            query = service.query

            def consume_budget_then_query(instance, question, **kwargs):
                for _ in range(89):
                    instance.model.budget.reserve()
                return query(instance, question, **kwargs)

            service.query = consume_budget_then_query
            return app, model, service

        with patch.object(self, "factories", side_effect=exhausting_factories):
            report = self.run_campaign(max_http_attempts=90)
            self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 90)
            self.assertEqual(report["stop_reason"], "campaign_http_budget_exhausted")
            self.assertEqual(report["completed_attempt_receipts"], 1)
            self.assertEqual(len(self.calls), 1)
            CampaignHttpBudget._clear_process_cache_for_test()
            resumed = self.run_campaign(resume=True, max_http_attempts=90)
        self.assertEqual(resumed["campaign_http_budget"]["reserved_http_attempts"], 90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_hash)

    def test_v11_rejects_v10_checkpoint_before_provider_dispatch(self):
        self.protocol = 11
        self.interrupt = True
        with self.assertRaises(SystemExit):
            self.run_campaign(question_ids=["q0"], max_http_attempts=90)
        attempt = self.root / "campaign/question-01/first"
        sessions = RecallSessionStore(attempt / "logs/recall_sessions")
        checkpoint = next((attempt / "logs/recall_sessions").glob("*.json"))
        state = sessions.read(checkpoint.stem)
        state["version"] = 10
        sessions.write(state)
        with self.assertRaisesRegex(ValueError, "checkpoint request or protocol differs"):
            self.run_campaign(resume=True, question_ids=["q0"], max_http_attempts=90)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(runner._read(self.root / "campaign/http-budget.json")["reserved_http_attempts"], 1)

    def test_resume_rejects_changed_code_models_questions_source_or_missing_budget(self):
        for kind in ("code", "model", "questions", "source", "budget"):
            with self.subTest(kind=kind):
                output = self.root / kind
                self.run_campaign(output, prepare_only=True)
                prior = deepcopy(self.config.model)
                if kind == "code":
                    self.fingerprint["test-file.py"] = "changed"
                elif kind == "model":
                    self.config.model.max_retries += 1
                elif kind == "questions":
                    self.manifest.write_text(json.dumps([{"id": "q0", "question": "changed"}]), encoding="utf-8")
                elif kind == "source":
                    con = sqlite3.connect(self.source)
                    con.execute("UPDATE source SET raw_text='changed' WHERE id=1")
                    con.commit(); con.close()
                else:
                    (output / "http-budget.json").unlink()
                with self.assertRaises(ValueError):
                    self.run_campaign(output, resume=True)
                self.config.model = prior
                self.fingerprint["test-file.py"] = "fixed-code"
                self.manifest.write_text(json.dumps({"questions": self.questions}), encoding="utf-8")
                if kind == "source":
                    con = sqlite3.connect(self.source)
                    con.execute("UPDATE source SET raw_text='original source' WHERE id=1")
                    con.commit(); con.close()
        self.assertEqual(self.calls, [])

    def test_receipt_tampering_and_clone_text_change_are_rejected(self):
        self.run_campaign(question_ids=["q0"])
        attempt = self.root / "campaign/question-01/first"
        result = attempt / "result.json"
        result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.run_campaign(resume=True, question_ids=["q0"])
        self.assertEqual(len(self.calls), 1)
        output = self.root / "clone-change"
        prepared = self.run_campaign(output, prepare_only=True, question_ids=["q0"])
        con = sqlite3.connect(prepared["prepared_questions"][0]["database"])
        con.execute("UPDATE episode SET text='changed'"); con.commit(); con.close()
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.run_campaign(output, resume=True, question_ids=["q0"])

    def test_runtime_manifest_rejects_gold_fields_and_duplicate_selections(self):
        self.manifest.write_text(json.dumps([{**self.questions[0], "evidence_groups": [123]}]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only"):
            self.run_campaign(prepare_only=True)
        self.manifest.write_text(json.dumps(self.questions), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_campaign(prepare_only=True, question_ids=["q0", "q0"])

    def test_shared_cap_stops_before_next_question_without_reset(self):
        report = self.run_campaign(max_http_attempts=1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["stop_reason"], "campaign_http_budget_exhausted")
        report = self.run_campaign(resume=True, max_http_attempts=1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["campaign_http_budget"]["reserved_http_attempts"], 1)

    def test_rolled_back_budget_is_rejected_even_in_a_new_process(self):
        self.run_campaign(max_http_attempts=1)
        path = self.root / "campaign/http-budget.json"
        value = runner._read(path)
        value["reserved_http_attempts"] = 0
        path.write_text(json.dumps(value), encoding="utf-8")
        CampaignHttpBudget._clear_process_cache_for_test()
        with self.assertRaisesRegex(ValueError, "recorded usage"):
            self.run_campaign(resume=True, max_http_attempts=1)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
