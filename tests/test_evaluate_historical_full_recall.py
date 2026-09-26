"""Offline scoring boundaries; no runtime or model import is required."""
import ast
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from benchmarks.evaluate_historical_full_recall import (
    IdentityMismatch, evaluate_campaign, evaluate_run, freeze_historical_manifest,
    read_semantic_snapshot, score_groups, verified_episode_ids, verify_identity,
)


# Exact audited source, extracted from the original unchanged scorer (lines 198-210).
ORIGINAL_SCORE = '''def _score_groups(item: dict, final_ids: set[int]) -> list[dict]:
    groups = []
    for group in item["evidence_groups"]:
        alternatives = {int(value) for value in group["alternatives"]}
        matches = sorted(final_ids & alternatives)
        groups.append(
            {
                **group,
                "matches": matches,
                "passed": bool(matches),
            }
        )
    return groups'''


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class HistoricalFullRecallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.database = self.directory / "reference.sqlite"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE source (id INTEGER PRIMARY KEY, raw_text TEXT)")
            db.execute("CREATE TABLE episode (id INTEGER PRIMARY KEY, source_id INTEGER, source_key TEXT, segment_index INTEGER, text TEXT)")
            db.executemany("INSERT INTO source VALUES (?,?)", [(i, f"Original Source {i}") for i in range(1, 91)])
            db.executemany("INSERT INTO episode VALUES (?,?,?,?,?)", [(i, i, f"source/{i}", 0, f"Original Episode {i}") for i in range(1, 91)])
        snapshot = read_semantic_snapshot(self.database)
        self.identity = {"passed": True, "historical_run_id": "synthetic-original-run",
                         "input_report": {"sha256": "original-report-sha"},
                         "semantic_snapshot_sha256": snapshot["semantic_snapshot_sha256"],
                         "matched_database": {"path": str(self.database)}}
        self.audit = {"passed": True, "audit_errors": [], "historical_run_errors": [],
                      "historical_run_id": "synthetic-original-run",
                      "input_files": [{"path": "report.json", "sha256": "original-report-sha"}], "questions": []}
        slot = 1
        for index in range(15):
            groups, bindings = [], []
            for _ in range(4 if index < 12 else 3):
                groups.append({"id": f"slot-{slot}", "description": f"Original slot {slot}",
                               "alternatives": [slot, 90] if slot == 1 else [slot],
                               "oracle_adjustment": "preserve original metadata", "matches": [slot], "passed": True})
                row = snapshot["episodes"][slot]
                bindings.append({"episode_id": slot, "source_key": row["source_key"], "segment_index": 0,
                                 "logged_episode_text_sha256": row["text_sha256"]})
                slot += 1
            self.audit["questions"].append({"id": f"q{index + 1}", "question": f"Original question {index + 1}?",
                                             "layers": {"final": {"groups": groups}}, "logged_final_episode_bindings": bindings})
        self.runtime, self.manifest = freeze_historical_manifest(self.audit, audit_sha256="audit-sha", identity=self.identity)

    def results(self):
        return {q["id"]: {"status": "complete", "complete": True,
                           "evidence": [{"episode_id": g["alternatives"][0]} for g in q["evidence_groups"]],
                           "needs": ["need"], "need_assessments": [{"need_index": 0, "status": "supported"}]}
                for q in self.manifest["questions"]}

    def score(self, results, **kwargs):
        return evaluate_run(self.manifest, results, identity=kwargs.pop("identity", self.identity),
                            database_path=kwargs.pop("database_path", self.database), **kwargs)

    def test_freeze_preserves_fifteen_questions_fifty_seven_slots_and_separates_runtime(self):
        self.assertEqual(len(self.runtime), 15)
        self.assertTrue(all(set(row) == {"id", "question"} for row in self.runtime))
        self.assertEqual(sum(len(q["evidence_groups"]) for q in self.manifest["questions"]), 57)
        group = self.manifest["questions"][0]["evidence_groups"][0]
        self.assertNotIn("passed", group)
        self.assertNotIn("matches", group)
        self.assertEqual(group["oracle_adjustment"], "preserve original metadata")
        corrupted = deepcopy(self.manifest)
        corrupted["questions"][0]["question"] = "Easier question"
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_run(corrupted, {}, identity=self.identity, database_path=self.database)

    def test_original_audited_function_is_exactly_equivalent(self):
        self.assertEqual(hashlib.sha256(ORIGINAL_SCORE.encode()).hexdigest(),
                         "cf78438a2c610ac3cc414ba9ee06098e12eecf0716ecdc3495bf9bd94ac516f8")
        namespace = {}
        module = ast.parse(ORIGINAL_SCORE)
        self.assertEqual(len(module.body), 1)
        exec(compile(module, "frozen-original-score", "exec"), namespace)
        original = namespace["_score_groups"]
        for selected in (set(), {90}, {1, 3, 90}, set(range(1, 58)), set(range(2, 58, 3))):
            for question in self.manifest["questions"]:
                self.assertEqual(score_groups(question, selected), original(question, selected))

    def test_full_verified_suite_alone_can_be_full_recall(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        result = self.score(self.results())
        self.assertTrue(result["full_suite"])
        self.assertTrue(result["full_recall"])
        self.assertEqual(result["summary"]["verified"]["matched_fact_slots"], 57)
        self.assertTrue(result["identity_recomputed_readonly"])
        self.assertEqual(hashlib.sha256(self.database.read_bytes()).hexdigest(), before)

    def test_candidate_presented_and_need_resolved_do_not_supply_verified_evidence(self):
        results = self.results()
        for result in results.values():
            ids = [item["episode_id"] for item in result["evidence"]]
            result.update(evidence=[], candidate_episode_ids=ids, presented_episode_ids=ids)
        result = self.score(results)
        self.assertEqual(result["summary"]["candidate"]["matched_fact_slots"], 57)
        self.assertEqual(result["summary"]["presented"]["matched_fact_slots"], 57)
        self.assertEqual(result["summary"]["verified"]["matched_fact_slots"], 0)
        self.assertEqual(result["questions"][0]["need_resolved"]["reported_resolved_count"], 1)
        self.assertFalse(result["full_recall"])

    def test_verified_dedup_is_before_first_thirty_limit_without_gold_selection(self):
        self.assertEqual(verified_episode_ids({"evidence": [{"episode_id": 1}] * 40 + [{"episode_id": 2}]}), [1, 2])
        result = {"status": "complete", "evidence": [{"episode_id": i} for i in range(60, 90)] + [{"episode_id": 1}]}
        evaluated = self.score({"q1": result})
        self.assertEqual(evaluated["questions"][0]["verified"]["episode_ids"], list(range(60, 90)))
        self.assertEqual(evaluated["questions"][0]["verified"]["matched_fact_slots"], 0)
        self.assertEqual(evaluated["questions"][0]["evidence_limit"]["unique_before_limit"], 31)

    def test_partial_suite_keeps_original_denominator(self):
        result = self.score({"q1": self.results()["q1"]})
        self.assertFalse(result["full_suite"])
        self.assertFalse(result["full_recall"])
        self.assertEqual(result["summary"]["verified"]["required_fact_slots"], 57)
        self.assertEqual(result["summary"]["verified"]["observed_required_fact_slots"], 4)
        self.assertEqual(result["summary"]["verified"]["fact_slot_recall"], 4 / 57)
        self.assertEqual(result["summary"]["verified"]["observed_fact_slot_recall"], 1)

    def test_technical_failure_and_interruption_prevent_full_claim(self):
        for status, error in (("provider_error", "timeout"), ("time_budget", None), ("complete", "technical failure")):
            with self.subTest(status=status, error=error):
                results = self.results()
                results["q1"].update(status=status, error=error)
                result = self.score(results)
                self.assertEqual(result["summary"]["verified"]["matched_fact_slots"], 57)
                self.assertFalse(result["full_recall"])
                self.assertTrue(result["technical_errors"] or result["interrupted_runs"])

    def test_need_resolution_is_separate_and_includes_refutation(self):
        results = self.results()
        results["q1"]["need_assessments"] = [{"need_index": 0, "status": "refuted"}, {"need_index": 1, "status": "unknown"}]
        results["q1"]["needs"] = ["one", "two"]
        result = self.score(results)
        self.assertEqual(result["questions"][0]["need_resolved"]["reported_resolved_count"], 1)
        self.assertTrue(result["full_recall"])

    def test_identity_flag_alone_is_insufficient_and_missing_identity_refuses(self):
        for identity in (None, {"passed": True}):
            with self.subTest(identity=identity), self.assertRaises(IdentityMismatch):
                self.score(self.results(), identity=identity)

    def test_actual_source_and_episode_changes_refuse_even_with_same_ids(self):
        for table, field in (("source", "raw_text"), ("episode", "text")):
            clone = self.directory / f"changed-{table}.sqlite"
            shutil.copyfile(self.database, clone)
            with closing(sqlite3.connect(clone)) as db, db:
                db.execute(f"UPDATE {table} SET {field} = 'different text' WHERE id=90")
            with self.subTest(table=table), self.assertRaisesRegex(IdentityMismatch, "actual SQLite"):
                self.score(self.results(), database_path=clone)
            forged = deepcopy(self.identity)
            forged["semantic_snapshot_sha256"] = read_semantic_snapshot(clone)["semantic_snapshot_sha256"]
            with self.assertRaisesRegex(IdentityMismatch, "pre-registered"):
                self.score(self.results(), identity=forged, database_path=clone)

    def test_schema_only_clone_change_allowed_and_every_clone_recomputed(self):
        clone = self.directory / "initialized.sqlite"
        shutil.copyfile(self.database, clone)
        with closing(sqlite3.connect(clone)) as db, db:
            db.execute("CREATE TABLE new_runtime_session (id TEXT)")
        self.assertNotEqual(clone.read_bytes(), self.database.read_bytes())
        databases = {qid: (clone if qid == "q1" else self.database) for qid in self.results()}
        result = self.score(self.results(), database_path=databases)
        self.assertTrue(result["full_recall"])
        self.assertEqual(result["verified_database_count"], 2)
        with closing(sqlite3.connect(clone)) as db, db:
            db.execute("UPDATE episode SET text='changed' WHERE id=1")
        with self.assertRaises(IdentityMismatch):
            self.score(self.results(), database_path=databases)

    def test_unknown_ids_and_boolean_ids_refuse(self):
        with self.assertRaisesRegex(ValueError, "unknown question"):
            self.score({"foreign": {}})
        with self.assertRaises(IdentityMismatch):
            self.score({"q1": {"status": "complete", "evidence": [{"episode_id": 999}]}})
        with self.assertRaises(ValueError):
            verified_episode_ids({"evidence": [{"episode_id": True}]})

    def test_missing_diagnostic_layers_are_unavailable_not_zero(self):
        results = {"q1": self.results()["q1"]}
        results["q1"]["presented_episode_ids"] = None
        evaluated = self.score(results)
        for layer in ("candidate", "presented"):
            self.assertIsNone(evaluated["questions"][0][layer]["matched_fact_slots"])
            self.assertIsNone(evaluated["summary"][layer]["fact_slot_recall"])
            self.assertEqual(evaluated["summary"][layer]["unavailable_questions"], 1)
        results["q1"]["candidate_episode_ids"] = []
        evaluated = self.score(results)
        self.assertEqual(evaluated["summary"]["candidate"]["matched_fact_slots"], 0)
        self.assertTrue(evaluated["questions"][0]["candidate"]["available"])

    def make_campaign(self):
        directory = self.directory / "campaign"
        attempt = directory / "question-01" / "first"
        attempt.mkdir(parents=True)
        database = attempt.parent / "memory.sqlite"
        shutil.copyfile(self.database, database)
        write_json(attempt / "result.json", self.results()["q1"])
        for name in ("provider.json", "checkpoint.json", "invocations.json"):
            write_json(attempt / name, {})
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in attempt.iterdir()}
        write_json(attempt / "receipt.json", {"identity": {"campaign_id": "campaign", "question_id": "q1"}, "files": files})
        write_json(directory / "report.json", {
            "campaign_id": "campaign", "binding": {"source_semantic_sha256": self.identity["semantic_snapshot_sha256"],
                                                        "runtime_questions": [self.runtime[0]]},
            "prepared_questions": [{**self.runtime[0], "database": str(database)}],
            "rows": [{"id": "q1", "phase": "first", "database": str(database), "result_path": str(attempt / "result.json")}]})
        return directory, attempt

    def test_campaign_scores_receipt_bound_partial_results(self):
        directory, _ = self.make_campaign()
        evaluated = evaluate_campaign(self.manifest, self.identity, directory)
        self.assertEqual(evaluated["result_receipts_verified"], 1)
        self.assertEqual(evaluated["summary"]["verified"]["matched_fact_slots"], 4)
        self.assertFalse(evaluated["full_suite"])

    def test_campaign_rejects_changed_result_or_missing_receipt(self):
        directory, attempt = self.make_campaign()
        result = json.loads((attempt / "result.json").read_text())
        result["evidence"] = []
        write_json(attempt / "result.json", result)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            evaluate_campaign(self.manifest, self.identity, directory)
        (attempt / "receipt.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_campaign(self.manifest, self.identity, directory)


if __name__ == "__main__":
    unittest.main()
