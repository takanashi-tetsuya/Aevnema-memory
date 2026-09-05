from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import unittest

from benchmarks.support.network_scorer import score_report


class Stage4NetworkScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            Path("validation/stage4-network-evidence-manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def _completed_result(self, source_keys: list[str]) -> dict:
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        answer_terms = list(criterion["required_terms"])
        answer_terms.extend(group[0] for group in criterion["required_term_groups"])
        return {
            "episode_ids": [group[0] for group in criterion["required_episode_groups"]],
            "evidence_episodes": [{"source_key": key} for key in source_keys],
            "answer": " ".join([*answer_terms, *source_keys]),
            "answer_audits": [{"valid": True}],
            "association_ids": [],
            "new_association_ids": [],
            "reinforced_association_ids": [],
        }

    def test_complete_vector_result_passes_all_network_groups(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": self._completed_result(criterion["required_sources"]),
                    }
                ]
            },
        }

        scored = score_report(report, self.manifest)

        row = scored["modes"]["vector_only"][0]
        self.assertTrue(scored["passed"])
        self.assertEqual(row["matched_group_count"], 7)
        self.assertEqual(row["required_group_count"], 7)

    def test_out_of_corpus_evidence_source_fails(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": self._completed_result(
                            [*criterion["required_sources"], "outside/story.json"]
                        ),
                    }
                ]
            },
        }

        scored = score_report(report, self.manifest)

        row = scored["modes"]["vector_only"][0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["unexpected_evidence_sources"], ["outside/story.json"])

    def test_equivalent_evidence_source_citation_does_not_require_every_name(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        result = self._completed_result(criterion["required_sources"])
        omitted = criterion["required_sources"][-1]
        result["answer"] = result["answer"].replace(omitted, "")
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": result,
                    }
                ]
            },
        }

        row = score_report(report, self.manifest)["modes"]["vector_only"][0]

        self.assertTrue(row["passed"])
        self.assertFalse(row["required_source_names_in_answer"][omitted])
        self.assertTrue(row["traceable_source_citation"])

    def test_answer_source_citation_is_a_diagnostic_when_evidence_is_structured(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        result = self._completed_result(criterion["required_sources"])
        for source in criterion["required_sources"]:
            result["answer"] = result["answer"].replace(source, "")
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": result,
                    }
                ]
            },
        }

        row = score_report(report, self.manifest)["modes"]["vector_only"][0]

        self.assertTrue(row["passed"])
        self.assertFalse(row["traceable_source_citation"])

    def test_episode_id_group_coverage_is_diagnostic_not_a_hard_gate(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        result = self._completed_result(criterion["required_sources"])
        result["episode_ids"] = result["episode_ids"][:-1]
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": result,
                    }
                ]
            },
        }

        row = score_report(report, self.manifest)["modes"]["vector_only"][0]

        self.assertTrue(row["passed"])
        self.assertFalse(row["episode_group_closure"])

    def test_expected_source_and_label_closure_are_diagnostics(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        result = self._completed_result(criterion["required_sources"])
        result["evidence_episodes"] = result["evidence_episodes"][:-1]
        for group in criterion["required_term_groups"]:
            result["answer"] = result["answer"].replace(group[0], "")
        report = {
            "mode_order": ["vector_only"],
            "modes": {
                "vector_only": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": result,
                    }
                ]
            },
        }

        row = score_report(report, self.manifest)["modes"]["vector_only"][0]

        self.assertTrue(row["passed"])
        self.assertFalse(row["required_source_closure"])
        self.assertFalse(row["term_group_closure"])

    def test_graph_growing_can_pass_with_safe_zero_write(self):
        criterion = self.manifest["questions"]["eden_obligation_to_new_institution"]
        result = self._completed_result(criterion["required_sources"])
        result["association_ids"] = [42]
        report = {
            "mode_order": ["graph_growing"],
            "modes": {
                "graph_growing": [
                    {
                        "id": "eden_obligation_to_new_institution",
                        "status": "completed",
                        "result": result,
                    }
                ]
            },
        }

        scored = score_report(report, self.manifest)

        row = scored["modes"]["graph_growing"][0]
        self.assertTrue(row["passed"])
        self.assertNotIn("growth_target", row)
        self.assertNotIn("growth_target_met", row)

    def test_manifest_anchors_exist_in_the_frozen_corpus(self):
        question_ids = {
            item["id"]
            for item in json.loads(
                Path("validation/evaluation-questions-stage4-network.json").read_text(
                    encoding="utf-8"
                )
            )
        }
        self.assertEqual(question_ids, set(self.manifest["questions"]))
        allowed_sources = set(self.manifest["corpus_boundary"]["allowed_source_keys"])
        connection = sqlite3.connect(self.manifest["database"])
        try:
            source_by_episode = {
                int(row[0]): str(row[1])
                for row in connection.execute("SELECT id, source_key FROM episode")
            }
        finally:
            connection.close()
        for question_id, criterion in self.manifest["questions"].items():
            self.assertGreaterEqual(
                len(criterion["required_episode_groups"]),
                6,
                question_id,
            )
            for source_key in criterion["required_sources"]:
                self.assertIn(source_key, allowed_sources, question_id)
            for group in criterion["required_episode_groups"]:
                self.assertTrue(group, question_id)
                for episode_id in group:
                    self.assertIn(episode_id, source_by_episode, (question_id, episode_id))
                    self.assertIn(
                        source_by_episode[episode_id], allowed_sources, (question_id, episode_id)
                    )


if __name__ == "__main__":
    unittest.main()
