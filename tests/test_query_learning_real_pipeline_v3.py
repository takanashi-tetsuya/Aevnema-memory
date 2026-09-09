from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig
from memory_demo.llm.prompts import (
    ANSWER_AUDIT_SYSTEM,
    ANSWER_SYSTEM,
    EVIDENCE_RERANK_SYSTEM,
    QUERY_SYSTEM,
)
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.engine import QueryEngine


class _LocalFixtureModel:
    """Explicit local provider responses; no ModelClient or network exists."""

    requirements = (
        "Where is the recorded signal?",
        "What color is the recorded signal?",
    )
    source_answers = ("The signal is in the tower.", "The recorded signal is blue.")
    answer = "The signal is in the tower. The recorded signal is blue."

    def __init__(self, *, provide_coverage: bool) -> None:
        self.provide_coverage = provide_coverage
        self.calls: list[str] = []
        self.embedding_batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append("embedding")
        self.embedding_batches.append(list(texts))
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def chat_json(self, system: str, prompt: str, **_kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            return {"search_queries": list(self.requirements)}
        if system == EVIDENCE_RERANK_SYSTEM:
            self.calls.append("rerank")
            return {
                "selected_episode_ids": [1, 2],
                "coverage": (
                    [
                        {"query": requirement, "episode_ids": [index]}
                        for index, requirement in enumerate(self.requirements, start=1)
                    ]
                    if self.provide_coverage
                    else []
                ),
            }
        if system == ANSWER_AUDIT_SYSTEM:
            self.calls.append("answer_audit")
            return {
                "reviews": [
                    {"claim": self.answer, "verdict": "supported_fact"}
                ]
            }
        raise AssertionError("Unexpected provider purpose in isolated local fixture")

    def chat_text(self, system: str, prompt: str, **_kwargs) -> str:
        if system != ANSWER_SYSTEM:
            raise AssertionError("Unexpected text provider purpose")
        self.calls.append("answer")
        return self.answer


class _ObservedQueryEngine(QueryEngine):
    """Observe the real capture before public-query cleanup, without changing it."""

    def _finalize_v3_query_learning(self, *args, **kwargs):
        self.observed_learning_capture = self._v3_learning_capture
        return super()._finalize_v3_query_learning(*args, **kwargs)


class QueryLearningRealPipelineV3Tests(unittest.TestCase):
    """Real query, requirements, vectors, selector and answer guard.

    The successful case uses two independent direct-source mappings, observed
    rerank work and an audited answer. No query method or learning capture is
    patched, and the application finalizer and RAM publication are real.
    """

    def _fixture(self, directory: str, *, provide_coverage: bool, invalid_source: bool = False):
        config = AppConfig(
            database_path=Path(directory) / "memory.db",
            log_dir=Path(directory) / "logs",
            model=ModelConfig(
                embedding_model="t13-real-local-fixture", embedding_dimension=3
            ),
        )
        config.retrieval.contextual_association_enabled = True
        config.retrieval.contextual_association_shadow = False
        config.retrieval.sparse_enabled = False
        config.retrieval.source_key_cohort_enabled = False
        config.retrieval.graph_max_hops = 0
        config.retrieval.growth_max_rounds = 0
        config.retrieval.followup_planning_mode = "off"
        config.retrieval.rerank_review_mode = "lean"
        config.retrieval.rerank_coverage_audit_enabled = False
        config.retrieval.rerank_audit_enabled = False
        app = MemoryApplication(config)
        marker = "2026-09-06T00:00:00+00:00"
        with app.db.transaction() as connection:
            for index, answer in enumerate(_LocalFixtureModel.source_answers, start=1):
                raw_text = f"[record: signal-{index}]\n{answer}"
                source_id = connection.execute(
                    "INSERT INTO source(raw_text) VALUES(?)", (raw_text,)
                ).lastrowid
                quote = "This quote is absent from the source." if invalid_source and index == 2 else raw_text
                connection.execute(
                    """
                    INSERT INTO episode(
                        source_id, source_key, segment_index, text,
                        evidence_origin, epistemic_status, generation,
                        evidence_quotes_json, evidence_spans_json, evidence_basis,
                        embedding, created_at, updated_at
                    ) VALUES(?, ?, 0, ?, 'source', 'observed', 0, ?, ?,
                             'literal_source_span', ?, ?, ?)
                    """,
                    (
                        source_id, f"local-fixture/signal-{index}.json", answer,
                        json.dumps([quote]), json.dumps([[1, 2]]),
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                        marker, marker,
                    ),
                )
        app.rebuild_indexes()
        model = _LocalFixtureModel(provide_coverage=provide_coverage)
        engine = _ObservedQueryEngine(
            config, model, app.episode_index, app.concept_index,
            app.episodes, app.concepts, app.sources, app.associations,
            contextual_matcher=ContextualAssociationMatcher(
                app.context_cue_index, app.need_cue_index, app.associations
            ),
            contextual_learning_finalizer=app.finalize_recall_event,
        )
        return app, model, engine

    def test_real_pipeline_creates_ready_receipt_and_rejects_incomplete_source_support(self):
        scenarios = ((True, False), (False, False), (True, True))
        for coverage, invalid_source in scenarios:
            with self.subTest(coverage=coverage, invalid_source=invalid_source), tempfile.TemporaryDirectory() as directory:
                app, model, engine = self._fixture(directory, provide_coverage=coverage, invalid_source=invalid_source)
                result = engine.query(
                    "Describe the signal in the source record.",
                    contextual_learning=True,
                    learning_request_id="real-local-pipeline-request",
                    contextual_domain="knowledge",
                )
                self.assertEqual(
                    ["requirements_planner", "embedding", "rerank", "answer", "answer_audit"],
                    model.calls,
                )
                self.assertEqual(1, len(model.embedding_batches))
                self.assertEqual(3, result["query_vector_bundle"]["logical_binding_count"])
                self.assertEqual(_LocalFixtureModel.answer, result["answer"])
                self.assertIs(True, result["answer_audits"][-1]["valid"])
                self.assertIs(False, result["answer_generation_skipped"])
                selector = result["evidence_slot_trace"]["slot_selector_v3"]
                self.assertEqual("resolved", selector["requirements_status"])
                self.assertEqual("contextual_double_key_contribution_selector_v3", result["contextual_association"]["backend"])
                self.assertEqual([1, 2], selector["base_endpoint_manifest"])
                self.assertEqual(
                    [
                        {"episode_id": 1, "status": "source_bound"},
                        {"episode_id": 2, "status": "source_evidence_quote_span_mismatch" if invalid_source else "source_bound"},
                    ],
                    selector["source_provenance_status"],
                )
                capture = engine.observed_learning_capture
                self.assertIsNotNone(capture)
                self.assertEqual((1, 2), capture.independent_base_episode_ids)
                self.assertTrue(capture.contributions)
                if coverage and not invalid_source:
                    self.assertEqual([1, 2], sorted(result["episode_ids"]))
                    self.assertEqual("no_unresolved_slots", selector["reason"])
                    self.assertFalse(capture.target_gate_completed)
                    self.assertTrue(capture.direct_base_validation_completed)
                    self.assertFalse(capture.missing_required_clauses)
                    learning = result["contextual_learning"]
                    self.assertEqual("ready", learning["status"])
                    self.assertEqual(1, len(learning["receipt_ids"]))
                    receipt = app.associations.get_contextual_creation_receipt(learning["receipt_ids"][0])
                    self.assertEqual("ready", receipt["status"])
                    with app.db.connection() as connection:
                        durable_receipt = connection.execute(
                            "SELECT * FROM contextual_creation_receipt WHERE id = ?",
                            (receipt["receipt_id"],),
                        ).fetchone()
                    self.assertEqual("source_bound", durable_receipt["verification_status"])
                    self.assertTrue(json.loads(durable_receipt["source_facts_json"]))
                    edge = app.associations.get(receipt["association_id"])
                    self.assertEqual({1, 2}, {edge["from_id"], edge["to_id"]})
                    self.assertEqual(1, app.context_cue_index.count)
                    self.assertEqual(1, app.need_cue_index.count)
                    expected_counts = (1, 2, 1)
                else:
                    self.assertTrue(capture.target_gate_completed)
                    self.assertTrue(capture.missing_required_clauses)
                    self.assertEqual("skipped", result["contextual_learning"]["status"])
                    self.assertEqual("selector_delivery_incomplete", result["contextual_learning"]["reason"])
                    expected_counts = (0, 0, 0)
                with app.db.connection() as connection:
                    for table, expected_count in zip((
                        "association", "association_cue_prototype",
                        "contextual_creation_receipt",
                    ), expected_counts):
                        self.assertEqual(expected_count, connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def test_public_evidence_only_uses_normal_retrieval_without_answer_or_edge_use_write(self):
        """Evidence-only stops after selected Source materialization, not at a mock."""

        with tempfile.TemporaryDirectory() as directory:
            app, model, engine = self._fixture(directory, provide_coverage=True)
            with patch.object(engine.associations, "mark_used", wraps=engine.associations.mark_used) as mark_used:
                result = engine.query(
                    "Describe the signal in the source record.",
                    contextual_domain="knowledge",
                    stop_after="evidence",
                )

            self.assertEqual(
                ["requirements_planner", "embedding", "rerank"], model.calls
            )
            self.assertEqual("live_evidence", result["execution_profile"])
            self.assertTrue(result["answer_generation_skipped"])
            self.assertFalse(result["association_usage_recorded"])
            self.assertEqual("", result["answer"])
            self.assertEqual([], result["answer_audits"])
            mark_used.assert_not_called()
            evidence = result["evidence_result"]
            self.assertEqual("complete", evidence["evidence_state"])
            self.assertEqual(
                "no_unresolved_slots", evidence["participation"]["entry_reason"]
            )
            self.assertFalse(evidence["learning_eligible"])
            self.assertFalse(evidence["delivery_states"]["answer_input_sent"])
            self.assertTrue(evidence["delivery_states"]["evidence_materialized"])
            self.assertEqual(2, len(evidence["materialized_source_refs"]))
            self.assertTrue(
                all(
                    item["source_evidence_delivery"] == "source_bound"
                    for item in evidence["materialized_source_refs"]
                )
            )
            excerpts = [
                item["source_excerpt"]
                for item in evidence["materialized_source_refs"]
            ]
            self.assertTrue(any("The signal is in the tower." in item for item in excerpts))
            self.assertTrue(any("The recorded signal is blue." in item for item in excerpts))
            with app.db.connection() as connection:
                self.assertEqual(
                    0,
                    connection.execute("SELECT COUNT(*) FROM association").fetchone()[0],
                )
                self.assertEqual(
                    0,
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_creation_receipt"
                    ).fetchone()[0],
                )

    def test_public_frozen_input_restores_requirement_vectors_without_a_second_model_call(self):
        """A frozen plan must retain slots, not merely opaque vector metadata."""

        with tempfile.TemporaryDirectory() as directory:
            _app, model, engine = self._fixture(directory, provide_coverage=True)
            question = "Describe the signal in the source record."
            engine.config.retrieval.rerank_atomic_query_limit = 1
            plan, prepared = engine.build_frozen_query_input(question)
            restored = engine.restore_frozen_query_vectors(question, plan)
            self.assertEqual(prepared.metadata(), restored.metadata())
            self.assertGreater(restored.logical_count, 0)
            self.assertTrue(any(item.slot_id for item in restored.queries))
            preparation_calls = list(model.calls)
            result = engine.query(
                question,
                frozen_plan=plan,
                query_vector_bundle=restored,
                strict_vector_bundle=True,
                contextual_domain="knowledge",
                stop_after="evidence",
            )
            self.assertEqual(preparation_calls, model.calls)
            self.assertTrue(result["query_plan_frozen"])
            self.assertEqual("input_frozen", result["execution_profile"])
            self.assertEqual("complete", result["evidence_result"]["evidence_state"])



if __name__ == "__main__":
    unittest.main()
