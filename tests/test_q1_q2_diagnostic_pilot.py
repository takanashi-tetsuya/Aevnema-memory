from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.run_q1_q2_diagnostic_pilot import (
    PilotCase,
    FinalizerObserver,
    ProviderLedger,
    Q2_ARM_NAMES,
    _clone_sqlite,
    _mask_engine_edge,
    _pilot_config,
    _pilot_scope_hash,
    _terminal_q2_arms,
    _write_json,
    run_diagnostic_pilot,
)
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


QUESTION = "Which organization did the witness say she supported?"
ANSWER = "The witness said she supported the Archive organization."
RICH_QUESTION = "圣园未花说自己一直在暗中支援哪个组织？"
RICH_ANSWER = "圣园未花说自己一直在暗中支援阿里乌斯。"
RICH_SCOPE_HASH = "test-scope:sha256:" + sha256(
    b"rich-requirement-v17-scope"
).hexdigest()


class _LocalPilotModel:
    """Local deterministic model: no provider client or network exists."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append("embedding")
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def chat_json(self, system: str, _prompt: str, **_kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            # The sole required slot is the whole raw question, the narrow
            # shape V17 may independently accept.  The test does not inject a
            # need, answer cache, association, receipt, or gold artifact.
            return {"search_queries": [QUESTION]}
        if system == EVIDENCE_RERANK_SYSTEM:
            self.calls.append("rerank")
            return {
                "selected_episode_ids": [1, 2],
                "coverage": [{"query": QUESTION, "episode_ids": [1, 2]}],
            }
        if system == ANSWER_AUDIT_SYSTEM:
            self.calls.append("answer_audit")
            return {"reviews": [{"claim": ANSWER, "verdict": "supported_fact"}]}
        raise AssertionError(f"unexpected local JSON purpose: {system}")

    def chat_text(self, system: str, _prompt: str, **_kwargs) -> str:
        if system != ANSWER_SYSTEM:
            raise AssertionError(f"unexpected local text purpose: {system}")
        self.calls.append("answer")
        return ANSWER


class _RerankTimeoutModel(_LocalPilotModel):
    """Local failure injection: no provider or network is involved."""

    def chat_json(self, system: str, prompt: str, **kwargs) -> dict:
        if system == EVIDENCE_RERANK_SYSTEM:
            self.calls.append("rerank_timeout")
            raise TimeoutError("local injected rerank timeout")
        return super().chat_json(system, prompt, **kwargs)


class _ParaphraseFirstPlannerModel(_LocalPilotModel):
    """Exercises the singleton raw-question requirement alignment."""

    def chat_json(self, system: str, prompt: str, **kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            return {"search_queries": ["Did the witness support an organization?"]}
        return super().chat_json(system, prompt, **kwargs)


class _RichRequirementPilotModel(_LocalPilotModel):
    """Exercise a live Q1 with the rich requirement shape from the pilot."""

    def chat_json(self, system: str, prompt: str, **kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            return {
                "language": "zh",
                "target_entities": ["圣园未花"],
                "search_queries": [RICH_QUESTION],
                "requested_relation": "暗中支援的组织",
                "temporal_constraint": "一直",
                "answer_shape": "单一实体",
                "uncertainty_required": False,
            }
        if system == EVIDENCE_RERANK_SYSTEM:
            self.calls.append("rerank")
            return {
                "selected_episode_ids": [1, 2],
                "coverage": [{"query": RICH_QUESTION, "episode_ids": [1, 2]}],
            }
        if system == ANSWER_AUDIT_SYSTEM:
            self.calls.append("answer_audit")
            return {"reviews": [{"claim": RICH_ANSWER, "verdict": "supported_fact"}]}
        return super().chat_json(system, prompt, **kwargs)

    def chat_text(self, system: str, _prompt: str, **_kwargs) -> str:
        if system != ANSWER_SYSTEM:
            raise AssertionError(f"unexpected local text purpose: {system}")
        self.calls.append("answer")
        return RICH_ANSWER


class _RichRequirementParaphraseModel(_RichRequirementPilotModel):
    """Preserve the rich slot while adding ordinary Q1 planner paraphrases."""

    def chat_json(self, system: str, prompt: str, **kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            return {
                "language": "zh",
                "target_entities": ["圣园未花"],
                "search_queries": [
                    "圣园未花曾经支援的是哪个组织？",
                    "圣园未花说自己一直在暗中支援哪个组织？",
                ],
                "requested_relation": "暗中支援的组织",
                "temporal_constraint": "一直",
                "answer_shape": "单一实体",
                "uncertainty_required": False,
            }
        return super().chat_json(system, prompt, **kwargs)


class _RichRequirementManyPlannerCuesModel(_RichRequirementPilotModel):
    """Model the actual rich slot plus several Q1-only planner retrieval cues."""

    def chat_json(self, system: str, prompt: str, **kwargs) -> dict:
        if system == QUERY_SYSTEM:
            self.calls.append("requirements_planner")
            return {
                "language": "zh",
                "target_entities": ["圣园未花"],
                # None is the whole raw question.  They are ordinary Q1
                # retrieval alternatives, not future Q2 needs.
                "search_queries": [
                    "圣园未花支援哪个组织？",
                    "圣园未花暗中帮助的对象是谁？",
                    "圣园未花一直支持哪个团体？",
                    "阿里乌斯是否获得圣园未花支援？",
                    "圣园未花的秘密支援对象",
                    "寻找圣园未花与组织支援的剧情",
                    "圣园未花的长期暗中支援关系",
                    "支援组织的身份是什么？",
                    "圣园未花说的那个组织",
                    "暗中支援组织的证据",
                ],
                "requested_relation": "暗中支援的组织",
                "temporal_constraint": "一直",
                "answer_shape": "单一实体",
                "uncertainty_required": False,
            }
        return super().chat_json(system, prompt, **kwargs)


def _config(directory: str) -> AppConfig:
    config = AppConfig(
        database_path=Path(directory) / "pilot.sqlite",
        log_dir=Path(directory) / "logs",
        model=ModelConfig(embedding_model="local-q1-q2", embedding_dimension=3),
    )
    config.retrieval.contextual_association_enabled = True
    config.retrieval.contextual_association_shadow = False
    config.retrieval.growth_max_rounds = 0
    config.retrieval.growth_persist_only_used = False
    config.retrieval.association_cue_enabled = False
    config.retrieval.association_cue_fast_path_enabled = False
    config.retrieval.sparse_enabled = False
    config.retrieval.source_key_cohort_enabled = False
    config.retrieval.graph_max_hops = 0
    config.retrieval.followup_planning_mode = "off"
    config.retrieval.rerank_review_mode = "lean"
    config.retrieval.rerank_coverage_audit_enabled = False
    config.retrieval.rerank_audit_enabled = False
    config.retrieval.answer_episode_limit = 2
    return config


def _engine(app: MemoryApplication, config: AppConfig, model: _LocalPilotModel) -> QueryEngine:
    return QueryEngine(
        config,
        model,
        app.episode_index,
        app.concept_index,
        app.episodes,
        app.concepts,
        app.sources,
        app.associations,
        contextual_matcher=ContextualAssociationMatcher(
            app.context_cue_index,
            app.need_cue_index,
            app.associations,
        ),
        contextual_learning_finalizer=app.finalize_recall_event,
    )


class Q1Q2DiagnosticPilotTests(unittest.TestCase):
    def test_terminal_trace_writer_accepts_mixed_runtime_map_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_local.json"
            _write_json(
                path,
                {"runtime_binding": {32: "episode", "sha256:scope": "scope"}},
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("episode", payload["runtime_binding"]["32"])
        self.assertEqual("scope", payload["runtime_binding"]["sha256:scope"])

    def test_pilot_config_preserves_discovery_and_disables_observed_redundant_compressor(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("\n", encoding="utf-8")
            config = _pilot_config(
                env_file,
                Path(directory) / "diagnostic.sqlite",
                Path(directory) / "logs",
            )

        self.assertEqual("missing_slots", config.retrieval.followup_planning_mode)
        self.assertEqual(1, config.retrieval.rerank_atomic_query_limit)
        self.assertEqual(2, config.retrieval.answer_episode_limit)
        self.assertFalse(config.retrieval.rerank_audit_enabled)
        self.assertEqual(25.0, config.model.timeout_seconds)
        self.assertEqual(0, config.model.max_retries)

    def test_provider_ledger_accepts_model_client_succeeded_status(self):
        ledger = ProviderLedger()
        for status in ("succeeded", "success", "timeout"):
            ledger.provider_call_finished({"status": status, "sent": True})
        self.assertEqual(
            {
                "logical_batches": 0,
                "http_attempts": 3,
                "sent": 3,
                "succeeded": 2,
                "failed_or_rejected": 1,
                "fallbacks": 0,
            },
            ledger.export()["counts"],
        )

    def test_q1_failure_branch_gives_all_dependent_arms_terminal_state(self):
        arms = _terminal_q2_arms("Q1 failed before public finalization")

        self.assertEqual(
            {
                "before_commit_probe",
                "post_send_immediate_q2",
                "learning_ready_q2",
                "edge_masked_q2",
                "ordinary_embedding_cache_q2",
                "edge_restart_q2",
                "source_changed_q2",
            },
            set(arms),
        )
        self.assertTrue(arms["edge_masked_q2"]["edge_mask"]["enabled"])
        self.assertFalse(arms["ordinary_embedding_cache_q2"]["edge_mask"]["enabled"])
        self.assertEqual("not_run", arms["learning_ready_q2"]["summary"]["run_status"])

    def test_in_progress_full_local_checkpoint_precedes_q1_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_database = root / "source.sqlite"
            source_app = MemoryApplication(
                AppConfig(
                    database_path=source_database,
                    log_dir=root / "source-logs",
                    model=ModelConfig(embedding_dimension=3),
                )
            )
            manifest = root / "case.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_source_slice.v1",
                        "formal_scoring_eligible": False,
                        "promotion_prohibited": True,
                        "pilot_input": {
                            "q1_text": QUESTION,
                            "q2_text": QUESTION,
                            "contextual_domain": "knowledge",
                            "contextual_revisit_scope": "checkpoint-test",
                        },
                    }
                ),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text("LOCAL_TEST=1\n", encoding="utf-8")
            output = root / "case-output"

            def open_local(_env, database, log_dir):
                return (
                    MemoryApplication(
                        AppConfig(
                            database_path=database,
                            log_dir=log_dir,
                            model=ModelConfig(embedding_dimension=3),
                        )
                    ),
                    AppConfig(
                        database_path=database,
                        log_dir=log_dir,
                        model=ModelConfig(embedding_dimension=3),
                    ),
                )

            def q1_fails_after_observing_checkpoint(*_args, **_kwargs):
                checkpoint = json.loads(
                    (output / "q1_q2_case.full_local.json").read_text(encoding="utf-8")
                )
                self.assertEqual("in_progress_q1", checkpoint["status"])
                self.assertEqual(
                    "not_observed_q1_in_progress", checkpoint["q1"]["record"]["status"]
                )
                return {"status": "failed", "provider": {"counts": {}}}

            with patch(
                "benchmarks.run_q1_q2_diagnostic_pilot._open_app", side_effect=open_local
            ), patch(
                "benchmarks.run_q1_q2_diagnostic_pilot._call_query",
                side_effect=q1_fails_after_observing_checkpoint,
            ):
                result = run_diagnostic_pilot(
                    source_database=source_database,
                    case_manifest=manifest,
                    output_dir=output,
                    env_file=env_file,
                )

            self.assertFalse(result["formal_score"])
            terminal = json.loads(
                (output / "q1_q2_case.full_local.json").read_text(encoding="utf-8")
            )
            self.assertEqual("diagnostic_complete", terminal["status"])

    def test_full_pilot_writes_runtime_receipt_and_all_zero_model_arms(self):
        """The terminal package survives a runtime manifest's mixed map keys.

        This is entirely local.  It exercises the public Q1 finalizer and
        the runner's public zero-model Q2 arms, then serializes the same
        receipt/manifest shape that previously left an in-progress package.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_app, _source_config = self._app(
                str(root / "source"),
                episode_texts=(
                    "圣园未花提到她与阿里乌斯结盟，以打倒共同敌人。",
                    "圣园未花说自己一直在暗中支援阿里乌斯。",
                ),
            )
            source_snapshot = root / "source-static.sqlite"
            _clone_sqlite(source_app.config.database_path, source_snapshot)
            manifest = root / "case.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_source_slice.v1",
                        "formal_scoring_eligible": False,
                        "promotion_prohibited": True,
                        "pilot_input": {
                            "q1_text": RICH_QUESTION,
                            "q2_text": RICH_QUESTION,
                            "contextual_domain": "knowledge",
                            "contextual_revisit_scope": "runtime-terminal-test",
                        },
                    }
                ),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text("LOCAL_TEST=1\n", encoding="utf-8")
            output = root / "case-output"

            def open_local(_env: Path, database: Path, log_dir: Path):
                config = _config(str(root / "local-config"))
                config.database_path = database
                config.log_dir = log_dir
                config.retrieval.rerank_atomic_query_limit = 1
                config.retrieval.answer_episode_limit = 2
                config.retrieval.followup_planning_mode = "missing_slots"
                app = MemoryApplication(config)
                app.rebuild_indexes()
                app.rebuild_contextual_indexes()
                model = _RichRequirementManyPlannerCuesModel()
                app.query_engine = lambda config=None: _engine(app, config or app.config, model)  # type: ignore[method-assign]
                return app, config

            with patch(
                "benchmarks.run_q1_q2_diagnostic_pilot._open_app", side_effect=open_local
            ):
                result = run_diagnostic_pilot(
                    source_database=source_snapshot,
                    case_manifest=manifest,
                    output_dir=output,
                    env_file=env_file,
                    deadline_seconds=3.0,
                )

            self.assertFalse(result["formal_score"])
            terminal = json.loads(
                (output / "q1_q2_case.full_local.json").read_text(encoding="utf-8")
            )
            self.assertEqual("diagnostic_complete", terminal["status"])
            self.assertEqual("ready", terminal["q1"]["learning_status"])
            self.assertEqual(
                "completed",
                terminal["arms"]["learning_ready_q2"]["summary"]["run_status"],
            )
            self.assertEqual(
                0,
                terminal["arms"]["learning_ready_q2"]["summary"]["provider_counts"]["http_attempts"],
            )
            self.assertEqual(
                "exact_revisit_miss",
                terminal["arms"]["before_commit_probe"]["summary"]["run_status"],
            )
            self.assertEqual(
                "exact_revisit_miss",
                terminal["arms"]["edge_masked_q2"]["summary"]["run_status"],
            )
            self.assertEqual(
                "completed",
                terminal["arms"]["edge_restart_q2"]["summary"]["run_status"],
            )
            self.assertEqual(
                "exact_revisit_miss",
                terminal["arms"]["source_changed_q2"]["summary"]["run_status"],
            )
            for name, arm in terminal["arms"].items():
                if arm["summary"]["run_status"] == "not_run":
                    continue
                self.assertEqual(0, arm["summary"]["provider_counts"]["http_attempts"], name)
            for name in Q2_ARM_NAMES:
                checkpoint = json.loads(
                    (output / f"q2_{name}.full_local.json").read_text(encoding="utf-8")
                )
                expected_status = (
                    "q2_arm_not_run"
                    if name == "ordinary_embedding_cache_q2"
                    else "q2_arm_terminal"
                )
                self.assertEqual(expected_status, checkpoint["status"], name)
            source_changed_checkpoint = json.loads(
                (output / "q2_source_changed_q2.full_local.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "append_nonsemantic_diagnostic_revision_marker_to_pilot_clone",
                source_changed_checkpoint["source_mutation"]["mutation"],
            )

            # A fault after the public Q1 finalizer must retain observed Q1
            # state without inventing dependent Q2 results.
            interrupted_output = root / "post-q1-fault"
            with patch(
                "benchmarks.run_q1_q2_diagnostic_pilot._open_app", side_effect=open_local
            ), patch(
                "benchmarks.run_q1_q2_diagnostic_pilot._q2_arm",
                side_effect=RuntimeError("local injected post-Q1 arm fault"),
            ), self.assertRaisesRegex(RuntimeError, "post-Q1 arm fault"):
                run_diagnostic_pilot(
                    source_database=source_snapshot,
                    case_manifest=manifest,
                    output_dir=interrupted_output,
                    env_file=env_file,
                    deadline_seconds=3.0,
                )
            interrupted = json.loads(
                (interrupted_output / "q1_q2_case.full_local.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("q1_terminal_q2_pending", interrupted["status"])
            self.assertEqual("completed", interrupted["q1"]["record"]["status"])
            self.assertEqual("ready", interrupted["q1"]["learning_status"])
            self.assertEqual(
                "not_run",
                interrupted["arms"]["before_commit_probe"]["summary"]["run_status"],
            )
            pending_arm = json.loads(
                (interrupted_output / "q2_before_commit_probe.full_local.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("q2_arm_pending_public_preflight", pending_arm["status"])

    def test_sqlite_clone_uses_udf_independent_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite"
            destination = Path(directory) / "clone.sqlite"
            app = MemoryApplication(
                AppConfig(
                    database_path=source,
                    log_dir=Path(directory) / "logs",
                    model=ModelConfig(embedding_dimension=3),
                )
            )
            _clone_sqlite(source, destination)
            self.assertTrue(destination.is_file())
            self.assertFalse(destination.with_name(destination.name + "-wal").exists())
            self.assertFalse(destination.with_name(destination.name + "-shm").exists())
            connection = sqlite3.connect(destination)
            try:
                self.assertEqual(0, connection.execute("SELECT COUNT(*) FROM source").fetchone()[0])
            finally:
                connection.close()

    def _app(
        self,
        directory: str,
        *,
        episode_texts: tuple[str, str] | None = None,
    ) -> tuple[MemoryApplication, AppConfig]:
        config = _config(directory)
        app = MemoryApplication(config)
        texts = episode_texts or (
            "The witness stated that she supports the Archive organization.",
            "A second source record independently says the Archive organization received the witness's support.",
        )
        with app.db.transaction() as connection:
            for episode_id, text in enumerate(texts, start=1):
                source_text = f"[record: pilot-{episode_id}]\n{text}"
                source_id = connection.execute(
                    "INSERT INTO source(raw_text) VALUES(?)", (source_text,)
                ).lastrowid
                connection.execute(
                    """
                    INSERT INTO episode(
                        source_id, source_key, segment_index, text,
                        evidence_origin, epistemic_status, generation,
                        evidence_quotes_json, evidence_spans_json, evidence_basis,
                        embedding, created_at, updated_at
                    ) VALUES (?, ?, 0, ?, 'source', 'observed', 0, ?, ?,
                              'literal_source_span', ?, '2026-09-07T00:00:00+00:00',
                              '2026-09-07T00:00:00+00:00')
                    """,
                    (
                        source_id,
                        f"pilot/source-{episode_id}.json",
                        text,
                        json.dumps([source_text]),
                        json.dumps([[1, 2]]),
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
                    ),
                )
        app.rebuild_indexes()
        return app, config

    def _rich_q1_fixture(
        self, directory: str
    ) -> tuple[MemoryApplication, AppConfig, _RichRequirementPilotModel, int]:
        """Create one source-bound Q1 edge through the public finalizer."""

        app, config = self._app(
            directory,
            episode_texts=(
                "圣园未花提到她与阿里乌斯结盟，以打倒共同敌人。",
                "圣园未花说自己一直在暗中支援阿里乌斯。",
            ),
        )
        # Mirror the v10 pilot's one-atomic-query requirement budget. The
        # test keeps the rich fields on that one whole-question requirement;
        # it does not delete them or change the question to evade projection.
        config.retrieval.rerank_atomic_query_limit = 1
        model = _RichRequirementPilotModel()
        engine = _engine(app, config, model)
        q1 = engine.query(
            RICH_QUESTION,
            contextual_learning=True,
            learning_request_id="rich-requirement-q1",
            contextual_domain="knowledge",
            contextual_revisit_scope_hash=RICH_SCOPE_HASH,
        )
        slot = q1["authoritative_requirements"]["requirements"][0]
        self.assertEqual(["圣园未花"], slot["subject_terms"])
        self.assertEqual("暗中支援的组织", slot["relation_hint"])
        self.assertEqual("一直", slot["temporal_hint"])
        self.assertEqual("explicit_negation", slot["negation_hint"])
        learning = q1["contextual_learning"]
        self.assertEqual("ready", learning["status"])
        self.assertEqual(
            ["ready"],
            [item["status"] for item in learning["revisit_projections"]],
        )
        self.assertEqual(
            ["runtime_manifest_ready"],
            [item["reason"] for item in learning["revisit_projections"]],
        )
        receipt_id = int(learning["receipt_ids"][0])
        receipt = app.associations.get_contextual_creation_receipt(receipt_id)
        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual("ready", receipt["status"])
        self.assertIsNotNone(app.associations.load_contextual_revisit_contract(receipt_id))
        self.assertEqual(
            "ready",
            app.associations.load_contextual_revisit_runtime_manifest(receipt_id).state,
        )
        return app, config, model, int(receipt["association_id"])

    def _public_rich_preflight(
        self,
        engine: QueryEngine,
        model: _RichRequirementPilotModel,
    ) -> dict | None:
        before_calls = list(model.calls)
        with patch.object(
            engine,
            "_query_impl",
            side_effect=AssertionError("public preflight must not fall through"),
        ) as ordinary_query:
            result = engine.try_contextual_revisit(
                RICH_QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash=RICH_SCOPE_HASH,
            )
        ordinary_query.assert_not_called()
        self.assertEqual(before_calls, model.calls)
        return result

    def test_real_q1_public_finalizer_masks_cache_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(directory)
            q1_engine = _engine(app, config, _LocalPilotModel())
            original_finalizer = q1_engine.contextual_learning_finalizer
            assert callable(original_finalizer)
            observer = FinalizerObserver()
            q1_engine.contextual_learning_finalizer = observer.wrap(original_finalizer)
            q1_result = q1_engine.query(
                QUESTION,
                contextual_learning=True,
                learning_request_id="local-real-q1",
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-v3-q1-q2-pilot",
            )
            learning = q1_result["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(1, observer.finalizer_calls)
            observation = observer.export(started_at=0.0)
            self.assertFalse(observation["q1_pause_injected"])
            self.assertIsNotNone(observation["before_public_finalizer_at_ms"])
            self.assertIsNotNone(observation["public_finalizer_returned_at_ms"])
            self.assertEqual(1, len(learning["receipt_ids"]))
            receipt = app.associations.get_contextual_creation_receipt(learning["receipt_ids"][0])
            self.assertIsNotNone(receipt)
            assert receipt is not None
            self.assertEqual("ready", receipt["status"])
            edge_id = int(receipt["association_id"])
            self.assertEqual(1, app.associations.stats()["edges"])
            # A receipt/edge must be real and ready.  A V17 runtime manifest
            # is intentionally not forced by this fixture: if the strict
            # projection shape is not met, the diagnostic preserves that
            # absence and Q2 remains an ordinary live query.
            runtime_manifest = app.associations.load_contextual_revisit_runtime_manifest(
                int(receipt["receipt_id"])
            )
            if runtime_manifest is not None:
                self.assertEqual("ready", runtime_manifest.state)

            ready = _engine(app, config, _LocalPilotModel()).query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-v3-q1-q2-pilot",
            )
            self.assertFalse(ready.get("answer_generation_skipped"))

            # Hiding the one real Q1 edge affects both matcher and all engine
            # association lookups; direct base retrieval remains intact.
            masked_engine = _engine(app, config, _LocalPilotModel())
            _mask_engine_edge(masked_engine, edge_id)
            masked = masked_engine.query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-v3-q1-q2-pilot",
            )
            self.assertNotIn(
                edge_id,
                masked.get("contextual_association", {}).get("attached_edges", []),
            )
            self.assertTrue(masked["candidate_episode_ids"])

            # The ordinary-cache arm receives only an independently made raw
            # question vector and has the same edge masked.  It cannot pass a
            # persisted Q1 plan, Q2 need, answer, selected IDs, or gold.
            cache_vector = _engine(app, config, _LocalPilotModel()).embed_query_text(QUESTION)
            cache_engine = _engine(app, config, _LocalPilotModel())
            _mask_engine_edge(cache_engine, edge_id)
            cached = cache_engine.query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-v3-q1-q2-pilot",
                query_embeddings_override={QUESTION: cache_vector},
                strict_vector_bundle=False,
            )
            self.assertGreaterEqual(cached["query_embedding_cache"]["hit_count"], 1)
            self.assertNotIn(
                edge_id,
                cached.get("contextual_association", {}).get("attached_edges", []),
            )

            # Fresh application/index objects prove the Q1 receipt survives a
            # restart.  We do not force a V17 hit: a runtime miss is a valid
            # observed result, but it must be a normal, runnable Q2.
            restarted = MemoryApplication(deepcopy(config))
            restarted.rebuild_indexes()
            restarted.rebuild_contextual_indexes()
            restart_result = _engine(restarted, restarted.config, _LocalPilotModel()).query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-v3-q1-q2-pilot",
            )
            self.assertFalse(restart_result.get("answer_generation_skipped"))

    def test_rich_requirement_projects_to_public_exact_revisit_and_fails_closed(self):
        """No field deletion, manual manifest, or internal matcher call is used."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, config, model, edge_id = self._rich_q1_fixture(str(root / "hit"))
            hit = self._public_rich_preflight(_engine(app, config, model), model)
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual("hit", hit["exact_revisit"]["status"])
            self.assertIn(edge_id, hit["association_ids"])
            self.assertIn(edge_id, hit["contextual_association"]["attached_edges"])
            executed = hit["exact_revisit"]["executed_modules"]
            self.assertFalse(executed["planner"])
            self.assertFalse(executed["embedding"])
            self.assertFalse(executed["reranker"])
            self.assertTrue(executed["contextual_matcher"])
            self.assertTrue(executed["source_closure"])

            masked_app, masked_config, masked_model, masked_edge = self._rich_q1_fixture(
                str(root / "masked")
            )
            masked_engine = _engine(masked_app, masked_config, masked_model)
            _mask_engine_edge(masked_engine, masked_edge)
            self.assertIsNone(self._public_rich_preflight(masked_engine, masked_model))

            source_app, source_config, source_model, _source_edge = self._rich_q1_fixture(
                str(root / "source-drift")
            )
            with source_app.db.transaction() as connection:
                source_id = int(
                    connection.execute("SELECT source_id FROM episode WHERE id = 2").fetchone()[0]
                )
                connection.execute(
                    "UPDATE source SET raw_text = ? WHERE id = ?",
                    ("[record: drift]\\nsource revision invalidates the original span", source_id),
                )
            self.assertIsNone(
                self._public_rich_preflight(
                    _engine(source_app, source_config, source_model), source_model
                )
            )

            policy_app, policy_config, policy_model, _policy_edge = self._rich_q1_fixture(
                str(root / "policy-change")
            )
            policy_config.retrieval.answer_episode_limit = 1
            self.assertIsNone(
                self._public_rich_preflight(
                    _engine(policy_app, policy_config, policy_model), policy_model
                )
            )

            restart_app, restart_config, _restart_model, restart_edge = self._rich_q1_fixture(
                str(root / "restart")
            )
            restarted = MemoryApplication(deepcopy(restart_config))
            restarted.rebuild_indexes()
            restarted.rebuild_contextual_indexes()
            restart_model = _RichRequirementPilotModel()
            restarted_hit = self._public_rich_preflight(
                _engine(restarted, restarted.config, restart_model), restart_model
            )
            self.assertIsNotNone(restarted_hit)
            assert restarted_hit is not None
            self.assertIn(restart_edge, restarted_hit["association_ids"])

    def test_rich_requirement_with_planner_paraphrase_keeps_exact_projection(self):
        """A paraphrase must not displace the existing whole-question binding."""

        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(
                directory,
                episode_texts=(
                    "圣园未花提到她与阿里乌斯结盟，以打倒共同敌人。",
                    "圣园未花说自己一直在暗中支援阿里乌斯。",
                ),
            )
            config.retrieval.rerank_atomic_query_limit = 1
            model = _RichRequirementParaphraseModel()
            engine = _engine(app, config, model)
            q1 = engine.query(
                RICH_QUESTION,
                contextual_learning=True,
                learning_request_id="rich-requirement-with-paraphrase",
                contextual_domain="knowledge",
                contextual_revisit_scope_hash=RICH_SCOPE_HASH,
            )

            requirement = q1["authoritative_requirements"]["requirements"][0]
            self.assertEqual(["圣园未花"], requirement["subject_terms"])
            self.assertEqual("暗中支援的组织", requirement["relation_hint"])
            self.assertEqual("一直", requirement["temporal_hint"])
            self.assertEqual("explicit_negation", requirement["negation_hint"])
            learning = q1["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(
                ["runtime_manifest_ready"],
                [item["reason"] for item in learning["revisit_projections"]],
            )
            self.assertIsNotNone(
                self._public_rich_preflight(_engine(app, config, model), model)
            )

    def test_rich_requirement_many_q1_cues_keeps_whole_exact_contract(self):
        """Multiple planner cues cannot make a whole-question slot ambiguous.

        This is a regression for the persisted N11-shaped requirement: its
        subject/relation/time/negation fields remain intact while all ten
        planner retrieval alternatives stay Q1-only.  The assertion uses the
        public Q1 finalizer and public Q2 entry; it does not inject a need,
        manifest, or edge.
        """

        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(
                directory,
                episode_texts=(
                    "圣园未花提到她与阿里乌斯结盟，以打倒共同敌人。",
                    "圣园未花说自己一直在暗中支援阿里乌斯。",
                ),
            )
            config.retrieval.rerank_atomic_query_limit = 1
            model = _RichRequirementManyPlannerCuesModel()
            q1 = _engine(app, config, model).query(
                RICH_QUESTION,
                contextual_learning=True,
                learning_request_id="rich-requirement-many-q1-cues",
                contextual_domain="knowledge",
                contextual_revisit_scope_hash=RICH_SCOPE_HASH,
            )

            requirement = q1["authoritative_requirements"]["requirements"][0]
            self.assertEqual(["圣园未花"], requirement["subject_terms"])
            self.assertEqual("暗中支援的组织", requirement["relation_hint"])
            self.assertEqual("一直", requirement["temporal_hint"])
            self.assertEqual("explicit_negation", requirement["negation_hint"])
            learning = q1["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(
                ["runtime_manifest_ready"],
                [item["reason"] for item in learning["revisit_projections"]],
            )
            hit = self._public_rich_preflight(_engine(app, config, model), model)
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertTrue(hit["answer_generation_skipped"])

    def test_rich_requirement_records_explicit_projection_rejection(self):
        """A ready edge is not silently reported as exact-reuse ready."""

        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(
                directory,
                episode_texts=(
                    "圣园未花提到她与阿里乌斯结盟，以打倒共同敌人。",
                    "圣园未花说自己一直在暗中支援阿里乌斯。",
                ),
            )
            config.retrieval.rerank_atomic_query_limit = 1
            result = _engine(app, config, _RichRequirementPilotModel()).query(
                RICH_QUESTION,
                contextual_learning=True,
                learning_request_id="rich-requirement-no-scope",
                contextual_domain="knowledge",
                # Deliberately do not provide a scope. This is an ordinary
                # Q1 edge creation, but automatic Q2 exact reuse must reject.
            )
            learning = result["contextual_learning"]
            self.assertEqual("ready", learning["status"])
            self.assertEqual(1, len(learning["receipt_ids"]))
            self.assertEqual(
                [
                    {
                        "status": "rejected_not_reconstructable",
                        "reason": "ordinary_contract_input_or_scope_invalid",
                    }
                ],
                [
                    {"status": item["status"], "reason": item["reason"]}
                    for item in learning["revisit_projections"]
                ],
            )
            receipt_id = learning["receipt_ids"][0]
            self.assertIsNone(
                app.associations.load_contextual_revisit_contract(receipt_id)
            )
            self.assertIsNone(
                app.associations.load_contextual_revisit_runtime_manifest(receipt_id)
            )

    def test_rerank_timeout_is_caught_and_current_flow_still_generates_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(directory)
            model = _RerankTimeoutModel()
            result = _engine(app, config, model).query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-rerank-timeout-control-flow",
            )

            self.assertIn("rerank_timeout", model.calls)
            self.assertIn("answer", model.calls)
            self.assertFalse(result["answer_generation_skipped"])
            self.assertIn("local injected rerank timeout", result["rerank_trace"]["error"])

    def test_missing_slots_mode_skips_followup_when_initial_candidates_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(directory)
            config.retrieval.followup_planning_mode = "missing_slots"
            model = _LocalPilotModel()
            result = _engine(app, config, model).query(
                QUESTION,
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-missing-slot-control-flow",
            )

            self.assertNotIn("followup_query_planning", model.calls)
            self.assertFalse(result["followup_planner_invoked"])
            self.assertEqual(
                "all_initial_candidate_slots_present",
                result["followup_planning_reason"],
            )

    def test_singleton_requirement_uses_raw_question_not_planner_paraphrase(self):
        with tempfile.TemporaryDirectory() as directory:
            app, config = self._app(directory)
            config.retrieval.rerank_atomic_query_limit = 1
            result = _engine(app, config, _ParaphraseFirstPlannerModel()).query(
                QUESTION,
                contextual_learning=True,
                learning_request_id="local-singleton-requirement-alignment",
                contextual_domain="knowledge",
                contextual_revisit_scope_hash="local-singleton-requirement-alignment",
            )

        requirement = result["authoritative_requirements"]["requirements"][0]
        self.assertEqual(QUESTION, requirement["question"])
        self.assertEqual("ready", result["contextual_learning"]["status"])

    def test_case_manifest_rejects_scoring_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_source_slice.v1",
                        "formal_scoring_eligible": False,
                        "promotion_prohibited": True,
                        "pilot_input": {
                            "q1_text": QUESTION,
                            "q2_text": QUESTION,
                            "contextual_domain": "knowledge",
                            "contextual_revisit_scope": "scope",
                            "answer": "must be rejected",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Q2 gold"):
                PilotCase.from_manifest(path)

    def test_case_manifest_projects_readable_scope_to_opaque_runtime_key(self):
        """The runner, rather than the caller, bridges label and V17 key."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            scope_label = "v3-real-source-diagnostic-pilot-32170"
            path.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_source_slice.v1",
                        "formal_scoring_eligible": False,
                        "promotion_prohibited": True,
                        "pilot_input": {
                            "q1_text": RICH_QUESTION,
                            "q2_text": RICH_QUESTION,
                            "contextual_domain": "knowledge",
                            "contextual_revisit_scope": scope_label,
                        },
                    }
                ),
                encoding="utf-8",
            )
            case = PilotCase.from_manifest(path)

        self.assertEqual(_pilot_scope_hash(scope_label), case.scope_hash)
        self.assertRegex(case.scope_hash, r"^pilot-scope:sha256:[0-9a-f]{64}$")
        self.assertNotEqual(scope_label, case.scope_hash)


if __name__ == "__main__":
    unittest.main()
