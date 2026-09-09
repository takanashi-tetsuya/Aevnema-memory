from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import requests

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.embeddings import EmbeddingIndex, encode_embedding
from memory_demo.llm.client import (
    ModelClient,
    ProviderCallObservation,
    TracePersistenceError,
)
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    SourceRepository,
)
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.trace import QueryTraceBridge, RecallTraceWriter, TraceStateError
from memory_demo.types import EpisodeDraft, QueryIntent


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_COMMIT = "e" * 40


def _manifest(*, run_id: str) -> dict[str, object]:
    return {
        "record_type": "run_manifest",
        "trace_version": "aevnema.recall-trace.v3",
        "run_id": run_id,
        "status": "planned",
        "core_commit": _COMMIT,
        "chatbot_commit": None,
        "database_sha256": _SHA_A,
        "source_manifest_sha256": _SHA_B,
        "gold_manifest_sha256": None,
        "split_manifest_sha256": _SHA_C,
        "config_sha256": _SHA_D,
        "replay_mode": "input_frozen",
        "arm": "unit-query-bridge",
        "requires_contextual": False,
        "answer_prose_cache_enabled": False,
        "actual_modules": ["memory_demo.trace", "memory_demo.retrieval.engine"],
        "artifact_refs": [],
    }


def _request_payload(query_artifact_id: str) -> dict[str, object]:
    """The complete p_request shape; no test-only shortcuts."""

    return {
        "query_artifact_id": query_artifact_id,
        "context_sha256": _SHA_A,
        "permission_scope_sha256": _SHA_B,
        "database_sha256": _SHA_C,
        "knowledge_epoch": "query-bridge-unit-epoch",
        "core_commit": _COMMIT,
        "chatbot_commit": None,
        "config_sha256": _SHA_D,
        "request_mode": "factual",
        "budget_ms": 1_000.0,
        "delivered_episode_budget": 5,
        "delivered_token_budget": 1_024,
    }


def _vector_payload(
    *,
    query_artifact_id: str,
    query_sha256: str,
    vector_artifact_id: str,
    vector_sha256: str,
) -> dict[str, object]:
    """Build the complete p_vectors shape for one observed vector bundle."""

    return {
        "query_refs": [
            {
                "query_id": "query-checkpoint-unit",
                "vector_id": "vector-checkpoint-unit",
                "text_sha256": query_sha256,
                "role": "whole",
                "slot_ids": ["slot-checkpoint-unit"],
                "text_artifact_id": query_artifact_id,
            }
        ],
        "physical_vectors": [
            {
                "vector_id": "vector-checkpoint-unit",
                "artifact_id": vector_artifact_id,
                "array_key": "whole",
                "sha256": vector_sha256,
                "dtype": "float32",
                "dimension": 1,
                "embedding_space_id": "embedding-space-checkpoint-unit",
                "normalization": "l2",
                "origin": "provider",
            }
        ],
        "whole_query_ref": "query-checkpoint-unit",
        "embedding_space_id": "embedding-space-checkpoint-unit",
        "logical_batches": 1,
        "http_attempts": 1,
        "stage_index": 0,
        "strict_single_batch": True,
    }


def _records(writer: RecallTraceWriter) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in writer.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_record(writer: RecallTraceWriter) -> dict[str, object]:
    return json.loads(writer.manifest_path.read_text(encoding="utf-8"))


def _new_bridge(
    writer: RecallTraceWriter,
    *,
    scope_id: str,
    private_question: str,
) -> tuple[QueryTraceBridge, object]:
    """Build a bridge whose raw question is confined to a full-local artifact."""

    query = writer.artifacts.put_text(private_question, visibility="full_local")
    session = writer.start_request(scope_id=scope_id)
    bridge = QueryTraceBridge(
        session,
        request_payload=_request_payload(query.artifact_id),
        request_artifact_refs=[query],
    )
    return bridge, session


class _Response:
    """Minimal successful requests response for the real ModelClient hook."""

    status_code = 200
    headers: dict[str, str] = {}
    content = b""

    def __init__(self) -> None:
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "model": "served-bge-unit-model",
            "usage": {"prompt_tokens": 7, "completion_tokens": 0},
            "data": [{"index": 0, "embedding": [0.25, 0.75]}],
        }

    def close(self) -> None:
        self.closed = True


class _ProviderResponse:
    """Response stand-in for mixed chat/embedding and fallback assertions."""

    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.content = b""
        self._payload = payload
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict[str, object]:
        return self._payload

    def close(self) -> None:
        self.closed = True


def _embedding_response(*_args: object, **kwargs: object) -> _ProviderResponse:
    """Return one deterministic embedding per real coordinator input."""

    request = json.loads(bytes(kwargs["data"]).decode("utf-8"))
    inputs = request.get("input", [])
    return _ProviderResponse(
        {
            "model": "served-engine-trace-bge-model",
            "usage": {"prompt_tokens": len(inputs), "completion_tokens": 0},
            "data": [
                {
                    "index": index,
                    "embedding": (
                        [0.25, 0.75] if index % 2 == 0 else [0.75, 0.25]
                    ),
                }
                for index, _value in enumerate(inputs)
            ],
        }
    )


class _TraceAwareModel:
    """A no-network model proving QueryEngine keeps its outer binding alive."""

    def __init__(self) -> None:
        self.active = False
        self.entered = 0
        self.exited = 0

    @contextmanager
    def trace_request(self, _sink: object):
        self.entered += 1
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.exited += 1


class QueryTraceBridgeTests(unittest.TestCase):
    def _query_engine(
        self,
        model: _TraceAwareModel,
        *,
        growth_max_rounds: int = 0,
        growth_staging_enabled: bool = False,
    ) -> QueryEngine:
        """Construct the query boundary without a database or indexes."""

        engine = object.__new__(QueryEngine)
        engine.config = SimpleNamespace(
            retrieval=SimpleNamespace(
                growth_max_rounds=growth_max_rounds,
                growth_staging_enabled=growth_staging_enabled,
            ),
            weights=object(),
        )
        engine.model = model
        engine.associations = object()
        engine.traverser = object()
        engine.growth = object()
        engine.chronology = object()
        engine.episodes = object()
        engine.logger = None
        return engine

    def _persistent_query_engine(self, client: ModelClient) -> QueryEngine:
        """Build a real, empty retrieval boundary for trace-stage tests."""

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        config = AppConfig(
            database_path=root / "memory.db",
            log_dir=root / "logs",
            model=client.config,
        )
        config.retrieval.rerank_enabled = False
        config.retrieval.followup_planning_mode = "off"
        config.retrieval.growth_max_rounds = 0
        config.retrieval.growth_persist_only_used = False
        database = Database(config.database_path)
        database.initialize()
        return QueryEngine(
            config,
            client,
            EmbeddingIndex(client.config.embedding_dimension),
            EmbeddingIndex(client.config.embedding_dimension),
            EpisodeRepository(database),
            ConceptRepository(database),
            SourceRepository(database),
            AssociationRepository(database, config.weights),
        )

    def test_query_engine_emits_only_observed_requirement_and_initial_vector_stages(self) -> None:
        """A real query traces only frozen requirements and its ready bundle."""

        config = ModelConfig(
            api_key="unit-engine-trace-key",
            embedding_model="unit-engine-bge-model",
            embedding_dimension=2,
            max_retries=0,
        )
        client = ModelClient(config)
        engine = self._persistent_query_engine(client)
        private_question = "private engine trace question must remain local"
        private_requirement = "private atomic requirement must remain local"
        private_source = "private source evidence must never enter the receipt"
        source_id = engine.sources.insert(private_source)
        episode_id = engine.episodes.insert(
            source_id,
            "private/source.json",
            0,
            EpisodeDraft(
                "local summary only",
                evidence_quotes=[private_source],
                evidence_spans=[(1, 1)],
            ),
            encode_embedding(np.asarray([0.25, 0.75], dtype=np.float32), 2),
        )
        engine.episode_index.add(episode_id, [0.25, 0.75])

        def planned_intent_response(
            *_args: object,
            **kwargs: object,
        ) -> _ProviderResponse:
            request = json.loads(bytes(kwargs["data"]).decode("utf-8"))
            if "input" in request:
                return _embedding_response(**kwargs)
            return _ProviderResponse(
                {
                    "model": "served-engine-intent-planner",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "language": "en",
                                        "target_entities": [],
                                        "search_queries": [private_requirement],
                                    }
                                )
                            }
                        }
                    ],
                }
            )

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="engine-observed-stages")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-engine-observed-stages",
                private_question=private_question,
            )
            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=planned_intent_response,
            ):
                result = engine.query(
                    private_question,
                    generate_answer=False,
                    followup_queries_override=[],
                    trace_bridge=bridge,
                )
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

            event_types = [str(record["event_type"]) for record in records]
            self.assertEqual(
                [
                    "request_received",
                    "provider_call",
                    "requirements_resolved",
                    "provider_call",
                    "vector_bundle_ready",
                    "request_completed",
                ],
                event_types,
            )
            received, planner, requirements, provider, vectors, completed = records
            self.assertEqual(
                [received["event_id"], planner["event_id"]],
                requirements["parent_event_ids"],
            )
            self.assertEqual(
                [requirements["event_id"], provider["event_id"]],
                vectors["parent_event_ids"],
            )
            self.assertEqual("planner", planner["payload"]["role"])
            self.assertEqual("embedding", provider["payload"]["role"])
            self.assertEqual(
                [
                    received["event_id"],
                    requirements["event_id"],
                    vectors["event_id"],
                    planner["event_id"],
                    provider["event_id"],
                ],
                completed["parent_event_ids"],
            )
            self.assertEqual(
                ["requirements_resolved", "vector_bundle_ready"],
                [
                    event_type
                    for event_type in event_types
                    if event_type
                    not in {"request_received", "provider_call", "request_completed"}
                ],
            )

            requirement_payload = requirements["payload"]
            self.assertEqual("cloud", requirement_payload["planner_origin"])
            self.assertTrue(
                str(requirement_payload["requirements"][0]["question"]).startswith(
                    "sha256:"
                )
            )
            self.assertTrue(requirements["artifact_refs"])
            self.assertTrue(
                all(
                    reference["visibility"] == "full_local"
                    for reference in requirements["artifact_refs"]
                )
            )
            vector_payload = vectors["payload"]
            self.assertEqual(1, vector_payload["logical_batches"])
            self.assertEqual(1, vector_payload["http_attempts"])
            self.assertFalse(vector_payload["strict_single_batch"])
            self.assertEqual(2, len(vector_payload["query_refs"]))
            self.assertEqual(2, len(vector_payload["physical_vectors"]))
            self.assertTrue(
                all(item["origin"] == "provider" for item in vector_payload["physical_vectors"])
            )
            vector_artifact_ids = {
                reference["artifact_id"] for reference in vectors["artifact_refs"]
            }
            self.assertTrue(
                {
                    item["text_artifact_id"] for item in vector_payload["query_refs"]
                }.issubset(vector_artifact_ids)
            )
            self.assertTrue(
                {
                    item["artifact_id"] for item in vector_payload["physical_vectors"]
                }.issubset(vector_artifact_ids)
            )
            self.assertTrue(
                all(
                    reference["visibility"] == "full_local"
                    for reference in vectors["artifact_refs"]
                )
            )
            self.assertEqual(
                2,
                result["query_vector_bundle"]["physical_vector_count"],
            )

        serialized = json.dumps(records, ensure_ascii=False)
        self.assertNotIn(private_question, serialized)
        self.assertNotIn(private_requirement, serialized)
        self.assertNotIn(private_source, serialized)

    def test_followup_bundle_gets_its_own_monotonic_stage_and_receipts(self) -> None:
        """A later real bundle is traced, without inventing an intervening stage."""

        config = ModelConfig(
            api_key="unit-engine-followup-trace-key",
            embedding_model="unit-engine-bge-model",
            embedding_dimension=2,
            max_retries=0,
        )
        client = ModelClient(config)
        engine = self._persistent_query_engine(client)
        engine.config.retrieval.followup_planning_mode = "always"
        private_question = "private followup trace question"
        private_requirement = "private initial trace requirement"
        private_followup = "private later trace discovery"

        def planned_followup_response(
            *_args: object,
            **kwargs: object,
        ) -> _ProviderResponse:
            request = json.loads(bytes(kwargs["data"]).decode("utf-8"))
            if "input" in request:
                return _embedding_response(**kwargs)
            return _ProviderResponse(
                {
                    "model": "served-engine-followup-planner",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"followup_queries": [private_followup]}
                                )
                            }
                        }
                    ],
                }
            )

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="engine-followup-vector-stages")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-engine-followup-vector-stages",
                private_question=private_question,
            )
            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=planned_followup_response,
            ):
                engine.query(
                    private_question,
                    generate_answer=False,
                    intent_override=QueryIntent(search_queries=[private_requirement]),
                    trace_bridge=bridge,
                )
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        self.assertEqual(
            [
                "request_received",
                "requirements_resolved",
                "provider_call",
                "vector_bundle_ready",
                "provider_call",
                "provider_call",
                "vector_bundle_ready",
                "request_completed",
            ],
            [record["event_type"] for record in records],
        )
        requirements = records[1]
        first_provider, first_vectors = records[2], records[3]
        planner_provider, second_provider, second_vectors = (
            records[4],
            records[5],
            records[6],
        )
        self.assertEqual(
            [requirements["event_id"], first_provider["event_id"]],
            first_vectors["parent_event_ids"],
        )
        self.assertEqual(
            [
                first_vectors["event_id"],
                planner_provider["event_id"],
                second_provider["event_id"],
            ],
            second_vectors["parent_event_ids"],
        )
        self.assertEqual("planner", planner_provider["payload"]["role"])
        self.assertEqual("embedding", second_provider["payload"]["role"])
        self.assertEqual(0, first_vectors["payload"]["stage_index"])
        self.assertEqual(1, second_vectors["payload"]["stage_index"])
        self.assertEqual(1, first_vectors["payload"]["logical_batches"])
        self.assertEqual(1, second_vectors["payload"]["logical_batches"])
        self.assertEqual(1, first_vectors["payload"]["http_attempts"])
        self.assertEqual(1, second_vectors["payload"]["http_attempts"])
        self.assertEqual(3, len(second_vectors["payload"]["query_refs"]))
        self.assertEqual(3, len(second_vectors["payload"]["physical_vectors"]))
        serialized = json.dumps(records, ensure_ascii=False)
        self.assertNotIn(private_question, serialized)
        self.assertNotIn(private_requirement, serialized)
        self.assertNotIn(private_followup, serialized)

    def test_vector_stage_persistence_fault_leaves_strict_receipt_incomplete(self) -> None:
        """A vector-stage write fault never turns into a false completion."""

        config = ModelConfig(
            api_key="unit-engine-trace-fault-key",
            embedding_model="unit-engine-bge-model",
            embedding_dimension=2,
            max_retries=0,
        )
        client = ModelClient(config)
        engine = self._persistent_query_engine(client)
        private_question = "private vector persistence fault question"
        original_emit = QueryTraceBridge.emit_observed_stage

        def fail_only_vector_stage(
            bridge: QueryTraceBridge,
            event_type: str,
            **kwargs: object,
        ) -> str:
            if event_type == "vector_bundle_ready":
                raise TracePersistenceError("private vector stage write failed")
            return original_emit(bridge, event_type, **kwargs)

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="engine-vector-stage-fault")
            )
            bridge, session = _new_bridge(
                writer,
                scope_id="scope-engine-vector-stage-fault",
                private_question=private_question,
            )
            with patch(
                "memory_demo.llm.client.requests.Session.post",
                side_effect=_embedding_response,
            ), patch.object(
                QueryTraceBridge,
                "emit_observed_stage",
                new=fail_only_vector_stage,
            ):
                with self.assertRaisesRegex(TracePersistenceError, "vector stage write failed"):
                    engine.query(
                        private_question,
                        generate_answer=False,
                        intent_override=QueryIntent(search_queries=["private requirement"]),
                        followup_queries_override=[],
                        trace_bridge=bridge,
                    )
            self.assertFalse(session.completed)
            writer.validate_persisted()
            records = _records(writer)
            writer.finalize("incomplete")

        self.assertEqual(
            ["request_received", "requirements_resolved", "provider_call"],
            [record["event_type"] for record in records],
        )

    def test_real_model_client_keeps_provider_events_and_done_counts_request_local(self) -> None:
        """Concurrent bridges sharing a ModelClient cannot cross-charge calls."""

        config = ModelConfig(
            api_key="unit-trace-key",
            embedding_model="unit-bge-model",
            embedding_dimension=2,
            max_retries=0,
            max_concurrent_requests=2,
        )
        client = ModelClient(config)
        entered_post = Barrier(2)

        def post(*_args: object, **_kwargs: object) -> _Response:
            entered_post.wait(timeout=5)
            return _Response()

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="request-local-provider-counts")
            )
            first_bridge, first_session = _new_bridge(
                writer,
                scope_id="scope-contextual",
                private_question="private contextual question must stay local",
            )
            second_bridge, second_session = _new_bridge(
                writer,
                scope_id="scope-base",
                private_question="private base question must stay local",
            )
            first_bridge.start()
            second_bridge.start()

            def run(
                bridge: QueryTraceBridge, *, purpose: str, marker: str
            ) -> None:
                with bridge.bind_model(client), client.provider_purpose(
                    "embedding", purpose
                ):
                    matrix = client.embed([f"private source evidence {marker}"])
                self.assertEqual((1, 2), matrix.shape)
                bridge.complete_success()

            with patch(
                "memory_demo.llm.client.requests.Session.post", side_effect=post
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(
                        run,
                        first_bridge,
                        purpose="contextual:unit-isolation",
                        marker="first",
                    )
                    second = executor.submit(
                        run,
                        second_bridge,
                        purpose="base:unit-isolation",
                        marker="second",
                    )
                    first.result(timeout=10)
                    second.result(timeout=10)

            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        by_request: dict[str, list[dict[str, object]]] = {}
        for record in records:
            by_request.setdefault(str(record["request_id"]), []).append(record)
        self.assertEqual({first_session.request_id, second_session.request_id}, set(by_request))

        for session, expected_contextual in (
            (first_session, 1),
            (second_session, 0),
        ):
            request_records = sorted(
                by_request[session.request_id], key=lambda item: int(item["sequence"])
            )
            self.assertEqual(
                ["request_received", "provider_call", "request_completed"],
                [record["event_type"] for record in request_records],
            )
            received, provider, completed = request_records
            provider_payload = provider["payload"]
            completed_payload = completed["payload"]
            self.assertEqual([received["event_id"]], provider["parent_event_ids"])
            self.assertEqual(
                [received["event_id"], provider["event_id"]],
                completed["parent_event_ids"],
            )
            self.assertEqual("embedding", provider_payload["role"])
            self.assertEqual("succeeded", provider_payload["status"])
            self.assertEqual("served-bge-unit-model", provider_payload["actual_model"])
            self.assertEqual(7, provider_payload["prompt_tokens"])
            self.assertEqual(200, provider_payload["http_status"])
            self.assertEqual(1, completed_payload["cloud_logical_calls"])
            self.assertEqual(1, completed_payload["cloud_http_attempts"])
            self.assertEqual(expected_contextual, completed_payload["contextual_http_attempts"])
            self.assertEqual(1, completed_payload["embedding_logical_batches"])
            self.assertIsNone(completed_payload["fallback_kind"])

        serialized = json.dumps(records, ensure_ascii=False)
        self.assertNotIn("private source evidence", serialized)
        self.assertNotIn("private contextual question", serialized)
        self.assertNotIn("private base question", serialized)

    def test_observed_stage_needs_explicit_parent_and_schema_complete_payload(self) -> None:
        """The bridge cannot infer a stage or silently attach it to the root."""

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="explicit-observed-stage")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-observed-stage",
                private_question="private stage query",
            )
            root = bridge.start()
            payload = {
                "requirements": [
                    {
                        "slot_id": "slot-1",
                        "question": f"sha256:{_SHA_A}",
                        "required": True,
                        "query_refs": ["query-ref-1"],
                        "origin": "user_explicit",
                        "support_mode": "alternative",
                        "clause_ids": ["clause-1"],
                        "subject_terms": [],
                        "object_terms": [],
                        "relation_hint": "",
                        "temporal_hint": "",
                        "epistemic_hint": "",
                    }
                ],
                "unresolved_slot_ids": ["slot-1"],
                "uncertain_slot_ids": [],
                "planner_origin": "explicit",
                "planner_call_ids": [],
            }
            with self.assertRaisesRegex(RuntimeError, "explicit parents"):
                bridge.emit_observed_stage(
                    "requirements_resolved",
                    stage="requirements",
                    payload=payload,
                    parent_event_ids=(),
                )
            event_id = bridge.emit_observed_stage(
                "requirements_resolved",
                stage="requirements",
                payload=payload,
                parent_event_ids=(root,),
            )
            bridge.complete_success()
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        self.assertEqual(
            ["request_received", "requirements_resolved", "request_completed"],
            [record["event_type"] for record in records],
        )
        self.assertEqual(event_id, records[1]["event_id"])
        self.assertEqual([root], records[1]["parent_event_ids"])

    def test_checkpoint_freezes_actual_provider_events_for_artifact_backed_stage(self) -> None:
        """A V3 stage may cite only observations present in its checkpoint."""

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="checkpointed-observed-stage")
            )
            query = writer.artifacts.put_text(
                "private checkpoint query stays in the full-local artifact"
            )
            session = writer.start_request(scope_id="scope-checkpointed-stage")
            bridge = QueryTraceBridge(
                session,
                request_payload=_request_payload(query.artifact_id),
                request_artifact_refs=[query],
            )
            root = bridge.start()
            before_provider = bridge.checkpoint_observations()
            bridge.ledger.logical_batch_started(
                logical_batch_id="batch-checkpoint-unit",
                operation="embedding",
                requested_model="unit-bge-model",
            )
            bridge.ledger.provider_call_finished(
                ProviderCallObservation(
                    call_id="call-checkpoint-unit",
                    logical_batch_id="batch-checkpoint-unit",
                    operation="embedding",
                    endpoint="https://private.example.invalid/embeddings",
                    requested_model="unit-bge-model",
                    actual_model="served-unit-bge-model",
                    role="embedding",
                    purpose="contextual:checkpoint",
                    status="succeeded",
                    sent=True,
                    queue_ms=0.0,
                    network_ms=1.0,
                    total_ms=1.0,
                    prompt_tokens=1,
                    completion_tokens=None,
                    http_status=200,
                    error_class=None,
                )
            )
            provider_delta = bridge.observations_since(before_provider)
            checkpoint = bridge.checkpoint_observations()
            self.assertEqual(session.request_id, checkpoint.request_id)
            self.assertEqual(root, checkpoint.request_event_id)
            self.assertEqual(1, checkpoint.provider.cloud_logical_calls)
            self.assertEqual(1, checkpoint.provider.cloud_http_attempts)
            self.assertEqual(1, checkpoint.provider.contextual_http_attempts)
            self.assertEqual(1, checkpoint.provider.embedding_logical_batches)
            self.assertEqual(
                checkpoint.provider.causal_parent_event_ids,
                checkpoint.observed_parent_event_ids,
            )
            self.assertEqual(
                checkpoint.provider_event_ids,
                provider_delta.provider_event_ids,
            )
            self.assertEqual(1, provider_delta.cloud_logical_calls)
            self.assertEqual(1, provider_delta.cloud_http_attempts)
            self.assertEqual(1, provider_delta.contextual_http_attempts)
            self.assertEqual(1, provider_delta.embedding_logical_batches)

            vector = writer.artifacts.put_bytes(
                b"unit-vector-artifact",
                media_type="application/x-npy",
                visibility="full_local",
                redacted=False,
            )
            vector_event = bridge.emit_observed_stage(
                "vector_bundle_ready",
                stage="vectors",
                payload=_vector_payload(
                    query_artifact_id=query.artifact_id,
                    query_sha256=query.sha256,
                    vector_artifact_id=vector.artifact_id,
                    vector_sha256=vector.sha256,
                ),
                artifact_refs=[query, vector],
                parent_event_ids=provider_delta.causal_parent_event_ids,
                observation_checkpoint=provider_delta,
            )
            after_stage = bridge.checkpoint()
            self.assertEqual((vector_event,), after_stage.observed_stage_event_ids)
            self.assertEqual(
                (root, vector_event, *checkpoint.provider_event_ids),
                after_stage.observed_parent_event_ids,
            )

            # The first snapshot predates ``vector_event``.  It cannot be used
            # to post-hoc claim that a following stage was caused by it.
            with self.assertRaisesRegex(TraceStateError, "supplied observation checkpoint"):
                bridge.emit_observed_stage(
                    "requirements_resolved",
                    stage="requirements",
                    payload={},
                    parent_event_ids=(vector_event,),
                    observation_checkpoint=checkpoint,
                )

            bridge.complete_success()
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        self.assertEqual(
            [
                "request_received",
                "provider_call",
                "vector_bundle_ready",
                "request_completed",
            ],
            [record["event_type"] for record in records],
        )
        self.assertEqual(
            [root, checkpoint.provider_event_ids[0]],
            records[2]["parent_event_ids"],
        )
        self.assertEqual(
            [root, vector_event, checkpoint.provider_event_ids[0]],
            records[3]["parent_event_ids"],
        )
        self.assertEqual(
            [query.artifact_id, vector.artifact_id],
            [reference["artifact_id"] for reference in records[2]["artifact_refs"]],
        )
        serialized = json.dumps(records, ensure_ascii=False)
        self.assertNotIn("private checkpoint query", serialized)
        self.assertNotIn("private.example.invalid", serialized)

    def test_checkpoint_rejects_foreign_or_faulted_observations_and_forged_provider_events(self) -> None:
        """The stage hook cannot impersonate a provider or cross request scope."""

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="checkpoint-scope-guards")
            )
            first_bridge, _first_session = _new_bridge(
                writer,
                scope_id="scope-checkpoint-first",
                private_question="private first checkpoint query",
            )
            second_bridge, _second_session = _new_bridge(
                writer,
                scope_id="scope-checkpoint-second",
                private_question="private second checkpoint query",
            )
            first_root = first_bridge.start()
            second_root = second_bridge.start()
            first_checkpoint = first_bridge.checkpoint()

            with self.assertRaisesRegex(TraceStateError, "provider"):
                first_bridge.emit_observed_stage(
                    "provider_call",
                    stage="provider",
                    payload={},
                    parent_event_ids=(first_root,),
                )
            with self.assertRaisesRegex(TraceStateError, "different trace request"):
                second_bridge.emit_observed_stage(
                    "requirements_resolved",
                    stage="requirements",
                    payload={},
                    parent_event_ids=(second_root,),
                    observation_checkpoint=first_checkpoint,
                )
            with self.assertRaisesRegex(TraceStateError, "different trace request"):
                second_bridge.observations_since(first_checkpoint)

            second_bridge.ledger.trace_persistence_failed(
                RuntimeError("private persistence fault")
            )
            with self.assertRaises(TracePersistenceError):
                second_bridge.checkpoint_observations()
            with self.assertRaises(TracePersistenceError):
                second_bridge.emit_observed_stage(
                    "requirements_resolved",
                    stage="requirements",
                    payload={},
                    parent_event_ids=(second_root,),
                )

            # A faulted request deliberately stays incomplete.  The unrelated
            # request can still complete, proving the guards remain scoped.
            first_bridge.complete_success()
            writer.finalize("incomplete")
            writer.validate_persisted()

    def test_query_engine_normal_exception_becomes_a_technical_failure_receipt(self) -> None:
        model = _TraceAwareModel()
        engine = self._query_engine(model)

        def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
            self.assertTrue(model.active)
            raise RuntimeError("private implementation diagnostic")

        engine._query_impl = explode
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="technical-failure-receipt")
            )
            bridge, session = _new_bridge(
                writer,
                scope_id="scope-technical-failure",
                private_question="private failing query",
            )

            with self.assertRaisesRegex(RuntimeError, "private implementation diagnostic"):
                engine.query("private failing query", trace_bridge=bridge)

            self.assertTrue(session.completed)
            self.assertEqual(1, model.entered)
            self.assertEqual(1, model.exited)
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        self.assertEqual(["request_received", "request_completed"], [record["event_type"] for record in records])
        completed = records[-1]
        self.assertEqual("failed", completed["payload"]["route"])
        self.assertEqual("technical_failure", completed["payload"]["status"])
        self.assertEqual("query_exception_RuntimeError", completed["payload"]["reason_code"])
        self.assertNotIn("private implementation diagnostic", json.dumps(records))

    def test_query_engine_keyboard_interrupt_leaves_the_trace_incomplete(self) -> None:
        model = _TraceAwareModel()
        engine = self._query_engine(model)

        def interrupted(*_args: object, **_kwargs: object) -> dict[str, object]:
            self.assertTrue(model.active)
            raise KeyboardInterrupt("operator stop")

        engine._query_impl = interrupted
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="interrupt-incomplete-receipt")
            )
            bridge, session = _new_bridge(
                writer,
                scope_id="scope-interrupt",
                private_question="private interrupted query",
            )

            with self.assertRaises(KeyboardInterrupt):
                engine.query("private interrupted query", trace_bridge=bridge)

            self.assertFalse(session.completed)
            self.assertEqual(1, model.entered)
            self.assertEqual(1, model.exited)
            writer.validate_persisted()
            manifest = _manifest_record(writer)
            records = _records(writer)
            writer.finalize("incomplete")

        self.assertEqual("incomplete", manifest["status"])
        self.assertEqual(["request_received"], [record["event_type"] for record in records])

    def test_query_engine_trace_uses_readonly_association_view(self) -> None:
        model = _TraceAwareModel()
        durable_mark_used: list[list[int]] = []
        readonly_mark_used: list[list[int]] = []

        class FakeDurableAssociations:
            def mark_used(self, association_ids: list[int]) -> None:
                durable_mark_used.append(list(association_ids))

        class FakeReadOnlyOverlay:
            """Minimal overlay proving trace execution cannot write durable state."""

            def __init__(self, durable: object) -> None:
                self.durable = durable

            def mark_used(self, association_ids: list[int]) -> None:
                readonly_mark_used.append(list(association_ids))

        engine = self._query_engine(
            model,
            growth_max_rounds=1,
            growth_staging_enabled=True,
        )
        durable_associations = FakeDurableAssociations()
        engine.associations = durable_associations

        def query_impl(*_args: object, **kwargs: object) -> dict[str, object]:
            self.assertTrue(model.active)
            self.assertIsInstance(engine.associations, FakeReadOnlyOverlay)
            self.assertFalse(kwargs["allow_association_learning"])
            engine.associations.mark_used([17])
            return {"answer": "unit answer"}

        engine._query_impl = query_impl
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="readonly-trace-scope")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-readonly-trace",
                private_question="private staged query",
            )
            with patch(
                "memory_demo.retrieval.engine.AssociationRepository",
                FakeDurableAssociations,
            ), patch(
                "memory_demo.retrieval.engine.AssociationOverlay",
                FakeReadOnlyOverlay,
            ), patch(
                "memory_demo.retrieval.engine.GraphTraverser",
                side_effect=lambda associations: ("traverser", associations),
            ), patch(
                "memory_demo.retrieval.engine.AssociationGrowthEngine",
                side_effect=lambda *_args: "growth",
            ), patch(
                "memory_demo.retrieval.engine.ChronologyService",
                side_effect=lambda *_args: "chronology",
            ):
                result = engine.query("private staged query", trace_bridge=bridge)

            self.assertEqual("unit answer", result["answer"])
            self.assertEqual(
                {
                    "enabled": False,
                    "committed": False,
                    "temporary_to_durable_ids": {},
                    "reason": "strict_trace_read_only",
                },
                result["growth_staging"],
            )
            self.assertTrue(result["strict_trace_read_only"])
            self.assertIs(engine.associations, durable_associations)
            self.assertEqual([], durable_mark_used)
            self.assertEqual([[17]], readonly_mark_used)
            self.assertFalse(model.active)
            self.assertEqual(1, model.entered)
            self.assertEqual(1, model.exited)
            writer.finalize("completed")
            writer.validate_persisted()

    def test_query_boundaries_emit_truthful_planner_contextual_answer_and_audit_purposes(self) -> None:
        """Strict receipts use literal roles at each real ModelClient boundary."""

        config = ModelConfig(
            api_key="unit-purpose-key",
            embedding_model="unit-bge-model",
            embedding_dimension=2,
            max_retries=0,
        )
        client = ModelClient(config)
        engine = object.__new__(QueryEngine)
        engine.model = client
        engine.config = SimpleNamespace(model=config)
        engine.logger = None
        chat_payloads = iter(
            (
                {
                    "model": "served-planner-model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "language": "zh",
                                        "target_entities": [],
                                        "search_queries": ["offline query"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                },
                {
                    "model": "served-answer-model",
                    "choices": [{"message": {"content": "offline answer"}}],
                },
                {
                    "model": "served-audit-model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "valid": True,
                                        "issues": [],
                                        "correction_instructions": [],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                },
            )
        )

        def post(*_args: object, **kwargs: object) -> _ProviderResponse:
            request = json.loads(bytes(kwargs["data"]).decode("utf-8"))
            if "input" in request:
                return _ProviderResponse(
                    {
                        "model": "served-bge-model",
                        "data": [{"index": 0, "embedding": [0.25, 0.75]}],
                    }
                )
            return _ProviderResponse(next(chat_payloads))

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="query-purpose-boundaries")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-purpose-boundaries",
                private_question="private query boundary question",
            )
            bridge.start()
            with patch(
                "memory_demo.llm.client.requests.Session.post", side_effect=post
            ):
                with bridge.bind_model(client):
                    intent = engine._parse_intent("private planner question")
                    bundle = engine.embed_query_bundle(
                        [("whole", "private contextual query")]
                    )
                    answer, audits, revisions = engine._generate_audited_answer(
                        "private answer question",
                        intent,
                        [],
                        [],
                        [],
                        [],
                    )
            self.assertEqual("offline answer", answer)
            self.assertEqual(1, len(audits))
            self.assertEqual(0, revisions)
            self.assertEqual(1, len(bundle.queries))
            bridge.complete_success()
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        provider_payloads = [
            record["payload"]
            for record in records
            if record["event_type"] == "provider_call"
        ]
        self.assertEqual(
            [
                ("planner", "query_intent"),
                ("embedding", "contextual:query_vector_bundle"),
                ("answer", "answer_generation"),
                ("audit", "answer_evidence_audit"),
            ],
            [
                (str(payload["role"]), str(payload["purpose"]))
                for payload in provider_payloads
            ],
        )
        completion = records[-1]["payload"]
        self.assertEqual(4, completion["cloud_logical_calls"])
        self.assertEqual(4, completion["cloud_http_attempts"])
        self.assertEqual(1, completion["contextual_http_attempts"])
        self.assertEqual(1, completion["embedding_logical_batches"])
        self.assertIsNone(completion["fallback_kind"])
        serialized = json.dumps(records, ensure_ascii=False)
        self.assertNotIn("private planner question", serialized)
        self.assertNotIn("private contextual query", serialized)
        self.assertNotIn("private answer question", serialized)

    def test_fallback_retains_boundary_purpose_and_is_receipted(self) -> None:
        """A model fallback stays attached to the original planner boundary."""

        config = ModelConfig(
            api_key="unit-fallback-key",
            embedding_dimension=2,
            max_retries=0,
        )
        client = ModelClient(config)
        engine = object.__new__(QueryEngine)
        engine.model = client
        attempts: list[str] = []

        def post(*_args: object, **kwargs: object) -> _ProviderResponse:
            request = json.loads(bytes(kwargs["data"]).decode("utf-8"))
            attempts.append(str(request["model"]))
            if len(attempts) == 1:
                return _ProviderResponse({}, status_code=503)
            return _ProviderResponse(
                {
                    "model": "served-glm-4.5v",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "language": "zh",
                                        "target_entities": [],
                                        "search_queries": [],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                }
            )

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="query-purpose-fallback")
            )
            bridge, _session = _new_bridge(
                writer,
                scope_id="scope-purpose-fallback",
                private_question="private fallback question",
            )
            bridge.start()
            with patch(
                "memory_demo.llm.client.requests.Session.post", side_effect=post
            ):
                with bridge.bind_model(client):
                    intent = engine._parse_intent("private fallback planner question")
            self.assertEqual([], intent.search_queries)
            bridge.complete_success()
            writer.finalize("completed")
            writer.validate_persisted()
            records = _records(writer)

        self.assertEqual([config.reasoning_model, config.fallback_model], attempts)
        provider_payloads = [
            record["payload"]
            for record in records
            if record["event_type"] == "provider_call"
        ]
        self.assertEqual(2, len(provider_payloads))
        self.assertEqual(
            [("planner", "query_intent"), ("planner", "query_intent")],
            [
                (str(payload["role"]), str(payload["purpose"]))
                for payload in provider_payloads
            ],
        )
        self.assertEqual(["failed", "succeeded"], [
            str(payload["status"]) for payload in provider_payloads
        ])
        completion = records[-1]["payload"]
        self.assertEqual(1, completion["cloud_logical_calls"])
        self.assertEqual(2, completion["cloud_http_attempts"])
        self.assertEqual("model_fallback", completion["fallback_kind"])

    def test_faulted_ledger_refuses_a_false_success_receipt(self) -> None:
        """A swallowed best-effort path cannot later complete a strict trace."""

        model = _TraceAwareModel()
        engine = self._query_engine(model)
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(
                Path(directory), _manifest(run_id="faulted-ledger-incomplete")
            )
            bridge, session = _new_bridge(
                writer,
                scope_id="scope-faulted-ledger",
                private_question="private faulted receipt question",
            )

            def supposedly_successful(*_args: object, **_kwargs: object) -> dict:
                bridge.ledger.trace_persistence_failed(
                    RuntimeError("private persistence failure")
                )
                return {"answer": "ordinary result"}

            engine._query_impl = supposedly_successful
            with self.assertRaises(TracePersistenceError):
                engine.query("private faulted receipt question", trace_bridge=bridge)

            self.assertFalse(session.completed)
            writer.validate_persisted()
            records = _records(writer)
            writer.finalize("incomplete")

        self.assertEqual(["request_received"], [record["event_type"] for record in records])


if __name__ == "__main__":
    unittest.main()
