from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from memory_demo.trace import (
    TRACE_CONTRACT_UPSTREAM_SHA256,
    RecallTraceWriter,
    TraceContractError,
    TracePrivacyError,
    TraceStateError,
    load_trace_contract,
    validate_case_summary,
    validate_persisted_trace_run,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_COMMIT = "e" * 40


def _manifest(*, run_id: str = "trace-unit-run") -> dict[str, object]:
    """Return a complete, schema-valid v3 run manifest."""
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
        "arm": "unit-trace",
        "requires_contextual": False,
        "answer_prose_cache_enabled": False,
        "actual_modules": ["memory_demo.trace"],
        "artifact_refs": [],
    }


def _request_payload(query_artifact_id: str) -> dict[str, object]:
    """Return the full p_request shape rather than a permissive test stub."""
    return {
        "query_artifact_id": query_artifact_id,
        "context_sha256": _SHA_A,
        "permission_scope_sha256": _SHA_B,
        "database_sha256": _SHA_C,
        "knowledge_epoch": "unit-epoch-1",
        "core_commit": _COMMIT,
        "chatbot_commit": None,
        "config_sha256": _SHA_D,
        "request_mode": "factual",
        "budget_ms": 1_000.0,
        "delivered_episode_budget": 5,
        "delivered_token_budget": 1_024,
    }


def _completed_payload() -> dict[str, object]:
    """Return the complete p_done shape used for a successful test request."""
    return {
        "route": "resolve",
        "status": "completed",
        "total_ms": 12.5,
        "cloud_logical_calls": 0,
        "cloud_http_attempts": 0,
        "contextual_http_attempts": 0,
        "embedding_logical_batches": 1,
        "actually_skipped_stages": [],
        "learning_receipt_ids": [],
        "fallback_kind": None,
        "reason_code": "unit_test_complete",
    }


def _provider_payload(*, purpose: str = "test:provider") -> dict[str, object]:
    """Return a schema-complete provider receipt for trace tamper tests."""
    return {
        "call_id": "call-unit-1",
        "logical_batch_id": "batch-unit-1",
        "attempt": 1,
        "role": "embedding",
        "requested_model": "unit-bge-model",
        "actual_model": "unit-bge-model",
        "purpose": purpose,
        "status": "succeeded",
        "queue_ms": 0.0,
        "network_ms": 1.0,
        "first_token_ms": None,
        "total_ms": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": None,
        "http_status": 200,
        "error_class": None,
        "late_result_discarded": False,
    }


def _vector_payload(
    *, query_artifact_id: str, query_sha256: str, vector_artifact_id: str, vector_sha256: str
) -> dict[str, object]:
    """Return a schema-complete nested payload for pointer-closure checks."""

    return {
        "query_refs": [
            {
                "query_id": "query-unit",
                "vector_id": "vector-unit",
                "text_sha256": query_sha256,
                "role": "whole",
                "slot_ids": ["slot-unit"],
                "text_artifact_id": query_artifact_id,
            }
        ],
        "physical_vectors": [
            {
                "vector_id": "vector-unit",
                "artifact_id": vector_artifact_id,
                "array_key": "whole",
                "sha256": vector_sha256,
                "dtype": "float32",
                "dimension": 1,
                "embedding_space_id": "embedding-space-unit",
                "normalization": "l2",
                "origin": "provider",
            }
        ],
        "whole_query_ref": "query-unit",
        "embedding_space_id": "embedding-space-unit",
        "logical_batches": 1,
        "http_attempts": 1,
        "stage_index": 0,
        "strict_single_batch": True,
    }


def _event_records(writer: RecallTraceWriter) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in writer.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_record(writer: RecallTraceWriter) -> dict[str, object]:
    return json.loads(writer.manifest_path.read_text(encoding="utf-8"))


def _write_event_records(
    writer: RecallTraceWriter, records: list[dict[str, object]]
) -> None:
    writer.events_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class RecallTraceWriterTests(unittest.TestCase):
    def test_vendored_contract_keeps_the_approved_identity_and_hash(self):
        contract = load_trace_contract()

        self.assertEqual(
            "urn:aevnema:recall-trace:v3:2026-09-05", contract["$id"]
        )
        self.assertEqual(
            "aevnema.recall-trace.v3",
            contract["$defs"]["traceEvent"]["properties"]["trace_version"]["const"],
        )
        self.assertEqual(
            17,
            len(
                contract["$defs"]["traceEvent"]["properties"]["event_type"][
                    "enum"
                ]
            ),
        )
        self.assertEqual(
            "b666b72935aded65d951e9fed3ba20bff10a5b19d2210fb9c59029012d9efc86",
            TRACE_CONTRACT_UPSTREAM_SHA256,
        )

    def _emit_complete_request(
        self,
        writer: RecallTraceWriter,
        *,
        scope_id: str = "scope-unit",
        query_text: str = "测试问题",
    ) -> tuple[object, str, str]:
        query = writer.artifacts.put_text(query_text, visibility="full_local")
        session = writer.start_request(scope_id=scope_id)
        received = session.emit(
            event_type="request_received",
            stage="request",
            payload=_request_payload(query.as_dict()["artifact_id"]),
            artifact_refs=[query],
            parent_event_ids=[],
        )
        completed = session.emit(
            event_type="request_completed",
            stage="complete",
            payload=_completed_payload(),
            artifact_refs=[],
            parent_event_ids=[received],
        )
        return session, received, completed

    def test_valid_trace_persists_complete_contract_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RecallTraceWriter(root, _manifest())
            session, received, completed = self._emit_complete_request(writer)
            writer.finalize("completed")
            writer.validate_persisted()

            manifest = _manifest_record(writer)
            records = _event_records(writer)

        self.assertEqual("completed", manifest["status"])
        self.assertEqual("trace-unit-run", manifest["run_id"])
        self.assertEqual(2, len(records))
        self.assertEqual("request_received", records[0]["event_type"])
        self.assertEqual("request_completed", records[1]["event_type"])
        self.assertEqual(session.request_id, records[0]["request_id"])
        self.assertEqual(0, records[0]["sequence"])
        self.assertEqual(1, records[1]["sequence"])
        self.assertEqual(received, records[0]["event_id"])
        self.assertEqual(completed, records[1]["event_id"])
        self.assertEqual([received], records[1]["parent_event_ids"])
        self.assertEqual(_request_payload(records[0]["payload"]["query_artifact_id"]), records[0]["payload"])
        self.assertEqual(_completed_payload(), records[1]["payload"])

    def test_contract_rejects_missing_payload_fields_and_extra_top_level_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest["unexpected"] = True
            with self.assertRaises(TraceContractError):
                RecallTraceWriter(root / "bad-manifest", manifest)

            missing_payload_writer = RecallTraceWriter(root / "missing-payload", _manifest())
            query = missing_payload_writer.artifacts.put_text(
                "合同缺字段测试", visibility="full_local"
            )
            session = missing_payload_writer.start_request(scope_id="scope-invalid")
            malformed = _request_payload(query.as_dict()["artifact_id"])
            del malformed["budget_ms"]
            with self.assertRaises(TraceContractError):
                session.emit(
                    event_type="request_received",
                    stage="request",
                    payload=malformed,
                    artifact_refs=[query],
                    parent_event_ids=[],
                )

            writer = RecallTraceWriter(root / "tampered-event", _manifest())
            # `validate_persisted` must revalidate what is on disk, not merely
            # trust in-memory records: a JSONL record with an unknown top-level
            # key violates the contract's additionalProperties=false rule.
            self._emit_complete_request(writer, scope_id="scope-valid")
            writer.finalize("completed")
            records = _event_records(writer)
            records[0]["unknown_top_level"] = "not permitted"
            writer.events_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaises(TraceContractError):
                writer.validate_persisted()

    def test_same_request_records_have_contiguous_sequence_and_causal_parent(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            session, received, completed = self._emit_complete_request(writer)
            writer.finalize("completed")
            records = [
                record
                for record in _event_records(writer)
                if record["request_id"] == session.request_id
            ]

        self.assertEqual([0, 1], [record["sequence"] for record in records])
        self.assertEqual([], records[0]["parent_event_ids"])
        self.assertEqual(received, records[0]["event_id"])
        self.assertEqual(completed, records[1]["event_id"])
        self.assertEqual(received, records[1]["parent_event_ids"][0])

    def test_concurrent_requests_keep_request_local_sequences_and_parentage(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest(run_id="parallel-run"))

            def emit(index: int) -> tuple[str, str]:
                session, received, _ = self._emit_complete_request(
                    writer,
                    scope_id=f"parallel-scope-{index}",
                    query_text=f"并发问题 {index}",
                )
                return session.request_id, received

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(emit, range(8)))
            writer.finalize("completed")
            writer.validate_persisted()
            records = _event_records(writer)

        self.assertEqual(16, len(records))
        self.assertEqual(16, len({str(record["event_id"]) for record in records}))
        self.assertEqual(8, len({request_id for request_id, _ in results}))
        by_request: dict[str, list[dict[str, object]]] = {}
        for record in records:
            by_request.setdefault(str(record["request_id"]), []).append(record)
        self.assertEqual({request_id for request_id, _ in results}, set(by_request))
        first_event_by_request = dict(results)
        for request_id, request_records in by_request.items():
            request_records.sort(key=lambda record: int(record["sequence"]))
            self.assertEqual([0, 1], [record["sequence"] for record in request_records])
            self.assertEqual("request_received", request_records[0]["event_type"])
            self.assertEqual("request_completed", request_records[1]["event_type"])
            self.assertEqual(
                [first_event_by_request[request_id]],
                request_records[1]["parent_event_ids"],
            )

    def test_foreign_request_parent_is_rejected(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            first_session, first_event, _ = self._emit_complete_request(writer)
            second_query = writer.artifacts.put_text("第二个问题")
            second_session = writer.start_request(scope_id="scope-second")
            second_received = second_session.emit(
                "request_received",
                stage="request",
                payload=_request_payload(second_query.artifact_id),
                artifact_refs=[second_query],
            )

            with self.assertRaises(TraceStateError):
                second_session.emit(
                    "request_completed",
                    stage="complete",
                    payload=_completed_payload(),
                    parent_event_ids=[first_event],
                )

            second_session.emit(
                "request_completed",
                stage="complete",
                payload=_completed_payload(),
                parent_event_ids=[second_received],
            )
            self.assertTrue(first_session.completed)
            writer.finalize("completed")

    def test_artifacts_deduplicate_and_shareable_artifacts_are_redacted(self):
        secret = "trace-shareable-secret"
        raw_prompt = "这是不能出现在 shareable 工件内的原始提示文本"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RecallTraceWriter(root, _manifest())
            first = writer.artifacts.put_text("content-addressed payload", visibility="full_local")
            second = writer.artifacts.put_text("content-addressed payload", visibility="full_local")
            shareable = writer.artifacts.put_json(
                {
                    "api_key": secret,
                    "headers": {"Authorization": f"Bearer {secret}"},
                    "prompt": raw_prompt,
                    "safe_label": "retain this",
                },
                visibility="shareable",
            )
            shareable_ref = shareable.as_dict()
            shareable_contents = (writer.run_dir / shareable_ref["path"]).read_text(
                encoding="utf-8"
            )

        self.assertEqual(first.as_dict()["artifact_id"], second.as_dict()["artifact_id"])
        self.assertEqual(first.as_dict()["sha256"], second.as_dict()["sha256"])
        self.assertEqual(first.as_dict()["path"], second.as_dict()["path"])
        self.assertEqual("shareable", shareable_ref["visibility"])
        self.assertTrue(shareable_ref["redacted"])
        self.assertNotIn(secret, shareable_contents)
        self.assertNotIn(raw_prompt, shareable_contents)
        self.assertIn("retain this", shareable_contents)

    def test_shareable_artifact_cannot_opt_out_of_redaction(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            with self.assertRaises(TracePrivacyError):
                writer.artifacts.put_json(
                    {"prompt": "raw"}, visibility="shareable", redacted=False
                )

    def test_live_writes_reject_noncanonical_or_duplicate_artifact_references(self):
        """Direct mapping inputs cannot bypass ArtifactStore's safe defaults."""

        mutations = {
            "artifact_id": lambda reference: reference.__setitem__(
                "artifact_id", f"sha256:{_SHA_A}"
            ),
            "path": lambda reference: reference.__setitem__(
                "path", f"artifacts/{_SHA_A}"
            ),
            "redaction": lambda reference: reference.update(
                {"visibility": "audit", "redacted": False}
            ),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    writer = RecallTraceWriter(root / label, _manifest(run_id=label))
                    query = writer.artifacts.put_text("artifact validation query")
                    reference = query.as_dict()
                    mutate(reference)
                    session = writer.start_request(scope_id=f"scope-{label}")
                    with self.assertRaises(TraceContractError):
                        session.emit(
                            "request_received",
                            stage="request",
                            payload=_request_payload(str(reference["artifact_id"])),
                            artifact_refs=[reference],
                        )

            writer = RecallTraceWriter(root / "duplicate", _manifest(run_id="duplicate"))
            query = writer.artifacts.put_text("duplicate artifact query")
            session = writer.start_request(scope_id="scope-duplicate")
            with self.assertRaises(TraceContractError):
                session.emit(
                    "request_received",
                    stage="request",
                    payload=_request_payload(query.artifact_id),
                    artifact_refs=[query, query],
                )

    def test_persisted_validator_rejects_ambiguous_json_objects(self):
        """The persisted reader must not silently use JSON's last-key-wins rule."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            event_writer = RecallTraceWriter(root / "events", _manifest(run_id="events"))
            self._emit_complete_request(event_writer)
            event_writer.finalize("completed")
            event_writer.events_path.write_text(
                '{"record_type":"trace_event","record_type":"trace_event"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TraceContractError, "duplicate object key"):
                event_writer.validate_persisted()

            manifest_writer = RecallTraceWriter(
                root / "manifest", _manifest(run_id="manifest")
            )
            manifest_writer.manifest_path.write_text(
                '{"record_type":"run_manifest","record_type":"run_manifest"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TraceContractError, "duplicate object key"):
                manifest_writer.validate_persisted()

    def test_persisted_validator_rejects_artifact_reference_tampering(self):
        """On-disk validation repeats canonical and cross-record reference checks."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for label, mutate in {
                "path": lambda records: records[0]["artifact_refs"][0].__setitem__(
                    "path", f"artifacts/{_SHA_A}"
                ),
                "redaction": lambda records: records[0]["artifact_refs"][0].update(
                    {"visibility": "audit", "redacted": False}
                ),
            }.items():
                with self.subTest(label=label):
                    writer = RecallTraceWriter(root / label, _manifest(run_id=label))
                    self._emit_complete_request(writer)
                    writer.finalize("completed")
                    records = _event_records(writer)
                    mutate(records)
                    _write_event_records(writer, records)
                    with self.assertRaises(TraceContractError):
                        writer.validate_persisted()

            writer = RecallTraceWriter(root / "conflict", _manifest(run_id="conflict"))
            query = writer.artifacts.put_text("conflicting metadata query")
            session = writer.start_request(scope_id="scope-conflict")
            received = session.emit(
                "request_received",
                stage="request",
                payload=_request_payload(query.artifact_id),
                artifact_refs=[query],
            )
            session.emit(
                "request_completed",
                stage="complete",
                payload=_completed_payload(),
                artifact_refs=[query],
                parent_event_ids=[received],
            )
            writer.finalize("completed")
            records = _event_records(writer)
            records[1]["artifact_refs"][0]["media_type"] = "application/octet-stream"
            _write_event_records(writer, records)
            with self.assertRaisesRegex(TraceContractError, "conflicts with an earlier"):
                writer.validate_persisted()

    def test_persisted_validator_rejects_lifecycle_and_scope_tampering(self):
        """Roots, terminals, and scope binding remain meaningful after restart."""

        mutations = {
            "second-root": lambda records: (
                records[0].__setitem__("sequence", 1),
                records[1].__setitem__("sequence", 2),
            ),
            "scope-change": lambda records: records[1].__setitem__(
                "scope_id", "scope-other"
            ),
            "post-terminal": lambda records: records.append(
                {
                    **deepcopy(records[1]),
                    "event_id": "event-after-terminal",
                    "sequence": 2,
                    "parent_event_ids": [records[1]["event_id"]],
                }
            ),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    writer = RecallTraceWriter(root / label, _manifest(run_id=label))
                    self._emit_complete_request(writer)
                    writer.finalize("completed")
                    records = _event_records(writer)
                    mutate(records)
                    _write_event_records(writer, records)
                    with self.assertRaises((TraceContractError, TraceStateError)):
                        writer.validate_persisted()

    def test_live_writes_reject_free_form_trace_labels_and_purposes(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            with self.assertRaises(TracePrivacyError):
                writer.start_request(scope_id="private scope text")

            query = writer.artifacts.put_text("label validation query")
            session = writer.start_request(scope_id="scope-labels")
            with self.assertRaises(TracePrivacyError):
                session.emit(
                    "request_received",
                    stage="private stage text",
                    payload=_request_payload(query.artifact_id),
                    artifact_refs=[query],
                )

            received = session.emit(
                "request_received",
                stage="request",
                payload=_request_payload(query.artifact_id),
                artifact_refs=[query],
            )
            with self.assertRaises(TracePrivacyError):
                session.emit(
                    "provider_call",
                    stage="provider",
                    payload=_provider_payload(purpose="private prompt text"),
                    parent_event_ids=[received],
                )

    def test_nested_payload_artifact_pointers_require_local_event_references(self):
        """A nested query-ref pointer cannot escape an event's own receipt."""

        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            query = writer.artifacts.put_text("nested pointer query")
            vector = writer.artifacts.put_bytes(
                b"not-a-real-vector-but-a-valid-artifact",
                media_type="application/x-npy",
                visibility="full_local",
                redacted=False,
            )
            session = writer.start_request(scope_id="scope-nested-pointers")
            received = session.emit(
                "request_received",
                stage="request",
                payload=_request_payload(query.artifact_id),
                artifact_refs=[query],
            )
            payload = _vector_payload(
                query_artifact_id=query.artifact_id,
                query_sha256=query.sha256,
                vector_artifact_id=vector.artifact_id,
                vector_sha256=vector.sha256,
            )
            payload["query_refs"][0]["text_artifact_id"] = f"sha256:{_SHA_A}"

            with self.assertRaisesRegex(TraceContractError, "artifact pointer"):
                session.emit(
                    "vector_bundle_ready",
                    stage="vectors",
                    payload=payload,
                    artifact_refs=[query, vector],
                    parent_event_ids=[received],
                )

    def test_case_summary_artifact_pointers_close_over_verified_run_catalog(self):
        with TemporaryDirectory() as directory:
            writer = RecallTraceWriter(Path(directory), _manifest())
            self._emit_complete_request(writer)
            writer.finalize("completed")
            trace_run = validate_persisted_trace_run(writer.run_dir)
            query_artifact_id = _event_records(writer)[0]["artifact_refs"][0][
                "artifact_id"
            ]
            summary = {
                "record_type": "case_summary",
                "trace_version": "aevnema.recall-trace.v3",
                "run_id": writer.run_dir.name,
                "family_id": "family-unit",
                "q1_request_id": _event_records(writer)[0]["request_id"],
                "q2_request_ids": [],
                "creation_receipt_ids": [],
                "comparison_artifact_ids": [query_artifact_id],
                "independent_gold_metrics_artifact_id": query_artifact_id,
                "provider_totals_artifact_id": query_artifact_id,
                "outcome": "no_op",
                "evidence_level": "unit_verified",
                "limitations": ["unit limitation"],
                "case_report_artifact_id": query_artifact_id,
            }

            validate_case_summary(summary, trace_run=trace_run)
            summary["comparison_artifact_ids"] = [f"sha256:{_SHA_A}"]
            with self.assertRaisesRegex(TraceContractError, "artifact pointer"):
                validate_case_summary(summary, trace_run=trace_run)

    def test_unfinalized_run_is_persisted_as_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RecallTraceWriter(root, _manifest())
            self._emit_complete_request(writer)
            writer.validate_persisted()
            manifest = _manifest_record(writer)

        self.assertEqual("incomplete", manifest["status"])


if __name__ == "__main__":
    unittest.main()
