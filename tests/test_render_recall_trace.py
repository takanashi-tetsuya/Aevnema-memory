from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.render_recall_trace import TraceRenderError, render_recall_trace
from memory_demo.trace import RecallTraceWriter


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_COMMIT = "e" * 40


def _manifest(*, run_id: str = "render-trace-unit") -> dict[str, object]:
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
        "arm": "renderer-unit",
        "requires_contextual": False,
        "answer_prose_cache_enabled": False,
        "actual_modules": ["memory_demo.trace"],
        "artifact_refs": [],
    }


def _request_payload(query_artifact_id: str) -> dict[str, object]:
    return {
        "query_artifact_id": query_artifact_id,
        "context_sha256": _SHA_A,
        "permission_scope_sha256": _SHA_B,
        "database_sha256": _SHA_C,
        "knowledge_epoch": "renderer-unit-epoch",
        "core_commit": _COMMIT,
        "chatbot_commit": None,
        "config_sha256": _SHA_D,
        "request_mode": "factual",
        "budget_ms": 1_000.0,
        "delivered_episode_budget": 5,
        "delivered_token_budget": 1_024,
    }


def _completed_payload() -> dict[str, object]:
    return {
        "route": "resolve",
        "status": "completed",
        "total_ms": 10.0,
        "cloud_logical_calls": 0,
        "cloud_http_attempts": 0,
        "contextual_http_attempts": 0,
        "embedding_logical_batches": 1,
        "actually_skipped_stages": [],
        "learning_receipt_ids": [],
        "fallback_kind": None,
        "reason_code": "renderer_unit_complete",
    }


def _complete_trace(root: Path) -> tuple[RecallTraceWriter, Path]:
    writer = RecallTraceWriter(root, _manifest())
    query = writer.artifacts.put_text("renderer unit query", visibility="full_local")
    session = writer.start_request(scope_id="renderer-unit-scope", request_id="request-renderer")
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
        parent_event_ids=[received],
    )
    writer.finalize("completed")
    return writer, writer.run_dir / query.path


class RenderRecallTraceTests(unittest.TestCase):
    def test_renders_observed_records_and_explicit_unobserved_event_types(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, _ = _complete_trace(root)
            output = root / "case-report.md"

            rendered = render_recall_trace(writer.run_dir, output)
            report = rendered.read_text(encoding="utf-8")

        self.assertEqual(output, rendered)
        self.assertIn('### Request `"request-renderer"`', report)
        self.assertIn('"request_received"', report)
        self.assertIn('"request_completed"', report)
        self.assertIn("Unobserved v3 event types:", report)
        self.assertIn('"vector_bundle_ready"', report)
        self.assertIn("does not infer or backfill unobserved stages", report)

    def test_rejects_a_tampered_artifact_before_creating_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, artifact = _complete_trace(root)
            artifact.write_bytes(b"tampered artifact")
            output = root / "case-report.md"

            with self.assertRaisesRegex(TraceRenderError, "artifact hash mismatch"):
                render_recall_trace(writer.run_dir, output)

            self.assertFalse(output.exists())

    def test_reuses_strict_payload_closure_and_opaque_label_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, _ = _complete_trace(root)
            records = [
                json.loads(line)
                for line in writer.events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records[0]["payload"]["query_artifact_id"] = f"sha256:{'f' * 64}"
            writer.events_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = root / "case-report.md"

            with self.assertRaisesRegex(TraceRenderError, "artifact pointer"):
                render_recall_trace(writer.run_dir, output)

            self.assertFalse(output.exists())

    def test_rejects_a_symlinked_artifact_even_when_its_content_matches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, artifact = _complete_trace(root)
            target = root / "same-content-artifact"
            target.write_bytes(artifact.read_bytes())
            artifact.unlink()
            try:
                artifact.symlink_to(target)
            except OSError:
                self.skipTest("this Windows test environment cannot create symlinks")
            output = root / "case-report.md"

            with self.assertRaisesRegex(TraceRenderError, "symbolic link"):
                render_recall_trace(writer.run_dir, output)

            self.assertFalse(output.exists())

            # Restore the local pointer, then verify the renderer also applies
            # the writer's opaque-stage boundary to persisted input.
            records[0]["payload"]["query_artifact_id"] = records[0]["artifact_refs"][0][
                "artifact_id"
            ]
            records[0]["stage"] = "private stage text"
            writer.events_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TraceRenderError, "opaque machine label"):
                render_recall_trace(writer.run_dir, output)

            self.assertFalse(output.exists())

    def test_rejects_output_inside_run_without_writing_to_the_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, _ = _complete_trace(root)
            output = writer.run_dir / "must-not-write.md"
            before = {
                path.relative_to(writer.run_dir).as_posix(): path.read_bytes()
                for path in writer.run_dir.rglob("*")
                if path.is_file()
            }

            with self.assertRaisesRegex(TraceRenderError, "outside the trace run directory"):
                render_recall_trace(writer.run_dir, output)

            after = {
                path.relative_to(writer.run_dir).as_posix(): path.read_bytes()
                for path in writer.run_dir.rglob("*")
                if path.is_file()
            }
            self.assertFalse(output.exists())
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
