"""Create and validate a no-network synthetic Aevnema v3 trace run.

This is a contract smoke test, not a retrieval-quality benchmark.  It writes
only synthetic text and verifies that the strict writer can produce a durable,
schema-valid run receipt for the renderer and later benchmark tooling.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from memory_demo.trace import (
    TRACE_CONTRACT_UPSTREAM_SHA256,
    TRACE_VERSION,
    RecallTraceWriter,
)


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _core_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    candidate = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(candidate) != 40:
        raise RuntimeError("could not resolve the current core git commit")
    return candidate


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest(run_id: str, core_commit: str) -> dict[str, Any]:
    return {
        "record_type": "run_manifest",
        "trace_version": TRACE_VERSION,
        "run_id": run_id,
        "status": "planned",
        "core_commit": core_commit,
        "chatbot_commit": None,
        "database_sha256": _sha("synthetic database"),
        "source_manifest_sha256": _sha("synthetic source manifest"),
        "gold_manifest_sha256": None,
        "split_manifest_sha256": _sha("synthetic split manifest"),
        "config_sha256": _sha("synthetic safe config"),
        "replay_mode": "input_frozen",
        "arm": "trace_contract_synthetic",
        "requires_contextual": False,
        "answer_prose_cache_enabled": False,
        "actual_modules": ["memory_demo.trace"],
        "artifact_refs": [],
    }


def build_synthetic_trace(output_dir: Path, run_id: str) -> dict[str, Any]:
    """Build one finite synthetic trace and return its validation receipt."""

    core_commit = _core_commit()
    writer = RecallTraceWriter(output_dir, _manifest(run_id, core_commit))
    query_text = "Synthetic query used only to validate the trace contract."
    query_artifact = writer.artifacts.put_text(query_text)
    request = writer.start_request(scope_id="synthetic-contract-scope")
    received_id = request.emit(
        "request_received",
        stage="request",
        payload={
            "query_artifact_id": query_artifact.artifact_id,
            "context_sha256": _sha("synthetic context"),
            "permission_scope_sha256": _sha("synthetic permission scope"),
            "database_sha256": _sha("synthetic database"),
            "knowledge_epoch": "synthetic-epoch-1",
            "core_commit": core_commit,
            "chatbot_commit": None,
            "config_sha256": _sha("synthetic safe config"),
            "request_mode": "factual",
            "budget_ms": 1_000.0,
            "delivered_episode_budget": 1,
            "delivered_token_budget": None,
        },
        artifact_refs=[query_artifact],
    )
    requirements_id = request.emit(
        "requirements_resolved",
        stage="requirements",
        payload={
            "requirements": [
                {
                    "slot_id": "synthetic-slot-1",
                    "question": f"sha256:{_sha(query_text)}",
                    "required": True,
                    "query_refs": [query_artifact.artifact_id],
                    "origin": "user_explicit",
                    "support_mode": "alternative",
                    "clause_ids": ["synthetic-clause-1"],
                    "subject_terms": [],
                    "object_terms": [],
                    "relation_hint": "",
                    "temporal_hint": "",
                    "epistemic_hint": "",
                }
            ],
            "unresolved_slot_ids": ["synthetic-slot-1"],
            "uncertain_slot_ids": [],
            "planner_origin": "explicit",
            "planner_call_ids": [],
        },
        parent_event_ids=[received_id],
    )
    request.emit(
        "request_completed",
        stage="complete",
        payload={
            "route": "not_applicable",
            "status": "completed",
            "total_ms": 0.0,
            "cloud_logical_calls": 0,
            "cloud_http_attempts": 0,
            "contextual_http_attempts": 0,
            "embedding_logical_batches": 0,
            "actually_skipped_stages": [
                "vector_bundle_ready",
                "base_retrieval",
                "provider_call",
                "evidence_delivered",
            ],
            "learning_receipt_ids": [],
            "fallback_kind": None,
            "reason_code": "synthetic_contract_validation",
        },
        parent_event_ids=[requirements_id],
    )
    writer.finalize("completed")
    writer.validate_persisted()
    event_count = len(writer.events_path.read_text(encoding="utf-8").splitlines())
    return {
        "validation_version": "aevnema.trace-contract-validation.v1",
        "status": "passed",
        "evidence_level": "unit_verified",
        "network_used": False,
        "model_calls": 0,
        "contract_sha256": TRACE_CONTRACT_UPSTREAM_SHA256,
        "run_id": run_id,
        "run_directory": str(writer.run_dir),
        "event_count": event_count,
        "checks": [
            "vendored_contract_hash",
            "draft_2020_12_record_validation",
            "jsonl_reparse",
            "request_sequence_and_parentage",
            "artifact_hash_closure",
        ],
        "limitations": [
            "Synthetic contract validation does not exercise a live query, provider, or retrieval lane.",
            "Unobserved v3 stages are explicitly listed in request_completed instead of fabricated.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive the immutable synthetic run and receipt.",
    )
    parser.add_argument(
        "--run-id",
        default="synthetic-trace-contract",
        help="Safe run identifier below --output-dir (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_synthetic_trace(args.output_dir, args.run_id)
    receipt_path = args.output_dir / "trace-contract-validation.json"
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
