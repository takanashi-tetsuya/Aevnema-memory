from __future__ import annotations

"""Render an already-persisted Aevnema v3 recall trace as Markdown.

This is deliberately a read-only verifier.  It validates the run manifest and
every JSONL event against the vendored v3 contract, checks every referenced
artifact's digest, and then reports only records that were actually observed.
It never invokes a model, opens a network connection, or writes inside the
trace run directory.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from memory_demo.trace import (
    TRACE_EVENT_TYPES,
    TraceContractError,
    TracePrivacyError,
    TraceStateError,
    validate_persisted_trace_run,
)


# The contract exposes a set, whereas reports benefit from a stable lifecycle
# order.  Keep this list tied to the same contract event names and fall back to
# lexical ordering if the vendored contract evolves before this renderer does.
_LIFECYCLE_EVENT_ORDER = (
    "request_received",
    "requirements_resolved",
    "vector_bundle_ready",
    "revisit_probe",
    "base_retrieval",
    "anchors_selected",
    "edge_scored",
    "target_checked",
    "candidate_merged",
    "selector_step",
    "recovery_step",
    "mask_evaluated",
    "evidence_delivered",
    "provider_call",
    "response_checked",
    "learning_committed",
    "request_completed",
)
if set(_LIFECYCLE_EVENT_ORDER) == set(TRACE_EVENT_TYPES):
    CANONICAL_EVENT_TYPES = _LIFECYCLE_EVENT_ORDER
else:  # pragma: no cover - compatibility protection for a future contract.
    CANONICAL_EVENT_TYPES = tuple(sorted(TRACE_EVENT_TYPES))


class TraceRenderError(RuntimeError):
    """A persisted run cannot be safely verified or rendered."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One artifact reference plus the record from which it came."""

    source: str
    index: int
    value: Mapping[str, Any]


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _collect_artifact_references(
    manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    records: list[tuple[str, Mapping[str, Any]]] = (
        [("run_manifest.json", manifest)]
        + [
            (f"events.jsonl line {index}", event)
            for index, event in enumerate(events, start=1)
        ]
    )
    for source, record in records:
        raw_references = record.get("artifact_refs", [])
        if not isinstance(raw_references, list):
            # The schema checker reports this too; retain a clear guard for
            # type checkers and for callers that reuse this helper.
            raise TraceRenderError(f"artifact_refs in {source} is not an array")
        for index, reference in enumerate(raw_references, start=1):
            if not isinstance(reference, Mapping):
                raise TraceRenderError(f"artifact reference {index} in {source} is not an object")
            references.append(ArtifactReference(source, index, reference))
    return references


def _markdown_code(value: Any) -> str:
    """Render untrusted trace metadata without letting it alter report syntax."""

    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"`{rendered.replace('`', '\\`')}`"


def _markdown_values(values: Sequence[Any]) -> str:
    if not values:
        return "_none observed_"
    return ", ".join(_markdown_code(value) for value in values)


def _append_artifact_refs(lines: list[str], references: Sequence[Mapping[str, Any]], *, indent: str = "") -> None:
    if not references:
        lines.append(f"{indent}- Artifact refs: _none observed_")
        return
    lines.append(f"{indent}- Artifact refs:")
    for reference in references:
        lines.append(
            f"{indent}  - {_markdown_code(reference['artifact_id'])}; "
            f"path {_markdown_code(reference['path'])}; "
            f"SHA-256 {_markdown_code(reference['sha256'])}; "
            f"media type {_markdown_code(reference['media_type'])}; "
            f"visibility {_markdown_code(reference['visibility'])}; "
            f"redacted {_markdown_code(reference['redacted'])}"
        )


def _render_markdown(
    run_dir: Path,
    manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    references: Sequence[ArtifactReference],
) -> str:
    by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_request[str(event["request_id"])].append(event)
    for records in by_request.values():
        records.sort(key=lambda record: int(record["sequence"]))

    unique_artifact_paths = {str(item.value["path"]) for item in references}
    lines = [
        "# Aevnema v3 Recall Trace Case Report",
        "",
        "This report is a read-only rendering of persisted trace records. It lists "
        "only observed events and declared artifact references; it does not infer or "
        "backfill unobserved stages.",
        "",
        "## Run verification",
        "",
        f"- Run directory: {_markdown_code(run_dir.name)}",
        f"- Run ID: {_markdown_code(manifest['run_id'])}",
        f"- Trace version: {_markdown_code(manifest['trace_version'])}",
        f"- Manifest status: {_markdown_code(manifest['status'])}",
        f"- JSONL events observed: {len(events)}",
        f"- Declared artifact references verified: {len(references)} ({len(unique_artifact_paths)} unique path(s))",
        "- Validation: run manifest and every observed JSONL event satisfy the vendored v3 JSON Schema; referenced artifact files exist and match their declared SHA-256 values.",
        "",
        "## Run-manifest artifact references",
        "",
    ]
    _append_artifact_refs(lines, manifest["artifact_refs"])
    lines.extend(["", "## Requests", ""])

    if not by_request:
        lines.extend(
            [
                "No request event was observed. `events.jsonl` may be absent for a planned run, "
                "or may contain no records; the renderer does not synthesize a request.",
                "",
            ]
        )
    for request_id in sorted(by_request):
        records = by_request[request_id]
        observed_types = [str(record["event_type"]) for record in records]
        observed_type_set = set(observed_types)
        missing_types = [
            event_type
            for event_type in CANONICAL_EVENT_TYPES
            if event_type not in observed_type_set
        ]
        observed_stages = list(dict.fromkeys(str(record["stage"]) for record in records))
        lines.extend(
            [
                f"### Request {_markdown_code(request_id)}",
                "",
                f"- Observed event types: {_markdown_values(observed_types)}",
                f"- Observed stage labels: {_markdown_values(observed_stages)}",
                f"- Unobserved v3 event types: {_markdown_values(missing_types)}",
                "- Note: `stage` is a free-form contract field, so the report does not invent a list of missing stage-label values.",
                "",
            ]
        )
        for record in records:
            lines.extend(
                [
                    f"#### Sequence {record['sequence']}: {_markdown_code(record['event_type'])} / stage {_markdown_code(record['stage'])}",
                    "",
                    f"- Event ID: {_markdown_code(record['event_id'])}",
                    f"- Parents: {_markdown_values(record['parent_event_ids'])}",
                ]
            )
            _append_artifact_refs(lines, record["artifact_refs"])
            lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "An item marked unobserved means this persisted run contains no event with that v3 `event_type` for the request. It is not evidence that the corresponding retrieval or learning operation happened, failed, or was skipped for any reason.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_output(out_path: Path, content: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def render_recall_trace(run_dir: Path, output: Path) -> Path:
    """Validate and render a trace run, writing only the requested output file."""

    try:
        resolved_run_dir = run_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TraceRenderError(f"trace run directory does not exist: {run_dir}") from exc
    if not resolved_run_dir.is_dir():
        raise TraceRenderError(f"trace run path is not a directory: {run_dir}")

    # Resolve before writing anything.  This prevents an explicit or symlinked
    # output target from modifying the trace run, including its artifacts.
    resolved_output = output.resolve(strict=False)
    if _is_within(resolved_output, resolved_run_dir):
        raise TraceRenderError("--output must be outside the trace run directory")

    try:
        persisted = validate_persisted_trace_run(run_dir)
    except (OSError, TraceContractError, TracePrivacyError, TraceStateError) as exc:
        raise TraceRenderError(f"strict trace validation failed: {exc}") from exc
    references = _collect_artifact_references(persisted.manifest, persisted.events)
    report = _render_markdown(
        persisted.run_dir,
        persisted.manifest,
        persisted.events,
        references,
    )
    _write_output(resolved_output, report)
    return resolved_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read and verify an Aevnema v3 recall trace run, then render a "
            "Markdown case report without changing that run."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing run_manifest.json, optional events.jsonl, and artifacts/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown report path; it must be outside the trace run directory",
    )
    args = parser.parse_args(argv)
    try:
        rendered = render_recall_trace(args.run_dir, args.output)
    except (TraceRenderError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Rendered verified trace report: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
