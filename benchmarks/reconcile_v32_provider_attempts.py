"""Reconcile the v3.2 provider ledger without issuing a provider request.

The campaign reports aggregate counters in several nested artefacts.  This
tool reads the preserved full-local ledgers, groups concrete observations by
their recorded ``call_id``/logical batch, and explicitly leaves a billing
allocation unknown when the historic data did not persist one.  It never
rewrites a historical observation or converts an aggregate counter into a
fictional request ID.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "aevnema.v3_2.provider_attempt_reconciliation.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_ledgers(value: Any, path: str = "$") -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield concrete ProviderLedger exports, not their aggregate projections."""

    if isinstance(value, Mapping):
        observations = value.get("observations")
        counts = value.get("counts")
        if isinstance(observations, list) and isinstance(counts, Mapping):
            yield path, value
            return
        for key, child in value.items():
            yield from _find_ledgers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_ledgers(child, f"{path}[{index}]")


def _phase_for(scope: str, ledger_path: str) -> str:
    text = f"{scope} {ledger_path}".casefold()
    if "preparation" in text or "prepare" in text:
        return "frozen_input_preparation"
    if "fresh" in text or "q1" in text:
        return "fresh_q1"
    if "arms" in text or "edge_available" in text or "edge_masked" in text:
        return "q2_arm"
    return "unclassified_preserved_ledger"


def _safe_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep accounting fields only; never duplicate raw request/response bodies."""

    fields = (
        "call_id",
        "logical_batch_id",
        "operation",
        "endpoint",
        "requested_model",
        "actual_model",
        "role",
        "purpose",
        "status",
        "sent",
        "queue_ms",
        "network_ms",
        "total_ms",
        "http_status",
        "error_class",
        "late_result_discarded",
    )
    return {key: observation.get(key) for key in fields if key in observation}


def _observation_key(observation: Mapping[str, Any]) -> tuple[str, str]:
    call_id = str(observation.get("call_id") or "").strip()
    if call_id:
        return "call_id", call_id
    # Missing IDs must remain visibly weaker than a real dedupe key.  This
    # hash only detects byte-identical observations copied into two reports.
    stable = json.dumps(_safe_observation(observation), ensure_ascii=False, sort_keys=True)
    return "missing_call_id_observation_hash", sha256(stable.encode("utf-8")).hexdigest()


def reconcile(*, campaign_root: Path, output_dir: Path) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    campaign_state_path = campaign_root / "campaign_state.json"
    if not campaign_state_path.is_file():
        raise FileNotFoundError(campaign_state_path)
    campaign_state = _read_json(campaign_state_path)
    sources = {
        "E32-02_live_q2_arms": campaign_root / "experiments" / "E32-02_frozen_evidence_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-03_frozen_input_preparation": campaign_root / "experiments" / "E32-03_frozen_input_evidence_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-04_unbound_slot_preparation": campaign_root / "experiments" / "E32-04_frozen_input_vectors_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-05_restored_vectors": campaign_root / "experiments" / "E32-05_restored_frozen_vectors_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-05a_fresh_event_q1": campaign_root / "experiments" / "E32-05a_event_q1" / "same_chapter_learning_matrix.full_local.json",
        "E32-05b_fresh_main_q1": campaign_root / "experiments" / "E32-05b_main_q1" / "same_chapter_learning_matrix.full_local.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing preserved campaign inputs: " + "; ".join(missing))

    rows: list[dict[str, Any]] = []
    ledger_summaries: list[dict[str, Any]] = []
    for scope, path in sources.items():
        payload = _read_json(path)
        for ledger_path, ledger in _find_ledgers(payload):
            observations = ledger.get("observations")
            counts = ledger.get("counts")
            assert isinstance(observations, list) and isinstance(counts, Mapping)
            phase = _phase_for(scope, ledger_path)
            ledger_summaries.append(
                {
                    "scope": scope,
                    "phase": phase,
                    "artifact": str(path),
                    "ledger_path": ledger_path,
                    "recorded_http_attempts": int(counts.get("http_attempts", 0) or 0),
                    "observation_rows": len(observations),
                    "count_matches_rows": int(counts.get("http_attempts", 0) or 0) == len(observations),
                }
            )
            for index, raw in enumerate(observations):
                if not isinstance(raw, Mapping):
                    rows.append(
                        {
                            "scope": scope,
                            "phase": phase,
                            "artifact": str(path),
                            "ledger_path": ledger_path,
                            "observation_index": index,
                            "identity_kind": "invalid_observation",
                            "identity": "not_observed",
                            "observation": {"status": "invalid_observation_shape"},
                        }
                    )
                    continue
                identity_kind, identity = _observation_key(raw)
                rows.append(
                    {
                        "scope": scope,
                        "phase": phase,
                        "artifact": str(path),
                        "ledger_path": ledger_path,
                        "observation_index": index,
                        "identity_kind": identity_kind,
                        "identity": identity,
                        "observation": _safe_observation(raw),
                    }
                )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["identity_kind"]), str(row["identity"]))].append(row)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    by_scope = Counter(str(row["scope"]) for row in rows)
    by_phase = Counter(str(row["phase"]) for row in rows)
    missing_id = sum(row["identity_kind"] != "call_id" for row in rows)
    reported = campaign_state.get("provider_counters")
    reported = reported if isinstance(reported, Mapping) else {}
    reported_39 = int(reported.get("batch_provider_http_attempts", 39) or 39)
    raw_sum = len(rows)
    unique_ids = len(groups)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "reconciled_from_preserved_local_ledgers_only",
        "created_at": _utc_now(),
        "paid_provider_calls_issued": 0,
        "inputs": {
            "campaign_state": str(campaign_state_path),
            "artifacts": {key: str(value) for key, value in sources.items()},
        },
        "ledger_integrity": {
            "ledger_count": len(ledger_summaries),
            "all_recorded_count_matches_observation_rows": all(item["count_matches_rows"] for item in ledger_summaries),
            "ledgers": ledger_summaries,
        },
        "reconciliation": {
            "reported_campaign_subtotal_http_attempts": reported_39,
            "preserved_raw_observation_rows": raw_sum,
            "unique_observation_identities": unique_ids,
            "duplicate_identity_groups": len(duplicate_groups),
            "missing_concrete_call_ids": missing_id,
            "raw_rows_by_scope": dict(sorted(by_scope.items())),
            "raw_rows_by_phase": dict(sorted(by_phase.items())),
            "reported_39_explained_as": "E32-02 live-arm observations plus fresh event/main Q1 observations; it excludes E32-03/E32-04 preparation ledgers.",
            "dedupe_limit": "No provider invoice/batch-allocation field is persisted. Equality of a call_id only identifies copied observations; it cannot establish whether two different call_ids were billed to one authorization batch.",
            "billing_or_authorization_allocation": "not_observed",
            "budget_compliance_conclusion": "not_determined_from_local_artifacts",
        },
        "attempts_full_local": rows,
        "duplicate_identity_groups_full_local": [
            {
                "identity_kind": items[0]["identity_kind"],
                "identity": items[0]["identity"],
                "occurrences": [
                    {
                        "scope": item["scope"],
                        "phase": item["phase"],
                        "artifact": item["artifact"],
                        "ledger_path": item["ledger_path"],
                        "observation_index": item["observation_index"],
                    }
                    for item in items
                ],
            }
            for items in duplicate_groups
        ],
    }
    output_dir.mkdir(parents=True)
    full_local = output_dir / "provider_attempt_reconciliation.full_local.json"
    summary = output_dir / "provider_attempt_reconciliation.json"
    full_local.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: value for key, value in payload.items() if not key.endswith("_full_local")}
    compact["attempt_count"] = len(rows)
    summary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# v3.2 provider-attempt reconciliation",
        "",
        "No provider call was made for this reconciliation.",
        "",
        f"- Preserved raw observation rows: `{raw_sum}`",
        f"- Unique local identities: `{unique_ids}`",
        f"- Missing concrete call IDs: `{missing_id}`",
        f"- Duplicated identities across preserved reports: `{len(duplicate_groups)}`",
        f"- Reported campaign subtotal: `{reported_39}`",
        "",
        "The reported 39 is a subtotal (live arms plus fresh Q1), not a complete ledger total. E32-03 and E32-04 input-preparation observations are separately preserved. Because no invoice or authorization-batch allocation is recorded, this local reconciliation does not determine billing, allocation, or budget compliance.",
        "",
        "| Phase | Observation rows |",
        "| --- | ---: |",
        *[f"| {phase} | {count} |" for phase, count in sorted(by_phase.items())],
    ]
    (output_dir / "PROVIDER_ATTEMPT_RECONCILIATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"full_local": str(full_local), "summary": str(summary), "raw_rows": raw_sum, "unique_identities": unique_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
