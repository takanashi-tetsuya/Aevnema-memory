from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.association_overlay import AssociationDelta, AssociationOverlay
from memory_demo.config import AppConfig
from memory_demo.llm import ModelClient
from memory_demo.retrieval import QueryEngine
from benchmarks.support.stage5 import (
    load_json,
    mechanism_check,
    score_evidence_retrieval,
    write_json,
)


def _overlay_engine(app: MemoryApplication, overlay: AssociationOverlay) -> QueryEngine:
    logger = app.new_logger("generation-replay-masked")
    return QueryEngine(
        app.config,
        ModelClient(app.config.model, logger),
        app.episode_index,
        app.concept_index,
        app.episodes,
        app.concepts,
        app.sources,
        overlay,
        logger,
        paragraph_index=app.paragraph_index,
        paragraphs=app.paragraphs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay frozen Stage-5 retrieval bundles with generation-aware edges."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "validation/evaluation-stage7-generation/pilot/block-001/T/graph.db"
        ),
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=Path(
            "validation/evaluation-stage7-generation/pilot/block-001/T/association-delta.json"
        ),
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        default=Path("validation/evaluation-stage5-causality/official"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/stage4-network-evidence-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/stage7-generation-replay-probe.json"),
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    config.database_path = args.database
    config.log_dir = args.output.parent / "stage7-generation-replay-logs"
    config.paragraph.enabled = False
    config.retrieval.growth_max_rounds = 0
    app = MemoryApplication(config)
    app.rebuild_indexes()
    delta = AssociationDelta.from_dict(load_json(args.delta))
    criterion = load_json(args.manifest)["questions"][
        "old_cathedral_symbol_infrastructure_and_limits"
    ]
    overlay = AssociationOverlay.from_delta(app.associations, delta)
    treatment_engine = app.query_engine(app.new_logger("generation-replay-treatment"))
    masked_engine = _overlay_engine(app, overlay)
    before = app.associations.snapshot()

    rows = []
    for bundle_path in sorted(args.bundles.glob("block-*/q2-replay-bundle.json")):
        bundle = load_json(bundle_path)
        treatment = treatment_engine.replay_retrieval(bundle)
        masked = masked_engine.replay_retrieval(bundle)
        treatment_score = score_evidence_retrieval(treatment, criterion)
        masked_score = score_evidence_retrieval(masked, criterion)
        rows.append(
            {
                "bundle": str(bundle_path),
                "treatment": treatment_score,
                "masked": masked_score,
                "recall_delta": (
                    float(treatment_score["recall_at_30"])
                    - float(masked_score["recall_at_30"])
                ),
                "mechanism": mechanism_check(
                    treatment, masked, delta, criterion
                ),
            }
        )

    endpoint_probes = []
    for changed in [
        *(item["after"] for item in delta.created),
        *(item["after"] for item in delta.reinforced),
    ]:
        if changed["from_type"] != "episode" or changed["to_type"] != "episode":
            continue
        for seed_id, target_id in (
            (int(changed["from_id"]), int(changed["to_id"])),
            (int(changed["to_id"]), int(changed["from_id"])),
        ):
            bundle = {
                "version": 1,
                "question": str(changed["relation_text"]),
                "intent": {},
                "final_seed_hits": [
                    {
                        "node_type": "episode",
                        "node_id": seed_id,
                        "score": 1.0,
                    }
                ],
                "episode_anchor_ids": [seed_id],
                "configuration": treatment_engine._replay_configuration(),
            }
            treatment = treatment_engine.replay_retrieval(bundle)
            masked = masked_engine.replay_retrieval(bundle)
            endpoint_probes.append(
                {
                    "association_id": int(changed["id"]),
                    "generation": int(changed.get("generation", 0)),
                    "seed_episode_id": seed_id,
                    "target_episode_id": target_id,
                    "treatment_candidate": target_id
                    in treatment["candidate_episode_ids"],
                    "masked_candidate": target_id in masked["candidate_episode_ids"],
                    "treatment_selected": target_id in treatment["episode_ids"],
                    "masked_selected": target_id in masked["episode_ids"],
                    "treatment_used_edge": int(changed["id"])
                    in treatment["association_ids"],
                }
            )

    if app.associations.snapshot() != before:
        raise RuntimeError("replay probe modified Association rows")
    changed_rows = [
        *(item["after"] for item in delta.created),
        *(item["after"] for item in delta.reinforced),
    ]
    deltas = [float(item["recall_delta"]) for item in rows]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.database),
        "delta": str(args.delta),
        "bundle_count": len(rows),
        "paragraph_enabled": False,
        "generation_distribution": dict(
            sorted(Counter(int(row.get("generation", 0)) for row in changed_rows).items())
        ),
        "recall_delta_summary": {
            "values": deltas,
            "mean": sum(deltas) / len(deltas) if deltas else 0.0,
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
        "bundles_with_delta_path": sum(
            row["mechanism"]["delta_used_in_treatment_paths"] is True
            for row in rows
        ),
        "bundles_with_rank_or_selection_change": sum(
            row["mechanism"]["rank_or_selection_changed_after_mask"] is True
            for row in rows
        ),
        "endpoint_probe_summary": {
            "probe_count": len(endpoint_probes),
            "candidate_gain": sum(
                row["treatment_candidate"] and not row["masked_candidate"]
                for row in endpoint_probes
            ),
            "selected_gain": sum(
                row["treatment_selected"] and not row["masked_selected"]
                for row in endpoint_probes
            ),
            "used_edge": sum(
                row["treatment_used_edge"] for row in endpoint_probes
            ),
        },
        "endpoint_probes": endpoint_probes,
        "rows": rows,
    }
    write_json(args.output, output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
