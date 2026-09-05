from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.llm import ModelClient
from memory_demo.retrieval import QueryEngine
from benchmarks.support.stage5 import load_json, score_evidence_retrieval, write_json


QUESTION_ID = "old_cathedral_symbol_infrastructure_and_limits"


def _engine(app: MemoryApplication, config: AppConfig, operation: str) -> QueryEngine:
    logger = app.new_logger(operation)
    return QueryEngine(
        config,
        ModelClient(config.model, logger),
        app.episode_index,
        app.concept_index,
        app.episodes,
        app.concepts,
        app.sources,
        app.associations,
        logger,
        paragraph_index=app.paragraph_index,
        paragraphs=app.paragraphs,
    )


def _group_recall(ids: list[int], criterion: dict) -> dict:
    selected = set(int(value) for value in ids)
    groups = criterion["required_episode_groups"]
    hits = sum(bool(selected.intersection(int(value) for value in group)) for group in groups)
    return {
        "hits": hits,
        "groups": len(groups),
        "recall": hits / len(groups) if groups else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact Top-20 recall on frozen Stage-5 replay bundles."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "validation/evaluation-stage7-generation/pilot/block-001/C/graph.db"
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
        default=Path("validation/top20-no-growth-replay-audit.json"),
    )
    args = parser.parse_args()

    base = AppConfig.from_env()
    base.database_path = args.database
    base.log_dir = args.output.parent / "top20-replay-logs"
    base.paragraph.enabled = False
    base.retrieval.growth_max_rounds = 0
    base.retrieval.answer_episode_limit = 20
    app = MemoryApplication(base)
    app.rebuild_indexes()

    vector_config = deepcopy(base)
    vector_config.retrieval.graph_max_hops = 0
    static_config = deepcopy(base)
    static_config.retrieval.graph_max_hops = 3
    engines = {
        "vector_only": _engine(app, vector_config, "top20-vector"),
        "graph_static": _engine(app, static_config, "top20-static"),
    }
    criterion = load_json(args.manifest)["questions"][QUESTION_ID]
    rows = []
    for bundle_path in sorted(args.bundles.glob("block-*/q2-replay-bundle.json")):
        original = load_json(bundle_path)
        mode_rows = {}
        for mode, engine in engines.items():
            bundle = deepcopy(original)
            bundle["version"] = 1
            bundle["configuration"] = engine._replay_configuration()
            result = engine.replay_retrieval(bundle)
            candidate_top20 = [
                int(value) for value in result["candidate_episode_ids"][:20]
            ]
            selected_top20 = [int(value) for value in result["episode_ids"]]
            mode_rows[mode] = {
                "candidate_top20": _group_recall(candidate_top20, criterion),
                "selected_top20": _group_recall(selected_top20, criterion),
                "candidate_episode_ids": candidate_top20,
                "selected_episode_ids": selected_top20,
            }
        rows.append({"bundle": str(bundle_path), "modes": mode_rows})

    summary = {}
    for mode in engines:
        summary[mode] = {}
        for layer in ("candidate_top20", "selected_top20"):
            values = [float(row["modes"][mode][layer]["recall"]) for row in rows]
            summary[mode][layer] = {
                "values": values,
                "mean": sum(values) / len(values) if values else 0.0,
                "minimum": min(values) if values else 0.0,
                "maximum": max(values) if values else 0.0,
                "runs_above_95_percent": sum(value > 0.95 for value in values),
            }
    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_id": QUESTION_ID,
            "top_k": 20,
            "growth_max_rounds": 0,
            "paragraph_enabled": False,
            "bundle_count": len(rows),
            "summary": summary,
            "rows": rows,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
