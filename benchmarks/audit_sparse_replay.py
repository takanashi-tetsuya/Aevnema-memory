from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.retrieval import QueryEngine
from benchmarks.support.stage5 import load_json, write_json
from memory_demo.types import QueryIntent, SearchHit


QUESTION_ID = "old_cathedral_symbol_infrastructure_and_limits"


def _engine(app: MemoryApplication, config: AppConfig) -> QueryEngine:
    return QueryEngine(
        config,
        None,
        app.episode_index,
        app.concept_index,
        app.episodes,
        app.concepts,
        app.sources,
        app.associations,
        paragraph_index=app.paragraph_index,
        paragraphs=app.paragraphs,
        episode_sparse_index=app.episode_sparse_index,
        source_sparse_index=app.source_sparse_index,
    )


def _group_recall(ids: list[int], criterion: dict) -> dict:
    selected = set(int(value) for value in ids)
    groups = criterion["required_episode_groups"]
    hits = sum(
        bool(selected.intersection(int(value) for value in group))
        for group in groups
    )
    return {
        "hits": hits,
        "groups": len(groups),
        "recall": hits / len(groups) if groups else 0.0,
    }


def _rebuild_bundle(engine: QueryEngine, original: dict) -> tuple[dict, dict]:
    intent = QueryIntent.from_dict(original["intent"])
    initial_queries = [str(value) for value in original["initial_queries"]]
    initial_matrix = np.asarray(
        original["initial_query_embeddings_float32"], dtype=np.float32
    )
    initial_anchors: list[int] = []
    initial_hits, initial_rankings = engine._vector_seed_hits_from_matrix(
        initial_queries,
        initial_matrix,
        initial_anchors,
        engine.config.retrieval.answer_whole_question_anchor_episodes,
    )
    alias_hits: list[SearchHit] = []
    for entity in intent.target_entities:
        for row in engine.concepts.find_by_alias(entity):
            alias_hits.append(
                SearchHit(
                    "concept",
                    int(row["canonical_concept_id"] or row["id"]),
                    1.0,
                )
            )
    seeds = engine._merge_hits(initial_hits, alias_hits)

    followup_queries = [str(value) for value in original.get("followup_queries", [])]
    followup_anchors: list[int] = []
    followup_rankings: dict[str, list] = {}
    followup_hits: list[SearchHit] = []
    if followup_queries:
        followup_matrix = np.asarray(
            original["followup_query_embeddings_float32"], dtype=np.float32
        )
        followup_hits, followup_rankings = engine._vector_seed_hits_from_matrix(
            followup_queries,
            followup_matrix,
            followup_anchors,
        )
        seeds = engine._merge_hits(seeds, followup_hits)
    anchors = (
        engine._interleave_anchor_ids(followup_anchors, initial_anchors)
        if followup_queries
        else list(initial_anchors)
    )
    rebuilt = deepcopy(original)
    rebuilt.update(
        {
            "version": 3 if engine.sparse_retrieval_enabled else 1,
            "initial_seed_hits": engine._serialize_hits(
                engine._merge_hits(initial_hits, alias_hits)
            ),
            "followup_seed_hits": engine._serialize_hits(followup_hits),
            "final_seed_hits": engine._serialize_hits(seeds),
            "initial_episode_anchor_ids": initial_anchors,
            "followup_episode_anchor_ids": followup_anchors,
            "episode_anchor_ids": anchors,
            "configuration": engine._replay_configuration(),
            "initial_rankings": initial_rankings,
            "followup_rankings": followup_rankings,
        }
    )
    return rebuilt, {
        "initial": initial_rankings,
        "followup": followup_rankings,
    }


def _summarize(rows: list[dict], modes: list[str]) -> dict:
    summary: dict[str, dict] = {}
    for mode in modes:
        summary[mode] = {}
        for layer in ("candidate_top100", "selected_top20"):
            values = [float(row["modes"][mode][layer]["recall"]) for row in rows]
            summary[mode][layer] = {
                "values": values,
                "mean": sum(values) / len(values) if values else 0.0,
                "minimum": min(values) if values else 0.0,
                "maximum": max(values) if values else 0.0,
                "runs_above_95_percent": sum(value > 0.95 for value in values),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-growth Dense/Sparse retrieval ablation on frozen query plans."
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
        default=Path("validation/sparse-no-growth-replay-audit.json"),
    )
    args = parser.parse_args()

    base = AppConfig.from_env()
    base.database_path = args.database
    base.log_dir = args.output.parent / "sparse-replay-logs"
    base.paragraph.enabled = False
    base.retrieval.growth_max_rounds = 0
    base.retrieval.answer_episode_limit = 20
    base.retrieval.rerank_enabled = False
    app = MemoryApplication(base)
    app.rebuild_indexes()

    configurations: dict[str, AppConfig] = {}
    dense = deepcopy(base)
    dense.retrieval.sparse_enabled = False
    dense.retrieval.graph_max_hops = 0
    configurations["dense_only"] = dense
    hybrid = deepcopy(base)
    hybrid.retrieval.sparse_enabled = True
    hybrid.retrieval.graph_max_hops = 0
    configurations["dense_sparse"] = hybrid
    static = deepcopy(base)
    static.retrieval.sparse_enabled = True
    static.retrieval.graph_max_hops = 3
    configurations["dense_sparse_static"] = static
    engines = {name: _engine(app, config) for name, config in configurations.items()}

    criterion = load_json(args.manifest)["questions"][QUESTION_ID]
    rows: list[dict] = []
    for bundle_path in sorted(args.bundles.glob("block-*/q2-replay-bundle.json")):
        original = load_json(bundle_path)
        mode_rows: dict[str, dict] = {}
        for mode, engine in engines.items():
            bundle, rankings = _rebuild_bundle(engine, original)
            result = engine.replay_retrieval(bundle)
            candidate = [
                int(value) for value in result["candidate_episode_ids"][:100]
            ]
            selected = [int(value) for value in result["episode_ids"]]
            mode_rows[mode] = {
                "candidate_top100": _group_recall(candidate, criterion),
                "selected_top20": _group_recall(selected, criterion),
                "candidate_episode_ids": candidate,
                "selected_episode_ids": selected,
                "atomic_anchor_episode_ids": result["atomic_anchor_episode_ids"],
                "rankings": rankings,
            }
        rows.append({"bundle": str(bundle_path), "modes": mode_rows})

    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_id": QUESTION_ID,
            "growth_max_rounds": 0,
            "paragraph_enabled": False,
            "bundle_count": len(rows),
            "thresholds": {
                "candidate_top100": 0.99,
                "selected_top20": 0.95,
            },
            "summary": _summarize(rows, list(engines)),
            "rows": rows,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
