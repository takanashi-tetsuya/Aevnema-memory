from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from memory_demo.types import QueryIntent, SearchHit


def build_frozen_query_plan(engine, question: str) -> dict[str, Any]:
    """Freeze stochastic query decomposition without freezing retrieval hits."""
    intent = engine._parse_intent(question)
    initial_queries = list(dict.fromkeys([question, *intent.search_queries]))
    initial_matrix = np.asarray(engine.model.embed(initial_queries), dtype=np.float32)
    initial_hits, _rankings = engine._vector_seed_hits_from_matrix(
        initial_queries,
        initial_matrix,
        [],
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
    traversed, _paths = engine.traverser.expand(
        seeds,
        engine.config.retrieval.graph_beam_width,
        engine.config.retrieval.graph_max_hops,
    )
    traversed = traversed[: engine.config.retrieval.candidate_limit]
    followup_queries = engine._plan_followup_queries(question, intent, traversed)
    followup_matrix = np.empty(
        (0, engine.config.model.embedding_dimension), dtype=np.float32
    )
    if followup_queries:
        followup_matrix = np.asarray(
            engine.model.embed(followup_queries), dtype=np.float32
        )
    return {
        "version": 1,
        "question": question,
        "intent": asdict(intent),
        "initial_queries": initial_queries,
        "followup_queries": followup_queries,
        "initial_query_embeddings_float32": initial_matrix.tolist(),
        "followup_query_embeddings_float32": followup_matrix.tolist(),
        "planner_configuration": {
            "embedding_dimension": engine.config.model.embedding_dimension,
            "graph_max_hops": engine.config.retrieval.graph_max_hops,
            "sparse_enabled": engine.sparse_retrieval_enabled,
            "paragraph_enabled": engine.paragraph_retrieval_enabled,
        },
    }


def run_frozen_query_plan(
    engine,
    plan: dict[str, Any],
    *,
    episode_limit: int = 20,
    rerank: bool = True,
    rerank_cache: dict[str, tuple[list[int], dict]] | None = None,
) -> dict[str, Any]:
    """Recompute every retrieval channel for one arm using a shared plan."""
    if int(plan.get("version", 0)) != 1:
        raise ValueError("unsupported Stage 11 query plan version")
    question = str(plan["question"])
    intent = QueryIntent.from_dict(dict(plan["intent"]))
    initial_queries = [str(value) for value in plan["initial_queries"]]
    followup_queries = [str(value) for value in plan.get("followup_queries", [])]
    initial_matrix = np.asarray(
        plan["initial_query_embeddings_float32"], dtype=np.float32
    )
    followup_matrix = np.asarray(
        plan.get("followup_query_embeddings_float32", []), dtype=np.float32
    )
    if not followup_queries:
        followup_matrix = np.empty(
            (0, engine.config.model.embedding_dimension), dtype=np.float32
        )

    initial_anchor_ids: list[int] = []
    initial_hits, initial_rankings = engine._vector_seed_hits_from_matrix(
        initial_queries,
        initial_matrix,
        initial_anchor_ids,
        engine.config.retrieval.answer_whole_question_anchor_episodes,
    )
    alias_hits: list[SearchHit] = []
    alias_concept_ids: list[int] = []
    for entity in intent.target_entities:
        for row in engine.concepts.find_by_alias(entity):
            concept_id = int(row["canonical_concept_id"] or row["id"])
            alias_concept_ids.append(concept_id)
            alias_hits.append(SearchHit("concept", concept_id, 1.0))
    seeds = engine._merge_hits(initial_hits, alias_hits)

    followup_anchor_ids: list[int] = []
    followup_rankings: dict[str, list] = {
        "episode": [],
        "concept": [],
        "paragraph": [],
        "paragraph_episode_expansion": [],
    }
    if followup_queries:
        followup_hits, followup_rankings = engine._vector_seed_hits_from_matrix(
            followup_queries,
            followup_matrix,
            followup_anchor_ids,
        )
        seeds = engine._merge_hits(seeds, followup_hits)

    episode_anchor_ids = (
        engine._interleave_anchor_ids(followup_anchor_ids, initial_anchor_ids)
        if followup_queries
        else list(initial_anchor_ids)
    )
    traversed, paths = engine.traverser.expand(
        seeds,
        engine.config.retrieval.graph_beam_width,
        engine.config.retrieval.graph_max_hops,
    )
    traversed = traversed[: engine.config.retrieval.candidate_limit]
    graph_candidate_ids = [
        int(item.node_id) for item in traversed if item.node_type == "episode"
    ]
    episodes, concepts = engine._materialize_nodes(traversed, include_sources=True)
    actual_limit = min(max(0, int(episode_limit)), len(episodes))
    atomic_queries = list(
        dict.fromkeys([*initial_queries, *followup_queries])
    )
    paragraph_rankings = [
        *initial_rankings.get("paragraph", []),
        *followup_rankings.get("paragraph", []),
    ]
    paragraph_context_by_source = engine._paragraph_context_by_source(
        paragraph_rankings
    )
    if rerank:
        preferred_ids, rerank_trace = engine._rerank_answer_episodes(
            question,
            intent,
            atomic_queries,
            episodes,
            actual_limit,
            episode_anchor_ids,
            paragraph_context_by_source,
            rerank_cache,
        )
    else:
        preferred_ids = episode_anchor_ids[:actual_limit]
        rerank_trace = {
            "enabled": False,
            "candidate_episode_ids": graph_candidate_ids[
                : engine.config.retrieval.rerank_candidate_limit
            ],
            "final_episode_ids": preferred_ids,
        }
    coverage_groups = (
        rerank_trace.get("merged_coverage", {}).get("coverage", [])
        if rerank
        else []
    )
    selected_episodes, answer_paths = engine._select_answer_evidence(
        episodes,
        paths,
        actual_limit,
        engine.config.retrieval.answer_path_limit,
        set(),
        question,
        preferred_ids or episode_anchor_ids,
        engine.config.retrieval.learned_bridge_slots,
        engine.config.retrieval.learned_bridge_min_query_relevance,
        engine.config.retrieval.learned_bridge_duplicate_threshold,
        coverage_groups,
        {},
    )

    paragraph_expansions = [
        *initial_rankings.get("paragraph_episode_expansion", []),
        *followup_rankings.get("paragraph_episode_expansion", []),
    ]
    concept_rankings = [
        *initial_rankings.get("concept", []),
        *followup_rankings.get("concept", []),
    ]
    concept_raw_rankings = [
        *initial_rankings.get("concept_raw", []),
        *followup_rankings.get("concept_raw", []),
    ]
    concept_gate_audits = [
        *initial_rankings.get("concept_gate", []),
        *followup_rankings.get("concept_gate", []),
    ]
    selected_ids = [int(item["id"]) for item in selected_episodes]
    selected_id_set = set(selected_ids)
    evidence_episodes = [
        {
            key: item[key]
            for key in (
                "id",
                "score",
                "text",
                "participants",
                "source_key",
                "segment_index",
                "story_time_text",
                "story_order",
                "timeline_scope",
                "evidence_origin",
                "epistemic_status",
                "generation",
                "epistemic_note",
            )
        }
        for item in selected_episodes
    ]
    return {
        "question": question,
        "intent": asdict(intent),
        "initial_queries": initial_queries,
        "followup_queries": followup_queries,
        "candidate_episode_ids": [
            int(value)
            for value in (
                rerank_trace.get("candidate_episode_ids")
                or graph_candidate_ids[
                    : engine.config.retrieval.rerank_candidate_limit
                ]
            )
        ],
        "graph_candidate_episode_ids": graph_candidate_ids,
        "atomic_anchor_episode_ids": episode_anchor_ids,
        "reranked_episode_ids": [int(value) for value in preferred_ids],
        "episode_ids": selected_ids,
        "concept_ids": [int(item["id"]) for item in concepts],
        "evidence_episodes": evidence_episodes,
        "association_paths": answer_paths,
        "rerank_trace": rerank_trace,
        "alias_concept_ids": list(dict.fromkeys(alias_concept_ids)),
        "concept_ranking_ids": [
            [int(item["id"]) for item in ranking] for ranking in concept_rankings
        ],
        "concept_raw_ranking_ids": [
            [int(item["id"]) for item in ranking]
            for ranking in concept_raw_rankings
        ],
        "concept_gate_audits": concept_gate_audits,
        "paragraph_ranking_ids": [
            [int(item["id"]) for item in ranking] for ranking in paragraph_rankings
        ],
        "paragraph_episode_expansion_ids": [
            [int(item["episode_id"]) for item in ranking]
            for ranking in paragraph_expansions
        ],
        "selected_paragraph_expansion_episode_ids": sorted(
            selected_id_set.intersection(
                int(item["episode_id"])
                for ranking in paragraph_expansions
                for item in ranking
            )
        ),
        "paragraph_retrieval_enabled": engine.paragraph_retrieval_enabled,
        "paragraph_context_source_ids": sorted(paragraph_context_by_source),
        "sparse_retrieval_enabled": engine.sparse_retrieval_enabled,
    }
