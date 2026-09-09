from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.associations.traversal import TraversedNode
from memory_demo.retrieval.coverage import (
    aggregate_contributions,
    select_contribution_evidence,
)
from memory_demo.retrieval.contextual_association import ContextualPreTargetProposal
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import RequirementResolution
from memory_demo.types import (
    ContextualSlotHit,
    EvidenceSelectionBudget,
    EvidenceSlot,
    QueryVector,
    QueryVectorBundle,
)


class _RowsRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = {int(row["id"]): dict(row) for row in rows}

    def get_many(self, ids):
        return [dict(self.rows[int(value)]) for value in ids if int(value) in self.rows]


class _ProposalMatcher:
    endpoint_limit = 3

    def __init__(self, proposals) -> None:
        self.proposals = list(proposals)
        self.calls: list[dict] = []

    def match_bundle(self, _bundle, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "hits": [],
            "pre_target_proposals": list(self.proposals),
            "context_hits": [],
            "need_hits": [],
            "external_calls": 0,
        }


def _source(source_id: int, raw_text: str) -> dict:
    return {"id": source_id, "raw_text": raw_text}


def _episode(
    episode_id: int,
    source_id: int,
    *,
    source_key: str,
    spans: list[list[int]],
    quotes: list[str],
) -> dict:
    return {
        "id": episode_id,
        "source_id": source_id,
        "source_key": source_key,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "evidence_origin": "source",
        "generation": 0,
        "epistemic_status": "asserted",
        "evidence_basis": "reasoning_view_nonempty_lines_v1",
        "evidence_quotes_json": json.dumps(quotes),
        "evidence_spans_json": json.dumps(spans),
    }


def _hit(edge_id: int, episode_id: int, slot_id: str, *, score: float = 0.8):
    return ContextualSlotHit(
        association_id=edge_id,
        anchor_episode_id=90,
        target_episode_id=episode_id,
        matched_slot_id=slot_id,
        matched_query_id=f"query-{slot_id}",
        target_support_score=score,
        total_score=score,
    )


def _proposal(edge_id: int, episode_id: int, slot_id: str, rank: int):
    return ContextualPreTargetProposal(
        association_id=edge_id,
        anchor_episode_id=90,
        target_episode_id=episode_id,
        context_cue_id=10,
        need_cue_id=20,
        matched_slot_id=slot_id,
        matched_query_id=f"query-{slot_id}",
        matched_query_role="atomic",
        matched_physical_id=f"physical-{slot_id}",
        embedding_space_id="test-space",
        context_similarity=0.9,
        need_similarity=0.9,
        context_gate_score=0.8,
        need_gate_score=0.8,
        anchor_activation=1.0,
        utility_weight=1.0,
        lifecycle_state="active",
        pre_target_score=0.9,
        proposal_key=f"proposal-{edge_id}",
        rank_before_endpoint_cap=rank,
    )


class EngineContributionSelectorV3Tests(unittest.TestCase):
    def _engine(self, episode_rows: list[dict], source_rows: list[dict]) -> QueryEngine:
        config = AppConfig(model=ModelConfig(embedding_dimension=3))
        config.retrieval.answer_episode_limit = 3
        return QueryEngine(
            config,
            model=object(),
            episode_index=EmbeddingIndex(3),
            concept_index=EmbeddingIndex(3),
            episodes=_RowsRepository(episode_rows),
            concepts=object(),
            sources=_RowsRepository(source_rows),
            associations=object(),
        )

    @staticmethod
    def _episode_view(episode_id: int, score: float) -> dict:
        return {"id": episode_id, "score": score, "source_key": "local-only"}

    def test_source_fact_uses_declared_reasoning_view_coordinates(self):
        raw = "\n".join(
            (
                "[source_key: story/structured.json]",
                "",
                "[record: 1]",
                "[speaker_raw: A]",
                "[script_raw: #na;A;anchor proof]",
                "zh-CN: 锚点证据。",
                "ja: 根拠です。",
                "en: Anchor proof.",
            )
        )
        reasoning_lines = MemoryExtractor._single_pass_source_lines(
            MemoryExtractor.compact_source_for_reasoning(raw)
        )[0]
        quote = "\n".join(reasoning_lines[1:])
        engine = self._engine(
            [
                _episode(
                    1,
                    1,
                    source_key="story/structured.json",
                    spans=[[2, len(reasoning_lines)]],
                    quotes=[quote],
                )
            ],
            [_source(1, raw)],
        )

        fact, reason = engine._v3_source_fact_for_closure(
            episode_id=1,
            episode=engine.episodes.get_many((1,))[0],
            source=engine.sources.get_many((1,))[0],
        )

        self.assertEqual("source_bound", reason)
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertIn(
            f"reasoning-view-nonempty-lines:2-{len(reasoning_lines)}",
            fact.record_span,
        )
        self.assertEqual(
            "sha256:" + hashlib.sha256(quote.encode("utf-8")).hexdigest(),
            fact.raw_span_hash,
        )

    def test_materialization_delivers_persisted_quote_not_keyword_neighbours(self):
        direct = "[record: 99]\n[speaker_raw: Morgan]\nzh-CN: Morgan 明确批准了请求。"
        raw = "\n\n".join(
            [
                "[source_key: story/a.json]\n[segment_index: 0]",
                *[
                    f"[record: {index}]\nzh-CN: Morgan 正在讨论其他事情。"
                    for index in range(1, 100)
                ],
                direct,
            ]
        )
        episode = _episode(
            1,
            1,
            source_key="story/a.json",
            spans=[[1, 3]],
            quotes=[direct],
        )
        episode.update(
            {
                "text": "Morgan 批准了请求。",
                "participants_json": json.dumps(["Morgan"]),
                "segment_index": 0,
                "story_time_text": "",
                "story_order": None,
                "timeline_scope": "",
                "epistemic_note": "",
            }
        )
        engine = self._engine([episode], [_source(1, raw)])
        engine.concepts = _RowsRepository([])

        episodes, _concepts = engine._materialize_nodes(
            [TraversedNode("episode", 1, 0.9)], include_sources=True
        )

        self.assertEqual("source_bound", episodes[0]["source_evidence_delivery"])
        self.assertEqual(1, episodes[0]["source_evidence_quote_count"])
        self.assertIn(direct, episodes[0]["source_text"])

    def test_configured_budget_keeps_base_and_two_additional_routes(self):
        raw_a = "[record: 1]\nA: base proof"
        raw_b = "[record: 1]\nB: second proof"
        raw_c = "[record: 1]\nC: third proof"
        rows = [
            _episode(1, 1, source_key="story/a.json", spans=[[1, 2]], quotes=[raw_a]),
            _episode(2, 2, source_key="story/b.json", spans=[[1, 2]], quotes=[raw_b]),
            _episode(3, 3, source_key="story/c.json", spans=[[1, 2]], quotes=[raw_c]),
        ]
        engine = self._engine(rows, [_source(1, raw_a), _source(2, raw_b), _source(3, raw_c)])
        slots = [
            EvidenceSlot(slot_id="base", question="base", clause_ids=("base",)),
            EvidenceSlot(slot_id="second", question="second", clause_ids=("second",)),
            EvidenceSlot(slot_id="third", question="third", clause_ids=("third",)),
        ]
        contributions, _reasons, _trace = engine._v3_build_candidate_contributions(
            episodes=[
                self._episode_view(1, 0.4),
                self._episode_view(2, 0.2),
                self._episode_view(3, 0.1),
            ],
            slots=slots,
            slot_support={1: {"base"}, 2: {"second"}, 3: {"third"}},
            contextual_hits=(_hit(11, 2, "second"), _hit(12, 3, "third")),
            target_relevance_scores={("second", 2): 0.8, ("third", 3): 0.7},
        )

        selected = select_contribution_evidence(
            aggregate_contributions(contributions),
            slots,
            EvidenceSelectionBudget(episode_limit=3),
        )

        self.assertEqual({1, 2, 3}, set(selected.selected_episode_ids))
        self.assertEqual(frozenset({"base", "second", "third"}), selected.covered_required_clauses)
        self.assertEqual(3, selected.actual_delivery_episode_count)

    def test_path_recovers_two_contextual_targets_beyond_one_base_pool(self):
        """Configured budget is not shrunk to the initial one-episode pool."""

        raw_a = "[record: 1]\nA: base proof"
        raw_b = "[record: 1]\nB: second proof"
        raw_c = "[record: 1]\nC: third proof"
        rows = [
            _episode(1, 1, source_key="story/a.json", spans=[[1, 2]], quotes=[raw_a]),
            _episode(2, 2, source_key="story/b.json", spans=[[1, 2]], quotes=[raw_b]),
            _episode(3, 3, source_key="story/c.json", spans=[[1, 2]], quotes=[raw_c]),
        ]
        matcher = _ProposalMatcher(
            [_proposal(31, 2, "second", 1), _proposal(32, 3, "third", 2)]
        )
        engine = self._engine(
            rows,
            [_source(1, raw_a), _source(2, raw_b), _source(3, raw_c)],
        )
        engine.contextual_matcher = matcher
        engine.config.retrieval.contextual_association_enabled = True
        engine.config.retrieval.contextual_association_shadow = False
        for episode_id in (2, 3):
            engine.episode_index.add(episode_id, [0.0, 1.0, 0.0])
        engine._materialize_nodes = lambda _nodes, include_sources=True: (
            [self._episode_view(2, 0.0), self._episode_view(3, 0.0)],
            [],
        )  # type: ignore[method-assign]
        slots = [
            EvidenceSlot(slot_id="base", question="base", query_id="query-base", clause_ids=("base",)),
            EvidenceSlot(slot_id="second", question="second", query_id="query-second", clause_ids=("second",)),
            EvidenceSlot(slot_id="third", question="third", query_id="query-third", clause_ids=("third",)),
        ]
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=(
                QueryVector(
                    query_id="query-base",
                    text_hash="base-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="base",
                    text="base",
                ),
                QueryVector(
                    query_id="query-second",
                    text_hash="second-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="second",
                    text="second",
                ),
                QueryVector(
                    query_id="query-third",
                    text_hash="third-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="third",
                    text="third",
                ),
            ),
        )

        selected, trace = engine._select_contextual_slots(
            episodes=[self._episode_view(1, 0.4)],
            baseline_selected=[self._episode_view(1, 0.4)],
            rerank_trace={
                "merged_coverage": {
                    "coverage": [
                        {"query": "base", "episode_ids": [1], "clause_ids": ["base"]}
                    ]
                }
            },
            reranked_episode_ids=[1],
            bundle=bundle,
            domain=None,
            endpoint_limit=3,
            anchor_activations={90: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of="2026-02-01T00:00:00+00:00",
            learning_initial_candidate_episode_ids=(1,),
            learning_initial_delivered_episode_ids=(1,),
        )

        selected_ids = [int(item["id"]) for item in selected]
        self.assertEqual({1, 2, 3}, set(selected_ids))
        self.assertEqual(3, len(selected_ids))
        self.assertEqual(selected_ids, trace["treatment_selected_episode_ids"])
        self.assertEqual(3, trace["selected_count"])
        self.assertEqual("contextual_double_key_contribution_selector_v3", trace["backend"])
        self.assertEqual(1, len(matcher.calls))
        counterfactual = trace["contribution_counterfactual"]
        self.assertTrue(counterfactual["candidate_universe_fingerprint"].startswith("sha256:"))
        self.assertTrue(counterfactual["requirements_fingerprint"].startswith("sha256:"))
        self.assertTrue(counterfactual["budget_fingerprint"].startswith("sha256:"))
        self.assertTrue(counterfactual["input_fingerprint"].startswith("sha256:"))
        self.assertEqual(
            trace["treatment_selected_episode_ids"],
            counterfactual["treatment"]["selected_episode_ids"],
        )
        self.assertEqual(
            trace["masked_selected_episode_ids"],
            counterfactual["masked"]["selected_episode_ids"],
        )
        self.assertEqual([31, 32], counterfactual["mask_evaluation"]["masked_edge_ids"])
        self.assertEqual([31, 32], [row["edge_id"] for row in counterfactual["edges"]])
        self.assertTrue(all("single_edge" in row and "leave_one_out" in row for row in counterfactual["edges"]))
        # The selector's complete base route universe remains [1] here, but
        # T13 learning retains the earlier caller-provided query snapshot.
        # A later non-contextual source-bound route can therefore be
        # classified as candidate-missing rather than being silently relabelled
        # as present merely because the final selector saw it.
        self.assertIsNotNone(engine._v3_learning_capture)
        self.assertEqual((1,), engine._v3_learning_capture.initial_candidate_episode_ids)
        self.assertEqual((1,), engine._v3_learning_capture.initial_delivered_episode_ids)
        serialized_trace = repr(trace)
        self.assertNotIn("story/a.json", serialized_trace)
        self.assertNotIn(raw_b, serialized_trace)

    def test_shadow_delivers_exact_masked_result_without_losing_same_episode_base(self):
        raw = "[record: 1]\nA: base proof and contextual candidate"
        rows = [
            _episode(1, 1, source_key="story/private.json", spans=[[1, 2]], quotes=[raw]),
        ]
        engine = self._engine(rows, [_source(1, raw)])
        engine.contextual_matcher = _ProposalMatcher([_proposal(77, 1, "missing", 1)])
        engine.config.retrieval.contextual_association_enabled = True
        engine.config.retrieval.contextual_association_shadow = True
        engine.episode_index.add(1, [0.0, 1.0, 0.0])
        slots = [
            EvidenceSlot(slot_id="base", question="base", query_id="query-base", clause_ids=("base",)),
            EvidenceSlot(slot_id="missing", question="missing", query_id="query-missing", clause_ids=("missing",)),
        ]
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=(
                QueryVector(
                    query_id="query-base",
                    text_hash="base-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="base",
                    text="base",
                ),
                QueryVector(
                    query_id="query-missing",
                    text_hash="missing-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="missing",
                    text="missing",
                ),
            ),
        )

        selected, trace = engine._select_contextual_slots(
            episodes=[self._episode_view(1, 0.4)],
            baseline_selected=[self._episode_view(1, 0.4)],
            rerank_trace={
                "merged_coverage": {
                    "coverage": [
                        {"query": "base", "episode_ids": [1], "clause_ids": ["base"]}
                    ]
                }
            },
            reranked_episode_ids=[1],
            bundle=bundle,
            domain=None,
            endpoint_limit=3,
            anchor_activations={90: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of="2026-02-01T00:00:00+00:00",
        )

        counterfactual = trace["contribution_counterfactual"]
        edge = counterfactual["edges"][0]
        masked_ids = counterfactual["masked"]["selected_episode_ids"]
        treatment_ids = counterfactual["treatment"]["selected_episode_ids"]
        self.assertTrue(trace["shadow"])
        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual(masked_ids, trace["masked_selected_episode_ids"])
        self.assertEqual(masked_ids, [int(item["id"]) for item in selected])
        self.assertEqual([1], masked_ids)
        self.assertEqual([1], treatment_ids)
        self.assertEqual([77], counterfactual["mask_evaluation"]["masked_edge_ids"])
        self.assertEqual(77, edge["edge_id"])
        self.assertTrue(
            set(edge["contribution_ids"])
            <= set(counterfactual["treatment"]["selected_contribution_ids"])
        )
        self.assertFalse(
            set(edge["contribution_ids"])
            & set(counterfactual["masked"]["selected_contribution_ids"])
        )
        self.assertIn("missing", counterfactual["treatment"]["missing_required_clauses"])
        self.assertFalse(edge["sufficient"])
        self.assertFalse(edge["necessary"])
        rendered = repr(counterfactual)
        self.assertNotIn("story/private.json", rendered)
        self.assertNotIn(raw, rendered)

    def test_edge_only_recovered_endpoint_is_removed_from_masked_delivery(self):
        raw_base = "[record: 1]\nA: pre-existing base proof"
        raw_target = "[record: 1]\nB: edge-recovered endpoint"
        rows = [
            _episode(1, 1, source_key="story/base.json", spans=[[1, 2]], quotes=[raw_base]),
            _episode(2, 2, source_key="story/target.json", spans=[[1, 2]], quotes=[raw_target]),
        ]
        engine = self._engine(
            rows,
            [_source(1, raw_base), _source(2, raw_target)],
        )
        engine.contextual_matcher = _ProposalMatcher([_proposal(61, 2, "missing", 1)])
        engine.config.retrieval.contextual_association_enabled = True
        engine.config.retrieval.contextual_association_shadow = True
        engine.episode_index.add(2, [0.0, 1.0, 0.0])
        engine._materialize_nodes = lambda _nodes, include_sources=True: (
            [self._episode_view(2, 0.99)],
            [],
        )  # type: ignore[method-assign]
        slots = [
            EvidenceSlot(slot_id="base", question="base", query_id="query-base", clause_ids=("base",)),
            EvidenceSlot(slot_id="missing", question="missing", query_id="query-missing", clause_ids=("missing",)),
        ]
        requirements = RequirementResolution(
            request_mode="factual",
            status="resolved",
            requirements=tuple(slots),
            planner_origin="explicit",
        )
        bundle = QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=(
                QueryVector(
                    query_id="query-base",
                    text_hash="base-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="base",
                    text="base",
                ),
                QueryVector(
                    query_id="query-missing",
                    text_hash="missing-hash",
                    role="atomic",
                    vector=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    slot_id="missing",
                    text="missing",
                ),
            ),
        )

        selected, trace = engine._select_contextual_slots(
            episodes=[self._episode_view(1, 0.4)],
            baseline_selected=[self._episode_view(1, 0.4)],
            rerank_trace={
                "merged_coverage": {
                    "coverage": [
                        {"query": "base", "episode_ids": [1], "clause_ids": ["base"]}
                    ]
                }
            },
            reranked_episode_ids=[1],
            bundle=bundle,
            domain=None,
            endpoint_limit=3,
            anchor_activations={90: 1.0},
            authoritative_requirements=requirements,
            evaluation_as_of="2026-02-01T00:00:00+00:00",
        )

        counterfactual = trace["contribution_counterfactual"]
        recovered_rows = [
            row for row in trace["merged_candidates"] if row["episode_id"] == 2
        ]
        self.assertTrue(trace["shadow"])
        self.assertEqual([1], trace["base_endpoint_manifest"])
        self.assertEqual([1, 2], counterfactual["treatment"]["selected_episode_ids"])
        self.assertEqual([1], counterfactual["masked"]["selected_episode_ids"])
        self.assertEqual([1], [int(item["id"]) for item in selected])
        self.assertEqual([1], trace["masked_selected_episode_ids"])
        self.assertEqual([61], counterfactual["mask_evaluation"]["masked_edge_ids"])
        self.assertTrue(recovered_rows)
        self.assertTrue(all(row["lane"] == "contextual" for row in recovered_rows))
        self.assertTrue(
            all(row["endpoint_provenance"] == "contextual_edge" for row in recovered_rows)
        )

    def test_same_episode_keeps_base_and_each_accepted_edge_slot_route(self):
        raw = "[record: 1]\nA: one proof\n[record: 2]\nA: another proof"
        rows = [
            _episode(1, 1, source_key="story/one.json", spans=[[1, 2]], quotes=["[record: 1]\nA: one proof"]),
        ]
        engine = self._engine(rows, [_source(1, raw)])
        slots = [EvidenceSlot(slot_id="fact", question="fact", clause_ids=("fact",))]
        contributions, _reasons, trace_rows = engine._v3_build_candidate_contributions(
            episodes=[self._episode_view(1, 0.5)],
            slots=slots,
            slot_support={1: {"fact"}},
            contextual_hits=(_hit(21, 1, "fact"), _hit(22, 1, "fact")),
            target_relevance_scores={("fact", 1): 0.9},
        )

        aggregate = aggregate_contributions(contributions)
        self.assertEqual(1, len(aggregate))
        self.assertEqual(2, len(aggregate[0].contextual_contribution_ids))
        self.assertGreaterEqual(len(aggregate[0].base_contribution_ids), 2)
        self.assertEqual({21, 22}, {row["edge_id"] for row in trace_rows if row.get("edge_id")})

    def test_same_source_distinct_spans_keep_distinct_fact_identities(self):
        raw = "[record: 1]\nA: first fact\n[record: 2]\nB: second fact"
        rows = [
            _episode(1, 1, source_key="story/long.json", spans=[[1, 2]], quotes=["[record: 1]\nA: first fact"]),
            _episode(2, 1, source_key="story/long.json", spans=[[3, 4]], quotes=["[record: 2]\nB: second fact"]),
        ]
        engine = self._engine(rows, [_source(1, raw)])
        slots = [
            EvidenceSlot(slot_id="first", question="first", clause_ids=("first",)),
            EvidenceSlot(slot_id="second", question="second", clause_ids=("second",)),
        ]
        contributions, _reasons, trace_rows = engine._v3_build_candidate_contributions(
            episodes=[self._episode_view(1, 0.4), self._episode_view(2, 0.3)],
            slots=slots,
            slot_support={1: {"first"}, 2: {"second"}},
        )
        aggregates = aggregate_contributions(contributions)
        fact_ids = {
            fact.fact_id
            for aggregate in aggregates
            for fact in aggregate.source_facts
        }

        self.assertEqual(2, len(fact_ids))
        self.assertTrue(all("source_key" not in row for row in trace_rows))

    def test_contextual_relevance_only_route_cannot_cover_a_requirement(self):
        raw = "[record: 1]\nA: source fact"
        rows = [
            _episode(5, 5, source_key="story/context.json", spans=[[1, 2]], quotes=[raw]),
        ]
        engine = self._engine(rows, [_source(5, raw)])
        slots = [EvidenceSlot(slot_id="need", question="need", clause_ids=("need",))]
        contributions, _reasons, trace_rows = engine._v3_build_candidate_contributions(
            episodes=[self._episode_view(5, 0.0)],
            slots=slots,
            slot_support={},
            contextual_hits=(_hit(99, 5, "need", score=0.95),),
            target_relevance_scores={("need", 5): 0.95},
        )
        selected = select_contribution_evidence(
            aggregate_contributions(contributions),
            slots,
            EvidenceSelectionBudget(episode_limit=1),
        )
        contextual_rows = [row for row in trace_rows if row.get("edge_id") == 99]

        self.assertEqual((5,), selected.selected_episode_ids)
        self.assertEqual(frozenset({"need"}), selected.missing_required_clauses)
        self.assertTrue(contextual_rows[0]["ranking_only"])
        self.assertEqual(
            "relevance_only",
            contextual_rows[0]["clause_supports"][0]["verification_status"],
        )


if __name__ == "__main__":
    unittest.main()
