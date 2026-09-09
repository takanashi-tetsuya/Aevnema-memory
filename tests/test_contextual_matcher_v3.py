from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.association_overlay import AssociationOverlay, StagedAssociationOverlay
from memory_demo.associations.plasticity import derive_contextual_candidates
from memory_demo.config import AppConfig, ModelConfig, WeightConfig
from memory_demo.embeddings import EmbeddingIndex
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.types import QueryVector, QueryVectorBundle


AS_OF = "2026-01-01T00:00:00+00:00"


def _row(
    association_id: int,
    *,
    anchor_id: int = 1,
    target_id: int = 2,
    context_id: int = 10,
    need_id: int = 20,
    utility: float = 1.0,
) -> dict:
    return {
        "id": association_id,
        "from_type": "episode",
        "from_id": anchor_id,
        "to_id": target_id,
        "association_mode": "contextual_recall",
        "context_cue_id": context_id,
        "need_cue_id": need_id,
        "lifecycle_state": "active",
        "utility_weight": utility,
        "cue_domain": "knowledge",
    }


class _AnchorAwareRepo:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.calls: list[tuple[object, object, dict]] = []

    def get_contextual_for_prototypes(self, context_ids, need_ids, **kwargs):
        self.calls.append((context_ids, need_ids, dict(kwargs)))
        anchors = kwargs.get("anchor_episode_ids")
        if anchors is None:
            raise AssertionError("v3 matcher must supply active anchor ids")
        if kwargs.get("evaluation_as_of") is None:
            raise AssertionError("v3 matcher must supply evaluation_as_of")
        allowed = {int(value) for value in anchors}
        return [
            dict(row) for row in self.rows if int(row["from_id"]) in allowed
        ]


class _OverlayRepo:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.calls: list[dict] = []
        self.weights = WeightConfig()

    def get_contextual_for_prototypes(self, _context_ids, _need_ids, **kwargs):
        self.calls.append(dict(kwargs))
        anchors = kwargs.get("anchor_episode_ids")
        if anchors is None:
            return [dict(row) for row in self.rows]
        allowed = {int(value) for value in anchors}
        return [
            dict(row) for row in self.rows if int(row["from_id"]) in allowed
        ]


class ContextualMatcherV3Tests(unittest.TestCase):
    @staticmethod
    def _indexes():
        context = EmbeddingIndex(3)
        need = EmbeddingIndex(3)
        context.add(10, [1.0, 0.0, 0.0])
        need.add(20, [0.0, 1.0, 0.0])
        return context, need

    @staticmethod
    def _bundle(*queries: QueryVector) -> QueryVectorBundle:
        return QueryVectorBundle(
            model_id="test",
            dimension=3,
            whole=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            queries=tuple(queries),
        )

    def test_same_need_prototype_retains_each_slot_and_logical_binding(self):
        context, need = self._indexes()
        repo = _AnchorAwareRepo([_row(17)])
        matcher = ContextualAssociationMatcher(
            context, need, repo, context_threshold=0.5, need_threshold=0.5,
            endpoint_limit=8,
        )
        bundle = self._bundle(
            QueryVector(
                "atomic-a", "same-need", "atomic",
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32), slot_id="slot-a",
            ),
            QueryVector(
                "followup-b", "same-need", "followup",
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32), slot_id="slot-b",
            ),
        )

        result = matcher.match_bundle(
            bundle,
            domain="knowledge",
            active_anchor_ids={1: 1.0},
            unresolved_slot_ids=["slot-a", "slot-b"],
            evaluation_as_of=AS_OF,
        )

        proposals = result["pre_target_proposals"]
        self.assertEqual(2, len(proposals))
        self.assertEqual({"slot-a", "slot-b"}, {item.matched_slot_id for item in proposals})
        self.assertEqual({"atomic", "followup"}, {item.matched_query_role for item in proposals})
        self.assertEqual(1, len({item.matched_physical_id for item in proposals}))
        self.assertEqual([1, 2], [item.rank_before_endpoint_cap for item in proposals])
        self.assertEqual(2, len({item.proposal_key for item in proposals}))
        self.assertEqual(2, len(result["hits"]))
        self.assertIsNone(repo.calls[0][0])
        self.assertIsNone(repo.calls[0][1])

    def test_anchor_first_ignores_global_unrelated_prototype_top_k(self):
        context = EmbeddingIndex(3)
        need = EmbeddingIndex(3)
        # IDs 99/88 are globally best but belong only to an unrelated anchor.
        context.add(99, [1.0, 0.0, 0.0])
        need.add(88, [0.0, 1.0, 0.0])
        context.add(10, [0.8, 0.6, 0.0])
        need.add(20, [0.0, 0.8, 0.6])
        repo = _AnchorAwareRepo([
            _row(1, anchor_id=1, target_id=2, context_id=10, need_id=20),
            _row(2, anchor_id=999, target_id=3, context_id=99, need_id=88),
        ])
        matcher = ContextualAssociationMatcher(
            context, need, repo, context_threshold=0.5, need_threshold=0.5,
            context_top_k=1, need_top_k=1,
        )
        bundle = self._bundle(
            QueryVector(
                "q", "need", "atomic", np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id="slot",
            )
        )

        result = matcher.match_bundle(
            bundle,
            active_anchor_ids={1: 1.0},
            unresolved_slot_ids=["slot"],
            evaluation_as_of=AS_OF,
        )

        self.assertEqual([1], [item.association_id for item in result["pre_target_proposals"]])
        self.assertEqual([2], [item.target_episode_id for item in result["hits"]])
        self.assertEqual((1,), repo.calls[0][2]["anchor_episode_ids"])
        self.assertIsNone(repo.calls[0][2]["limit"])
        self.assertTrue(result["gate_trace"]["repository_anchor_filter_applied"])

    def test_pre_target_proposals_are_uncapped_and_deterministically_ranked(self):
        context, need = self._indexes()
        repo = _AnchorAwareRepo([
            _row(21, target_id=2, utility=0.9),
            _row(22, target_id=3, utility=0.8),
            _row(23, target_id=4, utility=0.7),
        ])
        matcher = ContextualAssociationMatcher(
            context, need, repo, context_threshold=0.5, need_threshold=0.5,
            endpoint_limit=1,
        )
        bundle = self._bundle(
            QueryVector(
                "q", "need", "atomic", np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id="slot",
            )
        )

        first = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["slot"],
            evaluation_as_of=AS_OF,
        )
        second = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["slot"],
            evaluation_as_of=AS_OF,
        )

        self.assertEqual(3, len(first["pre_target_proposals"]))
        self.assertEqual(1, len(first["hits"]))
        self.assertEqual(
            [1, 2, 3],
            [item.rank_before_endpoint_cap for item in first["pre_target_proposals"]],
        )
        self.assertEqual(
            [item.proposal_key for item in first["pre_target_proposals"]],
            [item.proposal_key for item in second["pre_target_proposals"]],
        )

    def test_none_empty_and_missing_timestamp_have_distinct_no_query_results(self):
        context, need = self._indexes()
        repo = _AnchorAwareRepo([_row(17)])
        matcher = ContextualAssociationMatcher(context, need, repo)
        bundle = self._bundle(
            QueryVector(
                "q", "need", "atomic", np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id="slot",
            )
        )

        invalid = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=None,
            evaluation_as_of=AS_OF,
        )
        empty = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=[],
            evaluation_as_of=AS_OF,
        )
        missing_as_of = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["slot"],
        )

        self.assertEqual("invalid_unresolved_slot_contract", invalid["reason"])
        self.assertTrue(invalid["invalid_request_contract"])
        self.assertEqual("no_unresolved_slots", empty["reason"])
        self.assertTrue(empty["residual_mode"])
        self.assertEqual("invalid_evaluation_as_of", missing_as_of["reason"])
        self.assertEqual([], repo.calls)

    def test_missing_slot_vector_and_embedding_space_mismatch_fail_closed(self):
        context, need = self._indexes()
        repo = _AnchorAwareRepo([_row(17)])
        bundle = self._bundle(
            QueryVector(
                "other", "need", "atomic", np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id="other-slot",
            )
        )
        matcher = ContextualAssociationMatcher(context, need, repo)
        missing = matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["wanted-slot"],
            evaluation_as_of=AS_OF,
        )
        mismatch_matcher = ContextualAssociationMatcher(
            context, need, repo, embedding_space_id="different-space",
        )
        mismatch = mismatch_matcher.match_bundle(
            bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["other-slot"],
            evaluation_as_of=AS_OF,
        )

        self.assertEqual("missing_slot_vector", missing["reason"])
        self.assertEqual(["wanted-slot"], missing["missing_slot_ids"])
        self.assertEqual("embedding_space_mismatch", mismatch["reason"])
        self.assertEqual([], repo.calls)

    def test_concurrent_endpoint_limits_do_not_mutate_shared_matcher(self):
        context, need = self._indexes()
        repo = _AnchorAwareRepo([_row(100 + index, target_id=10 + index) for index in range(8)])
        matcher = ContextualAssociationMatcher(
            context, need, repo, context_threshold=0.5, need_threshold=0.5,
            endpoint_limit=2,
        )
        bundle = self._bundle(
            QueryVector(
                "q", "need", "atomic", np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                slot_id="slot",
            )
        )

        def invoke(limit: int) -> tuple[int, int]:
            result = matcher.match_bundle(
                bundle, active_anchor_ids={1: 1.0}, unresolved_slot_ids=["slot"],
                endpoint_limit=limit, evaluation_as_of=AS_OF,
            )
            return len(result["hits"]), len(result["pre_target_proposals"])

        with ThreadPoolExecutor(max_workers=2) as executor:
            short, long = list(executor.map(invoke, (1, 8)))

        self.assertEqual((1, 8), short)
        self.assertEqual((8, 8), long)
        self.assertEqual(2, matcher.endpoint_limit)

    def test_repository_v3_filters_anchor_before_limit_and_replay_time_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            app = MemoryApplication(
                AppConfig(
                    database_path=Path(directory) / "memory.db",
                    log_dir=Path(directory) / "logs",
                    model=ModelConfig(embedding_dimension=3),
                )
            )
            with app.db.transaction() as connection:
                source_id = connection.execute(
                    "INSERT INTO source(raw_text) VALUES('x')"
                ).lastrowid
                for episode_id in range(1, 5):
                    connection.execute(
                        """
                        INSERT INTO episode(
                            id, source_id, source_key, segment_index, text,
                            embedding, created_at, updated_at
                        ) VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (
                            episode_id, source_id, f"episode-{episode_id}", 0,
                            str(episode_id), np.asarray([1, 0, 0], dtype=np.float32).tobytes(),
                            AS_OF, AS_OF,
                        ),
                    )
            context_id = app.associations.get_or_create_cue_prototype(
                domain="knowledge", cue_kind="context", model_id="test", dimension=3,
                vector=[1, 0, 0], text_hash="context",
            )
            need_id = app.associations.get_or_create_cue_prototype(
                domain="knowledge", cue_kind="need", model_id="test", dimension=3,
                vector=[0, 1, 0], text_hash="need",
            )
            valid = derive_contextual_candidates(
                base_episode_ids=[1], selected_episode_ids=[1, 2], target_episode_ids=[2],
                context_query_id="c", need_query_id="n", anchor_id=1,
            )[0]
            unrelated = derive_contextual_candidates(
                base_episode_ids=[3], selected_episode_ids=[3, 4], target_episode_ids=[4],
                context_query_id="c", need_query_id="n", anchor_id=3,
            )[0]
            valid_id = app.associations.create_contextual(
                valid, context_cue_id=context_id, need_cue_id=need_id, utility_weight=0.1,
            )
            app.associations.create_contextual(
                unrelated, context_cue_id=context_id, need_cue_id=need_id, utility_weight=1.0,
            )

            rows = app.associations.get_contextual_for_prototypes(
                None, None, domain="knowledge", anchor_episode_ids=[1],
                evaluation_as_of="2026-01-01T00:00:00Z", limit=1,
            )
            repeated = app.associations.get_contextual_for_prototypes(
                None, None, domain="knowledge", anchor_episode_ids=[1],
                evaluation_as_of=AS_OF, limit=1,
            )

            self.assertEqual([valid_id], [int(row["id"]) for row in rows])
            self.assertEqual([valid_id], [int(row["id"]) for row in repeated])

    def test_read_only_and_staged_overlays_keep_v3_anchor_scope_after_filtering(self):
        hidden = _row(1, anchor_id=1, target_id=2)
        visible = _row(2, anchor_id=1, target_id=3)
        repo = _OverlayRepo([hidden, visible, _row(3, anchor_id=99, target_id=4)])
        overlay = AssociationOverlay(repo, hidden_ids=[1])
        rows = overlay.get_contextual_for_prototypes(
            None, None, anchor_episode_ids=[1], evaluation_as_of=AS_OF, limit=1,
        )
        self.assertEqual([2], [int(row["id"]) for row in rows])
        self.assertIsNone(repo.calls[0]["limit"])

        staged_repo = _OverlayRepo([])
        staged = StagedAssociationOverlay(staged_repo)
        staged._rows[-1] = {
            **_row(-1, anchor_id=1, target_id=5),
            "expires_at": "2026-02-01T00:00:00+00:00",
        }
        staged_rows = staged.get_contextual_for_prototypes(
            None, None, anchor_episode_ids=[1], evaluation_as_of=AS_OF, limit=None,
        )
        self.assertEqual([-1], [int(row["id"]) for row in staged_rows])
        self.assertIsNone(staged_repo.calls[0]["limit"])


if __name__ == "__main__":
    unittest.main()
