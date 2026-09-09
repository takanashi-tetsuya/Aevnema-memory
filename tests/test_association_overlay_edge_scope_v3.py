from __future__ import annotations

import unittest

from memory_demo.association_overlay import AssociationDelta, AssociationOverlay


AS_OF = "2026-01-01T00:00:00+00:00"


def _contextual_row(
    association_id: int,
    *,
    anchor_id: int = 7,
    target_id: int,
    context_cue_id: int = 101,
    need_cue_id: int = 202,
) -> dict:
    return {
        "id": association_id,
        "from_type": "episode",
        "from_id": anchor_id,
        "to_id": target_id,
        "association_mode": "contextual_recall",
        "context_cue_id": context_cue_id,
        "need_cue_id": need_cue_id,
        "lifecycle_state": "active",
        "utility_weight": 1.0,
    }


class _AnchorScopedContextualRepository:
    """Small no-I/O repository double that records the v3 lookup contract."""

    def __init__(self, rows: list[dict]):
        self.rows = [dict(row) for row in rows]
        self.calls: list[dict] = []

    def get_contextual_for_prototypes(self, _context_ids, _need_ids, **kwargs):
        self.calls.append(dict(kwargs))
        anchors = {int(value) for value in kwargs["anchor_episode_ids"]}
        return [
            dict(row)
            for row in self.rows
            if int(row["from_id"]) in anchors
        ]


class AssociationOverlayEdgeScopeV3Tests(unittest.TestCase):
    def test_delta_masks_only_created_edge_and_restores_only_reinforced_edge(self):
        created = _contextual_row(11, target_id=80)
        sibling = _contextual_row(12, target_id=81)
        before_reinforced = _contextual_row(13, target_id=82)
        after_reinforced = _contextual_row(13, target_id=999)
        outside_anchor = _contextual_row(14, anchor_id=99, target_id=83)
        repository = _AnchorScopedContextualRepository(
            [created, sibling, after_reinforced, outside_anchor]
        )
        overlay = AssociationOverlay.from_delta(
            repository,
            AssociationDelta(
                created=[{"id": 11, "before": None, "after": created}],
                reinforced=[
                    {
                        "id": 13,
                        "before": before_reinforced,
                        "after": after_reinforced,
                    }
                ],
            ),
        )

        rows = overlay.get_contextual_for_prototypes(
            [101],
            [202],
            anchor_episode_ids=[7],
            evaluation_as_of=AS_OF,
            limit=10,
        )

        # The created treatment edge is absent, but the sibling that shares
        # both cue prototypes remains.  A reinforcement is restored only for
        # its own association id, without affecting that sibling either.
        self.assertEqual([12, 13], [int(row["id"]) for row in rows])
        self.assertEqual(82, int(rows[1]["to_id"]))
        self.assertIsNone(repository.calls[0]["limit"])
        self.assertEqual([7], repository.calls[0]["anchor_episode_ids"])
        self.assertEqual(AS_OF, repository.calls[0]["evaluation_as_of"])

    def test_legacy_cue_mask_arguments_cannot_hide_a_shared_cue_sibling(self):
        hidden = _contextual_row(21, target_id=90)
        sibling = _contextual_row(22, target_id=91)
        repository = _AnchorScopedContextualRepository([hidden, sibling])
        overlay = AssociationOverlay(
            repository,
            hidden_ids=[21],
            hidden_context_cue_ids=[101],
            hidden_need_cue_ids=[202],
        )

        rows = overlay.get_contextual_for_prototypes(
            [101],
            [202],
            anchor_episode_ids=[7],
            evaluation_as_of=AS_OF,
            limit=1,
        )

        self.assertEqual([22], [int(row["id"]) for row in rows])


if __name__ == "__main__":
    unittest.main()
