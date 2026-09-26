from __future__ import annotations

import json
import math
import unittest

from memory_demo.associations.spreading import (
    SpreadingEdge,
    SpreadingRecall,
    SpreadingSeed,
)


def edge(identifier: int, source: int, target: int, weight: float = 1.0):
    return SpreadingEdge(identifier, "episode", source, "episode", target, weight)


def recall(count: int, edges, **kwargs):
    kwargs.setdefault("monotonic", lambda: 0.0)
    return SpreadingRecall([("episode", i) for i in range(1, count + 1)], edges, **kwargs)


def node(result, identifier):
    return next(item for item in result.nodes if item.node_id == identifier)


class SpreadingRecallTests(unittest.TestCase):
    def test_independent_weak_clues_combine_and_keep_both_provenance_paths(self):
        graph = recall(3, [edge(1, 1, 3, 0.4), edge(2, 2, 3, 0.4)],
                       damping=1.0, degree_exponent=0.0)
        first = graph.search([SpreadingSeed("episode", 1)])
        both = graph.search([SpreadingSeed("episode", 1), SpreadingSeed("episode", 2)])
        self.assertAlmostEqual(node(first, 3).activation, 0.4)
        self.assertAlmostEqual(node(both, 3).activation, 0.64)
        self.assertEqual(node(both, 3).paths, {"episode:1": [1], "episode:2": [2]})

    def test_same_source_clues_do_not_count_as_independent_support(self):
        graph = recall(3, [edge(1, 1, 3, 0.4), edge(2, 2, 3, 0.4)],
                       damping=1.0, degree_exponent=0.0)
        result = graph.search([
            SpreadingSeed("episode", 1, root_id="source:7"),
            SpreadingSeed("episode", 2, root_id="source:7"),
            SpreadingSeed("episode", 1, root_id="source:7"),
        ])
        self.assertAlmostEqual(node(result, 3).activation, 0.4)
        self.assertEqual(len(node(result, 3).contributions), 1)

    def test_converging_clues_prioritize_exploration_over_a_stronger_single_clue(self):
        graph = recall(7, [edge(1, 1, 4, 0.6), edge(2, 2, 4, 0.6),
                           edge(3, 3, 5, 0.75), edge(4, 4, 6), edge(5, 5, 7)],
                       damping=1.0, degree_exponent=0.0)
        result = graph.search([SpreadingSeed("episode", i) for i in (1, 2, 3)],
                              max_expansions=4)
        self.assertGreater(node(result, 4).activation, node(result, 5).activation)
        self.assertIn(4, result.explored_edge_ids)
        self.assertNotIn(5, result.explored_edge_ids)

    def test_later_stronger_support_revisits_an_already_expanded_node(self):
        graph = recall(5, [edge(1, 1, 3, 0.2), edge(2, 2, 3, 0.8),
                           edge(3, 1, 4, 0.6), edge(4, 4, 3, 0.6), edge(5, 3, 5)],
                       damping=1.0, degree_exponent=0.0)
        seeds = [SpreadingSeed("episode", 1, 0.9, "a"), SpreadingSeed("episode", 2, 1, "b")]
        result = graph.search(seeds)
        self.assertAlmostEqual(node(result, 3).contributions["a"], 0.9 * 0.6 * 0.6)
        self.assertAlmostEqual(node(result, 5).contributions["a"], 0.9 * 0.6 * 0.6)
        self.assertEqual(node(result, 5).paths["a"], [3, 4, 5])

    def test_cycle_and_parallel_paths_never_amplify_one_root(self):
        graph = recall(3, [edge(1, 1, 2), edge(2, 2, 3), edge(3, 3, 1),
                           edge(4, 1, 2), edge(5, 2, 2)],
                       damping=1.0, degree_exponent=0.0)
        result = graph.search([SpreadingSeed("episode", 1, 0.3)], max_expansions=100)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.expansions, 8)
        self.assertEqual(result.explored_edge_ids, [1, 2, 3, 4])
        for item in result.nodes:
            self.assertAlmostEqual(item.activation, 0.3)
            self.assertEqual(len(item.contributions), 1)
        self.assertEqual(node(result, 1).paths, {"episode:1": []})

    def test_hub_degree_dampens_generic_nodes_without_parallel_edge_penalty(self):
        edges = [edge(1, 1, 2, 0.8), edge(2, 1, 3, 0.8)]
        edges.extend(edge(i, 2, i + 1) for i in range(3, 7))
        graph = recall(7, edges, damping=1.0)
        result = graph.search([SpreadingSeed("episode", 1)])
        self.assertAlmostEqual(node(result, 2).activation, 0.8 / math.sqrt(5))
        self.assertAlmostEqual(node(result, 3).activation, 0.8)
        duplicate = recall(7, [*edges, edge(7, 1, 2, 0.8)], damping=1.0)
        self.assertAlmostEqual(node(duplicate.search([SpreadingSeed("episode", 1)]), 2).activation,
                               node(result, 2).activation)

    def test_weak_frontier_survives_many_batches_and_matches_uninterrupted_search(self):
        graph = recall(9, [edge(i, i, i + 1, 0.5) for i in range(1, 9)])
        seeds = [SpreadingSeed("episode", 1), SpreadingSeed("episode", 9, 0.3)]
        expected = graph.search(seeds)
        result = graph.search(seeds, max_expansions=1)
        self.assertEqual(result.status, "expansion_budget")
        self.assertGreater(result.pending_count, 0)
        attempts = 0
        while result.status != "complete":
            # A new object and JSON round trip exercise actual persistence.
            graph = recall(9, [edge(i, i, i + 1, 0.5) for i in reversed(range(1, 9))])
            result = graph.search(seeds, max_expansions=2,
                                  checkpoint=json.loads(json.dumps(result.checkpoint)))
            attempts += 1
            self.assertLess(attempts, 100)
        self.assertEqual(result.to_dict(), expected.to_dict())
        self.assertGreater(node(result, 9).contributions["episode:1"], 0)
        self.assertEqual(len(node(result, 9).paths["episode:1"]), 8)

    def test_single_step_budget_can_pause_in_the_middle_of_a_hub(self):
        graph = recall(101, [edge(i, 1, i + 1) for i in range(1, 101)])
        result = graph.search([SpreadingSeed("episode", 1)], max_expansions=1)
        self.assertEqual(result.expansions, 1)
        self.assertEqual(len(result.explored_edge_ids), 1)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(result.status, "expansion_budget")

    def test_weight_reinforcement_prioritizes_the_relevant_edge(self):
        seeds = [SpreadingSeed("episode", 1)]
        before = recall(3, [edge(1, 1, 2, 0.2), edge(2, 1, 3, 0.8)])
        after = recall(3, [edge(1, 1, 2, 0.9), edge(2, 1, 3, 0.8)])
        self.assertEqual(before.search(seeds, max_expansions=1).explored_edge_ids, [2])
        self.assertEqual(after.search(seeds, max_expansions=1).explored_edge_ids, [1])

    def test_checkpoint_rejects_changed_graph_inputs_or_settings(self):
        graph = recall(3, [edge(1, 1, 2), edge(2, 2, 3)])
        seeds = [SpreadingSeed("episode", 1)]
        checkpoint = graph.search(seeds, max_expansions=1).checkpoint
        changed = recall(3, [edge(1, 1, 2, 0.8), edge(2, 2, 3)])
        with self.assertRaisesRegex(ValueError, "graph fingerprint"):
            changed.search(seeds, checkpoint=checkpoint)
        with self.assertRaisesRegex(ValueError, "seeds or propagation settings"):
            graph.search([SpreadingSeed("episode", 1, 0.5)], checkpoint=checkpoint)
        changed = recall(3, [edge(1, 1, 2), edge(2, 2, 3)], damping=0.7)
        with self.assertRaisesRegex(ValueError, "seeds or propagation settings"):
            changed.search(seeds, checkpoint=checkpoint)

    def test_checkpoint_detects_corruption(self):
        graph = recall(2, [edge(1, 1, 2)])
        seeds = [SpreadingSeed("episode", 1)]
        checkpoint = graph.search(seeds, max_expansions=1).checkpoint
        checkpoint["supports"][0]["activation"] = 0.123
        with self.assertRaisesRegex(ValueError, "checksum"):
            graph.search(seeds, checkpoint=checkpoint)

    def test_injected_clock_stops_search_and_retains_work(self):
        ticks = iter(range(100))
        graph = recall(5, [edge(i, i, i + 1) for i in range(1, 5)],
                       monotonic=lambda: next(ticks))
        seeds = [SpreadingSeed("episode", 1)]
        partial = graph.search(seeds, max_seconds=3)
        self.assertEqual(partial.status, "time_budget")
        self.assertEqual(partial.expansions, 2)
        self.assertGreater(partial.pending_count, 0)
        resumed = graph.search(seeds, checkpoint=partial.checkpoint)
        self.assertEqual(resumed.status, "complete")
        self.assertEqual(len(resumed.nodes), 5)
        self.assertGreater(resumed.elapsed_seconds, partial.elapsed_seconds)

    def test_no_budget_does_not_sleep_or_apply_wall_clock_decay(self):
        edges = [edge(1, 1, 2, 0.5)]
        early = recall(2, edges, monotonic=lambda: 1.0)
        later = recall(2, edges, monotonic=lambda: 1_000_000.0)
        seeds = [SpreadingSeed("episode", 1)]
        self.assertEqual(early.search(seeds).to_dict(), later.search(seeds).to_dict())

    def test_relation_taxonomy_is_ignored_and_optional_confidence_is_supported(self):
        row = {"id": 1, "from_type": "episode", "from_id": 1,
               "to_type": "episode", "to_id": 2, "weight": 0.5,
               "relation_type": "due_to", "polarity": -1}
        first = recall(2, [row], damping=1.0)
        second = recall(2, [{**row, "relation_type": "is_a", "polarity": 1}], damping=1.0)
        seeds = [SpreadingSeed("episode", 1)]
        self.assertEqual(first.search(seeds).to_dict(), second.search(seeds).to_dict())
        uncertain = recall(2, [{**row, "confidence": 0.2}], damping=1.0)
        self.assertAlmostEqual(node(uncertain.search(seeds), 2).activation, 0.1)

    def test_zero_budgets_empty_graph_and_isolated_seed(self):
        self.assertEqual(recall(0, []).search([]).status, "complete")
        self.assertEqual(recall(1, []).search([SpreadingSeed("episode", 1)]).pending_count, 0)
        graph = recall(2, [edge(1, 1, 2)])
        seeds = [SpreadingSeed("episode", 1)]
        self.assertEqual(graph.search(seeds, max_expansions=0).status, "expansion_budget")
        self.assertEqual(graph.search(seeds, max_seconds=0).status, "time_budget")

    def test_invalid_inputs_cannot_create_amplifying_or_nonfinite_edges(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(weight=value), self.assertRaises(ValueError):
                recall(2, [edge(1, 1, 2, value)])
        with self.assertRaisesRegex(ValueError, "missing node"):
            recall(2, [edge(1, 1, 3)])
        with self.assertRaisesRegex(ValueError, "duplicate edge"):
            recall(2, [edge(1, 1, 2), edge(1, 1, 2)])
        graph = recall(2, [edge(1, 1, 2)])
        with self.assertRaises(ValueError):
            graph.search([SpreadingSeed("episode", 1)], max_expansions=-1)


if __name__ == "__main__":
    unittest.main()
