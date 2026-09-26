"""Fair scheduling must not let duplicates consume another need's opportunity."""
import unittest
from memory_demo.retrieval.recall_frontier import round_robin_sources


class RecallFrontierTests(unittest.TestCase):
    def test_distinct_needs_receive_opportunities_before_one_gets_a_second(self):
        result = round_robin_sources([[1, 2, 3], [4, 5, 6]], {i: i for i in range(1, 7)}, limit=4)
        self.assertEqual([x.episode_id for x in result], [1, 4, 2, 5])
        self.assertEqual([x.need_index for x in result], [0, 1, 0, 1])

    def test_duplicate_episodes_same_source_and_prior_reads_do_not_consume_turns(self):
        result = round_robin_sources([[1, 1, 2, 3], [2, 4, 5]],
                                    {1: 10, 2: 10, 3: 30, 4: 40, 5: 50}, limit=3, excluded_sources={10})
        self.assertEqual([x.source_id for x in result], [30, 40, 50])
        self.assertEqual(result[0].need_rank, 4)

    def test_exhausted_need_does_not_block_other_need_and_unknown_ids_are_skipped(self):
        result = round_robin_sources([[], [99, 1, 2]], {1: 10, 2: 20}, limit=8)
        self.assertEqual([x.source_id for x in result], [10, 20])

    def test_rotating_start_avoids_always_favoring_first_need_under_small_budget(self):
        rankings = [[1, 2], [3, 4], [5, 6]]
        mapping = {i: i for i in range(1, 7)}
        self.assertEqual(round_robin_sources(rankings, mapping, limit=1, start_need=1)[0].need_index, 1)
        self.assertEqual(round_robin_sources(rankings, mapping, limit=1, start_need=5)[0].need_index, 2)

    def test_empty_and_invalid_budgets(self):
        self.assertEqual(round_robin_sources([], {}, limit=4), [])
        self.assertEqual(round_robin_sources([[1]], {1: 1}, limit=0), [])
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                round_robin_sources([], {}, limit=limit)


if __name__ == '__main__':
    unittest.main()
