"""Batched verdicts cannot promote evidence or partially commit bad targets."""
from copy import deepcopy
import unittest

from config.prompt_config.recall_v6_prompts import NEEDS
from memory_demo.retrieval.recall_review_v3 import ReviewStageError
from memory_demo.retrieval.recall_review_v5 import RecordReviewV5
from memory_demo.retrieval.recall_review_v6 import RecordReviewV6
import test_recall_review_v4 as fixtures


class RecordReviewV6Tests(unittest.TestCase):
    def setUp(self):
        fixtures.RecordReviewV4Tests.setUp(self)
        fixtures.RecordReviewV4Tests.accept(self)
        self.response = {'assessments': [
            {'need_index': 0, 'status': 'supported', 'fact_ids': ['F1'],
             'answer': '甲批准，仅限当晚。', 'reason': '批准和限定均在原文。'},
            {'need_index': 1, 'status': 'unknown', 'fact_ids': [],
             'answer': '', 'reason': '没有丙回来的日期。'}]}
        self.reviewer = RecordReviewV6(self.sources, self.episodes, self.batch_call)

    call = fixtures.RecordReviewV4Tests.call
    commit = fixtures.RecordReviewV4Tests.commit

    def batch_call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        self.assertEqual(system, NEEDS)
        return deepcopy(self.response)

    def test_independent_targets_share_complete_context_and_keep_evidence_unchanged(self):
        before = deepcopy(self.state)
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(self.state, before)
        self.assertNotIn('facts', result.updates)
        self.assertNotIn('verified_links', result.updates)
        self.assertEqual(result.updates['resolved_needs'], [0])
        self.assertEqual(result.updates['need_assessments'][1]['status'], 'unknown')
        payload = result.trace['request']
        self.assertEqual(payload['request_context'], self.state['question'])
        self.assertEqual([t['question'] for t in payload['targets']], self.state['needs'])
        self.assertEqual(len(payload['records']), 5)
        self.assertEqual(len(payload['facts'][0]['record_ids']), 5)
        control = RecordReviewV5(self.sources, self.episodes, self.call).review_need(self.state, 0)
        self.assertEqual(payload['records'], control.trace['request']['records'])
        self.assertEqual(payload['facts'], control.trace['request']['facts'])
        self.assertEqual(result.trace['version'], 6)

    def test_invalid_row_rejects_the_entire_batch_atomically(self):
        original = deepcopy(self.response)
        cases = [[], original['assessments'][:1], original['assessments'] * 2]
        for field, value in [('need_index', True), ('need_index', 99), ('fact_ids', ['F999']),
                             ('status', 'complete'), ('status', []), ('extra', 'unexpected')]:
            rows = deepcopy(original['assessments'])
            rows[1][field] = value
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows):
                self.response = {'assessments': rows}
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError) as caught:
                    self.reviewer.review_needs(self.state, [0, 1])
                self.assertEqual(self.state, before)
                self.assertEqual(caught.exception.trace['response'], self.response)

    def test_supported_and_refuted_require_accepted_citations(self):
        for status in ('supported', 'refuted'):
            self.response['assessments'][1]['status'] = status
            with self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(self.state, [0, 1])

    def test_reassignment_is_allowed_but_pending_facts_are_never_accepted_citations(self):
        self.response['assessments'][1].update(status='supported', fact_ids=['F1'],
            answer='仅用于检验可重新指向未映射需求。')
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(result.updates['resolved_needs'], [0, 1])
        self.state['pending_facts'] = self.state.pop('facts')
        self.state['facts'] = []
        count = len(self.calls)
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(self.calls), count)
        self.assertEqual(result.updates['resolved_needs'], [])

    def test_changed_source_or_excess_context_cannot_be_hidden_by_batching(self):
        count = len(self.calls)
        self.state['review_context_max_chars'] = 1
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0, 1])
        self.state.pop('review_context_max_chars')
        self.sources[1] += 'changed'
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(self.calls), count)

    def test_invalid_target_sets_fail_before_provider(self):
        count = len(self.calls)
        for indices in ([], [0, 0], [True], [2], '0'):
            with self.subTest(indices=indices), self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(self.state, indices)
        self.assertEqual(len(self.calls), count)


if __name__ == '__main__':
    unittest.main()
