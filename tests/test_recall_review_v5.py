"""Optional cue isolation never repairs or certifies the required facts."""
from copy import deepcopy
import unittest

from memory_demo.retrieval.recall_review_v3 import ReviewStageError
from memory_demo.retrieval.recall_review_v4 import RecordReviewV4
from memory_demo.retrieval.recall_review_v5 import RecordReviewV5
import test_recall_review_v4 as fixtures


class RecordReviewV5Tests(unittest.TestCase):
    def setUp(self):
        fixtures.RecordReviewV4Tests.setUp(self)

    call = fixtures.RecordReviewV4Tests.call

    def test_mixed_cues_keep_facts_and_raw_response_unchanged(self):
        response = self.call(fixtures.MAP_SYSTEM, {})
        response['followup_cues'] = [
            {'record_id': 'R1', 'cue': self.records[1][0]['text']},
            {'record_id': 'R1', 'cue': '原文不存在的推測'},
            {'record_id': 'R999', 'cue': '未知記錄'},
            {'record_id': [], 'cue': '錯誤格式'},
        ]
        original = deepcopy(response)
        callback = lambda system, payload: deepcopy(response)
        before = deepcopy(self.state)
        with self.assertRaises(ReviewStageError):
            RecordReviewV4(self.sources, self.episodes, callback).map(self.records[1], self.state)
        result = RecordReviewV5(self.sources, self.episodes, callback).map(self.records[1], self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(result.trace['response'], original)
        self.assertEqual(result.trace['admitted_response']['facts'], original['facts'])
        self.assertEqual(result.trace['admitted_response']['links'], original['links'])
        self.assertEqual(len(result.trace['rejected_optional_cues']), 3)
        self.assertEqual(len(result.followup_cues), 1)
        self.assertEqual(len(result.updates['pending_facts']), 1)
        self.assertNotIn('facts', result.updates)

    def test_malformed_optional_container_is_isolated(self):
        for cues in (None, 'invalid', {'record_id': 'R1', 'cue': 'x'}):
            with self.subTest(cues=cues):
                response = self.call(fixtures.MAP_SYSTEM, {})
                response['followup_cues'] = cues
                result = RecordReviewV5(self.sources, self.episodes, lambda *_: response).map(self.records[1], self.state)
                self.assertEqual(len(result.updates['pending_facts']), 1)
                self.assertEqual(result.followup_cues, [])

    def test_invalid_required_fact_still_rejects_entire_stage(self):
        response = self.call(fixtures.MAP_SYSTEM, {})
        response['facts'][0]['record_ids'] = ['R999']
        response['followup_cues'] = [{'record_id': 'R1', 'cue': 'not present'}]
        before = deepcopy(self.state)
        with self.assertRaises(ReviewStageError) as caught:
            RecordReviewV5(self.sources, self.episodes, lambda *_: response).map(self.records[1], self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(caught.exception.trace['response'], response)
        self.assertFalse(caught.exception.trace['schema_ok'])

    def test_timeout_retains_exact_request_and_record_mapping(self):
        requests = []
        def timeout(system, payload):
            requests.append(deepcopy(payload))
            raise TimeoutError('synthetic provider deadline')
        with self.assertRaises(ReviewStageError) as caught:
            RecordReviewV5(self.sources, self.episodes, timeout).map(self.records[1], self.state)
        trace = caught.exception.trace
        self.assertEqual(trace['request'], requests[0])
        self.assertEqual(len(trace['record_ids']), 5)
        self.assertEqual(trace['version'], 5)
        self.assertNotIn('response', trace)


if __name__ == '__main__':
    unittest.main()
