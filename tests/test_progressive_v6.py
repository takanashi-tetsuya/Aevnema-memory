"""Real SQLite coverage of bounded sufficiency batches and crash recovery."""
import json
import unittest

from config.prompt_config import recall_v3_prompts as v3, recall_v6_prompts as v6
from memory_demo.retrieval.progressive_v6 import ProgressiveRecallV6
from benchmarks import run_associative_recall_experiment as runner
from test_progressive_v5 import RecoveringReviewer
import test_progressive_v3 as fixtures


class BatchReviewer(RecoveringReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.need_batches = []
        self.fail_batch = False
        self.fail_target = None
        self.need_delay = 0

    def chat_json(self, system, content):
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        if system == v3.PLAN:
            return {'needs': [f'关于求助信的问题 {i}' for i in range(8)], 'cues': ['求助信']}
        if system == v6.NEEDS:
            indices = [t['need_index'] for t in payload['targets']]
            self.need_batches.append(indices)
            self.clock.value += self.need_delay
            if self.fail_batch and len(indices) > 1 or self.fail_target in indices:
                raise TimeoutError('synthetic NEED deadline')
            return {'assessments': [{'need_index': i, 'status': 'supported',
                'fact_ids': [f['fact_id'] for f in payload['facts']],
                'answer': '离线fixture回答', 'reason': '离线fixture依据'} for i in indices]}
        return super().chat_json(system, content)


class ProgressiveV6Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.old_service = self.service
        self.model = BatchReviewer(self.clock)
        self.service = ProgressiveRecallV6(self.config, self.db, self.model, clock=self.clock)

    def test_eight_targets_use_two_calls_with_unchanged_original_evidence(self):
        result = self.service.query('求助信后来怎么样？', learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.need_batches, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
        self.assertEqual(result['review_protocol_version'], 6)
        state = self.service.sessions.read(result['session_id'])
        self.assertTrue(all(t['version'] == 6 for t in state['review_trace']))
        self.assertIs(runner._service_factory(6), ProgressiveRecallV6)

    def test_failed_batches_split_once_and_clear_only_recovered_failures(self):
        self.model.fail_batch = True
        self.model.fail_target = 2
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertEqual(self.model.need_batches, [[0, 1, 2, 3], [4, 5, 6, 7]] + [[i] for i in range(8)])
        self.assertEqual(result['completion_blockers']['failed_need_indices'], [2])
        self.assertEqual(len(result['evidence']), 2)
        self.assertEqual(result['need_assessments'][2]['status'], 'unknown')

    def test_global_deadline_cannot_start_singleton_recovery(self):
        self.model.need_delay = 10
        result = self.service.query('求助信后来怎么样？', timeout_seconds=5, learn=False)
        self.assertEqual(result['status'], 'time_budget')
        self.assertEqual(self.model.need_batches, [[0, 1, 2, 3]])
        self.assertEqual(len(result['evidence']), 2)
        self.assertFalse(result['complete'])

    def test_checkpoint_after_commit_or_failure_does_not_repeat_batch(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                self.model.need_batches.clear()
                self.model.fail_batch = failure
                self.service = ProgressiveRecallV6(self.config, self.db, self.model, clock=self.clock)
                write = self.service.sessions.write
                interrupted = []
                key = 'failed_stages' if failure else 'committed_stages'
                def crash(state):
                    write(state)
                    work = state.get('review_work') or {}
                    if not interrupted and 'need:0' in work.get(key, {}) and work['need_cursor'] == 0:
                        interrupted.append(True)
                        raise KeyboardInterrupt()
                self.service.sessions.write = crash
                first = self.service.query('求助信后来怎么样？', learn=False)
                self.assertEqual(first['status'], 'cancelled')
                restored = ProgressiveRecallV6(self.config, self.db, self.model, clock=self.clock)
                result = restored.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
                self.assertTrue(result['complete'], result)
                self.assertEqual(self.model.need_batches.count([0, 1, 2, 3]), 1)
                self.assertEqual(self.model.need_batches.count([4, 5, 6, 7]), 1)

    def test_old_checkpoint_is_not_reinterpreted(self):
        first = self.old_service.query('求助信后来怎么样？', learn=False)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertEqual(self.model.need_batches, [])


if __name__ == '__main__':
    unittest.main()
