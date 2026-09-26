"""Offline SQLite integration for MAP fact isolation on the unchanged V10 loop.

Provider judgments are scripted. Passing establishes scheduler, checkpoint and
admission behavior, never semantic correctness or a new recall measurement.
"""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from config.prompt_config import recall_v4_prompts as v4
from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
from memory_demo.retrieval.progressive_v11 import ProgressiveRecallV11
from memory_demo.retrieval.progressive_v12 import (
    ProgressiveRecallV12, isolation_implementation_hashes,
)
from memory_demo.retrieval.recall_map_isolation import FACT_ISOLATION_PROTOCOL
import test_progressive_v3 as fixtures
from test_progressive_v8 import QUESTION
from test_progressive_v10 import FragmentReviewer


class IsolationScheduleReviewer(FragmentReviewer):
    """Change only synthetic MAP replies; PLAN/CONTRACT/FACTS/NEED stay fixed."""

    def __init__(self, clock):
        super().__init__(clock)
        self.mode = 'legal'
        self.map_inputs = []
        self.first_map_delay = 0

    def chat_json(self, system, content):
        if system != v4.MAP:
            return super().chat_json(system, content)
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        self.map_inputs.append((system, deepcopy(payload)))
        response = super().chat_json(system, content)
        multiple = len({r['source_id'] for r in payload['records']}) > 1
        if multiple and self.mode in {'partial_batch', 'partial_bad_link_batch'}:
            response['facts'][1]['record_ids'] = ['R597']
        if self.mode == 'all_invalid_all' or multiple and self.mode == 'all_invalid_batch':
            for fact in response['facts']:
                fact['record_ids'] = ['R597']
        if multiple and self.mode == 'partial_bad_link_batch':
            response['links'] = [{'from_episode_id': 1, 'to_episode_id': 999,
                                  'rationale': 'offline invalid global endpoint'}]
        if len(self.map_inputs) == 1:
            self.clock.value += self.first_map_delay
        return response


class ProgressiveV12Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.model = IsolationScheduleReviewer(self.clock)
        self.service = ProgressiveRecallV12(self.config, self.db, self.model, clock=self.clock)

    def restore(self):
        return ProgressiveRecallV12(self.config, self.db, self.model, clock=self.clock)

    def interrupt_after_write(self, predicate):
        original = self.service.sessions.write
        fired = []

        def observed(state):
            original(state)
            if not fired and predicate(state):
                fired.append(True)
                raise KeyboardInterrupt()

        self.service.sessions.write = observed
        return fired

    @staticmethod
    def map_traces(state):
        return [t for t in state['review_trace'] if t['stage'] == 'map']

    def source_groups(self, model=None):
        return [sorted({r['source_id'] for r in payload['records']})
                for _, payload in (model or self.model).map_inputs]

    def test_snapshot_adds_only_isolation_binding_to_v10(self):
        baseline = ProgressiveRecallV10(self.config, self.db, self.model, clock=self.clock)
        old = baseline._snapshot()
        new = self.service._snapshot()
        payload = deepcopy(self.service._snapshot_payload)
        isolation = payload.pop('map_fact_isolation')
        self.assertEqual(payload, baseline._snapshot_payload)
        self.assertNotEqual(new[-1], old[-1])
        self.assertEqual(isolation, {
            'base_protocol_version': 10,
            'protocol': FACT_ISOLATION_PROTOCOL,
            'implementation_sha256': isolation_implementation_hashes(),
        })
        self.assertEqual(set(isolation['implementation_sha256']),
                         {'progressive_v12.py', 'recall_review_v12.py', 'recall_map_isolation.py'})
        self.assertFalse(issubclass(ProgressiveRecallV12, ProgressiveRecallV11))
        self.assertEqual(self.service.learning_verifier, 'progressive_record_review_v12')

    def test_all_legal_query_matches_v10_calls_and_evidence_without_gap_feedback(self):
        result = self.service.query(QUESTION, learn=False)
        other_model = IsolationScheduleReviewer(self.clock)
        baseline = ProgressiveRecallV10(self.config, self.db, other_model, clock=self.clock)
        old = baseline.query(QUESTION, learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['evidence'], old['evidence'])
        self.assertEqual(result['need_assessments'], old['need_assessments'])
        self.assertEqual(self.model.pipeline, other_model.pipeline)
        self.assertEqual(self.model.map_inputs, other_model.map_inputs)
        self.assertEqual(self.model.fact_payloads, other_model.fact_payloads)
        self.assertEqual(self.model.item_payloads, other_model.item_payloads)
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual([t['stage'] for t in state['review_trace']], ['contract', 'map', 'facts', 'need'])
        self.assertTrue(all(t['version'] == 12 for t in state['review_trace']))
        self.assertEqual(result['review_protocol_version'], 12)
        self.assertEqual(result['map_fact_isolation_protocol'], FACT_ISOLATION_PROTOCOL)
        self.assertNotIn('map_feedback_snapshots', state)
        self.assertNotIn('need_feedback', self.model.map_inputs[0][1])
        self.assertEqual(self.map_traces(state)[0]['map_fact_isolation']['filter_status'], 'unchanged')
        self.assertEqual(result['learning']['status'], 'disabled')

    def test_partial_success_commits_all_windows_but_only_valid_pending_before_facts(self):
        self.model.mode = 'partial_batch'
        fired = self.interrupt_after_write(lambda s: (s.get('review_work') or {}).get('phase') == 'facts')
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertEqual(first['evidence'], [])
        state = self.service.sessions.read(first['session_id'])
        self.assertEqual(len(state['pending_facts']), 1)
        self.assertEqual(state['pending_facts'][0]['episode_id'], self.eids[0])
        self.assertEqual(len(state['mapped_source_windows']), 2)
        self.assertEqual(len(state['source_offsets']), 2)
        self.assertEqual(state['review_work']['map_cursor'], 1)
        self.assertEqual(len(state['review_work']['map_groups']), 1)
        self.assertEqual(self.source_groups(), [[1, 2]])
        self.assertNotIn('facts', self.model.pipeline)
        trace = self.map_traces(state)[0]
        self.assertTrue(trace['committed'])
        self.assertEqual(trace['map_fact_isolation']['filter_status'], 'partial')
        self.assertEqual(trace['map_fact_isolation']['retained_ordinals'], [1])
        self.assertEqual(trace['response']['facts'][1]['record_ids'], ['R597'])
        self.assertEqual(len(trace['admitted_response']['facts']), 1)
        final = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual({f['episode_id'] for f in final['evidence']}, {self.eids[0]})
        self.assertEqual(self.source_groups(), [[1, 2]])

    def test_v10_same_partial_reply_splits_while_v12_does_not_retry_rejected_fact(self):
        self.model.mode = 'partial_batch'
        new = self.service.query(QUESTION, learn=False, max_waves=1)
        other_model = IsolationScheduleReviewer(self.clock)
        other_model.mode = 'partial_batch'
        baseline = ProgressiveRecallV10(self.config, self.db, other_model, clock=self.clock)
        old = baseline.query(QUESTION, learn=False, max_waves=1)
        self.assertEqual(self.source_groups(), [[1, 2]])
        self.assertEqual(self.source_groups(other_model), [[1, 2], [1], [2]])
        self.assertEqual({f['episode_id'] for f in new['evidence']}, {self.eids[0]})
        self.assertEqual({f['episode_id'] for f in old['evidence']}, set(self.eids))
        self.assertEqual(new['completion_blockers']['deferred_source_windows'], [])

    def test_all_invalid_group_still_splits_and_singletons_recover(self):
        self.model.mode = 'all_invalid_batch'
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.source_groups(), [[1, 2], [1], [2]])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
        state = self.service.sessions.read(result['session_id'])
        traces = self.map_traces(state)
        self.assertEqual([t['committed'] for t in traces], [False, True, True])
        self.assertEqual(traces[0]['map_fact_isolation']['filter_status'], 'all_rejected')
        self.assertEqual(len(state['mapped_source_windows']), 2)
        self.assertEqual([a['token'] for a in state['stage_attempts'] if a['stage'] == 'map'],
                         ['map:0', 'map:1', 'map:2'])

    def test_all_invalid_singletons_are_deferred_without_pending_or_window_commit(self):
        self.model.mode = 'all_invalid_all'
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertEqual(self.source_groups(), [[1, 2], [1], [2]])
        self.assertEqual(result['evidence'], [])
        self.assertEqual(result['pending_evidence_count'], 0)
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual(state['mapped_source_windows'], [])
        self.assertEqual(state['source_offsets'], {})
        self.assertEqual(len(state['deferred_source_windows']), 2)
        self.assertTrue(all(not t['committed'] for t in self.map_traces(state)))
        self.assertNotIn('facts', self.model.pipeline)

    def test_global_bad_link_still_discards_partial_fact_result_and_splits(self):
        self.model.mode = 'partial_bad_link_batch'
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual(self.source_groups(), [[1, 2], [1], [2]])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
        trace = self.map_traces(state)[0]
        self.assertFalse(trace['committed'])
        self.assertEqual(trace['map_fact_isolation']['filter_status'], 'partial')
        self.assertEqual(trace['map_fact_isolation']['retained_ordinals'], [1])
        self.assertEqual(state['pending_links'], [])

    def test_late_partial_reply_rolls_back_and_resume_does_not_repeat_failed_group(self):
        self.model.mode = 'partial_batch'
        self.model.first_map_delay = 10
        first = self.service.query(QUESTION, timeout_seconds=5, learn=False)
        self.assertEqual(first['status'], 'time_budget', first)
        state = self.service.sessions.read(first['session_id'])
        self.assertEqual(state['facts'], [])
        self.assertEqual(state['pending_facts'], [])
        self.assertEqual(state['mapped_source_windows'], [])
        self.assertEqual(state['source_offsets'], {})
        self.assertIn('map:0', state['review_work']['failed_stages'])
        self.assertNotIn('map:0', state['review_work']['committed_stages'])
        self.assertFalse(self.map_traces(state)[0]['committed'])
        final = self.restore().query(QUESTION, resume=first['session_id'], learn=False, max_waves=1)
        self.assertEqual(self.source_groups(), [[1, 2], [1], [2]])
        self.assertEqual({f['episode_id'] for f in final['evidence']}, set(self.eids))
        self.assertEqual(self.model.pipeline.count('plan'), 1)
        self.assertEqual(self.model.pipeline.count('contract'), 1)

    def test_committed_partial_map_resumes_before_cursor_without_another_callback(self):
        self.model.mode = 'partial_batch'
        fired = self.interrupt_after_write(lambda s: 'map:0' in (s.get('review_work') or {}).get('committed_stages', {})
                                          and s['review_work']['map_cursor'] == 0)
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        state = self.service.sessions.read(first['session_id'])
        self.assertEqual(len(state['pending_facts']), 1)
        self.assertEqual(state['mapped_source_windows'], [])
        self.assertEqual(state['review_work']['map_cursor'], 0)
        self.assertEqual(len(self.map_traces(state)), 1)
        final = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.source_groups(), [[1, 2]])
        self.assertEqual({f['episode_id'] for f in final['evidence']}, {self.eids[0]})
        resumed = self.service.sessions.read(first['session_id'])
        self.assertEqual(len(resumed['mapped_source_windows']), 2)
        self.assertEqual(len(self.map_traces(resumed)), 1)
        self.assertEqual(len([a for a in resumed['stage_attempts'] if a['stage'] == 'map']), 1)

    def test_recorded_all_invalid_group_resumes_before_cursor_into_split_recovery(self):
        self.model.mode = 'all_invalid_batch'
        fired = self.interrupt_after_write(lambda s: 'map:0' in (s.get('review_work') or {}).get('failed_stages', {}))
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        state = self.service.sessions.read(first['session_id'])
        self.assertEqual(state['review_work']['map_cursor'], 0)
        self.assertEqual(len(state['review_work']['map_groups']), 1)
        self.assertEqual(state['pending_facts'], [])
        final = self.restore().query(QUESTION, resume=first['session_id'], learn=False, max_waves=1)
        self.assertEqual(self.source_groups(), [[1, 2], [1], [2]])
        self.assertEqual({f['episode_id'] for f in final['evidence']}, set(self.eids))
        self.assertEqual(len(self.map_traces(self.service.sessions.read(first['session_id']))), 3)

    def test_changed_isolation_policy_rejects_checkpoint_before_any_provider_call(self):
        first = self.service.query(QUESTION, learn=False)
        before = deepcopy(self.model.pipeline)
        with patch.dict(FACT_ISOLATION_PROTOCOL, {'all_invalid_nonempty_facts': 'different-policy'}):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, before)

    def test_changed_isolation_implementation_rejects_checkpoint_before_any_provider_call(self):
        first = self.service.query(QUESTION, learn=False)
        before = deepcopy(self.model.pipeline)
        changed = isolation_implementation_hashes()
        changed['recall_map_isolation.py'] = '0' * 64
        with patch('memory_demo.retrieval.progressive_v12.isolation_implementation_hashes', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, before)

    def test_v10_and_v11_checkpoints_are_not_reinterpreted_as_v12(self):
        for cls in (ProgressiveRecallV10, ProgressiveRecallV11):
            with self.subTest(protocol=cls.protocol_version):
                model = IsolationScheduleReviewer(self.clock)
                old = cls(self.config, self.db, model, clock=self.clock)
                first = old.query(QUESTION, learn=False)
                self.assertTrue(first['complete'], first)
                with self.assertRaisesRegex(ValueError, 'another review protocol'):
                    self.service.query(QUESTION, resume=first['session_id'], learn=False)
                self.assertEqual(self.model.pipeline, [])

    def test_v12_checkpoint_cannot_resume_under_atomic_v10(self):
        first = self.service.query(QUESTION, learn=False)
        model = IsolationScheduleReviewer(self.clock)
        old = ProgressiveRecallV10(self.config, self.db, model, clock=self.clock)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            old.query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(model.pipeline, [])

    def test_runner_factory_and_cli_offer_v12_without_changing_default(self):
        from benchmarks import run_associative_recall_experiment as runner
        self.assertIs(runner._service_factory(12), ProgressiveRecallV12)
        self.assertIs(runner._service_factory(10), ProgressiveRecallV10)
        directory = Path(self.config.log_dir).parent
        argv = ['--source-database', str(self.db.path), '--output', str(directory / 'campaign'),
                '--manifest', str(directory / 'offline-manifest.json'), '--prepare-only',
                '--max-http-attempts', '90']
        report = {'campaign_id': 'offline', 'prepared_only': True, 'completed_attempt_receipts': 0,
                  'expected_attempts': 3, 'stop_reason': None, 'campaign_http_budget': {}}
        for extra, protocol in ((['--protocol', '12'], 12), ([], 3)):
            with self.subTest(protocol=protocol), patch.object(runner, 'run', return_value=report) as run:
                with patch('sys.stdout', new=io.StringIO()):
                    self.assertEqual(runner.main(argv + extra), 0)
                self.assertEqual(run.call_args.kwargs['protocol'], protocol)
                self.assertEqual(run.call_args.kwargs['max_http_attempts'], 90)
                self.assertTrue(run.call_args.kwargs['prepare_only'])
                self.assertFalse(run.call_args.kwargs['learn'])


if __name__ == '__main__':
    unittest.main()
