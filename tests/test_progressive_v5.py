"""Bounded recovery, honest completion and durable attempt identity on SQLite."""
import json
import unittest

from config.prompt_config import recall_v3_prompts as v3, recall_v4_prompts as v4
from memory_demo.retrieval.progressive_v5 import ProgressiveRecallV5
from benchmarks import run_associative_recall_experiment as runner
import test_progressive_v3 as fixtures
from test_progressive_v4 import FocusedReviewer


class RecoveringReviewer(FocusedReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.fact_sizes = []
        self.fail_first_batch = False
        self.failed_episode = None
        self.map_failure = None
        self.map_sources = []
        self.needs_context_episode = None

    def chat_json(self, system, content):
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        if system == v4.MAP:
            sources = {r['source_id'] for r in payload['records']}
            self.map_sources.append(sources)
            if self.map_failure is not None and self.map_failure in sources:
                raise TimeoutError('synthetic MAP deadline')
        if system == v3.FACTS:
            self.fact_sizes.append(len(payload['facts']))
            if (self.fail_first_batch and len(self.fact_sizes) == 1
                    or any(f['episode_id'] == self.failed_episode for f in payload['facts'])):
                raise TimeoutError('synthetic FACTS deadline')
        reply = super().chat_json(system, content)
        if system == v3.FACTS:
            for fact, decision in zip(payload['facts'], reply['fact_decisions']):
                if fact['episode_id'] == self.needs_context_episode:
                    decision.update(decision='needs_context', episode_alignment='unknown', statement='')
        return reply


class ProgressiveV5Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.old_service = self.service
        self.model = RecoveringReviewer(self.clock)
        self.service = ProgressiveRecallV5(self.config, self.db, self.model, clock=self.clock)

    def test_success_preserves_original_evidence_and_scoped_prompt(self):
        result = self.service.query('求助信后来怎么样？', learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['review_protocol_version'], 5)
        self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
        self.assertFalse(any(result['completion_blockers'].values()))
        state = self.service.sessions.read(result['session_id'])
        self.assertTrue(all(t['version'] == 5 for t in state['review_trace']))

    def test_failed_batch_retries_as_single_facts_before_need_completion(self):
        self.model.fail_first_batch = True
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.fact_sizes, [2, 1, 1])
        self.assertEqual(result['pending_evidence_count'], 0)
        self.assertEqual(len(self.model.map_sources), 1)
        attempts = [a for a in result['stage_attempts'] if a['stage'] == 'facts']
        self.assertEqual([a['status'] for a in attempts], ['failed', 'committed', 'committed'])
        self.assertEqual(set(attempts[0]['fact_ids']), {a['fact_ids'][0] for a in attempts[1:]})
        state = self.service.sessions.read(result['session_id'])
        trace = next(t for t in state['review_trace'] if t['stage'] == 'facts' and not t['committed'])
        self.assertEqual(len(trace['request']['facts']), 2)
        self.assertEqual(trace['provider_error_type'], 'TimeoutError')

    def test_permanent_fact_failure_cannot_be_hidden_by_supported_need(self):
        self.model.failed_episode = self.eids[0]
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=2)
        self.assertFalse(result['complete'], result)
        self.assertEqual(result['need_assessments'][0]['status'], 'supported')
        self.assertEqual(len(result['evidence']), 1)
        self.assertEqual(result['pending_evidence_count'], 1)
        self.assertEqual(len(result['completion_blockers']['failed_fact_ids']), 1)
        self.assertTrue(result['missing_requirements'])
        # Initial batch + one singleton recovery; subsequent waves may recheck
        # accepted evidence but do not retry the exhausted failed candidate.
        failed = [a for a in result['stage_attempts'] if a['stage'] == 'facts' and a['status'] == 'failed']
        self.assertEqual(len(failed), 2)

    def test_needs_context_is_not_implicitly_resolved_by_other_evidence(self):
        self.model.needs_context_episode = self.eids[0]
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=1)
        self.assertEqual(result['need_assessments'][0]['status'], 'supported')
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['completion_blockers']['pending_fact_ids']), 1)

    def test_failed_multi_source_map_salvages_other_source_without_advancing_failed_cursor(self):
        self.model.map_failure = 1
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=2)
        self.assertEqual(self.model.map_sources, [{1, 2}, {1}, {2}])
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['evidence']), 1)
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual(state['source_offsets'].get('1', 0), 0)
        self.assertGreater(state['source_offsets']['2'], 0)
        self.assertEqual(len(result['completion_blockers']['deferred_source_windows']), 1)

    def test_global_deadline_does_not_start_recovery_with_extra_budget(self):
        self.model.fact_delay = 50
        result = self.service.query('求助信后来怎么样？', timeout_seconds=5, learn=False)
        self.assertEqual(result['status'], 'time_budget')
        self.assertEqual(self.model.fact_sizes, [2])
        self.assertEqual(result['pending_evidence_count'], 2)
        self.assertFalse(result['complete'])

    def test_committed_fact_batch_is_not_repeated_after_cursor_interrupt(self):
        self.model.reject_first = True
        write = self.service.sessions.write
        interrupted = []
        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if not interrupted and 'facts:0' in work.get('committed_stages', {}) and work['batch_cursor'] == 0:
                interrupted.append(True)
                raise KeyboardInterrupt()
        self.service.sessions.write = crash
        first = self.service.query('求助信后来怎么样？', learn=False)
        self.assertEqual(first['status'], 'cancelled')
        restored = ProgressiveRecallV5(self.config, self.db, self.model, clock=self.clock)
        result = restored.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.fact_sizes, [2])
        self.assertEqual(len(self.model.map_sources), 1)

    def test_old_protocol_checkpoint_is_not_reinterpreted(self):
        first = self.old_service.query('求助信后来怎么样？', learn=False)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertEqual(self.model.actual, [])

    def test_recorded_failure_is_not_repeated_after_cursor_interrupt(self):
        self.model.fail_first_batch = True
        write = self.service.sessions.write
        interrupted = []
        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if not interrupted and 'facts:0' in work.get('failed_stages', {}) and work['batch_cursor'] == 0:
                interrupted.append(True)
                raise KeyboardInterrupt()
        self.service.sessions.write = crash
        first = self.service.query('求助信后来怎么样？', learn=False)
        self.assertEqual(first['status'], 'cancelled')
        restored = ProgressiveRecallV5(self.config, self.db, self.model, clock=self.clock)
        result = restored.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.fact_sizes, [2, 1, 1])
        failed = [a for a in result['stage_attempts'] if a['status'] == 'failed']
        self.assertEqual(len(failed), 1)

    def test_runner_recovers_completion_blockers_without_model_call(self):
        self.model.failed_episode = self.eids[0]
        first = self.service.query('求助信后来怎么样？', learn=False, max_waves=1)
        state = self.service.sessions.read(first['session_id'])
        before = len(self.model.actual)
        recovered = runner._recover_result(state, self.service, 5)
        self.assertEqual(recovered['completion_blockers'], first['completion_blockers'])
        self.assertEqual(recovered['stage_attempts'], first['stage_attempts'])
        self.assertFalse(recovered['complete'])
        self.assertEqual(len(self.model.actual), before)
        self.assertIs(runner._service_factory(5), ProgressiveRecallV5)


if __name__ == '__main__':
    unittest.main()
