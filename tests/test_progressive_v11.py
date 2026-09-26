"""Offline SQLite checks for durable previous-NEED feedback in later MAPs."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from config.prompt_config import recall_v4_prompts as v4, recall_v9_prompts as v9
from config.prompt_config.recall_v11_prompts import GAP_MAP
from memory_demo.embeddings.codec import encode_embedding
from memory_demo.repositories import SourceRepository, EpisodeRepository
from memory_demo.types import EpisodeDraft
from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
from memory_demo.retrieval.progressive_v11 import ProgressiveRecallV11
from memory_demo.retrieval.recall_map_feedback import FEEDBACK_PROTOCOL, validate_feedback
import test_progressive_v3 as fixtures
from test_progressive_v8 import QUESTION
from test_progressive_v10 import FragmentReviewer


class FeedbackReviewer(FragmentReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.contract_constraint = True
        self.constraint_unknown = True
        self.map_inputs = []
        self.need_round = 0
        self.split_feedback_map = False
        self.emit_followup_cue = False
        self.observe_map = None
        self.map_observations = []

    def chat_json(self, system, content):
        if system in (v4.MAP, GAP_MAP):
            payload = (getattr(content, "structured_payload", None) or json.loads(content))
            self.map_inputs.append((system, deepcopy(payload)))
            if self.observe_map:
                self.map_observations.append(deepcopy(self.observe_map()))
            if self.split_feedback_map and system == GAP_MAP and len({r['source_id'] for r in payload['records']}) > 1:
                raise TimeoutError('offline split for later-wave MAP')
            result = super().chat_json(v4.MAP, content)
            if self.emit_followup_cue and len(self.map_inputs) == 1:
                record = next(r for r in payload['records'] if r['text'].startswith('第'))
                result['followup_cues'] = [{'record_id': record['id'], 'cue': record['text']}]
            return result
        result = super().chat_json(system, content)
        if system == v9.NEEDS:
            self.need_round += 1
            for row in result['assessments']:
                for item in row['items']:
                    if item['status'] == 'unknown':
                        item['reason'] = f'离线已提交NEED第{self.need_round}次留下的具体限定缺口'
        return result


class ProgressiveV11Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        sources, episodes = SourceRepository(self.db), EpisodeRepository(self.db)
        # Ten real Sources leave unread windows for at least three normal waves.
        for i in range(3, 11):
            text = f'第{i}份求助信有新的回应记录。'
            sid = sources.insert(text)
            episodes.insert(sid, f'feedback/{i}', 0, EpisodeDraft(text), encode_embedding([1., 0.], 2))
        self.model = FeedbackReviewer(self.clock)
        self.service = self.restore()
        self.last_written = None
        self.install_write_observer(self.service)
        self.model.observe_map = lambda: self.last_written

    def restore(self):
        return ProgressiveRecallV11(self.config, self.db, self.model, clock=self.clock)

    def install_write_observer(self, service, interrupt=None):
        write = service.sessions.write
        fired = []

        def observed(state):
            write(state)
            self.last_written = deepcopy(state)
            if interrupt and not fired and interrupt(state):
                fired.append(True)
                raise KeyboardInterrupt()

        service.sessions.write = observed
        return fired

    def query(self, service=None, **kwargs):
        return (service or self.service).query(QUESTION, learn=False, **kwargs)

    def test_first_wave_map_payload_and_prompt_equal_v10(self):
        old_model = FeedbackReviewer(self.clock)
        old = ProgressiveRecallV10(self.config, self.db, old_model, clock=self.clock)
        baseline = old.query(QUESTION, learn=False, max_waves=1)
        result = self.query(max_waves=1)
        self.assertEqual(self.model.map_inputs, old_model.map_inputs)
        self.assertEqual(self.model.map_inputs[0][0], v4.MAP)
        self.assertNotIn('need_feedback', self.model.map_inputs[0][1])
        self.assertEqual(result['map_feedback_snapshots'][0]['payload']['items'], [])
        self.assertEqual(result['need_assessments'], baseline['need_assessments'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['review_protocol_version'], 11)

    def test_second_wave_receives_only_committed_unresolved_item(self):
        result = self.query(max_waves=2)
        self.assertEqual(result['metrics']['review_waves'], 2, result)
        first, second = result['map_feedback_snapshots']
        self.assertEqual(first['payload']['items'], [])
        self.assertEqual(len(second['payload']['items']), 1)
        gap = second['payload']['items'][0]
        self.assertEqual((gap['need_index'], gap['item_id'], gap['kind']), (0, 'C2', 'constraint'))
        self.assertEqual(gap['previous_status'], 'unknown')
        self.assertEqual(gap['previous_value'], '')
        self.assertEqual(gap['prior_review_wave'], 1)
        self.assertIn('第1次', gap['previous_reason'])
        self.assertNotIn('fact_ids', gap)
        self.assertNotIn('record_ids', gap)
        feedback_calls = [(prompt, payload) for prompt, payload in self.model.map_inputs if 'need_feedback' in payload]
        self.assertTrue(feedback_calls)
        self.assertTrue(all(prompt == GAP_MAP and payload['need_feedback'] == second['payload']
                            for prompt, payload in feedback_calls))
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual(validate_feedback(state, second), second['payload'])

    def test_snapshot_write_before_review_work_interrupt_reuses_same_snapshot(self):
        fired = self.install_write_observer(self.service, lambda s:
            len(s.get('map_feedback_snapshots', [])) == 2 and not s.get('review_work'))
        first = self.query(max_waves=2)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        snapshot = deepcopy(first['map_feedback_snapshots'][1])
        before = len(self.model.map_inputs)
        restored = self.restore()
        self.install_write_observer(restored)
        with patch('memory_demo.retrieval.progressive_v11.capture_feedback', side_effect=AssertionError('must reuse entry')):
            result = self.query(restored, resume=first['session_id'], max_waves=1)
        self.assertEqual(result['metrics']['review_waves'], 2, result)
        self.assertEqual(result['map_feedback_snapshots'][1], snapshot)
        self.assertEqual(len(self.model.map_inputs), before + 1)
        self.assertEqual(self.model.map_inputs[-1][1]['need_feedback'], snapshot['payload'])

    def test_early_fact_invalidation_keeps_all_same_wave_map_feedback_identical(self):
        self.model.split_feedback_map = True
        result = self.query(max_waves=2)
        self.assertEqual(result['metrics']['review_waves'], 2, result)
        second = result['map_feedback_snapshots'][1]
        calls = [(payload, state) for (_, payload), state in zip(self.model.map_inputs, self.model.map_observations)
                 if 'need_feedback' in payload]
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(all(p['need_feedback'] == second['payload'] for p, _ in calls))
        after_early = [s for _, s in calls if any(t.get('stage') == 'facts' and t.get('wave') == 2
                      and t.get('attempt_context', {}).get('kind') == 'early' and t.get('committed')
                      for t in s['review_trace'])]
        self.assertTrue(after_early)
        self.assertTrue(all(a['status'] == 'unknown' for a in after_early[0]['need_assessments']))
        self.assertIn('Evidence review changed', after_early[0]['need_assessments'][0]['reason'])

    def test_new_wave_uses_new_committed_need_not_old_diagnostic(self):
        result = self.query(max_waves=3)
        self.assertEqual(result['metrics']['review_waves'], 3, result)
        snapshots = result['map_feedback_snapshots']
        self.assertEqual([s['wave'] for s in snapshots], [1, 2, 3])
        self.assertIn('第1次', snapshots[1]['payload']['items'][0]['previous_reason'])
        self.assertIn('第2次', snapshots[2]['payload']['items'][0]['previous_reason'])
        self.assertNotEqual(snapshots[1]['snapshot_sha256'], snapshots[2]['snapshot_sha256'])

    def test_resume_after_early_fact_commit_keeps_entry_feedback_after_need_reset(self):
        self.model.split_feedback_map = True
        fired = self.install_write_observer(self.service, lambda s:
            any(t.get('stage') == 'facts' and t.get('wave') == 2 and t.get('committed')
                and t.get('attempt_context', {}).get('kind') == 'early'
                for t in s.get('review_trace', [])))
        first = self.query(max_waves=2)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertIn('Evidence review changed', first['need_assessments'][0]['reason'])
        snapshot = deepcopy(first['map_feedback_snapshots'][1])
        restored = self.restore()
        self.install_write_observer(restored)
        before = len(self.model.map_inputs)
        with patch('memory_demo.retrieval.progressive_v11.capture_feedback', side_effect=AssertionError('must reuse entry')):
            result = self.query(restored, resume=first['session_id'], max_waves=1)
        self.assertEqual(result['metrics']['review_waves'], 2, result)
        self.assertEqual(result['map_feedback_snapshots'][1], snapshot)
        self.assertGreater(len(self.model.map_inputs), before)
        self.assertTrue(all(p['need_feedback'] == snapshot['payload'] for _, p in self.model.map_inputs[before:]))

    def test_map_commit_before_cursor_interrupt_does_not_repeat_map_or_feedback_capture(self):
        fired = self.install_write_observer(self.service, lambda s:
            len(s.get('map_feedback_snapshots', [])) == 2
            and 'map:0' in (s.get('review_work') or {}).get('committed_stages', {})
            and s['review_work']['map_cursor'] == 0)
        first = self.query(max_waves=2)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        count = len(self.model.map_inputs)
        restored = self.restore()
        self.install_write_observer(restored)
        with patch('memory_demo.retrieval.progressive_v11.capture_feedback', side_effect=AssertionError('must reuse entry')):
            result = self.query(restored, resume=first['session_id'], max_waves=1)
        self.assertEqual(result['metrics']['review_waves'], 2, result)
        self.assertEqual(len(self.model.map_inputs), count)
        self.assertEqual(result['map_feedback_snapshots'], first['map_feedback_snapshots'])

    def test_missing_snapshot_for_started_map_work_cannot_be_recaptured(self):
        self.install_write_observer(self.service, lambda s:
            len(s.get('map_feedback_snapshots', [])) == 2 and bool(s.get('review_work')))
        first = self.query(max_waves=2)
        state = self.service.sessions.read(first['session_id'])
        state['map_feedback_snapshots'].pop()
        self.service.sessions.write(state)
        before = len(self.model.map_inputs)
        result = self.query(self.restore(), resume=first['session_id'], max_waves=1)
        self.assertEqual(result['status'], 'technical_error')
        self.assertIn('feedback', result['error']['message'])
        self.assertEqual(len(self.model.map_inputs), before)

    def test_changed_feedback_payload_rejected_before_next_map(self):
        self.install_write_observer(self.service, lambda s:
            len(s.get('map_feedback_snapshots', [])) == 2 and not s.get('review_work'))
        first = self.query(max_waves=2)
        state = self.service.sessions.read(first['session_id'])
        state['map_feedback_snapshots'][1]['payload']['items'][0]['previous_reason'] = 'tampered gap'
        self.service.sessions.write(state)
        before = len(self.model.map_inputs)
        result = self.query(self.restore(), resume=first['session_id'], max_waves=1)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(len(self.model.map_inputs), before)

    def test_invalid_history_with_pending_cue_is_rejected_before_any_provider_work(self):
        self.model.emit_followup_cue = True
        first = self.query(max_waves=1)
        original = self.service.sessions.read(first['session_id'])
        self.assertTrue(original['cue_queue']['pending'])
        self.assertEqual(original['metrics']['followup_cues_processed'], 0)
        for corruption in ('tampered', 'deleted', 'missing'):
            with self.subTest(corruption=corruption):
                state = deepcopy(original)
                if corruption == 'tampered':
                    state['map_feedback_snapshots'][0]['payload']['scope'] = 'tampered'
                elif corruption == 'deleted':
                    state['map_feedback_snapshots'] = []
                else:
                    state.pop('map_feedback_snapshots')
                self.service.sessions.write(state)
                before = deepcopy(self.model.pipeline)
                result = self.query(self.restore(), resume=first['session_id'], max_waves=1)
                self.assertEqual(result['status'], 'technical_error')
                self.assertEqual(self.model.pipeline, before)
                after = self.service.sessions.read(first['session_id'])
                self.assertEqual(after['cue_queue'], original['cue_queue'])
                self.assertEqual(after['metrics']['followup_cues_processed'], 0)

    def test_changed_raw_need_response_breaks_historical_snapshot_before_provider_work(self):
        first = self.query(max_waves=2)
        state = self.service.sessions.read(first['session_id'])
        origin = next(t for t in state['review_trace']
                      if t['stage'] == 'need' and t.get('committed') and t.get('wave') == 1)
        origin['response']['assessments'][0]['items'][0]['reason'] = 'changed raw provider judgment'
        self.service.sessions.write(state)
        before = deepcopy(self.model.pipeline)
        result = self.query(self.restore(), resume=first['session_id'], max_waves=1)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(self.model.pipeline, before)

    def test_changed_prompt_or_protocol_rejects_resume_before_provider_work(self):
        first = self.query(max_waves=1)
        before = deepcopy(self.model.pipeline)
        with patch('memory_demo.retrieval.progressive_v11.GAP_MAP', GAP_MAP + '\nchanged'):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.query(self.restore(), resume=first['session_id'], max_waves=1)
        with patch.dict(FEEDBACK_PROTOCOL, {'truncation': 'different-rule'}):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.query(self.restore(), resume=first['session_id'], max_waves=1)
        self.assertEqual(self.model.pipeline, before)

    def test_v10_checkpoint_is_rejected_before_any_provider_work(self):
        old_model = FeedbackReviewer(self.clock)
        old = ProgressiveRecallV10(self.config, self.db, old_model, clock=self.clock)
        first = old.query(QUESTION, learn=False, max_waves=1)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.query(resume=first['session_id'], max_waves=1)
        self.assertEqual(self.model.pipeline, [])

    def test_supported_first_wave_still_stops_without_feedback_map(self):
        self.model.constraint_unknown = False
        result = self.query()
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['metrics']['review_waves'], 1)
        self.assertEqual(len(result['map_feedback_snapshots']), 1)
        self.assertTrue(all(prompt == v4.MAP for prompt, _ in self.model.map_inputs))


if __name__ == '__main__':
    unittest.main()
