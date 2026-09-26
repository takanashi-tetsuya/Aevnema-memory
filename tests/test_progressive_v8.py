"""SQLite evidence-delivery and recovery checks for local FACTS scheduling.

The reviewer is deterministic and offline. These tests assert timing, durable
stage identity and context-sensitive reconsideration, not semantic accuracy.
"""
from copy import deepcopy
import json
import unittest

from config.prompt_config import recall_v3_prompts as v3, recall_v4_prompts as v4
from memory_demo.retrieval.progressive_v7 import ProgressiveRecallV7
from memory_demo.retrieval.progressive_v8 import ProgressiveRecallV8
from test_progressive_v6 import BatchReviewer
from test_progressive_v7 import source
import test_progressive_v3 as fixtures


QUESTION = '求助信后来怎么样？'


class LocalScheduleReviewer(BatchReviewer):
    """Split an initial MAP and control only subsequent local stage outcomes."""

    def __init__(self, clock):
        super().__init__(clock)
        self.events = []
        self.split_initial = True
        self.second_map_delay = 0
        self.fail_second_map = False
        self.first_verdict = 'accept'
        self.final_verdict = 'accept'
        self.first_need_indices = [0]
        self.fact_payloads = []

    def chat_json(self, system, content):
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        if system == v4.MAP:
            sids = sorted({record['source_id'] for record in payload['records']})
            self.events.append(('map', sids))
            if self.split_initial and len(sids) > 1:
                raise TimeoutError('offline initial multi-Source MAP failure')
            if sids == [2]:
                self.clock.value += self.second_map_delay
                if self.fail_second_map:
                    raise TimeoutError('offline second Source MAP failure')
        elif system == v3.FACTS:
            self.fact_payloads.append(deepcopy(payload))
            self.events.append(('facts', [fact['episode_id'] for fact in payload['facts']]))
        reply = super().chat_json(system, content)
        if system == v4.MAP:
            for fact in reply['facts']:
                if fact['episode_id'] == 1:
                    fact['need_indices'] = list(self.first_need_indices)
        elif system == v3.FACTS:
            context = {record['source_id'] for record in payload['records']}
            for fact, verdict in zip(payload['facts'], reply['fact_decisions']):
                if fact['episode_id'] == 1:
                    decision = self.first_verdict if context == {1} else self.final_verdict
                    verdict.update(decision=decision,
                                   statement='离线fixture原文事件' if decision == 'accept' else '',
                                   episode_alignment='unknown' if decision == 'needs_context' else 'supported')
        return reply


class ProgressiveV8Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.model = LocalScheduleReviewer(self.clock)
        self.service = ProgressiveRecallV8(self.config, self.db, self.model, clock=self.clock)

    def fact_calls_for(self, episode_id):
        return [payload for payload in self.model.fact_payloads
                if any(fact['episode_id'] == episode_id for fact in payload['facts'])]

    def test_first_source_is_delivered_before_later_map_exhausts_global_deadline(self):
        self.model.second_map_delay = 10
        result = self.service.query(QUESTION, timeout_seconds=5, learn=False)
        self.assertEqual(result['status'], 'time_budget', result)
        self.assertEqual(self.model.events[:4],
                         [('map', [1, 2]), ('map', [1]), ('facts', [1]), ('map', [2])])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, {self.eids[0]})
        self.assertEqual(result['review_protocol_version'], 8)
        self.assertTrue(result['schedule_protocol'])
        self.assertTrue(result['local_review_trace'])
        self.assertFalse(result['complete'])

        baseline_model = LocalScheduleReviewer(self.clock)
        baseline_model.second_map_delay = 10
        baseline = ProgressiveRecallV7(self.config, self.db, baseline_model, clock=self.clock)
        original = baseline.query(QUESTION, timeout_seconds=5, learn=False)
        self.assertEqual(original['status'], 'time_budget', original)
        self.assertEqual(original['evidence'], [])
        self.assertEqual(original['pending_evidence_count'], 1)
        self.assertEqual(baseline_model.events,
                         [('map', [1, 2]), ('map', [1]), ('map', [2])])

    def test_context_growth_rechecks_each_early_outcome_including_rejected_candidate(self):
        for initial in ('accept', 'needs_context', 'reject'):
            with self.subTest(initial=initial):
                self.model = LocalScheduleReviewer(self.clock)
                self.model.first_verdict = initial
                service = ProgressiveRecallV8(self.config, self.db, self.model, clock=self.clock)
                result = service.query(QUESTION, learn=False, max_waves=1)
                self.assertTrue(result['complete'], result)
                self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
                calls = self.fact_calls_for(self.eids[0])
                self.assertEqual(len(calls), 2)
                self.assertEqual({r['source_id'] for r in calls[0]['records']}, {1})
                self.assertEqual({r['source_id'] for r in calls[1]['records']}, {1, 2})
                later_fact = next(fact for fact in calls[1]['facts'] if fact['episode_id'] == 1)
                self.assertEqual(calls[0]['facts'][0]['record_ids'], later_fact['record_ids'])
                attempts = [a for a in result['stage_attempts'] if a['stage'] == 'facts']
                self.assertEqual([a['token'] for a in attempts],
                                 [f'facts:{i}' for i in range(len(attempts))])

    def test_early_acceptance_can_be_retracted_when_new_context_changes_verdict(self):
        self.model.final_verdict = 'reject'
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertEqual({f['episode_id'] for f in result['evidence']}, {self.eids[1]})
        self.assertEqual(len(self.fact_calls_for(self.eids[0])), 2)
        state = self.service.sessions.read(result['session_id'])
        transitions = [transition for trace in state['review_trace'] if trace['stage'] == 'facts'
                       for transition in trace.get('transitions', [])]
        first_id = next(transition['fact_id'] for transition in transitions
                        if transition['evidence'][0]['source_id'] == 1)
        self.assertEqual([t['to'] for t in transitions if t['fact_id'] == first_id], ['accept', 'reject'])
        self.assertFalse(any(f['fact_id'] == first_id for f in state['pending_facts']))

    def test_no_new_visible_records_does_not_repeat_early_semantic_decision(self):
        for initial in ('accept', 'needs_context', 'reject'):
            with self.subTest(initial=initial):
                self.model = LocalScheduleReviewer(self.clock)
                self.model.first_verdict = initial
                self.model.fail_second_map = True
                service = ProgressiveRecallV8(self.config, self.db, self.model, clock=self.clock)
                result = service.query(QUESTION, learn=False, max_waves=1)
                self.assertEqual(len(self.fact_calls_for(self.eids[0])), 1)
                self.assertEqual(len(result['completion_blockers']['deferred_source_windows']), 1)
                self.assertEqual(len(result['evidence']), int(initial == 'accept'))
                self.assertEqual(result['pending_evidence_count'], int(initial == 'needs_context'))
                self.assertFalse(result['complete'])

    def test_exhausted_local_failure_gets_no_final_or_later_wave_retry(self):
        self.model.failed_episode = self.eids[0]
        result = self.service.query(QUESTION, learn=False, max_waves=2)
        self.assertEqual(len(self.fact_calls_for(self.eids[0])), 2)
        self.assertFalse(result['complete'])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, {self.eids[1]})
        self.assertEqual(len(result['completion_blockers']['failed_fact_ids']), 1)
        failed = [a for a in result['stage_attempts'] if a['stage'] == 'facts' and a['status'] == 'failed']
        self.assertEqual(len(failed), 2)
        self.assertEqual([a['fact_ids'] for a in failed], [failed[0]['fact_ids']] * 2)

    def test_early_batch_failure_recovers_as_singletons_before_next_source_map(self):
        with self.db.connection() as connection:
            connection.execute('UPDATE source SET raw_text=? WHERE id=1',
                               (source('甲寄出了求助信。', '甲等待求助信回应。'),))
            connection.commit()
        self.model.fail_first_batch = True
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        second_map_index = self.model.events.index(('map', [2]))
        self.assertEqual(self.model.events[:second_map_index],
                         [('map', [1, 2]), ('map', [1]), ('facts', [1, 1]),
                          ('facts', [1]), ('facts', [1])])
        self.assertEqual(self.model.fact_sizes[:3], [2, 1, 1])
        self.assertFalse(result['completion_blockers']['failed_fact_ids'])

    def test_large_local_candidate_set_still_uses_at_most_four_facts_per_call(self):
        with self.db.connection() as connection:
            connection.execute('UPDATE source SET raw_text=? WHERE id=1',
                               (source(*(f'甲第{i}次寄出了求助信。' for i in range(6))),))
            connection.commit()
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        self.assertEqual(len(result['evidence']), 7)
        second_map_index = self.model.events.index(('map', [2]))
        early_batches = [items for stage, items in self.model.events[:second_map_index] if stage == 'facts']
        self.assertEqual([len(items) for items in early_batches], [4, 2])
        self.assertTrue(all(0 < len(payload['facts']) <= 4 for payload in self.model.fact_payloads))

    def test_non_gap_candidate_waits_for_final_review_without_extra_calls(self):
        self.model.first_need_indices = []
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.events,
                         [('map', [1, 2]), ('map', [1]), ('map', [2]), ('facts', [1, 2])])

    def test_resolved_at_wave_entry_is_not_reclassified_as_a_gap_after_map(self):
        review = self.service._review

        def seed_resolved_need(windows, state, sources, episodes):
            # Other needs remain open, but this MAP's fact only concerns 0.
            state['resolved_needs'] = [0]
            return review(windows, state, sources, episodes)

        self.service._review = seed_resolved_need
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.events,
                         [('map', [1, 2]), ('map', [1]), ('map', [2]), ('facts', [1, 2])])

    def test_pre_wave_accepted_facts_keep_the_existing_two_fact_rotation(self):
        with self.db.connection() as connection:
            connection.execute('UPDATE source SET raw_text=? WHERE id=1',
                               (source(*(f'甲第{i}次寄出了求助信。' for i in range(3))),))
            connection.commit()
        self.model.split_initial = False
        review = self.service._review
        captured = []

        def capture_windows(windows, state, sources, episodes):
            captured.append((deepcopy(windows), sources, episodes))
            return review(windows, state, sources, episodes)

        self.service._review = capture_windows
        first = self.service.query(QUESTION, learn=False)
        state = self.service.sessions.read(first['session_id'])
        self.assertEqual(len(state['facts']), 4)
        original_ids = [fact['fact_id'] for fact in state['facts']]
        state['old_fact_review_cursor'] = 1
        self.model.events.clear()
        windows, sources, episodes = captured[0]
        review(windows, state, sources, episodes)
        attempts = [a for a in state['stage_attempts'] if a['stage'] == 'facts' and a['wave'] == 2]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['fact_ids'], original_ids[1:3])
        self.assertEqual(state['old_fact_review_cursor'], 3)
        self.assertEqual(state['metrics']['review_waves'], 2)
        self.assertEqual(self.model.events, [('map', [1, 2]), ('facts', [1, 1])])

    def test_no_split_has_same_calls_and_evidence_as_v7(self):
        self.model.split_initial = False
        result = self.service.query(QUESTION, learn=False)
        baseline_model = LocalScheduleReviewer(self.clock)
        baseline_model.split_initial = False
        baseline = ProgressiveRecallV7(self.config, self.db, baseline_model, clock=self.clock)
        original = baseline.query(QUESTION, learn=False)
        self.assertEqual(result['evidence'], original['evidence'])
        self.assertEqual(result['need_assessments'], original['need_assessments'])
        self.assertEqual(self.model.events, baseline_model.events)
        self.assertEqual(self.model.actual, baseline_model.actual)
        self.assertEqual(self.model.need_batches, baseline_model.need_batches)

    def test_committed_local_fact_batch_is_not_resent_after_cursor_interrupt(self):
        self.model.fail_second_map = True
        write = self.service.sessions.write
        interrupted = []

        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if (not interrupted and 'facts:0' in work.get('committed_stages', {})
                    and work['batch_cursor'] == 0):
                interrupted.append(True)
                raise KeyboardInterrupt()

        self.service.sessions.write = crash
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'cancelled', first)
        self.assertEqual(len(first['evidence']), 1)
        checkpoint = self.service.sessions.read(first['session_id'])
        self.assertEqual(checkpoint['review_work']['batch_cursor'], 0)
        restored = ProgressiveRecallV8(self.config, self.db, self.model, clock=self.clock)
        result = restored.query(QUESTION, resume=first['session_id'], learn=False, max_waves=1)
        self.assertEqual(len(self.fact_calls_for(self.eids[0])), 1)
        self.assertEqual(self.model.events,
                         [('map', [1, 2]), ('map', [1]), ('facts', [1]), ('map', [2])])
        self.assertEqual(len(result['evidence']), 1)

    def test_recorded_local_failure_is_not_resent_after_cursor_interrupt(self):
        self.model.failed_episode = self.eids[0]
        write = self.service.sessions.write
        interrupted = []

        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if (not interrupted and 'facts:0' in work.get('failed_stages', {})
                    and work['batch_cursor'] == 0):
                interrupted.append(True)
                raise KeyboardInterrupt()

        self.service.sessions.write = crash
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'cancelled', first)
        restored = ProgressiveRecallV8(self.config, self.db, self.model, clock=self.clock)
        result = restored.query(QUESTION, resume=first['session_id'], learn=False, max_waves=1)
        self.assertEqual(len(self.fact_calls_for(self.eids[0])), 2)
        self.assertEqual(len(result['completion_blockers']['failed_fact_ids']), 1)
        attempts = [a for a in result['stage_attempts'] if a['stage'] == 'facts']
        self.assertEqual(len({a['token'] for a in attempts}), len(attempts))

    def test_v7_checkpoint_is_rejected_before_any_v8_model_call(self):
        baseline_model = LocalScheduleReviewer(self.clock)
        baseline_model.split_initial = False
        baseline = ProgressiveRecallV7(self.config, self.db, baseline_model, clock=self.clock)
        original = baseline.query(QUESTION, learn=False)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query(QUESTION, resume=original['session_id'], learn=False)
        self.assertEqual(self.model.events, [])
        self.assertEqual(self.model.actual, [])

    def test_changed_schedule_checkpoint_is_rejected_before_any_model_call(self):
        first = self.service.query(QUESTION, learn=False)

        class ChangedSchedule(ProgressiveRecallV8):
            schedule_protocol = {**ProgressiveRecallV8.schedule_protocol, 'version': 2}

        changed = ChangedSchedule(self.config, self.db, self.model, clock=self.clock)
        events = deepcopy(self.model.events)
        calls = deepcopy(self.model.actual)
        with self.assertRaisesRegex(ValueError, 'changed knowledge'):
            changed.query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.events, events)
        self.assertEqual(self.model.actual, calls)


if __name__ == '__main__':
    unittest.main()
