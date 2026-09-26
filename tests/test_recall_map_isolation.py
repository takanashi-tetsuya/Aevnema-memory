"""Offline MAP fault isolation; scripted callbacks do not test model semantics."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from memory_demo.retrieval.progressive_v5 import ProgressiveRecallV5
from memory_demo.retrieval.recall_map_isolation import FACT_ISOLATION_PROTOCOL, isolated_reviewer
from memory_demo.retrieval.recall_records import project_source
from memory_demo.retrieval.recall_review_v3 import FACTS_SYSTEM, ReviewStageError
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10
from memory_demo.retrieval.recall_review_v11 import RecordReviewV11
import test_recall_review_v11 as feedback_fixtures


def proposal(record_ids=('R1',), *, episode_id=1, need_indices=(0,), interpretation='甲只是猜测。'):
    return {'record_ids': list(record_ids), 'episode_id': episode_id,
            'need_indices': list(need_indices), 'interpretation': interpretation}


class MapIsolationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            1: '[record: 1]\nzh-CN: 甲只是猜测门已打开。\n\n'
               '[record: 2]\nzh-CN: 乙说现在还不能确认。\n\n'
               '[record: 3]\nzh-CN: 甲要求继续核对。',
            2: '[record: 1]\nzh-CN: 门实际仍然关闭。\n\n'
               '[record: 2]\nzh-CN: 丙仍未回来。',
        }
        self.episodes = {1: {'id': 1, 'source_id': 1, 'text': '甲的猜测'},
                         11: {'id': 11, 'source_id': 1, 'text': '乙的保留意见'},
                         2: {'id': 2, 'source_id': 2, 'text': '门仍关闭'}}
        self.records = project_source(1, self.sources[1], [1, 11]) + project_source(2, self.sources[2], [2])
        self.state = {'question': '门是否打开，丙是否回来？', 'needs': ['门是否打开？', '丙是否回来？'],
                      'facts': [], 'pending_facts': [], 'pending_links': [], 'verified_links': [],
                      'need_assessments': [], 'covered_needs': [], 'resolved_needs': []}
        self.calls = []
        self.first = proposal()
        self.second = proposal(('R4',), episode_id=2, need_indices=(1,), interpretation='门仍关闭。')
        self.bad = proposal(('R597',))
        self.response = {'facts': [self.first, self.bad, self.second], 'links': []}
        self.failure = None
        self.reviewer = isolated_reviewer(10, self.sources, self.episodes, self.call)

    def call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if self.failure:
            raise self.failure
        if system == FACTS_SYSTEM:
            return {'fact_decisions': [{'fact_id': f['fact_id'], 'decision': 'accept',
                'episode_alignment': 'supported', 'statement': '仅确认原文中的说法。',
                'reason': 'Synthetic offline verdict.'} for f in payload['facts']]}
        return self.response

    def baseline(self, response, *, state=None, records=None):
        calls = []
        def callback(system, payload):
            calls.append((system, deepcopy(payload)))
            return deepcopy(response)
        result = RecordReviewV10(self.sources, self.episodes, callback).map(
            self.records if records is None else records, self.state if state is None else state)
        return result, calls

    def assert_failed(self, *, state=None, records=None, expected_calls=1):
        state = self.state if state is None else state
        records = self.records if records is None else records
        before = deepcopy((state, records, self.response, self.sources, self.episodes))
        count = len(self.calls)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.map(records, state)
        self.assertEqual(len(self.calls) - count, expected_calls)
        self.assertEqual((state, records, self.response, self.sources, self.episodes), before)
        return caught.exception.trace

    def test_good_bad_good_only_commits_original_legal_facts_in_order(self):
        before = deepcopy((self.state, self.records, self.response))
        result = self.reviewer.map(self.records, self.state)
        expected, calls = self.baseline({'facts': [self.first, self.second], 'links': []})
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.followup_cues, expected.followup_cues)
        self.assertEqual(self.calls, calls)
        self.assertEqual([f['episode_id'] for f in result.updates['pending_facts']], [1, 2])
        self.assertEqual(result.trace['response'], self.response)
        self.assertTrue(result.trace['schema_ok'])
        diagnostic = result.trace['map_fact_isolation']
        self.assertEqual(diagnostic['protocol'], FACT_ISOLATION_PROTOCOL)
        self.assertEqual(diagnostic['filter_status'], 'partial')
        self.assertEqual(diagnostic['raw_fact_count'], 3)
        self.assertEqual(diagnostic['retained_ordinals'], [1, 3])
        self.assertEqual(len(diagnostic['rejected_proposals']), 1)
        rejected = diagnostic['rejected_proposals'][0]
        self.assertEqual(rejected['proposal_ordinal'], 2)
        self.assertEqual(rejected['proposal'], self.bad)
        self.assertEqual(rejected['error_type'], 'ValueError')
        self.assertIn('unknown or duplicate', rejected['reason'])
        self.assertEqual(result.trace['admitted_response']['facts'], [self.first, self.second])
        self.assertEqual((self.state, self.records, self.response), before)

    def test_invalid_first_middle_and_last_do_not_change_valid_merge_order(self):
        expected, _ = self.baseline({'facts': [self.first, self.second], 'links': []})
        for position in range(3):
            with self.subTest(position=position):
                facts = [deepcopy(self.first), deepcopy(self.second)]
                facts.insert(position, deepcopy(self.bad))
                self.response = {'facts': facts, 'links': []}
                count = len(self.calls)
                result = self.reviewer.map(self.records, self.state)
                self.assertEqual(result.updates, expected.updates)
                self.assertEqual(result.trace['response'], self.response)
                self.assertEqual(len(self.calls) - count, 1)

    def test_every_original_fact_validation_remains_strict_without_repair(self):
        invalid = [
            {}, None, [], 'not a fact',
            {k: v for k, v in self.first.items() if k != 'need_indices'},
            {**self.first, 'quote': 'handwritten'},
            {**self.first, 'record_ids': []},
            {**self.first, 'record_ids': ['R1', 'R1']},
            {**self.first, 'record_ids': ['R1', 'R4']},
            {**self.first, 'record_ids': [1]},
            {**self.first, 'episode_id': 2},
            {**self.first, 'episode_id': True},
            {**self.first, 'episode_id': 999},
            {**self.first, 'interpretation': ''},
            {**self.first, 'need_indices': [True]},
            {**self.first, 'need_indices': [99]},
            {**self.first, 'need_indices': [0, 0]},
        ]
        expected, _ = self.baseline({'facts': [self.second], 'links': []})
        for bad in invalid:
            with self.subTest(bad=bad):
                self.response = {'facts': [bad, self.second], 'links': []}
                result = self.reviewer.map(self.records, self.state)
                self.assertEqual(result.updates, expected.updates)
                self.assertEqual(result.trace['response'], self.response)

    def test_nonempty_all_invalid_facts_fail_even_with_valid_cue(self):
        missing = {k: v for k, v in self.first.items() if k != 'need_indices'}
        for facts in ([self.bad], [missing] * 6):
            with self.subTest(facts=facts):
                self.response = {'facts': facts, 'links': [],
                    'followup_cues': [{'record_id': 'R1', 'cue': '门已打开'}]}
                trace = self.assert_failed()
                self.assertEqual(trace['response'], self.response)
                self.assertFalse(trace['schema_ok'])
                diagnostic = trace['map_fact_isolation']
                self.assertEqual(diagnostic['filter_status'], 'all_rejected')
                self.assertEqual(diagnostic['retained_ordinals'], [])
                self.assertEqual([r['proposal_ordinal'] for r in diagnostic['rejected_proposals']],
                                 list(range(1, len(facts) + 1)))
                self.assertEqual([r['proposal'] for r in diagnostic['rejected_proposals']], facts)

    def test_bad_envelope_or_nonlist_is_global_failure(self):
        cases = (None, [], {}, {'facts': []}, {'links': []},
                 {'facts': [self.first], 'links': [], 'unexpected': True},
                 {'facts': 'bad', 'links': []}, {'facts': {}, 'links': []})
        for response in cases:
            with self.subTest(response=response):
                self.response = response
                trace = self.assert_failed()
                self.assertEqual(trace.get('response'), response)

    def test_eight_raw_facts_allowed_but_nine_fail_before_filtering(self):
        self.response = {'facts': [deepcopy(self.first) for _ in range(8)], 'links': []}
        self.assertEqual(len(self.reviewer.map(self.records, self.state).updates['pending_facts']), 1)
        self.response['facts'] += [deepcopy(self.bad)]
        self.assert_failed()
        self.response = {'facts': [deepcopy(self.bad) for _ in range(8)] + [self.first], 'links': []}
        self.assert_failed()

    def test_original_empty_facts_and_valid_cues_are_not_all_invalid(self):
        self.response = {'facts': [], 'links': [], 'followup_cues': [{'record_id': 'R1', 'cue': '门已打开'}]}
        result = self.reviewer.map(self.records, self.state)
        expected, calls = self.baseline(self.response)
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.followup_cues, expected.followup_cues)
        self.assertEqual(self.calls, calls)
        self.assertEqual(len(result.followup_cues), 1)

    def test_empty_records_skip_provider(self):
        result = self.reviewer.map([], self.state)
        expected, calls = self.baseline(self.response, records=[])
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.followup_cues, expected.followup_cues)
        self.assertEqual(result.trace['skipped'], 'no_visible_records')
        self.assertEqual(self.calls, calls)
        self.assertEqual(self.calls, [])

    def test_bad_link_remains_global_failure_after_good_facts(self):
        invalid_links = (None, {}, [{'from_episode_id': 1, 'to_episode_id': 1, 'rationale': 'self'}],
            [{'from_episode_id': 1, 'to_episode_id': 999, 'rationale': 'unknown'}],
            [{'from_episode_id': 1, 'to_episode_id': 2}],
            [{'from_episode_id': True, 'to_episode_id': 2, 'rationale': 'bool'}])
        for links in invalid_links:
            with self.subTest(links=links):
                self.response = {'facts': [self.first, self.bad, self.second], 'links': links}
                trace = self.assert_failed()
                self.assertEqual(trace['response'], self.response)
                self.assertFalse(trace['schema_ok'])
                self.assertEqual(trace['map_fact_isolation']['filter_status'], 'partial')
                self.assertEqual(trace['map_fact_isolation']['retained_ordinals'], [1, 3])

    def test_valid_links_and_optional_cues_keep_original_independence(self):
        # A current record may supply a cue without its fact being proposed.
        self.response = {'facts': [self.first, self.bad], 'links': [
            {'from_episode_id': 1, 'to_episode_id': 2, 'rationale': '对照说法。'}],
            'followup_cues': [{'record_id': 'R4', 'cue': '仍然关闭'}]}
        expected, _ = self.baseline({**self.response, 'facts': [self.first]})
        result = self.reviewer.map(self.records, self.state)
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.followup_cues, expected.followup_cues)
        self.assertEqual(result.followup_cues[0]['source_id'], 2)

    def test_invalid_optional_cues_keep_raw_response_and_do_not_discard_good_facts(self):
        self.response['followup_cues'] = [
            {'record_id': 'R597', 'cue': '不存在'},
            {'record_id': 'R1', 'cue': '不是原文的缺口描述'},
            {'record_id': 'R4', 'cue': '仍然关闭'}]
        result = self.reviewer.map(self.records, self.state)
        self.assertEqual(len(result.updates['pending_facts']), 2)
        self.assertEqual([c['cue'] for c in result.followup_cues], ['仍然关闭'])
        self.assertEqual(len(result.trace['rejected_optional_cues']), 2)
        self.assertEqual(result.trace['response'], self.response)
        self.assertEqual(result.trace['admitted_response']['facts'], [self.first, self.second])
        self.assertEqual(result.trace['admitted_response']['followup_cues'],
                         [{'record_id': 'R4', 'cue': '仍然关闭'}])

    def test_damaged_records_registry_or_source_fail_before_model(self):
        for kind in ('records', 'registry', 'source'):
            with self.subTest(kind=kind):
                state, records = deepcopy(self.state), deepcopy(self.records)
                if kind == 'records':
                    records[0]['text'] += ' changed'
                elif kind == 'registry':
                    state['record_registry'] = {records[0]['record_id']: deepcopy(records[0])}
                    state['record_registry'][records[0]['record_id']]['source_sha256'] = 'forged'
                else:
                    records[0]['source_id'] = 999
                self.assert_failed(state=state, records=records, expected_calls=0)

    def test_damaged_stored_pending_or_accepted_fact_fail_before_model(self):
        seeded, _ = self.baseline({'facts': [self.first], 'links': []})
        for field in ('pending_facts', 'facts'):
            state = deepcopy(self.state)
            state.update(deepcopy(seeded.updates))
            fact = state['pending_facts'].pop()
            fact['fact_id'] = 'forged-id'
            state[field] = [fact]
            with self.subTest(field=field):
                self.assert_failed(state=state, expected_calls=0)

    def test_context_budget_still_rejects_before_provider(self):
        self.state['review_context_max_chars'] = 1
        trace = self.assert_failed(expected_calls=0)
        self.assertGreater(trace['input_chars'], trace['input_char_budget'])

    def test_provider_deadline_is_not_treated_as_salvageable_fact_error(self):
        self.failure = TimeoutError('scripted provider deadline')
        trace = self.assert_failed()
        self.assertEqual(trace.get('provider_error_type'), 'TimeoutError')
        self.assertNotIn('response', trace)
        self.assertEqual(trace['map_fact_isolation']['filter_status'], 'not_processed')
        self.assertIsNone(trace['map_fact_isolation']['raw_fact_count'])

    def test_duplicates_preserve_pending_union_interpretation_order_and_last_writer(self):
        seeded, _ = self.baseline({'facts': [self.first], 'links': []})
        self.state.update(deepcopy(seeded.updates))
        self.state['pending_facts'][0].update(verification_status='needs_context', statement='earlier statement')
        a = proposal(need_indices=(1,), interpretation='新的解释一。')
        b = proposal(need_indices=(0,), interpretation='最后解释。')
        self.response = {'facts': [a, self.bad, b, deepcopy(a)], 'links': []}
        expected, _ = self.baseline({'facts': [a, b, a], 'links': []})
        result = self.reviewer.map(self.records, self.state)
        self.assertEqual(result.updates, expected.updates)
        pending = result.updates['pending_facts'][0]
        self.assertEqual(pending['need_indices'], [0, 1])
        self.assertEqual(pending['interpretation'], '新的解释一。')
        self.assertEqual(pending['verification_status'], 'pending')
        self.assertEqual(result.updates['interpretation_proposals'][pending['fact_id']],
                         [self.first['interpretation'], a['interpretation'], b['interpretation']])

    def test_previously_accepted_fact_is_not_overwritten_or_returned_to_pending(self):
        seeded, _ = self.baseline({'facts': [self.first], 'links': []})
        self.state.update(deepcopy(seeded.updates))
        accepted = RecordReviewV10(self.sources, self.episodes, self.call).review_facts(self.state)
        self.state.update(deepcopy(accepted.updates))
        before = deepcopy(self.state['facts'])
        revised = proposal(need_indices=(1,), interpretation='新的提案不得改写已核原文。')
        self.response = {'facts': [revised, self.bad, self.second], 'links': []}
        expected, _ = self.baseline({'facts': [revised, self.second], 'links': []})
        result = self.reviewer.map(self.records, self.state)
        self.assertEqual(result.updates, expected.updates)
        self.state.update(result.updates)
        self.assertEqual(self.state['facts'], before)
        self.assertEqual([f['episode_id'] for f in self.state['pending_facts']], [2])

    def test_all_legal_payload_updates_and_cues_match_base_exactly(self):
        self.response = {'facts': [self.first, self.second], 'links': [
            {'from_episode_id': 1, 'to_episode_id': 2, 'rationale': '原文说法的对照。'}],
            'followup_cues': [{'record_id': 'R3', 'cue': '继续核对'}]}
        expected, calls = self.baseline(self.response)
        result = self.reviewer.map(self.records, self.state)
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.followup_cues, expected.followup_cues)
        self.assertEqual(result.trace['transitions'], expected.trace['transitions'])
        self.assertEqual(self.calls, calls)
        trace = deepcopy(result.trace)
        diagnostic = trace.pop('map_fact_isolation')
        self.assertEqual(trace, expected.trace)
        self.assertEqual(diagnostic['filter_status'], 'unchanged')
        self.assertEqual(diagnostic['retained_ordinals'], [1, 2])
        self.assertEqual(diagnostic['rejected_proposals'], [])

    def test_same_instance_after_failed_call_has_no_stale_isolation_context(self):
        self.response = {'facts': [self.bad], 'links': []}
        self.assert_failed()
        self.response = {'facts': [self.second], 'links': []}
        result = self.reviewer.map(self.records, self.state)
        expected, _ = self.baseline(self.response)
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result.trace['map_fact_isolation']['rejected_proposals'], [])

    def test_fact_isolation_does_not_invent_a_minimum_need_index_rule(self):
        # Original MAP admits an empty need_indices list; don't conflate it
        # with an absent required field or silently add a target.
        self.response = {'facts': [proposal(need_indices=()), self.bad], 'links': []}
        result = self.reviewer.map(self.records, self.state)
        expected, _ = self.baseline({'facts': [proposal(need_indices=())], 'links': []})
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(result.updates['pending_facts'][0]['need_indices'], [])

    def test_factory_rejects_unregistered_protocols_without_model_calls(self):
        for protocol in (True, 9, 12, '10', None, []):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                isolated_reviewer(protocol, self.sources, self.episodes, self.call)
        self.assertEqual(self.calls, [])

    def test_successful_isolated_stage_commit_before_cursor_resume_does_not_resend(self):
        self.state.update(metrics={'review_waves': 0}, review_trace=[], review_work={
            'map_cursor': 0, 'committed_stages': {}, 'attempt_context': {}})
        engine = object.__new__(ProgressiveRecallV5)
        engine._ensure_live = lambda: None
        saved = []
        def write(state):
            saved.append(deepcopy(state))
            if 'map:0' in state['review_work']['committed_stages']:
                raise InterruptedError('commit persisted before scheduler cursor advance')
        engine.sessions = SimpleNamespace(write=write)
        with self.assertRaises(InterruptedError):
            engine._commit_stage(lambda: self.reviewer.map(self.records, self.state), self.state, 'map')
        restored = deepcopy(saved[-1])
        self.assertEqual(len(restored['pending_facts']), 2)
        self.assertEqual(restored['review_work']['map_cursor'], 0)
        engine.sessions = SimpleNamespace(write=lambda state: None)
        resumed = engine._commit_stage(lambda: self.fail('committed MAP must not run again'), restored, 'map')
        self.assertEqual(resumed.trace, {'resumed_committed_stage': 'map:0'})
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(restored['review_trace']), 1)
        self.assertTrue(restored['review_trace'][0]['committed'])

    def test_late_after_reviewer_return_is_not_committed_or_resent_on_recovery(self):
        self.state.update(metrics={'review_waves': 0}, review_trace=[], review_work={
            'map_cursor': 0, 'committed_stages': {}, 'attempt_context': {}})
        engine = object.__new__(ProgressiveRecallV5)
        expired, saved = [False], []
        def ensure_live():
            if expired[0]:
                raise TimeoutError('original stage deadline expired after MAP returned')
        def operation():
            result = self.reviewer.map(self.records, self.state)
            self.assertEqual(len(result.updates['pending_facts']), 2)
            expired[0] = True
            return result
        engine._ensure_live = ensure_live
        engine.sessions = SimpleNamespace(write=lambda state: saved.append(deepcopy(state)))
        with self.assertRaises(TimeoutError):
            engine._commit_stage(operation, self.state, 'map')
        restored = deepcopy(saved[-1])
        self.assertEqual(restored['pending_facts'], [])
        self.assertNotIn('record_registry', restored)
        self.assertEqual(restored['review_work']['committed_stages'], {})
        self.assertIn('map:0', restored['review_work']['failed_stages'])
        self.assertFalse(restored['review_trace'][-1]['committed'])
        self.assertEqual(restored['stage_attempts'][0]['status'], 'failed')
        engine._ensure_live = lambda: None
        replayed = engine._commit_stage(lambda: self.fail('failed MAP must not run again'), restored, 'map')
        self.assertIsNone(replayed)
        self.assertEqual(len(self.calls), 1)


class MapIsolationV11Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = feedback_fixtures.RecordReviewV11Tests()
        self.fixture.setUp()
        self.f = self.fixture
        self.reviewer = isolated_reviewer(11, self.f.sources, self.f.episodes, self.f.call)

    def test_feedback_prompt_and_snapshot_unchanged_with_one_call_and_isolated_fact(self):
        self.f.map_response = {'facts': [proposal(('R4',), interpretation='甲批准申请。'), proposal(('R597',))], 'links': []}
        before = deepcopy(self.f.state)
        result = self.reviewer.map(self.f.records[1], self.f.state)
        self.assertEqual(self.f.state, before)
        self.assertEqual(len(self.f.calls), 1)
        self.assertEqual(result.trace['map_gap_feedback_snapshot'], self.f.snapshot)
        self.assertEqual(self.f.calls[0][1]['need_feedback'], self.f.snapshot['payload'])
        self.assertEqual(result.trace['response'], self.f.map_response)
        response = deepcopy(self.f.map_response)
        self.f.map_response['facts'].pop()
        expected = RecordReviewV11(self.f.sources, self.f.episodes, self.f.call).map(self.f.records[1], self.f.state)
        self.assertEqual(result.updates, expected.updates)
        self.assertEqual(self.f.calls[0], self.f.calls[1])
        self.assertEqual(response['facts'][1]['record_ids'], ['R597'])

    def test_forged_feedback_is_global_failure_before_provider(self):
        self.f.state['map_feedback_snapshots'][0]['payload']['items'][0]['previous_reason'] = 'forged'
        with self.assertRaises(ReviewStageError):
            self.reviewer.map(self.f.records[1], self.f.state)
        self.assertEqual(self.f.calls, [])

    def test_fact_and_need_review_stages_remain_exactly_base(self):
        base = RecordReviewV11(self.f.sources, self.f.episodes, self.f.call)
        before = deepcopy(self.f.state)
        for operation in (lambda reviewer: reviewer.review_facts(self.f.state, fact_ids=[self.f.state['facts'][0]['fact_id']]),
                          lambda reviewer: reviewer.review_needs(self.f.state, [0, 1])):
            with self.subTest(operation=operation):
                expected = operation(base)
                result = operation(self.reviewer)
                self.assertEqual(result.updates, expected.updates)
                self.assertEqual(result.followup_cues, expected.followup_cues)
                self.assertEqual(result.trace, expected.trace)
                self.assertEqual(self.f.calls[-2], self.f.calls[-1])
                self.assertEqual(self.f.state, before)


if __name__ == '__main__':
    unittest.main()
