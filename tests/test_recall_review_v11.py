"""Historical wave-entry feedback gates; synthetic verdicts are not semantics."""
from copy import deepcopy
import json
import unittest

from config.prompt_config.recall_v11_prompts import GAP_MAP
from memory_demo.retrieval.recall_contract_anchors import anchor_binding
from memory_demo.retrieval.recall_map_feedback import capture_feedback, validate_feedback, active_feedback
from memory_demo.retrieval.recall_need_contracts import unknown_assessment
from memory_demo.retrieval.recall_review_v3 import ReviewStageError, _hash
from memory_demo.retrieval.recall_review_v4 import MAP_SYSTEM
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10
from memory_demo.retrieval.recall_review_v11 import RecordReviewV11
import test_recall_review_v9 as v9


class RecordReviewV11Tests(unittest.TestCase):
    def setUp(self):
        self.commit_wave = 1
        v9.RecordReviewV9Tests.setUp(self)
        self.state['metrics'] = {'review_waves': 1}
        self.state['contract_anchor_binding'] = anchor_binding(self.state)
        self.response['assessments'][0]['items'][1] = v9.item('C2', value='仅限当晚的限定尚缺支持。')
        self.response['assessments'][1] = v9.RecordReviewV9Tests.refuted_second_need(self)
        self.v10 = RecordReviewV10(self.sources, self.episodes, self.call)
        self.commit(self.v10.review_needs(self.state, [0, 1]))
        self.snapshot = capture_feedback(self.state, 2)
        self.state['map_feedback_snapshots'] = [deepcopy(self.snapshot)]
        self.reviewer = RecordReviewV11(self.sources, self.episodes, self.call)
        self.map_response = {'facts': [], 'links': []}
        self.calls.clear()

    def commit(self, result):
        self.state.update(deepcopy(result.updates))
        self.state.setdefault('review_trace', []).append({**deepcopy(result.trace),
            'committed': True, 'wave': self.commit_wave})
        return result

    def call(self, system, payload):
        if system in (GAP_MAP, MAP_SYSTEM) and hasattr(self, 'map_response'):
            self.calls.append((system, deepcopy(payload)))
            return deepcopy(self.map_response)
        return v9.RecordReviewV9Tests.call(self, system, payload)

    def snapshot_state(self):
        self.state['map_feedback_snapshots'] = [capture_feedback(self.state, 2)]

    def test_effective_unknown_constraint_is_feedback_but_supported_and_refuted_are_excluded(self):
        items = validate_feedback(self.state, self.snapshot)['items']
        self.assertEqual([(i['need_index'], i['item_id']) for i in items], [(0, 'C2')])
        self.assertEqual(items[0]['previous_value'], '仅限当晚的限定尚缺支持。')
        self.assertEqual(items[0]['prior_review_wave'], 1)
        self.assertEqual(items[0]['request_anchor'], self.state['need_contracts'][0]['items'][1]['anchor'])
        self.assertEqual(self.snapshot['origins'][0]['trace_index'], 2)

    def test_capture_and_active_access_do_not_mutate_or_expose_caller_objects(self):
        before = deepcopy(self.state)
        captured = capture_feedback(self.state, 2)
        self.assertEqual(self.state, before)
        active = active_feedback(self.state)
        active['payload']['items'][0]['description'] = 'changed'
        self.assertEqual(self.state, before)
        self.assertEqual(captured, self.snapshot)

    def test_failed_skipped_no_response_or_bad_schema_need_is_not_feedback(self):
        for changes in ({'committed': False}, {'skipped': 'no_accepted_facts'},
                        {'response': None}, {'schema_ok': False}):
            with self.subTest(changes=changes):
                state = deepcopy(self.state)
                state['review_trace'][-1].update(changes)
                self.assertEqual(capture_feedback(state, 2)['payload']['items'], [])

    def test_local_synthetic_unknown_without_need_response_does_not_become_model_diagnosis(self):
        state = deepcopy(self.state)
        state['review_trace'] = state['review_trace'][:2]
        state['need_assessments'] = [unknown_assessment(c, state['need_contract_hash'],
            'No current assessment bound to the contract and accepted evidence.')
            for c in state['need_contracts']]
        self.assertEqual(capture_feedback(state, 2)['payload']['items'], [])

    def test_same_wave_need_cannot_be_retrospectively_called_prior_wave_feedback(self):
        self.assertEqual(capture_feedback(self.state, 1)['payload']['items'], [])

    def test_full_supported_or_refuted_set_has_no_map_prompt_or_payload_change(self):
        self.response['assessments'][0]['items'][1] = v9.item('C2', 'supported',
            value='批准仅限当晚。', facts=['F1'], records=['R5'])
        self.commit(self.v10.review_needs(self.state, [0, 1]))
        self.snapshot_state()
        self.assertEqual(active_feedback(self.state)['payload']['items'], [])
        old = self.v10.map(self.records[1], self.state)
        old_call = deepcopy(self.calls[-1])
        new = self.reviewer.map(self.records[1], self.state)
        self.assertEqual(self.calls[-1], old_call)
        self.assertEqual(self.calls[-1][0], MAP_SYSTEM)
        self.assertEqual(new.updates, old.updates)
        self.assertNotIn('need_feedback', new.trace['request'])

    def test_nonempty_feedback_changes_only_prompt_and_one_payload_field(self):
        old = self.v10.map(self.records[1] + self.records[2], self.state)
        before = deepcopy(self.state)
        new = self.reviewer.map(self.records[1] + self.records[2], self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(self.calls[-1][0], GAP_MAP)
        payload = deepcopy(new.trace['request'])
        self.assertEqual(payload.pop('need_feedback'), self.snapshot['payload'])
        self.assertEqual(payload, old.trace['request'])
        self.assertEqual(new.updates, old.updates)
        self.assertEqual(new.trace['record_ids'], old.trace['record_ids'])
        self.assertEqual(new.trace['map_gap_feedback_snapshot'], self.snapshot)

    def test_current_fact_or_assessment_mismatch_prevents_wave_entry_capture(self):
        for field in ('facts', 'need_assessments'):
            state = deepcopy(self.state)
            state[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                capture_feedback(state, 2)

    def test_forged_snapshot_rehashed_payload_origin_and_prefix_are_rejected(self):
        mutations = (
            lambda s: s['payload']['items'][0].update(previous_reason='forged diagnosis'),
            lambda s: s['payload']['items'][0].update(need_index=1),
            lambda s: s.update(origins=[]),
            lambda s: s.update(trace_prefix_length=0),
            lambda s: s.update(trace_prefix_sha256='0' * 64),
        )
        for mutate in mutations:
            snap = deepcopy(self.snapshot)
            mutate(snap)
            snap['snapshot_sha256'] = _hash({k: v for k, v in snap.items() if k != 'snapshot_sha256'})
            with self.subTest(snapshot=snap), self.assertRaises(ValueError):
                validate_feedback(self.state, snap)

    def test_trace_semantic_tamper_is_rejected_even_with_unchanged_snapshot(self):
        state = deepcopy(self.state)
        state['review_trace'][-1]['need_assessments'][0]['item_assessments'][1]['reason'] += ' tampered'
        with self.assertRaises(ValueError):
            validate_feedback(state, self.snapshot)

    def test_raw_need_response_tamper_is_detected_even_when_parsed_assessment_stays_same(self):
        state = deepcopy(self.state)
        state['review_trace'][-1]['response']['assessments'][0]['items'][1]['reason'] += ' tampered'
        self.assertEqual(state['review_trace'][-1]['need_assessments'],
                         self.state['review_trace'][-1]['need_assessments'])
        with self.assertRaises(ValueError):
            validate_feedback(state, self.snapshot)

    def test_exported_token_redaction_does_not_change_semantic_snapshot_binding(self):
        state = deepcopy(self.state)
        for trace in state['review_trace']:
            trace['token'] = '[REDACTED]'
            trace['attempt_context'] = {'token': '[REDACTED]'}
        self.assertEqual(validate_feedback(state, self.snapshot), self.snapshot['payload'])

    def test_early_facts_reset_keeps_current_wave_entry_feedback_but_not_next_wave(self):
        self.commit_wave = 2
        self.commit(self.v10.review_facts(self.state, fact_ids=[self.state['facts'][0]['fact_id']]))
        self.assertEqual(self.state['resolved_needs'], [])
        self.assertTrue(all('item_assessments' not in a for a in self.state['need_assessments']))
        self.assertEqual(active_feedback(self.state), self.snapshot)
        self.reviewer.map(self.records[1], self.state)
        self.assertEqual(self.calls[-1][1]['need_feedback'], self.snapshot['payload'])
        self.assertEqual(capture_feedback(self.state, 3)['payload']['items'], [])

    def test_absent_wrong_wave_or_tampered_anchor_refuses_map_before_model(self):
        for mutate in (lambda s: s.update(map_feedback_snapshots=[]),
                       lambda s: s['metrics'].update(review_waves=2),
                       lambda s: s.update(contract_anchor_binding={'bad': True})):
            state = deepcopy(self.state)
            mutate(state)
            count = len(self.calls)
            with self.assertRaises(ReviewStageError):
                self.reviewer.map(self.records[1], state)
            self.assertEqual(len(self.calls), count)

    def test_record_admission_is_identical_and_feedback_is_not_a_record_alias(self):
        self.map_response = {'facts': [{'record_ids': ['R4', 'R5'], 'episode_id': 1,
            'interpretation': '甲批准这次申请，仅限当晚。', 'need_indices': [0]}], 'links': []}
        old = self.v10.map(self.records[1], self.state)
        new = self.reviewer.map(self.records[1], self.state)
        self.assertEqual(new.updates, old.updates)
        self.assertEqual(new.trace['transitions'], old.trace['transitions'])
        self.map_response['facts'][0]['record_ids'] = ['C2']
        before = deepcopy(self.state)
        with self.assertRaises(ReviewStageError):
            self.reviewer.map(self.records[1], self.state)
        self.assertEqual(self.state, before)

    def test_v5_optional_cue_filter_still_removes_invalid_cues_without_erasing_facts(self):
        self.map_response = {'facts': [{'record_ids': ['R4'], 'episode_id': 1,
            'interpretation': '甲批准申请。', 'need_indices': [0]}], 'links': [],
            'followup_cues': [{'record_id': 'R4', 'cue': '批准'},
                             {'record_id': 'R999', 'cue': '伪造'},
                             {'record_id': 'R4', 'cue': '缺少身份类别'}]}
        result = self.reviewer.map(self.records[1], self.state)
        self.assertEqual(len(result.followup_cues), 1)
        self.assertEqual(result.followup_cues[0]['cue'], '批准')
        self.assertEqual(len(result.trace['transitions']), 1)
        self.assertEqual(len(result.trace['rejected_optional_cues']), 2)
        self.assertEqual(result.trace['response'], self.map_response)
        self.assertEqual(len(result.trace['admitted_response']['followup_cues']), 1)

    def test_feedback_shares_actual_context_budget_without_truncating_records(self):
        result = self.reviewer.map(self.records[1], self.state)
        payload = result.trace['request']
        size = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        self.state['review_context_max_chars'] = size
        self.reviewer.map(self.records[1], self.state)
        self.state['review_context_max_chars'] = size - 1
        before, count = deepcopy(self.state), len(self.calls)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.map(self.records[1], self.state)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(self.state, before)
        self.assertEqual(caught.exception.trace['request'], payload)
        self.assertEqual(caught.exception.trace['input_chars'], size)

    def test_default_48k_budget_applies_to_historical_reason_without_silent_clipping(self):
        self.response['assessments'][0]['items'][1]['reason'] = '原文尚不足。' * 10000
        self.commit(self.v10.review_needs(self.state, [0, 1]))
        self.snapshot_state()
        count = len(self.calls)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.map(self.records[1], self.state)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(caught.exception.trace['input_char_budget'], 48000)
        self.assertGreater(caught.exception.trace['input_chars'], 48000)
        self.assertEqual(caught.exception.trace['request']['need_feedback']['items'][0]['previous_reason'],
                         self.response['assessments'][0]['items'][1]['reason'])

    def test_prompt_callback_is_restored_after_failed_gap_map(self):
        original = self.reviewer.call
        self.map_response = {'facts': [{'record_ids': ['R999'], 'episode_id': 1,
            'interpretation': '伪造', 'need_indices': [0]}], 'links': []}
        with self.assertRaises(ReviewStageError):
            self.reviewer.map(self.records[1], self.state)
        self.assertIs(self.reviewer.call, original)


if __name__ == '__main__':
    unittest.main()
