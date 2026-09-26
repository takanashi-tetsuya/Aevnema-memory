"""Offline anchor admission and unchanged V9 item gates; no semantic model test."""
from copy import deepcopy
import unittest

from config.prompt_config.recall_v10_prompts import CONTRACT
from memory_demo.retrieval.recall_contract_anchors import (
    anchor_binding, anchor_input, admit_anchor_contracts, request_fragments,
)
from memory_demo.retrieval.recall_need_contracts import bound_contracts, contract_gaps
from memory_demo.retrieval.recall_review_v3 import ReviewStageError
from memory_demo.retrieval.recall_review_v9 import RecordReviewV9
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10
import test_recall_review_v9 as v9


class RequestAnchorTests(unittest.TestCase):
    def setUp(self):
        self.state = {'question': ' 她是谁？\n是否去过新 ETO？🙂保留。尾句',
                      'context': '仅依据原文；不要推断!\n追加',
                      'needs': ['渚是谁？', '是否去过新ETO？']}
        self.response = {'contracts': [
            {'need_index': 0, 'answer_type': 'identity', 'items': [
                {'kind': 'answer', 'description': '她的身份', 'anchor_id': 'Q1'}]},
            {'need_index': 1, 'answer_type': 'boolean', 'items': [
                {'kind': 'answer', 'description': '是否去过新 ETO', 'anchor_id': 'Q2'},
                {'kind': 'constraint', 'description': '不做原文外推断', 'anchor_id': 'X2'}]},
        ]}

    def test_unicode_offsets_and_sentence_punctuation_preserve_original_text(self):
        records = request_fragments(self.state)
        self.assertEqual([r['anchor_id'] for r in records],
                         ['Q1', 'Q2', 'Q3', 'Q4', 'X1', 'X2', 'X3'])
        self.assertEqual(records[0]['text'], ' 她是谁？\n')
        self.assertEqual(records[1]['text'], '是否去过新 ETO？')
        self.assertEqual(records[2]['text'], '🙂保留。')
        for record in records:
            self.assertEqual(record['text'], self.state[record['field']][record['start']:record['end']])
        for field in ('question', 'context'):
            self.assertEqual(''.join(r['text'] for r in records if r['field'] == field), self.state[field])

    def test_empty_context_whitespace_and_mixed_punctuation_have_no_fake_ids(self):
        state = {'question': '甲？！；\n乙;丙!尾部', 'context': ' \n', 'needs': ['甲']}
        records = request_fragments(state)
        self.assertEqual([r['text'] for r in records], ['甲？！；\n', '乙;', '丙!', '尾部'])
        self.assertEqual([r['anchor_id'] for r in records], ['Q1', 'Q2', 'Q3', 'Q4'])
        self.assertEqual(request_fragments({'question': '', 'context': ''}), [])

    def test_input_separates_literal_request_from_rewritten_plan_without_mutation(self):
        before = deepcopy(self.state)
        payload = anchor_input(self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(set(payload), {'original_request_fragments', 'targets'})
        self.assertEqual(payload['targets'][0], {'need_index': 0, 'planned_need': '渚是谁？'})
        self.assertEqual(payload['original_request_fragments'][0]['text'], ' 她是谁？\n')
        payload['original_request_fragments'][0]['text'] = '改写'
        self.assertEqual(self.state, before)

    def test_admission_locally_binds_quote_and_keeps_caller_objects_unchanged(self):
        before, original = deepcopy(self.state), deepcopy(self.response)
        updates, admitted = admit_anchor_contracts(self.state, self.response)
        self.assertEqual(self.state, before)
        self.assertEqual(self.response, original)
        first = updates['need_contracts'][0]['items'][0]
        self.assertEqual(first['item_id'], 'C1')
        self.assertEqual(first['anchor'], {'field': 'question', 'quote': '她是谁？'})
        self.assertEqual(updates['need_contracts'][1]['items'][0]['anchor']['quote'], '是否去过新 ETO？')
        self.assertEqual(updates['need_contracts'][1]['items'][1]['anchor']['field'], 'context')
        self.assertNotIn('anchor_id', admitted['contracts'][0]['items'][0])
        committed = dict(deepcopy(self.state), **updates)
        self.assertEqual(bound_contracts(committed), updates['need_contracts'])

    def test_unknown_nonstring_or_wrong_case_ids_are_rejected_atomically(self):
        for anchor_id in ('Q0', 'Q999', 'q1', 'X99', 'Q1 ', '', 1, True, None, [], {}):
            with self.subTest(anchor_id=anchor_id):
                response = deepcopy(self.response)
                response['contracts'][0]['items'][0]['anchor_id'] = anchor_id
                before = deepcopy(self.state)
                original = deepcopy(response)
                with self.assertRaises(ValueError):
                    admit_anchor_contracts(self.state, response)
                self.assertEqual(self.state, before)
                self.assertEqual(response, original)

    def test_model_quotes_offsets_fields_or_old_anchor_schema_are_rejected(self):
        for extra in ({'quote': '她是谁？'}, {'field': 'question'}, {'start': 1}, {'end': 5},
                      {'anchor': {'field': 'question', 'quote': '她是谁？'}}):
            with self.subTest(extra=extra):
                response = deepcopy(self.response)
                response['contracts'][0]['items'][0].update(extra)
                with self.assertRaises(ValueError):
                    admit_anchor_contracts(self.state, response)
        old = deepcopy(self.response)
        item = old['contracts'][0]['items'][0]
        del item['anchor_id']
        item['anchor'] = {'field': 'question', 'quote': '她是谁？'}
        with self.assertRaises(ValueError):
            admit_anchor_contracts(self.state, old)

    def test_contract_shape_rules_still_reject_incomplete_or_extra_answers(self):
        for change in ('missing_need', 'duplicate_need', 'two_answers', 'no_answer', 'extra_response_key'):
            response = deepcopy(self.response)
            if change == 'missing_need':
                response['contracts'].pop()
            elif change == 'duplicate_need':
                response['contracts'].append(deepcopy(response['contracts'][0]))
            elif change == 'two_answers':
                response['contracts'][1]['items'][1]['kind'] = 'answer'
            elif change == 'no_answer':
                response['contracts'][0]['items'][0]['kind'] = 'constraint'
            else:
                response['status'] = 'complete'
            with self.subTest(change=change), self.assertRaises(ValueError):
                admit_anchor_contracts(self.state, response)

    def test_anchor_binding_is_deterministic_and_sensitive_to_original_spacing(self):
        expected = anchor_binding(self.state)
        self.assertEqual(anchor_binding(deepcopy(self.state)), expected)
        changed = deepcopy(self.state)
        changed['question'] = changed['question'].replace('新 ETO', '新ETO')
        self.assertNotEqual(anchor_binding(changed), expected)
        changed = deepcopy(self.state)
        changed['context'] += '。'
        self.assertNotEqual(anchor_binding(changed), expected)
        expected['protocol']['version'] = 99
        self.assertEqual(anchor_binding(self.state)['protocol']['version'], 1)


class RecordReviewV10Tests(unittest.TestCase):
    commit = v9.RecordReviewV9Tests.commit

    def setUp(self):
        v9.RecordReviewV9Tests.setUp(self)
        self.reviewer = RecordReviewV10(self.sources, self.episodes, self.call)
        self.anchor_response = {'contracts': [
            {'need_index': 0, 'answer_type': 'boolean', 'items': [
                {'kind': 'answer', 'description': '甲是否批准这次申请', 'anchor_id': 'Q1'},
                {'kind': 'constraint', 'description': '批准是否仅限当晚', 'anchor_id': 'Q1'}]},
            {'need_index': 1, 'answer_type': 'time', 'items': [
                {'kind': 'answer', 'description': '丙回来的时间', 'anchor_id': 'Q2'},
                {'kind': 'premise', 'description': '丙已经回来', 'anchor_id': 'Q2'}]},
        ]}

    def call(self, system, payload):
        if system == CONTRACT:
            self.calls.append((system, deepcopy(payload)))
            return deepcopy(self.anchor_response)
        return v9.RecordReviewV9Tests.call(self, system, payload)

    def unobserved(self):
        return {key: deepcopy(self.state[key]) for key in ('question', 'context', 'needs')}

    def test_build_contracts_records_model_ids_and_separate_local_admission(self):
        state = self.unobserved()
        before = deepcopy(state)
        result = self.reviewer.build_contracts(state)
        self.assertEqual(state, before)
        self.assertEqual(result.trace['version'], 10)
        self.assertEqual(result.trace['response'], self.anchor_response)
        self.assertEqual(result.trace['request'], anchor_input(state))
        self.assertEqual(result.updates['contract_anchor_binding'], anchor_binding(state))
        self.assertEqual(result.trace['admitted_response']['contracts'][0]['items'][0]['anchor']['quote'],
                         state['question'].split('？')[0] + '？')
        self.assertNotIn('facts', result.trace['request'])
        self.assertNotIn('records', result.trace['request'])

    def test_contract_still_refuses_existing_evidence_without_model_call(self):
        for field in ('facts', 'pending_facts', 'record_registry'):
            state = self.unobserved()
            state[field] = {'already': 'observed'}
            calls = len(self.calls)
            with self.subTest(field=field), self.assertRaises(ReviewStageError):
                self.reviewer.build_contracts(state)
            self.assertEqual(len(self.calls), calls)

    def test_bad_anchor_response_records_failed_trace_and_does_not_commit(self):
        self.anchor_response['contracts'][0]['items'][0]['anchor_id'] = 'Q99'
        state = self.unobserved()
        before = deepcopy(state)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.build_contracts(state)
        self.assertEqual(state, before)
        self.assertEqual(caught.exception.trace['response'], self.anchor_response)
        self.assertNotIn('need_contract_hash', caught.exception.trace)

    def test_need_updates_match_v9_for_supported_partial_unknown_and_refuted(self):
        baseline = deepcopy(self.response)
        cases = [baseline]
        partial = deepcopy(baseline)
        partial['assessments'][0]['items'][1] = v9.item('C2')
        cases.append(partial)
        refuted = deepcopy(baseline)
        refuted['assessments'][1] = v9.RecordReviewV9Tests.refuted_second_need(self)
        cases.append(refuted)
        old = RecordReviewV9(self.sources, self.episodes, self.call)
        for response in cases:
            self.response = response
            before = deepcopy(self.state)
            prior = old.review_needs(self.state, [0, 1])
            result = self.reviewer.review_needs(self.state, [0, 1])
            self.assertEqual(result.updates, prior.updates)
            self.assertEqual(result.trace['request'], prior.trace['request'])
            self.assertEqual(result.trace['version'], 10)
            self.assertEqual(self.state, before)

    def test_newly_admitted_anchor_contract_runs_existing_support_gate(self):
        updates = self.reviewer.build_contracts(self.unobserved()).updates
        self.state.update(updates)
        self.response['assessments'][0]['items'][1] = v9.item('C2')
        result = self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'partial')
        self.assertNotIn(0, self.state['resolved_needs'])
        self.assertTrue(any(g['need_index'] == 0 and g['item_id'] == 'C2' for g in contract_gaps(self.state)))

    def test_need_direct_record_citations_and_batch_atomicity_remain_enforced(self):
        self.response['assessments'][0]['items'][0]['record_ids'] = ['R1']
        before = deepcopy(self.state)
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(self.state, before)

    def test_withdrawn_fact_invalidates_previously_supported_need(self):
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertIn(0, self.state['resolved_needs'])
        self.state['facts'] = self.state['facts'][1:]
        self.assertTrue(any(g['need_index'] == 0 for g in contract_gaps(self.state)))


if __name__ == '__main__':
    unittest.main()
