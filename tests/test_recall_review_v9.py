"""Offline contract gates, direct citations and immutable NEEDS updates.

Provider verdicts here are scripted. These tests validate program invariants,
not whether a language model correctly judges a real story's meaning.
"""
from copy import deepcopy
import unittest

from config.prompt_config.recall_v9_prompts import CONTRACT, NEEDS
from memory_demo.retrieval.recall_need_contracts import (
    bound_contracts, contract_gaps, validate_contracts,
)
from memory_demo.retrieval.recall_review_v3 import FACTS_SYSTEM, ReviewStageError
from memory_demo.retrieval.recall_review_v4 import MAP_SYSTEM
from memory_demo.retrieval.recall_review_v9 import RecordReviewV9
import test_recall_review_v4 as fixtures


def item(item_id, status='unknown', *, value='', facts=(), records=()):
    return {'item_id': item_id, 'status': status, 'value': value,
            'fact_ids': list(facts), 'record_ids': list(records),
            'reason': 'Scripted offline evidence judgment.'}


class RecordReviewV9Tests(unittest.TestCase):
    def setUp(self):
        fixtures.RecordReviewV4Tests.setUp(self)
        self.state.update(
            question='停电后，甲是否批准乙的申请，批准是否仅限当晚？丙已经回来，他何时回来？',
            context='只依据已呈现记录，不推断次日。',
            needs=['甲是否批准乙的申请，且批准是否仅限当晚？', '丙何时回来？'])
        self.commit(self.reviewer.map(self.records[1] + self.records[2], self.state))
        self.commit(self.reviewer.review_facts(self.state))
        self.contract_rows = [
            {'need_index': 0, 'answer_type': 'boolean', 'items': [
                {'kind': 'answer', 'description': '甲是否批准这次申请',
                 'anchor': {'field': 'question', 'quote': '甲是否批准乙的申请'}},
                {'kind': 'constraint', 'description': '批准是否仅限当晚',
                 'anchor': {'field': 'question', 'quote': '仅限当晚'}}]},
            {'need_index': 1, 'answer_type': 'time', 'items': [
                {'kind': 'answer', 'description': '丙回来的时间',
                 'anchor': {'field': 'question', 'quote': '何时回来'}},
                {'kind': 'premise', 'description': '丙已经回来',
                 'anchor': {'field': 'question', 'quote': '丙已经回来'}}]},
        ]
        self.state.update(validate_contracts(self.state, self.contract_rows))
        self.state['contract_work'] = {'status': 'committed'}
        self.response = {'assessments': [
            {'need_index': 0, 'items': [
                item('C1', 'supported', value='甲批准了这次申请。', facts=['F1'], records=['R4']),
                item('C2', 'supported', value='批准仅限当晚。', facts=['F1'], records=['R5'])]},
            {'need_index': 1, 'items': [item('C1'), item('C2')]},
        ]}
        self.calls.clear()
        self.reviewer = RecordReviewV9(self.sources, self.episodes, self.call)

    commit = fixtures.RecordReviewV4Tests.commit

    def call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if system == MAP_SYSTEM:
            return {'facts': [
                {'record_ids': ['R4', 'R5'], 'episode_id': 1,
                 'interpretation': '甲批准了申请，仅限当晚。', 'need_indices': [0]},
                {'record_ids': ['R6'], 'episode_id': 2,
                 'interpretation': '丙仍未回来。', 'need_indices': [1]}], 'links': []}
        if system == FACTS_SYSTEM:
            return {'fact_decisions': [
                {'fact_id': fact['fact_id'], 'decision': 'accept', 'episode_alignment': 'supported',
                 'statement': '甲的批准仅限当晚。' if fact['episode_id'] == 1 else '丙仍未回来。',
                 'reason': 'Synthetic original records support their corresponding Episode.'}
                for fact in payload['facts']]}
        if system == CONTRACT:
            return {'contracts': deepcopy(self.contract_rows)}
        self.assertEqual(system, NEEDS)
        return deepcopy(self.response)

    def refuted_second_need(self, *, fact='F2'):
        return {'need_index': 1, 'items': [item('C1'),
            item('C2', 'contradicted', value='前提不成立：丙仍未回来。',
                 facts=[fact], records=['R6'])]}

    def test_contract_normalization_uses_original_anchors_and_local_item_ids(self):
        before = deepcopy(self.state)
        rows = deepcopy(self.contract_rows[::-1])
        rows[1]['items'].append({'kind': 'constraint', 'description': '不推断次日',
                                'anchor': {'field': 'context', 'quote': '不推断次日'}})
        original = deepcopy(rows)
        updates = validate_contracts(self.state, rows)
        self.assertEqual(self.state, before)
        self.assertEqual(rows, original)
        self.assertEqual([row['need_index'] for row in updates['need_contracts']], [0, 1])
        self.assertEqual([i['item_id'] for i in updates['need_contracts'][0]['items']], ['C1', 'C2', 'C3'])
        self.assertEqual([i['item_id'] for i in updates['need_contracts'][1]['items']], ['C1', 'C2'])
        self.assertNotEqual(updates['need_contract_hash'], self.state['need_contract_hash'])

    def test_invalid_contract_shapes_or_nonliteral_anchors_do_not_update_state(self):
        cases = [[], self.contract_rows[:1], self.contract_rows * 2]
        for mutate in (
            lambda rows: rows[0].update(need_index=True),
            lambda rows: rows[0].update(need_index=7),
            lambda rows: rows[0].update(answer_type='complete'),
            lambda rows: rows[0].update(extra='not permitted'),
            lambda rows: rows[0]['items'][0].update(kind='constraint'),
            lambda rows: rows[0]['items'][1].update(kind='answer'),
            lambda rows: rows[0]['items'][0].update(item_id='C1'),
            lambda rows: rows[0]['items'][0]['anchor'].update(field='needs'),
            lambda rows: rows[0]['items'][0]['anchor'].update(quote='原文没有的锚点'),
            lambda rows: rows[0]['items'][0]['anchor'].update(quote=''),
            lambda rows: rows[0]['items'].extend([deepcopy(rows[0]['items'][1])] * 5),
        ):
            rows = deepcopy(self.contract_rows)
            mutate(rows)
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows):
                before = deepcopy(self.state)
                with self.assertRaises(ValueError):
                    validate_contracts(self.state, rows)
                self.assertEqual(self.state, before)

    def test_contract_model_sees_question_and_context_but_no_evidence(self):
        unobserved = {key: deepcopy(self.state[key]) for key in ('question', 'context', 'needs')}
        before = deepcopy(unobserved)
        result = self.reviewer.build_contracts(unobserved)
        self.assertEqual(unobserved, before)
        self.assertEqual(result.updates['need_contract_hash'], self.state['need_contract_hash'])
        system, payload = self.calls[-1]
        self.assertEqual(system, CONTRACT)
        self.assertEqual(set(payload), {'request_context', 'context', 'targets'})
        self.assertEqual(payload['request_context'], self.state['question'])
        self.assertEqual(payload['context'], self.state['context'])
        self.assertEqual([target['question'] for target in payload['targets']], self.state['needs'])
        self.assertEqual(result.trace['version'], 9)

    def test_contract_construction_refuses_prior_evidence_before_provider(self):
        for field in ('facts', 'pending_facts', 'record_registry'):
            with self.subTest(field=field):
                state = {key: deepcopy(self.state[key]) for key in ('question', 'context', 'needs')}
                state[field] = deepcopy(self.state['facts'] if field == 'pending_facts' else self.state[field])
                before = deepcopy(state)
                count = len(self.calls)
                with self.assertRaises(ReviewStageError):
                    self.reviewer.build_contracts(state)
                self.assertEqual(state, before)
                self.assertEqual(len(self.calls), count)

    def test_nonstring_contract_enum_values_fail_as_schema_errors(self):
        for field in ('answer_type', 'kind', 'field'):
            with self.subTest(field=field):
                rows = deepcopy(self.contract_rows)
                target = (rows[0] if field == 'answer_type' else rows[0]['items'][0]
                          if field == 'kind' else rows[0]['items'][0]['anchor'])
                target[field] = []
                before = deepcopy(self.state)
                with self.assertRaises(ValueError):
                    validate_contracts(self.state, rows)
                self.assertEqual(self.state, before)

    def test_sufficient_and_unknown_targets_are_independent_and_do_not_change_facts(self):
        before = deepcopy(self.state)
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(self.state, before)
        self.assertNotIn('facts', result.updates)
        self.assertNotIn('verified_links', result.updates)
        self.assertEqual(result.updates['resolved_needs'], [0])
        self.assertEqual(result.updates['covered_needs'], [0])
        verdicts = result.updates['need_assessments']
        self.assertEqual([v['status'] for v in verdicts], ['supported', 'unknown'])
        self.assertEqual(verdicts[0]['answer'], '甲批准了这次申请。')
        self.assertEqual(verdicts[0]['contract_hash'], self.state['need_contract_hash'])
        self.assertEqual(verdicts[0]['item_assessments'][0]['fact_ids'], [self.state['facts'][0]['fact_id']])
        self.assertEqual(verdicts[0]['item_assessments'][0]['record_ids'], [self.records[1][3]['record_id']])
        self.assertEqual(result.trace['version'], 9)

    def test_known_answer_with_unknown_constraint_keeps_gap_open(self):
        self.response['assessments'][0]['items'][1] = item('C2')
        result = self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'partial')
        self.assertEqual(result.updates['resolved_needs'], [])
        gaps = contract_gaps(self.state)
        self.assertEqual([gap['item_id'] for gap in gaps if gap['need_index'] == 0], ['C2'])

    def test_answer_type_still_requires_an_answer_even_when_constraint_is_supported(self):
        self.response['assessments'][0]['items'][0] = item('C1')
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'partial')
        self.assertEqual(result.updates['need_assessments'][0]['answer'], '')
        self.assertEqual(result.updates['resolved_needs'], [])

    def test_explicit_premise_refutation_resolves_without_inventing_normal_answer(self):
        self.response['assessments'][1] = self.refuted_second_need()
        result = self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertEqual(result.updates['resolved_needs'], [0, 1])
        self.assertEqual(result.updates['covered_needs'], [0])
        self.assertEqual(result.updates['need_assessments'][1]['status'], 'refuted')
        self.assertEqual(result.updates['need_assessments'][1]['answer'], '前提不成立：丙仍未回来。')
        self.assertEqual(contract_gaps(self.state), [])

    def test_nonpremise_constraint_contradiction_cannot_refute_whole_need(self):
        self.response['assessments'][0]['items'][1] = item(
            'C2', 'contradicted', value='本项限定被明确反证。', facts=['F1'], records=['R5'])
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'partial')
        self.assertEqual(result.updates['resolved_needs'], [])

    def test_supported_answer_and_contradicted_premise_reject_entire_batch(self):
        self.response['assessments'][1] = self.refuted_second_need()
        self.response['assessments'][1]['items'][0] = item(
            'C1', 'supported', value='模型编造的归来日期。', facts=['F2'], records=['R6'])
        before = deepcopy(self.state)
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(self.state, before)

    def test_complete_visible_context_is_not_permission_to_cite_unbound_records(self):
        baseline = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(baseline.trace['request']['records']), 6)
        self.assertEqual(baseline.trace['request']['facts'][0]['record_ids'], ['R4', 'R5'])
        self.response['assessments'][0]['items'][0]['record_ids'] = ['R1']
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0, 1])

    def test_invalid_citation_sets_or_empty_supported_values_are_atomic(self):
        original = deepcopy(self.response)
        for changes in (
            {'fact_ids': []}, {'record_ids': []}, {'value': ''}, {'value': '   '},
            {'fact_ids': ['F99']}, {'record_ids': ['R99']},
            {'fact_ids': ['F1', 'F1']}, {'record_ids': ['R4', 'R4']},
            {'fact_ids': ['F1'], 'record_ids': ['R6']},
            {'fact_ids': ['F1', 'F2'], 'record_ids': ['R4']},
        ):
            with self.subTest(changes=changes):
                self.response = deepcopy(original)
                self.response['assessments'][0]['items'][0].update(changes)
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError):
                    self.reviewer.review_needs(self.state, [0, 1])
                self.assertEqual(self.state, before)

    def test_multiple_cited_facts_each_require_their_own_direct_record(self):
        self.response['assessments'][0]['items'][0].update(
            fact_ids=['F1', 'F2'], record_ids=['R4', 'R6'])
        result = self.reviewer.review_needs(self.state, [0, 1])
        refs = result.updates['need_assessments'][0]['item_assessments'][0]
        self.assertEqual(refs['fact_ids'], [f['fact_id'] for f in self.state['facts']])
        self.assertEqual(len(refs['record_ids']), 2)

    def test_missing_duplicate_extra_or_old_response_schema_rejects_entire_batch(self):
        base = deepcopy(self.response)
        cases = [{}, {'assessments': []}, {'assessments': base['assessments'][:1]},
                 {'assessments': base['assessments'] * 2}]
        for mutate in (
            lambda rows: rows[0].update(need_index=True),
            lambda rows: rows[0].update(status='supported'),
            lambda rows: rows[0].update(items=rows[0]['items'][:1]),
            lambda rows: rows[0]['items'].append(deepcopy(rows[0]['items'][0])),
            lambda rows: rows[0]['items'][0].update(item_id='C99'),
            lambda rows: rows[0]['items'][0].update(status='partial'),
            lambda rows: rows[0]['items'][0].update(extra='not allowed'),
        ):
            rows = deepcopy(base['assessments'])
            mutate(rows)
            cases.append({'assessments': rows})
        cases.append({'assessments': [{'need_index': 0, 'status': 'supported',
                      'fact_ids': ['F1'], 'answer': '旧协议答案', 'reason': '旧协议理由'}]})
        for response in cases:
            with self.subTest(response=response):
                self.response = response
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError):
                    self.reviewer.review_needs(self.state, [0, 1])
                self.assertEqual(self.state, before)

    def test_nonstring_item_identifiers_and_statuses_are_recoverable_schema_errors(self):
        for field in ('item_id', 'status'):
            with self.subTest(field=field):
                original = deepcopy(self.response)
                self.response['assessments'][0]['items'][0][field] = []
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError):
                    self.reviewer.review_needs(self.state, [0, 1])
                self.assertEqual(self.state, before)
                self.response = original

    def test_unknown_items_can_cite_partial_evidence_without_resolving_gap(self):
        self.response['assessments'][0]['items'][0] = item(
            'C1', value='相关但尚不足以给出答案。', facts=['F1'], records=['R4'])
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'partial')
        self.assertEqual(result.updates['resolved_needs'], [])

    def test_withdrawing_support_invalidates_prior_unreviewed_need(self):
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertEqual(self.state['resolved_needs'], [0])
        self.state['facts'] = self.state['facts'][1:]
        self.response = {'assessments': [self.refuted_second_need(fact='F1')]}
        result = self.reviewer.review_needs(self.state, [1])
        self.assertEqual(result.updates['need_assessments'][0]['status'], 'unknown')
        self.assertEqual(result.updates['need_assessments'][0]['fact_ids'], [])
        self.assertEqual(result.updates['resolved_needs'], [1])

    def test_unchanged_prior_need_remains_valid_when_another_target_is_reviewed(self):
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        original = deepcopy(self.state['need_assessments'][0])
        self.response = {'assessments': [self.refuted_second_need()]}
        result = self.reviewer.review_needs(self.state, [1])
        self.assertEqual(result.updates['need_assessments'][0], original)
        self.assertEqual(result.updates['resolved_needs'], [0, 1])

    def test_contract_binding_changes_fail_before_provider(self):
        original = deepcopy(self.state)
        for mutate in (
            lambda state: state.update(question=state['question'] + ' changed'),
            lambda state: state.update(context=state['context'] + ' changed'),
            lambda state: state['needs'].__setitem__(0, '另一个问题'),
            lambda state: state.update(need_contract_hash='changed'),
            lambda state: state['need_contracts'][0]['items'][0].update(item_id='C99'),
            lambda state: state['need_contracts'][0]['items'][0].update(description='changed'),
        ):
            self.state = deepcopy(original)
            mutate(self.state)
            before = deepcopy(self.state)
            count = len(self.calls)
            with self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(self.state, [0])
            self.assertEqual(self.state, before)
            self.assertEqual(len(self.calls), count)

    def test_uncommitted_or_invalid_contract_retains_completion_blockers(self):
        self.response['assessments'][1] = self.refuted_second_need()
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.assertEqual(contract_gaps(self.state), [])
        self.state['contract_work']['status'] = 'started'
        self.assertEqual({gap['status'] for gap in contract_gaps(self.state)}, {'contract_not_committed'})
        self.state['contract_work']['status'] = 'committed'
        self.state['need_contract_hash'] = 'invalid'
        self.assertEqual({gap['status'] for gap in contract_gaps(self.state)}, {'invalid_contract'})

    def test_no_accepted_facts_does_not_call_provider_or_keep_resolved_items(self):
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.state['pending_facts'] = self.state['facts']
        self.state['facts'] = []
        count = len(self.calls)
        result = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(self.calls), count)
        self.assertEqual(result.updates['resolved_needs'], [])
        self.assertTrue(all(v['status'] == 'unknown' for v in result.updates['need_assessments']))

    def test_invalid_targets_or_changed_source_are_rejected_before_provider(self):
        count = len(self.calls)
        for indices in ([], [0, 0], [True], [2], '0'):
            with self.subTest(indices=indices), self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(self.state, indices)
        self.state['review_context_max_chars'] = 1
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0])
        self.state.pop('review_context_max_chars')
        self.sources[1] += ' changed'
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_needs(self.state, [0])
        self.assertEqual(len(self.calls), count)


if __name__ == '__main__':
    unittest.main()
