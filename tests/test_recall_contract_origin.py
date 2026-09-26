"""Source-free origin sidecar and unchanged V12 evidence gates, entirely offline.

Origin labels and evidence judgments are scripted here. Hashes bind inputs;
they do not prove a model's semantic classification or authenticate a provider.
The projection never has authority to complete a recall.
"""
from copy import deepcopy
import json
import unittest

from config.prompt_config.recall_v9_prompts import NEEDS
from memory_demo.retrieval.recall_contract_origin import (
    ORIGIN_NEEDS, ORIGIN_PROTOCOL, RecordReviewOrigin, bound_origin,
    origin_input, origin_projection, validate_origin,
)
from memory_demo.retrieval.recall_need_contracts import validate_contracts
from memory_demo.retrieval.recall_review_v3 import ReviewStageError, _hash
from memory_demo.retrieval.recall_review_v12 import RecordReviewV12
import test_recall_review_v9 as v9


class ContractOriginTests(unittest.TestCase):
    commit = v9.RecordReviewV9Tests.commit

    def setUp(self):
        v9.RecordReviewV9Tests.setUp(self)
        self.origin_response = {'origins': [
            {'need_index': contract['need_index'], 'item_id': item['item_id'],
             'origin': 'original_obligation', 'reason': '仅依据原问的脚本分类。'}
            for contract in self.state['need_contracts'] for item in contract['items']]}
        self.sidecar = validate_origin(self.state, self.origin_response)
        self.base = RecordReviewV12(self.sources, self.episodes, self.call)
        self.reviewer = self.treatment()

    def call(self, system, payload):
        if system == ORIGIN_NEEDS:
            self.calls.append((system, deepcopy(payload)))
            return deepcopy(self.response)
        return v9.RecordReviewV9Tests.call(self, system, payload)

    def treatment(self, sidecar=None, call=None):
        return RecordReviewOrigin(self.sources, self.episodes, call or self.call,
                                  origin=self.sidecar if sidecar is None else sidecar)

    def labels(self, **changes):
        response = deepcopy(self.origin_response)
        for row in response['origins']:
            row['origin'] = changes.get(f"{row['need_index']}_{row['item_id']}", row['origin'])
        return validate_origin(self.state, response)

    def judgments(self):
        return self.base.review_needs(self.state, [0, 1]).updates['need_assessments']

    def test_origin_input_allowlists_request_and_complete_contracts_without_evidence(self):
        state = deepcopy(self.state)
        state.update(source_text='PRIVATE_SOURCE_SENTINEL', answer='ANSWER_SENTINEL',
                     review_trace=[{'response': 'TRACE_SENTINEL'}],
                     cue_queue={'pending': ['CUE_SENTINEL']},
                     provider={'api_key': 'KEY_SENTINEL'}, gold={'ids': ['GOLD_SENTINEL']})
        before = deepcopy(state)
        payload = origin_input(state)
        self.assertEqual(set(payload), {'request_context', 'context', 'contracts'})
        self.assertEqual(payload['contracts'], state['need_contracts'])
        self.assertEqual(payload['request_context'], state['question'])
        self.assertEqual(payload['context'], state['context'])
        serialized = json.dumps(payload, ensure_ascii=False)
        for marker in ('PRIVATE_SOURCE_SENTINEL', 'ANSWER_SENTINEL', 'TRACE_SENTINEL',
                       'CUE_SENTINEL', 'KEY_SENTINEL', 'GOLD_SENTINEL',
                       self.state['facts'][0]['fact_id'], self.records[1][0]['record_id']):
            self.assertNotIn(marker, serialized)
        self.assertEqual(state, before)

    def test_origin_input_and_bound_output_are_detached(self):
        before, sidecar = deepcopy(self.state), deepcopy(self.sidecar)
        payload = origin_input(self.state)
        payload['contracts'][0]['items'][0]['description'] = 'changed'
        bound = bound_origin(self.state, self.sidecar)
        bound['origins'][0]['reason'] = 'changed'
        bound['protocol']['version'] = 99
        self.assertEqual(self.state, before)
        self.assertEqual(self.sidecar, sidecar)
        self.assertEqual(ORIGIN_PROTOCOL['version'], 1)

    def test_origin_admission_normalizes_order_without_losing_any_item(self):
        response = deepcopy(self.origin_response)
        response['origins'].reverse()
        before = deepcopy(response)
        result = validate_origin(self.state, response)
        self.assertEqual(result, self.sidecar)
        self.assertEqual(response, before)
        self.assertEqual([(r['need_index'], r['item_id']) for r in result['origins']],
                         [(0, 'C1'), (0, 'C2'), (1, 'C1'), (1, 'C2')])
        self.assertEqual(result['question_contract_sha256'], _hash(origin_input(self.state)))
        digest_input = {k: v for k, v in result.items() if k != 'sidecar_sha256'}
        self.assertEqual(result['sidecar_sha256'], _hash(digest_input))

    def test_origin_response_unknown_duplicate_missing_or_extra_items_are_rejected(self):
        cases = [{}, {'origins': []}, {'origins': {}},
                 {'origins': self.origin_response['origins'][:-1]},
                 {'origins': self.origin_response['origins'] * 2}]
        for mutation in (
            lambda r: r.update(extra='not permitted'),
            lambda r: r['origins'][0].update(need_index=True),
            lambda r: r['origins'][0].update(need_index=99),
            lambda r: r['origins'][0].update(item_id='C99'),
            lambda r: r['origins'][0].update(item_id=[]),
            lambda r: r['origins'][0].update(origin='unknown'),
            lambda r: r['origins'][0].update(origin=[]),
            lambda r: r['origins'][0].update(reason=''),
            lambda r: r['origins'][0].update(reason='字' * 181),
            lambda r: r['origins'][0].update(fact_ids=['F1']),
            lambda r: r['origins'][0].pop('item_id'),
        ):
            response = deepcopy(self.origin_response)
            mutation(response)
            cases.append(response)
        for response in cases:
            with self.subTest(response=response):
                before, original = deepcopy(self.state), deepcopy(response)
                with self.assertRaises(ValueError):
                    validate_origin(self.state, response)
                self.assertEqual(self.state, before)
                self.assertEqual(response, original)

    def test_sidecar_tampering_is_rejected_even_after_rehashing_wrong_input_binding(self):
        for mutate in (
            lambda x: x.update(question_contract_sha256='forged'),
            lambda x: x.update(need_contract_hash='forged'),
            lambda x: x['protocol'].update(evidence_access=True),
            lambda x: x['origins'].pop(),
            lambda x: x.update(completion_authorized=True),
        ):
            changed = deepcopy(self.sidecar)
            mutate(changed)
            changed['sidecar_sha256'] = _hash({k: v for k, v in changed.items()
                                              if k != 'sidecar_sha256'})
            with self.subTest(sidecar=changed), self.assertRaises(ValueError):
                bound_origin(self.state, changed)
        changed = deepcopy(self.sidecar)
        changed['origins'][0]['origin'] = 'model_hypothesis'
        with self.assertRaises(ValueError):
            bound_origin(self.state, changed)

    def test_sidecar_cannot_transfer_to_another_valid_question_context_or_contract(self):
        for field in ('question', 'context', 'description', 'needs'):
            state, rows = deepcopy(self.state), deepcopy(self.contract_rows)
            if field in ('question', 'context'):
                state[field] += ' 新的限定。'
            elif field == 'description':
                rows[0]['items'][0]['description'] += '另一解释'
            else:
                state['needs'][0] += '新限定'
            state.update(validate_contracts(state, rows))
            with self.subTest(field=field), self.assertRaises(ValueError):
                bound_origin(state, self.sidecar)

    def test_nonliteral_or_relabelled_contract_items_fail_before_origin_input(self):
        for mutation in (
            lambda s: s['need_contracts'][0]['items'][0]['anchor'].update(quote='不在原题的文字'),
            lambda s: s['need_contracts'][0]['items'][0]['anchor'].update(field='source'),
            lambda s: s['need_contracts'][0]['items'][0].update(item_id='C99'),
            lambda s: s['need_contracts'][0]['items'][0].update(description='伪造契约'),
            lambda s: s['need_contracts'][0]['items'].pop(),
        ):
            state = deepcopy(self.state)
            mutation(state)
            before = deepcopy(state)
            with self.subTest(state=state), self.assertRaises(ValueError):
                origin_input(state)
            self.assertEqual(state, before)

    def test_projection_preserves_legacy_judgment_and_never_authorizes_completion(self):
        before, sidecar = deepcopy(self.state), deepcopy(self.sidecar)
        assessments = self.judgments()
        original = deepcopy(assessments)
        projected = origin_projection(self.state, self.sidecar, assessments)
        self.assertEqual([r['legacy_status'] for r in projected['rows']], ['supported', 'unknown'])
        self.assertEqual([r['original_only_supported_diagnostic'] for r in projected['rows']], [True, False])
        self.assertIs(projected['completion_authorized'], False)
        self.assertIs(projected['search_evidence_created'], False)
        self.assertEqual(self.state, before)
        self.assertEqual(self.sidecar, sidecar)
        self.assertEqual(assessments, original)

    def test_hypothesis_unknown_can_only_change_diagnostic_not_legacy_gate(self):
        self.response['assessments'][0]['items'][1] = v9.item('C2')
        sidecar = self.labels(**{'0_C2': 'model_hypothesis'})
        result = self.treatment(sidecar).review_needs(self.state, [0, 1])
        self.assertEqual(result.updates['resolved_needs'], [])
        projected = origin_projection(self.state, sidecar, result.updates['need_assessments'])
        self.assertEqual(projected['rows'][0]['legacy_status'], 'partial')
        self.assertTrue(projected['rows'][0]['original_only_supported_diagnostic'])
        self.assertEqual(projected['rows'][0]['hypotheses'][0]['status'], 'unknown')
        self.assertFalse(projected['completion_authorized'])

    def test_mixed_or_unknown_required_items_remain_conservative(self):
        self.response['assessments'][0]['items'][1] = v9.item('C2')
        assessments = self.judgments()
        for origin in ('original_obligation', 'mixed_uncertain'):
            sidecar = self.labels(**{'0_C2': origin})
            with self.subTest(origin=origin):
                projected = origin_projection(self.state, sidecar, assessments)
                self.assertFalse(projected['rows'][0]['original_only_supported_diagnostic'])
                self.assertFalse(projected['completion_authorized'])

    def test_missing_original_answer_and_all_hypotheses_cannot_pass_vacuously(self):
        assessments = self.judgments()
        for changes in ({'0_C1': 'model_hypothesis'}, {'0_C1': 'mixed_uncertain'},
                        {'0_C1': 'model_hypothesis', '0_C2': 'model_hypothesis'}):
            sidecar = self.labels(**changes)
            with self.subTest(changes=changes):
                projected = origin_projection(self.state, sidecar, assessments)
                self.assertFalse(projected['rows'][0]['original_only_supported_diagnostic'])
                self.assertFalse(projected['completion_authorized'])
        empty = origin_projection(self.state, self.sidecar, [])
        self.assertEqual(empty['rows'], [])
        self.assertFalse(empty['completion_authorized'])

    def test_hypothesis_premise_refutation_does_not_authorize_original_completion(self):
        self.response['assessments'][1] = v9.RecordReviewV9Tests.refuted_second_need(self)
        sidecar = self.labels(**{'1_C2': 'model_hypothesis'})
        result = self.treatment(sidecar).review_needs(self.state, [0, 1])
        # The probe deliberately preserves the old validator, including this
        # legacy refutation. The diagnostic does not promote it to authority.
        self.assertEqual(result.updates['need_assessments'][1]['status'], 'refuted')
        projected = origin_projection(self.state, sidecar, result.updates['need_assessments'])
        self.assertFalse(projected['rows'][1]['original_only_supported_diagnostic'])
        self.assertFalse(projected['completion_authorized'])

    def test_projection_rejects_unbound_incomplete_or_duplicate_assessments(self):
        original = self.judgments()
        cases = [original * 2]
        for mutation in (
            lambda rows: rows[0].update(need_index=True),
            lambda rows: rows[0].update(need_index=99),
            lambda rows: rows[0].update(contract_hash='wrong'),
            lambda rows: rows[0]['item_assessments'].pop(),
            lambda rows: rows[0]['item_assessments'].append(deepcopy(rows[0]['item_assessments'][0])),
            lambda rows: rows[0]['item_assessments'][0].update(item_id='C99'),
            lambda rows: rows[0]['item_assessments'][0].update(fact_ids=['not-accepted']),
            lambda rows: rows[0]['item_assessments'][0].update(record_ids=[self.records[1][0]['record_id']]),
        ):
            changed = deepcopy(original)
            mutation(changed)
            cases.append(changed)
        for assessments in cases:
            with self.subTest(assessments=assessments):
                before = deepcopy(self.state)
                with self.assertRaises(ValueError):
                    origin_projection(self.state, self.sidecar, assessments)
                self.assertEqual(self.state, before)

    def test_treatment_adds_only_origin_payload_key_and_keeps_all_updates_equal(self):
        before = deepcopy(self.state)
        control = self.base.review_needs(self.state, [0, 1])
        treatment = self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][0], NEEDS)
        self.assertEqual(self.calls[1][0], ORIGIN_NEEDS)
        actual = deepcopy(treatment.trace['request'])
        added = actual.pop('contract_origin')
        self.assertEqual(actual, control.trace['request'])
        self.assertEqual(added['origins'], self.sidecar['origins'])
        self.assertEqual(treatment.updates, control.updates)
        self.assertEqual(treatment.followup_cues, control.followup_cues)
        self.assertEqual(treatment.trace['response'], self.response)
        self.assertEqual(treatment.trace['version'], 12)
        self.assertEqual(self.state, before)
        self.assertIsNone(self.reviewer._current_origin)

    def test_subset_need_batch_receives_only_its_original_sidecar_rows(self):
        self.response = {'assessments': self.response['assessments'][1:]}
        result = self.reviewer.review_needs(self.state, [1])
        rows = result.trace['request']['contract_origin']['origins']
        self.assertEqual([(r['need_index'], r['item_id']) for r in rows], [(1, 'C1'), (1, 'C2')])
        self.assertEqual(result.trace['request']['targets'], [self.state['need_contracts'][1]])

    def test_constructor_detaches_sidecar_and_bad_binding_fails_before_provider(self):
        original = deepcopy(self.sidecar)
        self.sidecar['origins'][0]['reason'] = 'changed after constructing reviewer'
        self.reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(self.reviewer.origin, original)
        count, before = len(self.calls), deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.treatment().review_needs(self.state, [0, 1])
        self.assertEqual(len(self.calls), count)
        self.assertEqual(self.state, before)

    def test_treatment_still_rejects_unaccepted_or_indirect_citations(self):
        original = deepcopy(self.response)
        for refs in ({'fact_ids': ['F99']}, {'record_ids': ['R99']},
                     {'fact_ids': []}, {'record_ids': []},
                     {'record_ids': ['R1']}, {'record_ids': ['R6']},
                     {'fact_ids': ['F1', 'F2'], 'record_ids': ['R4']},
                     {'record_ids': ['R4', 'R4']}, {'value': ''}):
            with self.subTest(refs=refs):
                self.response = deepcopy(original)
                self.response['assessments'][0]['items'][0].update(refs)
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError):
                    self.reviewer.review_needs(self.state, [0, 1])
                self.assertEqual(self.state, before)
                self.assertIsNone(self.reviewer._current_origin)

    def test_treatment_still_rejects_missing_duplicate_and_unknown_response_items(self):
        original = deepcopy(self.response)
        for mutation in (
            lambda r: r['assessments'].pop(),
            lambda r: r['assessments'].append(deepcopy(r['assessments'][0])),
            lambda r: r['assessments'][0]['items'].pop(),
            lambda r: r['assessments'][0]['items'].append(deepcopy(r['assessments'][0]['items'][0])),
            lambda r: r['assessments'][0]['items'][0].update(item_id='C99'),
            lambda r: r['assessments'][0]['items'][0].update(origin='model_hypothesis'),
            lambda r: r.update(completion_authorized=True),
        ):
            self.response = deepcopy(original)
            mutation(self.response)
            before = deepcopy(self.state)
            with self.subTest(response=self.response), self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(self.state, [0, 1])
            self.assertEqual(self.state, before)

    def test_no_accepted_evidence_skips_provider_and_keeps_all_items_unknown(self):
        state = deepcopy(self.state)
        state['facts'] = []
        before = deepcopy(state)
        result = self.reviewer.review_needs(state, [0, 1])
        self.assertEqual(self.calls, [])
        self.assertEqual(result.trace['skipped'], 'no_accepted_facts')
        self.assertEqual(result.updates['resolved_needs'], [])
        self.assertTrue(all(i['status'] == 'unknown' for a in result.updates['need_assessments']
                            for i in a['item_assessments']))
        self.assertEqual(state, before)

    def test_damaged_registry_and_fact_status_fail_before_provider(self):
        for mutation in (
            lambda s: s['record_registry'][self.records[1][0]['record_id']].update(text='corrupted'),
            lambda s: s['facts'][0].update(verification_status='pending'),
            lambda s: s['facts'][0].update(record_ids=['missing-record']),
        ):
            state = deepcopy(self.state)
            mutation(state)
            before, count = deepcopy(state), len(self.calls)
            with self.subTest(state=state), self.assertRaises(ReviewStageError):
                self.reviewer.review_needs(state, [0, 1])
            self.assertEqual(len(self.calls), count)
            self.assertEqual(state, before)

    def test_provider_failure_retains_state_and_clears_temporary_origin(self):
        calls = []
        def fail(system, payload):
            calls.append((system, deepcopy(payload)))
            raise TimeoutError('scripted deadline; no network')
        reviewer, before = self.treatment(call=fail), deepcopy(self.state)
        with self.assertRaises(ReviewStageError) as error:
            reviewer.review_needs(self.state, [0, 1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(error.exception.trace['request']['contract_origin']['sidecar_sha256'],
                         self.sidecar['sidecar_sha256'])
        self.assertIsNone(reviewer._current_origin)
        self.assertEqual(self.state, before)

    def test_other_stages_keep_original_prompts_payload_and_updates(self):
        before = deepcopy(self.state)
        control = self.base.map(self.records[1] + self.records[2], self.state)
        treatment = self.reviewer.map(self.records[1] + self.records[2], self.state)
        self.assertEqual(self.calls[-2], self.calls[-1])
        self.assertEqual(control.updates, treatment.updates)
        self.assertNotIn('contract_origin', treatment.trace['request'])
        control = self.base.review_facts(self.state, fact_ids=[self.state['facts'][0]['fact_id']])
        treatment = self.reviewer.review_facts(self.state, fact_ids=[self.state['facts'][0]['fact_id']])
        self.assertEqual(self.calls[-2], self.calls[-1])
        self.assertEqual(control.updates, treatment.updates)
        self.assertEqual(self.state, before)


if __name__ == '__main__':
    unittest.main()
