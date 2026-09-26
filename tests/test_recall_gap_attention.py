"""Pure current-gap selection and one-vector activation, using synthetic facts.

The scripted NEED judgments test provenance and transformation mechanics, not
the semantic reliability of a model's missing-evidence diagnosis.
"""
from copy import deepcopy
from dataclasses import asdict
import unittest

import numpy as np

from memory_demo.associations.spreading import SpreadingSeed
from memory_demo.retrieval.recall_contract_anchors import anchor_binding
from memory_demo.retrieval.recall_cues import _sha
from memory_demo.retrieval.recall_gap_attention import (
    GAP_ATTENTION_PROTOCOL, select_gap_attention, validate_gap_attention,
    expand_attention_vector,
)
from memory_demo.retrieval.recall_need_contracts import validate_contracts
from memory_demo.retrieval.recall_review_v3 import _hash
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10
import test_recall_review_v9 as fixture


class GapAttentionTests(unittest.TestCase):
    def setUp(self):
        self.commit_wave = 1
        fixture.RecordReviewV9Tests.setUp(self)
        self.state['contract_anchor_binding'] = anchor_binding(self.state)
        self.state['metrics'] = {'review_waves': 0}
        self.response['assessments'][0]['items'][1] = fixture.item('C2', value='已有局部线索，但尚不知批准范围。')
        self.response['assessments'][0]['items'][1]['reason'] = '直接记录尚未支持所问的时间限定。'
        self.response['assessments'][1] = fixture.RecordReviewV9Tests.refuted_second_need(self)
        self.reviewer = RecordReviewV10(self.sources, self.episodes, self.call)
        self.commit(self.reviewer.review_needs(self.state, [0, 1]))
        self.selection = select_gap_attention(self.state)
        self.calls.clear()

    def call(self, system, payload):
        return fixture.RecordReviewV9Tests.call(self, system, payload)

    def commit(self, result):
        self.state.update(deepcopy(result.updates))
        self.state.setdefault('review_trace', []).append({**deepcopy(result.trace),
            'committed': True, 'wave': self.commit_wave})
        return result

    def assess(self):
        return self.commit(self.reviewer.review_needs(self.state, [0, 1]))

    def test_one_lowest_current_gap_preserves_exact_criterion_value_and_reason(self):
        selected = self.selection['selected']
        self.assertEqual((selected['need_index'], selected['item_id']), (0, 'C2'))
        self.assertEqual(selected['criterion'], self.contract_rows[0]['items'][1]['description'])
        raw = self.response['assessments'][0]['items'][1]
        self.assertEqual(selected['current_value'], raw['value'])
        self.assertEqual(selected['missing_reason'], raw['reason'])
        self.assertEqual(self.selection['control_text'], '仅限当晚')
        self.assertTrue(self.selection['treatment_text'].startswith(self.selection['control_text'] + '\n'))
        self.assertIn(raw['reason'], self.selection['treatment_text'])
        self.assertIn('未验证', self.selection['treatment_text'])
        self.assertEqual(self.selection['protocol']['embedding_slots_per_arm'], 1)
        self.assertEqual(self.selection['provenance']['origin_trace_index'], 2)
        self.assertEqual(self.selection['provenance']['origin_wave'], 1)
        self.assertEqual(self.selection['provenance']['review_boundary_wave'], 2)

    def test_unknown_answer_precedes_constraint_and_higher_need(self):
        self.response['assessments'][0]['items'][0] = fixture.item('C1')
        self.response['assessments'][1] = {'need_index': 1, 'items': [fixture.item('C1'), fixture.item('C2')]}
        self.assess()
        self.assertEqual(select_gap_attention(self.state)['selected']['item_id'], 'C1')
        self.assertEqual(select_gap_attention(self.state)['selected']['need_index'], 0)

    def test_supported_and_refuted_needs_are_not_reopened_by_selector(self):
        self.response['assessments'][0]['items'][1] = fixture.item('C2', 'supported',
            value='这是脚本判定，并非真实语义准确率。', facts=['F1'], records=['R5'])
        self.assess()
        self.assertEqual([a['status'] for a in self.state['need_assessments']], ['supported', 'refuted'])
        self.assertIsNone(select_gap_attention(self.state))

    def test_contradicted_nonpremise_constraint_is_not_attention_and_next_unknown_is_selected(self):
        self.response['assessments'][0]['items'][1] = fixture.item('C2', 'contradicted',
            value='证据与限定冲突。', facts=['F1'], records=['R5'])
        self.assess()
        self.assertIsNone(select_gap_attention(self.state))
        self.response['assessments'][1] = {'need_index': 1, 'items': [fixture.item('C1'), fixture.item('C2')]}
        self.assess()
        selected = select_gap_attention(self.state)['selected']
        self.assertEqual((selected['need_index'], selected['item_id'], selected['status']), (1, 'C1', 'unknown'))

    def test_failed_skipped_noresponse_and_invalid_schema_are_never_origins(self):
        for changed in ({'committed': False}, {'schema_ok': False}, {'response': None},
                        {'skipped': 'no_accepted_facts'}):
            with self.subTest(changed=changed):
                state = deepcopy(self.state)
                state['review_trace'][-1].update(changed)
                self.assertIsNone(select_gap_attention(state))

    def test_same_fact_accept_again_invalidates_historical_gap_and_synthetic_reset(self):
        self.commit_wave = 2
        self.commit(self.reviewer.review_facts(self.state, fact_ids=[self.state['facts'][0]['fact_id']]))
        self.assertIsNone(select_gap_attention(self.state))
        self.assertTrue(all(a['status'] == 'unknown' for a in self.state['need_assessments']))
        with self.assertRaises(ValueError):
            validate_gap_attention(self.state, self.selection)

    def test_failed_later_need_does_not_supply_its_uncommitted_reason(self):
        failed = deepcopy(self.state['review_trace'][-1])
        failed.update(committed=False, schema_ok=False, error='provider deadline exceeded')
        failed['response']['assessments'][0]['items'][1]['reason'] = 'uncommitted hypothesis'
        self.state['review_trace'].append(failed)
        selected = select_gap_attention(self.state)
        self.assertEqual(selected['provenance']['origin_trace_index'], 2)
        self.assertNotIn('uncommitted hypothesis', selected['treatment_text'])

    def test_current_assessment_or_accepted_fact_changes_are_rejected(self):
        for change in (
            lambda s: s['need_assessments'][0].update(reason='synthetic replacement'),
            lambda s: s['facts'][0].update(statement='changed accepted meaning'),
            lambda s: s['facts'][0]['evidence'][0].update(quote='forged quote'),
            lambda s: s['facts'].pop(),
        ):
            state = deepcopy(self.state)
            change(state)
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_gap_attention(state)

    def test_raw_need_response_must_replay_to_its_parsed_decision(self):
        state = deepcopy(self.state)
        state['review_trace'][-1]['response']['assessments'][0]['items'][1]['reason'] = 'tampered missing diagnosis'
        with self.assertRaisesRegex(ValueError, 'raw NEED response'):
            select_gap_attention(state)

    def test_need_request_cannot_hide_changed_fact_statement_or_contract_target(self):
        for change in (
            lambda t: t['request']['facts'][0].update(statement='different evidence'),
            lambda t: t['request']['targets'][0]['items'][1].update(description='easier criterion'),
            lambda t: t['request'].update(request_context='different original question'),
        ):
            state = deepcopy(self.state)
            change(state['review_trace'][-1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_gap_attention(state)

    def test_need_request_records_reject_literal_speaker_alias_and_catalog_tamper(self):
        for change in (
            lambda t: t['request']['records'][0].update(text='invented visible evidence'),
            lambda t: t['request']['records'][0].update(speaker='different speaker'),
            lambda t: t['request']['records'][0].update(aliases={'zh-CN': 'invented alias'}),
            lambda t: t['request']['records'].pop(),
            lambda t: t['record_ids'].update(R1=t['record_ids']['R2']),
        ):
            state = deepcopy(self.state)
            change(state['review_trace'][-1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_gap_attention(state)

    def test_selection_rehashed_tamper_or_changed_protocol_is_rejected(self):
        for change in (
            lambda a: a['selected'].update(missing_reason='invented diagnosis'),
            lambda a: a.update(treatment_text='generated answer pretending to be search'),
            lambda a: a['provenance'].update(origin_trace_index=0),
            lambda a: a['protocol'].update(version=99),
        ):
            selected = deepcopy(self.selection)
            change(selected)
            selected['selection_sha256'] = _hash({k: v for k, v in selected.items() if k != 'selection_sha256'})
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_gap_attention(self.state, selected)

    def test_metadata_changes_do_not_rebind_selection_or_expose_mutable_references(self):
        before = deepcopy(self.state)
        selected = select_gap_attention(self.state)
        selected['selected']['criterion'] = 'changed detached output'
        self.assertEqual(self.state, before)
        self.state['metrics'].update(review_waves=12, embedding_batches=9)
        self.state['f1_attention_attempt'] = {'status': 'started', 'arm': 'gap'}
        self.state['elapsed_seconds'] = 99
        for trace in self.state['review_trace']:
            trace['stage_token'] = '[REDACTED]'
        validated = validate_gap_attention(self.state, self.selection)
        self.assertEqual(validated, self.selection)
        validated['provenance'].clear()
        self.assertTrue(self.selection['provenance'])
        self.assertEqual(self.calls, [])

    def test_context_anchor_is_used_literally_in_both_arms(self):
        self.contract_rows[0]['items'][1]['anchor'] = {'field': 'context', 'quote': '不推断次日'}
        self.state.update(validate_contracts(self.state, self.contract_rows))
        self.assess()
        selected = select_gap_attention(self.state)
        self.assertEqual(selected['selected']['request_anchor']['field'], 'context')
        self.assertEqual(selected['control_text'], '不推断次日')
        self.assertTrue(selected['treatment_text'].startswith('不推断次日\n'))

    def test_2048_character_bound_is_exact_and_never_selects_later_shorter_gap(self):
        row = self.response['assessments'][0]['items'][1]
        size = len(self.selection['treatment_text'])
        row['reason'] += '限' * (2048 - size)
        self.assess()
        self.assertEqual(len(select_gap_attention(self.state)['treatment_text']), 2048)
        row['reason'] += '限'
        self.assess()
        with self.assertRaisesRegex(ValueError, 'no truncation or alternative'):
            select_gap_attention(self.state)

    def test_missing_contract_is_inapplicable_but_changed_binding_is_invalid(self):
        state = deepcopy(self.state)
        state['contract_work']['status'] = 'failed'
        self.assertIsNone(select_gap_attention(state))
        state = deepcopy(self.state)
        state['contract_anchor_binding']['request_fragments_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            select_gap_attention(state)


class GapVectorTests(unittest.TestCase):
    call = GapAttentionTests.call
    commit = GapAttentionTests.commit

    def setUp(self):
        GapAttentionTests.setUp(self)
        self.sources[3] = '另一个尚未读取的来源。'
        self.episodes.update({3: {'id': 3, 'source_id': 3, 'text': '候选三'},
                              4: {'id': 4, 'source_id': 3, 'text': '候选四'}})
        self.ids = [1, 2, 3, 4]
        self.matrix = np.array([[1, 0], [0, 1], [.6, .8], [.6, .8]], dtype=np.float32)
        self.state.update(seeds=[asdict(SpreadingSeed('episode', 1, .8, self.root(1)))],
            fallback_episode_scores={'1': .9, '2': .1, '3': .2, '4': .2},
            fallback_episode_ids=[1, 3, 4, 2], spreading={'expansions': 7, 'opaque': 'checkpoint'},
            spreading_expansion_base=5, cue_queue={'pending': [], 'processed': ['unchanged']},
            learning={'status': 'disabled'})
        self.state['metrics'].update(seed_candidates_scored=4, edge_expansions=12,
            spreading_restarts=2, embedding_batches=3, followup_cues_processed=6)

    def root(self, sid):
        return 'source:' + _sha(self.sources[sid])

    def expand(self, vector=(0, 1), **kwargs):
        options = {'arm': 'gap', 'episode_ids': self.ids, 'episode_matrix': self.matrix,
            'episodes': self.episodes, 'sources': self.sources, 'seeds_per_cue': 1}
        options.update(kwargs)
        return expand_attention_vector(self.state, self.selection, vector, **options)

    def test_vector_adds_seed_with_source_root_and_accumulates_epoch_work(self):
        before, matrix = deepcopy(self.state), self.matrix.copy()
        updated, diag = self.expand()
        self.assertEqual(self.state, before)
        np.testing.assert_array_equal(self.matrix, matrix)
        self.assertEqual(diag['new_seed_episode_ids'], [2])
        self.assertEqual(diag['source_roots']['2'], self.root(2))
        self.assertTrue(diag['fingerprint_changed'])
        self.assertTrue(diag['restart'])
        self.assertIsNone(updated['spreading'])
        self.assertEqual(updated['spreading_expansion_base'], 12)
        self.assertEqual(updated['metrics']['spreading_restarts'], 3)
        self.assertEqual(updated['metrics']['seed_candidates_scored'], 8)
        self.assertEqual(updated['metrics']['edge_expansions'], 12)
        self.assertEqual(updated['fallback_episode_ids'], [2, 1, 3, 4])
        self.assertEqual(updated['fallback_episode_scores']['1'], .9)

    def test_only_allowed_search_fields_change_and_entire_result_is_detached(self):
        before = deepcopy(self.state)
        updated, diag = self.expand()
        changed = {k for k in updated if updated[k] != before.get(k)}
        self.assertLessEqual(changed, {'seeds', 'fallback_episode_scores', 'fallback_episode_ids',
            'spreading', 'spreading_expansion_base', 'metrics'})
        self.assertEqual(updated['metrics']['embedding_batches'], 3)
        self.assertEqual(updated['metrics']['followup_cues_processed'], 6)
        for key in ('facts', 'pending_facts', 'need_assessments', 'cue_queue', 'learning', 'review_trace'):
            self.assertEqual(updated[key], before[key])
            self.assertIsNot(updated[key], self.state[key])
        updated['facts'][0]['statement'] = 'detached'
        diag['seed_before'].clear()
        self.assertEqual(self.state, before)

    def test_same_vector_arms_have_identical_activation_and_one_slot_diagnostics(self):
        control, c = self.expand(arm='control')
        treatment, g = self.expand(arm='gap')
        self.assertEqual(control, treatment)
        self.assertNotEqual(c['query_sha256'], g['query_sha256'])
        self.assertEqual(c['query_sha256'], _sha(self.selection['control_text']))
        self.assertEqual(g['query_sha256'], _sha(self.selection['treatment_text']))
        self.assertEqual(c['embedding_slots'], g['embedding_slots'])
        self.assertEqual(c['embedding_slots'], 1)

    def test_no_seed_change_keeps_graph_checkpoint_but_fallback_retains_maxima(self):
        self.state['seeds'].append(asdict(SpreadingSeed('episode', 2, 1., self.root(2))))
        updated, diag = self.expand()
        self.assertFalse(diag['restart'])
        self.assertEqual(updated['spreading'], self.state['spreading'])
        self.assertEqual(updated['spreading_expansion_base'], 5)
        self.assertEqual(updated['metrics']['spreading_restarts'], 2)
        self.assertEqual(updated['fallback_episode_scores']['1'], .9)
        self.assertAlmostEqual(updated['fallback_episode_scores']['3'], .8)

    def test_shared_source_and_old_alias_roots_use_max_not_independent_support(self):
        self.state['seeds'] = [asdict(SpreadingSeed('episode', 1, .8, 'question:a')),
                               asdict(SpreadingSeed('episode', 1, .6, 'question:b'))]
        updated, diag = self.expand((.6, .8), seeds_per_cue=2)
        old = [s for s in updated['seeds'] if s['node_id'] == 1]
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0]['activation'], .8)
        self.assertEqual(old[0]['root_id'], self.root(1))
        new = [s for s in updated['seeds'] if s['node_id'] in (3, 4)]
        self.assertEqual({s['root_id'] for s in new}, {self.root(3)})
        self.assertEqual(diag['new_seed_episode_ids'], [3, 4])

    def test_equal_scores_use_episode_id_ties_independent_of_matrix_row_order(self):
        first, _ = self.expand((.6, .8), seeds_per_cue=1)
        swapped, _ = self.expand((.6, .8), seeds_per_cue=1,
            episode_ids=[4, 3, 2, 1], episode_matrix=self.matrix[::-1])
        self.assertEqual(first, swapped)
        self.assertIn(3, [s['node_id'] for s in first['seeds']])
        self.assertNotIn(4, [s['node_id'] for s in first['seeds']])

    def test_invalid_vector_matrix_ids_and_source_leave_every_input_unchanged(self):
        before = deepcopy(self.state)
        for vector in ([0, 0], [float('nan'), 1], [float('inf'), 1], [1, 0, 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.expand(vector)
        for kwargs in ({'episode_ids': [1, 1, 3, 4]}, {'seeds_per_cue': 0}, {'arm': 'treatment'},
                       {'episode_matrix': self.matrix * 2}, {'episode_matrix': self.matrix[:2]},
                       {'sources': {**self.sources, 1: 'changed actual Source'}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.expand(**kwargs)
        self.assertEqual(self.state, before)

    def test_stale_attention_and_bad_fallback_are_rejected_before_any_transition(self):
        state = deepcopy(self.state)
        self.state['fallback_episode_scores']['1'] = float('nan')
        with self.assertRaises(ValueError):
            self.expand()
        self.state = state
        self.commit(self.reviewer.review_facts(self.state, fact_ids=[self.state['facts'][0]['fact_id']]))
        before = deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.expand()
        self.assertEqual(self.state, before)


if __name__ == '__main__':
    unittest.main()
