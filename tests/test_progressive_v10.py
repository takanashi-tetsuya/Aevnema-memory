"""Offline SQLite integration for original-request fragment contracts.

These check construction and recovery, not a model's semantic interpretation.
V9 fixtures and frozen tests remain unchanged.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from config.prompt_config import recall_v10_prompts as v10
from memory_demo.retrieval.progressive_v9 import ProgressiveRecallV9
from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
from memory_demo.retrieval.recall_contract_anchors import (
    ANCHOR_PROTOCOL, anchor_binding, request_fragments,
)
import test_progressive_v3 as fixtures
from test_progressive_v8 import QUESTION
from test_progressive_v9 import ContractReviewer


class FragmentReviewer(ContractReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.selected_anchor = 'Q1'

    def chat_json(self, system, content):
        if system != v10.CONTRACT:
            return super().chat_json(system, content)
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        self.pipeline.append('contract')
        self.contract_payloads.append(deepcopy(payload))
        self.clock.value += self.contract_delay
        if self.contract_failure == 'interrupt':
            raise KeyboardInterrupt()
        rows = []
        for target in payload['targets']:
            items = [{'kind': 'answer', 'description': '说明原问求助信的回应',
                      'anchor_id': self.selected_anchor}]
            if self.contract_constraint:
                items.append({'kind': 'constraint', 'description': '保留原问限定',
                              'anchor_id': self.selected_anchor})
            rows.append({'need_index': target['need_index'], 'answer_type': 'description',
                         'items': items})
        return {'contracts': rows}


class ProgressiveV10Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.model = FragmentReviewer(self.clock)
        self.service = ProgressiveRecallV10(self.config, self.db, self.model, clock=self.clock)

    def restore(self):
        return ProgressiveRecallV10(self.config, self.db, self.model, clock=self.clock)

    def interrupt_after_write(self, predicate):
        write = self.service.sessions.write
        fired = []

        def crash(state):
            write(state)
            if not fired and predicate(state):
                fired.append(True)
                raise KeyboardInterrupt()

        self.service.sessions.write = crash
        return fired

    def test_contract_constructs_once_before_embedding_with_local_literal_anchor(self):
        result = self.service.query(QUESTION, learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.pipeline[:4], ['plan', 'contract', 'embed', 'map'])
        self.assertEqual(len(self.model.contract_payloads), 1)
        payload = self.model.contract_payloads[0]
        self.assertEqual(set(payload), {'original_request_fragments', 'targets'})
        self.assertEqual(payload['targets'], [{'need_index': 0, 'planned_need': '求助信得到什么回应'}])
        self.assertEqual(payload['original_request_fragments'], [
            {'anchor_id': 'Q1', 'field': 'question', 'start': 0, 'end': len(QUESTION), 'text': QUESTION}])
        state = self.service.sessions.read(result['session_id'])
        self.assertEqual(result['need_contracts'][0]['items'][0]['anchor'],
                         {'field': 'question', 'quote': QUESTION})
        self.assertEqual(result['contract_anchor_binding'], anchor_binding(state))
        self.assertEqual(result['review_protocol_version'], 10)
        self.assertEqual(result['contract_work']['status'], 'committed')
        self.assertEqual(result['need_contract_gaps'], [])
        self.assertEqual([t['stage'] for t in state['review_trace']], ['contract', 'map', 'facts', 'need'])
        self.assertTrue(all(t['version'] == 10 for t in state['review_trace']))

    def test_original_context_and_sentence_offsets_survive_canonical_expansion(self):
        question = '说明求助信。后来怎么样？'
        context = '  只问这封信；不要新增日期。  '
        self.model.selected_anchor = 'X1'
        result = self.service.query(question, context=context, learn=False)
        self.assertTrue(result['complete'], result)
        payload = self.model.contract_payloads[0]
        original = {'question': question, 'context': context}
        self.assertEqual([r['anchor_id'] for r in payload['original_request_fragments']],
                         ['Q1', 'Q2', 'X1', 'X2'])
        for r in payload['original_request_fragments']:
            self.assertEqual(original[r['field']][r['start']:r['end']], r['text'])
        item = result['need_contracts'][0]['items'][0]
        self.assertEqual(item['anchor'], {'field': 'context', 'quote': '只问这封信；'})
        self.assertIn(item['anchor']['quote'], context)

    def test_unknown_anchor_id_fails_before_embedding_and_is_not_retried(self):
        self.model.selected_anchor = 'Q999'
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'technical_error')
        self.assertFalse(first['complete'])
        self.assertFalse(first['resumable'])
        self.assertIsNone(first['need_contracts'])
        self.assertEqual(first['contract_work']['status'], 'failed')
        self.model.selected_anchor = 'Q1'
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertNotIn('embed', self.model.pipeline)

    def test_started_construction_interrupt_does_not_resend(self):
        fired = self.interrupt_after_write(lambda s: s.get('contract_work', {}).get('status') == 'started')
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertFalse(first['resumable'])
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(self.model.pipeline, ['plan'])

    def test_provider_interrupt_keeps_single_attempt_with_no_binding(self):
        self.model.contract_failure = 'interrupt'
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'cancelled')
        self.assertFalse(first['resumable'])
        self.assertIsNone(first['contract_anchor_binding'])
        self.model.contract_failure = None
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertNotIn('embed', self.model.pipeline)

    def test_committed_binding_and_contract_are_reused_after_interrupt(self):
        fired = self.interrupt_after_write(lambda s: s.get('contract_work', {}).get('status') == 'committed')
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertTrue(first['resumable'])
        self.assertNotIn('embed', self.model.pipeline)
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['contract_anchor_binding'], first['contract_anchor_binding'])
        self.assertEqual(result['need_contract_hash'], first['need_contract_hash'])
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertEqual(self.model.pipeline.count('embed'), 1)

    def test_missing_or_changed_anchor_binding_blocks_gate_annotation_and_resume(self):
        first = self.service.query(QUESTION, learn=False)
        original = self.service.sessions.read(first['session_id'])
        variants = [None, {**original['contract_anchor_binding'], 'request_fragments_sha256': 'changed'}]
        for binding in variants:
            with self.subTest(binding=binding):
                state = deepcopy(original)
                if binding is None:
                    state.pop('contract_anchor_binding')
                else:
                    state['contract_anchor_binding'] = binding
                self.assertFalse(self.service._resolved(state))
                annotated = self.service.annotate_completion(deepcopy(first), state)
                self.assertFalse(annotated['complete'])
                self.assertEqual(annotated['completion_blockers']['contract_anchor_input_invalid'], [True])
                self.service.sessions.write(state)
                before = deepcopy(self.model.pipeline)
                result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
                self.assertEqual(result['status'], 'technical_error')
                self.assertFalse(result['complete'])
                self.assertEqual(self.model.pipeline, before)

    def test_fragment_catalog_change_without_protocol_bump_is_rejected(self):
        first = self.service.query(QUESTION, learn=False)
        before = deepcopy(self.model.pipeline)

        def changed_fragments(state):
            records = request_fragments(state)
            records[0]['anchor_id'] = 'Q42'
            return records

        with patch('memory_demo.retrieval.recall_contract_anchors.request_fragments', changed_fragments):
            state = self.service.sessions.read(first['session_id'])
            self.assertFalse(self.service._resolved(state))
            result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertFalse(result['complete'])
        self.assertEqual(self.model.pipeline, before)

    def test_changed_v10_prompt_and_segmentation_protocol_reject_snapshot(self):
        first = self.service.query(QUESTION, learn=False)
        before = deepcopy(self.model.pipeline)
        with patch('memory_demo.retrieval.progressive_v10.CONTRACT', v10.CONTRACT + '\nchanged'):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        with patch.dict(ANCHOR_PROTOCOL, {'segmentation': 'different-segmentation'}):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, before)

    def test_v9_checkpoint_is_not_reinterpreted_as_id_anchored_construction(self):
        model = ContractReviewer(self.clock)
        old = ProgressiveRecallV9(self.config, self.db, model, clock=self.clock)
        first = old.query(QUESTION, learn=False)
        self.assertTrue(first['complete'], first)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, [])

    def test_global_deadline_includes_contract_and_prevents_late_binding_commit(self):
        self.model.plan_delay = 4
        self.model.contract_delay = 2
        result = self.service.query(QUESTION, timeout_seconds=5, learn=False)
        self.assertEqual(result['status'], 'time_budget', result)
        self.assertEqual(result['contract_work']['status'], 'failed')
        self.assertIsNone(result['contract_anchor_binding'])
        self.assertNotIn('embed', self.model.pipeline)
        call = next(c for c in result['stage_calls'] if c['stage'] == 'contract')
        self.assertEqual(call['deadline_seconds'], 1)

    def test_local_contract_deadline_remains_forty_seconds(self):
        self.model.contract_delay = 41
        result = self.service.query(QUESTION, timeout_seconds=100, learn=False)
        self.assertEqual(result['status'], 'technical_error', result)
        self.assertFalse(result['resumable'])
        self.assertIsNone(result['contract_anchor_binding'])
        self.assertNotIn('embed', self.model.pipeline)
        call = next(c for c in result['stage_calls'] if c['stage'] == 'contract')
        self.assertEqual(call['deadline_seconds'], 40)

    def test_v9_per_item_gap_semantics_are_preserved(self):
        self.model.contract_constraint = True
        self.model.constraint_unknown = True
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['need_assessments'][0]['status'], 'partial')
        self.assertEqual([g['kind'] for g in result['need_contract_gaps']], ['constraint'])
        self.assertEqual(len(result['evidence']), 2)
        self.assertNotIn('contract_anchor_input_invalid', result['completion_blockers'])


if __name__ == '__main__':
    unittest.main()
