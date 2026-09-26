"""SQLite contract construction, durable gaps, and deadline/recovery checks.

The provider is deterministic and offline; these verify the contract gate and
state transitions, not whether a real model understands identity or causation.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from config.prompt_config import recall_v3_prompts as v3, recall_v4_prompts as v4
from config.prompt_config import recall_v9_prompts as v9
from memory_demo.retrieval.progressive_v8 import ProgressiveRecallV8
from memory_demo.retrieval.progressive_v9 import ProgressiveRecallV9
from memory_demo.retrieval.recall_need_contracts import contract_gaps
from test_progressive_v8 import LocalScheduleReviewer, QUESTION
import test_progressive_v3 as fixtures


class ContractReviewer(LocalScheduleReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.split_initial = False
        self.pipeline = []
        self.contract_payloads = []
        self.item_payloads = []
        self.contract_delay = 0
        self.plan_delay = 0
        self.contract_failure = None
        self.contract_constraint = False
        self.constraint_unknown = False
        self.contract_premise = False
        self.premise_conflict = False
        self.unbound_record = False
        self.fact_statement = None
        self.plan_need_count = 1

    def embed(self, cues):
        self.pipeline.append('embed')
        return super().embed(cues)

    def chat_json(self, system, content):
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        if system == v3.PLAN:
            self.pipeline.append('plan')
            self.clock.value += self.plan_delay
            needs = ['求助信得到什么回应'] if self.plan_need_count == 1 else [
                f'求助信事项 {i}' for i in range(self.plan_need_count)]
            return {'needs': needs, 'cues': ['求助信']}
        if system == v9.CONTRACT:
            self.pipeline.append('contract')
            self.contract_payloads.append(deepcopy(payload))
            self.clock.value += self.contract_delay
            if self.contract_failure == 'interrupt':
                raise KeyboardInterrupt()
            if self.contract_failure == 'schema':
                return {'contracts': []}
            if self.contract_failure == 'transport':
                raise TimeoutError('offline contract transport failure')
            rows = []
            for target in payload['targets']:
                anchor = {'field': 'question', 'quote': payload['request_context']}
                items = [{'kind': 'answer', 'description': '说明求助信的回应', 'anchor': anchor}]
                if self.contract_constraint:
                    items.append({'kind': 'constraint', 'description': '对应原问的求助信',
                                  'anchor': {'field': 'question', 'quote': '求助信'}})
                if self.contract_premise:
                    items.append({'kind': 'premise', 'description': '离线fixture整项前提',
                                  'anchor': anchor})
                rows.append({'need_index': target['need_index'], 'answer_type': 'description',
                             'items': items})
            return {'contracts': rows}
        if system == v9.NEEDS:
            self.pipeline.append('need')
            self.item_payloads.append(deepcopy(payload))
            self.clock.value += self.need_delay
            fact = next((f for f in payload['facts'] if f['episode_id'] == 2), payload['facts'][0])
            assessments = []
            for target in payload['targets']:
                items = []
                for item in target['items']:
                    status = 'supported'
                    if item['kind'] == 'constraint' and self.constraint_unknown:
                        status = 'unknown'
                    if item['kind'] == 'premise' and self.premise_conflict:
                        status = 'contradicted'
                    records = [fact['record_ids'][0]]
                    if self.unbound_record:
                        records = [next(r['id'] for r in payload['records']
                                        if r['id'] not in fact['record_ids'])]
                    items.append({'item_id': item['item_id'], 'status': status,
                                  'value': '' if status == 'unknown' else '离线fixture的直接回应',
                                  'fact_ids': [] if status == 'unknown' else [fact['fact_id']],
                                  'record_ids': [] if status == 'unknown' else records,
                                  'reason': '离线fixture逐项判断，不代表真实语义验证'})
                assessments.append({'need_index': target['need_index'], 'items': items})
            return {'assessments': assessments}
        self.pipeline.append('map' if system == v4.MAP else 'facts' if system == v3.FACTS else 'other')
        result = super().chat_json(system, content)
        if system == v3.FACTS and self.fact_statement is not None:
            for item in result['fact_decisions']:
                if item['decision'] == 'accept':
                    item['statement'] = self.fact_statement
        return result


class ProgressiveV9Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.model = ContractReviewer(self.clock)
        self.service = ProgressiveRecallV9(self.config, self.db, self.model, clock=self.clock)

    def restore(self):
        return ProgressiveRecallV9(self.config, self.db, self.model, clock=self.clock)

    def interrupt_after_write(self, predicate):
        write = self.service.sessions.write
        interrupted = []

        def crash(state):
            write(state)
            if not interrupted and predicate(state):
                interrupted.append(True)
                raise KeyboardInterrupt()

        self.service.sessions.write = crash
        return interrupted

    def test_contract_is_committed_once_before_embeddings_and_original_records(self):
        result = self.service.query(QUESTION, learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(self.model.pipeline[:4], ['plan', 'contract', 'embed', 'map'])
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertEqual(set(self.model.contract_payloads[0]), {'request_context', 'context', 'targets'})
        self.assertEqual(result['contract_work']['status'], 'committed')
        self.assertEqual(result['review_protocol_version'], 9)
        self.assertEqual(result['need_contract_gaps'], [])
        self.assertEqual(result['completion_blockers']['need_contract_gaps'], [])
        self.assertEqual(len(result['evidence']), 2)
        state = self.service.sessions.read(result['session_id'])
        self.assertTrue(self.service._resolved(state))
        self.assertEqual([t['stage'] for t in state['review_trace']], ['contract', 'map', 'facts', 'need'])

    def test_answer_without_required_constraint_keeps_a_durable_gap(self):
        self.model.contract_constraint = True
        self.model.constraint_unknown = True
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['need_assessments'][0]['status'], 'partial')
        self.assertEqual(len(result['evidence']), 2)
        self.assertEqual([g['kind'] for g in result['need_contract_gaps']], ['constraint'])
        state = self.service.sessions.read(result['session_id'])
        self.assertFalse(self.service._resolved(state))
        self.assertEqual(contract_gaps(state), result['need_contract_gaps'])

    def test_eight_needs_still_use_two_item_assessment_calls(self):
        self.model.plan_need_count = 8
        result = self.service.query(QUESTION, learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertEqual([[t['need_index'] for t in p['targets']] for p in self.model.item_payloads],
                         [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_interrupted_started_contract_is_not_sent_on_resume(self):
        fired = self.interrupt_after_write(lambda s: s.get('contract_work', {}).get('status') == 'started')
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertFalse(first['resumable'])
        self.assertEqual(len(self.model.contract_payloads), 0)
        before = deepcopy(self.model.pipeline)
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertFalse(result['complete'])
        self.assertFalse(result['resumable'])
        self.assertEqual(self.model.pipeline, before)
        self.assertNotIn('embed', self.model.pipeline)

    def test_provider_interrupt_is_recorded_failed_and_never_repeated(self):
        self.model.contract_failure = 'interrupt'
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'cancelled')
        self.assertEqual(first['contract_work']['status'], 'failed')
        self.assertFalse(first['resumable'])
        self.model.contract_failure = None
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertNotIn('embed', self.model.pipeline)

    def test_contract_schema_failure_cannot_regenerate_an_easier_contract(self):
        self.model.contract_failure = 'schema'
        first = self.service.query(QUESTION, learn=False)
        self.assertEqual(first['status'], 'technical_error')
        self.assertEqual(first['contract_work']['status'], 'failed')
        self.assertTrue(first['need_contract_gaps'])
        self.model.contract_failure = None
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertEqual(result['status'], 'technical_error')
        self.assertFalse(result['resumable'])

    def test_committed_contract_resumes_after_interrupt_without_another_call(self):
        fired = self.interrupt_after_write(lambda s: s.get('contract_work', {}).get('status') == 'committed')
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        self.assertTrue(first['resumable'])
        self.assertNotIn('embed', self.model.pipeline)
        digest = first['need_contract_hash']
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['need_contract_hash'], digest)
        self.assertEqual(len(self.model.contract_payloads), 1)
        self.assertEqual(self.model.pipeline.count('embed'), 1)

    def test_contract_uses_remaining_global_deadline_and_cannot_commit_late(self):
        self.model.plan_delay = 4
        self.model.contract_delay = 2
        first = self.service.query(QUESTION, timeout_seconds=5, learn=False)
        self.assertEqual(first['status'], 'time_budget', first)
        self.assertEqual(first['contract_work']['status'], 'failed')
        self.assertFalse(first['resumable'])
        self.assertFalse(first['need_contracts'])
        self.assertNotIn('embed', self.model.pipeline)
        contract_call = next(c for c in first['stage_calls'] if c['stage'] == 'contract')
        self.assertAlmostEqual(contract_call['deadline_seconds'], 1)
        before = deepcopy(self.model.pipeline)
        self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, before)

    def test_contract_local_forty_second_limit_does_not_gain_a_new_budget(self):
        self.model.contract_delay = 41
        result = self.service.query(QUESTION, timeout_seconds=100, learn=False)
        self.assertEqual(result['status'], 'technical_error', result)
        self.assertFalse(result['resumable'])
        self.assertFalse(result['need_contracts'])
        self.assertNotIn('embed', self.model.pipeline)
        call = next(c for c in result['stage_calls'] if c['stage'] == 'contract')
        self.assertEqual(call['deadline_seconds'], 40)
        self.assertEqual(result['elapsed_seconds'], 41)

    def test_committed_need_is_not_resent_between_commit_and_cursor(self):
        fired = self.interrupt_after_write(lambda s:
            'need:0' in (s.get('review_work') or {}).get('committed_stages', {})
            and s['review_work']['need_cursor'] == 0)
        first = self.service.query(QUESTION, learn=False)
        self.assertTrue(fired)
        self.assertEqual(first['status'], 'cancelled')
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(len(self.model.item_payloads), 1)
        self.assertEqual(len(self.model.contract_payloads), 1)

    def test_changed_contract_is_rejected_before_new_embedding_or_review(self):
        first = self.service.query(QUESTION, learn=False)
        state = self.service.sessions.read(first['session_id'])
        state['need_contracts'][0]['items'][0]['description'] = '篡改后的较弱要求'
        self.service.sessions.write(state)
        before = deepcopy(self.model.pipeline)
        result = self.restore().query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(result['status'], 'technical_error')
        self.assertFalse(result['complete'])
        self.assertEqual(self.model.pipeline, before)
        self.assertTrue(all(g['status'] == 'invalid_contract' for g in result['need_contract_gaps']))

    def test_changed_prompt_or_contract_deadline_rejects_snapshot_before_provider_work(self):
        first = self.service.query(QUESTION, learn=False)
        before = deepcopy(self.model.pipeline)
        with patch('memory_demo.retrieval.progressive_v9.CONTRACT', v9.CONTRACT + '\nchanged'):
            with self.assertRaisesRegex(ValueError, 'changed knowledge'):
                self.restore().query(QUESTION, resume=first['session_id'], learn=False)

        class ChangedDeadline(ProgressiveRecallV9):
            stage_seconds = {**ProgressiveRecallV9.stage_seconds, 'contract': 39.0}

        with self.assertRaisesRegex(ValueError, 'changed knowledge'):
            ChangedDeadline(self.config, self.db, self.model, clock=self.clock).query(
                QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, before)

    def test_v8_checkpoint_is_not_reinterpreted_as_a_contract_run(self):
        model = LocalScheduleReviewer(self.clock)
        model.split_initial = False
        old = ProgressiveRecallV8(self.config, self.db, model, clock=self.clock)
        first = old.query(QUESTION, learn=False)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query(QUESTION, resume=first['session_id'], learn=False)
        self.assertEqual(self.model.pipeline, [])

    def test_construction_receipt_is_required_by_query_gate_and_annotation(self):
        first = self.service.query(QUESTION, learn=False)
        original = self.service.sessions.read(first['session_id'])
        for work in (None, {'status': 'started'}, {'status': 'failed'}):
            with self.subTest(work=work):
                state = deepcopy(original)
                if work is None:
                    state.pop('contract_work')
                else:
                    state['contract_work'] = work
                self.assertFalse(self.service._resolved(state))
                annotated = self.service.annotate_completion(deepcopy(first), state)
                self.assertFalse(annotated['complete'])
                self.assertTrue(annotated['need_contract_gaps'])

    def test_withdrawn_cited_fact_reopens_gaps_even_with_stale_resolved_indices(self):
        first = self.service.query(QUESTION, learn=False)
        state = self.service.sessions.read(first['session_id'])
        cited = set(state['need_assessments'][0]['fact_ids'])
        state['facts'] = [f for f in state['facts'] if f['fact_id'] not in cited]
        self.assertEqual(state['resolved_needs'], [0])
        self.assertFalse(self.service._resolved(state))
        annotated = self.service.annotate_completion(deepcopy(first), state)
        self.assertFalse(annotated['complete'])
        self.assertTrue(annotated['missing_requirements'])
        self.assertTrue(annotated['need_contract_gaps'])

    def test_fact_recheck_with_same_ids_invalidates_all_previous_contract_items(self):
        capture = []
        review = self.service._review

        def capture_review(windows, state, sources, episodes):
            capture.append((deepcopy(windows), sources, episodes))
            return review(windows, state, sources, episodes)

        self.service._review = capture_review
        first = self.service.query(QUESTION, learn=False)
        state = self.service.sessions.read(first['session_id'])
        ids = [f['fact_id'] for f in state['facts']]
        contract_hash = state['need_contract_hash']
        self.model.fact_statement = '更窄的离线fixture事实陈述'
        self.interrupt_after_write(lambda s:
            any(t.get('stage') == 'facts' and t.get('wave') == 2 and t.get('committed')
                for t in s.get('review_trace', [])))
        with self.assertRaises(KeyboardInterrupt):
            review(capture[0][0], state, capture[0][1], capture[0][2])
        self.assertEqual([f['fact_id'] for f in state['facts']], ids)
        self.assertTrue(all(f['statement'] == self.model.fact_statement for f in state['facts']))
        self.assertEqual(state['need_contract_hash'], contract_hash)
        self.assertEqual(state['resolved_needs'], [])
        self.assertFalse(self.service._resolved(state))
        self.assertTrue(contract_gaps(state))
        self.assertTrue(all(g['status'] == 'unknown' for g in contract_gaps(state)))

    def test_refuted_premise_cannot_also_normally_support_answer_in_live_query(self):
        self.model.contract_premise = True
        self.model.premise_conflict = True
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertTrue(result['need_contract_gaps'])
        self.assertEqual(result['completion_blockers']['failed_need_indices'], [0])
        # V8 retries failed multi-need batches as singletons; a singleton
        # failure has already exhausted that unchanged recovery policy.
        self.assertEqual(len(self.model.item_payloads), 1)

    def test_true_visible_record_cannot_support_a_different_cited_fact(self):
        self.model.unbound_record = True
        result = self.service.query(QUESTION, learn=False, max_waves=1)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['evidence']), 2)
        self.assertEqual(result['completion_blockers']['failed_need_indices'], [0])
        self.assertTrue(result['need_contract_gaps'])
        self.assertTrue(any('directly bound' in e['message'] for e in result['stage_errors']))


if __name__ == '__main__':
    unittest.main()
