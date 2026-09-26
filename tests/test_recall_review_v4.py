"""Synthetic offline checks for the combined v4 record/need protocol."""
from copy import deepcopy
import hashlib
import json
import unittest

from config.prompt_config import recall_v3_prompts, recall_v4_prompts
from memory_demo.retrieval.recall_records import project_source
from memory_demo.retrieval.recall_review_v3 import (
    FACTS_SYSTEM, NEED_SYSTEM as V3_NEED_SYSTEM, MAP_SYSTEM as V3_MAP_SYSTEM,
    RecordReview, ReviewStageError,
)
from memory_demo.retrieval.recall_review_v4 import RecordReviewV4, MAP_SYSTEM, NEED_SYSTEM


class RecordReviewV4Tests(unittest.TestCase):
    def setUp(self):
        lines = ['停电之后，乙提出这次申请。', '甲确认收到了申请。', '乙说明了申请的用途。',
                 '甲批准了这次申请。', '甲补充说批准仅限于当晚。']
        raw = '\n\n'.join(f'[record: {i}]\n[speaker_raw: 甲]\nzh-CN: {line}'
                          for i, line in enumerate(lines, 1))
        self.sources = {1: raw, 2: '[record: 1]\nzh-CN: 丙仍未回来。'}
        self.episodes = {
            1: {'id': 1, 'source_id': 1, 'text': '甲在停电后批准乙的申请，但仅限当晚。'},
            2: {'id': 2, 'source_id': 2, 'text': '丙仍未回来。'},
        }
        self.records = {sid: project_source(sid, text, [sid]) for sid, text in self.sources.items()}
        self.state = {
            'question': '在停电后的那次申请中，甲是否批准乙的申请，以及丙何时回来？',
            'needs': ['她是否批准这次申请？', '丙何时回来？'],
            'facts': [], 'pending_facts': [], 'pending_links': [], 'verified_links': [],
            'need_assessments': [], 'covered_needs': [], 'resolved_needs': [],
        }
        self.calls = []
        self.map_refs = ['R1', 'R2', 'R3', 'R4', 'R5']
        self.map_episode = 1
        self.alignment = 'supported'
        self.status = 'supported'
        self.need_refs = None
        self.reviewer = RecordReviewV4(self.sources, self.episodes, self.call)

    def call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if system in (MAP_SYSTEM, V3_MAP_SYSTEM):
            return {'facts': [{'record_ids': list(self.map_refs), 'episode_id': self.map_episode,
                              'interpretation': '甲批准了停电后的申请，仅限于当晚。',
                              'need_indices': [0]}], 'links': []}
        if system == FACTS_SYSTEM:
            return {'fact_decisions': [{'fact_id': item['fact_id'], 'decision': 'accept',
                'episode_alignment': self.alignment, 'statement': '甲的批准带有当晚限定。',
                'reason': '原文记录完整的申请、批准和当晚限定，与此经历对应。'} for item in payload['facts']]}
        if system in (NEED_SYSTEM, V3_NEED_SYSTEM):
            return {'status': self.status,
                    'fact_ids': ([item['fact_id'] for item in payload['facts']]
                                 if self.need_refs is None else list(self.need_refs)),
                    'answer': '甲批准了这次申请，仅限当晚。', 'reason': '记录包含申请、明确批准及限定。'}
        raise AssertionError('Unexpected system prompt')

    def commit(self, result):
        self.state.update(deepcopy(result.updates))
        return result

    def accept(self):
        self.commit(self.reviewer.map(self.records[1], self.state))
        return self.commit(self.reviewer.review_facts(self.state))

    def test_v3_default_still_allows_three_but_rejects_five_records(self):
        self.assertEqual(RecordReview.max_fact_records, 3)
        base = RecordReview(self.sources, self.episodes, self.call)
        self.map_refs = ['R1', 'R2', 'R3']
        three = base.map(self.records[1], self.state)
        self.assertEqual(len(three.updates['pending_facts'][0]['evidence']), 3)
        self.assertEqual(three.trace['version'], 3)
        self.map_refs += ['R4', 'R5']
        with self.assertRaises(ReviewStageError):
            base.map(self.records[1], self.state)
        self.commit(self.reviewer.map(self.records[1], self.state))
        # Revalidated persisted groups obey the same per-class limit as map.
        with self.assertRaises(ReviewStageError):
            base.review_facts(self.state)

    def test_five_necessary_records_keep_all_original_spans_and_are_accepted(self):
        before = deepcopy(self.state)
        mapped = self.reviewer.map(self.records[1], self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(mapped.trace['version'], 4)
        self.commit(mapped)
        self.assertEqual(len(self.state['pending_facts'][0]['record_ids']), 5)
        reviewed = self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(reviewed.trace['version'], 4)
        self.assertEqual(self.state['pending_facts'], [])
        fact, = self.state['facts']
        self.assertEqual(fact['episode_alignment'], 'supported')
        self.assertEqual(fact['verification_status'], 'accepted')
        self.assertEqual(len(fact['evidence']), 5)
        for evidence, record in zip(fact['evidence'], self.records[1]):
            self.assertEqual(evidence['quote'], record['text'])
            self.assertEqual(evidence['quote'], self.sources[1][evidence['start']:evidence['end']])
            self.assertEqual(evidence['source_sha256'], hashlib.sha256(self.sources[1].encode()).hexdigest())
            self.assertEqual(evidence['episode_id'], 1)
        self.assertIn('仅限于当晚', fact['claim'])
        self.assertEqual(len(self.calls[-1][1]['facts'][0]['record_ids']), 5)

    def test_unknown_duplicate_cross_source_and_wrong_episode_still_reject_atomically(self):
        cases = [(['R99'], 1), (['R1', 'R1'], 1), (['R1', 'R6'], 1),
                 (['R1', 'R2', 'R3', 'R4', 'R5'], 2), ([], 1)]
        for refs, eid in cases:
            with self.subTest(refs=refs, eid=eid):
                self.map_refs, self.map_episode = refs, eid
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError) as caught:
                    self.reviewer.map(self.records[1] + self.records[2], self.state)
                self.assertEqual(self.state, before)
                self.assertFalse(caught.exception.trace['schema_ok'])
                self.assertEqual(caught.exception.trace['version'], 4)
                self.assertEqual(caught.exception.trace['request'], self.calls[-1][1])

    def test_episode_alignment_is_still_required_for_five_record_fact(self):
        for alignment, expected in [('unknown', 'pending_facts'), ('mismatch', None)]:
            with self.subTest(alignment=alignment):
                self.alignment = alignment
                state = deepcopy(self.state)
                state.update(self.reviewer.map(self.records[1], state).updates)
                result = self.reviewer.review_facts(state)
                self.assertEqual(result.updates['facts'], [])
                self.assertEqual(len(result.updates['pending_facts']), 1 if expected else 0)

    def test_source_change_invalidates_five_record_evidence_before_another_call(self):
        self.accept()
        before, count = deepcopy(self.state), len(self.calls)
        self.sources[1] += '\n变更后的来源。'
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_need(self.state, 0)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(self.state, before)

    def test_need_targets_one_question_with_original_reference_context_and_full_evidence(self):
        self.accept()
        before = deepcopy(self.state)
        class UnscopedMultiRecordReview(RecordReview):
            max_fact_records = None
        unscoped = UnscopedMultiRecordReview(self.sources, self.episodes, self.call)
        original = unscoped.review_need(self.state, 0).trace['request']
        result = self.reviewer.review_need(self.state, 0)
        prompt, actual = self.calls[-1]
        self.assertEqual(prompt, NEED_SYSTEM)
        self.assertEqual(list(actual)[:2], ['question', 'request_context'])
        self.assertEqual(actual['question'], self.state['needs'][0])
        self.assertEqual(actual['request_context'], self.state['question'])
        self.assertIn('停电后', actual['request_context'])
        self.assertNotIn('needs', actual)
        self.assertNotIn('need', actual)
        self.assertEqual(actual['records'], original['records'])
        self.assertEqual(actual['facts'], original['facts'])
        self.assertEqual(len(actual['records']), 5)
        self.assertEqual(actual['facts'][0]['record_ids'], ['R1', 'R2', 'R3', 'R4', 'R5'])
        self.assertEqual(result.trace['request'], actual)
        self.assertEqual(result.trace['input_chars'], len(json.dumps(actual, ensure_ascii=False, separators=(',', ':'))))
        self.assertEqual(self.state, before)
        self.assertEqual(result.updates['covered_needs'], [0])
        self.assertEqual(result.updates['need_assessments'][1]['status'], 'unknown')

    def test_need_budget_applies_to_actual_scoped_payload_before_callback(self):
        self.accept()
        request = self.reviewer.review_need(self.state, 0).trace['request']
        exact_size = len(json.dumps(request, ensure_ascii=False, separators=(',', ':')))
        self.state['review_context_max_chars'] = exact_size
        self.reviewer.review_need(self.state, 0)
        self.state['review_context_max_chars'] = exact_size - 1
        count, before = len(self.calls), deepcopy(self.state)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.review_need(self.state, 0)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(self.state, before)
        self.assertEqual(caught.exception.trace['request'], request)
        self.assertEqual(caught.exception.trace['input_chars'], exact_size)
        self.assertEqual(len(caught.exception.trace['request']['records']), 5)

    def test_default_48k_budget_is_not_removed_with_record_count_limit(self):
        raw = '\n\n'.join(f'[record: {i}]\nzh-CN: ' + ('完整原文' * 2600) for i in range(1, 6))
        self.sources[1] = raw
        records = project_source(1, raw, [1])
        self.assertEqual(len(records), 5)
        with self.assertRaises(ReviewStageError) as caught:
            self.reviewer.map(records, self.state)
        self.assertEqual(self.calls, [])
        self.assertEqual(caught.exception.trace['input_char_budget'], 48000)
        self.assertGreater(caught.exception.trace['input_chars'], 48000)
        self.assertEqual(len(caught.exception.trace['request']['records']), 5)
        self.assertEqual(caught.exception.trace['request']['records'][-1]['text'], records[-1]['text'])

    def test_unknown_partial_and_refuted_remain_distinct_after_target_scoping(self):
        self.accept()
        for status, refs, covered, resolved in [
                ('unknown', [], [], []), ('partial', ['F1'], [], []),
                ('refuted', ['F1'], [], [0]), ('supported', ['F1'], [0], [0])]:
            with self.subTest(status=status):
                self.status, self.need_refs = status, refs
                result = self.reviewer.review_need(self.state, 0)
                self.assertEqual(result.updates['covered_needs'], covered)
                self.assertEqual(result.updates['resolved_needs'], resolved)
                self.assertEqual(result.updates['need_assessments'][0]['status'], status)
                self.assertEqual(result.trace['request'], self.calls[-1][1])

    def test_resolved_need_cannot_use_empty_or_unknown_fact_ids(self):
        self.accept()
        for status in ['supported', 'refuted']:
            for refs in [[], ['F99']]:
                with self.subTest(status=status, refs=refs):
                    self.status, self.need_refs = status, refs
                    before = deepcopy(self.state)
                    with self.assertRaises(ReviewStageError) as caught:
                        self.reviewer.review_need(self.state, 0)
                    self.assertEqual(self.state, before)
                    self.assertEqual(caught.exception.trace['request'], self.calls[-1][1])
                    self.assertEqual(caught.exception.trace['request']['question'], self.state['needs'][0])

    def test_only_the_map_record_limit_and_need_prompt_change(self):
        old = '每条事实选择同一 Source 的1至3条记录，'
        new = '每条事实选择同一 Source 的一条或多条必要完整记录，不重复引用记录，'
        self.assertEqual(recall_v3_prompts.MAP.replace(old, new, 1), recall_v4_prompts.MAP)
        for name in ['PLAN', 'FACTS', 'LINKS']:
            self.assertEqual(getattr(recall_v3_prompts, name), getattr(recall_v4_prompts, name))
        self.assertIn('request_context', NEED_SYSTEM)
        self.assertIn('不得把其中的其他问题当成额外任务', NEED_SYSTEM)
        self.assertIn('未问的额外姓名、类别或更细粒度细节', NEED_SYSTEM)
        self.assertIn('仍须满足 question 本身及由上下文明确限定的要点', NEED_SYSTEM)


if __name__ == '__main__':
    unittest.main()
