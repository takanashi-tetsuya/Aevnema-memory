"""Pure local tests for separately committed record-review stages."""
from copy import deepcopy
import unittest

from memory_demo.associations.feedback import SourceEvidence, VerifiedRecallLink
from memory_demo.retrieval.recall_records import project_source
from memory_demo.retrieval.recall_review_v3 import (
    FACTS_SYSTEM, LINKS_SYSTEM, MAP_SYSTEM, NEED_SYSTEM,
    RecordReview, ReviewStageError,
)


class RecordReviewV3Tests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            1: '[speaker_alias_legend]\na: zh-CN=甲 | en=A\nb: zh-CN=乙 | en=B\n\n[record: 1]\n[speaker_raw: a]\nzh-CN: 我猜门已经打开了。\n\n[record: 2]\n[speaker_raw: b]\nzh-CN: 现在还不能确认。',
            2: '[record: 1]\n[speaker_raw: c]\nzh-CN: 门实际仍然关闭。',
            3: '[record: 1]\nzh-CN: 天气不错。',
            4: '[record: 1]\nzh-CN: 申请已批准。',
        }
        self.episodes = {i: {'id': i, 'source_id': i, 'text': f'Source {i} summary'} for i in self.sources}
        self.episodes[11] = {'id': 11, 'source_id': 1, 'text': 'The speaker qualifies the earlier guess.'}
        self.records = {sid: project_source(sid, raw, [1, 11] if sid == 1 else [sid]) for sid, raw in self.sources.items()}
        self.state = {'question': '门是否打开，以及申请是否批准？', 'needs': ['门是否打开', '申请是否批准'],
                      'facts': [], 'pending_facts': [], 'pending_links': [], 'verified_links': [],
                      'need_assessments': [], 'covered_needs': [], 'resolved_needs': []}
        self.calls = []
        self.responses = {
            MAP_SYSTEM: lambda p: {'facts': [{'record_ids': ['R1'], 'episode_id': 1, 'interpretation': '甲推测门已开。', 'need_indices': [0]}], 'links': [], 'followup_cues': []},
            FACTS_SYSTEM: lambda p: {'fact_decisions': [{'fact_id': f['fact_id'], 'decision': 'accept', 'episode_alignment': 'supported', 'statement': '只确认这是角色的说法。', 'reason': '保留说话人及限定。'} for f in p['facts']]},
            NEED_SYSTEM: lambda p: {'status': 'partial', 'fact_ids': [p['facts'][0]['fact_id']], 'answer': '已有角色陈述，但不足以判断。', 'reason': '缺少完整依据。'},
            LINKS_SYSTEM: lambda p: {'link_decisions': [{'link_id': link['link_id'], 'decision': 'accept', 'reason': '两端原文支持这项有限联系。'} for link in p['links']]},
        }
        self.reviewer = RecordReview(self.sources, self.episodes, self.call)

    def call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        result = self.responses[system]
        return deepcopy(result(payload) if callable(result) else result)

    def commit(self, result):
        self.state.update(deepcopy(result.updates))
        return result

    def mapped(self, records=None):
        return self.commit(self.reviewer.map(records or self.records[1], self.state))

    def accepted(self):
        self.mapped()
        return self.commit(self.reviewer.review_facts(self.state))

    def test_mapper_uses_short_ids_episode_hints_and_only_commits_pending(self):
        before = deepcopy(self.state)
        result = self.reviewer.map(self.records[1], self.state)
        self.assertEqual(before, self.state)
        self.assertNotIn('facts', result.updates)
        candidate = result.updates['pending_facts'][0]
        self.assertEqual(candidate['verification_status'], 'pending')
        self.assertEqual(candidate['quote'], '我猜门已经打开了。')
        self.assertIn('甲（a）：', candidate['claim'])
        self.assertNotEqual(candidate['claim'], candidate['interpretation'])
        payload = self.calls[0][1]
        self.assertEqual([r['id'] for r in payload['records']], ['R1', 'R2'])
        self.assertEqual({h['episode_id'] for h in payload['episode_hints']}, {1, 11})
        self.assertEqual(result.trace['record_ids']['R1'], self.records[1][0]['record_id'])

    def test_multiple_records_keep_separate_standard_spans_and_first_quote(self):
        self.responses[MAP_SYSTEM] = {'facts': [{'record_ids': ['R2', 'R1'], 'episode_id': 11, 'interpretation': '猜测及随后的保留意见。', 'need_indices': [0]}], 'links': []}
        self.mapped()
        fact = self.state['pending_facts'][0]
        self.assertEqual(fact['quote'], '我猜门已经打开了。')
        self.assertEqual(len(fact['evidence']), 2)
        self.assertIn('乙（b）：现在还不能确认。', fact['claim'])
        for evidence in fact['evidence']:
            SourceEvidence(**evidence)
            self.assertEqual(evidence['quote'], self.sources[evidence['source_id']][evidence['start']:evidence['end']])
        self.assertNotEqual(fact['quote'], ''.join(e['quote'] for e in fact['evidence']))

    def test_duplicate_record_groups_merge_interpretations_without_duplicate_claims(self):
        self.responses[MAP_SYSTEM] = {'facts': [
            {'record_ids': ['R1'], 'episode_id': 1, 'interpretation': '一个猜测。', 'need_indices': [0]},
            {'record_ids': ['R1'], 'episode_id': 1, 'interpretation': '保留说话人的意见。', 'need_indices': [1]}], 'links': []}
        self.mapped()
        self.assertEqual(len(self.state['pending_facts']), 1)
        fid = self.state['pending_facts'][0]['fact_id']
        self.assertEqual(self.state['pending_facts'][0]['need_indices'], [0, 1])
        self.assertEqual(len(self.state['interpretation_proposals'][fid]), 2)
        self.commit(self.reviewer.review_facts(self.state))
        accepted_before = deepcopy(self.state['facts'])
        self.mapped()
        self.assertEqual(self.state['facts'], accepted_before)
        self.assertEqual(self.state['pending_facts'], [])
        self.assertEqual(len(self.state['facts']), 1)

    def test_mapper_rejects_unknown_cross_source_wrong_episode_or_handwritten_quote(self):
        base = {'record_ids': ['R1'], 'episode_id': 1, 'interpretation': '解释。', 'need_indices': [0]}
        cases = [{}, {'facts': [{**base, 'record_ids': ['R99']}], 'links': []},
                 {'facts': [{**base, 'record_ids': ['R1', 'R3']}], 'links': []},
                 {'facts': [{**base, 'episode_id': 2}], 'links': []},
                 {'facts': [{**base, 'quote': '模型自己写的引文'}], 'links': []},
                 {'facts': [{**base, 'need_indices': [True]}], 'links': []}]
        for response in cases:
            with self.subTest(response=response):
                self.responses[MAP_SYSTEM] = response
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError) as caught:
                    self.reviewer.map(self.records[1]+self.records[2], self.state)
                self.assertEqual(self.state, before)
                self.assertEqual(caught.exception.trace['response'], response)

    def test_fact_batches_are_at_most_four_and_commit_independently(self):
        self.responses[MAP_SYSTEM] = lambda p: {'facts': [{'record_ids': [r['id']], 'episode_id': r['episode_ids'][0], 'interpretation': '原文陈述。', 'need_indices': [0]} for r in p['records']], 'links': []}
        self.mapped(sum(self.records.values(), []))
        self.assertEqual(len(self.state['pending_facts']), 5)
        first = self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(len(first.trace['fact_ids']), 4)
        self.assertEqual(len(self.state['facts']), 4)
        self.assertEqual(len(self.state['pending_facts']), 1)
        self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(len(self.state['facts']), 5)
        self.assertEqual(self.state['pending_facts'], [])
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_facts(self.state, fact_ids=[f['fact_id'] for f in self.state['facts']])

    def test_fact_review_keeps_primary_raw_claim_and_can_narrow_statement(self):
        self.mapped()
        before = deepcopy(self.state['pending_facts'][0])
        self.responses[FACTS_SYSTEM] = {'fact_decisions': [{'fact_id': 'F1', 'decision': 'accept', 'episode_alignment': 'supported', 'statement': '甲只是猜测，并未确认。', 'reason': '邻近记录明确保留意见。'}]}
        result = self.commit(self.reviewer.review_facts(self.state))
        fact = self.state['facts'][0]
        self.assertEqual(fact['fact_id'], before['fact_id'])
        self.assertEqual(fact['claim'], before['claim'])
        self.assertEqual(fact['quote'], before['quote'])
        self.assertEqual(fact['statement'], '甲只是猜测，并未确认。')
        self.assertEqual(result.trace['fact_ids'], {'F1': fact['fact_id']})
        self.assertIn('现在还不能确认。', [r['text'] for r in self.calls[-1][1]['records']])

    def test_needs_context_and_reject_are_not_delivered_or_learned(self):
        self.mapped()
        self.responses[FACTS_SYSTEM] = {'fact_decisions': [{'fact_id': 'F1', 'decision': 'needs_context', 'episode_alignment': 'supported', 'statement': '', 'reason': '须补上下文。'}]}
        self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(self.state['facts'], [])
        self.assertEqual(self.state['pending_facts'][0]['verification_status'], 'needs_context')
        self.responses[FACTS_SYSTEM]['fact_decisions'][0].update(decision='reject', reason='证据无关。')
        self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(self.state['facts'], [])
        self.assertEqual(self.state['pending_facts'], [])
        self.assertEqual(self.state['verified_links'], [])

    def test_bad_fact_decisions_leave_previously_committed_pending_intact(self):
        self.mapped()
        for response in [{}, {'fact_decisions': []}, {'fact_decisions': [{'fact_id': 'F99', 'decision': 'accept', 'episode_alignment': 'supported', 'statement': '窄义陈述。', 'reason': '理由。'}]},
                         {'fact_decisions': [{'fact_id': 'F1', 'decision': 'accept', 'episode_alignment': 'supported', 'statement': '窄义陈述。', 'reason': ''}]}]:
            self.responses[FACTS_SYSTEM] = response
            before = deepcopy(self.state)
            with self.assertRaises(ReviewStageError):
                self.reviewer.review_facts(self.state)
            self.assertEqual(before, self.state)
            self.assertEqual(len(self.state['pending_facts']), 1)

    def test_need_can_reassociate_an_accepted_fact_without_mapper_need_labels(self):
        self.accepted()
        fact = self.state['facts'][0]
        self.assertEqual(fact['need_indices'], [0])
        self.responses[NEED_SYSTEM] = {'status': 'supported', 'fact_ids': ['F1'], 'answer': '这个独立判决重新选择引用。', 'reason': '按本需求独立确认。'}
        result = self.commit(self.reviewer.review_need(self.state, 1))
        self.assertEqual(self.state['resolved_needs'], [1])
        self.assertEqual(self.state['need_assessments'][1]['fact_ids'], [fact['fact_id']])
        self.assertNotIn('facts', result.updates)
        self.assertEqual(fact['need_indices'], [0])

    def test_need_schema_or_transport_failure_cannot_erase_verified_facts(self):
        self.accepted()
        for response in [{}, {'status': 'supported', 'fact_ids': [], 'answer': '没有依据的回答。', 'reason': '无引用。'},
                         {'status': 'refuted', 'fact_ids': ['F99'], 'answer': '反证。', 'reason': '假编号。'}]:
            self.responses[NEED_SYSTEM] = response
            before = deepcopy(self.state)
            with self.assertRaises(ReviewStageError):
                self.reviewer.review_need(self.state, 0)
            self.assertEqual(before, self.state)
        def timeout(_):
            raise TimeoutError('local fake timeout')
        self.responses[NEED_SYSTEM] = timeout
        before = deepcopy(self.state)
        with self.assertRaises(TimeoutError):
            self.reviewer.review_need(self.state, 0)
        self.assertEqual(before, self.state)
        self.assertEqual(len(self.state['facts']), 1)

    def test_unknown_partial_and_empty_evidence_cannot_resolve(self):
        calls = len(self.calls)
        self.commit(self.reviewer.review_need(self.state, 0))
        self.assertEqual(len(self.calls), calls)
        self.accepted()
        for status in ('unknown', 'partial'):
            self.responses[NEED_SYSTEM] = {'status': status, 'fact_ids': ['F1'], 'answer': '仍需材料。', 'reason': '尚不充分。'}
            self.commit(self.reviewer.review_need(self.state, 0))
            self.assertEqual(self.state['resolved_needs'], [])
            self.assertEqual(self.state['covered_needs'], [])
        self.responses[NEED_SYSTEM] = {'status': 'refuted', 'fact_ids': ['F1'], 'answer': '原文明确反对这一前提。', 'reason': '引用该反证。'}
        self.commit(self.reviewer.review_need(self.state, 0))
        self.assertEqual(self.state['resolved_needs'], [0])
        self.assertEqual(self.state['covered_needs'], [])

    def test_completed_fact_rereview_invalidates_old_need_and_link_dependencies(self):
        self.responses[MAP_SYSTEM] = lambda p: {'facts': [{'record_ids': [r['id']], 'episode_id': r['episode_ids'][0], 'interpretation': '角色陈述。', 'need_indices': [0]} for r in p['records'] if r['source_id'] in (1, 2)],
                                               'links': [{'from_episode_id': 1, 'to_episode_id': 2, 'rationale': '两段围绕同一门的状态。'}]}
        self.mapped(self.records[1]+self.records[2])
        self.commit(self.reviewer.review_facts(self.state))
        self.responses[NEED_SYSTEM] = {'status': 'supported', 'fact_ids': ['F1'], 'answer': '已有判断。', 'reason': '依据角色陈述。'}
        self.commit(self.reviewer.review_need(self.state, 0))
        self.commit(self.reviewer.review_links(self.state))
        self.assertTrue(self.state['verified_links'])
        old_id = self.state['facts'][0]['fact_id']
        self.responses[FACTS_SYSTEM] = {'fact_decisions': [{'fact_id': 'F1', 'decision': 'reject', 'episode_alignment': 'supported', 'statement': '', 'reason': '新记录指出该旧解释错误。'}]}
        self.commit(self.reviewer.review_facts(self.state, fact_ids=[old_id]))
        self.assertNotIn(old_id, [f['fact_id'] for f in self.state['facts']])
        self.assertEqual(self.state['resolved_needs'], [])
        self.assertEqual(self.state['covered_needs'], [])
        self.assertEqual(self.state['verified_links'], [])
        self.assertTrue(self.state['pending_links'])

    def test_context_keeps_same_source_neighbors_and_latest_counterevidence_not_old_unrelated(self):
        self.accepted()
        self.responses[MAP_SYSTEM] = {'facts': [], 'links': []}
        self.mapped(self.records[3])
        self.mapped(self.records[2])
        fid = self.state['facts'][0]['fact_id']
        result = self.reviewer.review_facts(self.state, fact_ids=[fid])
        records = self.calls[-1][1]['records']
        self.assertEqual({r['source_id'] for r in records}, {1, 2})
        self.assertIn('现在还不能确认。', [r['text'] for r in records])
        self.assertIn('门实际仍然关闭。', [r['text'] for r in records])
        self.assertTrue(any(r['source_id'] == 3 for r in self.state['record_registry'].values()))
        self.assertGreater(result.trace['input_chars'], 0)

    def test_context_budget_failure_leaves_full_pending_and_never_truncates_or_calls(self):
        self.mapped()
        self.state['review_context_max_chars'] = 20
        before, calls = deepcopy(self.state), len(self.calls)
        with self.assertRaisesRegex(ReviewStageError, 'context budget exceeded') as caught:
            self.reviewer.review_facts(self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(len(self.calls), calls)
        self.assertGreater(caught.exception.trace['input_chars'], 20)
        self.assertEqual(len(caught.exception.trace['request']['records']), 2)

    def test_links_require_two_accepted_endpoints_and_a_separate_successful_review(self):
        self.responses[MAP_SYSTEM] = {'facts': [{'record_ids': ['R1'], 'episode_id': 1, 'interpretation': '甲的猜测。', 'need_indices': [0]},
                                               {'record_ids': ['R3'], 'episode_id': 2, 'interpretation': '另一条陈述。', 'need_indices': [0]}],
                                      'links': [{'from_episode_id': 1, 'to_episode_id': 2, 'rationale': '同一问题的两段材料。'}]}
        self.mapped(self.records[1]+self.records[2])
        calls = len(self.calls)
        self.commit(self.reviewer.review_links(self.state))
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(self.state['verified_links'], [])
        self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(self.state['verified_links'], [])
        result = self.commit(self.reviewer.review_links(self.state))
        self.assertEqual(len(self.state['verified_links']), 1)
        link = self.state['verified_links'][0]
        evidence = tuple(SourceEvidence(**item) for item in link['evidence'])
        VerifiedRecallLink(link['from_episode_id'], link['to_episode_id'], evidence, link['verifier'], link['rationale'], verified=True)
        self.assertEqual(result.trace['link_ids'], {'L1': link['link_id']})
        self.assertTrue(all(set(item) == {'source_id', 'episode_id', 'source_sha256', 'start', 'end', 'quote'} for item in link['evidence']))

    def test_stale_source_and_forged_speaker_aliases_reject_before_provider(self):
        forged = deepcopy(self.records[1])
        forged[0]['speaker_aliases']['zh-CN'] = '伪造名字'
        with self.assertRaises(ReviewStageError):
            self.reviewer.map(forged, self.state)
        self.assertEqual(self.calls, [])
        self.mapped()
        self.sources[1] += '\n原文已改变。'
        before, calls = deepcopy(self.state), len(self.calls)
        with self.assertRaises(ReviewStageError):
            self.reviewer.review_facts(self.state)
        self.assertEqual(self.state, before)
        self.assertEqual(len(self.calls), calls)

    def test_literal_followup_cue_returns_legacy_four_field_shape(self):
        self.responses[MAP_SYSTEM] = {'facts': [], 'links': [], 'followup_cues': [{'record_id': 'R1', 'cue': '已经打开'}]}
        result = self.reviewer.map(self.records[1], self.state)
        cue = result.followup_cues[0]
        self.assertEqual(set(cue), {'cue', 'source_id', 'episode_id', 'quote'})
        self.assertIn(cue['cue'], cue['quote'])
        self.responses[MAP_SYSTEM]['followup_cues'][0]['cue'] = '原文不存在的新名字'
        with self.assertRaises(ReviewStageError):
            self.reviewer.map(self.records[1], self.state)

    def test_promoted_unverified_candidate_is_not_accepted_by_need_stage(self):
        self.mapped()
        self.state['facts'] = deepcopy(self.state['pending_facts'])
        calls = len(self.calls)
        with self.assertRaisesRegex(ReviewStageError, 'unverified candidate'):
            self.reviewer.review_need(self.state, 0)
        self.assertEqual(len(self.calls), calls)

    def test_provider_payload_mutation_cannot_mutate_committed_state(self):
        self.mapped()
        before = deepcopy(self.state)
        def mutate(payload):
            payload['records'][0]['aliases']['zh-CN'] = '修改'
            payload['facts'][0]['record_ids'].clear()
            return {'fact_decisions': [{'fact_id': 'F1', 'decision': 'accept', 'episode_alignment': 'supported', 'statement': '限定解释。', 'reason': '引用原文。'}]}
        self.responses[FACTS_SYSTEM] = mutate
        self.reviewer.review_facts(self.state)
        self.assertEqual(self.state, before)


    def test_same_source_mismatched_episode_verdict_cannot_be_accepted(self):
        self.responses[MAP_SYSTEM] = {'facts': [{'record_ids': ['R1'], 'episode_id': 11,
            'interpretation': '甲猜测门已经打开。', 'need_indices': [0]}], 'links': []}
        self.mapped()
        self.assertEqual(self.state['pending_facts'][0]['episode_id'], 11)
        self.responses[FACTS_SYSTEM] = {'fact_decisions': [{'fact_id': 'F1', 'decision': 'accept',
            'episode_alignment': 'mismatch', 'statement': '只确认甲的猜测。',
            'reason': 'Episode 描述随后的保留意见；所引记录只有先前猜测，对应错误。'}]}
        result = self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(self.state['facts'], [])
        self.assertEqual(self.state['pending_facts'], [])
        verdict = result.trace['transitions'][0]
        self.assertEqual(verdict['model_decision'], 'accept')
        self.assertEqual(verdict['episode_alignment'], 'mismatch')
        self.assertEqual(verdict['to'], 'reject')
        self.assertIn('local_adjustment', verdict)

    def test_unknown_episode_alignment_suspends_acceptance_and_never_resolves(self):
        self.mapped()
        self.responses[FACTS_SYSTEM] = {'fact_decisions': [{'fact_id': 'F1', 'decision': 'accept',
            'episode_alignment': 'unknown', 'statement': '真实记录，但 Episode 对应不确定。',
            'reason': '缺少能够确认该 Episode 核心内容的记录。'}]}
        result = self.commit(self.reviewer.review_facts(self.state))
        self.assertEqual(self.state['facts'], [])
        self.assertEqual(self.state['pending_facts'][0]['episode_alignment'], 'unknown')
        self.assertEqual(self.state['pending_facts'][0]['verification_status'], 'needs_context')
        self.assertEqual(result.trace['transitions'][0]['model_decision'], 'accept')
        self.assertEqual(result.trace['transitions'][0]['to'], 'needs_context')
        self.assertEqual(self.state['resolved_needs'], [])
        self.assertEqual(self.state['verified_links'], [])

    def test_alignment_field_is_required_and_invalid_verdict_is_atomic(self):
        self.mapped()
        item = {'fact_id': 'F1', 'decision': 'accept', 'statement': '引用原文。', 'reason': '有依据。'}
        for response in [{'fact_decisions': [item]}, {'fact_decisions': [{**item, 'episode_alignment': 'same_source'}]}]:
            with self.subTest(response=response):
                self.responses[FACTS_SYSTEM] = response
                before = deepcopy(self.state)
                with self.assertRaises(ReviewStageError) as caught:
                    self.reviewer.review_facts(self.state)
                self.assertEqual(self.state, before)
                self.assertEqual(caught.exception.trace['response'], response)

    def test_supported_alignment_remains_an_auditable_model_verdict(self):
        self.accepted()
        fact = self.state['facts'][0]
        self.assertEqual(fact['episode_alignment'], 'supported')
        self.assertEqual(fact['verification_status'], 'accepted')
        # No deterministic meaning test is claimed: Source/span checks and the
        # independent model's explicit alignment verdict are separate evidence.
        before, calls = deepcopy(self.state), len(self.calls)
        self.state['facts'][0].pop('episode_alignment')
        with self.assertRaisesRegex(ReviewStageError, 'Episode alignment'):
            self.reviewer.review_need(self.state, 0)
        self.assertEqual(len(self.calls), calls)
        self.state = before


if __name__ == '__main__':
    unittest.main()
