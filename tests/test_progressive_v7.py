"""Episode navigation, complete records and durable MAP recovery on SQLite."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest

from config.prompt_config import recall_v3_prompts as v3, recall_v4_prompts as v4
from memory_demo.retrieval.progressive_v6 import ProgressiveRecallV6
from memory_demo.retrieval.progressive_v7 import ProgressiveRecallV7
from memory_demo.retrieval.recall_policy import RecallPolicy
from memory_demo.retrieval.recall_records import project_window
from memory_demo.retrieval.recall_review import stable_fact_id
from test_progressive_v6 import BatchReviewer
import test_progressive_v3 as fixtures


def source(*texts):
    return '\n\n'.join(f'[record: {i + 1}]\nzh-CN: {text}' for i, text in enumerate(texts))


class WindowReviewer(BatchReviewer):
    def __init__(self, clock):
        super().__init__(clock)
        self.map_payloads = []
        self.fail_text = None

    def chat_json(self, system, content):
        payload = (getattr(content, "structured_payload", None) or json.loads(content))
        if system == v4.MAP:
            self.map_payloads.append(deepcopy(payload))
            if self.fail_text and any(self.fail_text in r['text'] for r in payload['records']):
                raise TimeoutError('synthetic record-window failure')
        return super().chat_json(system, content)


class ProgressiveV7Tests(unittest.TestCase):
    def setUp(self):
        fixtures.ProgressiveV3Tests.setUp(self)
        self.model = WindowReviewer(self.clock)
        self.service = ProgressiveRecallV7(self.config, self.db, self.model, clock=self.clock)

    def window_fixture(self, *, chars=40):
        sources = {1: source('红色苹果' * 8, '蓝色海洋' * 8, '地下通道入口' * 8),
                   2: source('另外一个来源')}
        episodes = {1: {'source_id': 1, 'text': '地下通道入口'},
                    2: {'source_id': 1, 'text': '红色苹果'},
                    3: {'source_id': 2, 'text': '另外一个来源'}}
        policy = replace(RecallPolicy.for_request('test'), source_window_chars=chars)
        state = {'source_offsets': {}, 'pending_facts': [], 'policy': asdict(policy),
                 'metrics': {'review_waves': 0}}
        self.service._source_episodes = {1: [1, 2], 2: [3]}
        return sources, episodes, policy, state

    def long_sqlite_source(self):
        raw = source('红色苹果' + '甲' * 3500, '蓝色海洋' + '乙' * 3500,
                     '地下通道入口' + '丙' * 3500)
        with self.db.connection() as connection:
            connection.execute('UPDATE source SET raw_text=? WHERE id=1', (raw,))
            connection.execute('UPDATE episode SET text=? WHERE id=?', ('地下通道入口', self.eids[0]))
            connection.commit()
        return raw

    def test_success_retains_v6_prompts_and_evidence_but_reports_v7_windows(self):
        result = self.service.query('求助信后来怎么样？', learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['review_protocol_version'], 7)
        state = self.service.sessions.read(result['session_id'])
        self.assertTrue(all(t['version'] == 7 for t in state['review_trace']))
        self.assertEqual(self.model.need_batches, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual({f['episode_id'] for f in result['evidence']}, set(self.eids))
        self.assertEqual(len(result['mapped_source_windows']), 2)
        self.assertEqual(len(result['window_selection_trace']), 1)
        self.assertEqual(result['window_protocol']['source_window_chars'], 6000)
        self.assertEqual(result['window_protocol']['sources_per_review'], 4)
        mapped = next(a for a in result['stage_attempts'] if a['stage'] == 'map')
        self.assertEqual(mapped['windows'], result['mapped_source_windows'])

    def test_tail_selection_does_not_consume_prefix_or_unused_proposal(self):
        sources, episodes, policy, state = self.window_fixture()
        tail = self.service._windows([1], state, episodes, sources, policy)
        self.assertGreater(tail[0]['start'], 0)
        self.assertEqual(tail[0]['target_episode_ids'], [1])
        self.assertEqual(tail[0]['episode_ids'], [1, 2])
        self.assertEqual(state['source_offsets'], {})
        self.assertEqual(state['mapped_source_windows'], [])
        self.assertEqual(state['window_selection_trace'], [])
        self.assertEqual(self.service._windows([1], state, episodes, sources, policy), tail)
        self.service._commit_mapped_windows(tail, state)
        self.assertEqual(state['source_offsets']['1'], 0)
        earlier = self.service._windows([1], state, episodes, sources, policy)
        self.assertEqual(earlier[0]['start'], 0)
        self.assertEqual(earlier[0]['method'], 'sequential_unread_record')
        self.assertIn('红色苹果', earlier[0]['text'])
        self.assertLessEqual(earlier[0]['end'], tail[0]['start'])

    def test_candidate_source_order_and_episode_order_preserve_all_hints(self):
        sources, episodes, policy, state = self.window_fixture()
        windows = self.service._windows([3, 1, 2], state, episodes, sources, policy)
        self.assertEqual([w['source_id'] for w in windows], [2, 1])
        self.assertEqual(windows[1]['target_episode_ids'], [1])
        self.service._defer_windows([windows[1]], state)
        next_window = self.service._windows([1, 2], state, episodes, sources, policy)[0]
        self.assertEqual(next_window['target_episode_ids'], [2])
        self.assertEqual(next_window['episode_ids'], [1, 2])
        self.assertIn('红色苹果', next_window['text'])
        self.assertTrue(next_window['anchor_record_id'])

    def test_fallback_handles_large_header_and_preserves_whole_records(self):
        sources, episodes, policy, state = self.window_fixture(chars=5)
        sources[1] = 'Unrelated header ' * 30 + '\n\n' + source('完整记录' * 20)
        episodes[1]['text'] = episodes[2]['text'] = 'quantum'
        window = self.service._windows([1], state, episodes, sources, policy)[0]
        projection = project_window(1, sources[1], [1, 2], start=window['start'], end=window['end'])
        self.assertEqual(len(projection.records), 1)
        self.assertFalse(projection.incomplete_ranges)
        self.assertEqual(window['start'], 0)
        self.assertEqual(window['end'], len(sources[1]))
        self.assertTrue(window['overflow'])
        self.assertEqual(window['overflow_chars'], len(sources[1]) - 5)

    def test_recheck_does_not_change_read_ledger_and_avoids_only_failed_window(self):
        sources, episodes, policy, state = self.window_fixture(chars=2)
        locator = self.service._bind_window_state(state, sources, episodes, policy)
        records = locator.records(1)
        facts = []
        for index, record in enumerate(records):
            fact = {'source_id': 1, 'episode_id': 1, 'start': record['start'], 'end': record['end'],
                    'quote': record['text'], 'claim': record['text'],
                    'source_sha256': record['source_sha256']}
            fact['fact_id'] = stable_fact_id(fact)
            facts.append(fact)
        state['pending_facts'] = [facts[2], facts[0]]
        tail = self.service._windows([1], state, episodes, sources, policy)
        self.service._defer_windows(tail, state)
        contexts = self.service._context_windows(state, sources, episodes, policy)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]['recheck_fact_ids'], [facts[0]['fact_id']])
        self.assertEqual(contexts[0]['episode_ids'], [1, 2])
        self.assertTrue(contexts[0]['recheck'])
        before = deepcopy(state['source_offsets'])
        self.service._commit_mapped_windows(contexts, state)
        self.assertEqual(state['mapped_source_windows'], [])
        self.assertEqual(state['source_offsets'], before)
        state['review_work'] = {'windows': contexts}
        self.assertEqual(self.service._context_windows(state, sources, episodes, policy), contexts)

    def test_sqlite_window_failure_does_not_defer_the_whole_source(self):
        self.long_sqlite_source()
        self.model.fail_text = '地下通道入口'
        result = self.service.query('求助信后来怎么样？', learn=False, max_waves=2)
        self.assertFalse(result['complete'])
        state = self.service.sessions.read(result['session_id'])
        deferred = result['completion_blockers']['deferred_source_windows']
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]['source_id'], 1)
        self.assertGreater(deferred[0]['start'], 6000)
        first_source = [w for w in result['mapped_source_windows'] if w['source_id'] == 1]
        self.assertEqual(len(first_source), 1)
        self.assertEqual(first_source[0]['start'], 0)
        self.assertGreater(state['source_offsets']['1'], 0)
        self.assertLessEqual(state['source_offsets']['1'], deferred[0]['start'])
        attempts = [a for a in result['stage_attempts'] if a['stage'] == 'map']
        self.assertEqual([a['status'] for a in attempts], ['failed', 'failed', 'committed', 'committed'])
        self.assertEqual({w['source_id'] for w in attempts[0]['windows']}, {1, 2})
        self.assertEqual(attempts[0]['windows'][0]['window_id'], attempts[1]['windows'][0]['window_id'])

    def test_sqlite_commit_before_map_cursor_resume_is_idempotent(self):
        self.long_sqlite_source()
        write = self.service.sessions.write
        interrupted = []
        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if not interrupted and 'map:0' in work.get('committed_stages', {}) and work['map_cursor'] == 0:
                interrupted.append(True)
                raise KeyboardInterrupt()
        self.service.sessions.write = crash
        first = self.service.query('求助信后来怎么样？', learn=False)
        self.assertEqual(first['status'], 'cancelled', first)
        checkpoint = self.service.sessions.read(first['session_id'])
        original_windows = deepcopy(checkpoint['review_work']['windows'])
        self.assertEqual(checkpoint['mapped_source_windows'], [])
        restored = ProgressiveRecallV7(self.config, self.db, self.model, clock=self.clock)
        result = restored.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertTrue(result['complete'], result)
        self.assertEqual(len(self.model.map_payloads), 1)
        self.assertEqual(result['mapped_source_windows'], self.service._window_refs(original_windows))
        state = restored.sessions.read(first['session_id'])
        self.assertEqual(state['source_offsets']['1'], 0)
        self.assertEqual(len(result['window_selection_trace']), 1)

    def test_sqlite_failed_group_before_cursor_resume_recovers_each_window_once(self):
        self.long_sqlite_source()
        self.model.fail_text = '地下通道入口'
        write = self.service.sessions.write
        interrupted = []
        def crash(state):
            write(state)
            work = state.get('review_work') or {}
            if not interrupted and 'map:0' in work.get('failed_stages', {}) and work['map_cursor'] == 0:
                interrupted.append(True)
                raise KeyboardInterrupt()
        self.service.sessions.write = crash
        first = self.service.query('求助信后来怎么样？', learn=False)
        self.assertEqual(first['status'], 'cancelled', first)
        restored = ProgressiveRecallV7(self.config, self.db, self.model, clock=self.clock)
        result = restored.query('求助信后来怎么样？', resume=first['session_id'], learn=False, max_waves=1)
        self.assertEqual(len(self.model.map_payloads), 3)
        self.assertEqual(len(result['completion_blockers']['deferred_source_windows']), 1)
        self.assertEqual(len(result['mapped_source_windows']), 1)
        self.assertEqual(result['mapped_source_windows'][0]['source_id'], 2)
        self.assertEqual(len(result['window_selection_trace']), 1)

    def test_v6_checkpoint_and_changed_locator_are_rejected_before_any_model_call(self):
        previous = ProgressiveRecallV6(self.config, self.db, self.model, clock=self.clock)
        old = previous.query('求助信后来怎么样？', learn=False)
        count = len(self.model.map_payloads)
        with self.assertRaisesRegex(ValueError, 'another review protocol'):
            self.service.query('求助信后来怎么样？', resume=old['session_id'], learn=False)
        self.assertEqual(len(self.model.map_payloads), count)
        first = self.service.query('求助信后来怎么样？', learn=False)
        class ChangedLocator(ProgressiveRecallV7):
            window_protocol_version = 2
        changed = ChangedLocator(self.config, self.db, self.model, clock=self.clock)
        count = len(self.model.actual)
        with self.assertRaisesRegex(ValueError, 'changed knowledge'):
            changed.query('求助信后来怎么样？', resume=first['session_id'], learn=False)
        self.assertEqual(len(self.model.actual), count)

    def test_changed_source_or_episode_invalidates_bound_locator_state(self):
        for change in ('source', 'episode'):
            with self.subTest(change=change):
                sources, episodes, policy, state = self.window_fixture()
                self.service._windows([1], state, episodes, sources, policy)
                if change == 'source':
                    sources[1] += '\nchanged'
                else:
                    episodes[1]['text'] = '另一个摘要'
                with self.assertRaisesRegex(ValueError, 'locator or policy changed'):
                    self.service._windows([1], state, episodes, sources, policy)


if __name__ == '__main__':
    unittest.main()
