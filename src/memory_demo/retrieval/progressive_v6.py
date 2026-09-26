"""Opt-in batched sufficiency; V5 search, evidence and recovery are preserved."""
from copy import deepcopy

from memory_demo.retrieval.progressive_v5 import ProgressiveRecallV5
from memory_demo.retrieval.recall_cues import enqueue_followup_cues
from memory_demo.retrieval.recall_records import project_window
from memory_demo.retrieval.recall_review_v6 import RecordReviewV6


class ProgressiveRecallV6(ProgressiveRecallV5):
    protocol_version = 6
    reviewer_class = RecordReviewV6
    learning_verifier = 'progressive_record_review_v6'

    def _commit_mapped_windows(self, windows, state):
        for window in windows:
            if not window.get('recheck'):
                sid = str(window['source_id'])
                state['source_offsets'][sid] = max(state['source_offsets'].get(sid, 0), window['end'])

    def _defer_windows(self, windows, state):
        state.setdefault('deferred_source_windows', []).extend(self._window_refs(windows))

    def _review(self, windows, state, sources, episodes):
        if not state.get('review_work'):
            state['review_work'] = {
                'windows': deepcopy(windows), 'phase': 'map', 'map_groups': [deepcopy(windows)],
                'map_cursor': 0, 'mapped_windows': [], 'old_pending': [f['fact_id'] for f in state['pending_facts']],
                'batch_cursor': 0, 'need_cursor': 0, 'committed_stages': {},
            }
            self.sessions.write(state)
        work = state['review_work']
        reviewer = self.reviewer_class(sources, episodes, lambda system, payload: self._call(system, payload, state))
        if work['phase'] == 'map':
            while work['map_cursor'] < len(work['map_groups']):
                group = work['map_groups'][work['map_cursor']]
                records = []
                for window in group:
                    projection = project_window(window['source_id'], sources[window['source_id']],
                        window.get('episode_ids', [window['episode_id']]), start=window['start'], end=window['end'])
                    if projection.incomplete_ranges:
                        raise ValueError('review window contains an incomplete record')
                    records.extend(projection.records)
                state['current_record_ids'] = [r['record_id'] for r in records]
                work['attempt_context'] = {'windows': self._window_refs(group)}
                mapped = self._commit_stage(lambda: reviewer.map(records, state), state, 'map')
                if mapped is None:
                    if len(group) > 1:
                        # Try each complete Source window once after a failed
                        # multi-Source request, without changing original text.
                        work['map_groups'].extend([[w] for w in group])
                    else:
                        self._defer_windows(group, state)
                else:
                    self._commit_mapped_windows(group, state)
                    cues = enqueue_followup_cues(state.get('cue_queue'), mapped.followup_cues,
                        visible=group, sources=sources, episodes=episodes,
                        initial_cues=[state['question'], *state['cues'], *state['needs']])
                    state['cue_queue'] = cues.queue
                    state['metrics']['followup_cues_accepted'] += cues.stats['accepted']
                    work['mapped_windows'].extend(deepcopy(group))
                work['map_cursor'] += 1
                self.sessions.write(state)
            # Preserve V4's full successful-wave context for fact review; the
            # intervention here is failure recovery, not context selection.
            state['current_record_ids'] = list(dict.fromkeys(
                r['record_id'] for w in work['mapped_windows']
                for r in project_window(w['source_id'], sources[w['source_id']],
                    w.get('episode_ids', [w['episode_id']]), start=w['start'], end=w['end']).records))
            failed = set(state.get('failed_fact_ids', []))
            pending = [f['fact_id'] for f in state['pending_facts'] if f['fact_id'] not in failed]
            old = set(work['old_pending'])
            pending = [fid for fid in pending if fid not in old] + [fid for fid in pending if fid in old]
            accepted = [f['fact_id'] for f in state['facts'] if f['fact_id'] not in failed]
            if accepted:
                cursor = int(state.get('old_fact_review_cursor', 0)) % len(accepted)
                pending.extend((accepted[cursor:] + accepted[:cursor])[:2])
                state['old_fact_review_cursor'] = (cursor + min(2, len(accepted))) % len(accepted)
            work.update(phase='facts', batches=[{'fact_ids': pending[i:i + 4], 'recovery': False}
                                               for i in range(0, len(pending), 4)])
            self.sessions.write(state)
        if work['phase'] == 'facts':
            while work['batch_cursor'] < len(work['batches']):
                batch = work['batches'][work['batch_cursor']]
                work['attempt_context'] = deepcopy(batch)
                reviewed = self._commit_stage(lambda: reviewer.review_facts(state, fact_ids=batch['fact_ids']), state, 'facts')
                if reviewed is None:
                    if not batch['recovery']:
                        work['batches'].extend({'fact_ids': [fid], 'recovery': True} for fid in batch['fact_ids'])
                    else:
                        state['failed_fact_ids'] = sorted(set(state.get('failed_fact_ids', [])) | set(batch['fact_ids']))
                work['batch_cursor'] += 1
                self.sessions.write(state)
            indices = list(range(len(state['needs'])))
            work.update(phase='needs', need_batches=[
                {'need_indices': indices[i:i + 4], 'recovery': False}
                for i in range(0, len(indices), 4)])
            self.sessions.write(state)
        if work['phase'] == 'needs':
            while state['facts'] and work['need_cursor'] < len(work['need_batches']):
                batch = work['need_batches'][work['need_cursor']]
                indices = batch['need_indices']
                work['attempt_context'] = deepcopy(batch)
                reviewed = self._commit_stage(lambda: reviewer.review_needs(state, indices), state, 'need')
                failed = set(state.get('failed_need_indices', []))
                if reviewed is None:
                    failed.update(indices)
                    if not batch['recovery'] and len(indices) > 1:
                        work['need_batches'].extend({'need_indices': [i], 'recovery': True} for i in indices)
                else:
                    failed.difference_update(indices)
                state['failed_need_indices'] = sorted(failed)
                work['need_cursor'] += 1
                self.sessions.write(state)
            work['phase'] = 'links'
            self.sessions.write(state)
        if work['phase'] == 'links' and state.get('pending_links'):
            work['attempt_context'] = {'link_ids': [link['link_id'] for link in state['pending_links']]}
            self._commit_stage(lambda: reviewer.review_links(state), state, 'links')
        rechecked = set(state.get('context_rechecked_fact_ids', []))
        for window in work['mapped_windows']:
            rechecked.update(window.get('recheck_fact_ids', []))
        state['context_rechecked_fact_ids'] = sorted(rechecked)
        state['metrics']['source_windows_reviewed'] += len(work['mapped_windows'])
        state['metrics']['review_waves'] += 1
        state['review_work'] = None
        self.sessions.write(state)
