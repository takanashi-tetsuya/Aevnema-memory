"""Incremental FACTS during MAP recovery, with durable context-aware rechecks.

V7 navigation, admission, prompts, graph and sufficiency rules stay unchanged.
Only newly admitted facts for wave-entry gaps can interrupt remaining MAP work.
An early verdict is rechecked when the complete wave adds visible records.
"""
from copy import deepcopy

from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v7 import ProgressiveRecallV7
from memory_demo.retrieval.recall_cues import enqueue_followup_cues
from memory_demo.retrieval.recall_records import project_window
from memory_demo.retrieval.recall_review_v8 import RecordReviewV8


class ProgressiveRecallV8(ProgressiveRecallV7):
    protocol_version = 8
    reviewer_class = RecordReviewV8
    learning_verifier = 'progressive_record_review_v8'
    schedule_protocol = {'version': 1, 'name': 'gap-facts-before-map-recovery',
                         'max_fact_batch': 4, 'early_reviews_per_fact_per_wave': 1}

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        self._snapshot_payload['local_review_schedule'] = deepcopy(self.schedule_protocol)
        return sources, episodes, rows, _hash(self._snapshot_payload)

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        result['schedule_protocol'] = deepcopy(state.get('schedule_protocol', cls.schedule_protocol))
        result['local_review_trace'] = deepcopy(state.get('local_review_trace', []))
        return result

    @staticmethod
    def _visible_ids(state, source_ids):
        current = set(state.get('current_record_ids', []))
        return sorted(r['record_id'] for r in state.get('record_registry', {}).values()
                      if r['source_id'] in source_ids or r['record_id'] in current)

    @staticmethod
    def _wave_records(work, sources):
        return list(dict.fromkeys(r['record_id'] for w in work['mapped_windows']
            for r in project_window(w['source_id'], sources[w['source_id']], w['episode_ids'],
                                    start=w['start'], end=w['end']).records))

    def _append_batches(self, work, facts, kind):
        for offset in range(0, len(facts), 4):
            work['batches'].append({'fact_ids': facts[offset:offset + 4],
                                    'recovery': False, 'kind': kind})

    def _drain_facts(self, reviewer, state):
        work = state['review_work']
        while work['batch_cursor'] < len(work['batches']):
            batch = work['batches'][work['batch_cursor']]
            if 'context_record_ids' not in batch:
                available = {f['fact_id']: f for f in [*state['facts'], *state['pending_facts']]}
                batch['context_record_ids'] = self._visible_ids(
                    state, {available[fid]['source_id'] for fid in batch['fact_ids']})
                self.sessions.write(state)
            work['attempt_context'] = deepcopy(batch)
            reviewed = self._commit_stage(
                lambda: reviewer.review_facts(state, fact_ids=batch['fact_ids']), state, 'facts')
            if reviewed is None:
                if not batch['recovery']:
                    work['batches'].extend({'fact_ids': [fid], 'recovery': True, 'kind': batch['kind']}
                                           for fid in batch['fact_ids'])
                else:
                    state['failed_fact_ids'] = sorted(set(state.get('failed_fact_ids', [])) | set(batch['fact_ids']))
            elif batch['kind'] == 'early':
                for fid in batch['fact_ids']:
                    work['early_contexts'][fid] = list(batch['context_record_ids'])
            state['local_review_trace'].append({
                'wave': state['metrics']['review_waves'] + 1, 'kind': batch['kind'],
                'stage_token': f"facts:{work['batch_cursor']}", 'fact_ids': list(batch['fact_ids']),
                'context_record_ids': list(batch['context_record_ids']),
                'remaining_map_groups': len(work['map_groups']) - work['map_cursor'],
                'recovery': batch['recovery'], 'committed': reviewed is not None,
            })
            work['batch_cursor'] += 1
            self.sessions.write(state)

    def _finish_map(self, state, sources):
        work = state['review_work']
        state['current_record_ids'] = self._wave_records(work, sources)
        failed = set(state.get('failed_fact_ids', []))
        early = work['early_candidates']
        available = {f['fact_id']: f for f in [*state['facts'], *state['pending_facts']]}
        recheck = []
        for fid, original in early.items():
            previous = work['early_contexts'].get(fid)
            visible = set(self._visible_ids(state, {original['source_id']}))
            if fid not in failed and previous is not None and visible - set(previous):
                # A rejected local proposal has left both fact pools. New
                # context permits another independent review, never acceptance.
                if fid not in available:
                    restored = deepcopy(original)
                    restored['verification_status'] = 'pending'
                    state['pending_facts'].append(restored)
                    available[fid] = restored
                recheck.append(fid)
        pending = [f['fact_id'] for f in state['pending_facts']
                   if f['fact_id'] not in failed and (f['fact_id'] not in early or f['fact_id'] in recheck)]
        old = set(work['old_pending'])
        pending = [fid for fid in pending if fid not in old] + [fid for fid in pending if fid in old]
        pending.extend(fid for fid in recheck if fid not in pending)
        # Early accepts are not pre-wave accepted facts and do not receive the
        # ordinary rotating review merely because their decision happened first.
        old_accepted = set(work['old_accepted'])
        accepted = [f['fact_id'] for f in state['facts']
                    if f['fact_id'] in old_accepted and f['fact_id'] not in failed]
        if accepted:
            cursor = int(state.get('old_fact_review_cursor', 0)) % len(accepted)
            pending.extend(fid for fid in (accepted[cursor:] + accepted[:cursor])[:2] if fid not in pending)
            state['old_fact_review_cursor'] = (cursor + min(2, len(accepted))) % len(accepted)
        self._append_batches(work, pending, 'wave_final')
        work['phase'] = 'facts'
        self.sessions.write(state)

    def _review(self, windows, state, sources, episodes):
        self._bind_window_state(state, sources, episodes)
        if state.get('schedule_protocol', self.schedule_protocol) != self.schedule_protocol:
            raise ValueError('local review schedule changed; start a fresh recall')
        state.setdefault('schedule_protocol', deepcopy(self.schedule_protocol))
        state.setdefault('local_review_trace', [])
        work = state.get('review_work')
        selected = [w for group in work['map_groups'] for w in group] if work else windows
        for window in selected:
            self._validate_window(window, state, sources, episodes)
        if not work:
            state['window_selection_trace'].append({'wave': state['metrics']['review_waves'] + 1,
                                                     'windows': self._window_refs(windows)})
            resolved = set(state.get('resolved_needs', []))
            state['review_work'] = {
                'windows': deepcopy(windows), 'phase': 'map', 'map_groups': [deepcopy(windows)],
                'map_cursor': 0, 'mapped_windows': [], 'old_pending': [f['fact_id'] for f in state['pending_facts']],
                'old_accepted': [f['fact_id'] for f in state['facts']],
                'unresolved_at_entry': [i for i in range(len(state['needs'])) if i not in resolved],
                'map_before': {}, 'early_candidates': {}, 'early_contexts': {},
                'batch_cursor': 0, 'batches': [], 'need_cursor': 0, 'committed_stages': {},
            }
            self.sessions.write(state)
        work = state['review_work']
        reviewer = self.reviewer_class(sources, episodes, lambda system, payload: self._call(system, payload, state))
        while work['phase'] in {'map', 'early_facts'}:
            if work['phase'] == 'early_facts':
                self._drain_facts(reviewer, state)
                work['phase'] = 'map'
                self.sessions.write(state)
                continue
            if work['map_cursor'] >= len(work['map_groups']):
                self._finish_map(state, sources)
                break
            group = work['map_groups'][work['map_cursor']]
            records = []
            for window in group:
                projection = project_window(window['source_id'], sources[window['source_id']],
                    window['episode_ids'], start=window['start'], end=window['end'])
                if projection.incomplete_ranges:
                    raise ValueError('review window contains an incomplete record')
                records.extend(projection.records)
            state['current_record_ids'] = [r['record_id'] for r in records]
            cursor = str(work['map_cursor'])
            if cursor not in work['map_before']:
                work['map_before'][cursor] = [f['fact_id'] for f in [*state['facts'], *state['pending_facts']]]
                self.sessions.write(state)
            work['attempt_context'] = {'windows': self._window_refs(group)}
            mapped = self._commit_stage(lambda: reviewer.map(records, state), state, 'map')
            if mapped is None:
                if len(group) > 1:
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
                if work['map_cursor'] + 1 < len(work['map_groups']):
                    before = set(work['map_before'][cursor])
                    gaps = set(work['unresolved_at_entry'])
                    failed = set(state.get('failed_fact_ids', []))
                    chosen = [f for f in state['pending_facts'] if f['fact_id'] not in before
                              and f['fact_id'] not in work['early_candidates'] and f['fact_id'] not in failed
                              and gaps.intersection(f['need_indices'])]
                    if chosen:
                        work['early_candidates'].update({f['fact_id']: deepcopy(f) for f in chosen})
                        self._append_batches(work, [f['fact_id'] for f in chosen], 'early')
                        state['current_record_ids'] = self._wave_records(work, sources)
                        work['phase'] = 'early_facts'
            work['map_cursor'] += 1
            self.sessions.write(state)
        if work['phase'] == 'facts':
            self._drain_facts(reviewer, state)
            indices = list(range(len(state['needs'])))
            work.update(phase='needs', need_batches=[{'need_indices': indices[i:i + 4], 'recovery': False}
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
