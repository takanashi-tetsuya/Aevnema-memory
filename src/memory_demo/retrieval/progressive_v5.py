"""Opt-in bounded failure recovery; V4 planning and retrieval stay unchanged."""
from copy import deepcopy

from memory_demo.llm.client import CampaignBudgetError, ModelClientError
from memory_demo.retrieval.progressive_v4 import ProgressiveRecallV4
from memory_demo.retrieval.recall_cues import enqueue_followup_cues
from memory_demo.retrieval.recall_records import project_window
from memory_demo.retrieval.recall_review_v3 import ReviewStageResult
from memory_demo.retrieval.recall_review_v5 import RecordReviewV5


class ProgressiveRecallV5(ProgressiveRecallV4):
    protocol_version = 5
    reviewer_class = RecordReviewV5
    learning_verifier = 'progressive_record_review_v5'

    @staticmethod
    def completion_blockers(state):
        return {
            'pending_fact_ids': [f['fact_id'] for f in state.get('pending_facts', [])],
            'failed_fact_ids': list(state.get('failed_fact_ids', [])),
            'deferred_source_windows': deepcopy(state.get('deferred_source_windows', [])),
            'failed_need_indices': list(state.get('failed_need_indices', [])),
        }

    @staticmethod
    def _resolved(state):
        return (ProgressiveRecallV4._resolved(state)
                and not any(ProgressiveRecallV5.completion_blockers(state).values()))

    @classmethod
    def annotate_completion(cls, result, state):
        result['completion_blockers'] = cls.completion_blockers(state)
        result['stage_attempts'] = deepcopy(state.get('stage_attempts', []))
        if any(result['completion_blockers'].values()):
            # Supported NEED verdicts alone cannot certify unresolved review.
            result.update(complete=False, answer_status='partial' if state.get('facts') else 'unknown')
            if not result.get('missing_requirements'):
                result['missing_requirements'] = list(state.get('needs', [])) or [state['question']]
        return result

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        return self.annotate_completion(result, self.sessions.read(result['session_id']))

    @staticmethod
    def _deferred_sources(state):
        return {w['source_id'] for w in state.get('deferred_source_windows', [])}

    def _windows(self, candidates, state, episodes, sources, policy):
        state['candidate_episode_ids'] = list(dict.fromkeys([
            *state.get('candidate_episode_ids', []), *candidates]))[:100]
        deferred = self._deferred_sources(state)
        return super()._windows([eid for eid in candidates if eid in episodes
                                 and episodes[eid]['source_id'] not in deferred], state, episodes, sources, policy)

    def _context_windows(self, state, sources, episodes, policy):
        if state.get('review_work'):
            return deepcopy(state['review_work']['windows'])
        deferred = self._deferred_sources(state)
        return [w for w in super()._context_windows(state, sources, episodes, policy)
                if w['source_id'] not in deferred]

    def _commit_stage(self, operation, state, stage):
        self._stage_name = stage
        self._ensure_live()
        work = state['review_work']
        cursor = work.get({'map': 'map_cursor', 'facts': 'batch_cursor', 'need': 'need_cursor'}.get(stage, ''), 0)
        token = f'{stage}:{cursor}'
        cached = work['committed_stages'].get(token)
        if cached is not None:
            return ReviewStageResult({}, {'resumed_committed_stage': token}, deepcopy(cached['followup_cues']))
        if token in work.get('failed_stages', {}):
            # A crash between recording failure and scheduling recovery must
            # not grant another attempt at the same failed batch.
            return None
        attempt = {'stage': stage, 'token': token, 'wave': state['metrics']['review_waves'] + 1,
                   'status': 'started', **deepcopy(work.get('attempt_context', {}))}
        state.setdefault('stage_attempts', []).append(attempt)
        self.sessions.write(state)
        try:
            result = operation()
            self._ensure_live()
        except CampaignBudgetError:
            attempt['status'] = 'budget_exhausted'
            self.sessions.write(state)
            raise
        except (ValueError, TimeoutError, ModelClientError) as exc:
            trace = deepcopy(getattr(exc, 'trace', None) or {})
            attempt.update(status='failed', error_type=trace.get('provider_error_type', type(exc).__name__))
            state.setdefault('review_trace', []).append({**trace, 'committed': False, 'stage': stage,
                'wave': attempt['wave'], 'attempt_context': deepcopy(work.get('attempt_context', {}))})
            state.setdefault('stage_errors', []).append({'stage': stage, 'type': attempt['error_type'],
                'message': str(exc)[:400], 'wave': attempt['wave'], 'token': token})
            work.setdefault('failed_stages', {})[token] = {'error_type': attempt['error_type']}
            self.sessions.write(state)
            self._ensure_live()
            return None
        state.update(result.updates)
        attempt['status'] = 'committed'
        state.setdefault('review_trace', []).append({**deepcopy(result.trace), 'committed': True,
            'stage': stage, 'wave': attempt['wave'], 'attempt_context': deepcopy(work.get('attempt_context', {}))})
        work['committed_stages'][token] = {'followup_cues': deepcopy(result.followup_cues)}
        self.sessions.write(state)
        return result

    @staticmethod
    def _window_refs(windows):
        return [{k: w[k] for k in ('source_id', 'start', 'end')} for w in windows]

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
                        state.setdefault('deferred_source_windows', []).extend(self._window_refs(group))
                else:
                    for window in group:
                        if not window.get('recheck'):
                            sid = str(window['source_id'])
                            state['source_offsets'][sid] = max(state['source_offsets'].get(sid, 0), window['end'])
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
            work['phase'] = 'needs'
            self.sessions.write(state)
        if work['phase'] == 'needs':
            while state['facts'] and work['need_cursor'] < len(state['needs']):
                index = work['need_cursor']
                work['attempt_context'] = {'need_index': index}
                reviewed = self._commit_stage(lambda: reviewer.review_need(state, index), state, 'need')
                failed = set(state.get('failed_need_indices', []))
                failed.add(index) if reviewed is None else failed.discard(index)
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
