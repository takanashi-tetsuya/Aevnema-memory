"""Opt-in Episode anchors with durable, non-contiguous Source reading progress.

Search ranking, Source-wide Episode hints, review prompts, evidence admission,
and completion remain V6. Anchor scores propose reading locations, never facts.
"""
from copy import deepcopy

from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v3 import ProgressiveRecallV3
from memory_demo.retrieval.progressive_v6 import ProgressiveRecallV6
from memory_demo.retrieval.recall_anchors import EpisodeAnchorIndex
from memory_demo.retrieval.recall_records import align_window, project_window
from memory_demo.retrieval.recall_review_v7 import RecordReviewV7


class ProgressiveRecallV7(ProgressiveRecallV6):
    protocol_version = 7
    reviewer_class = RecordReviewV7
    learning_verifier = 'progressive_record_review_v7'
    window_protocol_version = 1

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        locator = self._locator_for(sources, episodes)
        # The base resumption check happens before PLAN/embedding/model work.
        # Keep this in the same payload used by committed-learning recovery.
        self._snapshot_payload['window_locator'] = {
            'version': self.window_protocol_version, 'fingerprint': locator.fingerprint,
        }
        return sources, episodes, rows, _hash(self._snapshot_payload)

    def _locator_for(self, sources, episodes):
        identity = _hash({
            'sources': [(sid, _hash(raw)) for sid, raw in sorted(sources.items())],
            'episodes': [(eid, e['source_id'], _hash(e['text'])) for eid, e in sorted(episodes.items())],
        })
        if getattr(self, '_locator_input_hash', None) != identity:
            self._locator = EpisodeAnchorIndex(sources, episodes)
            self._locator_input_hash = identity
        return self._locator

    def _bind_window_state(self, state, sources, episodes, policy=None):
        locator = self._locator_for(sources, episodes)
        settings = state.get('policy', {})
        chars = policy.source_window_chars if policy is not None else settings.get('source_window_chars')
        count = policy.sources_per_review if policy is not None else settings.get('sources_per_review')
        identity = {
            'version': self.window_protocol_version, 'locator_fingerprint': locator.fingerprint,
            'source_window_chars': chars, 'sources_per_review': count,
        }
        previous = state.get('window_protocol')
        if previous is not None and previous != identity:
            raise ValueError('recall window locator or policy changed; start a fresh recall')
        state.setdefault('window_protocol', identity)
        state.setdefault('mapped_source_windows', [])
        state.setdefault('window_selection_trace', [])
        return locator

    @staticmethod
    def _overlaps(start, end, intervals):
        return any(start < hi and end > lo for lo, hi in intervals)

    @staticmethod
    def _intervals(state, source_id, *, deferred_only=False):
        rows = state.get('deferred_source_windows', [])
        if not deferred_only:
            rows = [*state.get('mapped_source_windows', []), *rows]
        return sorted({(w['start'], w['end']) for w in rows if w['source_id'] == source_id})

    def _describe_window(self, window, state, sources, episodes, *, method, target_episode_ids,
                         anchor_record_id=None, anchor_score=None):
        result = deepcopy(window)
        sid = result['source_id']
        # Keep V3/V6's source-wide hint and binding scope. The target is a
        # diagnostic navigation choice, not a new verified Episode binding.
        result['episode_ids'] = sorted(eid for eid, e in episodes.items() if e['source_id'] == sid)
        result.update(
            source_sha256=_hash(sources[sid]),
            window_protocol_version=self.window_protocol_version,
            locator_fingerprint=state['window_protocol']['locator_fingerprint'],
            method=method, target_episode_ids=list(target_episode_ids),
            anchor_record_id=anchor_record_id, anchor_score=anchor_score,
        )
        result['window_id'] = 'w_' + _hash({k: result[k] for k in (
            'source_id', 'source_sha256', 'start', 'end', 'window_protocol_version',
            'locator_fingerprint', 'method', 'anchor_record_id')})
        return result

    def _fallback_window(self, sid, eid, locator, unavailable, sources, episodes, policy):
        records = locator.records(sid)
        for index, record in enumerate(records):
            lo, hi = record['context_start'], record['context_end']
            if self._overlaps(lo, hi, unavailable):
                continue
            # Include a possible leading legend/header only when the first
            # record is still unread, so contiguous-prefix progress stays exact.
            start = 0 if index == 0 else lo
            if self._overlaps(start, lo, unavailable):
                start = lo
            boundary = min([len(sources[sid])] + [a for a, b in unavailable if a >= hi])
            # A long header or first record must never yield an empty or
            # truncated record projection, even if it exceeds the nominal cap.
            end = min(boundary, max(hi, start + policy.source_window_chars))
            start, end = align_window(sources[sid], start, end)
            if self._overlaps(start, end, unavailable):
                continue
            return {'source_id': sid, 'episode_id': eid, 'episode_hint': episodes[eid]['text'],
                    'start': start, 'end': end, 'text': sources[sid][start:end],
                    'overflow': end - start > policy.source_window_chars,
                    'overflow_chars': max(0, end - start - policy.source_window_chars)}
        return None

    def _windows(self, candidates, state, episodes, sources, policy):
        locator = self._bind_window_state(state, sources, episodes, policy)
        state['candidate_episode_ids'] = list(dict.fromkeys([
            *state.get('candidate_episode_ids', []), *candidates]))[:100]
        by_source = {}
        for eid in candidates:
            if eid in episodes and episodes[eid]['source_id'] in sources:
                sid = episodes[eid]['source_id']
                if eid not in by_source.setdefault(sid, []):
                    by_source[sid].append(eid)
        windows = []
        for sid, eids in by_source.items():
            unavailable = self._intervals(state, sid)
            selected, selected_eid = None, eids[0]
            for eid in eids:
                anchored = locator.window(eid, unavailable=unavailable, max_chars=policy.source_window_chars)
                if anchored is not None:
                    selected_eid = eid
                    selected = {**anchored, 'source_id': sid, 'episode_id': eid,
                                'episode_hint': episodes[eid]['text'],
                                'text': sources[sid][anchored['start']:anchored['end']]}
                    break
            if selected is None:
                selected = self._fallback_window(sid, selected_eid, locator, unavailable,
                                                 sources, episodes, policy)
                if selected is None:
                    continue
                selected['method'] = 'sequential_unread_record'
            windows.append(self._describe_window(
                selected, state, sources, episodes, method=selected['method'],
                target_episode_ids=[selected_eid], anchor_record_id=selected.get('anchor_record_id'),
                anchor_score=selected.get('anchor_score')))
            if len(windows) >= policy.sources_per_review:
                break
        # This is only a proposal. query() also calls _windows for shortcut
        # candidates when a context recheck will actually take precedence.
        return windows

    def _context_windows(self, state, sources, episodes, policy):
        self._bind_window_state(state, sources, episodes, policy)
        if state.get('review_work'):
            return deepcopy(state['review_work']['windows'])
        # Bypass only V5's Source-wide deferred filter; retain complete-record
        # expansion and each pending fact's bounded recheck identity from V3.
        eligible = []
        for fact in state.get('pending_facts', []):
            sid = fact['source_id']
            start, end = align_window(sources[sid],
                max(0, fact['start'] - policy.source_window_chars),
                min(len(sources[sid]), fact['end'] + policy.source_window_chars))
            if not self._overlaps(start, end, self._intervals(state, sid, deferred_only=True)):
                eligible.append(fact)
        # Filter before the source/window cap so failed early candidates do not
        # starve unrelated pending context later in the queue.
        windows = ProgressiveRecallV3._context_windows(
            self, {**state, 'pending_facts': eligible}, sources, episodes, policy)
        result = []
        for window in windows:
            failed = self._intervals(state, window['source_id'], deferred_only=True)
            if self._overlaps(window['start'], window['end'], failed):
                continue
            result.append(self._describe_window(
                window, state, sources, episodes, method='context_recheck',
                target_episode_ids=sorted({f['episode_id'] for f in state['pending_facts']
                                          if f['fact_id'] in window.get('recheck_fact_ids', [])})))
        return result

    @staticmethod
    def _window_refs(windows):
        keys = ('source_id', 'start', 'end', 'source_sha256', 'window_id',
                'window_protocol_version', 'locator_fingerprint', 'method',
                'target_episode_ids', 'anchor_record_id', 'anchor_score',
                'recheck', 'recheck_fact_ids', 'overflow', 'overflow_chars')
        return [{k: deepcopy(w[k]) for k in keys if k in w} for w in windows]

    def _validate_window(self, window, state, sources, episodes):
        sid = window['source_id']
        if (sid not in sources or window.get('source_sha256') != _hash(sources[sid])
                or window.get('locator_fingerprint') != state['window_protocol']['locator_fingerprint']
                or window.get('window_protocol_version') != self.window_protocol_version):
            raise ValueError('recall window identity changed')
        start, end = window['start'], window['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sources[sid]):
            raise ValueError('invalid recall window bounds')
        if window['text'] != sources[sid][start:end]:
            raise ValueError('recall window text changed')
        allowed = sorted(eid for eid, e in episodes.items() if e['source_id'] == sid)
        if window.get('episode_ids') != allowed or window['episode_id'] not in allowed:
            raise ValueError('recall window Episode bindings changed')
        canonical = self._describe_window(
            window, state, sources, episodes, method=window['method'],
            target_episode_ids=window['target_episode_ids'],
            anchor_record_id=window.get('anchor_record_id'), anchor_score=window.get('anchor_score'))
        if canonical['window_id'] != window.get('window_id'):
            raise ValueError('recall window identifier changed')
        projection = project_window(sid, sources[sid], allowed, start=start, end=end)
        if projection.incomplete_ranges or not projection.records:
            raise ValueError('review window must contain complete original records')

    def _review(self, windows, state, sources, episodes):
        self._bind_window_state(state, sources, episodes)
        work = state.get('review_work')
        selected = [w for group in work['map_groups'] for w in group] if work else windows
        for window in selected:
            self._validate_window(window, state, sources, episodes)
        if not work:
            state['window_selection_trace'].append({
                'wave': state['metrics']['review_waves'] + 1,
                'windows': self._window_refs(windows),
            })
        return super()._review(windows, state, sources, episodes)

    def _commit_mapped_windows(self, windows, state):
        recorded = {w['window_id'] for w in state['mapped_source_windows']}
        for window in windows:
            if not window.get('recheck') and window['window_id'] not in recorded:
                state['mapped_source_windows'].extend(self._window_refs([window]))
                recorded.add(window['window_id'])
        # Only the union's contiguous prefix is a Source reading cursor.
        # Successful jumps never claim that earlier text has been consumed.
        source_ids = {w['source_id'] for w in state['mapped_source_windows']}
        for sid in source_ids:
            prefix = 0
            for lo, hi in sorted((w['start'], w['end']) for w in state['mapped_source_windows']
                                 if w['source_id'] == sid):
                if lo > prefix:
                    break
                prefix = max(prefix, hi)
            state['source_offsets'][str(sid)] = prefix

    def _defer_windows(self, windows, state):
        deferred = state.setdefault('deferred_source_windows', [])
        existing = {w['window_id'] for w in deferred}
        for window in windows:
            if window['window_id'] not in existing:
                deferred.extend(self._window_refs([window]))
                existing.add(window['window_id'])

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        for key in ('window_protocol', 'mapped_source_windows', 'window_selection_trace'):
            result[key] = deepcopy(state.get(key, {} if key == 'window_protocol' else []))
        return result
