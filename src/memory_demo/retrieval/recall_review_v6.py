"""Atomic sufficiency batches; all evidence and context rules remain V5."""
from copy import deepcopy

from config.prompt_config.recall_v6_prompts import NEEDS
from memory_demo.retrieval.recall_review_v3 import (
    RecordReview, _indices, _list, _object, _refs, _text,
)
from memory_demo.retrieval.recall_review_v5 import RecordReviewV5


class RecordReviewV6(RecordReviewV5):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 6
            return work(trace)
        return RecordReview._run(self, stage, versioned)

    def review_need(self, state, need_index):
        return self.review_needs(state, [need_index])

    def review_needs(self, state, need_indices):
        def work(trace):
            needs = self._needs(state)
            selected = _indices(need_indices, len(needs))
            if not 1 <= len(selected) <= 4:
                raise ValueError('need batch must contain one to four targets')
            registry = self._registry(state)
            accepted = self._facts(state, registry, 'facts')
            payload, record_short = self._common(state, registry, trace,
                source_ids={f['source_id'] for f in accepted.values()})
            # Keep all accepted facts and full previously visible context,
            # including evidence initially mapped to another requirement.
            payload = {'request_context': payload['question'],
                       'targets': [{'need_index': i, 'question': needs[i]} for i in selected],
                       'records': payload['records']}
            payload['facts'], short = self._fact_payload(accepted, record_short, state)
            trace['fact_ids'] = deepcopy(short)
            trace['need_indices'] = list(selected)
            decisions = {}
            if accepted:
                response = self._call(NEEDS, payload, trace)
                _object(response, {'assessments'})
                for row in _list(response['assessments'], 'assessments'):
                    _object(row, {'need_index', 'status', 'fact_ids', 'answer', 'reason'})
                    i = row['need_index']
                    if type(i) is not int or i not in selected or i in decisions:
                        raise ValueError('unknown or duplicate need_index')
                    if row['status'] not in {'supported', 'partial', 'unknown', 'refuted'}:
                        raise ValueError('unknown requirement status')
                    aliases = _refs(row['fact_ids'], short, 'fact_ids')
                    resolved = row['status'] in {'supported', 'refuted'}
                    if resolved and not aliases:
                        raise ValueError('resolved requirement must cite accepted facts')
                    decisions[i] = {'need_index': i, 'status': row['status'],
                        'fact_ids': [short[a] for a in aliases],
                        'answer': _text(row['answer'], 'answer', empty=not resolved),
                        'reason': _text(row['reason'], 'reason')}
                if set(decisions) != set(selected):
                    raise ValueError('need batch omitted a target')
            else:
                trace['skipped'] = 'no_accepted_facts'
                decisions = {i: self._unknown(i, 'No accepted original evidence for this requirement yet.')
                             for i in selected}
            previous = {a['need_index']: deepcopy(a) for a in state.get('need_assessments', [])}
            previous.update(decisions)
            assessments = []
            for i in range(len(needs)):
                item = previous.get(i, self._unknown(i, 'Not reviewed yet.'))
                if (any(fid not in accepted for fid in item['fact_ids'])
                        or item['status'] in {'supported', 'refuted'} and not item['fact_ids']):
                    item = self._unknown(i, 'Previous supporting evidence is no longer accepted.')
                assessments.append(item)
            trace['need_assessments'] = [deepcopy(decisions[i]) for i in selected]
            return {'need_assessments': assessments,
                    'covered_needs': [a['need_index'] for a in assessments if a['status'] == 'supported'],
                    'resolved_needs': [a['need_index'] for a in assessments if a['status'] in {'supported', 'refuted'}]}, []
        return self._run('need', work)

    @staticmethod
    def _unknown(index, reason):
        return {'need_index': index, 'status': 'unknown', 'fact_ids': [], 'answer': '', 'reason': reason}
