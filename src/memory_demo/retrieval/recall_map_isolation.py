"""Experimental fact isolation over the unchanged V10/V11 MAP transaction.

Only invalid individual fact proposals are filtered. The original reviewer
still validates and commits the complete effective response, including links,
cues, duplicate merging and pending-only updates. No new pipeline protocol is
registered here: partial success changes scheduler recovery and needs its own
end-to-end experiment.
"""
from copy import deepcopy

from memory_demo.retrieval.recall_review_v3 import (
    MAP_SYSTEM, _object, _list, _refs, _text, _indices,
)
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10
from memory_demo.retrieval.recall_review_v11 import RecordReviewV11


FACT_ISOLATION_PROTOCOL = {
    'version': 1,
    'name': 'individual-map-fact-validation',
    'scope': 'fact-filter-before-unchanged-map-transaction',
    'all_invalid_nonempty_facts': 'fail-stage',
    'envelope_and_original_fact_limit': 'unchanged',
    'links_and_optional_cues': 'unchanged',
    'fact_merging': 'unchanged',
}


class _FactIsolation:
    def map(self, records, state):
        # V11 already uses a per-call callback wrapper. Keep the same sequential
        # instance contract and always clear context after failure or success.
        if hasattr(self, '_fact_isolation_context'):
            raise ValueError('a MAP reviewer cannot be reentered')
        self._fact_isolation_context = (records, state)
        try:
            return super().map(records, state)
        finally:
            del self._fact_isolation_context

    def _run(self, stage, work):
        if stage != 'map':
            return super()._run(stage, work)

        def annotated(trace):
            trace['map_fact_isolation'] = {
                'protocol': deepcopy(FACT_ISOLATION_PROTOCOL),
                'filter_status': 'not_processed',
                'raw_fact_count': None,
                'retained_ordinals': [],
                'rejected_proposals': [],
            }
            return work(trace)

        return super()._run(stage, annotated)

    def _call(self, system, payload, trace):
        if system != MAP_SYSTEM:
            return super()._call(system, payload, trace)
        records, state = self._fact_isolation_context
        # These are global checks, outside the per-item rejection handler.
        # The parent MAP has already checked stored facts before reaching here.
        registry = self._registry(state, records)
        needs = self._needs(state)
        short = trace['record_ids']
        response = super()._call(system, payload, trace)
        _object(response, {'facts', 'links'}, {'followup_cues'})
        facts = _list(response['facts'], 'facts')
        diagnostic = trace['map_fact_isolation']
        diagnostic['raw_fact_count'] = len(facts)
        if len(facts) > 8:
            raise ValueError('mapper may propose at most eight facts')

        retained = []
        for ordinal, item in enumerate(facts, 1):
            try:
                _object(item, {'record_ids', 'episode_id', 'interpretation', 'need_indices'})
                aliases = _refs(item['record_ids'], short, 'record_ids',
                                minimum=1, maximum=self.max_fact_records)
                interpretation = _text(item['interpretation'], 'interpretation')
                self._fact([short[a] for a in aliases], item['episode_id'], interpretation,
                           _indices(item['need_indices'], len(needs)), registry)
            except ValueError as error:
                diagnostic['rejected_proposals'].append({
                    'proposal_ordinal': ordinal, 'proposal': deepcopy(item),
                    'error_type': type(error).__name__, 'reason': str(error),
                })
            else:
                retained.append(deepcopy(item))
                diagnostic['retained_ordinals'].append(ordinal)

        if facts and not retained:
            diagnostic['filter_status'] = 'all_rejected'
            raise ValueError('no valid fact proposals remain; MAP stage not committed')
        if not diagnostic['rejected_proposals']:
            diagnostic['filter_status'] = 'unchanged'
            return response

        # Retained means eligible for the parent's full transaction validation,
        # not live admission. A bad link can still reject all of these facts.
        diagnostic['filter_status'] = 'partial'
        clean = {**deepcopy(response), 'facts': retained}
        trace['admitted_response'] = deepcopy(clean)
        return clean


class FactIsolatedReviewV10(_FactIsolation, RecordReviewV10):
    pass


class FactIsolatedReviewV11(_FactIsolation, RecordReviewV11):
    pass


def isolated_reviewer(protocol, sources, episodes, call):
    """Return an experimental reviewer; its base prompt and version are intact."""
    if type(protocol) is not int or protocol not in (10, 11):
        raise ValueError('fact isolation requires base protocol 10 or 11')
    cls = {10: FactIsolatedReviewV10, 11: FactIsolatedReviewV11}[protocol]
    return cls(sources, episodes, call)
