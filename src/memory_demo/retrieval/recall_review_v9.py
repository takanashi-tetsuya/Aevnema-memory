"""V8 fact/source rules with explicit, question-bound NEED support items."""
from copy import deepcopy

from config.prompt_config.recall_v9_prompts import CONTRACT, NEEDS
from memory_demo.retrieval.recall_need_contracts import (
    bound_contracts, contract_input, derive_assessment, unknown_assessment,
    valid_assessment, validate_contracts,
)
from memory_demo.retrieval.recall_review_v3 import RecordReview, _indices, _list, _object, _refs, _text
from memory_demo.retrieval.recall_review_v8 import RecordReviewV8


class RecordReviewV9(RecordReviewV8):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 9
            return work(trace)
        return RecordReview._run(self, stage, versioned)

    def build_contracts(self, state):
        def work(trace):
            self._needs(state)
            if state.get('record_registry') or state.get('facts') or state.get('pending_facts'):
                raise ValueError('contracts must be fixed before reading evidence')
            response = self._call(CONTRACT, contract_input(state), trace)
            _object(response, {'contracts'})
            updates = validate_contracts(state, response['contracts'])
            trace.update(deepcopy(updates))
            return updates, []
        return self._run('contract', work)

    def review_needs(self, state, need_indices):
        def work(trace):
            contracts = bound_contracts(state)
            selected = _indices(need_indices, len(contracts))
            if not 1 <= len(selected) <= 4:
                raise ValueError('need batch must contain one to four targets')
            registry = self._registry(state)
            accepted = self._facts(state, registry, 'facts')
            common, record_short = self._common(state, registry, trace,
                source_ids={f['source_id'] for f in accepted.values()})
            payload = {'request_context': state['question'], 'context': state.get('context', ''),
                       'targets': [deepcopy(contracts[i]) for i in selected],
                       'records': common['records']}
            payload['facts'], fact_short = self._fact_payload(accepted, record_short, state)
            trace.update(fact_ids=deepcopy(fact_short), need_indices=list(selected),
                         need_contract_hash=state['need_contract_hash'])
            decisions = {}
            if accepted:
                response = self._call(NEEDS, payload, trace)
                _object(response, {'assessments'})
                for row in _list(response['assessments'], 'assessments'):
                    _object(row, {'need_index', 'items'})
                    index = row['need_index']
                    if type(index) is not int or index not in selected or index in decisions:
                        raise ValueError('unknown or duplicate need_index')
                    contract = contracts[index]
                    allowed_items = {i['item_id'] for i in contract['items']}
                    parsed, seen = [], set()
                    for item in _list(row['items'], 'item assessments'):
                        _object(item, {'item_id', 'status', 'value', 'fact_ids', 'record_ids', 'reason'})
                        item_id, status = item['item_id'], item['status']
                        if not isinstance(item_id, str) or item_id not in allowed_items or item_id in seen:
                            raise ValueError('unknown or duplicate contract item')
                        seen.add(item_id)
                        if not isinstance(status, str) or status not in {'supported', 'unknown', 'contradicted'}:
                            raise ValueError('unknown contract item status')
                        minimum = int(status != 'unknown')
                        fids = [fact_short[a] for a in _refs(item['fact_ids'], fact_short, 'fact_ids', minimum=minimum)]
                        rids = [record_short[a] for a in _refs(item['record_ids'], record_short, 'record_ids', minimum=minimum)]
                        direct = {rid for fid in fids for rid in accepted[fid]['record_ids']}
                        if not set(rids) <= direct or any(not set(rids).intersection(accepted[fid]['record_ids']) for fid in fids):
                            raise ValueError('item records must be directly bound to every cited accepted fact')
                        parsed.append({'item_id': item_id, 'status': status,
                            'value': _text(item['value'], 'item value', empty=not minimum),
                            'fact_ids': fids, 'record_ids': rids,
                            'reason': _text(item['reason'], 'item reason')})
                    parsed.sort(key=lambda i: next(n for n, x in enumerate(contract['items']) if x['item_id'] == i['item_id']))
                    decisions[index] = derive_assessment(contract, parsed, state['need_contract_hash'])
                if set(decisions) != set(selected):
                    raise ValueError('need batch omitted a target')
            else:
                trace['skipped'] = 'no_accepted_facts'
                decisions = {i: unknown_assessment(contracts[i], state['need_contract_hash'],
                    'No accepted original evidence for this requirement yet.') for i in selected}
            previous = {a['need_index']: deepcopy(a) for a in state.get('need_assessments', [])}
            previous.update(decisions)
            assessments = []
            for contract in contracts:
                item = previous.get(contract['need_index'])
                if item is None or not valid_assessment(contract, item, state):
                    item = unknown_assessment(contract, state['need_contract_hash'],
                        'No current assessment bound to the contract and accepted evidence.')
                assessments.append(item)
            trace['need_assessments'] = [deepcopy(decisions[i]) for i in selected]
            return {'need_assessments': assessments,
                    'covered_needs': [a['need_index'] for a in assessments if a['status'] == 'supported'],
                    'resolved_needs': [a['need_index'] for a in assessments if a['status'] in {'supported', 'refuted'}]}, []
        return self._run('need', work)
