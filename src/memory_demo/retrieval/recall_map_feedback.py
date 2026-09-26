"""Wave-bound historical NEED diagnostics for MAP attention, never evidence."""
from copy import deepcopy

from memory_demo.retrieval.recall_contract_anchors import anchor_binding
from memory_demo.retrieval.recall_need_contracts import bound_contracts, valid_assessment
from memory_demo.retrieval.recall_review_v3 import _hash

FEEDBACK_PROTOCOL = {'version': 1, 'name': 'committed-need-gap-to-map',
    'scope': 'wave-entry-historical-attention-only', 'truncation': 'none-shared-input-budget'}


def _view(traces):
    """Only semantic fields; exported token redaction must not alter the binding."""
    result = []
    for index, trace in enumerate(traces):
        if trace.get('stage') not in {'facts', 'need'}:
            continue
        result.append({'index': index, 'stage': trace['stage'], 'wave': trace.get('wave'),
            'committed': trace.get('committed') is True, 'schema_ok': trace.get('schema_ok') is True,
            'has_response': isinstance(trace.get('response'), dict),
            'need_response': deepcopy(trace.get('response')) if trace['stage'] == 'need' else None,
            'skipped': trace.get('skipped'), 'need_contract_hash': trace.get('need_contract_hash'),
            'need_assessments': deepcopy(trace.get('need_assessments', [])),
            'transitions': [{k: deepcopy(t.get(k)) for k in ('fact_id', 'to', 'record_ids')}
                            for t in trace.get('transitions', [])]})
    return result


def _derive(state, traces, wave):
    contracts = bound_contracts(state)
    facts, latest = {}, {}
    for trace in _view(traces):
        if not trace['committed'] or not trace['schema_ok']:
            continue
        if trace['stage'] == 'facts':
            latest.clear()
            for transition in trace['transitions']:
                fid = transition['fact_id']
                if transition['to'] == 'accept':
                    facts[fid] = {'fact_id': fid, 'record_ids': transition['record_ids']}
                else:
                    facts.pop(fid, None)
        elif (trace['has_response'] and not trace['skipped']
              and trace['need_contract_hash'] == state['need_contract_hash']
              and type(trace['wave']) is int and trace['wave'] < wave):
            for assessment in trace['need_assessments']:
                latest[assessment['need_index']] = (trace['index'], trace['wave'], assessment)
    virtual = {'need_contract_hash': state['need_contract_hash'], 'facts': list(facts.values())}
    items, origins, eligible = [], [], {}
    for contract in contracts:
        index = contract['need_index']
        if index not in latest:
            continue
        trace_index, prior_wave, assessment = latest[index]
        if not valid_assessment(contract, assessment, virtual):
            continue
        eligible[index] = deepcopy(assessment)
        if assessment['status'] in {'supported', 'refuted'}:
            continue
        origins.append({'need_index': index, 'trace_index': trace_index,
                        'assessment_sha256': _hash(assessment)})
        by_id = {item['item_id']: item for item in assessment['item_assessments']}
        for requirement in contract['items']:
            prior = by_id[requirement['item_id']]
            if prior['status'] == 'supported':
                continue
            items.append({'need_index': index, 'item_id': requirement['item_id'],
                'answer_type': contract['answer_type'], 'kind': requirement['kind'],
                'description': requirement['description'], 'request_anchor': deepcopy(requirement['anchor']),
                'prior_review_wave': prior_wave, 'previous_status': prior['status'],
                'previous_value': prior['value'], 'previous_reason': prior['reason']})
    payload = {'scope': 'historical_search_attention_not_evidence', 'items': items}
    return payload, origins, eligible, facts


def capture_feedback(state, wave):
    if type(wave) is not int or wave < 1:
        raise ValueError('feedback wave must be a positive integer')
    traces = state.get('review_trace', [])
    payload, origins, eligible, facts = _derive(state, traces, wave)
    actual_facts = {f['fact_id']: {'fact_id': f['fact_id'], 'record_ids': f['record_ids']}
                    for f in state.get('facts', [])}
    if actual_facts != facts:
        raise ValueError('wave-entry accepted records do not match committed FACTS')
    actual = {a['need_index']: a for a in state.get('need_assessments', [])}
    if any(actual.get(index) != assessment for index, assessment in eligible.items()):
        raise ValueError('wave-entry assessment does not match its committed NEED origin')
    if state.get('contract_anchor_binding') != anchor_binding(state):
        raise ValueError('feedback original request binding changed')
    snapshot = {'protocol': deepcopy(FEEDBACK_PROTOCOL), 'wave': wave,
        'need_contract_hash': state['need_contract_hash'], 'anchor_binding': anchor_binding(state),
        'trace_prefix_length': len(traces), 'trace_prefix_sha256': _hash(_view(traces)),
        'origins': origins, 'payload': payload}
    snapshot['snapshot_sha256'] = _hash(snapshot)
    return snapshot


def validate_feedback(state, snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError('missing wave feedback snapshot')
    body = {k: deepcopy(v) for k, v in snapshot.items() if k != 'snapshot_sha256'}
    if snapshot.get('snapshot_sha256') != _hash(body) or body.get('protocol') != FEEDBACK_PROTOCOL:
        raise ValueError('feedback snapshot or protocol changed')
    length, wave = body.get('trace_prefix_length'), body.get('wave')
    traces = state.get('review_trace', [])
    if (type(length) is not int or not 0 <= length <= len(traces)
            or type(wave) is not int or wave < 1):
        raise ValueError('invalid feedback origin boundary')
    prefix = traces[:length]
    payload, origins, _, _ = _derive(state, prefix, wave)
    if (body.get('need_contract_hash') != state['need_contract_hash']
            or body.get('anchor_binding') != anchor_binding(state)
            or state.get('contract_anchor_binding') != anchor_binding(state)
            or body.get('trace_prefix_sha256') != _hash(_view(prefix))
            or body.get('payload') != payload or body.get('origins') != origins):
        raise ValueError('feedback differs from its committed historical origin')
    return deepcopy(payload)


def active_feedback(state):
    wave = state['metrics']['review_waves'] + 1
    snapshots = state.get('map_feedback_snapshots', [])
    if not snapshots or snapshots[-1].get('wave') != wave:
        raise ValueError('MAP requires the current wave entry feedback')
    validate_feedback(state, snapshots[-1])
    return deepcopy(snapshots[-1])
