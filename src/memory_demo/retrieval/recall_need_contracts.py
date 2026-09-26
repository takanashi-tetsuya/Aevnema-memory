"""Question-bound requirement contracts and deterministic completion gates.

Anchors prove origin, not semantic adequacy. Item entailment remains a model
judgment; this module prevents omitted or unbound judgments from closing gaps.
"""
from copy import deepcopy

from memory_demo.retrieval.recall_review_v3 import _hash, _list, _object, _text

CONTRACT_PROTOCOL = {'version': 1, 'name': 'pre-evidence-need-items',
                     'max_items_per_need': 6, 'evidence_scope': 'accepted_fact_records'}
ANSWER_TYPES = {'identity', 'action', 'reason', 'time', 'place', 'relation',
                'description', 'quantity', 'boolean', 'other'}


def contract_input(state):
    return {'request_context': state['question'], 'context': state.get('context', ''),
            'targets': [{'need_index': i, 'question': n} for i, n in enumerate(state['needs'])]}


def validate_contracts(state, rows):
    needs = state['needs']
    if not needs:
        raise ValueError('contracts require planned needs')
    by_index = {}
    for row in _list(rows, 'contracts'):
        _object(row, {'need_index', 'answer_type', 'items'})
        index = row['need_index']
        if type(index) is not int or not 0 <= index < len(needs) or index in by_index:
            raise ValueError('unknown or duplicate contract need_index')
        if not isinstance(row['answer_type'], str) or row['answer_type'] not in ANSWER_TYPES:
            raise ValueError('unknown requested answer type')
        items = _list(row['items'], 'contract items')
        if not 1 <= len(items) <= CONTRACT_PROTOCOL['max_items_per_need']:
            raise ValueError('contract item count exceeds policy')
        parsed = []
        for ordinal, item in enumerate(items, 1):
            _object(item, {'kind', 'description', 'anchor'})
            if not isinstance(item['kind'], str) or item['kind'] not in {'answer', 'constraint', 'premise'}:
                raise ValueError('unknown contract item kind')
            anchor = item['anchor']
            _object(anchor, {'field', 'quote'})
            if not isinstance(anchor['field'], str) or anchor['field'] not in {'question', 'context'}:
                raise ValueError('contract anchor must refer to original request')
            quote = _text(anchor['quote'], 'contract anchor')
            text = state.get(anchor['field'], '')
            if quote not in text:
                raise ValueError('contract anchor is not literal original request text')
            parsed.append({'item_id': f'C{ordinal}', 'kind': item['kind'],
                'description': _text(item['description'], 'contract description'),
                'anchor': {'field': anchor['field'], 'quote': quote}})
        if sum(item['kind'] == 'answer' for item in parsed) != 1:
            raise ValueError('each need requires exactly one requested answer item')
        by_index[index] = {'need_index': index, 'question': needs[index],
                           'answer_type': row['answer_type'], 'items': parsed}
    if set(by_index) != set(range(len(needs))):
        raise ValueError('contracts must cover every planned need exactly')
    contracts = [by_index[i] for i in range(len(needs))]
    digest = _hash({'protocol': CONTRACT_PROTOCOL, 'input': contract_input(state), 'contracts': contracts})
    return {'need_contract_protocol': deepcopy(CONTRACT_PROTOCOL),
            'need_contracts': contracts, 'need_contract_hash': digest}


def bound_contracts(state):
    if state.get('need_contract_protocol') != CONTRACT_PROTOCOL:
        raise ValueError('need contract protocol missing or changed')
    contracts = state.get('need_contracts')
    rows = []
    for contract in _list(contracts, 'need_contracts'):
        rows.append({'need_index': contract['need_index'], 'answer_type': contract['answer_type'],
            'items': [{k: item[k] for k in ('kind', 'description', 'anchor')}
                      for item in contract['items']]})
    expected = validate_contracts(state, rows)
    if contracts != expected['need_contracts'] or state.get('need_contract_hash') != expected['need_contract_hash']:
        raise ValueError('need contracts or original question binding changed')
    return deepcopy(contracts)


def derive_assessment(contract, items, contract_hash):
    by_id = {item['item_id']: item for item in items}
    expected = {item['item_id'] for item in contract['items']}
    if len(by_id) != len(items) or set(by_id) != expected:
        raise ValueError('item assessments must cover the immutable contract exactly')
    answer = next(by_id[i['item_id']] for i in contract['items'] if i['kind'] == 'answer')
    refutations = [by_id[i['item_id']] for i in contract['items']
                   if i['kind'] == 'premise' and by_id[i['item_id']]['status'] == 'contradicted']
    if refutations and answer['status'] == 'supported':
        raise ValueError('premise refutation conflicts with normal supported answer')
    if refutations:
        status, value = 'refuted', '\n'.join(i['value'] for i in refutations)
    elif all(i['status'] == 'supported' for i in items):
        status, value = 'supported', answer['value']
    else:
        status = 'partial' if any(i['status'] != 'unknown' for i in items) else 'unknown'
        value = answer['value']
    return {'need_index': contract['need_index'], 'status': status,
            'fact_ids': list(dict.fromkeys(fid for i in items for fid in i['fact_ids'])),
            'answer': value, 'reason': '\n'.join(f"{i['item_id']}: {i['reason']}" for i in items),
            'contract_hash': contract_hash, 'item_assessments': deepcopy(items)}


def unknown_assessment(contract, contract_hash, reason):
    return derive_assessment(contract, [{'item_id': i['item_id'], 'status': 'unknown',
        'value': '', 'fact_ids': [], 'record_ids': [], 'reason': reason}
        for i in contract['items']], contract_hash)


def valid_assessment(contract, assessment, state):
    """Recheck references after evidence withdrawal or checkpoint resumption."""
    try:
        if assessment['contract_hash'] != state['need_contract_hash']:
            return False
        facts = {f['fact_id']: f for f in state.get('facts', [])}
        items = assessment['item_assessments']
        for item in items:
            if item['status'] not in {'supported', 'unknown', 'contradicted'}:
                return False
            fids, rids = item['fact_ids'], item['record_ids']
            if len(set(fids)) != len(fids) or len(set(rids)) != len(rids):
                return False
            if any(fid not in facts for fid in fids):
                return False
            allowed = {rid for fid in fids for rid in facts[fid]['record_ids']}
            if not set(rids) <= allowed or any(not set(rids).intersection(facts[fid]['record_ids']) for fid in fids):
                return False
            if item['status'] != 'unknown' and (not fids or not rids or not item['value'].strip()):
                return False
        return derive_assessment(contract, items, state['need_contract_hash']) == assessment
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def contract_gaps(state):
    if not state.get('needs'):
        return [{'need_index': None, 'item_id': None, 'status': 'unplanned'}]
    if not state.get('need_contracts'):
        return [{'need_index': i, 'item_id': None, 'status': 'contract_unavailable'}
                for i in range(len(state['needs']))]
    if state.get('contract_work', {}).get('status') != 'committed':
        return [{'need_index': i, 'item_id': None, 'status': 'contract_not_committed'}
                for i in range(len(state['needs']))]
    try:
        contracts = bound_contracts(state)
    except (ValueError, KeyError, TypeError):
        return [{'need_index': i, 'item_id': None, 'status': 'invalid_contract'}
                for i in range(len(state['needs']))]
    assessments = {a['need_index']: a for a in state.get('need_assessments', [])}
    gaps = []
    for contract in contracts:
        a = assessments.get(contract['need_index'])
        if a is None or not valid_assessment(contract, a, state):
            a = unknown_assessment(contract, state['need_contract_hash'], 'No current bound assessment.')
        if a['status'] in {'supported', 'refuted'}:
            continue
        states = {i['item_id']: i['status'] for i in a['item_assessments']}
        for item in contract['items']:
            if states[item['item_id']] != 'supported':
                gaps.append({'need_index': contract['need_index'], **deepcopy(item),
                             'status': states[item['item_id']]})
    return gaps
