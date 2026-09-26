"""Locally assigned request fragment IDs; models do not reproduce source text."""
from copy import deepcopy
import re

from memory_demo.retrieval.recall_need_contracts import validate_contracts
from memory_demo.retrieval.recall_review_v3 import _hash, _list, _object

ANCHOR_PROTOCOL = {'version': 1, 'name': 'original-request-fragment-ids',
                   'segmentation': 'sentence-punctuation-preserve-v1'}


def request_fragments(state):
    records = []
    for field, prefix in [('question', 'Q'), ('context', 'X')]:
        text = state.get(field, '')
        ordinal = 0
        for match in re.finditer(r'.*?(?:[。！？!?；;\n]+|$)', text, flags=re.S):
            if not match.group().strip():
                continue
            ordinal += 1
            records.append({'anchor_id': f'{prefix}{ordinal}', 'field': field,
                'start': match.start(), 'end': match.end(), 'text': match.group()})
    return records


def anchor_input(state):
    return {'original_request_fragments': request_fragments(state),
            'targets': [{'need_index': i, 'planned_need': n} for i, n in enumerate(state['needs'])]}


def admit_anchor_contracts(state, response):
    _object(response, {'contracts'})
    catalog = {r['anchor_id']: r for r in request_fragments(state)}
    rows = []
    for contract in _list(response['contracts'], 'contracts'):
        _object(contract, {'need_index', 'answer_type', 'items'})
        items = []
        for item in _list(contract['items'], 'items'):
            _object(item, {'kind', 'description', 'anchor_id'})
            key = item['anchor_id']
            if not isinstance(key, str) or key not in catalog:
                raise ValueError('contract anchor must select an original request fragment ID')
            record = catalog[key]
            items.append({'kind': item['kind'], 'description': item['description'],
                          'anchor': {'field': record['field'], 'quote': record['text'].strip()}})
        rows.append({'need_index': contract['need_index'], 'answer_type': contract['answer_type'], 'items': items})
    updates = validate_contracts(state, rows)
    return updates, {'contracts': rows}


def anchor_binding(state):
    return {'protocol': deepcopy(ANCHOR_PROTOCOL), 'request_fragments_sha256': _hash(request_fragments(state))}
