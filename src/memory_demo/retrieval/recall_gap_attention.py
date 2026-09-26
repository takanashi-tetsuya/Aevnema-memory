"""Current committed NEED gaps as search attention, never evidence or cues.

Pure functions only: the caller binds the input checkpoint to its receipt,
embeds exactly one selected text, and owns HTTP/deadline/commit accounting.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from memory_demo.associations.spreading import SpreadingSeed
from memory_demo.embeddings.codec import normalize_embedding
from memory_demo.retrieval.recall_cues import _canonical_seeds, _sha, restart_spreading_epoch
from memory_demo.retrieval.recall_map_feedback import capture_feedback
from memory_demo.retrieval.recall_need_contracts import bound_contracts, derive_assessment, valid_assessment
from memory_demo.retrieval.recall_review_v3 import _hash, _indices, _list, _object, _refs, _text


GAP_ATTENTION_PROTOCOL = {
    'version': 1, 'name': 'current-committed-need-single-gap-attention',
    'scope': 'search_attention_not_evidence_or_cue',
    'selection_order': 'need-index-then-contract-item-order',
    'eligible_item_status': 'unknown', 'eligible_need_status': ['unknown', 'partial'],
    'max_input_chars': 2048, 'oversize': 'reject-without-truncation-or-next-item',
    'embedding_slots_per_arm': 1, 'source_roots': 'original-source-text-sha256',
}


def _accepted_view(state):
    """Bind current fact contents to their last committed semantic decisions."""
    latest = {}
    for trace in state.get('review_trace', []):
        if trace.get('stage') == 'facts' and trace.get('committed') is True and trace.get('schema_ok') is True:
            if not isinstance(trace.get('response'), dict) or trace.get('skipped'):
                raise ValueError('accepted evidence requires a real committed FACTS response')
            for transition in trace.get('transitions', []):
                latest[transition['fact_id']] = transition
    registry = state.get('record_registry', {})
    records = {r['record_id']: r for r in registry.values()}
    if len(records) != len(registry):
        raise ValueError('duplicate current record identity')
    facts, direct = {}, {}
    for fact in _list(state.get('facts', []), 'accepted facts'):
        fid = fact['fact_id']
        decision = latest.get(fid, {})
        if (fid in facts or fact.get('verification_status') != 'accepted'
                or fact.get('episode_alignment') != 'supported'
                or decision.get('to') != 'accept'
                or any(decision.get(k) != fact.get(k) for k in ('record_ids', 'statement', 'evidence'))):
            raise ValueError('current accepted evidence differs from committed FACTS')
        rids = _refs(fact['record_ids'], records, 'accepted record_ids', minimum=1)
        if len(rids) != len(fact['evidence']):
            raise ValueError('accepted evidence record count changed')
        for rid, evidence in zip(rids, fact['evidence']):
            record = records[rid]
            expected = {'episode_id': fact['episode_id'], 'source_id': record['source_id'],
                'source_sha256': record['source_sha256'], 'start': record['start'],
                'end': record['end'], 'quote': record['text']}
            if evidence != expected or fact['source_id'] != record['source_id']:
                raise ValueError('accepted evidence direct record binding changed')
            direct[rid] = deepcopy(record)
        facts[fid] = deepcopy(fact)
    return {'facts': [facts[k] for k in sorted(facts)],
            'records': [direct[k] for k in sorted(direct)]}


def _replay_need_origin(state, trace, contracts):
    """Reparse saved model output locally, retaining V9's direct-reference rule."""
    indices = _indices(trace['need_indices'], len(contracts))
    if not 1 <= len(indices) <= 4:
        raise ValueError('invalid committed NEED target count')
    request = trace['request']
    if (request.get('request_context') != state['question']
            or request.get('context') != state.get('context', '')
            or request.get('targets') != [contracts[i] for i in indices]):
        raise ValueError('committed NEED original request or targets changed')
    accepted = {f['fact_id']: f for f in state['facts']}
    fact_short, record_short = trace['fact_ids'], trace['record_ids']
    if (len(set(fact_short.values())) != len(fact_short)
            or set(fact_short.values()) != set(accepted)
            or len(set(record_short.values())) != len(record_short)):
        raise ValueError('committed NEED accepted evidence catalog changed')
    registry = {r['record_id']: r for r in state['record_registry'].values()}
    request_records = _list(request['records'], 'request records')
    seen_records = set()
    for row in request_records:
        alias = row.get('id')
        if not isinstance(alias, str) or alias not in record_short or alias in seen_records:
            raise ValueError('unknown or duplicate NEED request record alias')
        seen_records.add(alias)
        record = registry.get(record_short[alias])
        if record is None:
            raise ValueError('NEED request record is no longer in current registry')
        expected = {'id': alias, 'source_id': record['source_id'],
            'episode_ids': list(record['episode_ids']), 'record_index': record['record_index'],
            'speaker': record['speaker_raw'], 'aliases': deepcopy(record['speaker_aliases']),
            'language': record['language'], 'text': record['text']}
        if row != expected:
            raise ValueError('NEED request record literal or speaker binding changed')
    if seen_records != set(record_short):
        raise ValueError('NEED request record catalog is incomplete')
    request_facts = request['facts']
    if len(request_facts) != len(fact_short):
        raise ValueError('committed NEED fact request changed')
    seen = set()
    for fact in request_facts:
        alias = fact['fact_id']
        if alias not in fact_short or alias in seen:
            raise ValueError('unknown or duplicate request fact alias')
        seen.add(alias)
        current = accepted[fact_short[alias]]
        rids = [record_short[a] for a in _refs(fact['record_ids'], record_short, 'request records', minimum=1)]
        if (rids != current['record_ids'] or any(fact.get(k) != current.get(k, '')
                for k in ('episode_id', 'statement', 'interpretation', 'need_indices'))):
            raise ValueError('committed NEED no longer describes current accepted evidence')
    response = trace['response']
    _object(response, {'assessments'})
    parsed = {}
    for row in _list(response['assessments'], 'assessments'):
        _object(row, {'need_index', 'items'})
        index = row['need_index']
        if type(index) is not int or index not in indices or index in parsed:
            raise ValueError('unknown or duplicate committed NEED target')
        contract = contracts[index]
        order = {i['item_id']: n for n, i in enumerate(contract['items'])}
        items, seen_items = [], set()
        for item in _list(row['items'], 'items'):
            _object(item, {'item_id', 'status', 'value', 'fact_ids', 'record_ids', 'reason'})
            iid, status = item['item_id'], item['status']
            if not isinstance(iid, str) or iid not in order or iid in seen_items:
                raise ValueError('unknown or duplicate committed NEED item')
            if not isinstance(status, str) or status not in {'supported', 'unknown', 'contradicted'}:
                raise ValueError('invalid committed NEED item status')
            seen_items.add(iid)
            minimum = int(status != 'unknown')
            fids = [fact_short[a] for a in _refs(item['fact_ids'], fact_short, 'fact_ids', minimum=minimum)]
            rids = [record_short[a] for a in _refs(item['record_ids'], record_short, 'record_ids', minimum=minimum)]
            items.append({'item_id': iid, 'status': status,
                'value': _text(item['value'], 'item value', empty=not minimum),
                'fact_ids': fids, 'record_ids': rids, 'reason': _text(item['reason'], 'item reason')})
        assessment = derive_assessment(contract, sorted(items, key=lambda i: order[i['item_id']]), state['need_contract_hash'])
        if not valid_assessment(contract, assessment, state):
            raise ValueError('committed NEED item references are no longer valid')
        parsed[index] = assessment
    if set(parsed) != set(indices) or [parsed[i] for i in indices] != trace['need_assessments']:
        raise ValueError('raw NEED response differs from committed parsed assessments')
    return parsed


def select_gap_attention(state):
    """Return one current model gap, or None; invalid/oversize bindings fail shut.

    A FACTS commit invalidates all prior NEED decisions even when it accepts the
    same fact again. Completed-wave counters are not an origin boundary: an
    unfinished wave may already contain a usable committed NEED decision.
    """
    if not state.get('needs') or state.get('contract_work', {}).get('status') != 'committed':
        return None
    contracts = bound_contracts(state)
    accepted = _accepted_view(state)
    traces = state.get('review_trace', [])
    wave = max([0, *(t['wave'] for t in traces if type(t.get('wave')) is int)]) + 1
    # Reuse the existing reset/origin gate, but capture the current state each
    # time; never reuse V11's historical wave-entry snapshot after FACTS.
    snapshot = capture_feedback(state, wave)
    candidates = [i for i in snapshot['payload']['items'] if i['previous_status'] == 'unknown']
    if not candidates:
        return None
    candidate = candidates[0]
    origin = next(o for o in snapshot['origins'] if o['need_index'] == candidate['need_index'])
    trace = traces[origin['trace_index']]
    parsed = _replay_need_origin(state, trace, contracts)
    assessment = parsed[candidate['need_index']]
    item = next(i for i in assessment['item_assessments'] if i['item_id'] == candidate['item_id'])
    requirement = next(i for i in contracts[candidate['need_index']]['items'] if i['item_id'] == candidate['item_id'])
    selected = {'need_index': candidate['need_index'], 'item_id': candidate['item_id'],
        'answer_type': candidate['answer_type'], 'kind': requirement['kind'],
        'criterion': requirement['description'], 'request_anchor': deepcopy(requirement['anchor']),
        'status': item['status'], 'current_value': item['value'], 'missing_reason': item['reason']}
    control = requirement['anchor']['quote']
    treatment = (control + '\n检索需求：' + selected['criterion']
        + '\n所求类型：' + selected['answer_type'] + '；约束类型：' + selected['kind']
        + '\n当前判定：' + selected['status'] + '\n当前值：' + selected['current_value']
        + '\n缺口诊断（未验证）：' + selected['missing_reason'])
    if max(len(control), len(treatment)) > GAP_ATTENTION_PROTOCOL['max_input_chars']:
        raise ValueError('gap attention input exceeds 2048 characters; no truncation or alternative selection')
    result = {'protocol': deepcopy(GAP_ATTENTION_PROTOCOL), 'selected': selected,
        'control_text': control, 'treatment_text': treatment,
        'provenance': {'need_contract_hash': state['need_contract_hash'],
            'anchor_binding': deepcopy(snapshot['anchor_binding']),
            'origin_trace_index': origin['trace_index'], 'origin_wave': trace['wave'],
            'review_boundary_wave': wave, 'trace_prefix_length': len(traces),
            'semantic_trace_prefix_sha256': snapshot['trace_prefix_sha256'],
            'raw_need_response_sha256': _hash(trace['response']),
            'need_request_sha256': _hash(trace['request']),
            'assessment_sha256': _hash(assessment), 'accepted_evidence_sha256': _hash(accepted)}}
    result['selection_sha256'] = _hash(result)
    return result


def validate_gap_attention(state, selection):
    if not isinstance(selection, dict) or selection != select_gap_attention(state):
        raise ValueError('gap attention differs from the current committed NEED origin')
    return deepcopy(selection)


def expand_attention_vector(state, selection, vector, *, arm, episode_ids,
                            episode_matrix, episodes, sources, seeds_per_cue):
    """Apply one externally embedded attention slot without creating a cue.

    Ranking, max merging, Source roots and graph-epoch restart match the
    existing follow-up expansion. No embedding/HTTP counter is claimed here.
    """
    selection = validate_gap_attention(state, selection)
    if arm not in {'control', 'gap'}:
        raise ValueError('attention arm must be control or gap')
    if type(seeds_per_cue) is not int or seeds_per_cue <= 0:
        raise ValueError('seeds_per_cue must be a positive integer')
    ids = list(episode_ids)
    if len(set(ids)) != len(ids) or any(type(eid) is not int or eid not in episodes for eid in ids):
        raise ValueError('episode matrix IDs must be unique existing episodes')
    matrix = np.asarray(episode_matrix, dtype=np.float32)
    if (matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()):
        raise ValueError('invalid episode embedding matrix')
    if len(ids) and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError('episode matrix must already be normalized')
    prior = _canonical_seeds(state.get('seeds', []))
    for eid in set(ids) | {s.node_id for s in prior if s.node_type == 'episode'}:
        if eid not in episodes or not isinstance(sources.get(episodes[eid]['source_id']), str):
            raise ValueError('attention seed Source is unavailable')
    for fact in state.get('facts', []):
        for evidence in fact['evidence']:
            sid, eid = evidence['source_id'], evidence['episode_id']
            raw = sources.get(sid)
            if (not isinstance(raw, str) or eid not in episodes or episodes[eid]['source_id'] != sid
                    or _sha(raw) != evidence['source_sha256']
                    or raw[evidence['start']:evidence['end']] != evidence['quote']):
                raise ValueError('attention accepted Source evidence changed')
    scores = matrix @ normalize_embedding(vector, matrix.shape[1])
    ranking = sorted(range(len(ids)), key=lambda p: (-float(scores[p]), ids[p]))
    positive = np.clip(scores, 0.0, 1.0)
    additions = [SpreadingSeed('episode', ids[p], float(positive[p]),
                 'source:' + _sha(sources[episodes[ids[p]]['source_id']]))
                 for p in ranking[:seeds_per_cue] if positive[p] > 0]
    source_prior = [SpreadingSeed(s.node_type, s.node_id, s.activation,
                    'source:' + _sha(sources[episodes[s.node_id]['source_id']]))
                    if s.node_type == 'episode' else s for s in prior]
    merged = _canonical_seeds([*source_prior, *additions])
    old_scores = state.get('fallback_episode_scores', {})
    if set(old_scores) - {str(eid) for eid in ids}:
        raise ValueError('attention matrix omits existing fallback episodes')
    for score in old_scores.values():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid existing fallback activation')
    combined = {str(eid): max(float(old_scores.get(str(eid), 0.0)), float(positive[p]))
                for p, eid in enumerate(ids)}
    detached = deepcopy(state)
    updated = restart_spreading_epoch(detached, merged)
    updated['fallback_episode_scores'] = combined
    updated['fallback_episode_ids'] = sorted(ids, key=lambda eid: (-combined[str(eid)], eid))
    updated.setdefault('metrics', {})['seed_candidates_scored'] = state.get('metrics', {}).get('seed_candidates_scored', 0) + len(ids)
    before, after = [asdict(s) for s in prior], [asdict(s) for s in merged]
    prior_ids = {s.node_id for s in prior if s.node_type == 'episode'}
    changed = before != after
    query = selection['control_text' if arm == 'control' else 'treatment_text']
    diagnostic = {'protocol': deepcopy(GAP_ATTENTION_PROTOCOL), 'arm': arm,
        'query_sha256': _sha(query), 'query_chars': len(query),
        'selection_sha256': selection['selection_sha256'], 'embedding_slots': 1,
        'episode_rows_scored': len(ids), 'seed_before': before, 'seed_after': after,
        'seed_fingerprint_before': _hash(before), 'seed_fingerprint_after': _hash(after),
        'fingerprint_changed': changed,
        'new_seed_episode_ids': sorted({s.node_id for s in merged if s.node_type == 'episode'} - prior_ids),
        'source_roots': {str(s.node_id): s.root_id for s in merged if s.node_type == 'episode'},
        'restart': changed, 'spreading_expansion_base_before': state.get('spreading_expansion_base', 0),
        'spreading_expansion_base_after': updated.get('spreading_expansion_base', 0),
        'fallback_episode_ids_before': deepcopy(state.get('fallback_episode_ids', [])),
        'fallback_episode_ids_after': deepcopy(updated['fallback_episode_ids'])}
    return updated, diagnostic
