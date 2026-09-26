"""Independent, atomic record-based map/fact/need/link review stages.

This opt-in protocol has no provider, database, or domain-specific logic.  Each
successful stage returns a state difference that its caller may commit before
attempting another stage.  A failed need review cannot erase verified facts.
Episode alignment is an explicit independent model verdict, not a deterministic
record-to-Episode mapping inferred from sharing a Source.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json
from typing import Callable

from memory_demo.retrieval.recall_records import project_source, resolve_record, RecordReferenceError
from memory_demo.retrieval.recall_review import stable_fact_id

_path = Path(__file__).resolve().parents[3] / 'config' / 'prompt_config' / 'recall_v3_prompts.py'
_spec = spec_from_file_location('memory_demo._record_review_v3_prompts', _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f'cannot load record review prompts: {_path}')
_catalog = module_from_spec(_spec)
_spec.loader.exec_module(_catalog)
MAP_SYSTEM, FACTS_SYSTEM, NEED_SYSTEM, LINKS_SYSTEM = (_catalog.MAP, _catalog.FACTS, _catalog.NEED, _catalog.LINKS)
_EVIDENCE = ('episode_id', 'source_id', 'source_sha256', 'start', 'end', 'quote')


class ReviewStageError(ValueError):
    def __init__(self, message: str, *, trace: dict | None = None):
        super().__init__(message)
        self.trace = deepcopy(trace or {})


@dataclass(frozen=True)
class ReviewStageResult:
    updates: dict
    trace: dict
    followup_cues: list = field(default_factory=list)


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f'{name} must be nonempty text')
    return value.strip()


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required).issubset(value) or set(value)-set(required)-set(optional):
        raise ValueError('response has missing or unexpected fields')
    return value


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    return value


def _refs(value, allowed, name, *, minimum=0, maximum=None):
    refs = _list(value, name)
    if len(refs) < minimum or (maximum is not None and len(refs) > maximum):
        raise ValueError(f'{name} has invalid length')
    if any(not isinstance(v, str) or v not in allowed for v in refs) or len(refs) != len(set(refs)):
        raise ValueError(f'{name} contains unknown or duplicate IDs')
    return list(refs)


def _indices(value, size):
    values = _list(value, 'need_indices')
    if any(type(v) is not int or not 0 <= v < size for v in values) or len(set(values)) != len(values):
        raise ValueError('invalid need_indices')
    return sorted(values)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _link(a, b, rationale):
    a, b = sorted((a, b))
    return {'link_id': 'l_'+_hash([a, b]), 'from_episode_id': a, 'to_episode_id': b, 'rationale': rationale}


def _primary_claim(records):
    parts = []
    for r in records:
        speaker = r.get('speaker_raw')
        if speaker:
            alias = r.get('speaker_aliases', {}).get(r['language'])
            label = f'{alias}（{speaker}）' if alias and alias != speaker else speaker
            parts.append(f'{label}：{r["text"]}')
        else:
            parts.append(f'原文记录：{r["text"]}')
    return '\n'.join(parts)


class RecordReview:
    """Sources/Episodes are caller-owned snapshots, read again at each stage."""
    # None removes only the per-fact count ceiling; literal record validation
    # and the complete-context character budget still apply at every stage.
    max_fact_records: int | None = 3

    def __init__(self, sources: dict[int, str], episodes: dict[int, dict], call: Callable):
        self.sources, self.episodes, self.call = sources, episodes, call

    def _run(self, stage, work):
        trace = {'version': 3, 'stage': stage, 'schema_ok': False, 'transitions': []}
        try:
            updates, cues = work(trace)
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, ReviewStageError):
                raise
            trace['error_type'], trace['error'] = type(exc).__name__, str(exc)
            raise ReviewStageError(str(exc), trace=trace) from exc
        trace['schema_ok'] = True
        return ReviewStageResult(deepcopy(updates), deepcopy(trace), deepcopy(cues))

    def _call(self, system, payload, trace):
        # The caller can archive the request's short-ID mapping before calling
        # a provider; successful responses are retained even if schema fails.
        trace['request'] = deepcopy(payload)
        trace['input_chars'] = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        if trace['input_chars'] > trace.get('input_char_budget', 48000):
            raise ValueError('review context budget exceeded; keep pending evidence for a complete context slice')
        response = self.call(system, deepcopy(payload))
        trace['response'] = deepcopy(response)
        return response

    def _needs(self, state):
        needs = _list(state.get('needs'), 'needs')
        if not needs or any(not isinstance(n, str) or not n.strip() for n in needs):
            raise ValueError('needs must contain nonempty questions')
        return list(needs)

    def _registry(self, state, records=()):
        stored = state.get('record_registry', {})
        if not isinstance(stored, dict):
            raise ValueError('record_registry must be an object')
        registry = {}
        for supplied in [*stored.values(), *records]:
            if not isinstance(supplied, dict):
                raise ValueError('record must be a locally projected object')
            sid = supplied.get('source_id')
            if type(sid) is not int or sid not in self.sources:
                raise ValueError('unknown record Source')
            eids = supplied.get('episode_ids')
            if not isinstance(eids, list) or not eids:
                raise ValueError('record has no Episode binding')
            for eid in eids:
                if type(eid) is not int or eid not in self.episodes or self.episodes[eid].get('source_id') != sid:
                    raise ValueError('record Source/Episode binding changed')
                resolve_record(supplied['record_id'], [supplied], episode_id=eid, raw=self.sources[sid])
            # Rebuild presentation metadata locally too: forged speaker aliases
            # must not survive merely because the quotation span is correct.
            canonical = project_source(sid, self.sources[sid], eids,
                                       start=supplied['context_start'], end=supplied['context_end'],
                                       language=supplied['language'])
            matches = [r for r in canonical if r['record_id'] == supplied['record_id']]
            if len(matches) != 1:
                raise ValueError('record is not a complete original record')
            record = matches[0]
            if any(record[k] != supplied[k] for k in ('text', 'start', 'end', 'speaker_raw', 'speaker_aliases')):
                raise ValueError('record presentation changed')
            rid = record['record_id']
            if rid in registry:
                old = registry[rid]
                if any(old[k] != record[k] for k in ('text', 'start', 'end', 'language')):
                    raise ValueError('ambiguous record language projection')
                record['episode_ids'] = sorted(set(old['episode_ids']) | set(record['episode_ids']))
            registry[rid] = record
        return registry

    def _records_payload(self, registry):
        short = {f'R{i}': rid for i, rid in enumerate(registry, 1)}
        payload = []
        for alias, rid in short.items():
            r = registry[rid]
            payload.append({'id': alias, 'source_id': r['source_id'], 'episode_ids': list(r['episode_ids']),
                            'record_index': r['record_index'], 'speaker': r['speaker_raw'],
                            'aliases': deepcopy(r['speaker_aliases']), 'language': r['language'], 'text': r['text']})
        return payload, short

    def _fact(self, record_ids, eid, interpretation, need_indices, registry):
        if type(eid) is not int or eid not in self.episodes:
            raise ValueError('unknown fact Episode')
        if (not record_ids or len(record_ids) != len(set(record_ids))
                or (self.max_fact_records is not None and len(record_ids) > self.max_fact_records)):
            raise ValueError('fact requires distinct records within the configured count limit')
        records = [registry[rid] for rid in record_ids]
        if len({r['source_id'] for r in records}) != 1:
            raise ValueError('fact records must belong to one Source')
        records.sort(key=lambda r: (r['context_start'], r['context_end']))
        evidence = []
        for r in records:
            resolved = resolve_record(r['record_id'], [r], episode_id=eid, raw=self.sources[r['source_id']])
            evidence.append({k: resolved[k] for k in _EVIDENCE})
        fact = {**evidence[0], 'claim': _primary_claim(records), 'record_ids': [r['record_id'] for r in records],
                'evidence': evidence, 'interpretation': interpretation, 'need_indices': sorted(need_indices),
                'context_start': min(r['context_start'] for r in records),
                'context_end': max(r['context_end'] for r in records)}
        fact['fact_id'] = stable_fact_id(fact)
        return fact

    def _facts(self, state, registry, key):
        result = {}
        for stored in _list(state.get(key, []), key):
            if key == 'facts' and stored.get('verification_status') != 'accepted':
                raise ValueError('unverified candidate cannot be treated as an accepted fact')
            if key == 'facts' and stored.get('episode_alignment') != 'supported':
                raise ValueError('accepted fact requires explicitly supported Episode alignment')
            need_indices = _indices(stored['need_indices'], len(self._needs(state)))
            fact = self._fact(stored['record_ids'], stored['episode_id'],
                              _text(stored.get('interpretation', ''), 'interpretation', empty=True), need_indices, registry)
            if any(stored[k] != fact[k] for k in (*_EVIDENCE, 'claim', 'evidence', 'fact_id')):
                raise ValueError('stored fact no longer matches its original records')
            if fact['fact_id'] in result:
                raise ValueError('duplicate stored fact')
            result[fact['fact_id']] = deepcopy(stored)
        return result

    def _common(self, state, registry, trace, *, source_ids=None):
        if source_ids is not None:
            current = set(state.get('current_record_ids', []))
            registry = {rid: r for rid, r in registry.items() if r['source_id'] in source_ids or rid in current}
        budget = state.get('review_context_max_chars', 48000)
        if type(budget) is not int or budget <= 0:
            raise ValueError('review_context_max_chars must be positive')
        trace['input_char_budget'] = budget
        records, short = self._records_payload(registry)
        trace['record_ids'] = deepcopy(short)
        return {'question': _text(state.get('question'), 'question'), 'needs': self._needs(state), 'records': records}, short

    def _fact_payload(self, facts, record_short, state):
        inverse = {v: k for k, v in record_short.items()}
        aliases = {f'F{i}': fid for i, fid in enumerate(facts, 1)}
        proposals = state.get('interpretation_proposals', {})
        payload = []
        for alias, fid in aliases.items():
            f = facts[fid]
            payload.append({'fact_id': alias, 'episode_id': f['episode_id'],
                            'record_ids': [inverse[rid] for rid in f['record_ids']],
                            'episode_hint': str(self.episodes[f['episode_id']].get('text', '')),
                            'interpretation': f.get('interpretation', ''), 'statement': f.get('statement', ''),
                            'interpretation_proposals': deepcopy(proposals.get(fid, [])),
                            'need_indices': list(f['need_indices'])})
        return payload, aliases

    def _decisions(self, response, key, id_key, short, *, statement=False):
        _object(response, {key})
        decisions = {}
        for item in _list(response[key], key):
            _object(item, {id_key, 'decision', 'reason', *(['statement', 'episode_alignment'] if statement else [])})
            alias = _text(item[id_key], id_key)
            if alias not in short or short[alias] in decisions:
                raise ValueError('unknown or duplicate verdict ID')
            if item['decision'] not in {'accept', 'reject', 'needs_context'}:
                raise ValueError('unknown evidence verdict')
            cleaned = {'decision': item['decision'], 'reason': _text(item['reason'], 'reason')}
            if statement:
                alignment = item['episode_alignment']
                if alignment not in {'supported', 'mismatch', 'unknown'}:
                    raise ValueError('unknown Episode alignment verdict')
                cleaned['episode_alignment'] = alignment
                cleaned['model_decision'] = item['decision']
                cleaned['statement'] = _text(item['statement'], 'statement', empty=item['decision'] != 'accept')
                if alignment == 'mismatch':
                    cleaned['decision'] = 'reject'
                elif alignment == 'unknown' and item['decision'] != 'reject':
                    cleaned['decision'] = 'needs_context'
                if cleaned['decision'] != item['decision']:
                    cleaned['local_adjustment'] = 'Episode alignment does not permit acceptance'
            decisions[short[alias]] = cleaned
        if set(decisions) != set(short.values()):
            raise ValueError('each candidate requires exactly one verdict')
        return decisions

    def map(self, records, state):
        """Commit candidates and original-record context without verifying them."""
        def work(trace):
            needs = self._needs(state)
            registry = self._registry(state, _list(records, 'records'))
            current = {r['record_id']: registry[r['record_id']] for r in records}
            accepted, pending = self._facts(state, registry, 'facts'), self._facts(state, registry, 'pending_facts')
            payload, short = self._common(state, current, trace)
            inverse = {rid: alias for alias, rid in short.items()}
            payload['existing_record_groups'] = [[inverse[rid] for rid in f['record_ids']] for f in [*accepted.values(), *pending.values()] if all(rid in inverse for rid in f['record_ids'])]
            eids = sorted({eid for r in current.values() for eid in r['episode_ids']})
            payload['episode_hints'] = [{'episode_id': eid, 'source_id': self.episodes[eid]['source_id'], 'text': str(self.episodes[eid].get('text', ''))} for eid in eids]
            if not current:
                trace['skipped'] = 'no_visible_records'
                return {}, []
            response = self._call(MAP_SYSTEM, payload, trace)
            _object(response, {'facts', 'links'}, {'followup_cues'})
            facts = _list(response['facts'], 'facts')
            if len(facts) > 8:
                raise ValueError('mapper may propose at most eight facts')
            proposals = deepcopy(state.get('interpretation_proposals', {}))
            for item in facts:
                _object(item, {'record_ids', 'episode_id', 'interpretation', 'need_indices'})
                aliases = _refs(item['record_ids'], short, 'record_ids', minimum=1, maximum=self.max_fact_records)
                interpretation = _text(item['interpretation'], 'interpretation')
                fact = self._fact([short[a] for a in aliases], item['episode_id'], interpretation,
                                  _indices(item['need_indices'], len(needs)), registry)
                fid = fact['fact_id']
                proposals[fid] = list(dict.fromkeys([*proposals.get(fid, []), interpretation]))
                if fid not in accepted:
                    if fid in pending:
                        fact['need_indices'] = sorted(set(fact['need_indices']) | set(pending[fid]['need_indices']))
                    fact['verification_status'] = 'pending'
                    pending[fid] = fact
                trace['transitions'].append({'fact_id': fid, 'record_ids': fact['record_ids'], 'to': 'already_accepted' if fid in accepted else 'pending'})
            links = {item['link_id']: deepcopy(item) for item in state.get('pending_links', [])}
            allowed_eids = set(eids) | {f['episode_id'] for f in accepted.values()} | {f['episode_id'] for f in pending.values()}
            for item in _list(response['links'], 'links'):
                _object(item, {'from_episode_id', 'to_episode_id', 'rationale'})
                a, b = item['from_episode_id'], item['to_episode_id']
                if type(a) is not int or type(b) is not int or a == b or not {a, b}.issubset(allowed_eids):
                    raise ValueError('invalid proposed link endpoints')
                link = _link(a, b, _text(item['rationale'], 'rationale'))
                links[link['link_id']] = link
            cues = []
            for item in _list(response.get('followup_cues', []), 'followup_cues'):
                _object(item, {'record_id', 'cue'})
                rid = _refs([item['record_id']], short, 'cue record_id', minimum=1)[0]
                record = registry[short[rid]]
                cue = _text(item['cue'], 'cue')
                if cue not in record['text']:
                    raise ValueError('cue is not a literal substring of its displayed record')
                cues.append({'cue': cue, 'source_id': record['source_id'],
                             'episode_id': record['episode_ids'][0], 'quote': record['text']})
            return {'record_registry': registry, 'current_record_ids': list(current), 'pending_facts': list(pending.values()), 'pending_links': list(links.values()),
                    'interpretation_proposals': proposals}, cues
        return self._run('map', work)

    def review_facts(self, state, *, fact_ids=None):
        """Review at most four pending or explicitly selected accepted facts."""
        def work(trace):
            registry = self._registry(state)
            accepted, pending = self._facts(state, registry, 'facts'), self._facts(state, registry, 'pending_facts')
            all_facts = {**accepted, **pending}
            selected = list(pending)[:4] if fact_ids is None else _refs(fact_ids, all_facts, 'fact_ids', maximum=4)
            if not selected:
                trace['skipped'] = 'no_fact_candidates'
                return {}, []
            candidates = {fid: all_facts[fid] for fid in selected}
            payload, record_short = self._common(state, registry, trace, source_ids={f['source_id'] for f in candidates.values()})
            payload['facts'], short = self._fact_payload(candidates, record_short, state)
            trace['fact_ids'] = deepcopy(short)
            decisions = self._decisions(self._call(FACTS_SYSTEM, payload, trace), 'fact_decisions', 'fact_id', short, statement=True)
            for fid, decision in decisions.items():
                before = 'accept' if fid in accepted else 'pending'
                accepted.pop(fid, None)
                pending.pop(fid, None)
                fact = deepcopy(candidates[fid])
                fact.update(statement=decision['statement'], review_reason=decision['reason'],
                            episode_alignment=decision['episode_alignment'])
                if decision['decision'] == 'accept':
                    fact['verification_status'] = 'accepted'
                    accepted[fid] = fact
                elif decision['decision'] == 'needs_context':
                    fact['verification_status'] = 'needs_context'
                    pending[fid] = fact
                trace['transitions'].append({'fact_id': fid, 'from': before, 'to': decision['decision'],
                                             **decision, 'record_ids': fact['record_ids'], 'evidence': fact['evidence']})
            # Even a narrower accepted statement can invalidate an old sufficiency
            # judgment.  Reassess needs separately, keeping verified facts intact.
            assessments = [{'need_index': i, 'status': 'unknown', 'fact_ids': [], 'answer': '',
                            'reason': 'Evidence review changed; requirement sufficiency must be reassessed.'}
                           for i in range(len(self._needs(state)))]
            pending_links = {link['link_id']: deepcopy(link) for link in state.get('pending_links', [])}
            verified_links = []
            touched_episodes = {f['episode_id'] for f in candidates.values()}
            for link in state.get('verified_links', []):
                depends = bool(set(link.get('fact_ids', [])) & set(selected))
                if not link.get('fact_ids'):
                    depends = bool({link['from_episode_id'], link['to_episode_id']} & touched_episodes)
                if depends:
                    clean = _link(link['from_episode_id'], link['to_episode_id'], link['rationale'])
                    pending_links[clean['link_id']] = clean
                else:
                    verified_links.append(deepcopy(link))
            return {'facts': list(accepted.values()), 'pending_facts': list(pending.values()),
                    'need_assessments': assessments, 'covered_needs': [], 'resolved_needs': [],
                    'verified_links': verified_links, 'pending_links': list(pending_links.values())}, []
        return self._run('facts', work)

    def review_need(self, state, need_index):
        """Judge one requirement independently against all accepted facts."""
        def work(trace):
            needs = self._needs(state)
            if type(need_index) is not int or not 0 <= need_index < len(needs):
                raise ValueError('invalid need_index')
            registry = self._registry(state)
            accepted = self._facts(state, registry, 'facts')
            payload, record_short = self._common(state, registry, trace, source_ids={f['source_id'] for f in accepted.values()})
            payload.update(need_index=need_index, need=needs[need_index])
            payload['facts'], short = self._fact_payload(accepted, record_short, state)
            trace['fact_ids'] = deepcopy(short)
            if accepted:
                response = self._call(NEED_SYSTEM, payload, trace)
                _object(response, {'status', 'fact_ids', 'answer', 'reason'})
                if response['status'] not in {'supported', 'partial', 'unknown', 'refuted'}:
                    raise ValueError('unknown requirement status')
                aliases = _refs(response['fact_ids'], short, 'fact_ids')
                resolved = response['status'] in {'supported', 'refuted'}
                if resolved and not aliases:
                    raise ValueError('resolved requirement must cite accepted facts')
                assessment = {'need_index': need_index, 'status': response['status'],
                              'fact_ids': [short[a] for a in aliases],
                              'answer': _text(response['answer'], 'answer', empty=not resolved),
                              'reason': _text(response['reason'], 'reason')}
            else:
                trace['skipped'] = 'no_accepted_facts'
                assessment = {'need_index': need_index, 'status': 'unknown', 'fact_ids': [], 'answer': '', 'reason': 'No accepted original evidence for this requirement yet.'}
            previous = {a['need_index']: deepcopy(a) for a in state.get('need_assessments', [])}
            previous[need_index] = assessment
            assessments = []
            for index in range(len(needs)):
                item = previous.get(index, {'need_index': index, 'status': 'unknown', 'fact_ids': [], 'answer': '', 'reason': 'Not reviewed yet.'})
                if any(fid not in accepted for fid in item['fact_ids']):
                    item = {'need_index': index, 'status': 'unknown', 'fact_ids': [], 'answer': '', 'reason': 'Previous supporting evidence is no longer accepted.'}
                if item['status'] in {'supported', 'refuted'} and not item['fact_ids']:
                    item = {'need_index': index, 'status': 'unknown', 'fact_ids': [], 'answer': '', 'reason': 'No accepted evidence supporting the previous judgment.'}
                assessments.append(item)
            trace['need_assessment'] = deepcopy(assessment)
            return {'need_assessments': assessments,
                    'covered_needs': [a['need_index'] for a in assessments if a['status'] == 'supported'],
                    'resolved_needs': [a['need_index'] for a in assessments if a['status'] in {'supported', 'refuted'}]}, []
        return self._run('need', work)

    def review_links(self, state, *, link_ids=None):
        """Return learnable links only after an independent accepted verdict."""
        def work(trace):
            registry = self._registry(state)
            accepted = self._facts(state, registry, 'facts')
            pending = {l['link_id']: deepcopy(l) for l in state.get('pending_links', [])}
            verified = {l['link_id']: deepcopy(l) for l in state.get('verified_links', [])}
            episodes = {f['episode_id'] for f in accepted.values()}
            for lid, link in list(verified.items()):
                if (not {link['from_episode_id'], link['to_episode_id']}.issubset(episodes)
                        or not link.get('fact_ids') or not set(link['fact_ids']).issubset(accepted)):
                    verified.pop(lid)
                    pending[lid] = _link(link['from_episode_id'], link['to_episode_id'], link['rationale'])
                    trace.setdefault('local_invalidations', []).append(lid)
            all_links = {**verified, **pending}
            selected = list(pending) if link_ids is None else _refs(link_ids, all_links, 'link_ids')
            episodes = {f['episode_id'] for f in accepted.values()}
            eligible = {lid: all_links[lid] for lid in selected if {all_links[lid]['from_episode_id'], all_links[lid]['to_episode_id']}.issubset(episodes)}
            trace['deferred_link_ids'] = [lid for lid in selected if lid not in eligible]
            if not eligible:
                trace['skipped'] = 'no_links_with_two_accepted_endpoints'
                return {'verified_links': list(verified.values()), 'pending_links': list(pending.values())}, []
            endpoints = {eid for link in eligible.values() for eid in (link['from_episode_id'], link['to_episode_id'])}
            link_facts = {fid: f for fid, f in accepted.items() if f['episode_id'] in endpoints}
            payload, record_short = self._common(state, registry, trace, source_ids={f['source_id'] for f in link_facts.values()})
            payload['facts'], fact_short = self._fact_payload(link_facts, record_short, state)
            trace['fact_ids'] = deepcopy(fact_short)
            short = {f'L{i}': lid for i, lid in enumerate(eligible, 1)}
            payload['links'] = [{'link_id': alias, 'from_episode_id': eligible[lid]['from_episode_id'],
                                 'to_episode_id': eligible[lid]['to_episode_id'], 'rationale': eligible[lid]['rationale']} for alias, lid in short.items()]
            trace['link_ids'] = deepcopy(short)
            decisions = self._decisions(self._call(LINKS_SYSTEM, payload, trace), 'link_decisions', 'link_id', short)
            for lid, decision in decisions.items():
                link = eligible[lid]
                pending.pop(lid, None)
                verified.pop(lid, None)
                if decision['decision'] == 'accept':
                    endpoint_facts = [f for f in accepted.values() if f['episode_id'] in {link['from_episode_id'], link['to_episode_id']}]
                    evidence = {}
                    for fact in endpoint_facts:
                        for item in fact['evidence']:
                            evidence[tuple(item[k] for k in _EVIDENCE)] = deepcopy(item)
                    verified[lid] = {**link, 'fact_ids': [f['fact_id'] for f in endpoint_facts], 'evidence': list(evidence.values()),
                                     'verified': True, 'verifier': 'record_review_v3', 'review_reason': decision['reason']}
                elif decision['decision'] == 'needs_context':
                    pending[lid] = deepcopy(link)
                trace['transitions'].append({'link_id': lid, 'to': decision['decision'], **decision})
            return {'verified_links': list(verified.values()), 'pending_links': list(pending.values())}, []
        return self._run('links', work)
