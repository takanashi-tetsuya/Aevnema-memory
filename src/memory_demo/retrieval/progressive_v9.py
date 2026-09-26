"""V8 navigation/scheduling with a pre-evidence, per-item NEED completion gate."""
from copy import deepcopy

from config.prompt_config.recall_v9_prompts import CONTRACT, NEEDS
from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v8 import ProgressiveRecallV8
from memory_demo.retrieval.recall_need_contracts import (
    CONTRACT_PROTOCOL, bound_contracts, contract_gaps,
)
from memory_demo.retrieval.recall_review_v9 import RecordReviewV9


class ProgressiveRecallV9(ProgressiveRecallV8):
    protocol_version = 9
    reviewer_class = RecordReviewV9
    learning_verifier = 'progressive_record_review_v9'
    stage_seconds = {**ProgressiveRecallV8.stage_seconds, 'contract': 40.0}

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        self._snapshot_payload['need_contract'] = {
            'protocol': deepcopy(CONTRACT_PROTOCOL), 'contract_prompt': _hash(CONTRACT),
            'need_prompt': _hash(NEEDS), 'contract_seconds': self.stage_seconds['contract']}
        return sources, episodes, rows, _hash(self._snapshot_payload)

    @staticmethod
    def completion_blockers(state):
        return {**ProgressiveRecallV8.completion_blockers(state),
                'need_contract_gaps': contract_gaps(state)}

    @staticmethod
    def _resolved(state):
        return ProgressiveRecallV8._resolved(state) and not contract_gaps(state)

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        for key in ('need_contract_protocol', 'need_contracts', 'need_contract_hash', 'contract_work'):
            result[key] = deepcopy(state.get(key))
        result['need_contract_gaps'] = contract_gaps(state)
        if state.get('contract_work', {}).get('status') in {'started', 'failed'}:
            result['resumable'] = False
        return result

    def _seed(self, state, episodes, sources, policy):
        # This hook follows the unchanged PLAN commit and precedes embeddings,
        # graph search, and the first Source read. Evidence cannot reshape the
        # contract to fit whichever facts happen to be retrieved.
        if state.get('need_contracts'):
            bound_contracts(state)
            if state.get('contract_work', {}).get('status') != 'committed':
                raise ValueError('need contracts lack a committed construction')
        else:
            if state.get('contract_work'):
                raise ValueError('contract construction already attempted; do not repeat an uncertain or failed call')
            self._ensure_live()
            state['contract_work'] = {'status': 'started', 'max_attempts': 1}
            self.sessions.write(state)
            self._stage_name = 'contract'
            reviewer = self.reviewer_class(sources, episodes,
                lambda system, payload: self._call(system, payload, state))
            try:
                result = reviewer.build_contracts(state)
                self._ensure_live()
            except BaseException as exc:
                trace = deepcopy(getattr(exc, 'trace', {}) or {})
                state['contract_work'].update(status='failed', error_type=type(exc).__name__)
                state.setdefault('review_trace', []).append({**trace, 'stage': 'contract',
                    'wave': 0, 'committed': False})
                state.setdefault('stage_errors', []).append({'stage': 'contract', 'wave': 0,
                    'type': type(exc).__name__, 'message': str(exc)[:400]})
                self.sessions.write(state)
                self._ensure_live()
                raise
            state.update(result.updates)
            state['contract_work']['status'] = 'committed'
            state.setdefault('review_trace', []).append({**deepcopy(result.trace),
                'stage': 'contract', 'wave': 0, 'committed': True})
            self.sessions.write(state)
        return super()._seed(state, episodes, sources, policy)
