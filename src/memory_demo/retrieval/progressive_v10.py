"""V9 contracts with local request fragment anchors instead of model-copied text."""
from copy import deepcopy

from config.prompt_config.recall_v10_prompts import CONTRACT
from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v9 import ProgressiveRecallV9
from memory_demo.retrieval.recall_contract_anchors import ANCHOR_PROTOCOL, anchor_binding
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10


class ProgressiveRecallV10(ProgressiveRecallV9):
    protocol_version = 10
    reviewer_class = RecordReviewV10
    learning_verifier = 'progressive_record_review_v10'

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        self._snapshot_payload['contract_anchor_input'] = {
            'protocol': deepcopy(ANCHOR_PROTOCOL), 'prompt_sha256': _hash(CONTRACT)}
        return sources, episodes, rows, _hash(self._snapshot_payload)

    @staticmethod
    def completion_blockers(state):
        blockers = ProgressiveRecallV9.completion_blockers(state)
        if state.get('need_contracts') and state.get('contract_anchor_binding') != anchor_binding(state):
            blockers['contract_anchor_input_invalid'] = [True]
        return blockers

    @staticmethod
    def _resolved(state):
        return (ProgressiveRecallV9._resolved(state)
                and state.get('contract_anchor_binding') == anchor_binding(state))

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        result['contract_anchor_binding'] = deepcopy(state.get('contract_anchor_binding'))
        return result

    def _seed(self, state, episodes, sources, policy):
        if state.get('need_contracts') and state.get('contract_anchor_binding') != anchor_binding(state):
            raise ValueError('original request fragment binding changed')
        return super()._seed(state, episodes, sources, policy)
