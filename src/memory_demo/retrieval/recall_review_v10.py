"""V9 NEED checks; construct contracts by selecting local request fragment IDs."""
from copy import deepcopy

from config.prompt_config.recall_v10_prompts import CONTRACT
from memory_demo.retrieval.recall_contract_anchors import anchor_input, admit_anchor_contracts, anchor_binding
from memory_demo.retrieval.recall_review_v3 import RecordReview
from memory_demo.retrieval.recall_review_v9 import RecordReviewV9


class RecordReviewV10(RecordReviewV9):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 10
            return work(trace)
        return RecordReview._run(self, stage, versioned)

    def build_contracts(self, state):
        def work(trace):
            self._needs(state)
            if state.get('record_registry') or state.get('facts') or state.get('pending_facts'):
                raise ValueError('contracts must be fixed before reading evidence')
            response = self._call(CONTRACT, anchor_input(state), trace)
            updates, admitted = admit_anchor_contracts(state, response)
            updates['contract_anchor_binding'] = anchor_binding(state)
            trace.update(deepcopy(updates), admitted_response=admitted)
            return updates, []
        return self._run('contract', work)
