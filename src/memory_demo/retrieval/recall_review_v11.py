"""V10 admission, with wave-entry NEED diagnostics in nonempty MAP inputs."""
from copy import deepcopy

from config.prompt_config.recall_v11_prompts import GAP_MAP
from memory_demo.retrieval.recall_map_feedback import active_feedback
from memory_demo.retrieval.recall_review_v3 import RecordReview, MAP_SYSTEM
from memory_demo.retrieval.recall_review_v4 import MAP_SYSTEM as V4_MAP_SYSTEM
from memory_demo.retrieval.recall_review_v10 import RecordReviewV10


class RecordReviewV11(RecordReviewV10):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 11
            return work(trace)
        return RecordReview._run(self, stage, versioned)

    def _common(self, state, registry, trace, *, source_ids=None):
        payload, short = super()._common(state, registry, trace, source_ids=source_ids)
        if trace['stage'] == 'map':
            snapshot = active_feedback(state)
            trace['map_gap_feedback_snapshot'] = snapshot
            if snapshot['payload']['items']:
                payload['need_feedback'] = deepcopy(snapshot['payload'])
        return payload, short

    def _call(self, system, payload, trace):
        if system != MAP_SYSTEM or not payload.get('need_feedback', {}).get('items'):
            return super()._call(system, payload, trace)
        # Keep V5 optional-cue admission and V4 payload tracing unchanged.
        # The reviewer is used sequentially; restore its callback even on errors.
        original = self.call
        self.call = lambda prompt, data: original(GAP_MAP if prompt == V4_MAP_SYSTEM else prompt, data)
        try:
            return super()._call(system, payload, trace)
        finally:
            self.call = original
