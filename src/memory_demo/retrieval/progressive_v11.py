"""V10 search and stopping; feed real previous NEED gaps into the next MAP."""
from copy import deepcopy

from config.prompt_config.recall_v11_prompts import GAP_MAP
from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
from memory_demo.retrieval.recall_map_feedback import FEEDBACK_PROTOCOL, capture_feedback, validate_feedback
from memory_demo.retrieval.recall_review_v11 import RecordReviewV11


class ProgressiveRecallV11(ProgressiveRecallV10):
    protocol_version = 11
    reviewer_class = RecordReviewV11
    learning_verifier = 'progressive_record_review_v11'

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        self._snapshot_payload['map_gap_feedback'] = {
            'protocol': deepcopy(FEEDBACK_PROTOCOL), 'prompt_sha256': _hash(GAP_MAP)}
        return sources, episodes, rows, _hash(self._snapshot_payload)

    def _seed(self, state, episodes, sources, policy):
        snapshots = state.get('map_feedback_snapshots', [])
        completed = state['metrics']['review_waves']
        if (not isinstance(snapshots, list) or len(snapshots) not in (completed, completed + 1)
                or state.get('review_work') and len(snapshots) != completed + 1):
            raise ValueError('feedback history does not match the review progress')
        for index, snapshot in enumerate(snapshots):
            if snapshot.get('wave') != index + 1:
                raise ValueError('feedback wave history is not consecutive')
            validate_feedback(state, snapshot)
        return super()._seed(state, episodes, sources, policy)

    def _review(self, windows, state, sources, episodes):
        wave = state['metrics']['review_waves'] + 1
        snapshots = state.setdefault('map_feedback_snapshots', [])
        if any(s.get('wave') != i + 1 for i, s in enumerate(snapshots)):
            raise ValueError('feedback wave history is not consecutive')
        if len(snapshots) == wave - 1:
            if state.get('review_work'):
                raise ValueError('started MAP work has no entry feedback')
            snapshots.append(capture_feedback(state, wave))
            self.sessions.write(state)
        elif len(snapshots) != wave:
            raise ValueError('feedback wave does not match the current review')
        validate_feedback(state, snapshots[-1])
        return super()._review(windows, state, sources, episodes)

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        result['map_feedback_snapshots'] = deepcopy(state.get('map_feedback_snapshots', []))
        return result
