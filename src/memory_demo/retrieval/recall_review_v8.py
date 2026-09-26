"""V7 evidence rules and prompts; V8 changes scheduling only."""
from memory_demo.retrieval.recall_review_v3 import RecordReview
from memory_demo.retrieval.recall_review_v7 import RecordReviewV7


class RecordReviewV8(RecordReviewV7):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 8
            return work(trace)
        return RecordReview._run(self, stage, versioned)
