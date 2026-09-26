"""V6 evidence admission and prompts; only the protocol trace version changes."""
from memory_demo.retrieval.recall_review_v3 import RecordReview
from memory_demo.retrieval.recall_review_v6 import RecordReviewV6


class RecordReviewV7(RecordReviewV6):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 7
            return work(trace)
        return RecordReview._run(self, stage, versioned)
