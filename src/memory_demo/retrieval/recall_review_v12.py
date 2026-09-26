"""V10 review with isolated validation of individual MAP fact proposals."""
from memory_demo.retrieval.recall_map_isolation import FactIsolatedReviewV10


class RecordReviewV12(FactIsolatedReviewV10):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 12
            return work(trace)
        # Keep the isolation wrapper and V10's unchanged stage transaction.
        return super()._run(stage, versioned)
