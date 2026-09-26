"""V10 full recall with fact isolation; preserve its scheduling and completion."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from memory_demo.retrieval import recall_map_isolation, recall_review_v12
from memory_demo.retrieval.progressive import _hash
from memory_demo.retrieval.progressive_v10 import ProgressiveRecallV10
from memory_demo.retrieval.recall_map_isolation import FACT_ISOLATION_PROTOCOL
from memory_demo.retrieval.recall_review_v12 import RecordReviewV12


def isolation_implementation_hashes():
    """Bind policy code as well as its declared version on checkpoint resume."""
    paths = [Path(__file__), Path(recall_map_isolation.__file__),
             Path(recall_review_v12.__file__)]
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


class ProgressiveRecallV12(ProgressiveRecallV10):
    protocol_version = 12
    reviewer_class = RecordReviewV12
    learning_verifier = 'progressive_record_review_v12'

    def _snapshot(self):
        sources, episodes, rows, _ = super()._snapshot()
        self._snapshot_payload['map_fact_isolation'] = {
            'base_protocol_version': 10,
            'protocol': deepcopy(FACT_ISOLATION_PROTOCOL),
            'implementation_sha256': isolation_implementation_hashes(),
        }
        return sources, episodes, rows, _hash(self._snapshot_payload)

    @classmethod
    def annotate_completion(cls, result, state):
        result = super().annotate_completion(result, state)
        result['map_fact_isolation_protocol'] = deepcopy(FACT_ISOLATION_PROTOCOL)
        return result
