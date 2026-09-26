"""Experimental recall with complete record groups and focused sufficiency."""
from memory_demo.retrieval.progressive_v3 import ProgressiveRecallV3
from memory_demo.retrieval.recall_review_v4 import RecordReviewV4


class ProgressiveRecallV4(ProgressiveRecallV3):
    protocol_version = 4
    reviewer_class = RecordReviewV4
    learning_verifier = "progressive_record_review_v4"
