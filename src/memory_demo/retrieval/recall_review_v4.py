"""Opt-in multi-record evidence with one explicit sufficiency target per call.

All record provenance, Episode alignment, atomic updates, and complete-context
budgets are inherited unchanged. Only the per-fact record count and map/need
requests differ from v3; this is a combined protocol revision, not an ablation.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from memory_demo.retrieval.recall_review_v3 import (
    FACTS_SYSTEM, LINKS_SYSTEM,
    MAP_SYSTEM as V3_MAP_SYSTEM, NEED_SYSTEM as V3_NEED_SYSTEM,
    RecordReview,
)

_path = Path(__file__).resolve().parents[3] / 'config' / 'prompt_config' / 'recall_v4_prompts.py'
_spec = spec_from_file_location('memory_demo._record_review_v4_prompts', _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f'cannot load record review prompts: {_path}')
_catalog = module_from_spec(_spec)
_spec.loader.exec_module(_catalog)
MAP_SYSTEM, NEED_SYSTEM = _catalog.MAP, _catalog.NEED


class RecordReviewV4(RecordReview):
    max_fact_records = None

    def _run(self, stage, work):
        def versioned_work(trace):
            trace['version'] = 4
            return work(trace)
        return super()._run(stage, versioned_work)

    def _call(self, system, payload, trace):
        if system == V3_MAP_SYSTEM:
            system = MAP_SYSTEM
        elif system == V3_NEED_SYSTEM:
            # Transform before the inherited tracing and size check so the
            # archived request is exactly the request sent to the provider.
            payload = {
                'question': payload['need'],
                'request_context': payload['question'],
                **{key: deepcopy(value) for key, value in payload.items()
                   if key not in {'question', 'needs', 'need'}},
            }
            system = NEED_SYSTEM
        return super()._call(system, payload, trace)
