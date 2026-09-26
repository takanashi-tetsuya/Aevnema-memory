"""V4 evidence rules with isolated optional cues and traceable failed calls."""
from copy import deepcopy

from memory_demo.llm.client import ModelClientError
from memory_demo.retrieval.recall_review_v3 import (
    MAP_SYSTEM, RecordReview, ReviewStageError, _object, _text,
)
from memory_demo.retrieval.recall_review_v4 import RecordReviewV4


class RecordReviewV5(RecordReviewV4):
    def _run(self, stage, work):
        def versioned(trace):
            trace['version'] = 5
            return work(trace)
        return RecordReview._run(self, stage, versioned)

    def _call(self, system, payload, trace):
        try:
            response = super()._call(system, payload, trace)
        except (TimeoutError, ModelClientError) as exc:
            # Preserve the exact request and short-ID maps even without a reply.
            trace.update(provider_error_type=type(exc).__name__, error=str(exc))
            raise ReviewStageError(str(exc), trace=trace) from exc
        if system != MAP_SYSTEM or not isinstance(response, dict) or 'followup_cues' not in response:
            return response
        cues = response['followup_cues']
        rejected, retained = [], []
        if not isinstance(cues, list):
            rejected.append({'index': None, 'original': deepcopy(cues), 'reason': 'followup_cues must be a list'})
        else:
            records = {r['id']: r for r in payload['records']}
            for index, item in enumerate(cues):
                try:
                    _object(item, {'record_id', 'cue'})
                    rid = _text(item['record_id'], 'record_id')
                    if rid != item['record_id'] or rid not in records:
                        raise ValueError('unknown record ID')
                    cue = _text(item['cue'], 'cue')
                    if cue not in records[rid]['text']:
                        raise ValueError('cue is not a literal substring of its displayed record')
                except ValueError as exc:
                    rejected.append({'index': index, 'original': deepcopy(item), 'reason': str(exc)})
                else:
                    retained.append(deepcopy(item))
        if not rejected:
            return response
        clean = {**deepcopy(response), 'followup_cues': retained}
        # Raw provider response remains untouched in trace['response']; the
        # effective response and each rejection are independently inspectable.
        trace['rejected_optional_cues'] = rejected
        trace['admitted_response'] = deepcopy(clean)
        return clean
