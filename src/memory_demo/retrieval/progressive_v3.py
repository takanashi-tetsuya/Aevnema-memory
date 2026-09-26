"""Experimental staged Source review; the v2 public default stays unchanged.

Only the review protocol changes. Seed vectors, weak-cue spreading, graph
weights and the complete fallback ranking are inherited from ProgressiveRecall.
No evaluator labels or story-specific query rules enter this service.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

from config.prompt_config import recall_v3_prompts as prompts
from memory_demo.associations.feedback import SourceEvidence, VerifiedRecallLink
from memory_demo.llm.client import CampaignBudgetError, ModelClient, ModelClientError
from memory_demo.llm.prompts import natural_prompt_data
from memory_demo.retrieval.progressive import NaturalPrompt, ProgressiveRecall, stable_fact_id, _hash
from memory_demo.retrieval.recall_cues import enqueue_followup_cues
from memory_demo.retrieval.recall_records import align_window, project_window
from memory_demo.retrieval.recall_review_v3 import RecordReview, ReviewStageResult


class ProgressiveRecallV3(ProgressiveRecall):
    protocol_version = 3
    reviewer_class = RecordReview
    learning_verifier = "progressive_record_review_v3"
    plan_system = prompts.PLAN
    stage_seconds = {"plan": 40.0, "map": 45.0, "facts": 45.0,
                     "need": 35.0, "links": 35.0}

    @staticmethod
    def _resolved(state):
        # A durable sufficiency verdict can precede the final link-stage
        # checkpoint. Resumption must finish that work before learning/return.
        return not state.get("review_work") and ProgressiveRecall._resolved(state)

    def _call(self, system, payload, state):
        self._ensure_live()
        stage = "plan" if system == self.plan_system else getattr(self, "_stage_name", "plan")
        started = self.clock()
        end = min(self._deadline, started + self.stage_seconds[stage])
        state["metrics"]["model_batches"] += 1
        readable = NaturalPrompt(natural_prompt_data(payload), payload)
        trace = {"stage": stage, "input_chars": len(readable),
                 "deadline_seconds": max(0.0, end - started)}
        scope = self.model.deadline_budget(end) if hasattr(self.model, "deadline_budget") else nullcontext()
        try:
            with scope:
                kwargs = {"allow_fallback": False, "max_retries": 0} if isinstance(self.model, ModelClient) else {}
                result = self.model.chat_json(system, readable, **kwargs)
            if stage == "map":
                # Count completed presentations even when schema validation
                # later rejects the mapping. Timeout visibility is unknown.
                state["presented_episode_ids"] = list(dict.fromkeys([
                    *state.get("presented_episode_ids", []),
                    *(item["episode_id"] for item in payload.get("episode_hints", []))]))
            self._ensure_live()
            if self.clock() >= end:
                raise TimeoutError("review stage deadline exhausted")
            if not isinstance(result, dict):
                raise ValueError("review stage must return an object")
            trace["status"] = "returned"
            return result
        except BaseException as exc:
            trace.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            trace["elapsed_seconds"] = max(0.0, self.clock() - started)
            state.setdefault("stage_calls", []).append(trace)

    def _snapshot(self):
        result = super()._snapshot()
        self._source_episodes = {}
        for eid, episode in result[1].items():
            self._source_episodes.setdefault(int(episode["source_id"]), []).append(eid)
        return result

    def _windows(self, candidates, state, episodes, sources, policy):
        # Diagnostic candidate budget is fixed independently of final evidence.
        state["candidate_episode_ids"] = list(dict.fromkeys(
            [*state.get("candidate_episode_ids", []), *candidates]))[:100]
        windows = super()._windows(candidates, state, episodes, sources, policy)
        for window in windows:
            sid = window["source_id"]
            start, end = align_window(sources[sid], window["start"], window["end"])
            window.update(start=start, end=end, text=sources[sid][start:end],
                          episode_ids=self._source_episodes[sid])
        return windows

    def _context_windows(self, state, sources, episodes, policy):
        if state.get("review_work"):
            return deepcopy(state["review_work"]["windows"])
        # Pending records were read at complete record boundaries and survive
        # independently of the Source cursor. Expanded context is still useful
        # for a suspended claim, and each claim gets at most one such reread.
        windows = super()._context_windows(state, sources, episodes, policy)
        for window in windows:
            sid = window["source_id"]
            start, end = align_window(sources[sid], window["start"], window["end"])
            window.update(start=start, end=end, text=sources[sid][start:end],
                          episode_ids=self._source_episodes[sid])
        return windows

    def _commit_stage(self, operation, state, stage):
        self._stage_name = stage
        self._ensure_live()
        work = state["review_work"]
        cursor = work.get("batch_cursor", 0) if stage == "facts" else work.get("need_cursor", 0) if stage == "need" else 0
        token = f"{stage}:{cursor}"
        cached = work.get("committed_stages", {}).get(token)
        if cached is not None:
            # The stage result and this token were one atomic checkpoint.
            # A crash before the caller advanced its cursor must not make an
            # already rejected fact an invalid repeated batch or repeat HTTP.
            return ReviewStageResult({}, {"resumed_committed_stage": token}, deepcopy(cached["followup_cues"]))
        try:
            result = operation()
            self._ensure_live()
        except CampaignBudgetError:
            raise
        except (ValueError, TimeoutError, ModelClientError) as exc:
            if getattr(exc, "trace", None):
                state.setdefault("review_trace", []).append({**deepcopy(exc.trace),
                    "committed": False, "stage": stage,
                    "wave": state["metrics"]["review_waves"] + 1})
            state.setdefault("stage_errors", []).append({"stage": stage,
                "type": type(exc).__name__, "message": str(exc)[:400],
                "wave": state["metrics"]["review_waves"] + 1})
            self.sessions.write(state)
            # Local provider deadlines cannot consume all subsequent waves.
            # Global deadlines and cancellation still stop the request.
            self._ensure_live()
            return None
        state.update(result.updates)
        state.setdefault("review_trace", []).append({**deepcopy(result.trace),
            "committed": True, "stage": stage,
            "wave": state["metrics"]["review_waves"] + 1})
        work.setdefault("committed_stages", {})[token] = {"followup_cues": deepcopy(result.followup_cues)}
        self.sessions.write(state)
        return result

    def _review(self, windows, state, sources, episodes):
        if not state.get("review_work"):
            state["review_work"] = {"windows": deepcopy(windows), "phase": "map",
                                    "map_failures": 0, "batch_cursor": 0, "need_cursor": 0,
                                    "committed_stages": {}}
            self.sessions.write(state)
        work = state["review_work"]
        reviewer = self.reviewer_class(sources, episodes, lambda system, payload: self._call(system, payload, state))
        if work["phase"] == "map":
            records = []
            for window in work["windows"]:
                projection = project_window(window["source_id"], sources[window["source_id"]],
                    window.get("episode_ids", [window["episode_id"]]), start=window["start"], end=window["end"])
                if projection.incomplete_ranges:
                    raise ValueError("review window contains an incomplete record")
                records.extend(projection.records)
            state["current_record_ids"] = [r["record_id"] for r in records]
            old_pending = {f["fact_id"] for f in state.get("pending_facts", [])}
            mapped = self._commit_stage(lambda: reviewer.map(records, state), state, "map")
            if mapped is None:
                work["map_failures"] += 1
                self.sessions.write(state)
                if work["map_failures"] >= 2:
                    raise ValueError("record mapping failed twice; staged windows remain resumable")
                return
            # Cursors advance only after all mapped candidates have a durable
            # pending record. Unverified candidates never enter result.evidence.
            visible = []
            for window in work["windows"]:
                sid = window["source_id"]
                if not window.get("recheck"):
                    state["source_offsets"][str(sid)] = max(state["source_offsets"].get(str(sid), 0), window["end"])
                visible.append({**window, "episode_id": window["episode_id"]})
            cues = enqueue_followup_cues(state.get("cue_queue"), mapped.followup_cues,
                visible=visible, sources=sources, episodes=episodes,
                initial_cues=[state["question"], *state["cues"], *state["needs"]])
            state["cue_queue"] = cues.queue
            state["metrics"]["followup_cues_accepted"] += cues.stats["accepted"]
            pending = [f["fact_id"] for f in state["pending_facts"]]
            pending = [fid for fid in pending if fid not in old_pending] + [fid for fid in pending if fid in old_pending]
            # Revisit a bounded rotating portion of old evidence against new
            # context, including counterevidence from another Source. Do not
            # repeatedly send every accepted fact to every fact reviewer.
            accepted_ids = [f["fact_id"] for f in state["facts"]]
            if accepted_ids:
                cursor = int(state.get("old_fact_review_cursor", 0)) % len(accepted_ids)
                rotated = accepted_ids[cursor:] + accepted_ids[:cursor]
                pending.extend(rotated[:2])
                state["old_fact_review_cursor"] = (cursor + min(2, len(accepted_ids))) % len(accepted_ids)
            work.update(phase="facts", batches=[pending[i:i + 4] for i in range(0, len(pending), 4)])
            self.sessions.write(state)
        if work["phase"] == "facts":
            while work["batch_cursor"] < len(work["batches"]):
                batch = work["batches"][work["batch_cursor"]]
                self._commit_stage(lambda: reviewer.review_facts(state, fact_ids=batch), state, "facts")
                work["batch_cursor"] += 1
                self.sessions.write(state)
            work["phase"] = "needs"
            self.sessions.write(state)
        if work["phase"] == "needs":
            # An empty evidence set cannot resolve a requirement; skip paid
            # sufficiency calls when mapping produced no admissible facts.
            while state["facts"] and work["need_cursor"] < len(state["needs"]):
                index = work["need_cursor"]
                self._commit_stage(lambda: reviewer.review_need(state, index), state, "need")
                work["need_cursor"] += 1
                self.sessions.write(state)
            work["phase"] = "links"
            self.sessions.write(state)
        if work["phase"] == "links" and state.get("pending_links"):
            self._commit_stage(lambda: reviewer.review_links(state), state, "links")
        rechecked = set(state.get("context_rechecked_fact_ids", []))
        for window in work["windows"]:
            rechecked.update(window.get("recheck_fact_ids", []))
        state["context_rechecked_fact_ids"] = sorted(rechecked)
        state["metrics"]["source_windows_reviewed"] += len(work["windows"])
        state["metrics"]["review_waves"] += 1
        state["review_work"] = None
        self.sessions.write(state)

    @classmethod
    def _learning_links(cls, state):
        return [VerifiedRecallLink(link["from_episode_id"], link["to_episode_id"],
            tuple(SourceEvidence(**{key: evidence[key] for key in (
                "episode_id", "source_id", "source_sha256", "start", "end", "quote")})
                for fact in link["evidence"] for evidence in fact.get("evidence", [fact])),
            verifier=cls.learning_verifier, rationale=link["rationale"], verified=True)
            for link in state["verified_links"]]

    def _validate_delivered_sources(self, state):
        super()._validate_delivered_sources(state)
        additional = [e for fact in [*state["facts"], *state.get("pending_facts", [])]
                      for e in fact.get("evidence", [])]
        if additional:
            super()._validate_delivered_sources({"facts": additional})

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        state = self.sessions.read(result["session_id"])
        result.update(review_protocol_version=self.protocol_version,
            coverage_basis=f"model_plan_and_independent_record_review_v{self.protocol_version}",
            candidate_episode_ids=state.get("candidate_episode_ids", []),
            presented_episode_ids=state.get("presented_episode_ids", []),
            stage_calls=state.get("stage_calls", []), stage_errors=state.get("stage_errors", []))
        return result
