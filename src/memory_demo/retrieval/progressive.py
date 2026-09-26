"""Resumable, source-checked associative recall with a small local graph rule.

The model interprets language and verifies Source evidence. It never chooses a
relation taxonomy for local spreading, and node activation never certifies a fact.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from time import monotonic
from typing import Callable
from uuid import uuid4

import numpy as np

from memory_demo.llm.recall_prompts import RECALL_MAP_SYSTEM, RECALL_PLAN_SYSTEM, RECALL_VERIFY_SYSTEM
from memory_demo.llm.prompts import natural_prompt_data
from memory_demo.associations.feedback import RecallFeedbackService, SourceEvidence, VerifiedRecallLink
from memory_demo.associations.spreading import SpreadingEdge, SpreadingRecall, SpreadingSeed
from memory_demo.database import transaction_liveness
from memory_demo.embeddings.codec import decode_embedding, normalize_embedding
from memory_demo.retrieval.recall_policy import RecallPolicy
from memory_demo.retrieval.recall_review import ReviewSchemaError, review_wave, stable_fact_id
from memory_demo.retrieval.recall_cues import (
    enqueue_followup_cues, expand_pending_cues, pending_cue_texts,
    restart_spreading_epoch,
)


def _hash(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NaturalPrompt(str):
    """Readable provider text with local-only payload for instrumentation.

    The structured payload is available to deterministic test doubles, while
    JSON encoding a NaturalPrompt sends only its string value to the provider.
    """

    def __new__(cls, text: str, structured_payload: dict):
        value = super().__new__(cls, text)
        value.structured_payload = structured_payload
        return value


def _indices(values: object, size: int) -> list[int]:
    if not isinstance(values, list):
        return []
    return sorted({v for v in values if type(v) is int and 0 <= v < size})


def _strings(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(v.strip() for v in values if isinstance(v, str) and v.strip()))[:limit]


class RecallSessionStore:
    """Atomic local checkpoints. User-provided IDs cannot name arbitrary files."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", str(session_id)):
            raise ValueError("invalid recall session id")
        return self.directory / f"{session_id}.json"

    def read(self, session_id: str) -> dict:
        value = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("session_id") != session_id:
            raise ValueError("invalid recall checkpoint")
        if value.get("checksum") != _hash({k: v for k, v in value.items() if k != "checksum"}):
            raise ValueError("recall checkpoint checksum mismatch")
        return value

    def write(self, state: dict) -> None:
        state["checksum"] = _hash({k: v for k, v in state.items() if k != "checksum"})
        target = self._path(state["session_id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


class ProgressiveRecall:
    protocol_version = 2
    plan_system = RECALL_PLAN_SYSTEM

    def __init__(self, config, db, model, *, clock: Callable[[], float] = monotonic):
        self.config, self.db, self.model, self.clock = config, db, model, clock
        self.sessions = RecallSessionStore(config.log_dir / "recall_sessions")
        self.feedback = RecallFeedbackService(db)

    def _snapshot(self) -> tuple[dict, dict, list, str]:
        # No schema initialization, index rebuilding, or learning is needed to
        # inspect an existing graph. A single read transaction binds the rows.
        with self.db.connection() as connection:
            connection.execute("BEGIN")
            sources = {int(r["id"]): str(r["raw_text"]) for r in connection.execute("SELECT id,raw_text FROM source")}
            episodes = {int(r["id"]): dict(r) for r in connection.execute("SELECT id,source_id,text,embedding FROM episode")}
            concepts = [int(r[0]) for r in connection.execute("SELECT id FROM concept ORDER BY id")]
            rows = [dict(r) for r in connection.execute("SELECT id,from_type,from_id,to_type,to_id,weight,confidence,polarity,association_mode,lifecycle_state FROM association ORDER BY id")]
        valid_links = self.feedback.valid_learned_edge_ids()
        edges = [r for r in rows if r["lifecycle_state"] != "retired" and float(r["weight"]) > 0 and (r["association_mode"] != "simple_recall" or r["id"] in valid_links)]
        # Time does not diminish a memory. Explicit retirement and changed
        # evidence are separate validity conditions, not age-based forgetting.
        self._nodes = [("episode", i) for i in episodes] + [("concept", i) for i in concepts]
        self._snapshot_payload = {"sources": [(i, _hash(t)) for i, t in sorted(sources.items())], "episodes": [(i, r["source_id"], _hash(r["text"]), hashlib.sha256(r["embedding"]).hexdigest()) for i, r in sorted(episodes.items())], "concepts": concepts, "edges": edges, "model": self.config.model.embedding_model}
        fingerprint = _hash(self._snapshot_payload)
        self._episode_vector_cache = None
        return sources, episodes, edges, fingerprint

    @staticmethod
    def _learning_links(state: dict) -> list[VerifiedRecallLink]:
        return [VerifiedRecallLink(v["from_episode_id"], v["to_episode_id"],
                    tuple(SourceEvidence(**{k: f[k] for k in ("episode_id", "source_id", "source_sha256", "start", "end", "quote")}) for f in v["evidence"]),
                    verifier="progressive_source_review_v2", rationale=v["rationale"], verified=True)
                for v in state["verified_links"]]

    def _recover_committed_learning(self, state: dict, *, refresh: bool) -> bool:
        if not self._resolved(state) or not state["verified_links"]:
            return False
        receipt = self.feedback.verified_receipt("recall:" + state["session_id"], self._learning_links(state))
        if receipt is None:
            return False
        self._validate_delivered_sources(state)
        if refresh:
            self._snapshot()
        current_hash = _hash(self._snapshot_payload)
        if current_hash != state["snapshot_hash"]:
            # Only this exact transaction may explain a changed snapshot.
            # Concurrent edits and later feedback must still reject resumption.
            changes = {change.association_id: change for change in receipt.changes}
            reviewed_pairs = {tuple(sorted((link["from_episode_id"], link["to_episode_id"]))) for link in state["verified_links"]}
            restored = []
            found = set()
            for row in self._snapshot_payload["edges"]:
                change = changes.get(row["id"])
                if change is None:
                    restored.append(row)
                    continue
                if float(row["weight"]) != change.after:
                    return False
                if change.created and (
                    row["from_type"] != "episode" or row["to_type"] != "episode"
                    or (row["from_id"], row["to_id"]) not in reviewed_pairs
                    or row["confidence"] != 1.0 or row["polarity"] != 1
                    or row["association_mode"] != "simple_recall" or row["lifecycle_state"] != "active"
                ):
                    return False
                found.add(row["id"])
                if not change.created and change.before is not None and change.before > 0:
                    restored.append({**row, "weight": change.before})
            if found != set(changes) or _hash({**self._snapshot_payload, "edges": restored}) != state["snapshot_hash"]:
                return False
        state["snapshot_hash"] = current_hash
        state["learning"] = {"status": "applied", "association_ids": list(receipt.association_ids)}
        state["status"] = "complete"
        state.pop("error", None)
        return True

    def _call(self, system: str, payload: dict, state: dict) -> dict:
        self._ensure_live()
        state["metrics"]["model_batches"] += 1
        result = self.model.chat_json(
            system, NaturalPrompt(natural_prompt_data(payload), payload)
        )
        self._ensure_live()
        if not isinstance(result, dict):
            raise ValueError("recall model must return a JSON object")
        return result

    def _ensure_live(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError("recall cancelled")
        if self.clock() >= self._deadline:
            raise TimeoutError("recall time budget exhausted")

    @staticmethod
    def _resolved(state: dict) -> bool:
        return bool(state["needs"]) and set(state.get("resolved_needs", [])) == set(range(len(state["needs"])))

    def _episode_vectors(self, episodes: dict) -> tuple[list[int], np.ndarray]:
        if getattr(self, "_episode_vector_cache", None) is None:
            ids = sorted(episodes)
            dimension = self.config.model.embedding_dimension
            matrix = np.stack([
                normalize_embedding(decode_embedding(episodes[i]["embedding"], dimension), dimension)
                for i in ids
            ]) if ids else np.empty((0, dimension), dtype=np.float32)
            self._episode_vector_cache = ids, matrix
        return self._episode_vector_cache

    def _seed(self, state: dict, episodes: dict, sources: dict, policy: RecallPolicy) -> list[SpreadingSeed]:
        if state.get("seeds") is not None:
            return [SpreadingSeed(**s) for s in state["seeds"]]
        cues = list(dict.fromkeys([state["question"], *state["cues"], *state["needs"]]))
        self._ensure_live()
        state["metrics"]["embedding_batches"] += 1
        vectors = self.model.embed(cues)
        self._ensure_live()
        ids, matrix = self._episode_vectors(episodes)
        if not ids:
            state["seeds"] = []
            state["fallback_episode_scores"] = {}
            state["fallback_episode_ids"] = []
            return []
        combined: dict[int, float] = {}
        all_scores = np.zeros(len(ids), dtype=np.float32)
        for vector in vectors:
            scores = matrix @ normalize_embedding(vector, self.config.model.embedding_dimension)
            all_scores = np.maximum(all_scores, scores)
            ranks = np.argsort(-scores, kind="stable")[:policy.seeds_per_cue]
            for position in ranks:
                score = min(1.0, max(0.0, float(scores[position])))
                if score > 0:
                    combined[ids[position]] = max(combined.get(ids[position], 0), score)
        # Different question rewrites finding the same Source are one root.
        seeds = [SpreadingSeed("episode", i, score, "source:" + _hash(sources[episodes[i]["source_id"]])) for i, score in sorted(combined.items())]
        state["seeds"] = [asdict(s) for s in seeds]
        all_scores = np.clip(all_scores, 0.0, 1.0)
        state["fallback_episode_scores"] = {str(eid): float(all_scores[p]) for p, eid in enumerate(ids)}
        state["fallback_episode_ids"] = [ids[p] for p in np.argsort(-all_scores, kind="stable")]
        state["metrics"]["seed_candidates_scored"] += len(ids) * len(cues)
        return seeds

    def _expand_followups(self, state: dict, episodes: dict, sources: dict, policy: RecallPolicy) -> bool:
        texts = pending_cue_texts(state.get("cue_queue"))
        if not texts:
            return False
        self._ensure_live()
        state["metrics"]["embedding_batches"] += 1
        vectors = self.model.embed(texts)
        self._ensure_live()
        ids, matrix = self._episode_vectors(episodes)
        expanded = expand_pending_cues(
            state.get("cue_queue"), vectors, episode_ids=ids, episode_matrix=matrix,
            episodes=episodes, sources=sources,
            existing_seeds=[SpreadingSeed(**item) for item in state["seeds"]],
            seeds_per_cue=policy.seeds_per_cue,
        )
        next_state = restart_spreading_epoch(state, expanded.seeds) if expanded.changed else dict(state)
        next_state["cue_queue"] = expanded.queue
        # Keep the strongest match across initial and later cues. A weak new
        # batch must not replace the complete ranking of earlier evidence.
        prior_scores = state.get("fallback_episode_scores", {})
        combined_scores = {str(eid): max(float(prior_scores.get(str(eid), 0.0)),
                                        expanded.episode_scores.get(eid, 0.0)) for eid in ids}
        next_state["fallback_episode_scores"] = combined_scores
        next_state["fallback_episode_ids"] = sorted(ids, key=lambda eid: (-combined_scores[str(eid)], eid))
        metrics = dict(next_state["metrics"])
        metrics["followup_cues_processed"] = metrics.get("followup_cues_processed", 0) + expanded.processed_count
        metrics["seed_candidates_scored"] += expanded.scored_count
        next_state["metrics"] = metrics
        self._ensure_live()
        state.update(next_state)
        self.sessions.write(state)
        return expanded.changed

    @staticmethod
    def _shortcut_targets(edges: list, seeds: list[SpreadingSeed]) -> dict[int, list[int]]:
        seed_scores: dict[int, float] = {}
        for seed in seeds:
            if seed.node_type == "episode":
                seed_scores[seed.node_id] = max(seed_scores.get(seed.node_id, 0.0), seed.activation)
        targets: dict[int, list[tuple[float, int]]] = {}
        for row in edges:
            if row["association_mode"] != "simple_recall" or min(float(row["weight"]), float(row["confidence"])) <= 0:
                continue
            a, b = int(row["from_id"]), int(row["to_id"])
            strength = float(row["weight"]) * float(row["confidence"])
            for anchor, target in ((a, b), (b, a)):
                score = seed_scores.get(anchor, 0.0) * strength
                # A nearly suppressed shortcut cannot permanently occupy the
                # early lane. It remains available in ordinary spreading.
                if score >= 0.05:
                    targets.setdefault(target, []).append((score, int(row["id"])))
        ordered = sorted(targets, key=lambda target: (-max(v[0] for v in targets[target]), target))
        return {target: [eid for _, eid in sorted(targets[target], key=lambda v: (-v[0], v[1]))] for target in ordered}

    def _windows(self, candidates: list[int], state: dict, episodes: dict, sources: dict, policy: RecallPolicy) -> list[dict]:
        windows = []
        seen_sources = set()
        for eid in candidates:
            episode = episodes.get(eid)
            if episode is None:
                continue
            sid = int(episode["source_id"])
            if sid in seen_sources or sid not in sources:
                continue
            seen_sources.add(sid)
            text = sources[sid]
            offset = int(state["source_offsets"].get(str(sid), 0))
            if offset >= len(text):
                continue
            end = min(len(text), offset + policy.source_window_chars)
            windows.append({"source_id": sid, "episode_id": eid, "episode_hint": episode["text"], "start": offset, "end": end, "text": text[offset:end]})
            if len(windows) >= policy.sources_per_review:
                break
        return windows

    def _review(self, windows: list[dict], state: dict, sources: dict, episodes: dict) -> None:
        stage = "map"

        def call(system, payload):
            nonlocal stage
            stage = "map" if system == RECALL_MAP_SYSTEM else "verify"
            return self._call(system, payload, state)

        window_refs = [{key: w[key] for key in ("source_id", "episode_id", "start", "end")} for w in windows]
        try:
            reviewed = review_wave(windows, state, sources, episodes, call)
            cues = enqueue_followup_cues(
                state.get("cue_queue"), reviewed.followup_cues,
                visible=reviewed.visible, sources=sources, episodes=episodes,
                initial_cues=[state["question"], *state["cues"], *state["needs"]],
            )
            self._ensure_live()
        except (Exception, KeyboardInterrupt) as error:
            trace = dict(error.trace or {}) if isinstance(error, ReviewSchemaError) else {}
            trace.update({"committed": False, "stage": stage, "error_type": type(error).__name__,
                          "windows": window_refs, "wave": state["metrics"]["review_waves"] + 1})
            state.setdefault("review_trace", []).append(trace)
            raise
        trace = dict(reviewed.trace)
        trace.update({"committed": True, "windows": window_refs, "cue_validation": cues.stats,
                      "wave": state["metrics"]["review_waves"] + 1})
        # Candidate construction and both contracts are pure. Only this point
        # commits the wave, including its reading cursor and diagnostic record.
        state.update(reviewed.updates)
        state["cue_queue"] = cues.queue
        state.setdefault("review_trace", []).append(trace)
        metrics = state["metrics"]
        metrics["source_windows_reviewed"] += len(windows)
        metrics["review_waves"] += 1
        metrics["verifier_calls_skipped"] = metrics.get("verifier_calls_skipped", 0) + int(trace.get("verifier_skipped", False))
        metrics["followup_cues_accepted"] = metrics.get("followup_cues_accepted", 0) + cues.stats["accepted"]
        rechecked = set(state.get("context_rechecked_fact_ids", []))
        for window in windows:
            rechecked.update(window.get("recheck_fact_ids", []))
        state["context_rechecked_fact_ids"] = sorted(rechecked)

    def _context_windows(self, state: dict, sources: dict, episodes: dict, policy: RecallPolicy) -> list[dict]:
        """One bounded expanded reread per suspended claim; never rewind cursors."""
        attempted = set(state.get("context_rechecked_fact_ids", []))
        by_source = {}
        for fact in state.get("pending_facts", []):
            fact_id = stable_fact_id(fact)
            if fact_id in attempted:
                continue
            sid, eid = fact["source_id"], fact["episode_id"]
            raw = sources[sid]
            start = max(0, fact["start"] - policy.source_window_chars)
            end = min(len(raw), fact["end"] + policy.source_window_chars)
            key = sid, start, end
            if key in by_source:
                by_source[key]["recheck_fact_ids"].append(fact_id)
                by_source[key]["episode_ids"] = sorted(set(by_source[key]["episode_ids"]) | {eid})
                continue
            if len(by_source) >= policy.sources_per_review:
                break
            by_source[key] = {
                "source_id": sid, "episode_id": eid, "episode_ids": [eid], "episode_hint": episodes[eid]["text"],
                "start": start, "end": end, "text": raw[start:end], "recheck": True,
                "recheck_fact_ids": [fact_id],
            }
        return list(by_source.values())

    def query(self, question: str, *, mode: str | None = None, context: str = "", resume: str | None = None, timeout_seconds: float | None = None, learn: bool = True, cancelled: Callable[[], bool] | None = None, max_waves: int | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is required")
        if max_waves is not None and (type(max_waves) is not int or max_waves <= 0):
            raise ValueError("max_waves must be a positive integer")
        policy = RecallPolicy.for_request(question, mode, timeout_seconds=timeout_seconds, learn=learn)
        started = self.clock()
        self._deadline, self._cancelled = started + policy.timeout_seconds, cancelled
        sources, episodes, rows, fingerprint = self._snapshot()
        request_hash = _hash({"question": question, "context": context, "database": str(self.config.database_path.resolve())})
        if resume:
            state = self.sessions.read(resume)
            if state.get("version") != self.protocol_version:
                raise ValueError(f"recall checkpoint uses another review protocol; start a fresh recall (version {self.protocol_version})")
            if state.get("request_hash") != request_hash or (state.get("snapshot_hash") != fingerprint and not self._recover_committed_learning(state, refresh=False)):
                raise ValueError("recall checkpoint belongs to another request or changed knowledge; start a fresh recall")
        else:
            state = {
                "version": self.protocol_version, "session_id": uuid4().hex, "request_hash": request_hash,
                "snapshot_hash": fingerprint, "question": question, "context": context,
                "needs": [], "cues": [], "facts": [], "pending_facts": [],
                "verified_links": [], "pending_links": [], "evidence_contexts": {},
                "covered_needs": [], "resolved_needs": [], "need_assessments": [],
                "source_offsets": {}, "spreading": None, "spreading_expansion_base": 0,
                "cue_queue": None, "context_rechecked_fact_ids": [], "review_trace": [],
                "used_edge_ids": [], "fact_edge_ids": {}, "elapsed_seconds": 0.0,
                "metrics": {"model_batches": 0, "embedding_batches": 0,
                            "seed_candidates_scored": 0, "source_windows_reviewed": 0,
                            "review_waves": 0, "edge_expansions": 0, "shortcut_candidates": 0,
                            "verifier_calls_skipped": 0, "followup_cues_accepted": 0,
                            "followup_cues_processed": 0, "spreading_restarts": 0},
                "learning": {"status": "not_attempted"},
            }
        state["policy"] = asdict(policy)
        state["status"] = "running"
        state.pop("error", None)
        self.sessions.write(state)
        deadline_scope = self.model.deadline_budget(self._deadline) if hasattr(self.model, "deadline_budget") else nullcontext()
        try:
            with deadline_scope:
                self._ensure_live()
                if not state["needs"]:
                    planned = self._call(self.plan_system, {"question": question, "context": context}, state)
                    needs = planned.get("needs")
                    if not isinstance(needs, list) or len(needs) > policy.max_needs or any(not isinstance(n, str) or not n.strip() for n in needs):
                        raise ValueError("recall plan has too many or invalid requirements; narrow the question")
                    state["needs"] = _strings(needs, policy.max_needs)
                    state["cues"] = _strings(planned.get("cues"), policy.max_cues)
                    if not state["needs"]:
                        raise ValueError("recall plan has no requirements")
                    self.sessions.write(state)
                seeds = self._seed(state, episodes, sources, policy)
                graph = SpreadingRecall(self._nodes, [SpreadingEdge(r["id"], r["from_type"], r["from_id"], r["to_type"], r["to_id"], float(r["weight"]), float(r["confidence"])) for r in rows], monotonic=self.clock)
                shortcuts = self._shortcut_targets(rows, seeds)
                state["metrics"]["shortcut_candidates"] = len(shortcuts)
                wave_count = 0
                spread = None
                while not self._resolved(state):
                    self._ensure_live()
                    if max_waves is not None and wave_count >= max_waves:
                        state["status"] = "wave_budget"
                        break
                    if self._expand_followups(state, episodes, sources, policy):
                        seeds = [SpreadingSeed(**item) for item in state["seeds"]]
                        shortcuts = self._shortcut_targets(rows, seeds)
                        state["metrics"]["shortcut_candidates"] = len(shortcuts)
                        spread = None
                    # Reuse may deliver the needed source before broad graph
                    # expansion. It still undergoes current-question review.
                    context_windows = self._context_windows(state, sources, episodes, policy)
                    shortcut_windows = self._windows(list(shortcuts), state, episodes, sources, policy)
                    if context_windows:
                        windows = context_windows
                    elif shortcut_windows:
                        windows = shortcut_windows
                    else:
                        spread = graph.search(seeds, max_expansions=policy.wave_expansions, max_seconds=max(0.0, self._deadline - self.clock()), checkpoint=state["spreading"])
                        state["spreading"] = spread.checkpoint
                        state["metrics"]["edge_expansions"] = state.get("spreading_expansion_base", 0) + spread.expansions
                        ranked = [n.node_id for n in spread.nodes if n.node_type == "episode"]
                        # Disconnected memories stay available through the
                        # complete vector ranking after the reachable graph.
                        if spread.status == "complete":
                            ranked = list(dict.fromkeys([*ranked, *state.get("fallback_episode_ids", [])]))
                        windows = self._windows(ranked, state, episodes, sources, policy)
                        if not windows:
                            if spread.status == "complete":
                                state["status"] = "search_exhausted"
                                break
                            self.sessions.write(state)
                            continue
                    self._review(windows, state, sources, episodes)
                    fact_edges = state.setdefault("fact_edge_ids", {})
                    for fact in state["facts"]:
                        key = stable_fact_id(fact)
                        if key in fact_edges:
                            continue
                        paths = set(shortcuts.get(fact["episode_id"], []))
                        if spread is not None:
                            for node in spread.nodes:
                                if node.node_type == "episode" and node.node_id == fact["episode_id"]:
                                    for path in node.paths.values():
                                        paths.update(path)
                        fact_edges[key] = sorted(paths)
                    used = {edge_id for f in state["facts"] for edge_id in fact_edges.get(stable_fact_id(f), [])}
                    state["used_edge_ids"] = sorted(used)
                    wave_count += 1
                    self.sessions.write(state)
                if self._resolved(state):
                    self._validate_delivered_sources(state)
                    state["status"] = "complete"
                    if learn and state["verified_links"] and state["learning"]["status"] != "applied":
                        self._ensure_live()
                        links = self._learning_links(state)
                        with transaction_liveness(self._ensure_live):
                            # A route to a true fact is not proof of every
                            # intermediate relation. Only reviewed links learn
                            # automatically; exploration paths remain diagnostic.
                            receipt = self.feedback.learn_verified("recall:" + state["session_id"], links)
                        state["learning"] = {"status": "applied", "association_ids": list(receipt.association_ids)}
                    elif not learn:
                        state["learning"] = {"status": "disabled"}
                    elif not state["verified_links"]:
                        state["learning"] = {"status": "no_verified_relation"}
        except (TimeoutError, InterruptedError, KeyboardInterrupt) as exc:
            state["status"] = "time_budget" if isinstance(exc, TimeoutError) else "cancelled"
        except Exception as exc:
            state["status"] = "technical_error"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)[:600]}
        finally:
            try:
                self._validate_delivered_sources(state)
                # A successful SQLite commit can precede an interrupt before
                # the JSON checkpoint is written. Recover its exact receipt;
                # never repeat strengthening or report a false resume promise.
                self._recover_committed_learning(state, refresh=True)
            except ValueError as exc:
                state["status"] = "source_changed"
                state["facts"], state["covered_needs"], state["used_edge_ids"] = [], [], []
                state["resolved_needs"], state["pending_facts"] = [], []
                state["verified_links"], state["pending_links"] = [], []
                state["evidence_contexts"], state["need_assessments"] = {}, []
                state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            valid_ids = self.feedback.valid_learned_edge_ids()
            state["feedback_edge_ids"] = sorted((set(state["used_edge_ids"]) | set(state.get("learning", {}).get("association_ids", []))) & valid_ids)
            state["elapsed_seconds"] += max(0.0, self.clock() - started)
            self.sessions.write(state)
        missing = [n for i, n in enumerate(state["needs"]) if i not in state["resolved_needs"]] or ([question] if not state["needs"] else [])
        claims = list(dict.fromkeys(f["claim"] for f in state["facts"]))
        return {"session_id": state["session_id"], "mode": policy.mode, "status": state["status"], "complete": state["status"] == "complete", "timeout_seconds": policy.timeout_seconds, "elapsed_seconds": state["elapsed_seconds"], "answer": "\n".join(claims), "needs": state["needs"], "missing_requirements": missing, "evidence": state["facts"], "used_edge_ids": state["used_edge_ids"], "feedback_edge_ids": state["feedback_edge_ids"], "learning": state["learning"], "metrics": state["metrics"], "checkpoint_path": str(self.sessions._path(state["session_id"]).resolve()), "resumable": state["status"] in {"time_budget", "cancelled", "wave_budget", "technical_error"}, "error": state.get("error"), "coverage_basis": "model_plan_and_independent_source_review_v2", "review_protocol_version": 2, "need_assessments": state["need_assessments"], "pending_evidence_count": len(state["pending_facts"]), "answer_status": "complete" if state["status"] == "complete" else "partial" if state["facts"] else "unknown"}

    def _validate_delivered_sources(self, state: dict) -> None:
        with self.db.connection() as connection:
            for fact in [*state["facts"], *state.get("pending_facts", [])]:
                row = connection.execute("SELECT s.raw_text,e.source_id FROM episode e JOIN source s ON s.id=e.source_id WHERE e.id=?", (fact["episode_id"],)).fetchone()
                if row is None or int(row[1]) != fact["source_id"] or _hash(str(row[0])) != fact["source_sha256"] or str(row[0])[fact["start"]:fact["end"]] != fact["quote"]:
                    raise ValueError("delivered Source or Episode binding changed since recall")

    def apply_feedback(self, session_id: str, *, positive: bool, feedback_id: str) -> dict:
        if not isinstance(positive, bool):
            raise ValueError("positive must be a boolean")
        if not isinstance(feedback_id, str) or not feedback_id.strip() or len(feedback_id.strip()) > 256:
            raise ValueError("feedback_id must be a nonempty string of at most 256 characters")
        state = self.sessions.read(session_id)
        expected = _hash({"question": state["question"], "context": state["context"], "database": str(self.config.database_path.resolve())})
        if state.get("request_hash") != expected:
            raise ValueError("feedback session belongs to another database")
        if not state.get("facts"):
            raise ValueError("session has no verified delivered evidence")
        # Feedback addresses the delivered result, never every visited edge.
        self._validate_delivered_sources(state)
        ids = state.get("feedback_edge_ids", [])
        if not ids:
            return {"feedback_id": feedback_id.strip(), "applied": False,
                    "changes": [], "reason": "no_verified_association"}
        receipt = self.feedback.apply_user_feedback(feedback_id, ids, positive=positive)
        return {"feedback_id": receipt.feedback_id, "applied": receipt.applied, "changes": [asdict(c) for c in receipt.changes]}
