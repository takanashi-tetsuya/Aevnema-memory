"""Atomic, source-bound review transitions for progressive recall.

This module does not call a provider, mutate its inputs, or write a database.
The caller owns deadlines, accounting, checkpointing, and learning.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

from memory_demo.llm.recall_prompts import RECALL_MAP_SYSTEM, RECALL_VERIFY_SYSTEM


class ReviewSchemaError(ValueError):
    def __init__(self, message: str, *, trace: dict | None = None):
        super().__init__(message)
        self.trace = deepcopy(trace or {})


@dataclass(frozen=True)
class ReviewResult:
    updates: dict
    trace: dict
    followup_cues: list
    visible: list


def _hash(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_fact_id(fact: dict) -> str:
    """A claim has stable identity independently of need/context annotations."""
    identity = {key: fact[key] for key in (
        "source_id", "episode_id", "source_sha256", "start", "end", "claim"
    )}
    identity["claim"] = identity["claim"].strip()
    return "f_" + _hash(identity)


def _keys(value, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - optional:
        raise ReviewSchemaError(f"{label} has missing or unexpected fields")
    return value


def _list(value, label: str) -> list:
    if not isinstance(value, list):
        raise ReviewSchemaError(f"{label} must be a list")
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewSchemaError(f"{label} must be a nonempty string")
    return value.strip()


def _integer(value, label: str) -> int:
    if type(value) is not int:
        raise ReviewSchemaError(f"{label} must be an integer")
    return value


def _indices(values, size: int, label: str) -> list[int]:
    values = _list(values, label)
    if any(type(v) is not int or not 0 <= v < size for v in values):
        raise ReviewSchemaError(f"{label} has a non-integer or out-of-range index")
    if len(values) != len(set(values)):
        raise ReviewSchemaError(f"{label} has duplicate indices")
    return sorted(values)


def _refs(values, allowed: set[str], label: str) -> list[str]:
    values = _list(values, label)
    if any(not isinstance(v, str) or v not in allowed for v in values):
        raise ReviewSchemaError(f"{label} contains an unknown ID")
    if len(values) != len(set(values)):
        raise ReviewSchemaError(f"{label} has duplicate IDs")
    return list(values)


def _context_id(record: dict) -> str:
    return "c_" + _hash({k: record[k] for k in ("source_id", "source_sha256", "start", "end")})


def _register_context(registry: dict, record: dict, sources: dict, episodes: dict) -> str:
    sid, start, end = (record.get(k) for k in ("source_id", "start", "end"))
    if type(sid) is not int or sid not in sources or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sources[sid]):
        raise ReviewSchemaError("invalid Source context range")
    digest = _hash(sources[sid])
    if record.get("source_sha256", digest) != digest:
        raise ReviewSchemaError("Source context changed since its review")
    eids = record.get("episode_ids", [record.get("episode_id")])
    if not isinstance(eids, list) or not eids or any(type(e) is not int or e not in episodes or int(episodes[e]["source_id"]) != sid for e in eids):
        raise ReviewSchemaError("Source context has an invalid Episode binding")
    clean = {"source_id": sid, "source_sha256": digest, "start": start, "end": end, "episode_ids": sorted(set(eids))}
    cid = _context_id(clean)
    if cid in registry:
        clean["episode_ids"] = sorted(set(clean["episode_ids"]) | set(registry[cid]["episode_ids"]))
    registry[cid] = clean
    return cid


def _visible_context(record: dict, sources: dict, episodes: dict) -> dict:
    eid = record["episode_ids"][0]
    return {**deepcopy(record), "context_id": _context_id(record), "episode_id": eid,
            "episode_hint": str(episodes[eid].get("text", "")),
            "episode_hints": [{"episode_id": e, "text": str(episodes[e].get("text", ""))} for e in record["episode_ids"]],
            "text": sources[record["source_id"]][record["start"]:record["end"]]}


def _stored_fact(item: dict, registry: dict, sources: dict, episodes: dict, need_count: int) -> dict:
    if not isinstance(item, dict):
        raise ReviewSchemaError("stored fact must be an object")
    fact = {k: deepcopy(item.get(k)) for k in ("source_id", "episode_id", "source_sha256", "start", "end", "quote", "claim", "need_indices")}
    sid, eid, start, end = (fact[k] for k in ("source_id", "episode_id", "start", "end"))
    if type(sid) is not int or sid not in sources or type(eid) is not int or eid not in episodes or int(episodes[eid]["source_id"]) != sid:
        raise ReviewSchemaError("stored fact Source/Episode binding changed")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sources[sid]) or sources[sid][start:end] != fact["quote"] or _hash(sources[sid]) != fact["source_sha256"]:
        raise ReviewSchemaError("stored fact Source span changed")
    fact["claim"] = _text(fact["claim"], "stored claim")
    fact["need_indices"] = _indices(fact["need_indices"], need_count, "stored need_indices")
    fact["fact_id"] = stable_fact_id(fact)
    if item.get("fact_id", fact["fact_id"]) != fact["fact_id"]:
        raise ReviewSchemaError("stored fact identity mismatch")
    context_ids = item.get("context_ids", [])
    if not isinstance(context_ids, list):
        raise ReviewSchemaError("stored context_ids must be a list")
    if not context_ids:
        # Version-one checkpoints are rejected by the caller. This fallback
        # supports direct callers with valid literal evidence but no registry.
        lo, hi = (0, len(sources[sid])) if len(sources[sid]) <= 6000 else (max(0, start - 400), min(len(sources[sid]), end + 400))
        context_ids = [_register_context(registry, {"source_id": sid, "start": lo, "end": hi, "episode_ids": [eid]}, sources, episodes)]
    for cid in context_ids:
        record = registry.get(cid)
        if not record or record["source_id"] != sid or eid not in record["episode_ids"] or not record["start"] <= start < end <= record["end"]:
            raise ReviewSchemaError("stored fact lost its supporting context")
    fact["context_ids"] = sorted(set(context_ids))
    return fact


def _validate_mapper(value: dict, need_count: int) -> dict:
    _keys(value, {"facts", "links"}, {"followup_cues"}, "mapper response")
    for fact in _list(value["facts"], "mapper facts"):
        _keys(fact, {"source_id", "episode_id", "quote", "claim", "need_indices"}, set(), "mapper fact")
        _integer(fact["source_id"], "source_id")
        _integer(fact["episode_id"], "episode_id")
        _text(fact["quote"], "quote")
        _text(fact["claim"], "claim")
        _indices(fact["need_indices"], need_count, "need_indices")
    for link in _list(value["links"], "mapper links"):
        _keys(link, {"from_episode_id", "to_episode_id", "rationale"}, set(), "mapper link")
        _integer(link["from_episode_id"], "from_episode_id")
        _integer(link["to_episode_id"], "to_episode_id")
        _text(link["rationale"], "rationale")
    for cue in _list(value.get("followup_cues", []), "followup_cues"):
        _keys(cue, {"cue", "source_id", "episode_id", "quote"}, set(), "followup cue")
        _text(cue["cue"], "followup cue")
        _text(cue["quote"], "followup quote")
        _integer(cue["source_id"], "followup source_id")
        _integer(cue["episode_id"], "followup episode_id")
    return value


def _decisions(rows, ids: set[str], field: str, all_fact_ids: set[str]) -> dict:
    result = {}
    for row in _list(rows, field + " decisions"):
        _keys(row, {field, "decision", "reason"}, {"basis_fact_ids"}, field + " decision")
        identifier = row[field]
        if not isinstance(identifier, str) or identifier not in ids or identifier in result:
            raise ReviewSchemaError(field + " decisions have unknown or duplicate IDs")
        if not isinstance(row["decision"], str) or row["decision"] not in {"accept", "reject", "needs_context"}:
            raise ReviewSchemaError("invalid decision status")
        reason = _text(row["reason"], "decision reason")
        basis = _refs(row.get("basis_fact_ids", []), all_fact_ids, "basis_fact_ids")
        result[identifier] = {field: identifier, "decision": row["decision"], "reason": reason, "basis_fact_ids": basis}
    if set(result) != ids:
        raise ReviewSchemaError(field + " decisions do not cover every candidate")
    return result


def _need_decisions(rows, needs: list, accepted: dict) -> list:
    result = {}
    for row in _list(rows, "need_decisions"):
        _keys(row, {"need_index", "status", "fact_ids", "reason"}, set(), "need decision")
        index = _integer(row["need_index"], "need_index")
        if not 0 <= index < len(needs) or index in result:
            raise ReviewSchemaError("need decisions have duplicate or out-of-range indices")
        if not isinstance(row["status"], str) or row["status"] not in {"supported", "partial", "unknown", "refuted"}:
            raise ReviewSchemaError("invalid need status")
        refs = _refs(row["fact_ids"], set(accepted), "need fact_ids")
        if row["status"] in {"supported", "refuted"} and (not refs or any(index not in accepted[fid]["need_indices"] for fid in refs)):
            raise ReviewSchemaError("resolved need must cite accepted facts mapped to that need")
        result[index] = {"need_index": index, "status": row["status"], "fact_ids": refs, "reason": _text(row["reason"], "need reason")}
    if set(result) != set(range(len(needs))):
        raise ReviewSchemaError("need decisions do not cover every requirement")
    return [result[i] for i in range(len(needs))]


def review_wave(windows: list[dict], state: dict, sources: dict, episodes: dict,
                call: Callable[[str, dict], dict]) -> ReviewResult:
    """Return a complete review commit, or raise without changing any input."""
    trace = {"version": 2, "schema_ok": False, "stage": "contexts", "verifier_skipped": False,
             "source_windows": [], "local_rejections": [], "fact_transitions": [], "link_transitions": []}
    try:
        return _review_wave(deepcopy(windows), deepcopy(state), sources, episodes, call, trace)
    except ReviewSchemaError as exc:
        trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
        exc.trace = deepcopy(trace)
        raise


def _review_wave(windows, state, sources, episodes, call, trace) -> ReviewResult:
    needs = state["needs"]
    registry = {}
    for cid, record in state.get("evidence_contexts", {}).items():
        actual = _register_context(registry, record, sources, episodes)
        if cid != actual:
            raise ReviewSchemaError("context identity mismatch")
    candidates = {}
    prior_status = {}

    def merge(fact):
        fid = fact["fact_id"]
        if fid in candidates:
            old = candidates[fid]
            old["need_indices"] = sorted(set(old["need_indices"]) | set(fact["need_indices"]))
            old["context_ids"] = sorted(set(old["context_ids"]) | set(fact["context_ids"]))
        else:
            candidates[fid] = deepcopy(fact)

    for status, field in (("accept", "facts"), ("needs_context", "pending_facts")):
        for item in state.get(field, []):
            fact = _stored_fact(item, registry, sources, episodes, len(needs))
            merge(fact)
            prior_status[fact["fact_id"]] = status
    visible_ids = list(dict.fromkeys(cid for f in candidates.values() for cid in f["context_ids"]))
    for window in windows:
        cid = _register_context(registry, window, sources, episodes)
        record = registry[cid]
        if window.get("text") != sources[record["source_id"]][record["start"]:record["end"]]:
            raise ReviewSchemaError("window text does not match the original Source")
        visible_ids.append(cid)
        trace["source_windows"].append({**deepcopy(record), "context_id": cid, "recheck": bool(window.get("recheck", False))})
    visible = [_visible_context(registry[cid], sources, episodes) for cid in dict.fromkeys(visible_ids)]
    # Rechecking an existing claim can depend on newly expanded context even
    # when the mapper correctly avoids proposing that same claim again.
    for fact in candidates.values():
        supporting = {w["context_id"] for w in visible if w["source_id"] == fact["source_id"]
                      and fact["episode_id"] in w["episode_ids"]
                      and w["start"] <= fact["start"] < fact["end"] <= w["end"]}
        fact["context_ids"] = sorted(set(fact["context_ids"]) | supporting)
    common = {"question": state["question"], "context": state.get("context", ""), "needs": needs, "sources": visible}
    trace["stage"] = "mapper_schema"
    proposal = _validate_mapper(call(RECALL_MAP_SYSTEM, deepcopy({**common, "existing_facts": list(candidates.values())})), len(needs))
    for index, item in enumerate(proposal["facts"]):
        sid, eid, quote = item["source_id"], item["episode_id"], item["quote"]
        matching = [w for w in visible if w["source_id"] == sid and eid in w["episode_ids"]]
        reason = None
        if eid not in episodes or int(episodes[eid]["source_id"]) != sid or not matching:
            reason = "source_episode_not_visible"
        spans = set()
        for window in matching:
            position = window["text"].find(quote)
            while position >= 0:
                spans.add((window["start"] + position, window["start"] + position + len(quote)))
                # A narrow overlap must not hide a second location visible in
                # a wider context. Include overlapping literal occurrences.
                position = window["text"].find(quote, position + 1)
        if reason is None and len(spans) != 1:
            reason = "quote_missing_or_ambiguous"
        if reason:
            trace["local_rejections"].append({"kind": "fact", "proposal_index": index, "source_id": sid, "episode_id": eid, "reason": reason})
            continue
        start, end = next(iter(spans))
        if sources[sid][start:end] != quote:
            raise ReviewSchemaError("validated quote does not match its absolute Source span")
        contexts = [w["context_id"] for w in matching if w["start"] <= start < end <= w["end"]]
        fact = {"source_id": sid, "episode_id": eid, "source_sha256": _hash(sources[sid]), "start": start, "end": end,
                "quote": quote, "claim": item["claim"].strip(), "need_indices": sorted(item["need_indices"]), "context_ids": sorted(set(contexts))}
        fact["fact_id"] = stable_fact_id(fact)
        merge(fact)

    links = {}
    old_link_status = {}
    for status, values in (("accept", state.get("verified_links", [])), ("needs_context", state.get("pending_links", [])), ("new", proposal["links"])):
        for item in values:
            a, b = item["from_episode_id"], item["to_episode_id"]
            if a == b or not {a, b}.issubset({f["episode_id"] for f in candidates.values()}):
                trace["local_rejections"].append({"kind": "link", "from_episode_id": a, "to_episode_id": b, "reason": "missing_distinct_endpoint_facts"})
                continue
            a, b = sorted((a, b))
            lid = "l_" + _hash([a, b])
            links[lid] = {"link_id": lid, "from_episode_id": a, "to_episode_id": b, "rationale": item["rationale"].strip()}
            if status != "new":
                old_link_status[lid] = status
    trace["candidate_counts"] = {"facts": len(candidates), "links": len(links), "mapper_facts": len(proposal["facts"]),
                                 "prior_facts": len(prior_status), "visible_contexts": len(visible)}
    trace["candidate_facts"] = [{k: deepcopy(f[k]) for k in ("fact_id", "claim", "source_id", "episode_id", "source_sha256", "start", "end", "context_ids", "need_indices")} for f in candidates.values()]
    trace["visible_contexts"] = [{k: deepcopy(w[k]) for k in ("context_id", "source_id", "source_sha256", "start", "end", "episode_ids")} for w in visible]
    trace["stage"] = "verifier_schema"
    if candidates or links:
        checked = call(RECALL_VERIFY_SYSTEM, deepcopy({**common, "proposal": {"facts": list(candidates.values()), "links": list(links.values())}}))
        _keys(checked, {"fact_decisions", "link_decisions", "need_decisions"}, set(), "verifier response")
        fact_decisions = _decisions(checked["fact_decisions"], set(candidates), "fact_id", set(candidates))
        link_decisions = _decisions(checked["link_decisions"], set(links), "link_id", set(candidates))
        accepted = {fid: f for fid, f in candidates.items() if fact_decisions[fid]["decision"] == "accept"}
        assessments = _need_decisions(checked["need_decisions"], needs, accepted)
    else:
        trace["verifier_skipped"] = True
        fact_decisions, link_decisions, accepted = {}, {}, {}
        assessments = [{"need_index": i, "status": "unknown", "fact_ids": [], "reason": "No admissible source-backed fact candidates in this wave."} for i in range(len(needs))]
    pending = {fid: f for fid, f in candidates.items() if fact_decisions[fid]["decision"] == "needs_context"}
    verified_links, pending_links = [], []
    for lid, link in links.items():
        decision = link_decisions[lid]
        if decision["decision"] == "accept":
            evidence = [f for f in accepted.values() if f["episode_id"] in {link["from_episode_id"], link["to_episode_id"]}]
            if {f["episode_id"] for f in evidence} != {link["from_episode_id"], link["to_episode_id"]}:
                raise ReviewSchemaError("accepted link lacks accepted evidence at both endpoints")
            verified_links.append({**link, "evidence": deepcopy(evidence)})
        elif decision["decision"] == "needs_context":
            pending_links.append(deepcopy(link))
        trace["link_transitions"].append({**decision, "from": old_link_status.get(lid, "new"), "to": decision["decision"],
                                          "from_episode_id": link["from_episode_id"], "to_episode_id": link["to_episode_id"]})
    for fid, fact in candidates.items():
        trace["fact_transitions"].append({**fact_decisions[fid], "from": prior_status.get(fid, "new"), "to": fact_decisions[fid]["decision"],
            "claim": fact["claim"], "source_id": fact["source_id"], "episode_id": fact["episode_id"], "source_sha256": fact["source_sha256"],
            "start": fact["start"], "end": fact["end"], "context_ids": list(fact["context_ids"]), "need_indices": list(fact["need_indices"])})
    covered = [a["need_index"] for a in assessments if a["status"] == "supported"]
    resolved = [a["need_index"] for a in assessments if a["status"] in {"supported", "refuted"}]
    trace["need_assessments"] = deepcopy(assessments)
    trace["coverage_change"] = {"before": list(state.get("covered_needs", [])), "after": covered,
                                "resolved_before": list(state.get("resolved_needs", state.get("covered_needs", []))), "resolved_after": resolved}
    offsets = deepcopy(state.get("source_offsets", {}))
    for window in windows:
        if window.get("recheck", False):
            continue
        sid = window["source_id"]
        end = window["end"]
        next_offset = end if end == len(sources[sid]) else max(window["start"] + 1, end - 400)
        offsets[str(sid)] = max(int(offsets.get(str(sid), 0)), next_offset)
    live_contexts = {cid for f in [*accepted.values(), *pending.values()] for cid in f["context_ids"]}
    updates = {"facts": list(accepted.values()), "pending_facts": list(pending.values()), "verified_links": verified_links, "pending_links": pending_links,
               "covered_needs": covered, "resolved_needs": resolved, "need_assessments": assessments,
               "evidence_contexts": {cid: deepcopy(registry[cid]) for cid in sorted(live_contexts)}, "source_offsets": offsets}
    trace["schema_ok"], trace["stage"] = True, "committed_update_ready"
    return ReviewResult(deepcopy(updates), deepcopy(trace), deepcopy(proposal.get("followup_cues", [])), deepcopy(visible))
