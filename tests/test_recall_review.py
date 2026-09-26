"""Local contract tests for atomic recall reviews; no provider or database."""
from __future__ import annotations

from copy import deepcopy
import unittest

from memory_demo.llm.recall_prompts import RECALL_MAP_SYSTEM, RECALL_VERIFY_SYSTEM
from memory_demo.retrieval.recall_review import ReviewSchemaError, review_wave, stable_fact_id


class RecallReviewTests(unittest.TestCase):
    def setUp(self):
        self.sources = {1: "Header: Alice's request.\nAlice accepted the request.\nEnd of record.",
                        2: "Bob delivered the parcel.", 3: "Correction: Alice did not accept the request."}
        self.episodes = {i: {"id": i, "source_id": i, "text": f"hint {i}"} for i in self.sources}
        self.state = {"question": "Was the request accepted and the parcel delivered?", "context": "",
                      "needs": ["acceptance", "delivery"], "facts": [], "pending_facts": [],
                      "verified_links": [], "pending_links": [], "covered_needs": [], "resolved_needs": [],
                      "source_offsets": {}, "evidence_contexts": {}}
        self.calls = []

    def window(self, sid, **extras):
        return {"source_id": sid, "episode_id": sid, "episode_hint": self.episodes[sid]["text"],
                "start": 0, "end": len(self.sources[sid]), "text": self.sources[sid], **extras}

    def fact(self, sid=1, *, claim=None, needs=None):
        quote = "Alice accepted the request." if sid == 1 else self.sources[sid]
        return {"source_id": sid, "episode_id": sid, "quote": quote, "claim": claim or quote,
                "need_indices": ([0] if sid != 2 else [1]) if needs is None else needs}

    def accepted(self, payload):
        facts = payload["proposal"]["facts"]
        return {"fact_decisions": [{"fact_id": f["fact_id"], "decision": "accept", "reason": "Exact original evidence supports this narrow claim."} for f in facts],
                "link_decisions": [{"link_id": link["link_id"], "decision": "accept", "reason": "Both original sources support this association."} for link in payload["proposal"]["links"]],
                "need_decisions": [{"need_index": i, "status": "supported" if any(i in f["need_indices"] for f in facts) else "unknown",
                                    "fact_ids": [f["fact_id"] for f in facts if i in f["need_indices"]],
                                    "reason": "Supported by cited evidence." if any(i in f["need_indices"] for f in facts) else "No evidence for this requirement yet."} for i in range(2)]}

    def run_wave(self, mapper, verifier=None, *, state=None, windows=None):
        def call(system, payload):
            self.calls.append((system, deepcopy(payload)))
            if system == RECALL_MAP_SYSTEM:
                return deepcopy(mapper)
            self.assertEqual(system, RECALL_VERIFY_SYSTEM)
            return (verifier or self.accepted)(payload)
        return review_wave(windows or [self.window(1)], self.state if state is None else state,
                           self.sources, self.episodes, call)

    def first_state(self):
        result = self.run_wave({"facts": [self.fact()], "links": []})
        return {**deepcopy(self.state), **result.updates}

    def test_success_returns_atomic_updates_and_preserves_original_context(self):
        original = deepcopy((self.state, self.sources, self.episodes))
        result = self.run_wave({"facts": [self.fact()], "links": []})
        self.assertEqual(original, (self.state, self.sources, self.episodes))
        self.assertEqual(result.updates["covered_needs"], [0])
        self.assertEqual(result.updates["resolved_needs"], [0])
        self.assertTrue(result.trace["schema_ok"])
        self.assertEqual(len(result.trace["fact_transitions"]), 1)
        self.assertTrue(result.updates["facts"][0]["fact_id"].startswith("f_"))
        context = next(iter(result.updates["evidence_contexts"].values()))
        self.assertEqual((context["start"], context["end"]), (0, len(self.sources[1])))
        self.assertNotIn("text", context)
        continued = {**self.state, **result.updates}
        self.run_wave({"facts": [], "links": []}, state=continued, windows=[self.window(2)])
        old = next(w for w in self.calls[-1][1]["sources"] if w["source_id"] == 1)
        self.assertEqual(old["text"], self.sources[1])
        self.assertEqual(old["episode_hint"], "hint 1")

    def test_malformed_mapper_fails_before_verifier_and_does_not_mutate(self):
        cases = [{}, {"facts": [], "links": [], "covered_needs": []},
                 {"facts": "bad", "links": []}, {"facts": [{**self.fact(), "need_indices": [True]}], "links": []},
                 {"facts": [], "links": [], "followup_cues": ["unbound guess"]}]
        for payload in cases:
            with self.subTest(payload=payload):
                before = deepcopy(self.state)
                self.calls.clear()
                with self.assertRaises(ReviewSchemaError) as caught:
                    self.run_wave(payload)
                self.assertEqual(before, self.state)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(caught.exception.trace["stage"], "mapper_schema")
                self.assertFalse(caught.exception.trace["schema_ok"])

    def test_missing_invalid_duplicate_or_unknown_fact_decision_is_atomic_error(self):
        state = self.first_state()
        def invalid(payload, case):
            response = self.accepted(payload)
            if case == "missing":
                response.pop("fact_decisions")
            elif case == "omitted":
                response["fact_decisions"] = []
            elif case == "duplicate":
                response["fact_decisions"] *= 2
            elif case == "unknown":
                response["fact_decisions"][0]["fact_id"] = "invented"
            elif case == "reason":
                response["fact_decisions"][0]["reason"] = " "
            else:
                response["fact_decisions"][0]["decision"] = {}
            return response
        for case in ("missing", "omitted", "duplicate", "unknown", "reason", "status"):
            with self.subTest(case=case):
                before = deepcopy(state)
                with self.assertRaises(ReviewSchemaError) as caught:
                    self.run_wave({"facts": [], "links": []}, lambda p: invalid(p, case), state=state, windows=[self.window(2)])
                self.assertEqual(state, before)
                self.assertEqual(caught.exception.trace["stage"], "verifier_schema")
                self.assertEqual(caught.exception.trace["candidate_facts"][0]["fact_id"], state["facts"][0]["fact_id"])

    def test_needs_context_stays_pending_and_can_be_reaccepted_next_wave(self):
        state = self.first_state()
        def pending(payload):
            response = self.accepted(payload)
            response["fact_decisions"][0].update(decision="needs_context", reason="Need the surrounding record to resolve attribution.")
            for need in response["need_decisions"]:
                need.update(status="unknown", fact_ids=[], reason="Attribution is not yet resolved.")
            return response
        result = self.run_wave({"facts": [], "links": []}, pending, state=state, windows=[self.window(2)])
        self.assertEqual(result.updates["facts"], [])
        self.assertEqual(len(result.updates["pending_facts"]), 1)
        self.assertEqual(result.updates["resolved_needs"], [])
        continued = {**state, **result.updates}
        restored = self.run_wave({"facts": [], "links": []}, state=continued, windows=[self.window(3)])
        self.assertEqual(restored.updates["pending_facts"], [])
        self.assertEqual(restored.updates["facts"][0]["fact_id"], state["facts"][0]["fact_id"])
        self.assertEqual(restored.trace["fact_transitions"][0]["from"], "needs_context")

    def test_explicit_counterevidence_retracts_old_fact_and_resolves_by_refutation(self):
        state = self.first_state()
        old_id = state["facts"][0]["fact_id"]
        def reject(payload):
            response = self.accepted(payload)
            replacement = next(f for f in payload["proposal"]["facts"] if f["source_id"] == 3)
            for decision in response["fact_decisions"]:
                if decision["fact_id"] == old_id:
                    decision.update(decision="reject", reason="Source 3 explicitly corrects the earlier account.", basis_fact_ids=[replacement["fact_id"]])
            response["need_decisions"][0].update(status="refuted", fact_ids=[replacement["fact_id"]], reason="The correction explicitly denies acceptance.")
            return response
        result = self.run_wave({"facts": [self.fact(3)], "links": []}, reject, state=state, windows=[self.window(3)])
        self.assertEqual([f["source_id"] for f in result.updates["facts"]], [3])
        self.assertEqual(result.updates["covered_needs"], [])
        self.assertEqual(result.updates["resolved_needs"], [0])
        removed = next(t for t in result.trace["fact_transitions"] if t["fact_id"] == old_id)
        self.assertEqual((removed["from"], removed["to"]), ("accept", "reject"))
        self.assertTrue(removed["basis_fact_ids"])

    def test_need_verdict_requires_all_needs_and_accepted_mapped_references(self):
        def bad(payload, case):
            response = self.accepted(payload)
            first = response["need_decisions"][0]
            if case == "missing":
                response["need_decisions"].pop()
            elif case == "duplicate":
                response["need_decisions"][1] = deepcopy(first)
            elif case == "no_evidence":
                first.update(status="refuted", fact_ids=[])
            elif case == "wrong_need":
                response["need_decisions"][1].update(status="supported", fact_ids=first["fact_ids"])
            elif case == "unaccepted":
                response["fact_decisions"][0].update(decision="reject")
            else:
                first["status"] = []
            return response
        for case in ("missing", "duplicate", "no_evidence", "wrong_need", "unaccepted", "status"):
            with self.subTest(case=case), self.assertRaises(ReviewSchemaError):
                self.run_wave({"facts": [self.fact()], "links": []}, lambda p: bad(p, case))

    def test_same_claim_merges_need_mapping_but_same_span_distinct_claims_survive(self):
        state = self.first_state()
        first_id = state["facts"][0]["fact_id"]
        result = self.run_wave({"facts": [self.fact(needs=[1]), self.fact(claim="The original record says Alice accepted.")], "links": []}, state=state)
        self.assertEqual(len(result.updates["facts"]), 2)
        original = next(f for f in result.updates["facts"] if f["fact_id"] == first_id)
        self.assertEqual(original["need_indices"], [0, 1])
        self.assertEqual(stable_fact_id({**original, "need_indices": []}), first_id)
        self.assertEqual(len(result.visible), 1)
        self.assertEqual(len(result.updates["evidence_contexts"]), 1)

    def test_link_needs_explicit_decision_and_accepted_fact_on_each_endpoint(self):
        mapper = {"facts": [self.fact(), self.fact(2)], "links": [{"from_episode_id": 1, "to_episode_id": 2, "rationale": "The accepted request was followed by delivery."}]}
        result = self.run_wave(mapper, windows=[self.window(1), self.window(2)])
        self.assertEqual(len(result.updates["verified_links"]), 1)
        self.assertEqual(len(result.updates["verified_links"][0]["evidence"]), 2)
        def bad(payload):
            response = self.accepted(payload)
            response["fact_decisions"][1]["decision"] = "needs_context"
            response["need_decisions"][1].update(status="unknown", fact_ids=[])
            return response
        with self.assertRaisesRegex(ReviewSchemaError, "both endpoints"):
            self.run_wave(mapper, bad, windows=[self.window(1), self.window(2)])

    def test_empty_candidates_skip_verifier_without_claiming_resolution(self):
        result = self.run_wave({"facts": [], "links": []})
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(result.trace["verifier_skipped"])
        self.assertEqual(result.updates["resolved_needs"], [])
        self.assertEqual(result.updates["source_offsets"], {"1": len(self.sources[1])})

    def test_wrong_literal_quote_is_local_rejection_not_a_fact(self):
        result = self.run_wave({"facts": [{**self.fact(), "quote": "not in this Source"}], "links": []})
        self.assertEqual(result.updates["facts"], [])
        self.assertTrue(result.trace["verifier_skipped"])
        self.assertEqual(result.trace["local_rejections"][0]["reason"], "quote_missing_or_ambiguous")

    def test_narrow_overlap_cannot_hide_ambiguous_quote_in_a_wider_window(self):
        for raw, quote in (("甲说：同意。乙说：同意。", "同意。"), ("aaaa", "aaa")):
            with self.subTest(raw=raw):
                self.sources[1] = raw
                first = raw.index(quote)
                narrow = {**self.window(1), "start": first, "end": first + len(quote), "text": quote}
                result = self.run_wave({"facts": [{**self.fact(), "quote": quote}], "links": []},
                                       windows=[narrow, self.window(1)])
                self.assertEqual(result.updates["facts"], [])
                self.assertEqual(result.trace["local_rejections"][0]["reason"], "quote_missing_or_ambiguous")
                self.assertTrue(result.trace["verifier_skipped"])

    def test_same_source_quote_cannot_borrow_episode_from_a_different_visible_block(self):
        self.episodes[4] = {"id": 4, "source_id": 1, "text": "second part"}
        split = self.sources[1].index("Alice accepted")
        first = {**self.window(1), "end": split, "text": self.sources[1][:split]}
        second = {**self.window(1), "episode_id": 4, "start": split, "text": self.sources[1][split:]}
        result = self.run_wave({"facts": [self.fact()], "links": []}, windows=[first, second])
        self.assertEqual(result.updates["facts"], [])
        accepted = self.run_wave({"facts": [{**self.fact(), "episode_id": 4}], "links": []}, windows=[first, second])
        self.assertEqual(len(accepted.updates["facts"]), 1)

    def test_recheck_window_does_not_move_a_source_cursor(self):
        state = self.first_state()
        old = deepcopy(state["source_offsets"])
        result = self.run_wave({"facts": [], "links": []}, state=state, windows=[self.window(1, recheck=True)])
        self.assertEqual(result.updates["source_offsets"], old)
        self.assertTrue(result.trace["source_windows"][0]["recheck"])

    def test_expanded_context_is_retained_without_reproposing_the_pending_claim(self):
        quote = self.fact()["quote"]
        start = self.sources[1].index(quote)
        narrow = {**self.window(1), "start": start, "end": start + len(quote), "text": quote}
        def pending(payload):
            response = self.accepted(payload)
            response["fact_decisions"][0]["decision"] = "needs_context"
            for need in response["need_decisions"]:
                need.update(status="unknown", fact_ids=[])
            return response
        first = self.run_wave({"facts": [self.fact()], "links": []}, pending, windows=[narrow])
        state = {**self.state, **first.updates}
        second = self.run_wave({"facts": [], "links": []}, state=state, windows=[self.window(1, recheck=True)])
        self.assertEqual(len(second.updates["facts"]), 1)
        contexts = second.updates["evidence_contexts"]
        self.assertTrue(any(c["start"] == 0 and c["end"] == len(self.sources[1]) for c in contexts.values()))
        state = {**state, **second.updates}
        third = self.run_wave({"facts": [], "links": []}, state=state, windows=[self.window(2)])
        self.assertTrue(any(w["source_id"] == 1 and w["text"] == self.sources[1] for w in third.visible))

    def test_duplicate_context_range_unifies_episode_bindings(self):
        self.episodes[4] = {"id": 4, "source_id": 1, "text": "alternate hint"}
        result = self.run_wave({"facts": [self.fact(), {**self.fact(), "episode_id": 4}], "links": []},
                               windows=[self.window(1), {**self.window(1), "episode_id": 4}])
        self.assertEqual(len(result.visible), 1)
        self.assertEqual(result.visible[0]["episode_ids"], [1, 4])
        self.assertEqual(len(result.updates["facts"]), 2)

    def test_transport_exception_preserves_all_inputs(self):
        state = self.first_state()
        before = deepcopy(state)
        def fail(payload):
            raise TimeoutError("local scripted timeout")
        with self.assertRaises(TimeoutError):
            self.run_wave({"facts": [], "links": []}, fail, state=state, windows=[self.window(2)])
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
