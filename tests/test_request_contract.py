from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unittest

from benchmarks.run_request_scoped_generation_eval import _build_snapshot
from config.prompt_config.request_scoped_answer_prompts import (
    request_scoped_answer_prompt,
)
from memory_demo.contracts import (
    Limitation,
    QuestionPremise,
    RequestAnswerContract,
    RequestedAction,
    RuntimeReceipt,
)


class RequestContractTests(unittest.TestCase):
    def _contract(self, request_id: str = "request-a") -> RequestAnswerContract:
        return RequestAnswerContract(
            request_id=request_id,
            question="store marker",
            requested_actions=(
                RequestedAction(
                    "memory_write",
                    "private:user-a",
                    "write the marker to this user's private memory",
                ),
            ),
            private_write_candidates=("marker",),
        )

    def test_snapshot_does_not_observe_late_receipt(self) -> None:
        snapshot = self._contract()
        completed = snapshot.with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-a",
                action="memory_write",
                target="private:user-a",
                status="succeeded",
                result_summary="stored marker",
            )
        )
        self.assertEqual(
            "unverified",
            snapshot.generation_view()["verified_action_outcomes"][0]["status"],
        )
        self.assertEqual(
            "succeeded",
            completed.generation_view()["verified_action_outcomes"][0]["status"],
        )

    def test_receipt_from_another_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "another request"):
            self._contract().with_receipt(
                RuntimeReceipt(
                    request_id="request-b",
                    receipt_id="receipt-b",
                    action="memory_write",
                    target="private:user-a",
                    status="succeeded",
                )
            )

    def test_wrong_target_receipt_is_not_exposed_to_generation(self) -> None:
        contract = self._contract().with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-wrong-target",
                action="memory_write",
                target="private:user-b",
                status="succeeded",
            )
        )
        self.assertEqual(
            "unverified",
            contract.generation_view()["verified_action_outcomes"][0]["status"],
        )
        self.assertEqual(
            ("receipt-wrong-target",), contract.ignored_receipt_ids()
        )

    def test_duplicate_receipt_is_idempotent_but_conflict_is_rejected(self) -> None:
        receipt = RuntimeReceipt(
            request_id="request-a",
            receipt_id="receipt-a",
            action="memory_write",
            target="private:user-a",
            status="succeeded",
        )
        contract = self._contract().with_receipt(receipt)
        self.assertIs(contract, contract.with_receipt(receipt))
        with self.assertRaisesRegex(ValueError, "reused"):
            contract.with_receipt(
                RuntimeReceipt(
                    request_id="request-a",
                    receipt_id="receipt-a",
                    action="memory_write",
                    target="private:user-a",
                    status="failed",
                )
            )

    def test_question_premise_state_is_immutable_and_conflict_is_rejected(self) -> None:
        premise = QuestionPremise("A met B", "unresolved")
        base = self._contract()
        updated = base.with_premise(premise)
        self.assertEqual((), base.question_premises)
        self.assertEqual("unresolved", updated.question_premises[0].status)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            updated.with_premise(QuestionPremise("A met B", "supported"))

    def test_response_directives_require_failed_action_and_public_error(self) -> None:
        contract = self._contract().with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-failed",
                action="memory_write",
                target="private:user-a",
                status="failed",
                error_summary="storage unavailable",
            )
        )
        directive = contract.response_directives()[0]
        self.assertTrue(directive["required"])
        self.assertEqual("failed", directive["status"])
        self.assertEqual("storage unavailable", directive["error_summary"])
        self.assertEqual(
            "write the marker to this user's private memory",
            directive["public_action_summary"],
        )

    def test_atomic_audit_contract_shares_request_and_response_directives(self) -> None:
        contract = self._contract().with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-success",
                action="memory_write",
                target="private:user-a",
                status="succeeded",
                result_summary="stored",
            )
        )
        audit = contract.atomic_audit_contract()
        self.assertEqual(
            [
                {
                    "action": "memory_write",
                    "target": "private:user-a",
                    "public_action_summary": (
                        "write the marker to this user's private memory"
                    ),
                }
            ],
            audit["requested_actions"],
        )
        self.assertEqual("succeeded", audit["response_directives"][0]["status"])

    def test_action_state_limitation_is_covered_by_action_outcome(self) -> None:
        contract = RequestAnswerContract(
            request_id="request-a",
            question="store marker",
            requested_actions=(
                RequestedAction(
                    "memory_write",
                    "private:user-a",
                    "write marker ALPHA to private memory",
                ),
            ),
            limitations=(
                Limitation(
                    "L1",
                    "whether marker ALPHA was written",
                    "action_state",
                    action="memory_write",
                    target="private:user-a",
                ),
            ),
        ).with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-failed",
                action="memory_write",
                target="private:user-a",
                status="failed",
                error_summary="storage unavailable",
            )
        )
        self.assertEqual((), contract.active_limitations())
        self.assertEqual(
            ["action_outcome"],
            [item["kind"] for item in contract.response_directives()],
        )

    def test_evidence_gap_remains_after_action_receipt(self) -> None:
        contract = RequestAnswerContract(
            request_id="request-a",
            question="what color is the wall",
            requested_actions=(
                RequestedAction(
                    "knowledge_recall",
                    "wall color",
                    "search the knowledge base for the wall color",
                ),
            ),
            limitations=(
                Limitation(
                    "L1",
                    "the wall color",
                    "evidence_gap",
                    claim_ref="wall_color",
                ),
            ),
        ).with_receipt(
            RuntimeReceipt(
                request_id="request-a",
                receipt_id="receipt-success",
                action="knowledge_recall",
                target="wall color",
                status="succeeded",
                result_summary="search completed without a supported color fact",
            )
        )
        self.assertEqual("L1", contract.active_limitations()[0].limitation_id)
        self.assertEqual(
            ["action_outcome", "limitation"],
            [item["kind"] for item in contract.response_directives()],
        )

    def test_action_state_limitation_must_reference_requested_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown action"):
            RequestAnswerContract(
                request_id="request-a",
                question="store marker",
                requested_actions=(
                    RequestedAction(
                        "memory_write",
                        "private:user-a",
                        "write marker ALPHA to private memory",
                    ),
                ),
                limitations=(
                    Limitation(
                        "L1",
                        "another action state",
                        "action_state",
                        action="weather_lookup",
                        target="Tokyo weather",
                    ),
                ),
            )

    def test_renderer_prompt_hides_internal_action_target_and_field_names(self) -> None:
        view = self._contract().generation_view()
        payload = json.loads(
            request_scoped_answer_prompt(
                question="store marker",
                persona={"name": "helper"},
                view=view,
            )
        )
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("memory_write", encoded)
        self.assertNotIn("private:user-a", encoded)
        self.assertNotIn("creative_private_items", encoded)
        directive = payload["render_contract"]["response_directives"][0]
        self.assertEqual(
            "write the marker to this user's private memory",
            directive["public_action_summary"],
        )

    def test_parallel_requests_do_not_share_receipts(self) -> None:
        def complete(index: int) -> tuple[str, str]:
            request_id = f"request-{index}"
            target = f"private:user-{index}"
            contract = RequestAnswerContract(
                request_id=request_id,
                question="store marker",
                requested_actions=(
                    RequestedAction(
                        "memory_write",
                        target,
                        f"write marker to private memory {index}",
                    ),
                ),
            )
            completed = contract.with_receipt(
                RuntimeReceipt(
                    request_id=request_id,
                    receipt_id=f"receipt-{index}",
                    action="memory_write",
                    target=target,
                    status="succeeded",
                )
            )
            view = completed.generation_view()["verified_action_outcomes"][0]
            return f"receipt-{index}", view["target"]

        with ThreadPoolExecutor(max_workers=8) as executor:
            actual = list(executor.map(complete, range(100)))
        self.assertEqual(100, len(set(actual)))
        self.assertEqual(
            [(f"receipt-{index}", f"private:user-{index}") for index in range(100)],
            actual,
        )

    def test_generation_manifest_quarantines_wrong_and_foreign_receipts(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks/manifests/request_scoped_generation_v2.json"
        )
        rows = {
            row["id"]: row
            for row in json.loads(path.read_text(encoding="utf-8"))["cases"]
        }
        wrong, wrong_diagnostics = _build_snapshot(
            rows["request_v1:write_wrong_target_receipt"]
        )
        self.assertEqual((), wrong.matching_receipts())
        self.assertEqual(["RW3"], wrong_diagnostics["ignored_receipt_ids"])
        foreign, foreign_diagnostics = _build_snapshot(
            rows["request_v1:write_foreign_request_receipt"]
        )
        self.assertEqual((), foreign.runtime_receipts)
        self.assertEqual(1, len(foreign_diagnostics["rejected_receipts"]))

    def test_generation_manifest_freezes_late_receipt_visibility(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks/manifests/request_scoped_generation_v2.json"
        )
        rows = {
            row["id"]: row
            for row in json.loads(path.read_text(encoding="utf-8"))["cases"]
        }
        before, before_diagnostics = _build_snapshot(
            rows["request_v1:late_receipt_after_snapshot"]
        )
        after, after_diagnostics = _build_snapshot(
            rows["request_v1:late_receipt_before_snapshot"]
        )
        self.assertEqual((), before.matching_receipts())
        self.assertEqual(["RL2"], before_diagnostics["next_snapshot_receipt_ids"])
        self.assertEqual("RL1", after.matching_receipts()[0].receipt_id)
        self.assertEqual(["RL1"], after_diagnostics["visible_receipt_ids"])


if __name__ == "__main__":
    unittest.main()
