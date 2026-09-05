"""Request-scoped evidence needed to describe runtime actions truthfully.

These objects are immutable snapshots.  Executors may return a new contract
with an appended receipt, but a renderer that already owns an older snapshot
cannot observe a late completion or a receipt from another request.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal


ReceiptStatus = Literal["succeeded", "failed", "cancelled"]
PremiseStatus = Literal["supported", "contradicted", "unresolved"]
LimitationKind = Literal["action_state", "evidence_gap"]

RECEIPT_STATUSES = {"succeeded", "failed", "cancelled"}
PREMISE_STATUSES = {"supported", "contradicted", "unresolved"}
LIMITATION_KINDS = {"action_state", "evidence_gap"}


def _required(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


@dataclass(frozen=True, slots=True)
class RequestedAction:
    action: str
    target: str
    public_action_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _required(self.action, "action"))
        object.__setattr__(self, "target", _required(self.target, "target"))
        object.__setattr__(
            self,
            "public_action_summary",
            _required(self.public_action_summary, "public_action_summary"),
        )

    def prompt_view(self) -> dict[str, str]:
        return {
            "action": self.action,
            "target": self.target,
            "public_action_summary": self.public_action_summary,
        }


@dataclass(frozen=True, slots=True)
class Limitation:
    limitation_id: str
    text: str
    kind: LimitationKind
    action: str = ""
    target: str = ""
    claim_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "limitation_id", _required(self.limitation_id, "limitation_id")
        )
        object.__setattr__(self, "text", _required(self.text, "limitation text"))
        kind = str(self.kind or "").strip().casefold()
        if kind not in LIMITATION_KINDS:
            raise ValueError(f"invalid limitation kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "action", str(self.action or "").strip())
        object.__setattr__(self, "target", str(self.target or "").strip())
        object.__setattr__(self, "claim_ref", str(self.claim_ref or "").strip())
        if kind == "action_state":
            _required(self.action, "action_state limitation action")
            _required(self.target, "action_state limitation target")
            if self.claim_ref:
                raise ValueError("action_state limitation cannot have claim_ref")
        else:
            _required(self.claim_ref, "evidence_gap limitation claim_ref")
            if self.action or self.target:
                raise ValueError("evidence_gap limitation cannot bind an action")

    def prompt_view(self) -> dict[str, str]:
        return {
            "limitation_id": self.limitation_id,
            "text": self.text,
            "kind": self.kind,
            "action": self.action,
            "target": self.target,
            "claim_ref": self.claim_ref,
        }


@dataclass(frozen=True, slots=True)
class RuntimeReceipt:
    request_id: str
    receipt_id: str
    action: str
    target: str
    status: ReceiptStatus
    result_summary: str = ""
    error_summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, "request_id"))
        object.__setattr__(self, "receipt_id", _required(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "action", _required(self.action, "action"))
        object.__setattr__(self, "target", _required(self.target, "target"))
        status = str(self.status or "").strip().casefold()
        if status not in RECEIPT_STATUSES:
            raise ValueError(f"invalid receipt status: {self.status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "result_summary", str(self.result_summary or "").strip())
        object.__setattr__(self, "error_summary", str(self.error_summary or "").strip())

    def prompt_view(self) -> dict[str, str]:
        return {
            "receipt_id": self.receipt_id,
            "action": self.action,
            "target": self.target,
            "status": self.status,
            "result_summary": self.result_summary,
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True, slots=True)
class QuestionPremise:
    premise: str
    status: PremiseStatus
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise", _required(self.premise, "premise"))
        status = str(self.status or "").strip().casefold()
        if status not in PREMISE_STATUSES:
            raise ValueError(f"invalid premise status: {self.status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(str(value).strip() for value in self.evidence_refs if str(value).strip()),
        )

    def prompt_view(self) -> dict[str, Any]:
        return {
            "premise": self.premise,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class RequestAnswerContract:
    request_id: str
    question: str
    intent_mode: str = "conversation"
    answer_language: str = "other"
    answerability: str = "not_applicable"
    requested_actions: tuple[RequestedAction, ...] = ()
    supported_facts: tuple[str, ...] = ()
    limitations: tuple[Limitation, ...] = ()
    private_write_candidates: tuple[str, ...] = ()
    knowledge_write_candidates: tuple[str, ...] = ()
    runtime_receipts: tuple[RuntimeReceipt, ...] = ()
    question_premises: tuple[QuestionPremise, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, "request_id"))
        object.__setattr__(self, "question", _required(self.question, "question"))
        action_keys = {(item.action, item.target) for item in self.requested_actions}
        if len(action_keys) != len(self.requested_actions):
            raise ValueError("duplicate requested action and target")
        limitation_ids = [item.limitation_id for item in self.limitations]
        if len(limitation_ids) != len(set(limitation_ids)):
            raise ValueError("duplicate limitation_id in one request contract")
        for limitation in self.limitations:
            if (
                limitation.kind == "action_state"
                and (limitation.action, limitation.target) not in action_keys
            ):
                raise ValueError(
                    f"limitation {limitation.limitation_id!r} references an unknown action"
                )
        for receipt in self.runtime_receipts:
            if receipt.request_id != self.request_id:
                raise ValueError(
                    f"receipt {receipt.receipt_id!r} belongs to another request"
                )
        receipt_ids = [receipt.receipt_id for receipt in self.runtime_receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError("duplicate receipt_id in one request contract")

    def with_receipt(self, receipt: RuntimeReceipt) -> "RequestAnswerContract":
        if receipt.request_id != self.request_id:
            raise ValueError(
                f"receipt {receipt.receipt_id!r} belongs to another request"
            )
        existing = next(
            (
                item
                for item in self.runtime_receipts
                if item.receipt_id == receipt.receipt_id
            ),
            None,
        )
        if existing is not None:
            if existing == receipt:
                return self
            raise ValueError(
                f"receipt_id {receipt.receipt_id!r} was reused with different content"
            )
        return replace(self, runtime_receipts=(*self.runtime_receipts, receipt))

    def with_premise(self, premise: QuestionPremise) -> "RequestAnswerContract":
        existing = next(
            (
                item
                for item in self.question_premises
                if item.premise == premise.premise
            ),
            None,
        )
        if existing is not None:
            if existing == premise:
                return self
            raise ValueError("the same question premise has conflicting states")
        return replace(self, question_premises=(*self.question_premises, premise))

    def matching_receipts(self) -> tuple[RuntimeReceipt, ...]:
        licensed = {(item.action, item.target) for item in self.requested_actions}
        return tuple(
            receipt
            for receipt in self.runtime_receipts
            if (receipt.action, receipt.target) in licensed
        )

    def ignored_receipt_ids(self) -> tuple[str, ...]:
        matched = {receipt.receipt_id for receipt in self.matching_receipts()}
        return tuple(
            receipt.receipt_id
            for receipt in self.runtime_receipts
            if receipt.receipt_id not in matched
        )

    def verified_action_outcomes(self) -> tuple[dict[str, str], ...]:
        """Resolve each requested action to the last exact-match request receipt."""
        outcomes: list[dict[str, str]] = []
        receipts = self.matching_receipts()
        for requested in self.requested_actions:
            matching = [
                receipt
                for receipt in receipts
                if receipt.action == requested.action
                and receipt.target == requested.target
            ]
            if not matching:
                outcomes.append(
                    {
                        "action": requested.action,
                        "target": requested.target,
                        "public_action_summary": requested.public_action_summary,
                        "status": "unverified",
                        "result_summary": "",
                        "error_summary": "",
                    }
                )
                continue
            receipt = matching[-1]
            outcomes.append(
                {
                    "action": requested.action,
                    "target": requested.target,
                    "public_action_summary": requested.public_action_summary,
                    "status": receipt.status,
                    "result_summary": receipt.result_summary,
                    "error_summary": receipt.error_summary,
                }
            )
        return tuple(outcomes)

    def active_limitations(self) -> tuple[Limitation, ...]:
        """Return unresolved fact gaps not already represented by action outcomes."""
        return tuple(
            limitation
            for limitation in self.limitations
            if limitation.kind == "evidence_gap"
        )

    def response_directives(self) -> tuple[dict[str, Any], ...]:
        """Compile permissions into the semantic units a renderer must cover."""
        directives: list[dict[str, Any]] = []
        for outcome in self.verified_action_outcomes():
            directives.append(
                {
                    "kind": "action_outcome",
                    "required": True,
                    **outcome,
                }
            )
        directives.extend(
            {"kind": "supported_fact", "required": True, "text": fact}
            for fact in self.supported_facts
        )
        directives.extend(
            {
                "kind": "limitation",
                "required": True,
                "limitation_id": limitation.limitation_id,
                "claim_ref": limitation.claim_ref,
                "text": limitation.text,
            }
            for limitation in self.active_limitations()
        )
        directives.extend(
            {
                "kind": "question_premise",
                "required": True,
                **premise.prompt_view(),
            }
            for premise in self.question_premises
        )
        return tuple(directives)

    def generation_view(self) -> dict[str, Any]:
        """Return deterministic action outcomes rather than raw receipt candidates."""
        return {
            "request_id": self.request_id,
            "question": self.question,
            "intent_mode": self.intent_mode,
            "answer_language": self.answer_language,
            "answerability": self.answerability,
            "requested_actions": [
                item.prompt_view()
                for item in self.requested_actions
            ],
            "allowed_supported_facts": list(self.supported_facts),
            "limitations": [
                limitation.prompt_view()
                for limitation in self.active_limitations()
            ],
            "all_limitations": [
                limitation.prompt_view() for limitation in self.limitations
            ],
            "creative_private_items": list(self.private_write_candidates),
            "knowledge_write_candidates": list(self.knowledge_write_candidates),
            "verified_action_outcomes": [
                dict(outcome) for outcome in self.verified_action_outcomes()
            ],
            "question_premises": [
                premise.prompt_view() for premise in self.question_premises
            ],
            "response_directives": [
                dict(directive) for directive in self.response_directives()
            ],
        }

    def atomic_audit_contract(self) -> dict[str, Any]:
        """Translate the immutable request snapshot into the existing audit shape."""
        return {
            "intent_mode": self.intent_mode,
            "answer_language": self.answer_language,
            "answerability": self.answerability,
            "claims": [
                {"claim": fact, "status": "supported", "evidence_refs": []}
                for fact in self.supported_facts
            ],
            "missing_requirements": [
                limitation.text for limitation in self.active_limitations()
            ],
            "limitations": [
                limitation.prompt_view()
                for limitation in self.active_limitations()
            ],
            "write_candidates": {
                "private": list(self.private_write_candidates),
                "knowledge": list(self.knowledge_write_candidates),
            },
            "requested_actions": [
                item.prompt_view()
                for item in self.requested_actions
            ],
            "runtime_receipts": [
                receipt.prompt_view() for receipt in self.matching_receipts()
            ],
            "question_premises": [
                premise.prompt_view() for premise in self.question_premises
            ],
            "response_directives": [
                dict(directive) for directive in self.response_directives()
            ],
        }
