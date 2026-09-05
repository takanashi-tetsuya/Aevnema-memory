from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import re

from memory_demo.associations.builder import AssociationBuilder
from memory_demo.config import WeightConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import (
    GROWTH_ADVERSARIAL_AUDIT_SYSTEM,
    GROWTH_AUDIT_SYSTEM,
    GROWTH_SYSTEM,
    fixed_endpoint_growth_prompt,
    growth_audit_prompt,
    growth_prompt,
)
from memory_demo.llm.validation import parse_growth_relationships
from memory_demo.repositories import AssociationRepository, EpisodeRepository
from memory_demo.types import AssociationDraft


_EXPLICIT_ROLE_CLAIM_RE = re.compile(
    r"(?:^|[，,。；;：:、（(\s])"
    r"(?P<organization>[A-Za-z0-9_·\u3400-\u9fff]{2,16}?)"
    r"(?:的)?"
    r"(?P<role>高层|成员|负责人|会长|主席|话事人)"
)


def _is_identity_role_claim(text: str, match: re.Match[str]) -> bool:
    """Return whether a role phrase is asserted as a node's identity.

    A role-shaped noun phrase used as a narrative subject is ordinary scene
    description. The integrity guard is aimed at explicit identity upgrades.
    Treating every such noun phrase as an identity assertion creates false
    positives for otherwise well-grounded relationship summaries.
    """
    raw_organization = match.group("organization")
    if re.search(r"(?:作为|身为|是|属于|担任|任职于|来自)", raw_organization):
        return True
    claim_start = match.start("organization")
    before = text[:claim_start]
    if before.rfind("（") > before.rfind("）"):
        return True
    if before.rfind("(") > before.rfind(")"):
        return True
    return bool(
        re.search(
            r"(?:作为|身为|是|属于|担任|任职于|来自)\s*$",
            before[-16:],
        )
    )


@dataclass(slots=True)
class GrowthOutcome:
    created_ids: list[int] = field(default_factory=list)
    reinforced_ids: list[int] = field(default_factory=list)
    reinforced_before: dict[int, dict] = field(default_factory=dict)

    @property
    def changed_ids(self) -> list[int]:
        return list(dict.fromkeys([*self.created_ids, *self.reinforced_ids]))

    @property
    def changed(self) -> bool:
        return bool(self.created_ids or self.reinforced_ids)


class AssociationGrowthEngine:
    def __init__(
        self,
        model,
        repository: AssociationRepository,
        episodes: EpisodeRepository,
        weights: WeightConfig,
        logger: JsonlEventLogger | None = None,
    ):
        self.model = model
        self.repository = repository
        self.episodes = episodes
        self.weights = weights
        self.logger = logger

    def _initial_weight(self, llm_score: float, confidence: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                self.weights.llm * llm_score
                + self.weights.embedding * confidence
                + self.weights.evidence,
            ),
        )

    def _request_audit_reviews(
        self,
        system: str,
        question: str,
        nodes: list[dict],
        relationships: list[dict],
        edges: list[dict],
        label: str,
        model_name: str | None = None,
    ) -> tuple[dict[int, dict], list[str]]:
        try:
            prompt = growth_audit_prompt(question, nodes, relationships, edges)
            if model_name:
                payload = self.model.chat_json(
                    system,
                    prompt,
                    model=model_name,
                    allow_fallback=False,
                )
            else:
                payload = self.model.chat_json(system, prompt)
        except Exception as exc:
            return {}, [f"{label} growth evidence audit request failed: {exc}"]
        if not isinstance(payload, dict) or not isinstance(
            payload.get("reviews"), list
        ):
            return {}, [f"{label} growth evidence audit reviews must be a list"]
        reviews: dict[int, dict] = {}
        errors: list[str] = []
        for position, raw_review in enumerate(payload["reviews"]):
            if not isinstance(raw_review, dict):
                errors.append(
                    f"{label} growth audit review {position}: must be an object"
                )
                continue
            try:
                index = int(raw_review["index"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label} growth audit review {position}: invalid index")
                continue
            if not 0 <= index < len(relationships):
                errors.append(
                    f"{label} growth audit review {position}: index out of range"
                )
                continue
            if index in reviews:
                errors.append(
                    f"{label} growth audit review {position}: duplicate index {index}"
                )
                continue
            reviews[index] = raw_review
        for index in range(len(relationships)):
            if index not in reviews:
                errors.append(f"{label} growth audit relationship {index}: missing review")
        return reviews, errors

    def _audit_relationships(
        self,
        question: str,
        nodes: list[dict],
        relationships: list[dict],
        edges: list[dict],
    ) -> tuple[list[dict], list[str]]:
        if not relationships:
            return [], []
        primary_reviews, errors = self._request_audit_reviews(
            GROWTH_AUDIT_SYSTEM,
            question,
            nodes,
            relationships,
            edges,
            "primary",
        )
        primary_indices = [
            index
            for index in range(len(relationships))
            if primary_reviews.get(index, {}).get("accept") is True
        ]
        primary_accepted = [relationships[index] for index in primary_indices]
        adversarial_reviews: dict[int, dict] = {}
        if primary_accepted:
            model_config = getattr(self.model, "config", None)
            independent_model = getattr(model_config, "fallback_model", None)
            adversarial_reviews, adversarial_errors = self._request_audit_reviews(
                GROWTH_ADVERSARIAL_AUDIT_SYSTEM,
                question,
                nodes,
                primary_accepted,
                edges,
                "adversarial",
                independent_model,
            )
            errors.extend(adversarial_errors)
        accepted: list[dict] = []
        decisions: list[dict] = []
        for index, relationship in enumerate(relationships):
            primary_review = primary_reviews.get(index)
            primary_accept = bool(
                primary_review is not None
                and primary_review.get("accept") is True
            )
            adversarial_position = (
                primary_indices.index(index) if index in primary_indices else None
            )
            adversarial_review = (
                adversarial_reviews.get(adversarial_position)
                if adversarial_position is not None
                else None
            )
            adversarial_accept = bool(
                adversarial_review is not None
                and adversarial_review.get("accept") is True
            )
            accept = primary_accept and adversarial_accept
            decision = {
                "index": index,
                "accept": accept,
                "primary_accept": primary_accept,
                "primary_reason": (
                    str(primary_review.get("reason", "")).strip()
                    if primary_review
                    else "missing review"
                ),
                "adversarial_accept": adversarial_accept,
                "adversarial_reason": (
                    str(adversarial_review.get("reason", "")).strip()
                    if adversarial_review
                    else (
                        "not run because primary rejected"
                        if not primary_accept
                        else "missing review"
                    )
                ),
            }
            decisions.append(decision)
            if accept:
                accepted.append({**relationship, "_audit_record": decision})
        if self.logger:
            self.logger.emit(
                "growth_relationship_audit",
                question=question,
                proposed_count=len(relationships),
                accepted_count=len(accepted),
                decisions=decisions,
                errors=errors,
            )
        return accepted, errors

    @staticmethod
    def _edge_association_id(edge: dict) -> int | None:
        raw_id = edge.get("association_id", edge.get("id"))
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _attach_visible_premises(
        cls,
        relationships: list[dict],
        edges: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Resolve cited inference premises against the visible graph only.

        Episode/Concept endpoint text is direct evidence and is deliberately not
        counted as an Association premise.  A relationship that cites an edge
        the model was not shown is rejected instead of silently receiving a low
        generation number.
        """
        edge_map = {
            association_id: dict(edge)
            for edge in edges
            if (association_id := cls._edge_association_id(edge)) is not None
        }
        accepted: list[dict] = []
        errors: list[str] = []
        for index, relationship in enumerate(relationships):
            premise_ids = list(
                dict.fromkeys(
                    int(value)
                    for value in relationship.get("premise_association_ids", [])
                )
            )
            missing = [value for value in premise_ids if value not in edge_map]
            if missing:
                errors.append(
                    f"relationship {index}: premise associations are not visible: {missing}"
                )
                continue
            premise_edges = [edge_map[value] for value in premise_ids]
            invalid_generation: list[int] = []
            for value, edge in zip(premise_ids, premise_edges, strict=True):
                try:
                    premise_generation = int(edge.get("generation", 0))
                except (TypeError, ValueError):
                    invalid_generation.append(value)
                    continue
                if premise_generation < 0:
                    invalid_generation.append(value)
            if invalid_generation:
                errors.append(
                    f"relationship {index}: premise associations have invalid generation: "
                    f"{invalid_generation}"
                )
                continue
            accepted.append(
                {
                    **relationship,
                    "premise_association_ids": premise_ids,
                    "_premise_edges": premise_edges,
                }
            )
        return accepted, errors

    @staticmethod
    def _generation(
        relation: dict,
        node_map: dict[tuple[str, int], dict] | None = None,
    ) -> int:
        premise_edges = relation.get("_premise_edges", [])
        generations = [int(edge.get("generation", 0)) for edge in premise_edges]
        for prefix in ("from", "to"):
            if str(relation.get(f"{prefix}_type")) != "episode":
                continue
            node = (node_map or {}).get(
                ("episode", int(relation.get(f"{prefix}_id", -1))), {}
            )
            generations.append(max(0, int(node.get("generation", 0) or 0)))
        # Every query-grown edge is at least one inference away from direct
        # Episode/Concept evidence.  Parallel premises use the longest chain;
        # summing them would measure breadth rather than distance.
        return max(generations, default=0) + 1

    @staticmethod
    def _unsupported_role_claim(
        relation: dict,
        node_map: dict[tuple[str, int], dict],
    ) -> str | None:
        """Reject explicit affiliations absent from visible direct evidence.

        Cooperation, knowledge of an internal conflict or ordinary co-occurrence
        cannot be upgraded into organization membership. Formal succession or
        replacing a leader remains an allowed, explicitly marked inference.
        """
        relation_text = str(relation.get("relation_text", ""))
        claims = list(_EXPLICIT_ROLE_CLAIM_RE.finditer(relation_text))
        if not claims:
            return None
        visible_texts = [
            str(
                node_map.get(
                    (str(relation[f"{prefix}_type"]), int(relation[f"{prefix}_id"])),
                    {},
                ).get("text", "")
            )
            for prefix in ("from", "to")
        ]
        visible_texts.extend(
            str(edge.get("relation_text", ""))
            for edge in relation.get("_premise_edges", [])
        )
        evidence = " ".join(visible_texts)
        # Extraction may keep bilingual glosses inside a role phrase. Ignore
        # an ASCII-only gloss when comparing it with the translated claim.
        comparable_evidence = re.sub(
            r"[（(][A-Za-z][A-Za-z0-9 _.'·/-]*[）)]",
            "",
            evidence,
        )
        compact_evidence = re.sub(r"[\s的]", "", comparable_evidence)
        for match in claims:
            if not _is_identity_role_claim(relation_text, match):
                continue
            organization = match.group("organization")
            organization = re.split(
                r"(?:作为|身为|是|属于|担任|任职于|来自)",
                organization,
            )[-1]
            role = match.group("role")
            compact_claim = re.sub(r"[\s的]", "", organization + role)
            if compact_claim in compact_evidence:
                continue
            return (
                "explicit organization role lacks visible direct evidence: "
                f"{organization}{role}"
            )
        return None

    @staticmethod
    def _integrity_rejection_reason(
        relation: dict,
        node_map: dict[tuple[str, int], dict],
    ) -> str | None:
        from_node = node_map.get(
            (str(relation["from_type"]), int(relation["from_id"])), {}
        )
        to_node = node_map.get(
            (str(relation["to_type"]), int(relation["to_id"])), {}
        )
        relation_text = str(relation.get("relation_text", ""))
        uncertain_endpoints = [
            node
            for node in (from_node, to_node)
            if str(node.get("epistemic_status", "unknown"))
            in {"reported", "speculative", "mixed"}
        ]
        if uncertain_endpoints and not any(
            marker in relation_text
            for marker in (
                "说法", "声称", "报告", "推测", "猜测", "可能", "尚未确认",
                "未证实", "未证明", "综合推论", "导入者", "不确定",
            )
        ):
            return "reported/speculative endpoint lost its epistemic qualifier"

        role_rejection = AssociationGrowthEngine._unsupported_role_claim(
            relation, node_map
        )
        if role_rejection:
            return role_rejection

        return None

    @staticmethod
    def _claim_level(relation: dict) -> str:
        return "supported_inference"

    @staticmethod
    def _endpoint_evidence(
        relation: dict,
        node_map: dict[tuple[str, int], dict],
    ) -> list[dict]:
        evidence: list[dict] = []
        for prefix in ("from", "to"):
            node_type = str(relation[f"{prefix}_type"])
            node_id = int(relation[f"{prefix}_id"])
            node = node_map.get((node_type, node_id), {})
            evidence.append(
                {
                    "type": node_type,
                    "id": node_id,
                    "source_key": str(node.get("source_key", "")),
                    "text": str(node.get("text", "")),
                    "evidence_origin": str(
                        node.get("evidence_origin", "unknown")
                    ),
                    "epistemic_status": str(
                        node.get("epistemic_status", "unknown")
                    ),
                    "generation": int(node.get("generation", 0) or 0),
                    "epistemic_note": str(node.get("epistemic_note", "")),
                }
            )
        for edge in relation.get("_premise_edges", []):
            evidence.append(
                {
                    "type": "association",
                    "id": int(edge.get("association_id", edge.get("id"))),
                    "generation": int(edge.get("generation", 0)),
                    "relation_type": str(edge.get("relation_type", "")),
                    "relation_key": str(edge.get("relation_key", "")),
                    "relation_text": str(edge.get("relation_text", "")),
                }
            )
        return evidence

    def grow(
        self,
        question: str,
        nodes: list[dict],
        edges: list[dict],
        seen_fingerprints: set[tuple] | None = None,
        relationships_override: list[dict] | None = None,
    ) -> GrowthOutcome:
        seen = seen_fingerprints if seen_fingerprints is not None else set()
        allowed_nodes = {(node["type"], int(node["id"])) for node in nodes}
        node_map = {
            (str(node["type"]), int(node["id"])): node for node in nodes
        }
        evidence_episode_ids = {
            int(node["id"]) for node in nodes if node["type"] == "episode"
        }
        if not evidence_episode_ids:
            return GrowthOutcome()
        payload = (
            {"relationships": relationships_override}
            if relationships_override is not None
            else self.model.chat_json(
                GROWTH_SYSTEM, growth_prompt(question, nodes, edges)
            )
        )
        relationships, errors = parse_growth_relationships(payload)
        relationships, premise_errors = self._attach_visible_premises(
            relationships, edges
        )
        errors.extend(premise_errors)
        if errors and self.logger:
            self.logger.emit("validation_failed", stage="association_growth", errors=errors)
        relationships, audit_errors = self._audit_relationships(
            question, nodes, relationships, edges
        )
        if audit_errors and self.logger:
            self.logger.emit(
                "validation_failed",
                stage="association_growth_evidence_audit",
                errors=audit_errors,
            )
        created: list[int] = []
        reinforced: list[int] = []
        reinforced_before: dict[int, dict] = {}
        skipped: Counter[str] = Counter()
        for relation in relationships:
            from_key = (relation["from_type"], relation["from_id"])
            to_key = (relation["to_type"], relation["to_id"])
            if from_key not in allowed_nodes or to_key not in allowed_nodes:
                skipped["node_not_in_retrieval_set"] += 1
                continue
            if not (
                (relation["from_type"] == "episode" and relation["from_id"] in evidence_episode_ids)
                or (relation["to_type"] == "episode" and relation["to_id"] in evidence_episode_ids)
            ):
                skipped["no_episode_evidence_endpoint"] += 1
                continue
            from_type = relation["from_type"]
            to_type = relation["to_type"]
            from_id = relation["from_id"]
            to_id = relation["to_id"]
            relation_type = relation["relation_type"]
            relation_key = relation["relation_key"]
            rejection_reason = self._integrity_rejection_reason(
                relation, node_map
            )
            if relation_type == "temporal":
                if from_type != "episode" or to_type != "episode":
                    rejection_reason = "temporal growth requires two Episode nodes"
                else:
                    rejection_reason = (
                        AssociationBuilder.episode_relationship_rejection_reason(
                            relation,
                            self.episodes.get(from_id),
                            self.episodes.get(to_id),
                        )
                    )
                    if not rejection_reason:
                        from_id, to_id, relation_key = (
                            AssociationBuilder.normalize_episode_relationship(
                                from_id, to_id, relation
                            )
                        )
            elif relation_type == "identity":
                if from_type != to_type:
                    rejection_reason = "identity growth requires matching node types"
                elif from_type == "episode":
                    rejection_reason = (
                        AssociationBuilder.episode_relationship_rejection_reason(
                            relation,
                            self.episodes.get(from_id),
                            self.episodes.get(to_id),
                        )
                    )
            if rejection_reason:
                skipped["integrity_guard_rejected"] += 1
                if self.logger:
                    self.logger.emit(
                        "association_rejected",
                        stage="query_growth",
                        question=question,
                        reason=rejection_reason,
                        relation=relation,
                    )
                continue
            fingerprint = (
                from_type,
                int(from_id),
                to_type,
                int(to_id),
                relation_type,
                "__semantic__" if relation_type == "semantic" else relation_key,
                int(relation["polarity"]),
            )
            if fingerprint in seen:
                skipped["duplicate_in_query"] += 1
                continue
            seen.add(fingerprint)
            generation = self._generation(relation, node_map)
            audit_record = {
                **relation.get("_audit_record", {}),
                "premise_association_ids": relation.get(
                    "premise_association_ids", []
                ),
                "generation": generation,
            }
            draft = AssociationDraft(
                from_type=from_type,
                from_id=from_id,
                to_type=to_type,
                to_id=to_id,
                relation_type=relation_type,
                relation_key=relation_key,
                relation_text=relation["relation_text"],
                polarity=relation["polarity"],
                weight=self._initial_weight(
                    relation["llm_score"], relation["confidence"]
                ),
                confidence=relation["confidence"],
                generation=generation,
                claim_level=self._claim_level(relation),
                audit_status="dual_accepted",
                evidence_json=json.dumps(
                    self._endpoint_evidence(relation, node_map),
                    ensure_ascii=False,
                ),
                audit_json=json.dumps(
                    [audit_record],
                    ensure_ascii=False,
                ),
                created_reason=f"查询中自主增长：{question}",
            )
            existing_id = self.repository.find_exact_id(draft)
            if existing_id is not None and existing_id not in reinforced_before:
                existing_row = self.repository.get(existing_id)
                if existing_row is not None:
                    reinforced_before[existing_id] = dict(existing_row)
            association_id = self.repository.upsert(draft)
            if existing_id is None:
                created.append(association_id)
            else:
                reinforced.append(association_id)
            if self.logger:
                self.logger.emit(
                    "association_growth",
                    question=question,
                    association_id=association_id,
                    draft=draft,
                    action="created" if existing_id is None else "reinforced",
                )
        if self.logger:
            self.logger.emit(
                "association_growth_result",
                question=question,
                proposed_count=len(relationships),
                created_association_ids=created,
                reinforced_association_ids=reinforced,
                skipped=dict(skipped),
                validation_errors=errors,
                audit_errors=audit_errors,
            )
        return GrowthOutcome(created, reinforced, reinforced_before)

    def audit_candidates(self, candidates: list[dict]) -> GrowthOutcome:
        """Audit exact answer-derived Episode edges without broad retrieval.

        Foreground consolidation has already selected two visible Episode
        endpoints.  Re-running a normal query can omit those endpoints and ask
        the growth model to invent nearby alternatives.  This path keeps the
        proposed edge fixed, loads its endpoints directly from SQLite, and
        applies the same primary, adversarial, integrity and generation gates
        used by ordinary autonomous growth.
        """

        type_mapping = {
            "causal": ("causal", "causal"),
            "motivation": ("semantic", "motivation"),
            "contrast": ("semantic", "thematic_contrast"),
            "trait": ("semantic", "trait_evidence"),
            # The two Episode endpoints are evidence for an entity/alias
            # identity claim; they are not themselves the same event.  Keep
            # true Episode identity edges reserved for same-scene evidence.
            "identity": ("semantic", "identity_evidence"),
            "temporal": ("temporal", "before"),
            "relationship": ("interpersonal", "relationship"),
            "recall_trigger": ("recall_trigger", "recall_trigger"),
            "thematic": ("semantic", "thematic_response"),
            # A retrieval bridge is deliberately weaker than a thematic or
            # causal world-model claim.  It records that two direct Episodes
            # jointly supplied distinct evidence slots in one successful
            # answer, so a close future query can retrieve both endpoints.
            "evidence_bridge": ("semantic", "evidence_bridge"),
        }
        relationships: list[dict] = []
        episode_ids: list[int] = []
        question_rows: list[dict] = []
        for raw in candidates[:12]:
            if not isinstance(raw, dict):
                continue
            claim = " ".join(str(raw.get("claim", "")).split())[:700]
            inference_type = str(raw.get("inference_type", "")).casefold()
            try:
                premise_ids = tuple(
                    dict.fromkeys(
                        int(value)
                        for value in raw.get("premise_episode_ids", [])
                    )
                )
                confidence = max(
                    0.0, min(1.0, float(raw.get("confidence", 0.0)))
                )
            except (TypeError, ValueError):
                continue
            if (
                not claim
                or len(premise_ids) != 2
                or inference_type not in type_mapping
            ):
                continue
            relation_type, relation_key = type_mapping[inference_type]
            relationships.append(
                {
                    "from_type": "episode",
                    "from_id": premise_ids[0],
                    "to_type": "episode",
                    "to_id": premise_ids[1],
                    "relation_type": relation_type,
                    "relation_key": relation_key,
                    "relation_text": (
                        claim
                        if claim.startswith("查询综合推论：")
                        else f"查询综合推论：{claim}"
                    ),
                    "polarity": 1,
                    "llm_score": confidence,
                    "confidence": confidence,
                    "premise_association_ids": [],
                }
            )
            episode_ids.extend(premise_ids)
            question_rows.append(
                {
                    "candidate_inference": claim,
                    "premise_episode_ids": list(premise_ids),
                    "inference_type": inference_type,
                }
            )
        if not relationships:
            return GrowthOutcome()

        rows = self.episodes.get_many(episode_ids)
        row_map = {int(row["id"]): row for row in rows}
        required_ids = set(episode_ids)
        if not required_ids.issubset(row_map):
            return GrowthOutcome()
        nodes: list[dict] = []
        for episode_id in dict.fromkeys(episode_ids):
            row = row_map[episode_id]
            nodes.append(
                {
                    "type": "episode",
                    "id": episode_id,
                    "source_key": str(row["source_key"] or ""),
                    "text": str(row["text"] or ""),
                    "story_time_text": str(row["story_time_text"] or ""),
                    "timeline_scope": str(row["timeline_scope"] or ""),
                    "evidence_origin": str(row["evidence_origin"] or "unknown"),
                    "epistemic_status": str(
                        row["epistemic_status"] or "unknown"
                    ),
                    "generation": int(row["generation"] or 0),
                    "epistemic_note": str(row["epistemic_note"] or ""),
                }
            )
        question = fixed_endpoint_growth_prompt(question_rows)
        return self.grow(
            question,
            nodes,
            [],
            set(),
            relationships_override=relationships,
        )
