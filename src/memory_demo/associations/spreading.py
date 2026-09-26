"""Resumable, read-only spreading activation over a fixed association snapshot.

Different roots combine by noisy-OR; repeated paths from one root contribute
only their strongest value. This keeps cycles and duplicate cues from creating
evidence. Activation influences scheduling, while each root propagates its own
support, so convergence cannot invent additional independent roots.

``complete`` means that this snapshot's frontier was exhausted. It does not
claim that a question's required relations have been found or source-verified.
No relation labels, embedding vectors, wall-clock age, or database writes are
involved. Callers are responsible for choosing independent seed root IDs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import time
from typing import Any


NodeKey = tuple[str, int]
_VERSION = "spreading-recall-v1"


def _unit(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _node(value: Any) -> NodeKey:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("node must be a (node_type, node_id) pair")
    if value[0] not in ("episode", "concept"):
        raise ValueError("node_type must be episode or concept")
    return value[0], _integer(value[1], "node_id")


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SpreadingEdge:
    id: int
    from_type: str
    from_id: int
    to_type: str
    to_id: int
    weight: float
    confidence: float = 1.0

    @property
    def source(self) -> NodeKey:
        return self.from_type, self.from_id

    @property
    def target(self) -> NodeKey:
        return self.to_type, self.to_id


@dataclass(frozen=True, slots=True)
class SpreadingSeed:
    node_type: str
    node_id: int
    activation: float = 1.0
    root_id: str | None = None


@dataclass(slots=True)
class SpreadingNode:
    node_type: str
    node_id: int
    activation: float
    contributions: dict[str, float]
    paths: dict[str, list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "activation": self.activation,
            "contributions": dict(self.contributions),
            "paths": {root: list(path) for root, path in self.paths.items()},
        }


@dataclass(slots=True)
class SpreadingResult:
    nodes: list[SpreadingNode]
    explored_edge_ids: list[int]
    expansions: int
    pending_count: int
    status: str
    elapsed_seconds: float
    checkpoint: dict[str, Any]

    def to_dict(self, *, include_checkpoint: bool = True) -> dict[str, Any]:
        result = {
            "nodes": [node.to_dict() for node in self.nodes],
            "explored_edge_ids": list(self.explored_edge_ids),
            "expansions": self.expansions,
            "pending_count": self.pending_count,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_checkpoint:
            result["checkpoint"] = self.checkpoint
        return result


@dataclass(frozen=True, slots=True)
class _Support:
    activation: float
    path: tuple[int, ...]
    path_nodes: tuple[NodeKey, ...]


def _activation(supports: Mapping[str, _Support]) -> float:
    values = [supports[root].activation for root in sorted(supports)]
    if any(value == 1.0 for value in values):
        return 1.0
    return -math.expm1(math.fsum(math.log1p(-value) for value in values))


class SpreadingRecall:
    """Search an immutable undirected snapshot with resumable edge-step budgets.

    Edge conductance is ``weight * confidence * damping / degree(target)**p``.
    Degree counts distinct positive-conductance neighbours, avoiding parallel
    edge count inflation. Both endpoint directions are explored, matching the
    existing association traverser. Seed activation, weight, and confidence
    must be in [0, 1]. No beam cutoff or hop limit discards weak candidates.

    ``max_expansions`` counts inspected incident edges for one root per call;
    ``max_seconds`` is also a per-call budget. Time includes state restoration
    but result/checkpoint serialization has to finish after a budget expires.
    This is a cooperative computation budget, not a hard process deadline.
    """

    def __init__(
        self,
        nodes: Iterable[NodeKey],
        edges: Iterable[SpreadingEdge | Mapping[str, Any]],
        *,
        damping: float = 0.85,
        degree_exponent: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.damping = _unit(damping, "damping")
        if self.damping == 0:
            raise ValueError("damping must be positive")
        self.degree_exponent = float(degree_exponent)
        if not math.isfinite(self.degree_exponent) or self.degree_exponent < 0:
            raise ValueError("degree_exponent must be finite and nonnegative")
        self.monotonic = monotonic
        self.nodes = frozenset(_node(node) for node in nodes)
        edge_map: dict[int, SpreadingEdge] = {}
        for item in edges:
            if isinstance(item, SpreadingEdge):
                edge = item
            else:
                edge = SpreadingEdge(
                    id=item["id"],
                    from_type=item["from_type"],
                    from_id=item["from_id"],
                    to_type=item["to_type"],
                    to_id=item["to_id"],
                    weight=item["weight"],
                    confidence=item["confidence"] if "confidence" in item.keys() else 1.0,
                )
            edge_id = _integer(edge.id, "edge id")
            source, target = _node(edge.source), _node(edge.target)
            if source not in self.nodes or target not in self.nodes:
                raise ValueError(f"edge {edge_id} references a missing node")
            if edge_id in edge_map:
                raise ValueError(f"duplicate edge id {edge_id}")
            edge_map[edge_id] = SpreadingEdge(
                edge_id, *source, *target,
                _unit(edge.weight, "weight"), _unit(edge.confidence, "confidence"),
            )
        self._edges = edge_map
        neighbours: dict[NodeKey, set[NodeKey]] = {node: set() for node in self.nodes}
        for edge in edge_map.values():
            if edge.source != edge.target and edge.weight * edge.confidence > 0:
                neighbours[edge.source].add(edge.target)
                neighbours[edge.target].add(edge.source)
        adjacency: dict[NodeKey, list[tuple[int, NodeKey, float]]] = {
            node: [] for node in self.nodes
        }
        for edge in edge_map.values():
            if edge.source == edge.target or edge.weight * edge.confidence == 0:
                continue
            for source, target in ((edge.source, edge.target), (edge.target, edge.source)):
                factor = (
                    edge.weight * edge.confidence * self.damping
                    * math.exp(-self.degree_exponent * math.log(len(neighbours[target])))
                )
                adjacency[source].append((edge.id, target, factor))
        self._adjacency = {
            node: tuple(sorted(items, key=lambda item: (-item[2], item[0], item[1])))
            for node, items in adjacency.items()
        }
        self._factors = {
            (source, edge_id): (target, factor)
            for source, items in self._adjacency.items()
            for edge_id, target, factor in items
        }
        self.graph_fingerprint = _digest({
            "nodes": sorted(self.nodes),
            "edges": [
                [edge.id, edge.source, edge.target, edge.weight, edge.confidence]
                for edge in sorted(edge_map.values(), key=lambda edge: edge.id)
            ],
        })

    def _seeds(self, seeds: Iterable[SpreadingSeed]) -> dict[tuple[NodeKey, str], float]:
        result: dict[tuple[NodeKey, str], float] = {}
        for seed in seeds:
            node = _node((seed.node_type, seed.node_id))
            if node not in self.nodes:
                raise ValueError(f"seed references a missing node: {node}")
            root = seed.root_id if seed.root_id is not None else f"{node[0]}:{node[1]}"
            if not isinstance(root, str) or not root.strip():
                raise ValueError("root_id must be a nonempty string")
            value = _unit(seed.activation, "seed activation")
            if value > 0:
                result[node, root] = max(result.get((node, root), 0.0), value)
        return result

    def _request_fingerprint(self, seeds: Mapping[tuple[NodeKey, str], float]) -> str:
        return _digest({
            "version": _VERSION,
            "graph": self.graph_fingerprint,
            "damping": self.damping,
            "degree_exponent": self.degree_exponent,
            "seeds": [[node, root, value] for (node, root), value in sorted(seeds.items())],
        })

    def _restore(
        self, checkpoint: Mapping[str, Any], request: str,
        seeds: Mapping[tuple[NodeKey, str], float],
    ) -> tuple[dict, dict, set[int], int, float]:
        try:
            payload = dict(checkpoint)
            checksum = payload.pop("checksum")
            if checksum != _digest(payload):
                raise ValueError("checkpoint checksum mismatch")
            if payload["version"] != _VERSION:
                raise ValueError("unsupported checkpoint version")
            if payload["graph_fingerprint"] != self.graph_fingerprint:
                raise ValueError("checkpoint graph fingerprint mismatch")
            if payload["request_fingerprint"] != request:
                raise ValueError("checkpoint seeds or propagation settings changed")
            best: dict[NodeKey, dict[str, _Support]] = {}
            for item in payload["supports"]:
                node, root = _node(item["node"]), item["root_id"]
                nodes = tuple(_node(value) for value in item["path_nodes"])
                path = tuple(_integer(value, "path edge id") for value in item["path"])
                value = _unit(item["activation"], "checkpoint activation")
                if (
                    node not in self.nodes or not nodes or nodes[-1] != node
                    or len(nodes) != len(path) + 1 or len(set(nodes)) != len(nodes)
                    or (nodes[0], root) not in seeds or root in best.get(node, {})
                ):
                    raise ValueError("invalid checkpoint support")
                expected = seeds[nodes[0], root]
                for source, target, edge_id in zip(nodes, nodes[1:], path):
                    matched = self._factors.get((source, edge_id))
                    if matched is None or matched[0] != target:
                        raise ValueError("invalid checkpoint support path")
                    expected *= matched[1]
                if value <= 0 or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0):
                    raise ValueError("invalid checkpoint support activation")
                best.setdefault(node, {})[root] = _Support(value, path, nodes)
            for (node, root), value in seeds.items():
                if root not in best.get(node, {}) or best[node][root].activation < value:
                    raise ValueError("checkpoint is missing a seed")
            pending: dict[NodeKey, dict[str, int]] = {}
            for item in payload["pending"]:
                node, root = _node(item["node"]), item["root_id"]
                cursor = _integer(item["next_edge"], "next_edge")
                if (
                    root not in best.get(node, {}) or root in pending.get(node, {})
                    or cursor >= len(self._adjacency[node])
                ):
                    raise ValueError("invalid checkpoint frontier")
                pending.setdefault(node, {})[root] = cursor
            explored = {_integer(value, "explored edge id") for value in payload["explored_edge_ids"]}
            if not explored.issubset(self._edges):
                raise ValueError("checkpoint references an unknown explored edge")
            expansions = _integer(payload["expansions"], "expansions")
            elapsed = float(payload["elapsed_seconds"])
            if not math.isfinite(elapsed) or elapsed < 0 or expansions < len(explored):
                raise ValueError("invalid checkpoint counters")
            return best, pending, explored, expansions, elapsed
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("malformed spreading checkpoint") from exc

    def search(
        self,
        seeds: Iterable[SpreadingSeed],
        *,
        max_seconds: float | None = None,
        max_expansions: int | None = None,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> SpreadingResult:
        if max_expansions is not None:
            max_expansions = _integer(max_expansions, "max_expansions")
        if max_seconds is not None:
            max_seconds = float(max_seconds)
            if not math.isfinite(max_seconds) or max_seconds < 0:
                raise ValueError("max_seconds must be finite and nonnegative")
        started = self.monotonic()
        seed_map = self._seeds(seeds)
        request = self._request_fingerprint(seed_map)
        if checkpoint is not None:
            best, pending, explored, expansions, elapsed = self._restore(checkpoint, request, seed_map)
        else:
            best, pending, explored, expansions, elapsed = {}, {}, set(), 0, 0.0
            for (node, root), value in seed_map.items():
                best.setdefault(node, {})[root] = _Support(value, (), (node,))
                if self._adjacency[node]:
                    pending.setdefault(node, {})[root] = 0
        activation = {node: _activation(supports) for node, supports in best.items()}
        versions: dict[NodeKey, int] = {}
        frontier: list[tuple[float, NodeKey, int]] = []

        def schedule(node: NodeKey) -> None:
            versions[node] = versions.get(node, 0) + 1
            if pending.get(node):
                heapq.heappush(frontier, (-activation[node], node, versions[node]))

        for node in pending:
            schedule(node)
        initial_expansions = expansions
        status = "complete"
        while pending:
            if max_expansions is not None and expansions - initial_expansions >= max_expansions:
                status = "expansion_budget"
                break
            if max_seconds is not None and self.monotonic() - started >= max_seconds:
                status = "time_budget"
                break
            while frontier:
                _, node, version = heapq.heappop(frontier)
                if version == versions[node] and pending.get(node):
                    break
            else:
                raise RuntimeError("spreading frontier lost pending work")
            root = min(pending[node], key=lambda key: (-best[node][key].activation, key))
            cursor = pending[node][root]
            edge_id, target, factor = self._adjacency[node][cursor]
            support = best[node][root]
            cursor += 1
            if cursor == len(self._adjacency[node]):
                del pending[node][root]
                if not pending[node]:
                    del pending[node]
            else:
                pending[node][root] = cursor
            expansions += 1
            explored.add(edge_id)
            value = support.activation * factor
            previous = best.get(target, {}).get(root)
            if (
                target not in support.path_nodes and value > 0
                and (previous is None or value > previous.activation)
            ):
                best.setdefault(target, {})[root] = _Support(
                    value, (*support.path, edge_id), (*support.path_nodes, target),
                )
                activation[target] = _activation(best[target])
                if self._adjacency[target]:
                    pending.setdefault(target, {})[root] = 0
                    schedule(target)
            schedule(node)
        elapsed += max(0.0, self.monotonic() - started)
        payload = {
            "version": _VERSION,
            "graph_fingerprint": self.graph_fingerprint,
            "request_fingerprint": request,
            "supports": [
                {"node": list(node), "root_id": root, "activation": support.activation,
                 "path": list(support.path), "path_nodes": [list(value) for value in support.path_nodes]}
                for node, supports in sorted(best.items())
                for root, support in sorted(supports.items())
            ],
            "pending": [
                {"node": list(node), "root_id": root, "next_edge": cursor}
                for node, roots in sorted(pending.items()) for root, cursor in sorted(roots.items())
            ],
            "explored_edge_ids": sorted(explored),
            "expansions": expansions,
            "elapsed_seconds": elapsed,
        }
        payload["checksum"] = _digest(payload)
        results = [
            SpreadingNode(
                *node, activation[node],
                {root: support.activation for root, support in sorted(supports.items())},
                {root: list(support.path) for root, support in sorted(supports.items())},
            ) for node, supports in best.items()
        ]
        results.sort(key=lambda node: (-node.activation, node.node_type, node.node_id))
        return SpreadingResult(
            results, sorted(explored), expansions,
            sum(len(roots) for roots in pending.values()), status, elapsed, payload,
        )
