"""Read-only whole-database structural exploration diagnostic; not a recall benchmark.

The fixed cues are Episode IDs from different source fragments, not semantic
user questions. Reachable raw sources have not been verified as useful evidence.
No gold, learning, model call, schema migration, or MemoryApplication is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from memory_demo.associations.spreading import SpreadingRecall, SpreadingSeed
from memory_demo.associations.traversal import GraphTraverser
from memory_demo.types import SearchHit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class SnapshotNeighbours:
    """Read-only equivalent of repository neighbours, with deterministic ID ties."""

    def __init__(self, edges):
        self.adjacency = defaultdict(list)
        for row in sorted(edges, key=lambda row: (-row["weight"], -row["confidence"], row["id"])):
            for endpoint in {(row["from_type"], row["from_id"]), (row["to_type"], row["to_id"])}:
                self.adjacency[endpoint].append(row)

    def neighbors_many(self, node_type, node_ids, limit=100):
        return {node_id: self.adjacency[node_type, node_id][:limit] for node_id in node_ids}


def reachable(nodes, episodes, nonempty_sources):
    episode_ids = {node.node_id for node in nodes if node.node_type == "episode"}
    source_ids = {episodes[episode_id]["source_id"] for episode_id in episode_ids}
    return {
        "discovered_nodes": len(nodes),
        "discovered_episodes": len(episode_ids),
        "reachable_raw_sources": len(source_ids & nonempty_sources),
    }


def comparable(checkpoint):
    return {key: value for key, value in checkpoint.items() if key not in {"elapsed_seconds", "checksum"}}


def run(database: Path, output: Path, sample_seconds: float, skip_legacy: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    before = sha256(database)
    loaded_at = time.perf_counter()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        episodes = {row["id"]: dict(row) for row in connection.execute(
            "SELECT id, source_id, source_key FROM episode ORDER BY id"
        )}
        nodes = [("episode", identifier) for identifier in episodes]
        nodes.extend(("concept", row[0]) for row in connection.execute("SELECT id FROM concept ORDER BY id"))
        source_lengths = {row[0]: row[1] for row in connection.execute("SELECT id, LENGTH(raw_text) FROM source")}
        edges = [dict(row) for row in connection.execute(
            "SELECT id, from_type, from_id, to_type, to_id, weight, confidence, polarity, "
            "relation_type, relation_key, relation_text, association_mode, generation, "
            "claim_level, audit_status, created_reason FROM association ORDER BY id"
        )]
    # sqlite3's context manager controls transactions; explicitly close the connection.
    connection.close()
    nonempty_sources = {identifier for identifier, length in source_lengths.items() if length > 0}
    graph = SpreadingRecall(nodes, edges)
    snapshot_seconds = time.perf_counter() - loaded_at
    adapter = SnapshotNeighbours(edges)
    positive_neighbours = defaultdict(set)
    for edge in edges:
        source, target = (edge["from_type"], edge["from_id"]), (edge["to_type"], edge["to_id"])
        if source != target and edge["weight"] * edge["confidence"] > 0:
            positive_neighbours[source].add(target)
            positive_neighbours[target].add(source)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_kind": "structural_exploration_without_semantic_query_or_gold",
        "database": str(database),
        "baseline_sha256_before": before,
        "read_only": {"sqlite_uri_mode": "ro", "query_only": True, "model_calls": 0,
                      "learned_edges": 0, "database_migrations": 0},
        "snapshot": {"sources": len(source_lengths), "episodes": len(episodes),
                     "concepts": len(nodes) - len(episodes), "edges": len(edges),
                     "isolated_episodes": sum(not positive_neighbours["episode", identifier] for identifier in episodes),
                     "load_and_graph_build_seconds": snapshot_seconds,
                     "graph_fingerprint": graph.graph_fingerprint},
        "method": {
            "seed_selection": "For each source_key category, first two Episodes by ID from distinct Source IDs with positive-conductance degree > 0. Isolated candidates skipped before selection are recorded.",
            "independence": "Source IDs are distinct structural roots; semantic independence has not been judged.",
            "cumulative_expansion_budgets": [128, 512, 2048, 8192],
            "per_sample_seconds": sample_seconds,
            "damping": 0.85, "degree_exponent": 0.5,
            "equivalence": "Every resumed stage compared with a fresh run at the same total expansion budget; ignores elapsed_seconds and checksum only.",
            "reachable_raw_sources": "Distinct nonempty Source rows referenced by discovered Episodes, without relevance or source-entailment verification.",
            "merged_targets": "Discovered nodes with positive support from both distinct Source-root IDs.",
            "legacy": "Unchanged GraphTraverser, beam_width=20, max_hops=3, memory-only neighbour adapter sorted weight/confidence/ID.",
            "limitations": ["No semantic question, answer, gold relations, source verification, learning, or repeat-question speedup evaluation.",
                            "Additional graph reach is not key-relation recall and may include irrelevant nodes.",
                            "Timing is a single local diagnostic; shared snapshot construction is measured separately.",
                            "Degree damping and tiny nonzero activations retain structural paths without imposing a relevance threshold."],
        },
        "samples": [],
    }
    for category in ("event", "main", "favor"):
        sample_started = time.perf_counter()
        selected, seen_sources, skipped_isolated = [], set(), []
        for episode in episodes.values():
            if episode["source_key"].startswith(category + "/") and episode["source_id"] not in seen_sources:
                degree = len(positive_neighbours["episode", episode["id"]])
                if degree == 0:
                    skipped_isolated.append(episode)
                    continue
                selected.append({**episode, "positive_conductance_degree": degree})
                seen_sources.add(episode["source_id"])
                if len(selected) == 2:
                    break
        if len(selected) != 2:
            raise ValueError(f"not enough distinct sources for {category}")
        seeds = [SpreadingSeed("episode", item["id"], root_id=f"source:{item['source_id']}") for item in selected]
        sample = {"category": category, "seeds": selected,
                  "skipped_isolated_candidates": skipped_isolated, "stages": []}
        previous = None
        for target_steps in (128, 512, 2048, 8192):
            remaining = max(0.0, sample_seconds - (time.perf_counter() - sample_started))
            stage_started = time.perf_counter()
            result = graph.search(seeds, max_expansions=target_steps - (previous.expansions if previous else 0),
                                  max_seconds=remaining,
                                  checkpoint=json.loads(encode(previous.checkpoint)) if previous else None)
            stage_seconds = time.perf_counter() - stage_started
            checkpoint_bytes = encode(result.checkpoint)
            remaining = max(0.0, sample_seconds - (time.perf_counter() - sample_started))
            replay_started = time.perf_counter()
            replay = graph.search(seeds, max_expansions=result.expansions, max_seconds=remaining)
            replay_seconds = time.perf_counter() - replay_started
            stage = {
                "target_cumulative_steps": target_steps,
                "actual_cumulative_steps": result.expansions,
                "stage_steps": result.expansions - (previous.expansions if previous else 0),
                "stage_seconds_including_checkpoint_restore_and_build": stage_seconds,
                "fresh_replay_seconds_including_checkpoint_build": replay_seconds,
                "status": result.status, "pending_root_node_states": result.pending_count,
                "explored_unique_edges": len(result.explored_edge_ids),
                "merged_targets": sum(len(item.contributions) >= 2 for item in result.nodes),
                "checkpoint_bytes": len(checkpoint_bytes),
                "continuation_equivalent": comparable(result.checkpoint) == comparable(replay.checkpoint),
                "fresh_replay_expansions": replay.expansions,
                **reachable(result.nodes, episodes, nonempty_sources),
            }
            sample["stages"].append(stage)
            print(json.dumps({"category": category, **stage}), flush=True)
            previous = result
            if result.status == "time_budget" or time.perf_counter() - sample_started >= sample_seconds:
                sample["time_limit_reached"] = True
                break
        (output / f"{category}_checkpoint.json").write_bytes(encode(previous.checkpoint))
        if not skip_legacy:
            legacy_started = time.perf_counter()
            legacy_nodes, legacy_paths = GraphTraverser(adapter).expand(
                [SearchHit("episode", item["id"], 1.0) for item in selected], beam_width=20, max_hops=3,
            )
            sample["legacy_3hop"] = {
                "seconds": time.perf_counter() - legacy_started,
                "inspected_path_records": len(legacy_paths),
                "explored_unique_edges": len({item["association_id"] for item in legacy_paths}),
                **reachable(legacy_nodes, episodes, nonempty_sources),
            }
        sample["total_sample_seconds"] = time.perf_counter() - sample_started
        report["samples"].append(sample)
    report["baseline_sha256_after"] = sha256(database)
    report["baseline_unchanged"] = before == report["baseline_sha256_after"]
    report["all_continuations_equivalent"] = all(
        stage["continuation_equivalent"] for sample in report["samples"] for stage in sample["stages"]
    )
    (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Whole-database structural exploration diagnostic", "",
             "These are fixed Episode-ID cues, not semantic user questions. There is no gold, answer, source-entailment check, learning, or recall percentage.", "",
             f"Snapshot: {len(source_lengths):,} Sources, {len(episodes):,} Episodes, {len(nodes) - len(episodes):,} Concepts, {len(edges):,} edges. Load/build: {snapshot_seconds:.3f}s.", "",
             f"The snapshot contains {report['snapshot']['isolated_episodes']:,} Episodes without a positive-conductance neighbour. Cue selection uses the first two nonisolated Episodes by ID from different Sources in each category; skipped isolated candidates are recorded in diagnostic.json.", "",
             "Reachable Sources are distinct nonempty raw-text rows referenced by discovered Episodes. They are not confirmed relevant evidence. Merged targets receive nonzero support from both distinct Source roots.", "",
             "| Category | Steps | Reachable Sources | Episodes | Merged targets | Pending states | Stage seconds | Checkpoint KiB | Resume equal |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for sample in report["samples"]:
        for stage in sample["stages"]:
            lines.append(f"| {sample['category']} | {stage['actual_cumulative_steps']} | {stage['reachable_raw_sources']} | {stage['discovered_episodes']} | {stage['merged_targets']} | {stage['pending_root_node_states']} | {stage['stage_seconds_including_checkpoint_restore_and_build']:.4f} | {stage['checkpoint_bytes'] / 1024:.1f} | {stage['continuation_equivalent']} |")
    lines.extend(["", "Stage time includes checkpoint restore/build; per-stage fresh replay timing is recorded in diagnostic.json. Checkpoint file output is included in each sample's total, not in stage timing.", ""])
    for sample in report["samples"]:
        cue_text = ", ".join(f"Episode {item['id']} / Source {item['source_id']} / `{item['source_key']}`" for item in sample["seeds"])
        lines.append(f"- {sample['category']} cues: {cue_text}. Total sample time: {sample['total_sample_seconds']:.3f}s.")
        if "legacy_3hop" in sample:
            legacy = sample["legacy_3hop"]
            lines.append(f"  Legacy 20-beam / 3-hop: {legacy['reachable_raw_sources']} reachable Sources, {legacy['discovered_episodes']} Episodes, {legacy['explored_unique_edges']} edges, {legacy['seconds']:.4f}s. This is a structural comparison with different budgets and scoring, not a speedup claim.")
    lines.extend(["", f"All continuation comparisons equal: **{report['all_continuations_equivalent']}**.",
                  f"Baseline unchanged: **{report['baseline_unchanged']}**.", "",
                  f"Before SHA-256: `{before}`", f"After SHA-256: `{report['baseline_sha256_after']}`", "",
                  "No HTTP/model calls, schema migrations, learned edges, or database writes. SQLite was opened with mode=ro and query_only enabled.", "",
                  "The next evidence needed is an independently judged semantic question set with source-backed required relations, followed by matched-quality first/second-search comparisons. This diagnostic does not establish either semantic recall or learned-edge speedup."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not report["baseline_unchanged"] or not report["all_continuations_equivalent"]:
        raise RuntimeError("diagnostic invariant failed; inspect diagnostic.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT / "validation/v3-freeze-20260906T233016Z/testing_kb.sqlite")
    parser.add_argument("--output", type=Path, default=PROJECT / "validation/progressive-recall-20260911")
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()
    if not 0 < args.sample_seconds <= 60:
        parser.error("--sample-seconds must be in (0, 60]")
    report = run(args.database.resolve(), args.output.resolve(), args.sample_seconds, args.skip_legacy)
    print(json.dumps({"baseline_unchanged": report["baseline_unchanged"],
                      "all_continuations_equivalent": report["all_continuations_equivalent"],
                      "report": str(args.output.resolve() / "REPORT.md")}), flush=True)


if __name__ == "__main__":
    main()
