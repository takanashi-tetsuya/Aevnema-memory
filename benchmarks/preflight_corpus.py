from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

from memory_demo.adapters import BlueArchiveJsonAdapter, TextAdapter
from memory_demo.adapters.base import logical_source_key
from memory_demo.config import AppConfig
from memory_demo.ingestion.ordering import natural_path_sort_key
from memory_demo.ingestion.segmenter import NaturalSegmenter


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Read-only TXT/Markdown/JSON import preflight"
    )
    parser.add_argument("story_root")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", help="optional UTF-8 JSON report path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "return exit code 2 for unsupported files, parse failures, empty "
            "files, or oversize segments"
        ),
    )
    args = parser.parse_args()
    input_path = Path(args.story_root).resolve()
    if not input_path.exists():
        raise SystemExit(f"story input does not exist: {input_path}")
    root = input_path if input_path.is_dir() else input_path.parent

    config = AppConfig.from_env(args.env_file)
    adapters = [BlueArchiveJsonAdapter(), TextAdapter()]
    segmenter = NaturalSegmenter(config.segment)
    all_files = (
        [input_path]
        if input_path.is_file()
        else sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=natural_path_sort_key,
        )
    )
    files: list[tuple[Path, object]] = []
    unsupported_files: list[str] = []
    for path in all_files:
        adapter = next(
            (candidate for candidate in adapters if candidate.supports(path)),
            None,
        )
        if adapter is None:
            unsupported_files.append(logical_source_key(path, root))
        else:
            files.append((path, adapter))
    category_stats = defaultdict(
        lambda: {"files": 0, "blocks": 0, "segments": 0, "bytes": 0}
    )
    languages: Counter[str] = Counter()
    segment_lengths: list[int] = []
    empty_files: list[str] = []
    failures: list[dict[str, str]] = []
    oversize_segments: list[dict[str, int | str]] = []

    for path, adapter in files:
        relative = path.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else "root"
        source_key = logical_source_key(path, root)
        stats = category_stats[category]
        stats["files"] += 1
        stats["bytes"] += path.stat().st_size
        try:
            blocks = adapter.read_blocks(path)
            segments = segmenter.segment(source_key, blocks)
        except Exception as exc:
            failures.append({"source_key": source_key, "error": str(exc)})
            continue
        if not blocks:
            empty_files.append(source_key)
        stats["blocks"] += len(blocks)
        stats["segments"] += len(segments)
        for block in blocks:
            languages.update(block.languages.keys())
        for segment in segments:
            length = len(segment.raw_text)
            segment_lengths.append(length)
            if length > config.segment.max_chars:
                oversize_segments.append(
                    {"source_key": source_key, "length": length}
                )

    report = {
        "input_root": str(root),
        "config": {
            "target_chars": config.segment.target_chars,
            "max_chars": config.segment.max_chars,
            "overlap_chars": config.segment.overlap_chars,
        },
        "totals": {
            "files": len(files),
            "parsed_files": len(files) - len(failures),
            "unsupported_files": len(unsupported_files),
            "blocks": sum(item["blocks"] for item in category_stats.values()),
            "segments": len(segment_lengths),
            "bytes": sum(item["bytes"] for item in category_stats.values()),
            "empty_files": len(empty_files),
            "parse_failures": len(failures),
            "oversize_segments": len(oversize_segments),
        },
        "categories": dict(sorted(category_stats.items())),
        "language_blocks": dict(languages.most_common()),
        "segment_length": {
            "min": min(segment_lengths, default=0),
            "mean": round(statistics.fmean(segment_lengths), 2)
            if segment_lengths else 0,
            "p50": percentile(segment_lengths, 0.50),
            "p95": percentile(segment_lengths, 0.95),
            "p99": percentile(segment_lengths, 0.99),
            "max": max(segment_lengths, default=0),
        },
        "empty_file_samples": empty_files[:20],
        "unsupported_file_samples": unsupported_files[:50],
        "failures": failures[:50],
        "oversize_samples": oversize_segments[:20],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    has_strict_failure = bool(
        unsupported_files or failures or empty_files or oversize_segments
    )
    return 2 if args.strict and has_strict_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
