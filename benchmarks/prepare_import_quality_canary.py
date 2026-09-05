from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

from memory_demo.adapters import TextAdapter
from memory_demo.config import AppConfig
from memory_demo.ingestion.segmenter import NaturalSegmenter


_SPEAKER = re.compile(r"(?m)^([^:\n]{1,80}):(?:\s|$)")
_HEADING = re.compile(r"(?m)^\s*(?:[-=]{2,}|[#【\[])")
_QUOTED = re.compile(r"[\"'“”‘’「」『』]")


def _metrics(source_key: str, segment_index: int, raw_text: str) -> dict:
    speakers = {
        " ".join(value.split()).casefold()
        for value in _SPEAKER.findall(raw_text)
        if value.strip()
    }
    heading_count = len(_HEADING.findall(raw_text))
    quote_count = len(_QUOTED.findall(raw_text))
    return {
        "source_key": source_key,
        "segment_index": segment_index,
        "chars": len(raw_text),
        "speaker_count": len(speakers),
        "heading_count": heading_count,
        "quote_count": quote_count,
        "scene_score": heading_count * 8 + len(speakers),
        "attribution_score": len(speakers) * 4 + min(20, quote_count),
        "raw_text": raw_text,
    }


def _take(rows: list[dict], count: int, used: set[tuple[str, int]]) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        identity = (row["source_key"], row["segment_index"])
        if identity in used:
            continue
        used.add(identity)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a structurally stratified Source-segment import canary"
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="explicit regression anchor in SOURCE_KEY:SEGMENT_INDEX form",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--pack-one-file",
        action="store_true",
        help="write selected segments as independent paragraphs in one file",
    )
    args = parser.parse_args()

    config = AppConfig.from_env(args.env_file)
    adapter = TextAdapter()
    segmenter = NaturalSegmenter(config.segment)
    rows: list[dict] = []
    for path in sorted(args.input_root.rglob("*.txt")):
        source_key = path.relative_to(args.input_root).as_posix()
        blocks = adapter.read_blocks(path)
        for segment in segmenter.segment(source_key, blocks):
            rows.append(
                _metrics(source_key, segment.segment_index, segment.raw_text)
            )
    by_identity = {
        (row["source_key"], row["segment_index"]): row for row in rows
    }
    selected: list[dict] = []
    used: set[tuple[str, int]] = set()
    for raw_anchor in args.anchor:
        source_key, separator, raw_index = raw_anchor.rpartition(":")
        if not separator:
            raise ValueError(f"invalid anchor: {raw_anchor}")
        identity = (source_key, int(raw_index))
        if identity not in by_identity:
            raise ValueError(f"anchor not found: {raw_anchor}")
        used.add(identity)
        selected.append(by_identity[identity])

    count = max(1, int(args.per_stratum))
    strata = {
        "scene_switch": sorted(
            rows,
            key=lambda row: (
                row["scene_score"], row["chars"], row["source_key"],
                -row["segment_index"],
            ),
            reverse=True,
        ),
        "attribution": sorted(
            rows,
            key=lambda row: (
                row["attribution_score"], row["chars"], row["source_key"],
                -row["segment_index"],
            ),
            reverse=True,
        ),
        "long": sorted(
            rows,
            key=lambda row: (
                row["chars"], row["speaker_count"], row["source_key"],
                -row["segment_index"],
            ),
            reverse=True,
        ),
        "ordinary": sorted(
            (
                row
                for row in rows
                if row["chars"] >= 800
            ),
            key=lambda row: (
                row["scene_score"],
                abs(row["chars"] - 4_000),
                row["speaker_count"],
                row["source_key"],
                row["segment_index"],
            ),
        ),
    }
    membership: dict[tuple[str, int], list[str]] = {
        identity: ["regression_anchor"] for identity in used
    }
    for name, ranked in strata.items():
        additions = _take(ranked, count, used)
        selected.extend(additions)
        for row in additions:
            membership[(row["source_key"], row["segment_index"])] = [name]

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(
            f"output_root must be absent or empty: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    packed_blocks: list[str] = []
    for output_index, row in enumerate(selected):
        safe_stem = re.sub(r"[^0-9A-Za-z_.\-\u3400-\u9fff]+", "_", Path(row["source_key"]).stem)
        output_name = (
            f"{output_index:03d}__{safe_stem[:80]}__segment-{row['segment_index']:04d}.txt"
        )
        if args.pack_one_file:
            output_name = "quality-canary-packed.txt"
            packed_blocks.append(
                re.sub(r"\n\s*\n+", "\n", row["raw_text"]).strip()
            )
        else:
            (args.output_root / output_name).write_text(
                row["raw_text"].rstrip() + "\n", encoding="utf-8"
            )
        manifest_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "raw_text"
            }
            | {
                "strata": membership[
                    (row["source_key"], row["segment_index"])
                ],
                "canary_file": output_name,
                "canary_record": output_index if args.pack_one_file else None,
            }
        )
    if args.pack_one_file:
        (args.output_root / "quality-canary-packed.txt").write_text(
            "\n\n".join(packed_blocks) + "\n", encoding="utf-8"
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "config": asdict(config.segment),
                "input_root": str(args.input_root.resolve()),
                "output_root": str(args.output_root.resolve()),
                "available_segments": len(rows),
                "selected_segments": len(manifest_rows),
                "segments": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "available_segments": len(rows),
                "selected_segments": len(manifest_rows),
                "output_root": str(args.output_root.resolve()),
                "manifest": str(args.manifest.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
