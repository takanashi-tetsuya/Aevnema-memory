from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from memory_demo.adapters import BlueArchiveJsonAdapter, TextAdapter
from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.ingestion.ordering import infer_timeline_scope
from memory_demo.ingestion.segmenter import NaturalSegmenter
from memory_demo.llm import ModelClient


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Run real Episode extraction without writing a database"
    )
    parser.add_argument("input_file")
    parser.add_argument("--source-root")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    path = Path(args.input_file).resolve()
    root = Path(args.source_root).resolve() if args.source_root else path.parent
    source_key = path.relative_to(root).as_posix()
    config = AppConfig.from_env(args.env_file)
    config.ensure_directories()
    adapter = next(
        candidate
        for candidate in (BlueArchiveJsonAdapter(), TextAdapter())
        if candidate.supports(path)
    )
    segments = NaturalSegmenter(config.segment).segment(
        source_key, adapter.read_blocks(path)
    )
    logger = JsonlEventLogger(config.log_dir / "probe-extraction.jsonl")
    extractor = MemoryExtractor(ModelClient(config.model, logger), logger)
    output = []
    for segment in segments:
        episodes, errors = extractor.extract_episodes(
            segment.raw_text, infer_timeline_scope(source_key)
        )
        output.append(
            {
                "source_key": source_key,
                "segment_index": segment.segment_index,
                "source_chars": len(segment.raw_text),
                "episodes": [asdict(episode) for episode in episodes],
                "errors": errors,
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
