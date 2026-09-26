from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from memory_demo.adapters import BlueArchiveJsonAdapter, TextAdapter
from memory_demo.adapters.base import logical_source_key
from memory_demo.app import MemoryApplication
from memory_demo.chronology import ChronologyService
from memory_demo.chronology.review import TimelineReviewer
from memory_demo.config import AppConfig
from memory_demo.evaluation import run_evaluation
from memory_demo.ingestion.paragraphs import ParagraphSegmenter
from memory_demo.ingestion.segmenter import NaturalSegmenter
from memory_demo.interfaces.telegram_bot import TelegramBot
from memory_demo.types import AssociationDraft


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _row_dict(row) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, bytes):
            result[key] = {
                "dtype": "float32",
                "bytes_length": len(value),
                "dimensions": len(value) // 4,
            }
    return result


def _configure_console_encoding() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory-demo")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--database", help="working SQLite database; use a copy for experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="initialize the database")
    prepare = subparsers.add_parser("prepare", help="parse and segment without API calls")
    prepare.add_argument("path")
    import_parser = subparsers.add_parser("import", help="import JSON/TXT memories")
    import_parser.add_argument("path")
    import_parser.add_argument(
        "--source-root",
        help="root used to derive source_key while importing a file or subtree",
    )
    import_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return exit code 0 even when some files or stages fail",
    )
    import_parser.add_argument(
        "--defer-inference-relations",
        action="store_true",
        help=(
            "persist Source, Episode, Concept and direct evidence links, but "
            "defer LLM-derived Episode/Episode and Concept/Concept relations"
        ),
    )
    query = subparsers.add_parser("query", help="query the memory network")
    query.add_argument("question")
    query.add_argument("--json", action="store_true")
    query.add_argument("--mode", choices=["light", "standard", "deep", "max_effort"], help="progressive recall: deep 6 minutes, max_effort 30 minutes")
    query.add_argument("--context", default="", help="visible context for progressive recall")
    query.add_argument("--resume", help="resume a progressive recall session id")
    query.add_argument("--timeout", type=float, help="shorter request deadline in seconds")
    query.add_argument("--no-learn", action="store_true", help="retrieve and verify without changing association strengths")
    feedback = subparsers.add_parser("recall-feedback", help="rate the actual edges used by a saved recall")
    feedback.add_argument("session_id")
    feedback.add_argument("verdict", choices=["positive", "negative"])
    feedback.add_argument("--feedback-id", required=True, help="stable event id; retries do not repeat learning")
    subparsers.add_parser("stats")
    subparsers.add_parser("rebuild-index")
    contextual = subparsers.add_parser(
        "contextual-cues", help="maintain local context/need association prototypes"
    )
    contextual_sub = contextual.add_subparsers(
        dest="contextual_command", required=True
    )
    contextual_sub.add_parser("stats")
    contextual_sub.add_parser("audit")
    prune = contextual_sub.add_parser("prune")
    prune.add_argument("--dry-run", action="store_true")
    contextual_sub.add_parser("rebuild-local-index")
    migrate = contextual_sub.add_parser("migrate-legacy")
    migrate.add_argument("--offline", action="store_true")
    subparsers.add_parser(
        "backfill-paragraphs",
        help="embed deterministic Paragraphs for existing Sources",
    )
    subparsers.add_parser(
        "augment-concepts",
        help="run the configured Concept profile over existing Episodes",
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("questions_json")
    evaluate.add_argument("output_dir")
    evaluate.add_argument(
        "--modes",
        nargs="+",
        choices=["vector_only", "graph_static", "graph_growing"],
        help="run only selected evaluation modes",
    )
    evaluate.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted report and skip completed questions",
    )
    evaluate.add_argument(
        "--question-order",
        choices=["original", "reverse", "interleaved"],
        default="original",
        help="question order for graph path-dependence evaluation",
    )
    evaluate.add_argument(
        "--query-plans-from",
        help=(
            "reuse frozen query-plans from another evaluation directory; "
            "missing or incompatible plans fail closed"
        ),
    )

    timeline = subparsers.add_parser("timeline")
    timeline_sub = timeline.add_subparsers(dest="timeline_command", required=True)
    export = timeline_sub.add_parser("export")
    export.add_argument("scope")
    export.add_argument("--output")
    set_order = timeline_sub.add_parser("set-order")
    set_order.add_argument("episode_id", type=int)
    set_order.add_argument("story_order", type=float)
    set_order.add_argument("scope")
    before = timeline_sub.add_parser("add-before")
    before.add_argument("earlier_episode_id", type=int)
    before.add_argument("later_episode_id", type=int)
    before.add_argument("relation_text")

    association = subparsers.add_parser("association")
    association_sub = association.add_subparsers(dest="association_command", required=True)
    add = association_sub.add_parser("add")
    add.add_argument("from_type", choices=["episode", "concept"])
    add.add_argument("from_id", type=int)
    add.add_argument("to_type", choices=["episode", "concept"])
    add.add_argument("to_id", type=int)
    add.add_argument("relation_type")
    add.add_argument("relation_key")
    add.add_argument("relation_text")
    add.add_argument("--weight", type=float, default=1.0)
    add.add_argument("--confidence", type=float, default=1.0)
    add.add_argument(
        "--generation",
        type=int,
        default=0,
        help="0=直接经验关系；n+1=使用最高 generation 为 n 的推断作为前提",
    )
    add.add_argument("--negative", action="store_true")
    inspect_association = association_sub.add_parser("inspect")
    inspect_association.add_argument("node_type", choices=["episode", "concept"])
    inspect_association.add_argument("node_id", type=int)
    inspect_association.add_argument("--limit", type=int, default=100)
    delete_association = association_sub.add_parser("delete")
    delete_association.add_argument("association_id", type=int)
    delete_association.add_argument("--yes", action="store_true")

    concept = subparsers.add_parser("concept")
    concept_sub = concept.add_subparsers(dest="concept_command", required=True)
    merge = concept_sub.add_parser("merge")
    merge.add_argument("concept_id", type=int)
    merge.add_argument("canonical_concept_id", type=int)
    unmerge = concept_sub.add_parser("unmerge")
    unmerge.add_argument("concept_id", type=int)
    inspect_concept = concept_sub.add_parser("inspect")
    inspect_concept.add_argument("concept_id", type=int)

    episode = subparsers.add_parser("episode")
    episode_sub = episode.add_subparsers(dest="episode_command", required=True)
    inspect_episode = episode_sub.add_parser("inspect")
    inspect_episode.add_argument("episode_id", type=int)
    delete_episode = episode_sub.add_parser("delete")
    delete_episode.add_argument("episode_id", type=int)
    delete_episode.add_argument("--yes", action="store_true")
    subparsers.add_parser("bot")
    return parser


def _prepare(path_value: str, config: AppConfig) -> dict:
    path = Path(path_value)
    root = path if path.is_dir() else path.parent
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    adapters = [BlueArchiveJsonAdapter(), TextAdapter()]
    segmenter = NaturalSegmenter(config.segment)
    paragraph_segmenter = ParagraphSegmenter(config.paragraph)
    result = {"files": [], "total_segments": 0}
    for file_path in files:
        adapter = next((item for item in adapters if item.supports(file_path)), None)
        if adapter is None:
            continue
        source_key = logical_source_key(file_path, root)
        blocks = adapter.read_blocks(file_path)
        segments = segmenter.segment(source_key, blocks)
        result["files"].append(
            {
                "source_key": source_key,
                "blocks": len(blocks),
                "segments": len(segments),
                "segment_lengths": [len(segment.raw_text) for segment in segments],
                "paragraphs_per_segment": [
                    len(paragraph_segmenter.segment(segment.raw_text))
                    for segment in segments
                ],
                "paragraph_lengths": [
                    [
                        len(paragraph.text)
                        for paragraph in paragraph_segmenter.segment(segment.raw_text)
                    ]
                    for segment in segments
                ],
            }
        )
        result["total_segments"] += len(segments)
    return result


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env(args.env_file)
    if args.database:
        config.database_path = Path(args.database)
    if args.command == "prepare":
        _print_json(_prepare(args.path, config))
        return 0
    # Evaluation must copy the evidence baseline before opening an application.
    # MemoryApplication initializes/migrates its database in __init__, so creating
    # it here would silently mutate the supposedly frozen source database.
    if args.command == "evaluate":
        _print_json(
            run_evaluation(
                config,
                args.questions_json,
                args.output_dir,
                selected_modes=args.modes,
                resume=args.resume,
                question_order=args.question_order,
                query_plans_from=args.query_plans_from,
            )
        )
        return 0

    app = MemoryApplication(config)
    if args.command == "recall-feedback":
        _print_json(app.recall_feedback(args.session_id, positive=args.verdict == "positive", feedback_id=args.feedback_id))
        return 0
    if args.command == "init":
        print(f"initialized {config.database_path}")
        return 0
    if args.command == "import":
        if args.defer_inference_relations:
            config.ingestion.build_inference_relations = False
        result = app.import_path(args.path, source_root=args.source_root)
        _print_json(result)
        if args.allow_partial:
            return 0
        return 0 if result.get("status") == "completed" else 2
    if args.command == "rebuild-index":
        app.rebuild_indexes()
        _print_json(app.stats())
        return 0
    if args.command == "contextual-cues":
        if args.contextual_command == "stats":
            rows = app.associations.list_contextual()
            _print_json(
                {
                    "association": app.associations.stats(),
                    "cue_prototypes": len(app.associations.list_cue_prototypes()),
                    "context_prototypes": len(
                        app.associations.list_cue_prototypes(cue_kind="context")
                    ),
                    "need_prototypes": len(
                        app.associations.list_cue_prototypes(cue_kind="need")
                    ),
                    "contextual_rows": len(rows),
                    "context_index_count": app.context_cue_index.count,
                    "need_index_count": app.need_cue_index.count,
                    "external_calls": 0,
                }
            )
        elif args.contextual_command == "audit":
            _print_json(app.db.audit_contextual_storage())
        elif args.contextual_command == "prune":
            if args.dry_run:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc).isoformat()
                candidates = [
                    row
                    for row in app.associations.list_contextual("probation")
                    if row["expires_at"] and str(row["expires_at"]) <= now
                ]
                _print_json({"dry_run": True, "would_retire": len(candidates)})
            else:
                _print_json({"retired": app.associations.prune_expired()})
        elif args.contextual_command == "rebuild-local-index":
            _print_json(app.rebuild_contextual_indexes())
        elif args.contextual_command == "migrate-legacy":
            if not args.offline:
                raise SystemExit("migrate-legacy requires --offline")
            _print_json(
                {
                    "offline": True,
                    "migrated": 0,
                    "note": "legacy relation cues are retained; no vectors were regenerated",
                }
            )
        return 0
    if args.command == "backfill-paragraphs":
        _print_json(app.backfill_paragraphs())
        return 0
    if args.command == "augment-concepts":
        _print_json(app.augment_concepts())
        return 0
    if args.command == "stats":
        _print_json(app.stats())
        return 0
    if args.command == "query":
        from memory_demo.retrieval.recall_policy import requests_maximum_recall

        if args.mode or args.resume or requests_maximum_recall(args.question):
            result = app.recall(
                args.question, mode=args.mode, context=args.context,
                resume=args.resume, timeout_seconds=args.timeout,
                learn=not args.no_learn,
            )
            if args.json:
                _print_json(result)
            else:
                print(result["answer"] or "尚未取得通过原文核验的证据。")
                print(f"\n回想状态：{result['status']}；会话：{result['session_id']}")
                if result["missing_requirements"]:
                    print("尚未补齐：" + "；".join(result["missing_requirements"]))
            return 0 if result["complete"] else 2
        if args.timeout is not None or args.context or args.no_learn:
            raise SystemExit("--timeout/--context/--no-learn require --mode for progressive recall")
        app.rebuild_indexes()
        result = app.query_engine().query(args.question)
        _print_json(result) if args.json else print(result["answer"])
        return 0
    if args.command == "timeline":
        logger = app.new_logger("review")
        service = ChronologyService(app.episodes, app.associations, logger)
        if args.timeline_command == "export":
            content = TimelineReviewer(
                app.episodes, app.sources, app.associations
            ).export_markdown(args.scope)
            if args.output:
                Path(args.output).write_text(content, encoding="utf-8")
            else:
                print(content)
        elif args.timeline_command == "set-order":
            service.set_order(args.episode_id, args.story_order, args.scope)
        elif args.timeline_command == "add-before":
            print(service.add_temporal_relation(
                args.earlier_episode_id, args.later_episode_id, args.relation_text
            ))
        return 0
    if args.command == "association":
        logger = app.new_logger("manual")
        if args.association_command == "add":
            association_id = app.associations.upsert(
                AssociationDraft(
                    from_type=args.from_type,
                    from_id=args.from_id,
                    to_type=args.to_type,
                    to_id=args.to_id,
                    relation_type=args.relation_type,
                    relation_key=args.relation_key,
                    relation_text=args.relation_text,
                    polarity=-1 if args.negative else 1,
                    weight=args.weight,
                    confidence=args.confidence,
                    generation=args.generation,
                    created_reason="CLI人工修改",
                )
            )
            logger.emit("manual_association_added", association_id=association_id)
            print(association_id)
        elif args.association_command == "inspect":
            _print_json(
                [
                    dict(row)
                    for row in app.associations.neighbors(
                        args.node_type, args.node_id, args.limit
                    )
                ]
            )
        elif args.association_command == "delete":
            if not args.yes:
                raise SystemExit("删除 Association 必须显式传入 --yes")
            deleted = app.associations.delete(args.association_id)
            logger.emit(
                "manual_association_deleted",
                association_id=args.association_id,
                deleted=deleted,
            )
            print("deleted" if deleted else "not found")
        return 0
    if args.command == "concept":
        logger = app.new_logger("manual")
        if args.concept_command == "merge":
            app.concepts.merge(args.concept_id, args.canonical_concept_id)
            logger.emit(
                "manual_concept_merged",
                concept_id=args.concept_id,
                canonical_concept_id=args.canonical_concept_id,
            )
        elif args.concept_command == "unmerge":
            app.concepts.unmerge(args.concept_id)
            logger.emit("manual_concept_unmerged", concept_id=args.concept_id)
        elif args.concept_command == "inspect":
            _print_json(
                {
                    "concept": _row_dict(app.concepts.get(args.concept_id)),
                    "aliases": [
                        dict(row) for row in app.concepts.list_aliases(args.concept_id)
                    ],
                    "associations": [
                        dict(row)
                        for row in app.associations.neighbors(
                            "concept", args.concept_id
                        )
                    ],
                }
            )
        return 0
    if args.command == "episode":
        logger = app.new_logger("manual")
        if args.episode_command == "inspect":
            row = app.episodes.get(args.episode_id)
            source = app.sources.get(int(row["source_id"])) if row else None
            _print_json(
                {
                    "episode": _row_dict(row),
                    "source": _row_dict(source),
                    "associations": [
                        dict(item)
                        for item in app.associations.neighbors(
                            "episode", args.episode_id
                        )
                    ],
                }
            )
        elif args.episode_command == "delete":
            if not args.yes:
                raise SystemExit("删除 Episode 及其 Association 必须显式传入 --yes")
            deleted = app.episodes.delete(args.episode_id)
            app.episode_index.remove(args.episode_id)
            logger.emit(
                "manual_episode_deleted",
                episode_id=args.episode_id,
                deleted=deleted,
            )
            print("deleted" if deleted else "not found")
        return 0
    if args.command == "bot":
        app.rebuild_indexes()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        TelegramBot(token, app.query_engine(app.new_logger("telegram"))).run()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
