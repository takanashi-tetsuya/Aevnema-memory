from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from memory_demo.database import Database


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _query_plan_path(root: Path, group: str, index: int, question_id: str) -> Path:
    fingerprint = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:10]
    return root / "query-plans" / group / f"{index:04d}-{fingerprint}.json"


def load_questions(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation questions must be a JSON list")
    questions: list[dict] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            questions.append({"id": str(index + 1), "question": item})
        elif isinstance(item, dict) and str(item.get("question", "")).strip():
            questions.append(
                {
                    "id": str(item.get("id", index + 1)),
                    "question": str(item["question"]),
                    "expected_capability": str(item.get("expected_capability", "")),
                }
            )
    return questions


def run_evaluation(
    config: AppConfig,
    questions_path: str | Path,
    output_dir: str | Path,
    selected_modes: list[str] | None = None,
    resume: bool = False,
    question_order: str = "original",
    query_plans_from: str | Path | None = None,
) -> dict:
    questions = load_questions(questions_path)
    if question_order == "reverse":
        questions = list(reversed(questions))
    elif question_order == "interleaved":
        questions = [*questions[::2], *questions[1::2]]
    elif question_order != "original":
        raise ValueError(f"unknown question order: {question_order}")
    root = Path(output_dir)
    external_plan_root = (
        Path(query_plans_from) if query_plans_from is not None else None
    )
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "evaluation-report.json"
    if not config.database_path.exists():
        raise FileNotFoundError(config.database_path)
    source_database = Database(config.database_path)
    available_modes = {
        "vector_only": {"graph_max_hops": 0, "growth_max_rounds": 0},
        "graph_static": {
            "graph_max_hops": config.retrieval.graph_max_hops,
            "growth_max_rounds": 0,
        },
        "graph_growing": {
            "graph_max_hops": config.retrieval.graph_max_hops,
            "growth_max_rounds": config.retrieval.growth_max_rounds,
            "growth_episode_limit": config.retrieval.growth_episode_limit,
        },
    }
    if selected_modes:
        unknown = sorted(set(selected_modes).difference(available_modes))
        if unknown:
            raise ValueError(f"unknown evaluation modes: {unknown}")
        modes = {
            name: available_modes[name]
            for name in selected_modes
        }
    else:
        modes = available_modes
    fresh_report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_database": str(config.database_path.resolve()),
        "questions": questions,
        "mode_order": list(modes),
        "question_order": question_order,
        "configuration": {
            "prompt_version": config.prompt_version,
            "embedding_model": config.model.embedding_model,
            "reasoning_model": config.model.reasoning_model,
            "fallback_model": config.model.fallback_model,
            "embedding_dimension": config.model.embedding_dimension,
            "episode_top_k": config.retrieval.episode_top_k,
            "concept_top_k": config.retrieval.concept_top_k,
            "paragraph_enabled": config.paragraph.enabled,
            "paragraph_top_k": config.retrieval.paragraph_top_k,
            "paragraph_episode_expansion_limit": (
                config.retrieval.paragraph_episode_expansion_limit
            ),
            "paragraph_rrf_weight": config.retrieval.paragraph_rrf_weight,
            "concept_extraction_profile": config.concept_extraction.profile,
            "episode_relation_candidate_k": config.retrieval.episode_relation_candidate_k,
            "graph_beam_width": config.retrieval.graph_beam_width,
            "graph_max_hops": config.retrieval.graph_max_hops,
            "growth_max_rounds": config.retrieval.growth_max_rounds,
            "growth_persist_only_used": (
                config.retrieval.growth_persist_only_used
            ),
            "growth_counterfactual_utility_enabled": (
                config.retrieval.growth_counterfactual_utility_enabled
            ),
            "growth_staging_enabled": config.retrieval.growth_staging_enabled,
            "source_key_cohort_enabled": (
                config.retrieval.source_key_cohort_enabled
            ),
            "source_key_cohort_min_anchor_hits": (
                config.retrieval.source_key_cohort_min_anchor_hits
            ),
            "source_key_cohort_max_keys": (
                config.retrieval.source_key_cohort_max_keys
            ),
            "source_key_cohort_max_episodes_per_key": (
                config.retrieval.source_key_cohort_max_episodes_per_key
            ),
            "source_key_cohort_total_limit": (
                config.retrieval.source_key_cohort_total_limit
            ),
            "rerank_frozen_before_growth": True,
            "answer_episode_limit": config.retrieval.answer_episode_limit,
            "answer_concept_limit": config.retrieval.answer_concept_limit,
            "answer_path_limit": config.retrieval.answer_path_limit,
            "answer_whole_question_anchor_episodes": (
                config.retrieval.answer_whole_question_anchor_episodes
            ),
            "answer_anchor_episodes_per_query": (
                config.retrieval.answer_anchor_episodes_per_query
            ),
            "rerank_atomic_floor_enabled": (
                config.retrieval.rerank_atomic_floor_enabled
            ),
            "rerank_atomic_floor_query_limit": (
                config.retrieval.rerank_atomic_floor_query_limit
            ),
            "rerank_atomic_floor_per_query": (
                config.retrieval.rerank_atomic_floor_per_query
            ),
            "rerank_atomic_floor_total_limit": (
                config.retrieval.rerank_atomic_floor_total_limit
            ),
            "rerank_constraint_floor_per_query": (
                config.retrieval.rerank_constraint_floor_per_query
            ),
            "rerank_constraint_floor_total_limit": (
                config.retrieval.rerank_constraint_floor_total_limit
            ),
            "rerank_question_sparse_floor_limit": (
                config.retrieval.rerank_question_sparse_floor_limit
            ),
            "rerank_combined_floor_limit": (
                config.retrieval.rerank_combined_floor_limit
            ),
            "query_plan_mode": "shared_frozen_v1",
            "query_plans_from": (
                str(external_plan_root.resolve())
                if external_plan_root is not None
                else None
            ),
        },
        "query_plans": {
            "schema": "frozen_query_plan_v1",
            "mode_groups": {
                mode: f"hops-{int(settings['graph_max_hops'])}"
                for mode, settings in modes.items()
            },
            "groups": {},
        },
        "status": "running",
        "modes": {},
    }
    if resume:
        if not report_path.exists():
            raise FileNotFoundError(f"cannot resume missing report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("questions") != questions:
            raise ValueError("resume questions do not match the existing report")
        if report.get("mode_order") != list(modes):
            raise ValueError("resume modes do not match the existing report")
        if report.get("question_order", "original") != question_order:
            raise ValueError("resume question order does not match the existing report")
        if report.get("source_database") != str(config.database_path.resolve()):
            raise ValueError("resume source database does not match the existing report")
        report["status"] = "running"
        for key in ("completed_at", "failed_at", "error_type", "error"):
            report.pop(key, None)
    else:
        report = fresh_report
    _write_report(report_path, report)
    try:
        plan_groups: dict[str, dict] = {}
        for mode, settings in modes.items():
            group = f"hops-{int(settings['graph_max_hops'])}"
            plan_groups.setdefault(group, settings)
        plans_by_group: dict[str, dict[str, dict]] = {}
        report.setdefault(
            "query_plans",
            {
                "schema": "frozen_query_plan_v1",
                "mode_groups": {
                    mode: f"hops-{int(settings['graph_max_hops'])}"
                    for mode, settings in modes.items()
                },
                "groups": {},
            },
        )
        report["configuration"].setdefault(
            "query_plan_mode", "shared_frozen_v1"
        )
        for group, settings in plan_groups.items():
            planner_config = deepcopy(config)
            planner_config.database_path = root / "query-plans" / f"{group}.db"
            planner_config.log_dir = root / "logs" / "query_planner" / group
            planner_config.retrieval.graph_max_hops = settings["graph_max_hops"]
            planner_config.retrieval.growth_max_rounds = 0
            group_plans: dict[str, dict] = {}
            group_records: list[dict] = []
            all_plan_files_exist = all(
                _query_plan_path(root, group, index, str(question["id"])).exists()
                for index, question in enumerate(questions, start=1)
            )
            if not (resume and planner_config.database_path.exists() and all_plan_files_exist):
                source_database.backup_to(planner_config.database_path)
            planner_app = MemoryApplication(planner_config)
            planner_app.rebuild_indexes()
            planner_engine = planner_app.query_engine(
                planner_app.new_logger("query-plan")
            )
            for index, question in enumerate(questions, start=1):
                question_id = str(question["id"])
                plan_path = _query_plan_path(root, group, index, question_id)
                if resume and plan_path.exists():
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    planner_engine._validate_query_plan(question["question"], plan)
                elif external_plan_root is not None:
                    source_plan_path = _query_plan_path(
                        external_plan_root, group, index, question_id
                    )
                    if not source_plan_path.exists():
                        raise FileNotFoundError(
                            f"missing external frozen query plan: {source_plan_path}"
                        )
                    plan = json.loads(
                        source_plan_path.read_text(encoding="utf-8")
                    )
                    planner_engine._validate_query_plan(question["question"], plan)
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_path.write_text(
                        json.dumps(plan, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    plan = planner_engine.build_query_plan(question["question"])
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_path.write_text(
                        json.dumps(plan, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                group_plans[question_id] = plan
                group_records.append(
                    {
                        "question_id": question_id,
                        "plan_id": str(plan["plan_id"]),
                        "path": str(plan_path.relative_to(root)),
                    }
                )
            plans_by_group[group] = group_plans
            report["query_plans"]["groups"][group] = group_records
            _write_report(report_path, report)

        for mode, settings in modes.items():
            mode_config = deepcopy(config)
            mode_config.database_path = root / f"{mode}.db"
            mode_config.log_dir = root / "logs" / mode
            mode_config.retrieval.graph_max_hops = settings["graph_max_hops"]
            mode_config.retrieval.growth_max_rounds = settings["growth_max_rounds"]
            # The source database is a frozen evidence baseline. Schema
            # migrations and query-time growth happen only in per-mode copies.
            existing_results = report["modes"].get(mode, [])
            completed_ids = {
                str(item.get("id"))
                for item in existing_results
                if item.get("status") == "completed"
            }
            if not (resume and completed_ids and mode_config.database_path.exists()):
                source_database.backup_to(mode_config.database_path)
            app = MemoryApplication(mode_config)
            app.rebuild_indexes()
            engine = app.query_engine(app.new_logger("evaluation"))
            existing_by_id = {
                str(item.get("id")): item for item in existing_results
            }
            results: list[dict] = []
            report["modes"][mode] = results
            _write_report(report_path, report)
            for question in questions:
                question_id = str(question["id"])
                result_record = existing_by_id.get(question_id)
                if result_record is not None and result_record.get("status") == "completed":
                    results.append(result_record)
                    continue
                result_record = {**question, "status": "running"}
                results.append(result_record)
                _write_report(report_path, report)
                try:
                    plan_group = report["query_plans"]["mode_groups"][mode]
                    frozen_plan = plans_by_group[plan_group][question_id]
                    result_record["query_plan_id"] = str(frozen_plan["plan_id"])
                    result_record["result"] = engine.query(
                        question["question"], frozen_plan=frozen_plan
                    )
                    result_record["status"] = "completed"
                except Exception as exc:
                    result_record["status"] = "failed"
                    result_record["error_type"] = type(exc).__name__
                    result_record["error"] = str(exc)
                    raise
                finally:
                    _write_report(report_path, report)
        report["status"] = "completed"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
    except Exception as exc:
        report["status"] = "failed"
        report["failed_at"] = datetime.now(timezone.utc).isoformat()
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        _write_report(report_path, report)
        raise
    return {
        "report_path": str(report_path),
        "mode_count": len(modes),
        "question_count": len(questions),
    }
