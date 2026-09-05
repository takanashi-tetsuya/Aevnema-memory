from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from memory_demo.cli import main
from memory_demo.config import AppConfig
from memory_demo.evaluation import run_evaluation


class _FakeDatabase:
    def backup_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class _FailingEngine:
    def build_query_plan(self, question: str) -> dict:
        return {
            "plan_schema": "frozen_query_plan_v1",
            "plan_id": f"plan:{question}",
            "question": question,
        }

    def _validate_query_plan(self, question: str, plan: dict) -> None:
        if plan.get("question") != question:
            raise ValueError("plan mismatch")

    def query(self, question: str, *, frozen_plan: dict | None = None) -> dict:
        raise RuntimeError(f"query failed: {question}")


class _SucceedingEngine:
    plan_builds = 0

    def build_query_plan(self, question: str) -> dict:
        type(self).plan_builds += 1
        return {
            "plan_schema": "frozen_query_plan_v1",
            "plan_id": f"plan:{question}",
            "question": question,
        }

    def _validate_query_plan(self, question: str, plan: dict) -> None:
        if plan.get("question") != question:
            raise ValueError("plan mismatch")

    def query(self, question: str, *, frozen_plan: dict | None = None) -> dict:
        return {
            "answer": f"completed: {question}",
            "query_plan_id": frozen_plan.get("plan_id") if frozen_plan else None,
        }


class _FakeApplication:
    def __init__(self, config: AppConfig):
        self.db = _FakeDatabase()

    def rebuild_indexes(self) -> None:
        return None

    def new_logger(self, name: str):
        return None

    def query_engine(self, logger=None) -> _FailingEngine:
        return _FailingEngine()


class _SucceedingApplication(_FakeApplication):
    def query_engine(self, logger=None) -> _SucceedingEngine:
        return _SucceedingEngine()


class EvaluationTests(unittest.TestCase):
    def test_cli_evaluate_does_not_initialize_source_application(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text("[]", encoding="utf-8")
            env_file = root / ".env"
            env_file.write_text(
                f"MEMORY_DB_PATH={root / 'frozen.db'}\n",
                encoding="utf-8",
            )
            evaluation = Mock(return_value={"status": "completed"})

            with patch(
                "memory_demo.cli.MemoryApplication",
                side_effect=AssertionError("source application must not be opened"),
            ), patch("memory_demo.cli.run_evaluation", evaluation):
                exit_code = main(
                    [
                        "--env-file",
                        str(env_file),
                        "evaluate",
                        str(questions),
                        str(root / "output"),
                        "--modes",
                        "vector_only",
                    ]
                )

            self.assertEqual(exit_code, 0)
            evaluation.assert_called_once()

    def test_failed_query_is_persisted_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(
                json.dumps([{"id": "q1", "question": "测试问题"}], ensure_ascii=False),
                encoding="utf-8",
            )
            config = AppConfig(database_path=root / "source.db")
            config.database_path.touch()

            with patch("memory_demo.evaluation.MemoryApplication", _FakeApplication):
                with self.assertRaisesRegex(RuntimeError, "query failed"):
                    run_evaluation(
                        config,
                        questions,
                        root / "evaluation",
                        selected_modes=["vector_only"],
                    )

            report = json.loads(
                (root / "evaluation" / "evaluation-report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_type"], "RuntimeError")
            self.assertIn("failed_at", report)
            result = report["modes"]["vector_only"][0]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "RuntimeError")
            self.assertIn("测试问题", result["error"])

    def test_resume_skips_completed_questions_and_retries_failed_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [
                        {"id": "done", "question": "已经完成"},
                        {"id": "retry", "question": "需要重试"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = AppConfig(database_path=root / "source.db")
            config.database_path.touch()
            output = root / "evaluation"
            output.mkdir()
            (output / "vector_only.db").touch()
            report = {
                "created_at": "2026-01-01T00:00:00+00:00",
                "source_database": str(config.database_path.resolve()),
                "questions": [
                    {
                        "id": "done",
                        "question": "已经完成",
                        "expected_capability": "",
                    },
                    {
                        "id": "retry",
                        "question": "需要重试",
                        "expected_capability": "",
                    },
                ],
                "mode_order": ["vector_only"],
                "configuration": {},
                "status": "failed",
                "error": "temporary network failure",
                "modes": {
                    "vector_only": [
                        {
                            "id": "done",
                            "question": "已经完成",
                            "expected_capability": "",
                            "status": "completed",
                            "result": {"answer": "keep me"},
                        },
                        {
                            "id": "retry",
                            "question": "需要重试",
                            "expected_capability": "",
                            "status": "failed",
                            "error": "temporary network failure",
                        },
                    ]
                },
            }
            (output / "evaluation-report.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )

            with patch(
                "memory_demo.evaluation.MemoryApplication", _SucceedingApplication
            ):
                run_evaluation(
                    config,
                    questions,
                    output,
                    selected_modes=["vector_only"],
                    resume=True,
                )

            resumed = json.loads(
                (output / "evaluation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(
                resumed["modes"]["vector_only"][0]["result"]["answer"],
                "keep me",
            )
            self.assertEqual(
                resumed["modes"]["vector_only"][1]["result"]["answer"],
                "completed: 需要重试",
            )
            self.assertNotIn("error", resumed)

    def test_reverse_question_order_is_recorded_and_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [
                        {"id": "first", "question": "第一题"},
                        {"id": "second", "question": "第二题"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = AppConfig(database_path=root / "source.db")
            config.database_path.touch()
            output = root / "evaluation"

            with patch(
                "memory_demo.evaluation.MemoryApplication", _SucceedingApplication
            ):
                run_evaluation(
                    config,
                    questions,
                    output,
                    selected_modes=["vector_only"],
                    question_order="reverse",
                )

            report = json.loads(
                (output / "evaluation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["question_order"], "reverse")
            self.assertEqual(
                [item["id"] for item in report["modes"]["vector_only"]],
                ["second", "first"],
            )

    def test_static_and_growing_modes_share_one_frozen_query_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [{"id": "shared", "question": "共享计划问题"}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = AppConfig(database_path=root / "source.db")
            config.database_path.touch()
            output = root / "evaluation"

            with patch(
                "memory_demo.evaluation.MemoryApplication", _SucceedingApplication
            ):
                run_evaluation(
                    config,
                    questions,
                    output,
                    selected_modes=["graph_static", "graph_growing"],
                )

            report = json.loads(
                (output / "evaluation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["query_plans"]["mode_groups"],
                {"graph_static": "hops-3", "graph_growing": "hops-3"},
            )
            self.assertEqual(len(report["query_plans"]["groups"]), 1)
            static_id = report["modes"]["graph_static"][0]["query_plan_id"]
            growing_id = report["modes"]["graph_growing"][0]["query_plan_id"]
            self.assertEqual(static_id, growing_id)
            self.assertEqual(
                report["modes"]["graph_static"][0]["result"]["query_plan_id"],
                static_id,
            )

    def test_external_frozen_query_plan_can_be_reused_without_replanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(
                json.dumps([{"id": "q1", "question": "复用计划"}], ensure_ascii=False),
                encoding="utf-8",
            )
            config = AppConfig(database_path=root / "source.db")
            config.database_path.touch()
            first = root / "first"
            second = root / "second"
            _SucceedingEngine.plan_builds = 0

            with patch(
                "memory_demo.evaluation.MemoryApplication", _SucceedingApplication
            ):
                run_evaluation(
                    config,
                    questions,
                    first,
                    selected_modes=["graph_static"],
                )
                builds_after_first = _SucceedingEngine.plan_builds
                run_evaluation(
                    config,
                    questions,
                    second,
                    selected_modes=["graph_growing"],
                    query_plans_from=first,
                )

            self.assertEqual(_SucceedingEngine.plan_builds, builds_after_first)
            first_report = json.loads(
                (first / "evaluation-report.json").read_text(encoding="utf-8")
            )
            second_report = json.loads(
                (second / "evaluation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_report["query_plans"]["groups"]["hops-3"][0]["plan_id"],
                second_report["query_plans"]["groups"]["hops-3"][0]["plan_id"],
            )


if __name__ == "__main__":
    unittest.main()
