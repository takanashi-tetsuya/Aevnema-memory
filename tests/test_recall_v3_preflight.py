from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import benchmarks.run_recall_v3 as recall_v3
from benchmarks.run_recall_v3 import (
    PREFLIGHT_SCHEMA,
    RUN_SPEC_SCHEMA,
    main,
    preflight_formal_run,
    preflight_formal_run_bytes,
    preflight_run,
)


_SHA = "sha256:" + "a" * 64


def _gold() -> dict[str, object]:
    return {
        "schema_version": "aevnema.source-gold.v1",
        "status": "approved_for_scoring",
        "review_status": "approved_for_scoring",
        "usable_for_scoring": True,
        "families": [
            {
                "family_id": "family-a",
                "review_status": "approved_for_scoring",
                "usable_for_scoring": True,
                "claim_groups": [
                    {
                        "claim_group_id": "family-a:claim-1",
                        "review_status": "approved_for_scoring",
                        "usable_for_scoring": True,
                        "evidence_atoms": [
                            {
                                "atom_id": "atom-1",
                                "review_status": "approved_for_scoring",
                                "usable_for_scoring": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _split() -> dict[str, object]:
    return {
        "schema_version": "aevnema.gold-split.v1",
        "status": "approved_for_scoring",
        "assignments": [{"family_id": "family-a", "split": "holdout"}],
    }


def _contextual_arm(*, candidate_state: str = "known_empty") -> dict[str, object]:
    candidate_count = 0 if candidate_state == "known_empty" else 2
    return {
        "arm_id": "contextual",
        "primary_score": True,
        "evaluation_role": "treatment",
        "requires_contextual": True,
        "replay_mode": "input_frozen",
        "frozen_input": {"artifact_id": "request-set-v1"},
        "execution_modules": {"matcher": True, "selector": True},
        "snapshot": {
            "database_sha256": _SHA,
            "source_manifest_sha256": _SHA,
            "scope_sha256": _SHA,
            "embedding_space": {
                "id": "bge-m3/normalized-v1",
                "model_id": "bge-m3",
                "preprocess_version": "normalized-v1",
                "dimension": 1024,
                "normalized": True,
            },
        },
        "contextual": {
            "matcher": {"enabled": True, "implementation": "v3-matcher"},
            "cues": [
                {
                    "cue_id": "cue-1",
                    "vector_binding_id": "cue-vector",
                    "source_manifest_sha256": _SHA,
                    "scope_sha256": _SHA,
                }
            ],
            "edges": [
                {
                    "edge_id": "edge-1",
                    "cue_id": "cue-1",
                    "target_ref": "episode:1",
                    "source_manifest_sha256": _SHA,
                    "scope_sha256": _SHA,
                }
            ],
            "query_refs": [
                {
                    "query_ref_id": "whole-query",
                    "role": "whole",
                    "vector_binding_id": "query-vector",
                    "source_manifest_sha256": _SHA,
                    "scope_sha256": _SHA,
                }
            ],
            "vector_bindings": [
                {
                    "binding_id": "cue-vector",
                    "embedding_space_id": "bge-m3/normalized-v1",
                    "dimension": 1024,
                    "normalized": True,
                },
                {
                    "binding_id": "query-vector",
                    "embedding_space_id": "bge-m3/normalized-v1",
                    "dimension": 1024,
                    "normalized": True,
                },
            ],
        },
        "candidate_inventory": {"state": candidate_state, "count": candidate_count},
    }


def _spec(*arms: dict[str, object]) -> dict[str, object]:
    return {
        "schema": RUN_SPEC_SCHEMA,
        "safety": {
            "network": "forbidden",
            "model_calls": "forbidden",
            "database_writes": "forbidden",
        },
        "arms": list(arms),
    }


def _codes(report: dict[str, object]) -> set[str]:
    return {str(issue["code"]) for issue in report["issues"]}  # type: ignore[index]


def _write_formal_source_gold_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object], dict[str, object]]:
    """Create one fully Source-anchored synthetic gold pair for formal CLI tests."""

    source_root = root / "sources"
    source_file = source_root / "main" / "example.json"
    source_file.parent.mkdir(parents=True)
    source_text = "synthetic Source span"
    source_file.write_text(
        json.dumps({"content": [{"Text": source_text}]}),
        encoding="utf-8",
    )
    source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    frozen_manifest = {
        "schema": "aevnema.source-gold.frozen-source-manifest.v1",
        "sources": [
            {
                "source_key": "main/example.json",
                "source_file_sha256": source_digest,
            }
        ],
    }
    frozen_manifest_path = root / "frozen-source-manifest.json"
    frozen_manifest_path.write_text(json.dumps(frozen_manifest), encoding="utf-8")
    frozen_manifest_digest = hashlib.sha256(
        frozen_manifest_path.read_bytes()
    ).hexdigest()

    gold = _gold()
    gold["source_manifest_sha256"] = frozen_manifest_digest
    atom = gold["families"][0]["claim_groups"][0]["evidence_atoms"][0]
    atom.update(
        {
            "source_key": "main/example.json",
            "source_file_sha256": source_digest,
            "record_locator": "/content/0/Text",
            "span_start": 0,
            "span_end": len(source_text),
            "raw_span_sha256": hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
        }
    )
    gold_path = root / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    gold_digest = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    split = {
        "schema_version": "aevnema.gold-split.v1",
        "status": "approved_for_scoring",
        "gold_manifest_sha256": gold_digest,
        "all_source_keys": ["main/example.json"],
        "assignments": [
            {
                "family_id": "family-a",
                "split": "holdout",
                "source_keys": ["main/example.json"],
            }
        ],
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return (
        source_root,
        frozen_manifest_path,
        gold_path,
        split_path,
        gold,
        split,
        frozen_manifest,
    )


def _bind_spec_to_frozen_source_manifest(
    spec: dict[str, object], frozen_manifest_digest: str
) -> None:
    """Keep every synthetic contextual declaration in one frozen scope."""

    binding = "sha256:" + frozen_manifest_digest
    for arm in spec["arms"]:
        arm["snapshot"]["source_manifest_sha256"] = binding
        for item in arm["contextual"]["cues"]:
            item["source_manifest_sha256"] = binding
        for item in arm["contextual"]["edges"]:
            item["source_manifest_sha256"] = binding
        for item in arm["contextual"]["query_refs"]:
            item["source_manifest_sha256"] = binding


class RecallV3PreflightTests(unittest.TestCase):
    def test_complete_contextual_setup_with_empty_candidates_is_ready_not_invalid(self) -> None:
        report = preflight_run(_spec(_contextual_arm()), _gold(), _split())

        self.assertEqual(PREFLIGHT_SCHEMA, report["schema"])
        self.assertEqual("ready", report["status"])
        self.assertTrue(report["can_execute"])
        self.assertEqual(["contextual"], report["valid_no_hit_arm_ids"])
        arm = report["arms"][0]
        self.assertEqual("ready", arm["status"])
        self.assertEqual("valid_no_hit", arm["execution_state"])
        self.assertEqual(0, report["network_calls"])
        self.assertEqual(0, report["model_calls"])
        self.assertEqual(0, report["database_writes"])

    def test_draft_or_unusable_source_gold_is_rejected_before_execution(self) -> None:
        draft = _gold()
        draft["status"] = "draft_pending_source_span_review"
        draft["families"][0]["claim_groups"][0]["evidence_atoms"][0][
            "usable_for_scoring"
        ] = False

        report = preflight_run(_spec(_contextual_arm()), draft, _split())

        self.assertEqual("invalid_setup", report["status"])
        self.assertIn("gold_draft_not_accepted", _codes(report))
        self.assertIn("gold_node_not_usable_for_scoring", _codes(report))

    def test_contextual_empty_mechanism_is_invalid_but_baseline_needs_no_edge(self) -> None:
        contextual = _contextual_arm()
        contextual["contextual"]["cues"] = []
        contextual["contextual"]["edges"] = []
        contextual["contextual"]["query_refs"] = []
        contextual["contextual"]["vector_bindings"] = []
        invalid_report = preflight_run(_spec(contextual), _gold(), _split())

        self.assertEqual("invalid_setup", invalid_report["status"])
        self.assertIn("missing_cue_prototype", _codes(invalid_report))
        self.assertIn("no_eligible_contextual_edges", _codes(invalid_report))
        self.assertIn("missing_query_ref", _codes(invalid_report))
        self.assertIn("missing_vector_binding", _codes(invalid_report))

        baseline = {
            "arm_id": "baseline",
            "primary_score": True,
            "evaluation_role": "treatment",
            "requires_contextual": False,
            "replay_mode": "live",
            "execution_modules": {"selector": True},
            "candidate_inventory": {"state": "known_empty", "count": 0},
        }
        baseline_report = preflight_run(_spec(baseline), _gold(), _split())
        self.assertEqual("ready", baseline_report["status"])
        self.assertEqual("valid_no_hit", baseline_report["arms"][0]["execution_state"])
        self.assertNotIn("no_eligible_contextual_edges", _codes(baseline_report))

        matcher_missing = _contextual_arm()
        matcher_missing["execution_modules"]["matcher"] = False
        matcher_missing["contextual"]["matcher"] = {
            "enabled": False,
            "implementation": "v3-matcher",
        }
        matcher_report = preflight_run(_spec(matcher_missing), _gold(), _split())
        self.assertEqual("invalid_setup", matcher_report["status"])
        self.assertIn("matcher_unavailable", _codes(matcher_report))
        self.assertIn("contextual_disabled", _codes(matcher_report))

    def test_legacy_replay_paths_cannot_supply_v3_primary_score(self) -> None:
        for marker, value in (
            ("frozen_plan", {"legacy": True}),
            ("replay_mode", "legacy_pipeline_replay"),
        ):
            with self.subTest(marker=marker):
                arm = _contextual_arm()
                arm[marker] = value

                report = preflight_run(_spec(arm), _gold(), _split())

                self.assertEqual("invalid_setup", report["status"])
                self.assertIn("legacy_frozen_plan_not_allowed", _codes(report))

    def test_cli_writes_only_a_preflight_json_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            output_path = root / "preflight.json"
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold_document,
                _split_document,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            frozen_digest = hashlib.sha256(
                frozen_manifest_path.read_bytes()
            ).hexdigest()
            spec = _spec(_contextual_arm())
            _bind_spec_to_frozen_source_manifest(spec, frozen_digest)
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            exit_code = main(
                [
                    "--spec",
                    str(spec_path),
                    "--gold",
                    str(gold_path),
                    "--split",
                    str(split_path),
                    "--source-root",
                    str(source_root),
                    "--frozen-source-manifest",
                    str(frozen_manifest_path),
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code, report)
        self.assertEqual("ready", report["status"])
        self.assertEqual(0, report["network_calls"])
        self.assertEqual(0, report["model_calls"])
        self.assertEqual(0, report["database_writes"])
        self.assertTrue(report["source_gold_physical"]["scoring_eligible"])

    def test_formal_preflight_blocks_json_ready_gold_without_a_verified_span(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                _gold_path,
                split_path,
                gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            frozen_digest = hashlib.sha256(
                frozen_manifest_path.read_bytes()
            ).hexdigest()
            spec = _spec(_contextual_arm())
            _bind_spec_to_frozen_source_manifest(spec, frozen_digest)
            # All JSON-level flags remain true.  The physical evaluator must
            # still reject an atom whose span cannot be re-derived.
            gold["families"][0]["claim_groups"][0]["evidence_atoms"][0][
                "raw_span_sha256"
            ] = "0" * 64
            report = preflight_formal_run_bytes(
                json.dumps(spec).encode("utf-8"),
                gold_bytes=json.dumps(gold).encode("utf-8"),
                split_bytes=split_path.read_bytes(),
                source_root=source_root,
                frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertFalse(report["can_execute"])
        self.assertIn("source_gold_physical_not_ready", _codes(report))
        physical_codes = {
            item["code"] for item in report["source_gold_physical"]["issues"]
        }
        self.assertIn("atom_raw_span_sha256_mismatch", physical_codes)

    def test_only_raw_spec_bytes_can_authorize_formal_preflight(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            spec = _spec(_contextual_arm())
            _bind_spec_to_frozen_source_manifest(
                spec,
                hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            )
            byte_bound = preflight_formal_run_bytes(
                json.dumps(spec).encode("utf-8"),
                gold_bytes=gold_path.read_bytes(),
                split_bytes=split_path.read_bytes(),
                source_root=source_root,
                frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
            )
            diagnostic = preflight_formal_run(
                spec,
                gold_bytes=gold_path.read_bytes(),
                split_bytes=split_path.read_bytes(),
                source_root=source_root,
                frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
            )

        self.assertEqual("ready", byte_bound["status"])
        self.assertTrue(byte_bound["can_execute"])
        self.assertEqual("invalid_setup", diagnostic["status"])
        self.assertFalse(diagnostic["can_execute"])
        self.assertIn(
            "formal_run_spec_raw_binding_unavailable", _codes(diagnostic)
        )

    def test_formal_report_never_reflects_untrusted_artifact_strings(self) -> None:
        sentinel = "SOURCE_TEXT_MUST_NOT_APPEAR_IN_FORMAL_REPORT"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                _gold_path,
                split_path,
                gold,
                split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            gold["status"] = sentinel
            gold["families"][0]["family_id"] = sentinel
            split["assignments"][0]["family_id"] = sentinel
            split_path.write_text(json.dumps(split), encoding="utf-8")
            spec = _spec(_contextual_arm())
            spec["arms"][0]["arm_id"] = sentinel
            _bind_spec_to_frozen_source_manifest(
                spec,
                hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            )
            report = preflight_formal_run_bytes(
                json.dumps(spec).encode("utf-8"),
                gold_bytes=json.dumps(gold).encode("utf-8"),
                split_bytes=split_path.read_bytes(),
                source_root=source_root,
                frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
            )

        self.assertNotIn(sentinel, json.dumps(report, ensure_ascii=False))
        self.assertEqual("invalid_setup", report["status"])
        self.assertEqual(0, report["arms"][0]["arm_index"])
        self.assertNotIn("arm_id", report["arms"][0])

    def test_formal_sanitizer_fails_closed_for_container_values_without_crashing(self) -> None:
        sentinel = "UNTRUSTED_CONTAINER_TEXT_MUST_NOT_APPEAR"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            gold["status"] = [sentinel]
            spec = _spec(_contextual_arm())
            spec["arms"][0]["candidate_inventory"]["state"] = [sentinel]
            _bind_spec_to_frozen_source_manifest(
                spec,
                hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            )
            report = preflight_formal_run_bytes(
                json.dumps(spec).encode("utf-8"),
                gold_bytes=json.dumps(gold).encode("utf-8"),
                split_bytes=split_path.read_bytes(),
                source_root=source_root,
                frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
            )

        self.assertEqual("invalid_setup", report["status"])
        self.assertFalse(report["can_execute"])
        self.assertNotIn(sentinel, json.dumps(report, ensure_ascii=False))

    def test_formal_cli_rejects_report_output_that_can_mutate_inputs_or_sources(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            spec = _spec(_contextual_arm())
            _bind_spec_to_frozen_source_manifest(
                spec,
                hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            )
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            protected_targets = (
                (gold_path, gold_path.read_bytes()),
                (
                    source_root / "main" / "example.json",
                    (source_root / "main" / "example.json").read_bytes(),
                ),
            )
            for output_path, original in protected_targets:
                with self.subTest(output_path=output_path), redirect_stdout(StringIO()):
                    exit_code = main(
                        [
                            "--spec",
                            str(spec_path),
                            "--gold",
                            str(gold_path),
                            "--split",
                            str(split_path),
                            "--source-root",
                            str(source_root),
                            "--frozen-source-manifest",
                            str(frozen_manifest_path),
                            "--output",
                            str(output_path),
                        ]
                    )
                self.assertEqual(2, exit_code)
                self.assertEqual(original, output_path.read_bytes())

            hard_link_alias = root / "gold-hard-link-alias.json"
            try:
                os.link(gold_path, hard_link_alias)
            except OSError:  # pragma: no cover - filesystem policy dependent
                return
            original_gold = gold_path.read_bytes()
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--spec",
                        str(spec_path),
                        "--gold",
                        str(gold_path),
                        "--split",
                        str(split_path),
                        "--source-root",
                        str(source_root),
                        "--frozen-source-manifest",
                        str(frozen_manifest_path),
                        "--output",
                        str(hard_link_alias),
                    ]
                )
            self.assertEqual(2, exit_code)
            self.assertEqual(original_gold, gold_path.read_bytes())

            missing_parent = root / "not-created-for-formal-report"
            missing_output = missing_parent / "preflight.json"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--spec",
                        str(spec_path),
                        "--gold",
                        str(gold_path),
                        "--split",
                        str(split_path),
                        "--source-root",
                        str(source_root),
                        "--frozen-source-manifest",
                        str(frozen_manifest_path),
                        "--output",
                        str(missing_output),
                    ]
                )
            self.assertEqual(2, exit_code)
            self.assertFalse(missing_parent.exists())

    def test_formal_cli_rejects_nonlocal_artifact_before_opening_it(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--spec",
                        r"\\server\share\untrusted-spec.json",
                        "--gold",
                        str(gold_path),
                        "--split",
                        str(split_path),
                        "--source-root",
                        str(source_root),
                        "--frozen-source-manifest",
                        str(frozen_manifest_path),
                    ]
                )
        self.assertEqual(2, exit_code)

    def test_formal_cli_keeps_relative_inputs_and_output_bound_after_cwd_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "working"
            other = root / "other"
            working.mkdir()
            other.mkdir()
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(working)
            spec = _spec(_contextual_arm())
            _bind_spec_to_frozen_source_manifest(
                spec,
                hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            )
            spec_path = working / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            original_loader = recall_v3.load_json_document_bytes
            loader_calls = 0

            def switch_cwd_after_preflight(path: Path) -> tuple[object, str, bytes]:
                nonlocal loader_calls
                loader_calls += 1
                if loader_calls == 1:
                    os.chdir(other)
                return original_loader(path)

            previous_cwd = Path.cwd()
            try:
                os.chdir(working)
                with patch.object(
                    recall_v3,
                    "load_json_document_bytes",
                    side_effect=switch_cwd_after_preflight,
                ):
                    exit_code = main(
                        [
                            "--spec",
                            "spec.json",
                            "--gold",
                            gold_path.name,
                            "--split",
                            split_path.name,
                            "--source-root",
                            source_root.name,
                            "--frozen-source-manifest",
                            frozen_manifest_path.name,
                            "--output",
                            "preflight.json",
                        ]
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(0, exit_code)
            self.assertTrue((working / "preflight.json").is_file())
            self.assertFalse((other / "preflight.json").exists())

    def test_public_formal_writer_keeps_relative_output_bound_after_cwd_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "working"
            other = root / "other"
            working.mkdir()
            other.mkdir()
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(working)
            original_guard = recall_v3._safe_report_output_path
            guard_calls = 0

            def switch_cwd_after_first_guard(*args: object, **kwargs: object) -> bool:
                nonlocal guard_calls
                result = original_guard(*args, **kwargs)  # type: ignore[arg-type]
                guard_calls += 1
                if guard_calls == 1:
                    os.chdir(other)
                return result

            previous_cwd = Path.cwd()
            try:
                os.chdir(working)
                with patch.object(
                    recall_v3,
                    "_safe_report_output_path",
                    side_effect=switch_cwd_after_first_guard,
                ):
                    written = recall_v3.write_preflight(
                        {"schema": "synthetic-formal-report"},
                        Path("public-writer-report.json"),
                        input_paths=(
                            Path(gold_path.name),
                            Path(split_path.name),
                            Path(frozen_manifest_path.name),
                        ),
                        source_root=Path(source_root.name),
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertTrue(written)
            self.assertTrue((working / "public-writer-report.json").is_file())
            self.assertFalse((other / "public-writer-report.json").exists())

    def test_formal_bytes_parser_rejects_overflow_and_deep_json_without_crashing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                frozen_manifest_path,
                gold_path,
                split_path,
                _gold,
                _split,
                _frozen_manifest,
            ) = _write_formal_source_gold_inputs(root)
            malformed_specs = (
                (b'{"probe": 1e999}', "input_load_failed"),
                # The decoder limit varies by supported Python build.  This
                # must never escape the formal gate whether it parses or
                # raises RecursionError on a particular runtime.
                (b"[" * 1200 + b"0" + b"]" * 1200, None),
            )
            for malformed_spec, expected_code in malformed_specs:
                with self.subTest(malformed_spec=malformed_spec[:20]):
                    report = preflight_formal_run_bytes(
                        malformed_spec,
                        gold_bytes=gold_path.read_bytes(),
                        split_bytes=split_path.read_bytes(),
                        source_root=source_root,
                        frozen_source_manifest_bytes=frozen_manifest_path.read_bytes(),
                    )
                    self.assertEqual("invalid_setup", report["status"])
                    self.assertFalse(report["can_execute"])
                    if expected_code is not None:
                        self.assertIn(expected_code, _codes(report))
                    else:
                        self.assertTrue(_codes(report))


if __name__ == "__main__":
    unittest.main()
